# Session KV reuse

The server keeps one KV cache alive between requests. If your next request **appends** to the same conversation, prefill is skipped for the shared prefix — only the new tokens are processed. This makes multi‑turn chat fast after the first message, and applies to both text‑only and vision conversations.

## The core problem

The naive approach — re‑tokenize the full conversation each turn and compare against the cached tokens — **never matches**. Re‑rendering old messages via `apply_chat_template` diverges from the raw tokens the model actually generated, because of:

1. `_safe_messages` converting a tool call's JSON `arguments` string into a dict (Jinja renders it with different whitespace),
2. whitespace stripping applied to reasoning content, and
3. Jinja template whitespace around tags never matching the model's byte‑exact autoregressive output.

Any of these shifts a token, the prefix comparison fails, and the whole conversation is re‑prefilled every turn.

## The fix: splice raw tokens

Instead of re‑tokenizing history, the engine **reuses the raw token IDs** from the previous turn (`_session_ids`) and only tokenizes the *new* messages:

```text
combined = _session_ids[:cache_seq_len]   # cached prefix (raw tokens)
         + _session_ids[cache_seq_len]     # one "bridge" token
         + tokenize(new_messages)          # only the appended messages
```

`get_seq_length()` is one less than the full previous sequence (the last generated token was never fed back), so the **bridge token** stitches that boundary. `_prepare_generation` then re‑prefills exactly `input_ids[:, cache_seq_len:]`, leaving one token for `generate`. Because the cached prefix is reused verbatim, the token‑level prefix check becomes a tautology — correctness rests on the guards below.

See [Inference engine](../internals/inference-engine.md) for the full mechanics.

## Guards

Since the prefix check no longer protects against edited history, two guards do:

- **Append‑only fingerprint** (`_cached_input_fp`): a hash of the prior input messages `(role, hash(content))`. Reuse only proceeds if the new turn's leading messages match the fingerprint. Editing, regenerating, or truncating a prior message changes the fingerprint → full prefill.
- **Injected‑system‑block stripping**: rendering the new‑message tail in isolation can make the template prepend a default system block that's already in the cached prefix. The engine drops a leading `<|im_start|>system` turn the new messages didn't ask for, so it isn't spliced in twice.

## Thinking turns

For a hit on turns with reasoning, the client must send the previous reply's reasoning **back** (`reasoning_content` on the assistant message, as the API returns it). If the reasoning is dropped, the re‑rendered history no longer matches the cached tokens and prefill restarts. The interactive CLI does this automatically.

## Vision

When an image appears in history, the vision encoder runs once for that turn and its KV states are cached; text follow‑ups extend the same cache instead of re‑encoding the image. New images (appended, or a swapped image detected by pixel fingerprint) invalidate and rebuild the cache.

## Constraint: one conversation per instance

There is **one** session cache per server, holding a single conversation. This is by design — two live caches would not fit in VRAM. A request that doesn't share the prefix (editing a prior message, switching conversations, or a shorter history than cached) invalidates the cache and re‑prefills. For multiple independent sessions, run multiple server instances on different ports.
