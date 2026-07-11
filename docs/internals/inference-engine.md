# Inference engine

`InferenceEngine` (`sqush/engine.py`) owns tokenization, the session KV cache, chunked prefill, and both generation paths (sync + streaming), for text and vision.

## Session state

| Field | Meaning |
|-------|---------|
| `_session_kv` | the live `SqushKVCache` (or a `DynamicCache`), reused across turns |
| `_session_ids` | the **raw** token IDs the cache was built from (full previous sequence) |
| `_session_num_messages` | message count at cache build (for vision new‑image detection) |
| `_cached_msg_count` | input messages covered by the cache, **+1** for the generated assistant reply now baked into `_session_ids` |
| `_cached_input_fp` | fingerprint of the cached input messages (append‑only guard) |
| `_last_prefill_s` | last prefill wall‑time (for tok/s logging) |
| `_last_prompt_tokens` | prompt token count of the last request (read by the server for usage) |

## Chunked prefill (`_chunked_prefill`)

Long prompts are prefilled in `PREFILL_CHUNK = 1024`‑token chunks to bound the FLA linear‑attention transient. It prefills all but the last token (generate handles the last token + decode). If given an existing cache it appends to it instead of creating a fresh one.

## Tokenization & the reuse splice (`_tokenize`)

Normal path: `apply_chat_template(_safe_messages(messages), add_generation_prompt=True, enable_thinking=…, preserve_thinking=True)`. `_safe_messages` converts a tool call's JSON `arguments` string into a dict (the Qwen template expects a mapping).

**Reuse path** (when a valid cache exists, no new images, and `cached_input_ids` is supplied): render only the *new* messages and splice raw tokens —

```python
combined = cached_input_ids                       # _session_ids[:cache_seq_len]
         + _session_ids[cache_seq_len:cache_seq_len+1]  # bridge token
         + tokenize(new_messages)
```

Before splicing, a leading auto‑injected `<|im_start|>system` block (which the template may prepend when rendering the tail in isolation) is stripped, since it's already in the cached prefix. See [Session KV reuse](../concepts/session-kv-reuse.md) for the rationale.

## `_prepare_generation`

Decides reuse vs. full prefill:

1. If `_session_kv` and `_session_ids` exist and `input_ids[:cache_seq_len]` matches `_session_ids[:cache_seq_len]`, reuse: prefill only `input_ids[:, cache_seq_len:]` onto the existing cache.
2. On a miss, log it and `_free_cache()`.
3. Otherwise full prefill — chunked if `> PREFILL_CHUNK`, else a fresh cache from the factory.

With the reuse splice, that prefix comparison is a tautology (the prefix is reused verbatim), so correctness rests on the **append‑only fingerprint guard** in the callers: reuse only proceeds if `_fingerprint(messages[:prior]) == _cached_input_fp`. A changed/edited/truncated history fails the check and falls back to full prefill.

## Generation

### Sync (`chat_completion_sync`)

Extract images → decide `has_new_images` (count **and** pixel‑fingerprint comparison, so a swapped image invalidates the cache) → tokenize (with reuse splice if eligible) →

- **New images**: `_free_cache()`, fresh cache, chunked **vision** prefill (`_chunked_vision_prefill`), then `model.generate`. Saves the KV so text follow‑ups extend it.
- **No images**: `_prepare_generation` → `model.generate` (wrapped in `try/except` that frees the cache on failure).

Returns `(text, prompt_tokens, completion_tokens)`. Logs prefill vs. decode tok/s separately.

### Streaming (`chat_completion_stream`)

A generator yielding decoded text chunks via `TextIteratorStreamer` on a worker thread. The thread wrapper captures exceptions into a dict and calls `streamer.end()` in a `finally` — so a failed `generate` surfaces the real error and doesn't hang the consumer waiting for tokens. On error the cache is freed. `_last_prompt_tokens` is set inside the generator body (only populated once the first token is pulled).

## Vision path (`_chunked_vision_prefill`)

Runs the vision encoder once, merges image embeddings into the input embeddings via `get_placeholder_mask` + `masked_scatter`, computes 3D position IDs, then prefills the language model in `PREFILL_CHUNK` chunks. The resulting KV is saved so subsequent text turns reuse it without re‑encoding the image.

## Cache lifecycle (`_free_cache`, `reset_session`)

`_free_cache()` deletes `_session_kv`, clears `_session_ids`/`_cached_msg_count`/`_cached_input_fp`, calls `torch.cuda.empty_cache()`, and logs freed positions + VRAM. It's called on cache miss, on new images, on generation exceptions, and by `reset_session()`.
