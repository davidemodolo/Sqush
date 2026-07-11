# Server & API

`sqush/server.py` builds a FastAPI app (`create_app`) with permissive CORS. The engine and its session cache are shared across requests — **one conversation at a time** by design (see [Session KV reuse](../concepts/session-kv-reuse.md)).

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness `{"status":"ok"}` |
| `GET` | `/health/vram` | GPU memory stats from `engine.get_vram_info()` |
| `GET` | `/v1/models` | Lists the configured model with context/output limits |
| `GET` | `/v1/models/{id}` | Single model (404 if unknown) |
| `POST` | `/v1/chat/completions` | Chat completion, streaming + non‑streaming |

## Chat completions

Request fields consumed: `messages`, `stream`, `max_tokens`, `temperature`, `top_p`, `tools`.

`enable_thinking` is derived from `_is_small_task()`: title‑generation and similar lightweight prompts (matched by marker phrases like "generate a short title") disable reasoning so `<think>` tags don't leak into short outputs.

- **Streaming** → `EventSourceResponse(_stream_response(...))`.
- **Non‑streaming** → `_sync_response` run via `loop.run_in_executor(...)` so a multi‑minute generation doesn't block the event loop (and `/health` stays responsive).

## Thinking‑block parsing

With `add_generation_prompt`, the model emits the closing `</think>` **only** (the opening tag lives in the prompt). Both paths treat everything up to `</think>` as reasoning:

- **Streaming** starts in `state="think"` and searches for `</think>`, emitting `reasoning_content` deltas until it flips to `post`.
- **Sync** mirrors this (guarded by `enable_thinking`): text up to `</think>` becomes `reasoning_content`, the rest is `content`. A stray opening `<think>`, if present, is stripped.

## Streaming state machine

`_stream_response` maintains `state ∈ {think, post, tool_call}` over a growing `buffer`, holding back `HOLD` characters so a partial tag isn't emitted mid‑split:

- **think** → emit `reasoning_content` until `</think>`.
- **post** → emit `content` until `<tool_call>`.
- **tool_call** → feed the incremental tool parser until `</tool_call>`.

At end‑of‑stream it flushes the remaining buffer for whichever state it's in — including a **finalize** of the tool parser if the stream ended mid‑`<tool_call>` (so the arguments delta is still emitted rather than a tool call with empty arguments).

## Tool calling

The model emits XML like `<tool_call><function=name><parameter=k>v</parameter></function></tool_call>`. `_make_tool_call_stream_parser` returns a stateful `parse(new_text, finalize=False)`:

- The **function name** is emitted as soon as `<function=…>` is seen (as an OpenAI `tool_calls` delta with an `id` and empty `arguments`).
- **Arguments** are emitted only on `finalize`, as a single complete JSON object — because streaming JSON char‑by‑char breaks when new keys insert commas, invalidating prefix concatenation.

The sync path reuses the same parser per `<tool_call>` match and assembles full `tool_calls` entries.

## Throughput logging

Streaming logs tok/s incrementally: every `logging.tps_interval_tokens` tokens it emits total vs. recent rate (counting only the newly appended chunk each token — not re‑encoding the whole buffer, which would be O(n²)). It logs a dedicated **think‑phase** tok/s when `</think>` closes, and a final summary (`prompt/completion/total tokens`, elapsed, avg tok/s). The engine separately logs **prefill vs. decode** tok/s.

## Delta & usage format

`_delta_chunk` emits standard `chat.completion.chunk` objects; `reasoning_content` is duplicated as `reasoning_text` for client compatibility. The final chunk carries `finish_reason` (`tool_calls` if any were emitted, else `stop`) and a `usage` block.

## OpenCode registration (`init`)

`python -m sqush init` writes a **provider‑only** entry to `~/.config/opencode/opencode.json`, merging without touching other providers:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "sqush": {
      "name": "Sqush (local)",
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "http://127.0.0.1:9898/v1", "apiKey": "local" },
      "models": {
        "qwen3.6-27b": {
          "name": "Qwen3.6 27B 4-bit (local)",
          "reasoning": true, "tools": true,
          "modalities": { "input": ["text", "image"], "output": ["text"] },
          "limit": { "context": 262144, "output": 65536 }
        },
        "qwen3.5-9b": { "…": "…" }
      }
    }
  }
}
```

Both model entries are registered regardless of the active tier. No `agent` block is written.
