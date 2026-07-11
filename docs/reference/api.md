# API reference

OpenAI‑compatible, served at `http://<host>:<port>` (default `127.0.0.1:9898`).

## `GET /health`

```json
{ "status": "ok" }
```

## `GET /health/vram`

Returns `engine.get_vram_info()` — `cuda_available`, `used_gb`, `total_gb`, `allocated_gb`, `reserved_gb` (fields present when CUDA is available).

## `GET /v1/models` · `GET /v1/models/{id}`

```json
{
  "object": "list",
  "data": [{
    "id": "Qwen/Qwen3.6-27B",
    "object": "model",
    "owned_by": "sqush",
    "context_window": 262144,
    "max_output_tokens": 65536
  }]
}
```

`GET /v1/models/{id}` returns the single model or `404` if the id doesn't match `model.repo`.

## `POST /v1/chat/completions`

### Request

| Field | Type | Notes |
|-------|------|-------|
| `messages` | array | required; OpenAI chat messages |
| `stream` | bool | default `false` |
| `max_tokens` | int? | falls back to `inference.max_new_tokens` |
| `temperature` | float? | falls back to config |
| `top_p` | float? | falls back to config |
| `tools` | array? | OpenAI tool/function schemas |

### Non‑streaming response

```json
{
  "id": "chatcmpl-…",
  "object": "chat.completion",
  "model": "Qwen/Qwen3.6-27B",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "…",
      "reasoning_content": "…",
      "tool_calls": [ … ]
    },
    "finish_reason": "stop"
  }],
  "usage": { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 }
}
```

`reasoning_content` (and its alias `reasoning_text`) appear when the model produced a thinking block. `finish_reason` is `tool_calls` when tool calls were emitted.

### Streaming response (SSE)

`text/event-stream` of `chat.completion.chunk` objects. The first chunk sets `delta.role`; subsequent chunks carry `delta.content`, `delta.reasoning_content`/`reasoning_text`, or `delta.tool_calls`. The final chunk has `finish_reason` and a `usage` block, followed by `data: [DONE]`.

### Image input

Send base64 data URLs as OpenAI `image_url` content parts:

```json
{
  "model": "Qwen/Qwen3.6-27B",
  "messages": [{
    "role": "user",
    "content": [
      { "type": "text", "text": "What is in this image?" },
      { "type": "image_url", "image_url": { "url": "data:image/png;base64,…" } }
    ]
  }]
}
```

Only base64 data URLs are supported — remote `http(s)` URLs are not fetched. On the 8 GB tier images are downscaled to the pixel cap before the vision encoder.

### Tool calls

Provide `tools` in OpenAI format. The model emits `<tool_call>` XML which the server converts to `tool_calls` deltas (streaming) or a full `tool_calls` array (sync). Function names stream immediately; arguments arrive as one complete JSON object.
