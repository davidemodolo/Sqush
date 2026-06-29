# QuantStar

**Run Qwen3.6-27B on a single 24GB GPU.**

4-bit weights + 4-bit KV cache + blockwise attention = 256k context in ~22 GB VRAM.
Multimodal text + image input. Drop-in OpenAI-compatible API. Local, private, zero config.

## Features

- **Full 256k context** on consumer hardware - custom blockwise GQA attention keeps KV cache at native 4 KV heads
- **4-bit everything** - NF4 weights (bitsandbytes) + int4 append-only KV cache (never re-quantizes)
- **OpenAI-compatible API** - swap `base_url` and use any OpenAI client, agents, or tools
- **Streaming (SSE)** - real token streaming with proper `reasoning_content` + `tool_calls` delta emission
- **Tool calling** - model uses `<tool_call>` XML, server parses it incrementally into OpenAI-format deltas
- **Image input** - Qwen3.6-VL vision encoder processes images via `image_url` content parts (base64 data URLs)
- **Interactive CLI** - Rich-based chat with session reuse, `/clear`, `/vram`, `/system` commands
- **Session KV reuse** - subsequent requests sharing the same prompt prefix skip redundant prefill

## Quickstart

```bash
./run.sh download    # download Qwen3.6-27B (one-time, ~52 GB)
./run.sh serve       # start the API on 127.0.0.1:9898
```

That's it. Test with:

```bash
curl http://127.0.0.1:9898/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3.6-27B","messages":[{"role":"user","content":"Hi!"}]}'
```

Other commands:

```bash
./run.sh chat         # interactive CLI
./run.sh info         # show config and VRAM
./run.sh init         # register in OpenCode config
```

`run.sh` handles everything: venv creation, CUDA 12.6 PyTorch install, dependencies, and a transformer docstring patch for Python 3.14.

## Use with OpenCode

Register QuantStar as a local provider:

```bash
./run.sh init
```

This writes the provider and agent config to `~/.config/opencode/opencode.json`. Then in OpenCode, run `/models` and select `quantstar/qwen3.6-27b`.

Or add it manually - in your `opencode.json`:

```json
{
  "provider": {
    "quantstar": {
      "name": "QuantStar (local)",
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://127.0.0.1:9898/v1",
        "apiKey": "local"
      },
      "models": {
        "qwen3.6-27b": {
          "name": "Qwen3.6 27B 4-bit (local)",
          "reasoning": true,
          "tools": true,
          "modalities": {
            "input": ["text", "image"],
            "output": ["text"]
          },
          "limit": { "context": 262144, "output": 65536 }
        }
      }
    }
  },
  "agent": {
    "quantstar": {
      "description": "Local QuantStar - Qwen3.6 27B 4-bit",
      "model": "quantstar/qwen3.6-27b",
      "temperature": 0
    }
  }
}
```

## Configuration

Edit `config.yaml` (or use `QUANTSTAR_*` env vars):

```yaml
model:
  repo: "Qwen/Qwen3.6-27B"
  cache_dir: "./models"

inference:
  max_context: 262144     # full 256k
  max_new_tokens: 65536
  temperature: 0.7

server:
  host: "127.0.0.1"
  port: 9898
```

## API

OpenAI-compatible at `http://127.0.0.1:9898`:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check |
| `GET` | `/health/vram` | GPU memory stats |
| `POST` | `/v1/chat/completions` | Chat completion (streaming + non-streaming) |

Streaming returns `reasoning_content` deltas for model thinking and `tool_calls` deltas when tools are provided.

### Image input

Send images as base64 data URLs using OpenAI-compatible `image_url` content parts:

```bash
curl http://127.0.0.1:9898/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"Qwen/Qwen3.6-27B",
    "messages":[{
      "role":"user",
      "content":[
        {"type":"text","text":"What do you see in this image?"},
        {"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}
      ]
    }]
  }'
```

Images are processed by Qwen3.6-VL's vision encoder. Vision requests use full-model `generate()` (no chunked prefill) and reset the session KV cache. Only base64 data URLs are supported — remote URLs are not yet handled.

## Text Performance

Measured on RTX 3090 24GB, Python 3.14, torch 2.12.1+cu126:

| Context | Peak VRAM | Prefill | Decode |
|---------|-----------|---------|--------|
| 3k | 17.1 GB | 3.0s | 11.0 tok/s |
| 16k | 17.3 GB | 20.8s | 6.7 tok/s |
| 32k | 17.6 GB | 52.9s | 4.4 tok/s |
| 64k | 18.1 GB | 151.1s | 2.6 tok/s |
| 128k | 19.1 GB | 480.3s | 1.4 tok/s |
| 256k | 21.2 GB | 1649.6s | 0.8 tok/s |

Weights: ~16.5 GB (NF4). KV cache: ~4.3 GB at 256k (int4, append-only). Headroom: ~2 GB.

Decode speed drops with context length because blockwise attention iterates over all cached blocks per token, dequantizing each block on the fly. At low context it's fast; at 256k it's compute-bound by the Python dispatch overhead. Future work: Triton kernel to fuse dequant + attention in a single pass.

## Session KV reuse

The server keeps the KV cache alive between requests in the same session. If your next request appends to the same conversation (same message prefix), prefill is skipped - only the new tokens are processed. This means multi-turn chat is fast after the first message.

**Constraint:** only one concurrent conversation per server instance. If you send a request that doesn't share the prefix (edit a prior message, switch conversations), the cache is invalidated and prefill restarts from scratch. For multiple independent sessions, run multiple server instances on different ports.

## Testing

```bash
./test_quantstar.sh
```

End-to-end: starts the server, tests streaming/non-streaming/concurrent, verifies no `<think>` tag leaks.
