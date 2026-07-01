# QuantStar

**Run Qwen models quantized on a single GPU — 8 GB or 24 GB VRAM.**

4-bit weights + 4-bit KV cache + blockwise attention.
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

## VRAM Tiers

QuantStar auto-detects your GPU and picks the right model and settings:

| VRAM | Model | Download | Context | Weight Bits | KV Bits |
|------|-------|----------|---------|-------------|---------|
| 8 GB | Qwen3.5-9B (pre-quantized) | 8.6 GB | 128k | 4-bit NF4 | 4-bit int4 |
| 24 GB | Qwen3.6-27B | 52 GB | 256k | 4-bit NF4 | 4-bit int4 |

Override auto-detection with `--vram`:

```bash
./run.sh --vram 8 serve      # force 8GB profile
python -m quantstar --vram 8 serve
```

The 8 GB tier uses [`techwithsergiu/Qwen3.5-9B-bnb-4bit`](https://huggingface.co/techwithsergiu/Qwen3.5-9B-bnb-4bit) — a pre-quantized bitsandbytes NF4 checkpoint, just 8.6 GB to download. On first run, QuantStar *bakes* it: the embedding table (`embed_tokens`, 1.93 GB in bfloat16) is quantized to 4-bit per-group asymmetric format and saved as a side-car file. The baked model is cached; subsequent starts load it directly. bitsandbytes does not quantize `nn.Embedding` layers, so QuantStar handles that itself. The `lm_head` projection (another 2.03 GB in bfloat16, kept unquantized by the upstream checkpoint) is quantized to NF4 at load time, saving a further ~1.45 GB. Together this keeps loading and inference well within the 8 GB budget, leaving headroom for long contexts.

Image inputs on the 8 GB tier are capped at 131,072 pixels (≈ 362 × 362) before being passed to the vision encoder. Images larger than this are downscaled to fit. This is necessary because the vision encoder's self-attention over image patches generates large bfloat16 activation tensors — at full 1024 × 1024 resolution (4096 patches) this exceeds the VRAM budget regardless of weight quantization. At the cap, the vision encoder sees 512 patches instead of 4096, keeping activation memory within budget.

The 24 GB tier downloads the full Qwen3.6-27B model and quantizes it at load time using bitsandbytes NF4.

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

Images are processed by Qwen3.6-VL's vision encoder via chunked prefill. The resulting KV cache is saved, so follow-up text messages in the same conversation reuse it without re-running the vision encoder. Only base64 data URLs are supported — remote URLs are not yet handled.

## Text Performance

All benchmarks: Python 3.14, torch 2.12.1+cu126, 4-bit NF4 weights, int4 KV cache.

### Qwen3.6-27B (24 GB VRAM, RTX 3090)

| Context | Peak VRAM | Prefill | Decode |
|---------|-----------|---------|--------|
| 3k | 17.1 GB | 3.0s | 11.0 tok/s |
| 16k | 17.3 GB | 20.8s | 6.7 tok/s |
| 32k | 17.6 GB | 52.9s | 4.4 tok/s |
| 64k | 18.1 GB | 151.1s | 2.6 tok/s |
| 128k | 19.1 GB | 480.3s | 1.4 tok/s |
| 256k | 21.2 GB | 1649.6s | 0.8 tok/s |

Weights: ~16.5 GB (NF4). KV cache: ~4.3 GB at 256k (int4, append-only). Headroom: ~2 GB.

### Qwen3.5-9B (8 GB VRAM, RTX 4060 Ti)

| Target | Actual | Prefill | Decode | Peak VRAM |
|--------|--------|---------|--------|-----------|
| 3,000 | 2,986 | 1.4s | 19.5t/s | 5.6 GB |
| 16,000 | 15,986 | 8.9s | 12.4t/s | 5.7 GB |
| 32,000 | 31,986 | 22.5s | 8.6t/s | 5.8 GB |
| 64,000 | 63,986 | 61.9s | 5.3t/s | 6.1 GB |
| 128,000 | 127,986 | 185.9s | 3.0t/s | 6.6 GB |
| 256,000 | 255,986 | 661.9s | 1.3t/s | 7.6 GB |

Weights: ~5.5 GB (NF4 + embedding quantization). KV cache: ~2.0 GB at 256k. 256k fits within 8 GB VRAM.

Decode speed drops with context length because blockwise attention iterates over all cached blocks per token, dequantizing each block on the fly. At low context it's fast; at 256k it's compute-bound by the Python dispatch overhead. Future work: Triton kernel to fuse dequant + attention in a single pass.

## Session KV reuse

The server keeps the KV cache alive between requests in the same session. If your next request appends to the same conversation, prefill is skipped — only the new tokens are processed. This makes multi-turn chat fast after the first message.

This applies to both text-only and vision conversations. When an image appears in the conversation history, the vision encoder runs once for that turn and the resulting KV states are cached; subsequent text follow-ups extend the same cache rather than re-processing the image.

Reuse is verified at the token level: the new prompt must start with exactly the tokens the cache was built from. For a hit on thinking turns, the client must send the previous reply's reasoning back (`reasoning_content` on the assistant message, as returned by the API) — if the reasoning is dropped, the re-rendered history no longer matches the cached tokens and prefill restarts from scratch.

**Constraint:** one concurrent conversation per server instance. If you send a request that doesn't share the prefix (editing a prior message, switching conversations, or a shorter history than what was cached), the cache is invalidated and prefill restarts. For multiple independent sessions, run multiple server instances on different ports.

## Testing

The test suite runs entirely without loading a model — all GPU-dependent code is replaced with mocks, so tests run on CPU in a few seconds and are safe to run in CI or on any machine.

```bash
pip install -e ".[test]"
python -m pytest tests/
```

The suite covers:

- **KV cache math** — int4 quantization, packed uint8 layout, round-trip accuracy, group-boundary edge cases, `QuantStarKVCache` layer structure
- **Blockwise GQA attention** — first prefill, cached prefill with offset, decode step, causal masking, GQA grouping, numerical stability, `blockwise_attention_from_cache`
- **Inference engine** — image extraction, message preprocessing, tokenization paths, `_prepare_generation` kwargs, session KV cache reuse (text and vision), stream and sync paths
- **FastAPI server** — health/VRAM endpoints, models list, sync and streaming completions, SSE format, CORS, thinking/reasoning content, tool call parsing and streaming
- **Config loading** — YAML overrides, env var precedence, defaults
- **CLI** — history trimming, `<think>` block stripping from display output
- **`__main__`** — OpenCode config init and path resolution

For end-to-end smoke testing against a live server (requires a GPU and the downloaded model):

```bash
./test_quantstar.sh
```

This starts the server, tests streaming and non-streaming requests, concurrent load, and verifies no `<think>` tags leak into content fields.
