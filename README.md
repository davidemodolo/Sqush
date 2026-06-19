# QuenStar v2

**Qwen3.6-27B quantized inference in 24 GB VRAM.**

- 4-bit weight quantization (bitsandbytes NF4)
- Flash Attention via PyTorch SDPA (cuDNN backend)
- OpenAI-compatible API server
- Interactive CLI chat

## Memory Budget

| Component | Size |
|-----------|------|
| Weights (4-bit NF4) | ~16.5 GB |
| KV cache per 1k tokens | ~64 KB |
| Full 64k context KV cache | ~4.2 GB |
| **Total at 64k context** | **~21 GB** |

## Quickstart

```bash
./run.sh download    # download Qwen3.6-27B (52 GB, one-time)
./run.sh chat        # interactive CLI
./run.sh serve       # OpenAI API on :9898
```

## Configuration

Edit `config.yaml` or use environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `QUENSTAR_MODEL_REPO` | `Qwen/Qwen3.6-27B` | HuggingFace model |
| `QUENSTAR_WEIGHT_BITS` | `4` | bitsandbytes 4-bit |
| `QUENSTAR_MAX_CONTEXT` | `65536` | Max context length |
| `QUENSTAR_HOST` | `127.0.0.1` | Server host |
| `QUENSTAR_PORT` | `9898` | Server port |

## API

OpenAI-compatible:

- `GET /v1/models`
- `POST /v1/chat/completions` (streaming + non-streaming)
- `GET /health`
- `GET /health/vram`

## Context length vs memory

| Context | KV cache | Total VRAM |
|---------|----------|------------|
| 32k | 2.1 GB | 18.6 GB |
| 64k | 4.2 GB | 20.7 GB |
| 96k | 6.3 GB | 22.8 GB |
| 128k | 8.4 GB | 24.9 GB |

Full 262k context requires KV cache quantization (not yet supported for Qwen3.6's hybrid architecture).
