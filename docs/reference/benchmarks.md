# Benchmarks

All runs: Python 3.14, torch 2.12.1+cu126, 4‑bit NF4 weights, int4 KV cache.

## Qwen3.6‑27B (24 GB VRAM, RTX 3090)

| Context | Peak VRAM | Prefill | Decode |
|---------|-----------|---------|--------|
| 3k | 17.1 GB | 3.0 s | 11.0 tok/s |
| 16k | 17.3 GB | 20.8 s | 6.7 tok/s |
| 32k | 17.6 GB | 52.9 s | 4.4 tok/s |
| 64k | 18.1 GB | 151.1 s | 2.6 tok/s |
| 128k | 19.1 GB | 480.3 s | 1.4 tok/s |
| 256k | 21.2 GB | 1649.6 s | 0.8 tok/s |

Weights ~16.5 GB (NF4). KV cache ~4.3 GB at 256k (int4, append‑only). Headroom ~2 GB.

## Qwen3.5‑9B (8 GB VRAM, RTX 4060 Ti)

| Target | Actual | Prefill | Decode | Peak VRAM |
|--------|--------|---------|--------|-----------|
| 3,000 | 2,986 | 1.4 s | 19.5 t/s | 5.6 GB |
| 16,000 | 15,986 | 8.9 s | 12.4 t/s | 5.7 GB |
| 32,000 | 31,986 | 22.5 s | 8.6 t/s | 5.8 GB |
| 64,000 | 63,986 | 61.9 s | 5.3 t/s | 6.1 GB |
| 128,000 | 127,986 | 185.9 s | 3.0 t/s | 6.6 GB |
| 256,000 | 255,986 | 661.9 s | 1.3 t/s | 7.6 GB |

Weights ~5.5 GB (NF4 + embedding quantization). KV cache ~2.0 GB at 256k. 256k fits in 8 GB.

## Why decode slows with context

Blockwise attention iterates over all cached 1024‑token blocks per token, dequantizing each on the fly. At low context it's fast; at 256k it's compute‑bound by Python dispatch overhead over the block loop. Future work: a Triton kernel fusing dequant + attention in a single pass.

## Bake timing (validated)

On an RTX 3090, baking the 27B (bf16 → NF4 `save_pretrained`) takes ~50 s and produces a 17.9 GB cooked checkpoint from the 52 GB raw; reload hits the pre‑quantized fast path at 17.9 GB peak.
