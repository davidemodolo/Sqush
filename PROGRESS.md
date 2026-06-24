# QuenStar — back2qwen

**Goal:** Run Qwen3.6-27B at full 256k context on a 24GB RTX 3090 (no RAM/SSD offload).
**Status: 256k works.** Decode speed needs improvement at low/mid context.

## Hardware & env
- RTX 3090 24GB (~23.5 GB usable), 32 GB RAM
- venv `./.venv`, Python 3.14.4, torch 2.12.1+cu126, transformers 5.12.1, bitsandbytes 0.49.2, fla 0.5.1, triton 3.7.1
- Model: `./models/Qwen__Qwen3.6-27B`

## Model (Qwen3.6-27B, `qwen3_5`)
64 layers hybrid: 48 `linear_attention` (DeltaNet, fixed-size state) + 16 `full_attention` (at indices 3,7,...,63). hidden 5120, head_dim 256, 24 attn heads, **4 KV heads (GQA 6:1)**, vocab 248320, max_pos 262144. Only the 16 full-attn layers grow a KV cache.

## What was fixed
Four bugs (found by profiling the real model), all resolved:
1. **lm_head computed logits for all prompt tokens** → pass `logits_to_keep=1` to prefill.
2. **fla not installed** → `pip install flash-linear-attention` (needed for DeltaNet triton kernels).
3. **ninja not on PATH** → `quantize.py:_ensure_cuda_libs()` prepends `.venv/bin`.
4. **KV cache re-quantized every decode step** → custom `Int4AttentionCacheLayer` (append-only 4-bit, per-64-token absmax scales, never re-quantizes stored data).

## The 256k solution — custom blockwise GQA attention

**Root cause of the prefill OOM** (at ~102k): during chunked prefill `kv_length > query_length`, so transformers materializes a 4D causal mask → `use_gqa_in_sdpa` returns False → `repeat_kv` expands K/V 4→24 heads (6× memory). At 131k that's ~3.2 GB/layer transient → OOM.

**Dead ends (all empirically verified):**
- flash-attn: no cp314 wheel (Python 3.14); source build risky.
- FlexAttention: triton register overflow for 24 query heads (non-power-of-2).
- SDPA `enable_gqa`+mask: falls back to math kernel → 24 GB score matrix.

**The fix** (`quantize.py`): reimplemented FlashAttention's online-softmax in PyTorch.
- `blockwise_attention_from_cache()`: K/V stay at 4 heads (no repeat_kv), causality applied per key-block, peak = one `[1,24,q,1024]` score block (~0.4 GB).
- **Block-by-block dequant**: cache exposes `append_only()`+`get_kv_block(start,end)` — dequantizes K/V one block at a time from int4 storage, never materializing the full dequantized KV (1.08 GB at 256k).
- `_patch_qwen_attention()`: monkey-patches `Qwen3_5Attention.forward` to use blockwise for both prefill-with-cache and decode; plain SDPA for first chunk.
- Registered as `attn_implementation="quenstar"` (in `ALL_ATTENTION_FUNCTIONS` only, not mask registry → `create_causal_mask` auto-skips, no 4D mask). Loaded with "sdpa", switched post-load.
- `PREFILL_CHUNK = 1024` (engine.py).

## Verified results

| context | peak VRAM | prefill | decode | response |
|---------|-----------|---------|--------|----------|
| 3k | 17.1 GB | 4s | — | "4" ✓ |
| 16k | 17.3 GB | — | **2.3 tok/s** | ✓ |
| 131k (was OOM) | 19.7 GB | 488s | — | "4" ✓ |
| 256k (target) | **21.1 GB** | 1685s | **0.7 tok/s** | "4" ✓ |

Budget: weights 16.4 + int4 cache 4.3 + blockwise transient 0.4 = ~21.1 GB. ~2.4 GB headroom.

## Improvement ideas (decode speed)

The blockwise decode loops in Python over N blocks × 16 layers per token and re-dequantizes the entire cache each token. At 16k (2.3 tok/s) and 256k (0.7 tok/s) this is the bottleneck. Ideas, easiest first:

1. **Adaptive decode threshold** (~10 lines, biggest low-context win): use full-dequant + SDPA when the dequant transient fits in budget, blockwise only when large. Threshold: if `stored_len × 4 heads × 256 × 2 bytes < 0.5 GB` (≈65k tokens) → `cache.update()` + SDPA with `enable_gqa=True, is_causal=False` (decode q_len=1 needs no mask). At 16k this restores original SDPA speed (~5+ tok/s); at 256k keeps blockwise to avoid OOM. Add a headroom check against `torch.cuda.mem_get_info()`.

2. **Larger decode block size** (~3 lines): at decode q_len=1 the score block is `[1,24,1,block]` = tiny (~1.5 MB even at block=16384). Use `block_size=8192` for decode, `1024` for prefill. At 256k: 32 blocks/layer instead of 256 → 8× fewer Python iterations. Doesn't reduce dequant cost but cuts dispatch overhead.

3. **Incremental dequant cache** (medium): keep the last dequantized block in bf16 across decode steps; each new token only dequants its own group. Invalidated on prefill. Saves re-dequantizing the whole cache every token — the true cost at 256k. ~30 lines, needs careful invalidation.

4. **Triton kernel for blockwise decode** (hard, fastest): fuse dequant + QK^T + online-softmax + AV into one triton kernel. Removes Python loop + PyTorch dispatch entirely. Best perf for 256k decode but most work (~100+ lines, debugging triton).

5. **Build flash-attn from source** (uncertain): `pip install flash-attn --no-build-isolation` (~30 min compile). If it builds on cp314+torch2.12, switch prefill+decode to FA2 → native GQA + causal-offset, no custom attention code needed. May not support Python 3.14 yet.

## Key code locations
- `quantize.py:45` — `_quantize_int4` / `_dequantize_int4`
- `quantize.py:99` — `Int4AttentionCacheLayer` (`append_only`/`get_kv_block` at ~207)
- `quantize.py:309` — `blockwise_gqa_attention` (full-dequant version, reference)
- `quantize.py:380` — `blockwise_attention_from_cache` (block-by-block dequant, the one used)
- `quantize.py:440` — `_patch_qwen_attention` (the patched `Qwen3_5Attention.forward`)
- `quantize.py:553` — `load_and_quantize_model`
- `engine.py:11` — `PREFILL_CHUNK = 1024`
- `engine.py:63` — `_chunked_prefill`

## Git state
- Branch `back2qwen` at `54ff2ff`. `quantize.py`, `engine.py`, `config.yaml` modified in working tree (NOT committed). `PROGRESS.md` untracked.
- `blockwise_gqa_attention` and `quenstar_attention_forward` are dead code (the patched forward bypasses them) — can remove.
