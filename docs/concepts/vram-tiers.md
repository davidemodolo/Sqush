# VRAM tiers

Sqush auto‑detects your GPU and picks a model + settings profile. There are two tiers, defined in `VRAM_PROFILES` (`sqush/config.py`).

| VRAM | Model | Download | Disk after bake | Context | Weight bits | KV bits |
|------|-------|----------|-----------------|---------|-------------|---------|
| **8 GB** (LOW) | Qwen3.5‑9B (pre‑quantized) | 8.6 GB | ~6.7 GB | 256k | 4‑bit NF4 | 4‑bit int4 |
| **24 GB** (HIGH) | Qwen3.6‑27B | 52 GB | ~18 GB | 256k | 4‑bit NF4 | 4‑bit int4 |

Classification (`classify_vram`): **≥ 20 GB → HIGH**, otherwise **LOW**. Override with `--vram`.

## What each profile sets

Profiles override only the keys they specify (null entries are skipped):

- **LOW** → `model.repo = techwithsergiu/Qwen3.5-9B-bnb-4bit`, `max_context = 262144`, and image pixel caps `max_image_pixels = 131072`, `min_image_pixels = 16384`.
- **HIGH** → `model.repo = Qwen/Qwen3.6-27B`, `max_context = 262144`.

Both use `weight_bits = 4`, `kv_cache_bits = 4`.

## LOW (8 GB) tier

The 8 GB tier uses [`techwithsergiu/Qwen3.5-9B-bnb-4bit`](https://huggingface.co/techwithsergiu/Qwen3.5-9B-bnb-4bit) — a **pre‑quantized** bitsandbytes NF4 checkpoint, only 8.6 GB to download. Its LM linear weights are already packed 4‑bit; baking handles the two things bitsandbytes leaves in bfloat16:

- **`embed_tokens`** (248320 × 4096, ~1.93 GB bf16) → quantized to a 4‑bit per‑group asymmetric side‑car ([Quantization](../internals/quantization.md#embeddings-4-bit-asymmetric)).
- **`lm_head`** (~2.03 GB bf16, kept unquantized upstream) → NF4‑quantized at load time, saving ~1.45 GB.

### Image pixel cap

On the 8 GB tier images are capped at **131,072 pixels** (≈ 362 × 362) before the vision encoder. Larger images are downscaled. The vision encoder's self‑attention over image patches produces large bf16 activation tensors — at 1024 × 1024 (4096 patches) this blows the budget regardless of weight quantization. At the cap the encoder sees ~512 patches, keeping activation memory in budget.

## HIGH (24 GB) tier

Downloads the full Qwen3.6‑27B (~52 GB bf16) and bakes it once into a compact NF4 checkpoint (~18 GB), deleting the raw. See [Baking](baking.md).

## Baking is common to both

Both tiers serve from a **cooked** checkpoint produced on first run, then delete the raw download. The mechanism differs per tier (side‑car vs. full NF4 re‑serialization) but the outcome is the same: a smaller on‑disk model that loads faster on subsequent starts.
