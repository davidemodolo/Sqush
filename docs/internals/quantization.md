# Weight quantization

Sqush quantizes four things independently, each with the scheme that fits it best. All live in `sqush/quantize.py`.

| Component | Scheme | Why not just bitsandbytes |
|-----------|--------|---------------------------|
| LM linear layers | NF4 (bitsandbytes) | — this is bitsandbytes' job |
| Visual encoder linears | NF4 (bitsandbytes) | left bf16 by pre‑quantized checkpoints; quantized post‑load |
| `lm_head` | NF4 (post‑load) | must stay in the skip list on pre‑quantized checkpoints |
| `embed_tokens` | 4‑bit per‑group **asymmetric** | bitsandbytes doesn't quantize `nn.Embedding` |

## LM weights — NF4

Loaded via `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=bf16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")`. NF4 (4‑bit NormalFloat) is a non‑uniform 4‑bit format matched to normally‑distributed weights; double‑quant further compresses the block scales. For the 27B this is ~16.5 GB of weights.

## `lm_head` — NF4 post‑load (`_quantize_lm_head`)

`lm_head` is 248320 × 4096 (~2.03 GB bf16). It **must remain in `llm_int8_skip_modules`** on a pre‑quantized checkpoint: `from_pretrained` expects any `Linear4bit`'s weights to already be packed 4‑bit + `quant_state` in the shard, so removing `lm_head` from the skip list would load raw bf16 into a `Linear4bit` with no `quant_state` and assert on the first forward. Instead it's loaded as a plain bf16 `nn.Linear` and quantized afterward:

1. Copy weight to CPU as fp16, free the 2 GB GPU tensor (keeps the transient low).
2. Build a `bnb.nn.Linear4bit(..., quant_type="nf4", compress_statistics=True)` with a `Params4bit`.
3. `.to("cuda")` — the move triggers NF4 quantization. Result ~0.57 GB (saves ~1.45 GB).

## Visual encoder — NF4 post‑load (`_quantize_visual_encoder`)

In a pre‑quantized checkpoint all LM linears are already `Linear4bit`; any remaining `nn.Linear` is the visual encoder that bitsandbytes skipped. The function walks the whole module tree (arch‑agnostic), replaces each remaining `Linear` with an NF4 `Linear4bit`, and — if the layer was CPU‑offloaded to a meta tensor — materializes it via `remove_hook_from_module` first, quantizes on GPU, then moves the packed int4 back to CPU. Freed ~X GB bf16 → ~X/4 GB int4.

## Embeddings — 4‑bit asymmetric (`_quantize_embeddings`, `QuantizedEmbedding`) { #embeddings-4-bit-asymmetric }

bitsandbytes does not quantize `nn.Embedding`, so Sqush does it with a **per‑group asymmetric int4** scheme (group size `_EMBED_GROUP_SIZE = 128`, along the hidden dimension):

**Quantize** (per row, per 128‑wide group):

```python
scale = (w_max - w_min).clamp(min=1e-9) / 15.0            # 4-bit unsigned range
zp    = (-w_min / scale).round().clamp(0, 15)             # zero-point
q     = ((w / scale) + zp).round().clamp(0, 15)           # uint4 codes [0,15]
```

Eight `uint4` codes are packed per `int32` word:

```python
packed |= (q[..., i] & 0xF) << (i * 4)   # for i in 0..7
```

Side‑car tensors: `_qw` (packed int32), `_sc` (bf16 scales), `_zp` (int32 zero‑points), plus `_vocab`/`_hidden` metadata.

**Dequantize** (`QuantizedEmbedding.forward`) only touches the rows the indices reference — never the full table:

```python
vals = (qw.unsqueeze(-1) >> (shift * 4)) & 0xF            # unpack 8 nibbles per word
w = (vals - zp) * sc                                      # asymmetric dequant
w = w.reshape(n, padded)[:, :embedding_dim]               # flatten groups, drop padding
```

!!! warning "Un‑padding happens after flattening"
    Padding is added at the **end of the flattened row** (to a multiple of 128), not along the group axis. Dequant flattens `[n, num_groups, 128] → [n, num_groups*128]` and slices to `embedding_dim`. Slicing the 128‑wide group axis instead would be a no‑op that only works when the hidden size is an exact multiple of 128.

## Two embedding code paths

- **Bake‑time** (`_bake_safetensors` in `__main__.py`) — quantizes `embed_tokens` on CPU during the LOW‑tier bake, writing the side‑car and a shard placeholder. Uses matching math.
- **Load‑time** (`_quantize_embeddings`) — used when serving a checkpoint that wasn't pre‑baked.

On a pre‑baked checkpoint (`qs_pre_baked_embeddings = true`), `_load_pre_baked_embeddings` reads the side‑car and installs a `QuantizedEmbedding` — matching by module name ending in `embed_tokens` (the visual encoder also has an `nn.Embedding` for positions, which must not be replaced).

## Memory budget (Qwen3.6‑27B, 24 GB)

- Weights ~16.5 GB (NF4), KV cache ~4.3 GB at 256k (int4, append‑only), ~2 GB headroom.
