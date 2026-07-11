# int4 KV cache

All line references are into `sqush/quantize.py`.

The KV cache stores keys/values at **4 bits** for the full‑attention layers of Qwen3.5's hybrid architecture. Goals: cut KV memory ~4× vs bf16, **never re‑quantize** already‑stored tokens (append‑only), and allow block‑by‑block dequantization so the full KV is never materialized (no OOM at 256k).

## Quantization primitives

`_GROUP_SIZE = 64` (line 72) — the quantization group is 64 tokens along the sequence axis.

### `_quantize_int4` (lines 75–105)

`_quantize_int4(tensor, group_size=64)` where `tensor` is `[1, n_heads, seq_len, head_dim]` bf16. Scheme: **symmetric, signed int4, per‑group absmax scaling**, grouped along the sequence axis in blocks of 64 tokens; each (group, head, head_dim) triple gets its own scale.

1. **Drop batch dim** (`t = tensor[0]`) → `[n_heads, seq_len, head_dim]` (batch assumed 1).
2. **Pad sequence to a multiple of 64** with zeros: `pad = (64 - seq_len % 64) % 64`.
3. **Reshape into groups** → `[n_heads, num_groups, 64, head_dim]`.
4. **Per‑group symmetric scale**: `scale = t.abs().amax(dim=2, keepdim=True).clamp(min=1e-8) / 7.0`. Divided by **7.0** (the positive int4 max) so the largest magnitude maps to ±7; `clamp` avoids divide‑by‑zero.
5. **Quantize**: `q = round(t / scale).clamp(-8, 7).to(int8)` — signed int4 range `[-8, 7]`.
6. **Relayout token‑major**: packed data → `[padded_len, n_heads, head_dim]`, scales → `[num_groups, n_heads, head_dim]`. Making axis 0 the token/group axis means appending is a `torch.cat` on dim 0.
7. **Nibble‑pack along head_dim**: two int4 per byte. `q_u = (q + 8).to(uint8)` biases signed→unsigned `[0,15]`; `packed = (q_u[...,0] << 4) | q_u[...,1]` — even index in the high nibble, odd in the low.

Returns:

| Tensor | Shape | Dtype |
|--------|-------|-------|
| `packed` | `[padded_len, n_heads, head_dim//2]` | uint8 |
| `scales` | `[num_groups, n_heads, head_dim]` | bf16 |
| `seq_len` | scalar int | — |

bf16 KV is 2 bytes/value; packed is 0.5 bytes/value → ~4× reduction. Scale overhead is amortized over 64 tokens (~0.03 bytes/value) — negligible.

### `_dequantize_int4` (lines 108–126)

`_dequantize_int4(packed, scales, num_tokens, group_size=64)` → `[1, n_heads, num_tokens, head_dim]` bf16.

1. **Unpack nibbles**: `high = (packed >> 4).to(int8) - 8` (even indices), `low = (packed & 0x0F).to(int8) - 8` (odd); `stack([high, low], dim=-1)` interleaves back to the original head_dim order. The `-8` undoes the pack‑time bias.
2. **Apply scales per group** (broadcast one scale over all 64 tokens): `q = q * scales.unsqueeze(1)`. Pure symmetric — no zero‑point.
3. **Trim padding & restore layout**: slice `[:num_tokens]`, `transpose(0,1).unsqueeze(0)` → `[1, n_heads, num_tokens, head_dim]`.

!!! note "KV is symmetric; embeddings are asymmetric"
    The KV scheme uses scale only. The embedding quantizer (`_quantize_embeddings`) is a separate, **asymmetric** int4 scheme (scale + zero‑point, 8 values per int32, group size 128).

## `Int4AttentionCacheLayer` (lines 129–307)

A per‑layer `CacheLayerMixin`. **Append‑only**: quantized tokens are never touched again. It keeps a bf16 *pending* buffer for tail tokens that don't yet fill a complete 64‑token group.

### State (lines 137–151)

| Attribute | Shape / type | Purpose |
|-----------|--------------|---------|
| `_packed_keys` / `_packed_values` | `[num_quantized, n_kv_heads, head_dim//2]` uint8 | flushed, quantized K/V |
| `_keys_scales` / `_values_scales` | `[num_quantized//64, n_kv_heads, head_dim]` bf16 | per‑group scales |
| `_pending_keys` / `_pending_values` | `[1, n_kv_heads, pending_len, head_dim]` bf16 | tail tokens not yet a full group |
| `_num_quantized` | int | tokens already packed |
| `_n_kv_heads`, `_head_dim` | int | KV geometry |

The stored sequence is conceptually `_packed_* (first _num_quantized tokens, quantized)` followed by `_pending_* (remainder, still bf16)`.

### Append path — `update` (lines 175–196)

1. Lazy‑init if needed.
2. **Accumulate into pending** — `torch.cat` new states on the seq axis.
3. **Flush complete groups**: `n_groups = pending_len // 64`; if > 0, quantize the largest 64‑aligned slice via `_quantize_and_append` and keep the < 64‑token remainder in bf16.
4. Return the full dequantized KV via `_build_full_kv`.

**Why append‑only works**: flushed groups are exactly group‑aligned so their scales never change; the partial tail (whose scale *would* change as tokens arrive) is deliberately kept bf16 until it completes a group.

### Other methods

- `_quantize_and_append` (198–213) — quantizes the flush slice for K and V independently, `cat`s onto dim 0 (cheap append), increments `_num_quantized`.
- `_build_full_kv` (215–234) — reconstructs full `[1, n_kv_heads, total, head_dim]` bf16 by concatenating the dequantized packed part and the bf16 pending tail. This dequant is transient (used for the SDPA fallback, then freed).
- `get_seq_length` (236–242) — `_num_quantized + pending_len`.
- `get_mask_sizes` (244–245) — `(get_seq_length() + query_length, 0)`.

### Block‑by‑block dequant API (lines 247–307)

Used by the patched attention so the full dequantized KV is never built:

- `append_only` (249–267) — same accumulate + flush as `update`, but returns nothing.
- `stored_length` (269–270) — alias for `get_seq_length`.
- `get_kv_block(start, end)` (272–292) — dequantizes only tokens `[start, end)`, stitching the packed part (`_dequant_packed_slice`) and the bf16 pending part.
- `_dequant_packed_slice` (294–307) — `start`/`end` need not be group‑aligned; it dequantizes the covering group range then slices to the exact window. This lets attention request arbitrary 1024‑token blocks while dequant stays group‑aligned.

## `SqushKVCache` (lines 310–336)

A `transformers.cache_utils.Cache` subclass assembling a **heterogeneous list** of per‑layer caches for Qwen3.5's mixed architecture (the docstring cites 48 linear‑attention + 16 full‑attention layers for the 27B):

- **linear_attention / conv / mamba / moe / hybrid layers** → stock `LinearAttentionLayer`.
- **everything else (e.g. `full_attention`)** → `Int4AttentionCacheLayer`.

Construction reads `layer_types`, `num_key_value_heads`, and `head_dim` from the decoder text config; each int4 layer is eagerly `pre_init`ed with geometry/dtype/device when available, otherwise lazy‑initialized on first `update`. `SqushKVCache` implements no per‑layer methods itself — the standard `Cache` dispatch (`self.layers[i].update(...)`, `get_seq_length`, …) routes to the right class.

Instances come from `_make_cache_factory(model)` (576–589), which closes over the model's config/dtype/device and returns a zero‑arg `factory()`, wired in at `load_and_quantize_model` (line 804).

## Runtime summary

During prefill/decode the patched `Qwen3_5Attention.forward` uses the block path whenever the int4 layer is initialized: `append_only(k, v)` stores new tokens (full groups → int4, tail → bf16), then `blockwise_attention_from_cache` loops over 1024‑token key blocks, calling `get_kv_block` to dequantize just that block. The full dequantized KV is **never** materialized — only one score block and one dequantized K/V block exist at a time. See [Attention](attention.md).
