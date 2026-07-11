# Blockwise GQA attention

All line references are into `sqush/quantize.py`.

## Why a custom kernel (lines 340–352)

No stock attention kernel serves **chunked prefill of a 256k context under GQA on limited VRAM**:

- **SDPA + GQA + mask** — during chunked prefill the query is a *suffix* of the cached keys (`offset = kv_len - q_len > 0`), needing a causal‑with‑offset mask. SDPA with `enable_gqa=True` **and** an explicit mask falls back to the math kernel, which materializes the full `[B, nq, q_len, kv_len]` score matrix → OOM at 256k.
- **`is_causal=True`** — wrong when `offset > 0` (assumes query/key co‑aligned).
- **FlashAttention‑2** — no cp314 wheel.
- **FlexAttention** — overflows Triton registers for 24 non‑power‑of‑2 query heads.

The solution is a hand‑rolled FlashAttention‑style **online‑softmax** in pure PyTorch that keeps K/V at `nkv` heads (no `repeat_kv`), applies causality per key‑block, and caps peak memory at one `[B, nq, q_len, block]` score tile. `_BLOCK_SIZE = 1024` (line 352) is the key‑block width.

## `blockwise_gqa_attention` (lines 355–407)

```python
def blockwise_gqa_attention(query, key, value, scaling, block_size=_BLOCK_SIZE)
```

| Tensor | Shape |
|--------|-------|
| `query` | `[B, nq, q_len, hd]` |
| `key`, `value` | `[B, nkv, kv_len, hd]` |
| return | `[B, nq, q_len, hd]` |

Derived: `G = nq // nkv` (GQA group size — query heads per KV head), `offset = kv_len - q_len` (absolute position of the first query token).

**Online‑softmax running state** (fp32 for stability, regardless of input dtype):

- `m` — running row max, `[B, nq, q_len, 1]`, init `-inf`.
- `l` — running softmax denominator, `[B, nq, q_len, 1]`, init `0`.
- `num` — unnormalized output accumulator, `[B, nq, q_len, hd]`, init `0`.

**GQA reshape without repeat** (line 375): `qg = query.view(B, nkv, G, q_len, hd)`. This is the crux of the memory saving — instead of expanding K/V to `nq` heads (×G memory), the query is grouped so a batched matmul against the unexpanded `[B, nkv, …]` K/V broadcasts the shared KV head across its `G` query heads.

**Per key‑block `[start:end)` of width `Bl`:**

Scores:
```python
kb = key[:, :, start:end, :]                              # [B, nkv, Bl, hd]
scores = matmul(qg, kb.unsqueeze(2).transpose(-1,-2)) * scaling
scores = scores.view(B, nq, q_len, Bl).float()
```

Causal‑with‑offset mask:
```python
key_pos = arange(start, end)          # absolute key positions
q_pos   = arange(q_len)
causal  = (q_pos + offset) >= key_pos # query i keeps key j iff i+offset >= j
scores  = scores.masked_fill(~causal, float("-inf"))
```

Online‑softmax update (the standard FlashAttention rescaling):
```python
mb    = scores.amax(-1, keepdim=True)
m_new = maximum(m, mb)
p     = exp(scores - m_new)
pg    = p.view(B, nkv, G, q_len, Bl)                      # regroup for the value matmul
block_num = matmul(pg, vb.unsqueeze(2).float()).view(B, nq, q_len, hd)
alpha = exp(m - m_new)                                    # rescale prior accumulator
num = num * alpha + block_num
l   = l   * alpha + p.sum(-1, keepdim=True)
m   = m_new
```

`alpha` rescales the previously accumulated numerator/denominator to the new maximum, making streaming softmax exact regardless of block order; all exponentials use the *global* running max so nothing overflows.

**Final normalization**: `out = (num / l.clamp(min=1e-20)).to(dtype)`. Peak memory is one `[B, nq, q_len, Bl]` fp32 tile — never the full `[B, nq, q_len, kv_len]`.

## `sqush_attention_forward` (lines 410–436)

The function registered as a transformers attention implementation. It receives the standard interface (`module`, projected `query/key/value` in `[B, heads, seq, hd]`, `attention_mask`, `dropout`, `scaling`, `is_causal`). `scale = scaling if scaling is not None else hd**-0.5`.

Dispatch:

- **Blockwise path** — when `has_gqa and q_len > 1 and kv_len > q_len` (GQA cached prefill, query is a suffix). Calls `blockwise_gqa_attention`, transposes to `[B, q_len, nq, hd]`, returns `(out, None)`.
- **SDPA path** — decode (`q_len == 1`) or first chunk (`kv_len == q_len`, offset 0). Uses `F.scaled_dot_product_attention` with `enable_gqa=True`, `is_causal = q_len > 1 and attention_mask is None and is_causal`. The 4D causal mask is never materialized, so `attention_mask` stays `None` — which is exactly what keeps GQA active (avoiding the mask→math fallback).

## `blockwise_attention_from_cache` (lines 439–487)

Same online‑softmax algorithm, but K/V are **dequantized one block at a time straight from the `Int4AttentionCacheLayer`** — the full dequant transient never exists.

```python
def blockwise_attention_from_cache(query, cache_layer, scaling, block_size=_BLOCK_SIZE)
```

`nkv = cache_layer._n_kv_heads`, `kv_len = cache_layer.stored_length()`; `G`, `offset`, running state, and the `qg` reshape are identical to the dense version. The only difference is the K/V source:

```python
for start in range(0, kv_len, block_size):
    end = min(start + block_size, kv_len)
    kb, vb = cache_layer.get_kv_block(start, end)   # [1, nkv, Bl, hd] bf16, dequantized on the fly
    # ... identical scores / mask / online-softmax update ...
```

`get_kv_block` dequantizes only `[start:end)` — the packed int4 part via `_dequant_packed_slice` (group‑aligned dequant then slice) and the bf16 pending part sliced directly. The cache's leading batch dim of 1 broadcasts across the query batch. This is the mechanism that lets a 256k‑token quantized cache be attended to without ever holding the dense KV.

## Injection: two layers

### `_patch_qwen_attention` (lines 490–549)

Monkeypatches `Qwen3_5Attention.forward` directly (idempotent via `_sqush_patched`). It reproduces the stock forward — QKV projection, the fused query/gate chunk (`q_proj` outputs `head_dim*2`, split into `query_states` and `gate`), `q_norm`/`k_norm`, transpose, `apply_rotary_pos_emb` — then:

```python
cache_layer = past_key_values.layers[self.layer_idx] if past_key_values else None
use_blockwise = (has_gqa and cache_layer is not None
                 and isinstance(cache_layer, Int4AttentionCacheLayer)
                 and cache_layer.is_initialized)
```

- **Blockwise branch** — `cache_layer.append_only(k, v)` then `blockwise_attention_from_cache(...)`. Used for **both** cached prefill (offset > 0) **and** decode (`q_len == 1`), because either would otherwise materialize the full dequantized KV → OOM at 256k. (Broader than `sqush_attention_forward`, which took SDPA for `q_len == 1`.)
- **SDPA fallback** — no int4 cache (first chunk before init, or non‑GQA): normal `past_key_values.update(...)` + SDPA.
- **Output** — reshape, apply Qwen3.5's **sigmoid gate** `attn_output * sigmoid(gate)`, then `o_proj`.

### `_register_sqush_attention` (lines 555–558)

Registers `sqush_attention_forward` as `"sqush"` in `ALL_ATTENTION_FUNCTIONS`, so `attn_implementation="sqush"` routes through it.

### Why both

- The **registry** is the clean, config‑selectable path handling the dense case and keeping GQA alive on SDPA.
- The **monkeypatch** is required because int4 dequant‑on‑the‑fly must happen *inside* the attention layer, around the cache append, and must cover decode — which the registry function (receiving already‑materialized K/V) cannot.

Together, at every phase — first chunk (SDPA), cached prefill (blockwise), decode (blockwise) — the model never materializes the full score matrix nor the full dequantized KV. That is what makes 256k context feasible on limited VRAM.
