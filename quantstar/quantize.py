from __future__ import annotations

import gc
import logging
import os
import sys
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from transformers.cache_utils import Cache, CacheLayerMixin, LinearAttentionLayer


# ---------------------------------------------------------------------------
# Triton autotuner race fix
# ---------------------------------------------------------------------------
# FLA kernels decorated with fla_cache_autotune (CachedAutotuner) can hit a
# triton race where Autotuner._bench reads self.nargs and finds None — even
# during a single-threaded call. The warmup cannot cover every sequence-length
# shape, so disable autotuning at runtime: when self.nargs is unexpectedly
# None, fall back to {} (empty positional args) and let the benchmark run.
# This is safe because FLA kernels pass all arguments as kwargs.

def _patch_triton_autotuner():
    from triton.runtime.autotuner import Autotuner

    if getattr(Autotuner, "_quantstar_nargs_fixed", False):
        return

    _original_bench = Autotuner._bench

    def _safe_bench(self, *args, config, **meta):
        if self.nargs is None:
            self.nargs = {}
        return _original_bench(self, *args, config=config, **meta)

    Autotuner._bench = _safe_bench
    Autotuner._quantstar_nargs_fixed = True


_patch_triton_autotuner()

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # avoids OOM from allocator fragmentation with large cache blocks

log = logging.getLogger(__name__)


def _ensure_cuda_libs():
    # bitsandbytes needs the CUDA 13 runtime lib. Search known local venv paths
    # and prepend to LD_LIBRARY_PATH. Also ensure ninja is on PATH for triton JIT.
    cu13_paths = [
        os.path.expanduser("~/.local/lib/python3.14/site-packages/nvidia/cu13/lib"),
        "/home/davide/Documents/Dev/genai_img_audio/.venv/lib/python3.14/site-packages/nvidia/cu13/lib",
        "/home/davide/Documents/Dev/alphamon/.venv/lib/python3.14/site-packages/nvidia/cu13/lib",
    ]
    for p in cu13_paths:
        if os.path.isdir(p) and p not in os.environ.get("LD_LIBRARY_PATH", ""):
            os.environ["LD_LIBRARY_PATH"] = p + ":" + os.environ.get("LD_LIBRARY_PATH", "")
            break

    venv_bin = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".venv", "bin")
    if os.path.isdir(venv_bin) and venv_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = venv_bin + ":" + os.environ.get("PATH", "")


_ensure_cuda_libs()


# ---------------------------------------------------------------------------
# Int4 KV cache — append-only, no re-quantization
# ---------------------------------------------------------------------------

_GROUP_SIZE = 64


def _quantize_int4(tensor: torch.Tensor, group_size: int = _GROUP_SIZE):
    """Quantize K/V tensor to 4-bit packed in uint8.

    Args:
        tensor: [1, n_heads, seq_len, head_dim] bf16
    Returns:
        packed: [padded_len, n_heads, head_dim//2] uint8 (2 int4 per byte, along head_dim)
        scales: [num_groups, n_heads, head_dim] bf16
        seq_len: int (original, before padding)
    """
    t = tensor[0]
    n_heads, seq_len, head_dim = t.shape

    pad = (group_size - seq_len % group_size) % group_size
    if pad:
        t = F.pad(t, (0, 0, 0, pad))
    padded_len = t.shape[1]
    num_groups = padded_len // group_size

    t = t.reshape(n_heads, num_groups, group_size, head_dim)
    scale = t.abs().amax(dim=2, keepdim=True).clamp(min=1e-8) / 7.0

    q = torch.round(t / scale).clamp(-8, 7).to(torch.int8)
    q = q.reshape(n_heads, padded_len, head_dim).transpose(0, 1).contiguous()
    scale = scale.squeeze(2).transpose(0, 1).contiguous()

    q = q.reshape(padded_len, n_heads, head_dim // 2, 2)
    q_u = (q + 8).to(torch.uint8)
    packed = (q_u[..., 0] << 4) | q_u[..., 1]

    return packed, scale, seq_len


def _dequantize_int4(packed: torch.Tensor, scales: torch.Tensor, num_tokens: int,
                     group_size: int = _GROUP_SIZE) -> torch.Tensor:
    """Dequantize packed int4 back to bf16.

    Returns: [1, n_heads, num_tokens, head_dim] bf16
    """
    padded_len, n_heads, half_dim = packed.shape
    head_dim = half_dim * 2
    num_groups = padded_len // group_size

    high = (packed >> 4).to(torch.int8) - 8
    low = (packed & 0x0F).to(torch.int8) - 8
    q = torch.stack([high, low], dim=-1).reshape(padded_len, n_heads, head_dim)

    q = q.reshape(num_groups, group_size, n_heads, head_dim)
    q = q.to(scales.dtype) * scales.unsqueeze(1)

    q = q.reshape(padded_len, n_heads, head_dim)[:num_tokens]
    return q.transpose(0, 1).unsqueeze(0).contiguous()


class Int4AttentionCacheLayer(CacheLayerMixin):
    """4-bit KV cache for full-attention layers. Append-only, never re-quantizes.

    Stores K/V as packed int4 (uint8) with per-group-of-64-token absmax scaling.
    New tokens are quantized and appended; existing data is never touched.
    On update(), dequantizes all stored K/V to bf16 for SDPA (transient, freed after layer).
    """

    def __init__(self, group_size: int = _GROUP_SIZE):
        super().__init__()
        self.group_size = group_size
        self._packed_keys: Optional[torch.Tensor] = None
        self._packed_values: Optional[torch.Tensor] = None
        self._keys_scales: Optional[torch.Tensor] = None
        self._values_scales: Optional[torch.Tensor] = None
        self._pending_keys: Optional[torch.Tensor] = None
        self._pending_values: Optional[torch.Tensor] = None
        self._num_quantized = 0
        self._n_kv_heads: Optional[int] = None
        self._head_dim: Optional[int] = None
        self.is_initialized = False
        self.dtype = None
        self.device = None

    def _lazy_init(self, key_states: torch.Tensor, value_states: torch.Tensor):
        self.dtype = key_states.dtype
        self.device = key_states.device
        self._n_kv_heads = key_states.shape[1]
        self._head_dim = key_states.shape[-1]
        self.is_initialized = True

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self._lazy_init(key_states, value_states)

    def get_max_cache_shape(self) -> int:
        return -1

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs):
        if not self.is_initialized:
            self._lazy_init(key_states, value_states)

        if self._pending_keys is None:
            self._pending_keys = key_states
            self._pending_values = value_states
        else:
            self._pending_keys = torch.cat([self._pending_keys, key_states], dim=-2)
            self._pending_values = torch.cat([self._pending_values, value_states], dim=-2)

        pending_len = self._pending_keys.shape[-2]
        n_groups = pending_len // self.group_size
        if n_groups > 0:
            flush_len = n_groups * self.group_size
            fk = self._pending_keys[:, :, :flush_len, :]
            fv = self._pending_values[:, :, :flush_len, :]
            self._quantize_and_append(fk, fv)
            self._pending_keys = self._pending_keys[:, :, flush_len:, :]
            self._pending_values = self._pending_values[:, :, flush_len:, :]

        return self._build_full_kv()

    def _quantize_and_append(self, keys: torch.Tensor, values: torch.Tensor):
        pk, sk, _ = _quantize_int4(keys, self.group_size)
        pv, sv, _ = _quantize_int4(values, self.group_size)

        if self._packed_keys is None:
            self._packed_keys = pk
            self._keys_scales = sk
            self._packed_values = pv
            self._values_scales = sv
        else:
            self._packed_keys = torch.cat([self._packed_keys, pk], dim=0)
            self._keys_scales = torch.cat([self._keys_scales, sk], dim=0)
            self._packed_values = torch.cat([self._packed_values, pv], dim=0)
            self._values_scales = torch.cat([self._values_scales, sv], dim=0)

        self._num_quantized += keys.shape[-2]

    def _build_full_kv(self) -> tuple[torch.Tensor, torch.Tensor]:
        parts_k = []
        if self._packed_keys is not None:
            parts_k.append(_dequantize_int4(self._packed_keys, self._keys_scales, self._num_quantized, self.group_size))
        if self._pending_keys is not None and self._pending_keys.shape[-2] > 0:
            parts_k.append(self._pending_keys)

        parts_v = []
        if self._packed_values is not None:
            parts_v.append(_dequantize_int4(self._packed_values, self._values_scales, self._num_quantized, self.group_size))
        if self._pending_values is not None and self._pending_values.shape[-2] > 0:
            parts_v.append(self._pending_values)

        if parts_k:
            return torch.cat(parts_k, dim=-2), torch.cat(parts_v, dim=-2)

        empty = (1, self._n_kv_heads, 0, self._head_dim)
        k = torch.zeros(*empty, dtype=self.dtype, device=self.device)
        v = torch.zeros(*empty, dtype=self.dtype, device=self.device)
        return k, v

    def get_seq_length(self) -> int:
        if not self.is_initialized:
            return 0
        total = self._num_quantized
        if self._pending_keys is not None:
            total += self._pending_keys.shape[-2]
        return total

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self.get_seq_length() + query_length, 0

    # --- block-by-block dequant API (avoids materializing the full KV) ---

    def append_only(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        """Append new tokens to int4 storage without dequantizing the full KV."""
        if not self.is_initialized:
            self._lazy_init(key_states, value_states)
        if self._pending_keys is None:
            self._pending_keys = key_states
            self._pending_values = value_states
        else:
            self._pending_keys = torch.cat([self._pending_keys, key_states], dim=-2)
            self._pending_values = torch.cat([self._pending_values, value_states], dim=-2)
        pending_len = self._pending_keys.shape[-2]
        n_groups = pending_len // self.group_size
        if n_groups > 0:
            flush_len = n_groups * self.group_size
            fk = self._pending_keys[:, :, :flush_len, :]
            fv = self._pending_values[:, :, :flush_len, :]
            self._quantize_and_append(fk, fv)
            self._pending_keys = self._pending_keys[:, :, flush_len:, :]
            self._pending_values = self._pending_values[:, :, flush_len:, :]

    def stored_length(self) -> int:
        return self.get_seq_length()

    def get_kv_block(self, start: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Dequantize tokens [start:end) only. start/end need not be group-aligned
        for the packed part (handled internally). Returns [1, nkv, end-start, hd]."""
        nq = self._num_quantized
        parts_k, parts_v = [], []
        # packed part
        if start < nq and self._packed_keys is not None:
            p_end = min(end, nq)
            parts_k.append(self._dequant_packed_slice(self._packed_keys, self._keys_scales, start, p_end))
            parts_v.append(self._dequant_packed_slice(self._packed_values, self._values_scales, start, p_end))
        # pending (bf16) part
        if end > nq and self._pending_keys is not None and self._pending_keys.shape[-2] > 0:
            ps = max(start, nq) - nq
            pe = end - nq
            parts_k.append(self._pending_keys[:, :, ps:pe, :])
            parts_v.append(self._pending_values[:, :, ps:pe, :])
        if parts_k:
            return torch.cat(parts_k, dim=-2), torch.cat(parts_v, dim=-2)
        empty = (1, self._n_kv_heads, 0, self._head_dim)
        z = torch.zeros(*empty, dtype=self.dtype, device=self.device)
        return z, z

    def _dequant_packed_slice(self, packed, scales, start: int, end: int) -> torch.Tensor:
        """Dequantize packed int4 for tokens [start:end). Dequantizes the covering
        groups then slices to the exact range."""
        gs = self.group_size
        g_start = start // gs
        g_end = (end + gs - 1) // gs  # ceil
        g_end = min(g_end, packed.shape[0] // gs)
        packed_slice = packed[g_start * gs: g_end * gs]
        scales_slice = scales[g_start:g_end]
        n = end - start
        full = _dequantize_int4(packed_slice, scales_slice, (g_end - g_start) * gs, gs)
        # full is [1, nkv, group_aligned_len, hd]; slice to [start:end]
        off = start - g_start * gs
        return full[:, :, off:off + n, :]


class QuantStarKVCache(Cache):
    """Hybrid KV cache for Qwen3.6's mixed architecture.

    48 linear_attention (DeltaNet) layers use LinearAttentionLayer (bf16 conv/recurrent state,
    fixed-size, no quantization needed).
    16 full_attention layers use Int4AttentionCacheLayer (4-bit append-only KV cache).
    """

    def __init__(self, config, group_size: int = _GROUP_SIZE):
        text_config = config.get_text_config(decoder=True)
        layer_types = getattr(text_config, "layer_types", None)

        layers = []
        for lt in (layer_types or []):
            if lt in ("linear_attention", "conv", "mamba", "moe", "hybrid"):
                layers.append(LinearAttentionLayer(config=None))
            else:
                layers.append(Int4AttentionCacheLayer(group_size=group_size))

        super().__init__(layers=layers)


# ---------------------------------------------------------------------------
# Blockwise online-softmax GQA attention
# ---------------------------------------------------------------------------
#
# During chunked prefill the query is a suffix of the cached keys (offset>0).
# SDPA cannot keep GQA in that case: a causal mask is required for the offset,
# and enable_gqa + any mask falls back to the math kernel (materializes the
# full [B,nq,q,kv] score matrix -> OOM). is_causal=True is wrong for offset.
# FlashAttention-2 has no cp314 wheel and FlexAttention overflows triton
# registers for 24 query heads (non power-of-2). So we reimplement the
# FlashAttention online-softmax in PyTorch: K/V stay at nkv heads (no repeat),
# causality is applied per key-block, peak memory is one [B,nq,q,block] block.

_BLOCK_SIZE = 1024


def blockwise_gqa_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                            scaling: float, block_size: int = _BLOCK_SIZE) -> torch.Tensor:
    """Online-softmax blockwise attention with GQA (no repeat_kv).

    q: [B, nq, q_len, hd], k/v: [B, nkv, kv_len, hd]. Causal-with-offset:
    query i (absolute pos offset+i) attends to keys 0..(offset+i).
    Returns [B, nq, q_len, hd].
    """
    B, nq, q_len, hd = query.shape
    nkv = key.shape[1]
    kv_len = key.shape[2]
    G = nq // nkv
    offset = kv_len - q_len
    device, dtype = query.device, query.dtype

    # running online-softmax state (fp32 for stability)
    m = torch.full((B, nq, q_len, 1), float("-inf"), device=device, dtype=torch.float32)
    l = torch.zeros((B, nq, q_len, 1), device=device, dtype=torch.float32)
    num = torch.zeros((B, nq, q_len, hd), device=device, dtype=torch.float32)

    qg = query.view(B, nkv, G, q_len, hd)  # for GQA batched matmul

    for start in range(0, kv_len, block_size):
        end = min(start + block_size, kv_len)
        kb = key[:, :, start:end, :]
        vb = value[:, :, start:end, :]
        Bl = end - start

        # GQA scores: [B, nkv, G, q_len, Bl] -> [B, nq, q_len, Bl]
        scores = torch.matmul(qg, kb.unsqueeze(2).transpose(-1, -2)) * scaling
        scores = scores.view(B, nq, q_len, Bl).float()

        # causal-with-offset mask
        key_pos = torch.arange(start, end, device=device).view(1, 1, 1, Bl)
        q_pos = torch.arange(q_len, device=device).view(1, 1, q_len, 1)
        causal = (q_pos + offset) >= key_pos
        scores = scores.masked_fill(~causal, float("-inf"))

        mb = scores.amax(dim=-1, keepdim=True)
        m_new = torch.maximum(m, mb)
        p = torch.exp(scores - m_new)

        pg = p.view(B, nkv, G, q_len, Bl)
        block_num = torch.matmul(pg, vb.unsqueeze(2).float()).view(B, nq, q_len, hd)
        block_sum = p.sum(dim=-1, keepdim=True)

        alpha = torch.exp(m - m_new)
        num = num * alpha + block_num
        l = l * alpha + block_sum
        m = m_new

    out = num / l.clamp(min=1e-20)
    return out.to(dtype)


def quantstar_attention_forward(module, query, key, value, attention_mask, dropout=0.0,
                               scaling=None, is_causal=None, **kwargs):
    """Custom attention: blockwise GQA for cached prefill (offset>0), SDPA otherwise.

    The 4D causal mask is never materialized for this implementation (see note
    above), so attention_mask is None and GQA stays active on the SDPA path.
    """
    q_len = query.shape[2]
    kv_len = key.shape[2]
    has_gqa = getattr(module, "num_key_value_groups", 1) > 1
    scale = scaling if scaling is not None else (query.shape[-1] ** -0.5)

    # cached prefill: query is a suffix of the keys -> blockwise to keep GQA
    if has_gqa and q_len > 1 and kv_len > q_len:
        out = blockwise_gqa_attention(query, key, value, scale)
        return out.transpose(1, 2).contiguous(), None

    # decode (q_len==1) or first chunk (kv_len==q_len): normal SDPA, GQA active
    sdpa_kwargs = {"enable_gqa": True} if has_gqa else {}
    if is_causal is None:
        is_causal = getattr(module, "is_causal", True)
    is_causal = q_len > 1 and attention_mask is None and is_causal
    attn_output = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=dropout,
        scale=scaling, is_causal=is_causal, **sdpa_kwargs,
    )
    return attn_output.transpose(1, 2).contiguous(), None


def blockwise_attention_from_cache(query: torch.Tensor, cache_layer, scaling: float,
                                   block_size: int = _BLOCK_SIZE) -> torch.Tensor:
    """Online-softmax blockwise attention that dequantizes K/V one block at a time
    from the int4 cache (avoids materializing the full dequantized KV).

    query: [B, nq, q_len, hd]; cache holds [1, nkv, kv_len, hd] in int4.
    Returns [B, nq, q_len, hd].
    """
    B, nq, q_len, hd = query.shape
    nkv = cache_layer._n_kv_heads
    kv_len = cache_layer.stored_length()
    G = nq // nkv
    offset = kv_len - q_len
    device, dtype = query.device, query.dtype

    m = torch.full((B, nq, q_len, 1), float("-inf"), device=device, dtype=torch.float32)
    l = torch.zeros((B, nq, q_len, 1), device=device, dtype=torch.float32)
    num = torch.zeros((B, nq, q_len, hd), device=device, dtype=torch.float32)

    qg = query.view(B, nkv, G, q_len, hd)

    for start in range(0, kv_len, block_size):
        end = min(start + block_size, kv_len)
        kb, vb = cache_layer.get_kv_block(start, end)  # [1, nkv, Bl, hd] bf16
        Bl = end - start

        scores = torch.matmul(qg, kb.unsqueeze(2).transpose(-1, -2)) * scaling
        scores = scores.view(B, nq, q_len, Bl).float()

        key_pos = torch.arange(start, end, device=device).view(1, 1, 1, Bl)
        q_pos = torch.arange(q_len, device=device).view(1, 1, q_len, 1)
        causal = (q_pos + offset) >= key_pos
        scores = scores.masked_fill(~causal, float("-inf"))

        mb = scores.amax(dim=-1, keepdim=True)
        m_new = torch.maximum(m, mb)
        p = torch.exp(scores - m_new)

        pg = p.view(B, nkv, G, q_len, Bl)
        block_num = torch.matmul(pg, vb.unsqueeze(2).float()).view(B, nq, q_len, hd)
        block_sum = p.sum(dim=-1, keepdim=True)

        alpha = torch.exp(m - m_new)
        num = num * alpha + block_num
        l = l * alpha + block_sum
        m = m_new

    out = num / l.clamp(min=1e-20)
    return out.to(dtype)


def _patch_qwen_attention():
    """Monkey-patch Qwen3_5Attention.forward to dequantize KV block-by-block
    during prefill (eliminates the full-dequant transient). Decode uses SDPA."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M

    if getattr(M.Qwen3_5Attention, "_quantstar_patched", False):
        return

    apply_rotary_pos_emb = M.apply_rotary_pos_emb

    def forward(self, hidden_states, position_embeddings, attention_mask,
                past_key_values=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)
        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        q_len = query_states.shape[2]
        has_gqa = self.num_key_value_groups > 1
        cache_layer = (past_key_values.layers[self.layer_idx]
                       if past_key_values is not None else None)
        # Use blockwise (block-by-block dequant) whenever the int4 cache is
        # initialized: cached prefill (offset>0) AND decode (q_len==1). Both
        # would otherwise materialize the full dequantized KV -> OOM at 256k.
        use_blockwise = (has_gqa and cache_layer is not None
                         and isinstance(cache_layer, Int4AttentionCacheLayer)
                         and cache_layer.is_initialized)

        if use_blockwise:
            cache_layer.append_only(key_states, value_states)
            attn_output = blockwise_attention_from_cache(query_states, cache_layer, self.scaling)
            attn_output = attn_output.transpose(1, 2).contiguous()
        else:
            if past_key_values is not None:
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self.layer_idx
                )
            sdpa_kwargs = {"enable_gqa": True} if has_gqa else {}
            is_causal = q_len > 1 and attention_mask is None and self.is_causal
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states, attn_mask=attention_mask,
                dropout_p=0.0, scale=self.scaling, is_causal=is_causal, **sdpa_kwargs,
            )
            attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    M.Qwen3_5Attention.forward = forward
    M.Qwen3_5Attention._quantstar_patched = True


_patch_qwen_attention()


def _register_quantstar_attention():
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    if "quantstar" not in ALL_ATTENTION_FUNCTIONS:
        ALL_ATTENTION_FUNCTIONS.register("quantstar", quantstar_attention_forward)


_register_quantstar_attention()


# ---------------------------------------------------------------------------
# Memory logging
# ---------------------------------------------------------------------------

def _print_memory_usage(prefix: str = "") -> None:
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    log.info(f"{prefix} GPU memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")


_CacheFactory: Optional[Callable] = None


def _make_cache_factory(model):
    """Return a callable that creates a fresh QuantStarKVCache for each request."""
    global _CacheFactory
    if _CacheFactory is not None:
        return _CacheFactory

    try:
        config = model.config

        def factory():
            return QuantStarKVCache(config=config)

        _CacheFactory = factory
        log.info("QuantStar int4 KV cache factory ready (append-only, no re-quantization)")
        return factory
    except Exception as e:
        log.warning(f"Failed to create int4 KV cache ({e}) — using dynamic cache")
        return None


def load_and_quantize_model(
    model_path: str,
    attn_implementation: str = "sdpa",
    torch_dtype_str: str = "bfloat16",
) -> tuple[torch.nn.Module, object, object, Optional[object]]:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    dtype = getattr(torch, torch_dtype_str) if torch_dtype_str != "auto" else torch.bfloat16

    _print_memory_usage("before model load")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # AutoModelForCausalLM resolves to Qwen3_5ForConditionalGeneration for Qwen3.6,
    # loading the full VL model (language + vision encoder). Load with sdpa then
    # switch to our custom "quantstar" attention.
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="cuda:0",
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )
    model.config._attn_implementation = "quantstar"
    log.info("Loaded with bitsandbytes 4-bit NF4 quantization (attn=quantstar blockwise GQA)")

    _print_memory_usage("after model load")

    gc.collect()
    torch.cuda.empty_cache()
    _print_memory_usage("after gc")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    from transformers.models.qwen3_vl import Qwen3VLProcessor
    processor = Qwen3VLProcessor.from_pretrained(model_path)

    cache_factory = _make_cache_factory(model)

    model.eval()
    return model, tokenizer, processor, cache_factory
