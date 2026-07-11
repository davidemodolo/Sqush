from __future__ import annotations

import gc
import logging
import os
import sys
from typing import Optional

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

    if getattr(Autotuner, "_sqush_nargs_fixed", False):
        return

    _original_bench = Autotuner._bench

    def _safe_bench(self, *args, config, **meta):
        if self.nargs is None:
            self.nargs = {}
        return _original_bench(self, *args, config=config, **meta)

    Autotuner._bench = _safe_bench
    Autotuner._sqush_nargs_fixed = True


_patch_triton_autotuner()

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # avoids OOM from allocator fragmentation with large cache blocks

log = logging.getLogger(__name__)


def _ensure_cuda_libs():
    py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    cu13_paths = [
        os.path.expanduser(f"~/.local/lib/{py_ver}/site-packages/nvidia/cu13/lib"),
        os.path.join(sys.prefix, "lib", py_ver, "site-packages", "nvidia", "cu13", "lib"),
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

    def pre_init(self, n_kv_heads: int, head_dim: int,
                 dtype: torch.dtype = torch.bfloat16,
                 device: torch.device = None):
        self._n_kv_heads = n_kv_heads
        self._head_dim = head_dim
        self.dtype = dtype
        self.device = device or torch.device("cuda:0")
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


class SqushKVCache(Cache):
    """Hybrid KV cache for Qwen3.6's mixed architecture.

    48 linear_attention (DeltaNet) layers use LinearAttentionLayer (bf16 conv/recurrent state,
    fixed-size, no quantization needed).
    16 full_attention layers use Int4AttentionCacheLayer (4-bit append-only KV cache).
    """

    def __init__(self, config, group_size: int = _GROUP_SIZE,
                 dtype: torch.dtype = torch.bfloat16,
                 device: torch.device = None):
        text_config = config.get_text_config(decoder=True)
        layer_types = getattr(text_config, "layer_types", None)
        n_kv_heads = getattr(text_config, "num_key_value_heads", None)
        head_dim = getattr(text_config, "head_dim", None)

        layers = []
        for lt in (layer_types or []):
            if lt in ("linear_attention", "conv", "mamba", "moe", "hybrid"):
                layers.append(LinearAttentionLayer(config=None))
            else:
                layer = Int4AttentionCacheLayer(group_size=group_size)
                if n_kv_heads is not None and head_dim is not None:
                    layer.pre_init(n_kv_heads, head_dim, dtype=dtype, device=device)
                layers.append(layer)

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


def sqush_attention_forward(module, query, key, value, attention_mask, dropout=0.0,
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

    if getattr(M.Qwen3_5Attention, "_sqush_patched", False):
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
    M.Qwen3_5Attention._sqush_patched = True


_patch_qwen_attention()


def _register_sqush_attention():
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    if "sqush" not in ALL_ATTENTION_FUNCTIONS:
        ALL_ATTENTION_FUNCTIONS.register("sqush", sqush_attention_forward)


_register_sqush_attention()


# ---------------------------------------------------------------------------
# Memory logging
# ---------------------------------------------------------------------------

def _print_memory_usage(prefix: str = "") -> None:
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    log.info(f"{prefix} GPU memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")


def _make_cache_factory(model):
    try:
        config = model.config
        _dtype = model.dtype if hasattr(model, "dtype") else torch.bfloat16
        _device = model.device if hasattr(model, "device") else torch.device("cuda:0")

        def factory():
            return SqushKVCache(config=config, dtype=_dtype, device=_device)

        log.info("Sqush int4 KV cache factory ready (append-only, no re-quantization)")
        return factory
    except Exception as e:
        log.warning(f"Failed to create int4 KV cache ({e}) — using dynamic cache")
        return None


def _model_is_pre_quantized(model_path: str) -> bool:
    import json
    import os
    config_path = os.path.join(model_path, "config.json")
    if os.path.isfile(config_path):
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            qc = cfg.get("quantization_config", {})
            return qc.get("quant_method") == "bitsandbytes"
        except Exception:
            pass
    return False


# The Qwen3.5-9B checkpoint's chat template unconditionally strips prior-turn
# <think> blocks (the Qwen3.6-27B template gained a `preserve_thinking` escape
# hatch). Stripping breaks session KV reuse: the re-rendered prompt no longer
# matches the tokens the cache was built from, so every follow-up turn misses.
# Patch the template in memory to honor preserve_thinking, matching the 27B.
_TEMPLATE_STRIP_THINKING = "{%- if loop.index0 > ns.last_query_index %}"
_TEMPLATE_PRESERVE_THINKING = (
    "{%- if (preserve_thinking is defined and preserve_thinking is true) "
    "or (loop.index0 > ns.last_query_index) %}"
)


def _patch_chat_template_preserve_thinking(tokenizer) -> None:
    tpl = getattr(tokenizer, "chat_template", None)
    if isinstance(tpl, str) and "preserve_thinking" not in tpl and _TEMPLATE_STRIP_THINKING in tpl:
        tokenizer.chat_template = tpl.replace(_TEMPLATE_STRIP_THINKING, _TEMPLATE_PRESERVE_THINKING)
        log.info("Patched chat template to honor preserve_thinking (required for session KV reuse)")


def _quantize_lm_head(model: torch.nn.Module) -> None:
    """Quantize lm_head to NF4 after loading.

    lm_head is a 248320×4096 bfloat16 tensor (2.03 GB); NF4 (~0.57 GB) saves ~1.45 GB
    on the 8 GB tier. It MUST stay in llm_int8_skip_modules: on a pre-quantized
    checkpoint, from_pretrained expects any Linear4bit module's weights to already be
    packed 4-bit + quant_state in the shard — removing lm_head from the skip list
    makes it load the raw bfloat16 weight into a Linear4bit with no quant_state and
    bitsandbytes asserts on the first forward. So we load it as a plain bf16
    nn.Linear and quantize it here, the same post-load approach as
    _quantize_visual_encoder.
    """
    try:
        import bitsandbytes as bnb
    except ImportError:
        log.warning("bitsandbytes not available — skipping lm_head quantization")
        return
    if not torch.cuda.is_available():
        log.warning("CUDA not available — skipping lm_head quantization (NF4 requires CUDA)")
        return

    head = getattr(model, "lm_head", None)
    if not isinstance(head, torch.nn.Linear) or isinstance(head, bnb.nn.Linear4bit):
        return

    target_device = head.weight.device
    out_f, in_f = head.weight.shape
    w_cpu = head.weight.data.cpu().to(torch.float16)
    b_cpu = head.bias.data.cpu() if head.bias is not None else None
    # Free the 2 GB bfloat16 GPU tensor before quantizing so the transient stays low.
    head.weight.data = torch.empty(0, device="cpu")
    if head.bias is not None:
        head.bias.data = torch.empty(0, device="cpu")
    gc.collect()
    torch.cuda.empty_cache()

    new = bnb.nn.Linear4bit(
        in_f, out_f,
        bias=b_cpu is not None,
        compute_dtype=torch.bfloat16,
        compress_statistics=True,
        quant_type="nf4",
    )
    new.weight = bnb.nn.Params4bit(
        w_cpu, requires_grad=False, quant_type="nf4", compress_statistics=True,
    )
    if b_cpu is not None:
        new.bias = torch.nn.Parameter(b_cpu.to(target_device), requires_grad=False)
    new = new.to(target_device if target_device.type == "cuda" else "cuda:0")  # .to(cuda) triggers NF4 quantization

    model.lm_head = new
    del w_cpu, b_cpu
    gc.collect()
    torch.cuda.empty_cache()
    log.info("Quantized lm_head to NF4 (%d × %d, ~2.0 GB bf16 → ~0.57 GB)", out_f, in_f)


def _patch_quant_config_for_visual_encoder(model: torch.nn.Module) -> None:
    """Remove visual encoder modules from bitsandbytes skip list.

    When saving a cooked model, this ensures the visual encoder's Linear4bit
    layers are included in the quantization_config so from_pretrained reloads
    them with the correct layer type (not nn.Linear).
    """
    qc = getattr(getattr(model, "config", None), "quantization_config", None)
    if qc is None:
        return
    skip = getattr(qc, "llm_int8_skip_modules", None)
    if not skip:
        return
    visual_prefixes = ("visual", "vision_model", "vision_tower", "img_processor")
    updated = [m for m in skip if not any(m.startswith(p) for p in visual_prefixes)]
    if len(updated) != len(skip):
        log.info(
            "Patched quantization_config: removed %d visual encoder entries from skip_modules",
            len(skip) - len(updated),
        )
        qc.llm_int8_skip_modules = updated


def load_and_quantize_model(
    model_path: str,
    attn_implementation: str = "sdpa",
    torch_dtype_str: str = "bfloat16",
    quantize_embeddings: bool = False,
    quantize_vision_encoder: bool = False,
    device_map: "str | dict" = "cuda:0",
) -> tuple[torch.nn.Module, object, object, Optional[object]]:
    from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

    dtype = getattr(torch, torch_dtype_str) if torch_dtype_str != "auto" else torch.bfloat16

    _print_memory_usage("before model load")

    already_quantized = _model_is_pre_quantized(model_path)

    if already_quantized:
        from transformers import AutoConfig
        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        pre_baked = getattr(model_config, "qs_pre_baked_embeddings", False)
        if pre_baked:
            # embed_tokens.weight was removed from the cooked safetensors during bake
            # and its 4-bit form saved to quantized_embeddings.safetensors. Route
            # embed_tokens to CPU so from_pretrained never allocates the 1.93 GB
            # bfloat16 tensor on GPU — this prevents CUDA allocator fragmentation that
            # would otherwise lock 2 GB in reserve permanently. The module is replaced
            # with QuantizedEmbedding on GPU immediately after loading.
            device_map = {"model.language_model.embed_tokens": "cpu", "": "cuda:0"}

        if isinstance(device_map, dict):
            qc = getattr(model_config, "quantization_config", None)
            if isinstance(qc, dict):
                qc["llm_int8_enable_fp32_cpu_offload"] = True
            elif qc is not None:
                qc.llm_int8_enable_fp32_cpu_offload = True
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            config=model_config,
            device_map=device_map,
            attn_implementation=attn_implementation,
            trust_remote_code=True,
            ignore_mismatched_sizes=pre_baked,
        )
        log.info("Loaded pre-quantized bitsandbytes model from disk")
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map=device_map,
            attn_implementation=attn_implementation,
            trust_remote_code=True,
        )
        log.info("Loaded with bitsandbytes 4-bit NF4 quantization (attn=sqush blockwise GQA)")

    model.config._attn_implementation = "sqush"

    _print_memory_usage("after model load")

    gc.collect()
    torch.cuda.empty_cache()
    _print_memory_usage("after gc")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _patch_chat_template_preserve_thinking(tokenizer)

    from transformers.models.qwen3_vl import Qwen3VLProcessor
    processor = Qwen3VLProcessor.from_pretrained(model_path)

    # Quantize the VL visual encoder and embeddings to 4-bit.
    # bitsandbytes leaves these in bfloat16 (~3.4 GB for visual encoder, ~1 GB for embeddings).
    if quantize_vision_encoder:
        _quantize_visual_encoder(model)
        gc.collect()
        torch.cuda.empty_cache()
        _print_memory_usage("after visual encoder quantization")
    if quantize_embeddings and not (already_quantized and getattr(model.config, "qs_pre_baked_embeddings", False)):
        _quantize_embeddings(model)
        gc.collect()
        torch.cuda.empty_cache()
        _print_memory_usage("after embedding quantization")
    if already_quantized and getattr(model.config, "qs_pre_baked_embeddings", False):
        _load_pre_baked_embeddings(model, model_path)
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        _print_memory_usage("after pre-baked embedding load")
        _quantize_lm_head(model)
        _print_memory_usage("after lm_head quantization")

    cache_factory = _make_cache_factory(model)

    model.eval()
    return model, tokenizer, processor, cache_factory


def bake_nf4_checkpoint(
    raw_path: str,
    cooked_path: str,
    torch_dtype_str: str = "bfloat16",
    attn_implementation: str = "sdpa",
) -> None:
    """One-time NF4 quantization of a full-precision checkpoint, saved to disk.

    Loads the model with bitsandbytes 4-bit (needs a GPU that fits the 4-bit
    model — the same one that serves it) and serializes the packed NF4 weights
    plus quantization_config via save_pretrained, so subsequent loads take the
    fast pre-quantized path instead of re-quantizing ~54 GB of bf16 every launch.

    Unlike the 8 GB tier's side-car bake, this genuinely quantizes the LM weights
    (bnb requires CUDA for that), so it cannot run on CPU. Serialization requires
    a pure-GPU device_map (no CPU offload), which the 24 GB tier provides.
    """
    import gc as _gc

    from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
    from transformers.models.qwen3_vl import Qwen3VLProcessor

    dtype = getattr(torch, torch_dtype_str) if torch_dtype_str != "auto" else torch.bfloat16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    log.info("Bake: loading %s with bitsandbytes NF4 (one-time) …", raw_path)
    model = AutoModelForImageTextToText.from_pretrained(
        raw_path,
        quantization_config=bnb_config,
        device_map="cuda:0",
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )
    _print_memory_usage("bake: after 4-bit load")

    log.info("Bake: saving NF4 checkpoint → %s", cooked_path)
    model.save_pretrained(cooked_path, safe_serialization=True)
    AutoTokenizer.from_pretrained(raw_path, trust_remote_code=True).save_pretrained(cooked_path)
    Qwen3VLProcessor.from_pretrained(raw_path).save_pretrained(cooked_path)

    del model
    _gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("Bake: NF4 checkpoint written.")


# ---------------------------------------------------------------------------
# Embedding quantization — saves ~1.5 GB on the 248k×4096 vocab table
# ---------------------------------------------------------------------------

_EMBED_GROUP_SIZE = 128


class QuantizedEmbedding(torch.nn.Module):
    """4-bit quantized embedding with per-group asymmetric quantization.

    Only dequantizes the rows accessed by the input indices, not the full table.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, qweight: torch.Tensor,
                 scales: torch.Tensor, zero_points: torch.Tensor):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.register_buffer("_qw", qweight)
        self.register_buffer("_sc", scales)
        self.register_buffer("_zp", zero_points)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        n = indices.numel()
        if n == 0:
            return torch.empty(0, self.embedding_dim, device=indices.device, dtype=torch.bfloat16)

        num_groups = (self.embedding_dim + _EMBED_GROUP_SIZE - 1) // _EMBED_GROUP_SIZE
        padded = num_groups * _EMBED_GROUP_SIZE

        flat = indices.flatten()
        device = indices.device

        qw = self._qw[flat]  # [n, num_groups, 16]
        sc = self._sc[flat]  # [n, num_groups]
        zp = self._zp[flat]  # [n, num_groups]
        shift = torch.arange(8, device=device, dtype=torch.int32)
        # qw: [n, num_groups, 16] → unpack → [n, num_groups, 128]
        vals = (qw.unsqueeze(-1) >> (shift * 4)) & 0xF  # [n, num_groups, 16, 8]
        vals = vals.reshape(n, num_groups, _EMBED_GROUP_SIZE)

        w = ((vals.to(torch.bfloat16) - zp.unsqueeze(-1).to(torch.bfloat16))
             * sc.unsqueeze(-1).to(torch.bfloat16))
        # Flatten the groups back to the padded row, then drop the padding — the
        # padding lives at the end of the flattened row, not along the group axis,
        # so slicing the group axis (size 128) against embedding_dim was a no-op.
        w = w.reshape(n, padded)[:, : self.embedding_dim]
        return w.reshape(*indices.shape, self.embedding_dim)


def _quantize_visual_encoder(model: torch.nn.Module) -> None:
    """Quantize all remaining bfloat16 Linear layers (the VL visual encoder) to 4-bit NF4.

    In a pre-quantized bitsandbytes model all LM Linear layers are already
    bnb.nn.Linear4bit. Any remaining nn.Linear layers are the visual encoder
    that bitsandbytes skipped. Walking the full module tree avoids depending
    on a specific attribute path (e.g. model.model.visual varies by arch).

    If the model was loaded with accelerate CPU-offload (device_map with cpu entries),
    weights appear as meta tensors. We materialise them via remove_hook_from_module
    before quantising, then quantise on GPU and store NF4 back on CPU for saving.
    """
    try:
        import bitsandbytes as bnb
    except ImportError:
        log.warning("bitsandbytes not available — skipping visual encoder quantization")
        return

    # Materialise any accelerate CPU-offloaded meta tensors so we can read their data.
    try:
        from accelerate.hooks import remove_hook_from_module
        remove_hook_from_module(model, recurse=True)
    except Exception:
        pass

    replacements: list[tuple[torch.nn.Module, str, torch.nn.Module]] = []
    for module in model.modules():
        for name, child in module.named_children():
            if isinstance(child, torch.nn.Linear) and not isinstance(child, bnb.nn.Linear4bit):
                replacements.append((module, name, child))

    if not replacements:
        log.info("Visual encoder: all Linear layers already quantized")
        return

    n_layers = len(replacements)
    freed_bytes = 0
    for parent, name, child in replacements:
        target_device = child.weight.device
        freed_bytes += child.weight.nelement() * child.weight.element_size()

        w_cpu = child.weight.data.cpu().to(torch.float16)
        b_cpu = child.bias.data.cpu() if child.bias is not None else None
        child.weight.data = torch.empty(0, device="cpu")
        if child.bias is not None:
            child.bias.data = torch.empty(0, device="cpu")
        del child
        gc.collect()
        torch.cuda.empty_cache()

        new = bnb.nn.Linear4bit(
            w_cpu.shape[1],
            w_cpu.shape[0],
            bias=b_cpu is not None,
            compute_dtype=torch.bfloat16,
            compress_statistics=True,
            quant_type="nf4",
        )
        new.weight = bnb.nn.Params4bit(
            w_cpu,
            requires_grad=False,
            quant_type="nf4",
            compress_statistics=True,
        )
        if b_cpu is not None:
            new.bias = torch.nn.Parameter(b_cpu.to(target_device), requires_grad=False)

        # NF4 quantisation requires CUDA. If the target is CPU (bake with offloaded
        # visual encoder), quantise on GPU then move the packed NF4 back to CPU.
        if target_device.type == "cpu" and torch.cuda.is_available():
            new = new.to("cuda")  # Params4bit.to("cuda") triggers NF4 quantisation
            new = new.to("cpu")   # moves packed int4 to CPU; stays quantised
        else:
            new = new.to(target_device)

        setattr(parent, name, new)
        del w_cpu, b_cpu

    del replacements
    gc.collect()
    torch.cuda.empty_cache()
    log.info(
        "Visual encoder: quantized %d Linear → NF4 (freed ~%.1f GB bfloat16, added ~%.1f GB int4)",
        n_layers,
        freed_bytes / 1e9,
        freed_bytes / 1e9 / 4,
    )


def _load_pre_baked_embeddings(model: torch.nn.Module, model_path: str) -> None:
    """Load pre-quantized embedding from quantized_embeddings.safetensors and replace embed_tokens."""
    import os as _os
    from safetensors.torch import load_file as _sf_load

    sidecar = _os.path.join(model_path, "quantized_embeddings.safetensors")
    if not _os.path.exists(sidecar):
        log.warning("qs_pre_baked_embeddings is set but %s not found — skipping", sidecar)
        return

    eq = _sf_load(sidecar, device="cpu")
    vocab   = int(eq["_vocab"].item())
    hidden  = int(eq["_hidden"].item())

    # Find the embed_tokens module by name — the visual encoder also has nn.Embedding
    # (pos_embed) that appears earlier in named_modules(), so we must match by name.
    target_name: str | None = None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Embedding) and name.endswith("embed_tokens"):
            target_name = name
            break

    if target_name is None:
        log.warning("No nn.Embedding found in model — cannot install pre-baked embedding")
        return

    # Resolve the parent module and attribute name.
    parts = target_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    attr = parts[-1]

    new_emb = QuantizedEmbedding(
        vocab, hidden,
        eq["_qw"].to("cuda:0"),
        eq["_sc"].to("cuda:0"),
        eq["_zp"].to("cuda:0"),
    )
    setattr(parent, attr, new_emb)
    del eq
    log.info(
        "Installed pre-baked QuantizedEmbedding (%d × %d) at %s",
        vocab, hidden, target_name,
    )


def _quantize_embeddings(model: torch.nn.Module) -> None:
    """Replace all nn.Embedding layers with QuantizedEmbedding."""
    replacements = []

    def _walk(module, prefix):
        for name, child in list(module.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, torch.nn.Embedding):
                replacements.append((module, name, child))
            else:
                _walk(child, path)

    _walk(model, "")

    for parent, name, child in replacements:
        target_device = child.weight.device

        # Move to CPU first, then convert dtype — doing it the other way
        # (.to(float32).cpu()) would create a ~4 GB float32 copy on GPU before
        # the move, which spikes VRAM to ~10 GB on top of the 5.94 GB model.
        w = child.weight.data.cpu().to(torch.float32)
        child.weight.data = torch.empty(0, device='cpu')
        gc.collect()
        torch.cuda.empty_cache()

        out_f, in_f = w.shape
        num_groups = (in_f + _EMBED_GROUP_SIZE - 1) // _EMBED_GROUP_SIZE
        pad = num_groups * _EMBED_GROUP_SIZE - in_f
        if pad:
            w = torch.nn.functional.pad(w, (0, pad), value=0)

        w_f = w.reshape(out_f, num_groups, _EMBED_GROUP_SIZE)
        w_min = w_f.amin(dim=-1)
        w_max = w_f.amax(dim=-1)
        scale = (w_max - w_min).clamp(min=1e-9) / 15.0
        zp = (-w_min / scale).round().clamp(0, 15).to(torch.int32)
        q = ((w_f / scale.unsqueeze(-1)).round() + zp.unsqueeze(-1)).clamp(0, 15).to(torch.int32)

        gs = _EMBED_GROUP_SIZE
        q = torch.nn.functional.pad(q, (0, (8 - gs % 8) % 8))
        q = q.reshape(out_f, num_groups, -1, 8)
        packed = torch.zeros(out_f, num_groups, q.shape[2], dtype=torch.int32)
        for i in range(8):
            packed |= (q[..., i] & 0xF) << (i * 4)

        new = QuantizedEmbedding(
            out_f, in_f,
            packed.to(target_device),
            scale.to(torch.bfloat16).to(target_device),
            zp.to(target_device),
        )
        setattr(parent, name, new)
        del w, w_f, scale, zp, q, packed, child
    del replacements
    gc.collect()
    torch.cuda.empty_cache()
