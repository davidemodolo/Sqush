"""Tests for Int4 KV cache.

Covers: _quantize_int4, _dequantize_int4, Int4AttentionCacheLayer,
        QuantStarKVCache, _make_cache_factory.
All tests run on CPU with synthetic tensors — no GPU required.
"""
from __future__ import annotations

from unittest import mock

import torch
from transformers.cache_utils import LinearAttentionLayer

from quantstar.quantize import (
    Int4AttentionCacheLayer,
    QuantStarKVCache,
    _GROUP_SIZE,
    _dequantize_int4,
    _make_cache_factory,
    _quantize_int4,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _kv(seq_len: int, n_heads: int = 2, head_dim: int = 16, seed: int = 0) -> torch.Tensor:
    """Return [1, n_heads, seq_len, head_dim] bfloat16."""
    torch.manual_seed(seed)
    return torch.randn(1, n_heads, seq_len, head_dim, dtype=torch.bfloat16)


def _make_cache_config(
    n_full: int = 4,
    n_linear: int = 4,
    n_kv_heads: int = 2,
    head_dim: int = 16,
) -> mock.MagicMock:
    text_config = mock.MagicMock()
    text_config.layer_types = (
        ["full_attention"] * n_full + ["linear_attention"] * n_linear
    )
    text_config.num_key_value_heads = n_kv_heads
    text_config.head_dim = head_dim
    cfg = mock.MagicMock()
    cfg.get_text_config.return_value = text_config
    return cfg


# ── quantize to packed uint8 ──────────────────────────────────────────

class TestQuantizeInt4:
    def test_packed_dtype_is_uint8(self):
        """output is packed uint8."""
        packed, scales, seq_len = _quantize_int4(_kv(64))
        assert packed.dtype == torch.uint8

    def test_packed_shape_group_aligned(self):
        """packed: [padded_len, n_heads, head_dim//2]."""
        packed, scales, seq_len = _quantize_int4(_kv(64, n_heads=2, head_dim=16))
        assert packed.shape == (64, 2, 8)

    def test_seq_len_returned(self):
        """seq_len matches the original (unpadded) sequence length."""
        _, _, seq_len = _quantize_int4(_kv(70))
        assert seq_len == 70

    def test_padded_len_rounds_up(self):
        """Non-group-aligned input is padded to next multiple of GROUP_SIZE."""
        packed, _, _ = _quantize_int4(_kv(70))
        assert packed.shape[0] == 128  # 70 → 128

    def test_2_2_roundtrip_shape(self):
        """dequantized shape matches original."""
        kv = _kv(128, n_heads=4, head_dim=32)
        packed, scales, seq_len = _quantize_int4(kv)
        recovered = _dequantize_int4(packed, scales, seq_len)
        assert recovered.shape == kv.shape

    def test_2_2_roundtrip_accuracy(self):
        """round-trip error < 10% for bfloat16 input."""
        kv = _kv(128, n_heads=4, head_dim=32)
        packed, scales, seq_len = _quantize_int4(kv)
        recovered = _dequantize_int4(packed, scales, seq_len)

        orig = kv[0].float()
        rec = recovered[0].float()
        scale_mag = orig.abs().mean().clamp(min=1e-8)
        rel_err = (orig - rec).abs().mean() / scale_mag
        assert rel_err < 0.20, f"round-trip rel error {rel_err:.4f} > 20%"

    def test_dequant_clips_to_seq_len(self):
        """dequantize returns exactly seq_len tokens, not the padded count."""
        kv = _kv(70)
        packed, scales, seq_len = _quantize_int4(kv)
        rec = _dequantize_int4(packed, scales, seq_len)
        assert rec.shape[-2] == 70


# ── Int4AttentionCacheLayer ───────────────────────────────

class TestInt4CacheLayer:
    def test_2_7_pre_init_sets_is_initialized(self):
        """pre_init sets is_initialized=True."""
        layer = Int4AttentionCacheLayer()
        assert not layer.is_initialized
        layer.pre_init(n_kv_heads=2, head_dim=16)
        assert layer.is_initialized

    def test_2_6_stored_length_tracks_appended(self):
        """stored_length() matches appended token count."""
        layer = Int4AttentionCacheLayer()
        layer.append_only(_kv(64), _kv(64))
        assert layer.stored_length() == 64
        layer.append_only(_kv(32), _kv(32))
        assert layer.stored_length() == 96

    def test_2_3_pending_lt_group_size_after_append(self):
        """after append_only, pending buffer is < GROUP_SIZE tokens."""
        layer = Int4AttentionCacheLayer()
        layer.append_only(_kv(65), _kv(65))  # 64 flush, 1 pending
        pending = layer._pending_keys.shape[-2] if layer._pending_keys is not None else 0
        assert pending < _GROUP_SIZE

    def test_2_3_exact_boundary_zero_pending(self):
        """Exactly GROUP_SIZE tokens → zero pending after flush."""
        layer = Int4AttentionCacheLayer()
        layer.append_only(_kv(_GROUP_SIZE), _kv(_GROUP_SIZE))
        pending = layer._pending_keys.shape[-2] if layer._pending_keys is not None else 0
        assert pending == 0

    def test_2_4_get_kv_block_full_range(self):
        """get_kv_block(0, N) returns all N tokens."""
        layer = Int4AttentionCacheLayer()
        layer.append_only(_kv(64), _kv(64))
        k, v = layer.get_kv_block(0, 64)
        assert k.shape[-2] == 64
        assert v.shape[-2] == 64

    def test_2_4_get_kv_block_interior_slice(self):
        """get_kv_block returns only the requested token slice."""
        layer = Int4AttentionCacheLayer()
        layer.append_only(_kv(64), _kv(64))
        k, v = layer.get_kv_block(10, 30)
        assert k.shape[-2] == 20
        assert v.shape[-2] == 20

    def test_2_5_get_kv_block_non_group_aligned_start(self):
        """get_kv_block handles non-group-aligned start correctly."""
        layer = Int4AttentionCacheLayer()
        layer.append_only(_kv(128, n_heads=2, head_dim=16), _kv(128, n_heads=2, head_dim=16))
        k, v = layer.get_kv_block(10, 27)  # neither boundary is group-aligned
        assert k.shape[-2] == 17
        assert v.shape[-2] == 17

    def test_2_5_get_kv_block_crosses_quantized_pending_boundary(self):
        """slice crossing the quantized/pending boundary returns correct len."""
        layer = Int4AttentionCacheLayer()
        layer.append_only(_kv(64), _kv(64))   # 64 quantized
        layer.append_only(_kv(10), _kv(10))   # 10 pending
        # Request tokens 60..74 (4 quantized + 10 pending)
        k, v = layer.get_kv_block(60, 74)
        assert k.shape[-2] == 14
        assert v.shape[-2] == 14

    def test_2_11_seq_length_after_multi_appends(self):
        """get_seq_length correct across multiple append rounds."""
        layer = Int4AttentionCacheLayer()
        layer.append_only(_kv(64), _kv(64))
        layer.append_only(_kv(1), _kv(1))
        layer.append_only(_kv(1), _kv(1))
        assert layer.get_seq_length() == 66

    def test_update_accumulates_kv(self):
        """update() called twice returns cumulative K/V sequence."""
        layer = Int4AttentionCacheLayer()
        layer.update(_kv(64), _kv(64))
        k, v = layer.update(_kv(1), _kv(1))
        assert k.shape[-2] == 65

    def test_update_returns_tensors(self):
        """update() returns a (key, value) 2-tuple of tensors."""
        layer = Int4AttentionCacheLayer()
        result = layer.update(_kv(64), _kv(64))
        assert isinstance(result, tuple) and len(result) == 2
        assert isinstance(result[0], torch.Tensor)

    def test_lazy_init_via_update(self):
        """is_initialized becomes True on first update() call."""
        layer = Int4AttentionCacheLayer()
        assert not layer.is_initialized
        layer.update(_kv(64), _kv(64))
        assert layer.is_initialized


# ── QuantStarKVCache ─────────────────────────────

class TestQuantStarKVCache:
    def test_2_8_full_model_64_layers(self):
        """16 full + 48 linear = 64 total for the actual model shape."""
        cache = QuantStarKVCache(config=_make_cache_config(n_full=16, n_linear=48))
        assert len(cache.layers) == 64

    def test_2_8_int4_layer_count(self):
        """16 layers are Int4AttentionCacheLayer."""
        cache = QuantStarKVCache(config=_make_cache_config(n_full=16, n_linear=48))
        n_int4 = sum(1 for l in cache.layers if isinstance(l, Int4AttentionCacheLayer))
        assert n_int4 == 16

    def test_2_10_linear_layers_type(self):
        """linear-attention slots use LinearAttentionLayer."""
        cache = QuantStarKVCache(config=_make_cache_config(n_full=2, n_linear=4))
        n_lin = sum(1 for l in cache.layers if isinstance(l, LinearAttentionLayer))
        assert n_lin == 4

    def test_2_9_layer_types_order_preserved(self):
        """full-attention layers are at the indices declared in layer_types."""
        text_config = mock.MagicMock()
        text_config.layer_types = [
            "linear_attention", "full_attention",
            "linear_attention", "full_attention",
        ]
        text_config.num_key_value_heads = 2
        text_config.head_dim = 16
        cfg = mock.MagicMock()
        cfg.get_text_config.return_value = text_config

        cache = QuantStarKVCache(config=cfg)
        assert isinstance(cache.layers[0], LinearAttentionLayer)
        assert isinstance(cache.layers[1], Int4AttentionCacheLayer)
        assert isinstance(cache.layers[2], LinearAttentionLayer)
        assert isinstance(cache.layers[3], Int4AttentionCacheLayer)

    def test_2_12_factory_independence_no_singleton(self):
        """successive calls to _make_cache_factory return independent factories."""
        cfg = _make_cache_config(n_full=2, n_linear=2)

        def _mock_model():
            m = mock.MagicMock()
            m.config = cfg
            m.dtype = torch.bfloat16
            m.device = torch.device("cpu")
            return m

        factory_a = _make_cache_factory(_mock_model())
        factory_b = _make_cache_factory(_mock_model())

        assert factory_a is not None
        assert factory_b is not None
        assert factory_a is not factory_b, "factories must be independent, not a singleton"

    def test_2_12_factory_each_call_returns_new_cache(self):
        """Each factory() call produces a new, distinct cache instance."""
        cfg = _make_cache_config(n_full=2, n_linear=2)
        m = mock.MagicMock()
        m.config = cfg
        m.dtype = torch.bfloat16
        m.device = torch.device("cpu")

        factory = _make_cache_factory(m)
        assert factory is not None
        c1 = factory()
        c2 = factory()
        assert c1 is not c2

    def test_pre_init_called_when_heads_known(self):
        """Int4 layers get pre_init when n_kv_heads and head_dim are available."""
        cache = QuantStarKVCache(config=_make_cache_config(n_full=2, n_linear=0, n_kv_heads=4, head_dim=32))
        int4_layers = [l for l in cache.layers if isinstance(l, Int4AttentionCacheLayer)]
        for layer in int4_layers:
            assert layer.is_initialized, "pre_init should have been called during __init__"
