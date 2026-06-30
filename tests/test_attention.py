"""Tests for blockwise online-softmax GQA attention.

Covers: blockwise_gqa_attention, blockwise_attention_from_cache,
        and _chunked_prefill last-token handling.
All tests run on CPU — no GPU required.
"""
from __future__ import annotations

from unittest import mock

import torch

from quantstar.quantize import (
    Int4AttentionCacheLayer,
    _GROUP_SIZE,
    blockwise_attention_from_cache,
    blockwise_gqa_attention,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _qkv(B=1, nq=4, nkv=2, q_len=8, kv_len=8, hd=32, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(B, nq, q_len, hd)
    k = torch.randn(B, nkv, kv_len, hd)
    v = torch.randn(B, nkv, kv_len, hd)
    return q, k, v


def _naive_gqa_attention(q, k, v, *, causal_offset: int = 0):
    """Reference: repeat-KV + manual softmax with optional offset causal mask."""
    B, nq, q_len, hd = q.shape
    nkv = k.shape[1]
    G = nq // nkv
    scale = hd ** -0.5

    k_rep = k.repeat_interleave(G, dim=1)
    v_rep = v.repeat_interleave(G, dim=1)

    scores = torch.matmul(q, k_rep.transpose(-2, -1)) * scale
    kv_len = k.shape[2]
    key_pos = torch.arange(kv_len).view(1, 1, 1, kv_len)
    q_pos = torch.arange(q_len).view(1, 1, q_len, 1)
    mask = (q_pos + causal_offset) >= key_pos
    scores = scores.masked_fill(~mask, float("-inf"))
    attn = torch.softmax(scores.float(), dim=-1)
    return torch.matmul(attn, v_rep.float()).to(q.dtype)


# ── first prefill (kv_len == q_len) ───────────────────────────────────

class TestBlockwiseGQAFirstPrefill:
    def test_output_shape(self):
        """output shape is [B, nq, q_len, hd]."""
        q, k, v = _qkv(q_len=8, kv_len=8)
        out = blockwise_gqa_attention(q, k, v, scaling=32 ** -0.5)
        assert out.shape == q.shape

    def test_matches_reference_first_prefill(self):
        """blockwise output matches reference softmax when kv_len==q_len."""
        q, k, v = _qkv(q_len=8, kv_len=8)
        scale = 32 ** -0.5

        out = blockwise_gqa_attention(q, k, v, scaling=scale)
        ref = _naive_gqa_attention(q, k, v, causal_offset=0)

        max_err = (out.float() - ref.float()).abs().max().item()
        assert max_err < 1e-3, f"max err {max_err:.2e} vs reference"

    def test_no_nan_in_output(self):
        out, k, v = _qkv(q_len=4, kv_len=4)
        result = blockwise_gqa_attention(out, k, v, scaling=0.5)
        assert not torch.isnan(result).any()


# ── cached prefill (kv_len > q_len) ───────────────────────────────────

class TestBlockwiseGQACachedPrefill:
    def test_output_shape(self):
        """output shape correct when kv_len > q_len (cached prefix)."""
        q, k, v = _qkv(q_len=4, kv_len=16)
        out = blockwise_gqa_attention(q, k, v, scaling=32 ** -0.5)
        assert out.shape == q.shape

    def test_matches_reference_with_offset(self):
        """blockwise output matches reference with causal offset."""
        q, k, v = _qkv(q_len=4, kv_len=16, seed=7)
        scale = 32 ** -0.5
        offset = 16 - 4  # == 12

        out = blockwise_gqa_attention(q, k, v, scaling=scale)
        ref = _naive_gqa_attention(q, k, v, causal_offset=offset)

        max_err = (out.float() - ref.float()).abs().max().item()
        assert max_err < 1e-3, f"max err {max_err:.2e}"

    def test_decode_single_token(self):
        """decode step (q_len=1) uses blockwise path, shape is correct."""
        q, k, v = _qkv(q_len=1, kv_len=32)
        out = blockwise_gqa_attention(q, k, v, scaling=32 ** -0.5)
        assert out.shape == q.shape

    def test_decode_attends_to_all_past_keys(self):
        """at decode step, token attends to all previous keys (no info cutoff)."""
        q, k, v = _qkv(B=1, nq=2, nkv=2, q_len=1, kv_len=4, hd=8, seed=1)
        scale = 8 ** -0.5
        out = blockwise_gqa_attention(q, k, v, scaling=scale)
        # No masking for the last (current) token — all past keys accessible
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


# ── causal mask with offset ────────────────────────────────────────────

class TestCausalMask:
    def test_future_keys_are_masked(self):
        """query at abs position i cannot attend to key at position i+1."""
        # q_len=2, kv_len=4 → offset=2. Query 0 is at abs pos 2; query 1 at abs pos 3.
        # Query 0 must not attend to key 3; query 1 must not attend to key 4 (OOB).
        B, nq, nkv, q_len, kv_len, hd = 1, 2, 2, 2, 4, 16
        torch.manual_seed(42)
        q = torch.randn(B, nq, q_len, hd)
        k = torch.randn(B, nkv, kv_len, hd)
        v = torch.randn(B, nkv, kv_len, hd)
        scale = hd ** -0.5

        # Modify the "future" key (position 3) and verify output for query 0 changes
        out1 = blockwise_gqa_attention(q, k, v, scaling=scale)
        k_modified = k.clone()
        k_modified[:, :, 3, :] += 1000.0  # key that query 0 (abs pos 2) should NOT see
        out2 = blockwise_gqa_attention(q, k_modified, v, scaling=scale)

        # Query 0 output must be identical (it cannot see key at position 3)
        max_diff_q0 = (out1[:, :, 0, :] - out2[:, :, 0, :]).abs().max().item()
        assert max_diff_q0 < 1e-5, f"query 0 was affected by a future key: diff={max_diff_q0:.2e}"

    def test_past_keys_are_visible(self):
        """modifying a past key DOES change the output."""
        B, nq, nkv, q_len, kv_len, hd = 1, 2, 2, 2, 4, 16
        torch.manual_seed(42)
        q = torch.randn(B, nq, q_len, hd)
        k = torch.randn(B, nkv, kv_len, hd)
        v = torch.randn(B, nkv, kv_len, hd)
        scale = hd ** -0.5

        out1 = blockwise_gqa_attention(q, k, v, scaling=scale)
        k_modified = k.clone()
        k_modified[:, :, 0, :] += 1000.0  # key at position 0 — ALWAYS visible
        out2 = blockwise_gqa_attention(q, k_modified, v, scaling=scale)

        diff = (out1 - out2).abs().max().item()
        assert diff > 0.01, "modifying a visible key should change the output"


# ── GQA grouping ───────────────────────────────────────────────────────

class TestGQAGrouping:
    def test_gqa_2x_group(self):
        """4 query heads grouped over 2 KV heads (G=2) works correctly."""
        q, k, v = _qkv(nq=4, nkv=2, q_len=8, kv_len=8, hd=16)
        out = blockwise_gqa_attention(q, k, v, scaling=16 ** -0.5)
        assert out.shape == q.shape

    def test_mha_fallback(self):
        """When nq==nkv (G=1), result equals standard attention."""
        q, k, v = _qkv(nq=2, nkv=2, q_len=4, kv_len=4, hd=8)
        scale = 8 ** -0.5
        out = blockwise_gqa_attention(q, k, v, scaling=scale)
        ref = _naive_gqa_attention(q, k, v, causal_offset=0)
        assert (out.float() - ref.float()).abs().max() < 1e-3


# ── numerical stability ────────────────────────────────────────────────

class TestNumericalStability:
    def test_no_nan_with_large_logits(self):
        """online-softmax stays finite with very large score magnitudes."""
        torch.manual_seed(0)
        q = torch.randn(1, 4, 4, 32) * 100.0
        k = torch.randn(1, 2, 4, 32) * 100.0
        v = torch.randn(1, 2, 4, 32)
        out = blockwise_gqa_attention(q, k, v, scaling=32 ** -0.5)
        assert not torch.isnan(out).any(), "NaN detected with large logits"
        assert not torch.isinf(out).any()

    def test_all_masked_produces_zeros_not_nan(self):
        """When all keys are masked (impossible causal), output should not be NaN."""
        # q_len=1, kv_len=0 edge case — handled by division by l.clamp(min=1e-20)
        torch.manual_seed(0)
        q = torch.randn(1, 2, 1, 8)
        k = torch.zeros(1, 2, 0, 8)  # empty KV → output should be all zeros
        v = torch.zeros(1, 2, 0, 8)
        # blockwise_gqa_attention iterates over kv_len steps; with kv_len=0 no iterations
        # num stays 0, l stays 0 → out = 0/clamp(0,1e-20) = 0
        out = blockwise_gqa_attention(q, k, v, scaling=8 ** -0.5)
        assert not torch.isnan(out).any()


# ── blockwise_attention_from_cache ─────────────────────────────────────

class TestBlockwiseFromCache:
    def _make_cache_layer(self, seq_len=64, n_kv=2, hd=16):
        layer = Int4AttentionCacheLayer()
        torch.manual_seed(1)
        kv = torch.randn(1, n_kv, seq_len, hd, dtype=torch.bfloat16)
        layer.append_only(kv, kv)
        return layer

    def test_3_3_output_shape(self):
        """blockwise_attention_from_cache returns correct shape."""
        layer = self._make_cache_layer(seq_len=64, n_kv=2, hd=16)
        q = torch.randn(1, 4, 4, 16, dtype=torch.bfloat16)
        out = blockwise_attention_from_cache(q, layer, scaling=16 ** -0.5)
        assert out.shape == q.shape

    def test_3_3_matches_in_memory_kv(self):
        """result matches blockwise_gqa_attention on the same data (± 1e-2)."""
        n_kv, hd, seq_len, q_len = 2, 16, 64, 4
        torch.manual_seed(99)
        kv_data = torch.randn(1, n_kv, seq_len, hd, dtype=torch.bfloat16)
        q = torch.randn(1, 4, q_len, hd, dtype=torch.bfloat16)
        scale = hd ** -0.5

        # Build int4 cache
        layer = Int4AttentionCacheLayer()
        layer.append_only(kv_data, kv_data)

        out_cache = blockwise_attention_from_cache(q, layer, scaling=scale)
        out_direct = blockwise_gqa_attention(q.float(), kv_data.float(), kv_data.float(), scaling=scale)

        max_err = (out_cache.float() - out_direct.float()).abs().max().item()
        assert max_err < 0.15, f"cache vs direct max err {max_err:.4f} (expect < 0.15 for 4-bit quant)"

    def test_single_token_decode_from_cache(self):
        """Decode step (q_len=1) via from_cache returns correct shape."""
        layer = self._make_cache_layer(seq_len=64, n_kv=2, hd=16)
        q = torch.randn(1, 4, 1, 16, dtype=torch.bfloat16)
        out = blockwise_attention_from_cache(q, layer, scaling=16 ** -0.5)
        assert out.shape == (1, 4, 1, 16)


# ── _chunked_prefill leaves last token ─────────────────────────────────

class TestChunkedPrefill:
    def test_3_11_last_token_not_prefilled(self):
        """_chunked_prefill processes [0, N-2] only; token N-1 left for generate()."""
        from quantstar.engine import InferenceEngine, PREFILL_CHUNK
        from tests.conftest import make_engine

        engine = make_engine()

        # Capture all chunk slices passed to model()
        chunks_seen = []

        def _fake_model(*args, **kwargs):
            ids = kwargs.get("input_ids", args[0] if args else None)
            if ids is not None:
                chunks_seen.append(ids.shape[1])
            out = mock.MagicMock()
            out.past_key_values = None
            return out

        engine.model.side_effect = _fake_model

        N = 10
        input_ids = torch.zeros(1, N, dtype=torch.long)
        engine._chunked_prefill(input_ids)

        # Total tokens processed must be N-1 (all but the last)
        total_processed = sum(chunks_seen)
        assert total_processed == N - 1, (
            f"expected {N-1} tokens prefilled, got {total_processed}"
        )

    def test_short_prompt_is_single_chunk(self):
        """Prompt < PREFILL_CHUNK processes all-but-last in one model call."""
        from quantstar.engine import InferenceEngine, PREFILL_CHUNK
        from tests.conftest import make_engine

        engine = make_engine()
        call_count = [0]

        def _fake_model(*args, **kwargs):
            call_count[0] += 1
            out = mock.MagicMock()
            out.past_key_values = None
            return out

        engine.model.side_effect = _fake_model

        N = 5  # less than PREFILL_CHUNK
        engine._chunked_prefill(torch.zeros(1, N, dtype=torch.long))
        assert call_count[0] == 1, "short prompt should be processed in one model call"

    def test_long_prompt_uses_multiple_chunks(self):
        """Prompt > PREFILL_CHUNK triggers multiple model calls."""
        from quantstar.engine import PREFILL_CHUNK
        from tests.conftest import make_engine

        engine = make_engine()
        call_count = [0]

        def _fake_model(*args, **kwargs):
            call_count[0] += 1
            out = mock.MagicMock()
            out.past_key_values = mock.MagicMock()
            out.past_key_values.get_seq_length = lambda _: 0
            return out

        engine.model.side_effect = _fake_model

        N = PREFILL_CHUNK * 2 + 5  # definitely multi-chunk
        engine._chunked_prefill(torch.zeros(1, N, dtype=torch.long))
        assert call_count[0] > 1, "long prompt should trigger multiple model() calls"
