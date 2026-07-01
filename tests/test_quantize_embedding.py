"""Tests for 8GB-tier embedding quantization.

Covers: _model_is_pre_quantized, QuantizedEmbedding forward pass,
        _quantize_embeddings model surgery.
All tests run on CPU — no GPU required.
"""
from __future__ import annotations

import json
import os
import tempfile

import torch


# ── _model_is_pre_quantized ────────────────────────────────────────────

class TestModelIsPreQuantized:
    def test_detects_bitsandbytes_quant_method(self):
        from quantstar.quantize import _model_is_pre_quantized
        with tempfile.TemporaryDirectory() as d:
            cfg = {"quantization_config": {"quant_method": "bitsandbytes"}}
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump(cfg, f)
            assert _model_is_pre_quantized(d) is True

    def test_returns_false_for_unquantized_model(self):
        from quantstar.quantize import _model_is_pre_quantized
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump({"model_type": "qwen3"}, f)
            assert _model_is_pre_quantized(d) is False

    def test_returns_false_for_different_quant_method(self):
        from quantstar.quantize import _model_is_pre_quantized
        with tempfile.TemporaryDirectory() as d:
            cfg = {"quantization_config": {"quant_method": "gptq"}}
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump(cfg, f)
            assert _model_is_pre_quantized(d) is False

    def test_returns_false_for_nonexistent_path(self):
        from quantstar.quantize import _model_is_pre_quantized
        assert _model_is_pre_quantized("/nonexistent/path/12345") is False

    def test_returns_false_for_malformed_json(self):
        from quantstar.quantize import _model_is_pre_quantized
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "config.json"), "w") as f:
                f.write("not json {{{")
            assert _model_is_pre_quantized(d) is False


# ── QuantizedEmbedding ─────────────────────────────────────────────────

def _make_quantized_embedding(num_emb: int, emb_dim: int, seed: int = 0):
    """Quantize a random nn.Embedding and return (QuantizedEmbedding, original_weight)."""
    from quantstar.quantize import QuantizedEmbedding, _EMBED_GROUP_SIZE

    torch.manual_seed(seed)
    w = torch.randn(num_emb, emb_dim)

    num_groups = (emb_dim + _EMBED_GROUP_SIZE - 1) // _EMBED_GROUP_SIZE
    pad = num_groups * _EMBED_GROUP_SIZE - emb_dim
    w_padded = torch.nn.functional.pad(w, (0, pad)) if pad else w

    w_f = w_padded.float().reshape(num_emb, num_groups, _EMBED_GROUP_SIZE)
    w_min = w_f.amin(dim=-1)
    w_max = w_f.amax(dim=-1)
    scale = (w_max - w_min).clamp(min=1e-9) / 15.0
    zp = (-w_min / scale).round().clamp(0, 15).to(torch.int32)
    q = ((w_f / scale.unsqueeze(-1)).round() + zp.unsqueeze(-1)).clamp(0, 15).to(torch.int32)

    gs = _EMBED_GROUP_SIZE
    q = torch.nn.functional.pad(q, (0, (8 - gs % 8) % 8))
    q = q.reshape(num_emb, num_groups, -1, 8)
    packed = torch.zeros(num_emb, num_groups, q.shape[2], dtype=torch.int32)
    for i in range(8):
        packed |= (q[..., i] & 0xF) << (i * 4)

    qemb = QuantizedEmbedding(num_emb, emb_dim, packed, scale.to(torch.bfloat16), zp)
    return qemb, w


class TestQuantizedEmbeddingForward:
    def test_output_shape_1d_indices(self):
        qemb, _ = _make_quantized_embedding(100, 64)
        indices = torch.tensor([0, 5, 99])
        out = qemb(indices)
        assert out.shape == (3, 64)

    def test_output_shape_2d_indices(self):
        qemb, _ = _make_quantized_embedding(100, 64)
        indices = torch.randint(0, 100, (2, 4))
        out = qemb(indices)
        assert out.shape == (2, 4, 64)

    def test_output_dtype_is_bfloat16(self):
        qemb, _ = _make_quantized_embedding(50, 128)
        out = qemb(torch.tensor([0, 1]))
        assert out.dtype == torch.bfloat16

    def test_empty_indices_returns_empty(self):
        qemb, _ = _make_quantized_embedding(50, 64)
        out = qemb(torch.tensor([], dtype=torch.long))
        assert out.shape == (0, 64)

    def test_round_trip_accuracy(self):
        """4-bit dequantized values are within expected tolerance of the originals.

        Random normal data spans ~±3σ; with 15 levels that gives a step of ~range/15,
        so max quantization error is ~range/30 ≈ 0.4 in the worst case. 0.30 is a
        reasonable bound that catches clearly broken quantization without being too tight.
        """
        qemb, w_orig = _make_quantized_embedding(200, 256, seed=42)
        indices = torch.arange(200)
        out = qemb(indices).float()
        max_err = (out - w_orig).abs().max().item()
        assert max_err < 0.30, f"max round-trip error {max_err:.4f} exceeds 4-bit tolerance"

    def test_embedding_dim_not_multiple_of_group_size(self):
        """Works when embedding_dim is not a multiple of _EMBED_GROUP_SIZE (128)."""
        qemb, _ = _make_quantized_embedding(50, 100)  # 100 < 128
        out = qemb(torch.tensor([0, 1, 2]))
        assert out.shape == (3, 100)
        assert not torch.isnan(out).any()

    def test_different_indices_give_different_outputs(self):
        qemb, _ = _make_quantized_embedding(100, 64, seed=7)
        out0 = qemb(torch.tensor([0]))
        out1 = qemb(torch.tensor([1]))
        assert not torch.allclose(out0, out1), "distinct rows should differ"


# ── _quantize_embeddings model surgery ────────────────────────────────

class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(200, 64)
        self.linear = torch.nn.Linear(64, 32)

    def forward(self, x):
        # cast to float32 to match the linear weight dtype (real models use bfloat16 throughout)
        return self.linear(self.embedding(x).float())


class _NestedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sub = _TinyModel()
        self.extra_emb = torch.nn.Embedding(50, 16)


class TestQuantizeEmbeddings:
    def test_replaces_embedding_with_quantized(self):
        from quantstar.quantize import _quantize_embeddings, QuantizedEmbedding
        model = _TinyModel()
        _quantize_embeddings(model)
        assert isinstance(model.embedding, QuantizedEmbedding)

    def test_linear_not_replaced(self):
        from quantstar.quantize import _quantize_embeddings
        model = _TinyModel()
        _quantize_embeddings(model)
        assert isinstance(model.linear, torch.nn.Linear)

    def test_nested_embeddings_all_replaced(self):
        from quantstar.quantize import _quantize_embeddings, QuantizedEmbedding
        model = _NestedModel()
        _quantize_embeddings(model)
        assert isinstance(model.sub.embedding, QuantizedEmbedding)
        assert isinstance(model.extra_emb, QuantizedEmbedding)

    def test_old_gpu_weight_freed(self):
        """After replacement the original Embedding weight tensor is freed (empty)."""
        from quantstar.quantize import _quantize_embeddings
        model = _TinyModel()
        orig_emb = model.embedding
        _quantize_embeddings(model)
        assert orig_emb.weight.data.numel() == 0

    def test_forward_still_works_after_replacement(self):
        """Model forward pass runs without error after embedding surgery."""
        from quantstar.quantize import _quantize_embeddings
        model = _TinyModel()
        _quantize_embeddings(model)
        indices = torch.randint(0, 200, (4,))
        out = model(indices)
        assert out.shape == (4, 32)
        assert not torch.isnan(out).any()

    def test_output_approximately_matches_original(self):
        """Quantized embedding forward output is close to the original float output."""
        from quantstar.quantize import _quantize_embeddings

        class _WideModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(200, 256)

        torch.manual_seed(0)
        model = _WideModel()
        indices = torch.randint(0, 200, (10,))
        with torch.no_grad():
            ref = model.embedding(indices).float()
        _quantize_embeddings(model)
        with torch.no_grad():
            out = model.embedding(indices).float()
        max_err = (out - ref).abs().max().item()
        assert max_err < 0.30, f"max err {max_err:.4f} after quantization"
