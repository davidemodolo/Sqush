"""Tests for _quantize_visual_encoder.

All tests run on CPU — no GPU required. bitsandbytes is mocked to avoid
requiring a CUDA device.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import patch

import torch
import torch.nn as nn

# Import at module level so quantstar.quantize lands in sys.modules once,
# before any patch.dict context can evict it on exit.
from quantstar.quantize import _quantize_visual_encoder


# ── helpers ────────────────────────────────────────────────────────────────

class _FakeLinear4bit(nn.Module):
    """Stand-in for bnb.nn.Linear4bit that accepts Params4bit as weight."""

    def __init__(self, in_features, out_features, bias=True, **_):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_parameter("bias", nn.Parameter(torch.zeros(out_features)) if bias else None)

    def __setattr__(self, name, value):
        if name == "weight":
            # bypass nn.Module's Parameter-only restriction for weight
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)


class _FakeParams4bit:
    def __init__(self, data, **_):
        self._data = data

    def __repr__(self):
        return f"FakeParams4bit({self._data.shape})"


def _make_bnb_mock():
    """Build a minimal bitsandbytes mock with nn.Linear4bit and nn.Params4bit."""
    bnb = types.ModuleType("bitsandbytes")
    bnb.nn = types.ModuleType("bitsandbytes.nn")
    bnb.nn.Linear4bit = _FakeLinear4bit
    bnb.nn.Params4bit = _FakeParams4bit
    return bnb


def _make_model_with_visual():
    """Model with a 'visual' submodule containing two Linear layers."""

    class _VisualEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(32, 64)
            self.out = nn.Linear(64, 16)

    class _VLModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = _VisualEncoder()
            self.lm_head = nn.Linear(16, 8)

    return _VLModel()


def _call_with_mock(model):
    bnb_mock = _make_bnb_mock()
    with patch.dict(sys.modules, {"bitsandbytes": bnb_mock, "bitsandbytes.nn": bnb_mock.nn}):
        _quantize_visual_encoder(model)


# ── tests ──────────────────────────────────────────────────────────────────

class TestQuantizeVisualEncoder:
    def test_replaces_linear_in_visual_with_linear4bit(self):
        model = _make_model_with_visual()
        _call_with_mock(model)
        assert isinstance(model.visual.proj, _FakeLinear4bit)
        assert isinstance(model.visual.out, _FakeLinear4bit)

    def test_replaces_all_unquantized_linears(self):
        """Walk-all-modules: every nn.Linear (including lm_head) is replaced.

        In the real pre-quantized model the LM layers are already bnb.Linear4bit
        so only the visual encoder's nn.Linear layers are touched. In this mock
        model nothing is pre-quantized, so all linears are replaced — which is
        the correct behaviour.
        """
        model = _make_model_with_visual()
        _call_with_mock(model)
        assert isinstance(model.lm_head, _FakeLinear4bit)

    def test_original_linear_weight_freed(self):
        model = _make_model_with_visual()
        orig_proj = model.visual.proj
        _call_with_mock(model)
        assert orig_proj.weight.data.numel() == 0

    def test_all_linear4bit_is_noop(self):
        """Model with only already-quantized layers returns without replacing anything."""

        class _AlreadyFullyQuantized(nn.Module):
            def __init__(self):
                super().__init__()
                self.lm_head = _FakeLinear4bit(16, 8)

        model = _AlreadyFullyQuantized()
        bnb_mock = _make_bnb_mock()
        with patch.dict(sys.modules, {"bitsandbytes": bnb_mock, "bitsandbytes.nn": bnb_mock.nn}):
            _quantize_visual_encoder(model)
        assert isinstance(model.lm_head, _FakeLinear4bit)

    def test_already_quantized_layers_skipped(self):
        """If all Linear layers are already Linear4bit, nothing changes."""

        class _AlreadyQuantized(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = _FakeLinear4bit(32, 64)

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.visual = _AlreadyQuantized()

        model = _Model()
        bnb_mock = _make_bnb_mock()
        with patch.dict(sys.modules, {"bitsandbytes": bnb_mock, "bitsandbytes.nn": bnb_mock.nn}):
            _quantize_visual_encoder(model)
        assert isinstance(model.visual.proj, _FakeLinear4bit)

    def test_vision_model_attr_also_found(self):
        """Linear layers under any attribute name are found (walk-all-modules)."""

        class _VisionModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(8, 16)

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.vision_model = _VisionModel()
                self.lm_head = nn.Linear(16, 4)

        model = _Model()
        _call_with_mock(model)
        assert isinstance(model.vision_model.proj, _FakeLinear4bit)
        assert isinstance(model.lm_head, _FakeLinear4bit)

    def test_nested_encoder_without_visual_attr(self):
        """Linears nested under model.model.encoder (no 'visual' attr) are found.

        Qwen3.5-VL puts the visual encoder at model.model.visual, not at the
        top level. The walk-all-modules approach must find it regardless.
        """

        class _Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(16, 32)

        class _InnerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = _Encoder()

        class _OuterModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = _InnerModel()

        model = _OuterModel()
        _call_with_mock(model)
        assert isinstance(model.model.encoder.proj, _FakeLinear4bit)

    def test_linear4bit_dimensions_match_original(self):
        model = _make_model_with_visual()
        _call_with_mock(model)
        assert model.visual.proj.in_features == 32
        assert model.visual.proj.out_features == 64
        assert model.visual.out.in_features == 64
        assert model.visual.out.out_features == 16

    def test_missing_bitsandbytes_logs_warning_and_returns(self):
        """If bitsandbytes is not installed, function returns without error."""
        model = _make_model_with_visual()
        with patch.dict(sys.modules, {"bitsandbytes": None, "bitsandbytes.nn": None}):
            _quantize_visual_encoder(model)
        assert isinstance(model.visual.proj, nn.Linear)

    def test_bias_preserved_when_present(self):
        model = _make_model_with_visual()
        _call_with_mock(model)
        assert model.visual.proj.bias is not None

    def test_no_bias_when_original_has_none(self):
        class _NoBiasEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(32, 64, bias=False)

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.visual = _NoBiasEncoder()

        model = _Model()
        _call_with_mock(model)
        assert isinstance(model.visual.proj, _FakeLinear4bit)
        assert model.visual.proj.bias is None
