"""Tests for load-time patches in quantstar.quantize:

- _patch_chat_template_preserve_thinking: the Qwen3.5-9B (8 GB tier) chat template
  unconditionally strips prior-turn <think> blocks, which breaks session KV reuse —
  the re-rendered prompt no longer matches the tokens the cache was built from.
  The patch adds the same `preserve_thinking` escape hatch the Qwen3.6-27B template has.
- _quantize_lm_head: lm_head (2.03 GB bfloat16) must be NF4-quantized AFTER load on
  the 8 GB tier. It must never be removed from llm_int8_skip_modules — on a
  pre-quantized checkpoint that makes from_pretrained build a Linear4bit around the
  raw bf16 weight with no quant_state, and bitsandbytes asserts on the first forward
  (`assert module.weight.shape[1] == 1`).
"""
from __future__ import annotations

from unittest import mock

import torch

from quantstar.quantize import (
    _TEMPLATE_PRESERVE_THINKING,
    _TEMPLATE_STRIP_THINKING,
    _patch_chat_template_preserve_thinking,
    _quantize_lm_head,
)


class TestPreserveThinkingTemplatePatch:
    def test_patches_9b_style_template(self):
        """The strip-thinking condition is replaced with a preserve_thinking-aware one."""
        tok = mock.MagicMock()
        tok.chat_template = "{%- for m in messages %}" + _TEMPLATE_STRIP_THINKING + "{{ x }}{%- endif %}{%- endfor %}"
        _patch_chat_template_preserve_thinking(tok)
        assert _TEMPLATE_PRESERVE_THINKING in tok.chat_template
        assert _TEMPLATE_STRIP_THINKING not in tok.chat_template.replace(_TEMPLATE_PRESERVE_THINKING, "")

    def test_noop_when_template_already_supports_preserve_thinking(self):
        """The 27B template already honors preserve_thinking — must not be touched."""
        tok = mock.MagicMock()
        original = "{%- if (preserve_thinking is defined and preserve_thinking is true) or (loop.index0 > ns.last_query_index) %}"
        tok.chat_template = original
        _patch_chat_template_preserve_thinking(tok)
        assert tok.chat_template == original

    def test_noop_when_no_template(self):
        tok = mock.MagicMock()
        tok.chat_template = None
        _patch_chat_template_preserve_thinking(tok)
        assert tok.chat_template is None

    def test_patched_template_renders_prior_thinking(self):
        """End-to-end with jinja2: after the patch, prior-turn reasoning_content is
        rendered when preserve_thinking=True (the prerequisite for KV cache hits)."""
        import jinja2

        # Minimal excerpt of the 9B template structure around the strip condition.
        template = (
            "{%- set ns = namespace(last_query_index=99) %}"
            "{%- for message in messages %}"
            + _TEMPLATE_STRIP_THINKING +
            "{{- message.reasoning_content }}"
            "{%- endif %}"
            "{{- message.content }}"
            "{%- endfor %}"
        )
        tok = mock.MagicMock()
        tok.chat_template = template
        _patch_chat_template_preserve_thinking(tok)

        env = jinja2.Environment()
        msgs = [{"role": "assistant", "content": "4.", "reasoning_content": "THINKING"}]

        unpatched = env.from_string(template).render(messages=msgs, preserve_thinking=True)
        patched = env.from_string(tok.chat_template).render(messages=msgs, preserve_thinking=True)
        assert "THINKING" not in unpatched
        assert "THINKING" in patched


class _FakeParams4bit:
    def __init__(self, w, **kw):
        self.w = w


class _FakeLinear4bit(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, **kw):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

    def to(self, *a, **kw):  # .to("cuda") triggers quantization in real bnb — no-op here
        return self


def _patch_bnb():
    """Patch the bitsandbytes classes _quantize_lm_head uses with CPU-safe fakes."""
    return (
        mock.patch("bitsandbytes.nn.Linear4bit", _FakeLinear4bit),
        mock.patch("bitsandbytes.nn.Params4bit", _FakeParams4bit),
        mock.patch("torch.cuda.is_available", return_value=True),
        mock.patch("torch.cuda.empty_cache"),
    )


class TestQuantizeLmHead:
    def _model(self, out_f=8, in_f=4):
        model = torch.nn.Module()
        model.lm_head = torch.nn.Linear(in_f, out_f, bias=False, dtype=torch.bfloat16)
        return model

    def test_replaces_plain_linear_lm_head(self):
        """A bf16 nn.Linear lm_head is replaced by a (quantized) Linear4bit.

        Regression guard for the 8 GB tier crash: lm_head must be quantized
        post-load, not by removing it from llm_int8_skip_modules — that path
        produces a Linear4bit with no quant_state and bitsandbytes asserts
        (`module.weight.shape[1] == 1`) on the first forward."""
        model = self._model()
        original_w = model.lm_head.weight.data.clone()
        patches = _patch_bnb()
        with patches[0], patches[1], patches[2], patches[3]:
            _quantize_lm_head(model)
        assert isinstance(model.lm_head, _FakeLinear4bit)
        assert model.lm_head.in_features == 4
        assert model.lm_head.out_features == 8
        # The weight handed to Params4bit is the original data (as fp16 on CPU)
        assert torch.equal(model.lm_head.weight.w, original_w.cpu().to(torch.float16))

    def test_noop_when_already_quantized(self):
        model = torch.nn.Module()
        head = _FakeLinear4bit(4, 8)
        model.lm_head = head
        patches = _patch_bnb()
        with patches[0], patches[1], patches[2], patches[3]:
            _quantize_lm_head(model)
        assert model.lm_head is head

    def test_noop_without_cuda(self):
        """NF4 quantization requires CUDA — lm_head is left untouched on CPU-only."""
        model = self._model()
        head = model.lm_head
        with mock.patch("torch.cuda.is_available", return_value=False):
            _quantize_lm_head(model)
        assert model.lm_head is head
        assert model.lm_head.weight.shape == (8, 4)

    def test_noop_when_no_lm_head(self):
        model = torch.nn.Module()
        patches = _patch_bnb()
        with patches[0], patches[1], patches[2], patches[3]:
            _quantize_lm_head(model)  # must not raise
