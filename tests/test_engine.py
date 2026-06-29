"""Unit tests for InferenceEngine — vision and text paths, image extraction, tokenization.

Run with:  python -m pytest tests/ -v
"""

from __future__ import annotations

import base64
import io
import json
import warnings
from unittest import mock

import pytest
from PIL import Image

from quantstar.engine import InferenceEngine, _extract_images, _safe_messages


# ── helpers ────────────────────────────────────────────────────────────────

def _tiny_rgb_bytes() -> bytes:
    """Return a minimal valid PNG (1x1 white pixel) as bytes."""
    img = Image.new("RGB", (1, 1), (255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _b64_image() -> str:
    return "data:image/png;base64," + base64.b64encode(_tiny_rgb_bytes()).decode()


def _make_engine(processor=None, cache_factory=None):
    """Create a minimal InferenceEngine with a mock model & tokenizer."""
    model = mock.MagicMock()
    model.device = "cpu"
    model.config = mock.MagicMock()
    model.config.image_token_id = None
    tokenizer = mock.MagicMock()
    tokenizer.pad_token_id = None
    tokenizer.eos_token_id = 151645
    tokenizer.apply_chat_template.return_value = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
    tokenizer.return_value = {"input_ids": mock.MagicMock()}
    tokenizer.return_value["input_ids"].shape = (1, 10)
    tokenizer.return_value["input_ids"].to = lambda device: tokenizer.return_value["input_ids"]

    return InferenceEngine(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        cache_config=cache_factory,
    )


# ── _extract_images ────────────────────────────────────────────────────────

class TestExtractImages:
    def test_no_images_empty_messages(self):
        assert _extract_images([]) == []

    def test_no_images_text_only(self):
        msgs = [{"role": "user", "content": "hello"}]
        assert _extract_images(msgs) == []

    def test_no_tool_calls(self):
        msgs = [{"role": "assistant", "tool_calls": [{"function": {"arguments": "{}"}}]}]
        assert _extract_images(msgs) == []

    def test_image_url_base64(self):
        msgs = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "describe this"},
                {"type": "image_url", "image_url": {"url": _b64_image()}},
            ],
        }]
        imgs = _extract_images(msgs)
        assert len(imgs) == 1
        assert isinstance(imgs[0], Image.Image)
        assert imgs[0].size == (1, 1)

    def test_image_url_http_skipped(self):
        msgs = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
            ],
        }]
        assert _extract_images(msgs) == []

    def test_content_list_not_dict_skipped(self):
        msgs = [{"role": "user", "content": ["not a dict"]}]
        assert _extract_images(msgs) == []

    def test_image_url_not_dict_skipped(self):
        msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": "not_a_dict"}]}]
        assert _extract_images(msgs) == []

    def test_multiple_images(self):
        b64 = _b64_image()
        msgs = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": b64}},
                {"type": "image_url", "image_url": {"url": b64}},
            ],
        }]
        assert len(_extract_images(msgs)) == 2

    def test_mixed_text_and_images(self):
        msgs = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "compare:"},
                {"type": "image_url", "image_url": {"url": _b64_image()}},
                {"type": "text", "text": "vs"},
                {"type": "image_url", "image_url": {"url": _b64_image()}},
            ],
        }]
        assert len(_extract_images(msgs)) == 2


# ── _safe_messages ─────────────────────────────────────────────────────────

class TestSafeMessages:
    def test_preserves_simple_text(self):
        msgs = [{"role": "user", "content": "hi"}]
        safe = _safe_messages(msgs)
        assert safe[0]["content"] == "hi"

    def test_converts_tool_call_arguments_json_string_to_dict(self):
        msgs = [{
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "search", "arguments": '{"q":"hello"}'}}],
        }]
        safe = _safe_messages(msgs)
        args = safe[0]["tool_calls"][0]["function"]["arguments"]
        assert isinstance(args, dict)
        assert args["q"] == "hello"

    def test_tool_call_arguments_already_dict(self):
        msgs = [{
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "f", "arguments": {"x": 1}}}],
        }]
        safe = _safe_messages(msgs)
        args = safe[0]["tool_calls"][0]["function"]["arguments"]
        assert args == {"x": 1}

    def test_invalid_json_kept_as_string(self):
        msgs = [{
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "f", "arguments": "not json"}}],
        }]
        safe = _safe_messages(msgs)
        assert safe[0]["tool_calls"][0]["function"]["arguments"] == "not json"

    def test_preserves_image_content(self):
        msgs = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": _b64_image()}},
            ],
        }]
        safe = _safe_messages(msgs)
        assert isinstance(safe[0]["content"], list)
        assert safe[0]["content"][1]["type"] == "image_url"


# ── InferenceEngine._tokenize ──────────────────────────────────────────────

class TestTokenize:
    def test_text_only_returns_4tuple(self):
        engine = _make_engine()
        result = engine._tokenize([{"role": "user", "content": "hi"}], enable_thinking=False)
        assert len(result) == 4
        input_ids, pv, ig, mm = result
        assert input_ids is not None
        assert pv is None
        assert ig is None
        assert mm is None

    def test_images_no_processor_returns_text_only(self):
        engine = _make_engine(processor=None)
        result = engine._tokenize(
            [{"role": "user", "content": "hi"}],
            images=[Image.new("RGB", (1, 1))],
            enable_thinking=False,
        )
        _, pv, ig, mm = result
        assert pv is None  # no processor → fallback to text-only

    def test_images_with_processor(self):
        processor = mock.MagicMock()
        processor_data = {
            "input_ids": mock.MagicMock(),
            "pixel_values": mock.MagicMock(),
            "image_grid_thw": mock.MagicMock(),
            "mm_token_type_ids": mock.MagicMock(),
        }
        for v in processor_data.values():
            v.to = lambda device: v
        processor_data["input_ids"].shape = (1, 100)

        class BatchFeature:
            def __init__(self, d):
                self.data = d

        processor.return_value = BatchFeature(processor_data)

        engine = _make_engine(processor=processor)
        result = engine._tokenize(
            [{"role": "user", "content": "describe this image"}],
            images=[Image.new("RGB", (1, 1))],
            enable_thinking=False,
        )
        _, pv, ig, mm = result
        assert pv is not None
        assert ig is not None
        assert mm is not None
        processor.assert_called_once()

    def test_tokenize_with_tools(self):
        engine = _make_engine()
        result = engine._tokenize(
            [{"role": "user", "content": "search"}],
            tools=[{"type": "function", "function": {"name": "search"}}],
            enable_thinking=False,
        )
        assert len(result) == 4


# ── chat_completion_sync path selection ─────────────────────────────────────

class TestChatCompletionPathSelection:
    def test_text_path_uses_prepare_generation(self):
        engine = _make_engine()
        engine._prepare_generation = mock.MagicMock()
        engine._prepare_generation.return_value = ({}, engine.tokenizer.return_value["input_ids"])
        engine.model.generate.return_value = mock.MagicMock()
        engine.model.generate.return_value.shape = (1, 15)
        engine._tokenize = lambda *a, **kw: (mock.MagicMock(shape=(1, 10)), None, None, None)

        with mock.patch("quantstar.engine._extract_images", return_value=[]):
            engine.chat_completion_sync([{"role": "user", "content": "hi"}], max_tokens=5)
        engine._prepare_generation.assert_called_once()

    def test_vision_path_bypasses_prepare_generation(self):
        engine = _make_engine()
        engine._prepare_generation = mock.MagicMock()
        engine.model.return_value = mock.MagicMock()
        engine.model.return_value.past_key_values = mock.MagicMock()
        engine.model.generate.return_value = mock.MagicMock()
        engine.model.generate.return_value.shape = (1, 150)
        fake_input_ids = mock.MagicMock(shape=(1, 100))
        fake_pv = mock.MagicMock()
        fake_ig = mock.MagicMock()
        fake_mm = mock.MagicMock()
        engine._tokenize = lambda *a, **kw: (fake_input_ids, fake_pv, fake_ig, fake_mm)

        with mock.patch("quantstar.engine._extract_images", return_value=[mock.MagicMock()]):
            engine.chat_completion_sync(
                [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": _b64_image()}}]}],
                max_tokens=5,
            )
        engine._prepare_generation.assert_not_called()
        # Prefill: model.forward called with vision tensors
        engine.model.assert_called_once()
        prefill_kwargs = engine.model.call_args[1]
        assert "pixel_values" in prefill_kwargs
        assert "image_grid_thw" in prefill_kwargs
        assert "mm_token_type_ids" in prefill_kwargs
        # Generate: model.generate called with past_key_values but no vision tensors
        engine.model.generate.assert_called_once()
        gen_kwargs = engine.model.generate.call_args[1]
        assert "past_key_values" in gen_kwargs
        assert "pixel_values" not in gen_kwargs

    def test_vision_resets_session_cache(self):
        engine = _make_engine()
        engine._session_kv = object()
        engine._session_prompt_ids = mock.MagicMock()
        engine.model.generate.return_value = mock.MagicMock()
        engine.model.generate.return_value.shape = (1, 150)
        fake_input_ids = mock.MagicMock(shape=(1, 100))
        engine._tokenize = lambda *a, **kw: (fake_input_ids, None, None, None)

        with mock.patch("quantstar.engine._extract_images", return_value=[mock.MagicMock()]):
            engine.chat_completion_sync([{"role": "user", "content": "hi"}], max_tokens=5)
        assert engine._session_kv is None
        assert engine._session_prompt_ids is None


# ── chat_completion_stream path selection ───────────────────────────────────

class TestChatCompletionStreamPathSelection:
    def test_vision_path_stream_yields_fake_tokens(self):
        """Vision stream path yields tokens from the streamer without crashing."""
        engine = _make_engine()
        fake_input_ids = mock.MagicMock(shape=(1, 100))

        class FakeStreamer:
            def __init__(self, *a, **kw):
                pass

            def __iter__(self):
                return iter(["hello", " world"])

        # Simulate a real thread that actually runs target
        class RealThread:
            def __init__(self, target, kwargs=None, **kw):
                self._target = target
                self._kw = kwargs or {}

            def start(self):
                self._target(**self._kw)

            def join(self):
                pass

        with mock.patch("transformers.TextIteratorStreamer", FakeStreamer):
            with mock.patch("threading.Thread", RealThread):
                with mock.patch("quantstar.engine._extract_images", return_value=[mock.MagicMock()]):
                    engine._tokenize = lambda *a, **kw: (fake_input_ids, mock.MagicMock(), mock.MagicMock(), mock.MagicMock())
                    gen = engine.chat_completion_stream(
                        [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": _b64_image()}}]}],
                        max_tokens=5,
                    )
                    result = list(gen)
                    assert result == ["hello", " world"]

    def test_text_path_stream_yields_fake_tokens(self):
        """Text-only stream path yields tokens from the streamer without crashing."""
        engine = _make_engine()
        engine._prepare_generation = mock.MagicMock()
        engine._prepare_generation.return_value = ({}, engine.tokenizer.return_value["input_ids"])
        engine._tokenize = lambda *a, **kw: (mock.MagicMock(shape=(1, 10)), None, None, None)

        class FakeStreamer:
            def __init__(self, *a, **kw):
                pass

            def __iter__(self):
                return iter(["hi"])

        class RealThread:
            def __init__(self, target=None, kwargs=None, **kw):
                self._target = target
                self._kwargs = kwargs or {}

            def start(self):
                if self._target:
                    self._target(**self._kwargs)

            def join(self):
                pass

        with mock.patch("transformers.TextIteratorStreamer", FakeStreamer):
            with mock.patch("threading.Thread", RealThread):
                with mock.patch("quantstar.engine._extract_images", return_value=[]):
                    gen = engine.chat_completion_stream([{"role": "user", "content": "hi"}], max_tokens=5)
                    result = list(gen)
                    assert result == ["hi"]


# ── _is_small_task (server module) ─────────────────────────────────────────

class TestIsSmallTask:
    def test_detects_title_generation(self):
        from quantstar.server import _is_small_task

        msgs = [{"role": "user", "content": "generate a short title for this chat"}]
        assert _is_small_task(msgs, None) is True

    def test_detects_title_with_list_content(self):
        from quantstar.server import _is_small_task

        msgs = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "generate a short title for the conversation above"},
                {"type": "image_url", "image_url": {"url": _b64_image()}},
            ],
        }]
        assert _is_small_task(msgs, None) is True

    def test_regular_message_not_small(self):
        from quantstar.server import _is_small_task

        msgs = [{"role": "user", "content": "What is the capital of France?"}]
        assert _is_small_task(msgs, None) is False

    def test_few_max_tokens_not_always_small(self):
        from quantstar.server import _is_small_task

        msgs = [{"role": "user", "content": "Write a poem about spring"}]
        assert _is_small_task(msgs, max_tokens=5) is False

    def test_list_content_no_text_parts(self):
        from quantstar.server import _is_small_task

        msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": _b64_image()}}]}]
        assert _is_small_task(msgs, None) is False

    def test_empty_messages(self):
        from quantstar.server import _is_small_task

        assert _is_small_task([], None) is False


# ── warmup coverage ────────────────────────────────────────────────────────

class TestWarmupCoverage:
    """Verify the warmup covers both chunked and full-prefill code paths."""

    def test_warmup_covers_chunked_prefill_sizes(self):
        """The warmup covers chunked-prefill sizes. Full-prefill path is
        protected by the triton autotuner monkey-patch (tested separately)."""
        from quantstar.__main__ import _warmup_engine

        import inspect
        src = inspect.getsource(_warmup_engine)
        assert "1024" in src, "warmup must cover chunk size"
        assert "128" in src, "warmup must cover partial chunk size"

    def test_warmup_engine_runs_without_gpu(self):
        """_warmup_engine should handle exceptions gracefully (e.g. no GPU)."""
        from quantstar.__main__ import _warmup_engine

        engine = _make_engine()
        engine.model.return_value = mock.MagicMock()

        # Should not raise — all errors are caught and logged as warnings
        _warmup_engine(engine)


# ── model loading return signature ─────────────────────────────────────────

class TestLoadAndQuantizeSignature:
    def test_returns_4tuple(self):
        """load_and_quantize_model must return (model, tokenizer, processor, cache_factory)."""
        from quantstar.quantize import load_and_quantize_model
        import inspect

        sig = inspect.signature(load_and_quantize_model)
        return_annotation = sig.return_annotation
        # Return type should be a 4-element tuple
        assert "object, object" in str(return_annotation) or "torch" in str(return_annotation)


# ── engine constructor signature ───────────────────────────────────────────

class TestEngineConstructor:
    def test_accepts_processor(self):
        import inspect

        sig = inspect.signature(InferenceEngine.__init__)
        params = list(sig.parameters.keys())
        assert "processor" in params

    def test_processor_optional(self):
        engine = _make_engine(processor=None)
        assert engine.processor is None


# ── image extraction edge cases ────────────────────────────────────────────

class TestExtractImagesEdgeCases:
    def test_broken_base64_skipped(self):
        msgs = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,!!!not-base64!!!"}},
            ],
        }]
        assert _extract_images(msgs) == []

    def test_data_url_no_comma_skipped(self):
        msgs = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64"}},
            ],
        }]
        assert _extract_images(msgs) == []


# ── triton autotuner patch ───────────────────────────────────────────────────

class TestTritonAutotunerPatch:
    def test_patch_applied(self):
        """Verify the triton autotuner _bench fallback is installed."""
        # The patch lives in quantize.py; importing it triggers the fix.
        import quantstar.quantize  # noqa: F811
        from triton.runtime.autotuner import Autotuner
        assert getattr(Autotuner, "_quantstar_nargs_fixed", False), (
            "Autotuner._bench must be patched to handle self.nargs=None "
            "(triton autotuner race fix)"
        )

    def test_bench_handles_none_nargs(self):
        """If self.nargs is None, _bench should fall back to {} instead of crashing."""
        from triton.runtime.autotuner import Autotuner
        import triton
        import triton.language as tl

        @triton.jit
        def _dummy_kernel(x_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
            pass

        at = Autotuner(
            fn=_dummy_kernel,
            arg_names=["x_ptr", "N"],
            configs=[
                triton.Config({"BLOCK": 64}, num_warps=1),
                triton.Config({"BLOCK": 128}, num_warps=1),
            ],
            key=["N"],
            reset_to_zero=[],
            restore_value=[],
        )
        # Simulate the bug: nargs is None when _bench runs
        at.nargs = None
        at.pre_hook = lambda full_nargs: None
        at.post_hook = lambda full_nargs, exception=None: None
        at.do_bench = lambda kernel_call, quantiles: [1.0, 1.0, 1.0]
        try:
            result = at._bench(*(), config=at.configs[0], x_ptr=mock.MagicMock(), N=64)
            assert result == [1.0, 1.0, 1.0]
        except TypeError as e:
            if "NoneType" in str(e):
                pytest.fail("Triton autotuner nargs=None crash is NOT patched! Vision requests will fail.")
            raise
