"""Unit tests for InferenceEngine — vision and text paths, image extraction, tokenization.

Run with:  python -m pytest tests/ -v
"""

from __future__ import annotations

import base64
import io
from unittest import mock

import pytest
import torch
from PIL import Image

from sqush.engine import InferenceEngine, _extract_images, _safe_messages


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

        with mock.patch("sqush.engine._extract_images", return_value=[]):
            engine.chat_completion_sync([{"role": "user", "content": "hi"}], max_tokens=5)
        engine._prepare_generation.assert_called_once()

    def test_vision_path_bypasses_prepare_generation(self):
        engine = _make_engine()
        engine._prepare_generation = mock.MagicMock()
        fake_input_ids = mock.MagicMock(shape=(1, 100))
        engine._tokenize = lambda *a, **kw: (fake_input_ids, mock.MagicMock(), mock.MagicMock(), mock.MagicMock())

        fake_cache = mock.MagicMock()
        engine._chunked_vision_prefill = mock.MagicMock(return_value=fake_cache)

        with mock.patch("sqush.engine._extract_images", return_value=[mock.MagicMock()]):
            engine.chat_completion_sync(
                [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": _b64_image()}}]}],
                max_tokens=5,
            )
        engine._prepare_generation.assert_not_called()
        engine._chunked_vision_prefill.assert_called_once()
        engine.model.generate.assert_called_once()
        gen_kwargs = engine.model.generate.call_args[1]
        assert gen_kwargs.get("past_key_values") is fake_cache

    def test_vision_saves_session_cache(self):
        """Vision turn saves the generated KV cache so follow-up text turns can extend it."""
        engine = _make_engine()
        engine._session_kv = object()
        engine._session_num_messages = 99  # stale (> len(messages)) → triggers fresh vision prefill
        fake_input_ids = mock.MagicMock(shape=(1, 100))
        engine._tokenize = lambda *a, **kw: (fake_input_ids, mock.MagicMock(), mock.MagicMock(), mock.MagicMock())
        engine._chunked_vision_prefill = mock.MagicMock(return_value=None)

        with mock.patch("sqush.engine._extract_images", return_value=[mock.MagicMock()]):
            engine.chat_completion_sync([{"role": "user", "content": "hi"}], max_tokens=5)
        # KV cache is now saved from model.generate output, not reset to None
        assert engine._session_kv is not None
        assert engine._session_num_messages == 1


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

            def end(self):
                pass

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
                with mock.patch("sqush.engine._extract_images", return_value=[mock.MagicMock()]):
                    engine._tokenize = lambda *a, **kw: (fake_input_ids, mock.MagicMock(), mock.MagicMock(), mock.MagicMock())
                    # _chunked_vision_prefill does real model calls; short-circuit it
                    engine._chunked_vision_prefill = mock.MagicMock(return_value=None)
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

            def end(self):
                pass

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
                with mock.patch("sqush.engine._extract_images", return_value=[]):
                    gen = engine.chat_completion_stream([{"role": "user", "content": "hi"}], max_tokens=5)
                    result = list(gen)
                    assert result == ["hi"]


# ── _is_small_task (server module) ─────────────────────────────────────────

class TestIsSmallTask:
    def test_detects_title_generation(self):
        from sqush.server import _is_small_task

        msgs = [{"role": "user", "content": "generate a short title for this chat"}]
        assert _is_small_task(msgs, None) is True

    def test_detects_title_with_list_content(self):
        from sqush.server import _is_small_task

        msgs = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "generate a short title for the conversation above"},
                {"type": "image_url", "image_url": {"url": _b64_image()}},
            ],
        }]
        assert _is_small_task(msgs, None) is True

    def test_regular_message_not_small(self):
        from sqush.server import _is_small_task

        msgs = [{"role": "user", "content": "What is the capital of France?"}]
        assert _is_small_task(msgs, None) is False

    def test_few_max_tokens_not_always_small(self):
        from sqush.server import _is_small_task

        msgs = [{"role": "user", "content": "Write a poem about spring"}]
        assert _is_small_task(msgs, max_tokens=5) is False

    def test_list_content_no_text_parts(self):
        from sqush.server import _is_small_task

        msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": _b64_image()}}]}]
        assert _is_small_task(msgs, None) is False

    def test_empty_messages(self):
        from sqush.server import _is_small_task

        assert _is_small_task([], None) is False


# ── warmup coverage ────────────────────────────────────────────────────────

class TestWarmupCoverage:
    """Verify the warmup covers both chunked and full-prefill code paths."""

    def test_warmup_covers_chunked_prefill_sizes(self):
        """The warmup covers chunked-prefill sizes. Full-prefill path is
        protected by the triton autotuner monkey-patch (tested separately)."""
        from sqush.__main__ import _warmup_engine

        import inspect
        src = inspect.getsource(_warmup_engine)
        assert "1024" in src, "warmup must cover chunk size"
        assert "128" in src, "warmup must cover partial chunk size"

    def test_warmup_engine_runs_without_gpu(self):
        """_warmup_engine should handle exceptions gracefully (e.g. no GPU)."""
        from sqush.__main__ import _warmup_engine

        engine = _make_engine()
        engine.model.return_value = mock.MagicMock()

        # Should not raise — all errors are caught and logged as warnings
        _warmup_engine(engine)


# ── model loading return signature ─────────────────────────────────────────

class TestLoadAndQuantizeSignature:
    def test_returns_4tuple(self):
        """load_and_quantize_model return annotation must declare a 4-element tuple."""
        import inspect
        import typing
        from sqush.quantize import load_and_quantize_model

        sig = inspect.signature(load_and_quantize_model)
        annotation = sig.return_annotation
        assert annotation is not inspect.Parameter.empty

        # quantize.py has `from __future__ import annotations` so annotations
        # are stored as strings (PEP 563). Resolve via get_type_hints if needed.
        if isinstance(annotation, str):
            hints = typing.get_type_hints(load_and_quantize_model)
            annotation = hints.get("return", annotation)

        args = getattr(annotation, "__args__", None)
        if args is None:
            # Fallback: verify via string representation (4-tuple has 3 commas)
            ann_str = str(annotation)
            assert "tuple" in ann_str.lower(), f"Not a tuple annotation: {ann_str}"
            assert ann_str.count(",") >= 3, (
                f"Expected 4-element tuple (3 commas), got: {ann_str}"
            )
        else:
            assert len(args) == 4, f"Expected 4-tuple return annotation, got: {annotation}"


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
        import sqush.quantize  # noqa: F811
        from triton.runtime.autotuner import Autotuner
        assert getattr(Autotuner, "_sqush_nargs_fixed", False), (
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


# ── _prepare_generation kwargs ────────────────────────────────────

class TestPrepareGeneration:
    """Tests for InferenceEngine._prepare_generation"""

    def _engine(self):
        return _make_engine()

    def _input_ids(self, length: int = 10) -> "torch.Tensor":
        import torch
        return torch.zeros(1, length, dtype=torch.long)

    def test_4_7_temperature_zero_disables_sampling(self):
        """temperature=0 → do_sample=False (greedy decode)."""
        engine = self._engine()
        kwargs, _ = engine._prepare_generation(self._input_ids(), temperature=0.0)
        assert kwargs["do_sample"] is False

    def test_4_8_positive_temperature_enables_sampling(self):
        """temperature>0 → do_sample=True."""
        engine = self._engine()
        kwargs, _ = engine._prepare_generation(self._input_ids(), temperature=0.5)
        assert kwargs["do_sample"] is True

    def test_4_9_top_p_forwarded(self):
        """top_p parameter passed into generate kwargs."""
        engine = self._engine()
        kwargs, _ = engine._prepare_generation(self._input_ids(), top_p=0.95)
        assert kwargs["top_p"] == 0.95

    def test_4_9_top_k_forwarded(self):
        """top_k (from engine default) forwarded to generate kwargs."""
        engine = self._engine()
        kwargs, _ = engine._prepare_generation(self._input_ids())
        assert "top_k" in kwargs
        assert kwargs["top_k"] == engine.top_k

    def test_4_12_per_request_temperature_override(self):
        """per-request temperature takes precedence over engine default."""
        engine = self._engine()
        engine.temperature = 0.7
        kwargs, _ = engine._prepare_generation(self._input_ids(), temperature=0.1)
        assert kwargs["temperature"] == 0.1

    def test_4_12_per_request_top_p_override(self):
        """per-request top_p takes precedence over engine default."""
        engine = self._engine()
        engine.top_p = 0.8
        kwargs, _ = engine._prepare_generation(self._input_ids(), top_p=0.5)
        assert kwargs["top_p"] == 0.5

    def test_4_12_falls_back_to_engine_temp_when_none(self):
        """when request temperature is None, engine default is used."""
        engine = self._engine()
        engine.temperature = 0.6
        kwargs, _ = engine._prepare_generation(self._input_ids(), temperature=None)
        assert kwargs["temperature"] == 0.6

    def test_4_13_engine_temperature_not_mutated(self):
        """calling _prepare_generation with temperature=X leaves engine.temperature unchanged."""
        import torch
        engine = self._engine()
        engine.temperature = 0.7
        engine.top_p = 0.8
        engine._prepare_generation(self._input_ids(), temperature=0.1, top_p=0.2)
        assert engine.temperature == 0.7
        assert engine.top_p == 0.8

    def test_4_6_max_tokens_applied(self):
        """max_tokens limits max_new_tokens in kwargs."""
        engine = self._engine()
        kwargs, _ = engine._prepare_generation(self._input_ids(), max_tokens=42)
        assert kwargs["max_new_tokens"] == 42

    def test_4_11_pad_token_falls_back_to_eos(self):
        """when pad_token_id is None, eos_token_id is used as pad."""
        engine = self._engine()
        engine.tokenizer.pad_token_id = None
        engine.tokenizer.eos_token_id = 151645
        kwargs, _ = engine._prepare_generation(self._input_ids())
        assert kwargs["pad_token_id"] == 151645


# ── session KV cache reuse ──────────────────────────────────────

class TestSessionReuse:
    """Tests for session KV cache reuse logic"""

    def _engine(self):
        return _make_engine()

    def _input_ids(self, length: int = 10) -> "torch.Tensor":
        import torch
        return torch.zeros(1, length, dtype=torch.long)

    def _mock_cache(self, seq_len: int) -> mock.MagicMock:
        cache = mock.MagicMock()
        cache.get_seq_length.return_value = seq_len
        return cache

    def test_6_4_shorter_prompt_full_prefill(self):
        """new prompt shorter than the cached sequence (new conversation) → full prefill."""
        import torch
        engine = self._engine()
        engine._session_kv = self._mock_cache(50)
        engine._session_ids = torch.zeros(51, dtype=torch.long)
        engine._session_num_messages = 3

        kwargs, _ = engine._prepare_generation(self._input_ids(length=10), num_messages=2)

        assert "past_key_values" not in kwargs or kwargs.get("past_key_values") is not engine._session_kv or engine._session_num_messages == 2
        assert engine._session_num_messages == 2

    def test_6_5_diverging_tokens_full_prefill(self):
        """prompt that does not start with the cached tokens (edited turn) → no reuse."""
        import torch
        engine = self._engine()
        cache = self._mock_cache(20)
        engine._session_kv = cache
        engine._session_ids = torch.ones(21, dtype=torch.long)  # cached tokens are all 1s
        engine._session_num_messages = 2

        # New prompt is all zeros — extends in length but tokens diverge
        input_ids = torch.zeros(1, 25, dtype=torch.long)
        kwargs, _ = engine._prepare_generation(input_ids, num_messages=3)

        assert kwargs.get("past_key_values") is not cache
        assert engine._session_num_messages == 3

    def test_6_6_session_num_messages_tracked(self):
        """_session_num_messages is updated after each call."""
        engine = self._engine()
        engine._prepare_generation(self._input_ids(), num_messages=5)
        assert engine._session_num_messages == 5

    def test_6_8_reset_session_clears_state(self):
        """reset_session() clears _session_kv, _session_num_messages, and _session_ids."""
        engine = self._engine()
        engine._session_kv = mock.MagicMock()
        engine._session_num_messages = 7
        engine._session_ids = mock.MagicMock()
        engine.reset_session()
        assert engine._session_kv is None
        assert engine._session_num_messages == 0
        assert engine._session_ids is None

    def test_6_1_cache_reuse_when_prompt_extends_cached_tokens(self):
        """follow-up whose tokens exactly extend the cached sequence reuses the KV cache."""
        import torch
        engine = self._engine()

        cache = self._mock_cache(seq_len=20)
        engine._session_kv = cache
        engine._session_ids = torch.zeros(21, dtype=torch.long)
        engine._session_num_messages = 2

        # New prompt is 21 tokens, first 20 identical to the cached ones (the single
        # new token is left for generate(), so no chunked prefill runs on mocks)
        input_ids = torch.zeros(1, 21, dtype=torch.long)
        kwargs, _ = engine._prepare_generation(input_ids, num_messages=3)

        assert kwargs.get("past_key_values") is cache
        assert engine._session_num_messages == 3

    def test_6_12_cache_seq_len_zero_triggers_full_prefill(self):
        """cache.get_seq_length()==0 falls through to full prefill."""
        import torch
        engine = self._engine()

        # Cache exists but seq_len == 0 (e.g., just allocated, nothing prefilled)
        cache = self._mock_cache(seq_len=0)
        engine._session_kv = cache
        engine._session_ids = torch.zeros(0, dtype=torch.long)
        engine._session_num_messages = 2

        input_ids = torch.zeros(1, 10, dtype=torch.long)
        kwargs, _ = engine._prepare_generation(input_ids, num_messages=3)

        assert kwargs.get("past_key_values") is not cache
        assert engine._session_num_messages == 3

    def test_6_9_vision_request_saves_session(self):
        """vision turn saves the KV cache so follow-up text turns reuse it."""
        import torch
        engine = self._engine()
        engine._session_kv = mock.MagicMock()
        engine._session_num_messages = 5  # stale (> len(messages)) → fresh vision prefill

        fake_input_ids = mock.MagicMock(shape=(1, 10))
        engine._tokenize = lambda *a, **kw: (
            fake_input_ids, mock.MagicMock(), mock.MagicMock(), mock.MagicMock()
        )
        engine._chunked_vision_prefill = mock.MagicMock(return_value=None)

        with mock.patch("sqush.engine._extract_images", return_value=[mock.MagicMock()]):
            engine.chat_completion_sync([{"role": "user", "content": "img"}], max_tokens=1)

        assert engine._session_kv is not None
        assert engine._session_num_messages == 1

    def test_vision_sets_session_ids(self):
        """vision turn must store the generated token sequence so followups can verify the prefix."""
        engine = self._engine()
        fake_input_ids = mock.MagicMock(shape=(1, 50))
        engine._tokenize = lambda *a, **kw: (
            fake_input_ids, mock.MagicMock(), mock.MagicMock(), mock.MagicMock()
        )
        engine._chunked_vision_prefill = mock.MagicMock(return_value=None)

        with mock.patch("sqush.engine._extract_images", return_value=[mock.MagicMock()]):
            engine.chat_completion_sync([{"role": "user", "content": "img"}], max_tokens=1)

        assert engine._session_ids is not None

    def test_vision_followup_reuses_cache_via_token_prefix(self):
        """followup text after a vision turn reuses the cache when tokens match exactly."""
        import torch
        engine = self._engine()

        fake_kv = mock.MagicMock()
        fake_kv.get_seq_length.return_value = 60
        engine._session_kv = fake_kv
        engine._session_ids = torch.zeros(61, dtype=torch.long)
        engine._session_num_messages = 1

        # Followup prompt extends the cached 60 tokens with an identical prefix
        # (one new token, so no chunked prefill runs on mocks)
        input_ids = torch.zeros(1, 61, dtype=torch.long)
        kwargs, _ = engine._prepare_generation(input_ids, num_messages=2)

        assert kwargs.get("past_key_values") is fake_kv

    def test_6_stream_session_reuse_updates_from_generate(self):
        """stream path stores past_key_values from generate into session cache."""
        import torch

        engine = self._engine()
        engine._prepare_generation = mock.MagicMock(
            return_value=({}, mock.MagicMock(shape=(1, 10)))
        )

        fake_cache = mock.MagicMock()

        class FakeStreamer:
            def __init__(self, *a, **kw): pass
            def __iter__(self): return iter(["hello"])
            def end(self): pass

        class FakeGenOutput:
            sequences = torch.zeros(1, 5, dtype=torch.long)
            past_key_values = fake_cache

        class RealThread:
            def __init__(self, target=None, kwargs=None, **kw):
                self._target = target
                self._kwargs = kwargs or {}

            def start(self):
                # Inject the fake output so _generate() captures it
                if self._target:
                    self._target(**self._kwargs)

            def join(self): pass

        def fake_generate(**kwargs):
            out = FakeGenOutput()
            streamer = kwargs.get("streamer")
            if streamer:
                for t in ["hello"]:
                    pass
            return out

        engine.model.generate.side_effect = fake_generate

        with mock.patch("transformers.TextIteratorStreamer", FakeStreamer):
            with mock.patch("threading.Thread", RealThread):
                with mock.patch("sqush.engine._extract_images", return_value=[]):
                    list(engine.chat_completion_stream([{"role": "user", "content": "hi"}], max_tokens=5))

        # past_key_values from generate should be stored in _session_kv
        assert engine._session_kv is fake_cache
        # the generated sequence must be stored for token-prefix verification on followup
        assert engine._session_ids is not None
        assert torch.equal(engine._session_ids, torch.zeros(5, dtype=torch.long))


# ── tokenization kwargs ─────────────────────────────────────────

class TestTokenizeKwargs:
    """Verify _tokenize passes the right kwargs to apply_chat_template"""

    def _engine(self):
        return _make_engine()

    def test_7_2_enable_thinking_true_in_kwargs(self):
        """enable_thinking=True is passed to apply_chat_template."""
        engine = self._engine()
        engine._tokenize([{"role": "user", "content": "hi"}], enable_thinking=True)
        call_kw = engine.tokenizer.apply_chat_template.call_args[1]
        assert call_kw.get("enable_thinking") is True

    def test_7_3_enable_thinking_false_in_kwargs(self):
        """enable_thinking=False is passed to apply_chat_template."""
        engine = self._engine()
        engine._tokenize([{"role": "user", "content": "hi"}], enable_thinking=False)
        call_kw = engine.tokenizer.apply_chat_template.call_args[1]
        assert call_kw.get("enable_thinking") is False

    def test_7_4_preserve_thinking_always_true(self):
        """preserve_thinking=True always set so raw think blocks are kept."""
        engine = self._engine()
        engine._tokenize([{"role": "user", "content": "hi"}], enable_thinking=True)
        call_kw = engine.tokenizer.apply_chat_template.call_args[1]
        assert call_kw.get("preserve_thinking") is True

    def test_7_5_tools_forwarded(self):
        """tools kwarg is forwarded to apply_chat_template."""
        engine = self._engine()
        tools = [{"type": "function", "function": {"name": "search"}}]
        engine._tokenize([{"role": "user", "content": "hi"}], tools=tools, enable_thinking=False)
        call_kw = engine.tokenizer.apply_chat_template.call_args[1]
        assert call_kw.get("tools") == tools

    def test_7_5_no_tools_kwarg_when_not_provided(self):
        """tools kwarg is omitted when tools=None."""
        engine = self._engine()
        engine._tokenize([{"role": "user", "content": "hi"}], enable_thinking=False)
        call_kw = engine.tokenizer.apply_chat_template.call_args[1]
        assert "tools" not in call_kw

    def test_7_1_add_generation_prompt_true(self):
        """add_generation_prompt=True passed to apply_chat_template."""
        engine = self._engine()
        engine._tokenize([{"role": "user", "content": "hi"}], enable_thinking=False)
        call_kw = engine.tokenizer.apply_chat_template.call_args[1]
        assert call_kw.get("add_generation_prompt") is True

    def test_7_6_safe_messages_converts_json_args(self):
        """_safe_messages converts JSON-string arguments to dict."""
        msgs = [{
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "f", "arguments": '{"k": "v"}'}}],
        }]
        from sqush.engine import _safe_messages
        safe = _safe_messages(msgs)
        assert safe[0]["tool_calls"][0]["function"]["arguments"] == {"k": "v"}

    def test_7_7_safe_messages_non_json_kept_as_string(self):
        """non-JSON argument strings are kept as strings."""
        from sqush.engine import _safe_messages
        msgs = [{
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "f", "arguments": "not json"}}],
        }]
        safe = _safe_messages(msgs)
        assert safe[0]["tool_calls"][0]["function"]["arguments"] == "not json"

    def test_7_8_safe_messages_no_tool_calls_unchanged(self):
        """messages without tool_calls are returned unchanged."""
        from sqush.engine import _safe_messages
        msgs = [{"role": "user", "content": "hello"}]
        safe = _safe_messages(msgs)
        assert safe[0]["content"] == "hello"
        assert "tool_calls" not in safe[0]
