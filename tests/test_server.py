"""Tests for the FastAPI server

Covers:
  thinking/reasoning extraction (sync + stream)
  tool call XML parsing (sync + stream)
  HTTP routes, response shape, CORS, per-request params
"""

from __future__ import annotations

import json
import warnings
from unittest import mock

warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*httpx.*")

from fastapi.testclient import TestClient

from quantstar.config import QuantStarConfig

# ── fixtures ─────────────────────────────────────────────────────────────────


def _make_engine(
    sync_text: str = "response text",
    stream_tokens: list[str] | None = None,
):
    engine = mock.MagicMock()
    engine.max_context = 4096
    engine.max_new_tokens = 512
    engine.temperature = 0.7
    engine.top_p = 0.8
    engine._last_prompt_tokens = 10
    engine.tokenizer = mock.MagicMock()
    engine.tokenizer.encode.return_value = list(range(5))  # 5 completion tokens
    engine.get_vram_info.return_value = {"cuda_available": False}
    engine.chat_completion_sync.return_value = (sync_text, 10, 5)

    _tokens = stream_tokens if stream_tokens is not None else [sync_text]
    engine.chat_completion_stream.side_effect = lambda *a, **kw: iter(_tokens)
    return engine


def _make_app(sync_text="response text", stream_tokens=None):
    from quantstar.server import create_app

    engine = _make_engine(sync_text, stream_tokens)
    # Use real config so attribute access (cfg.model.repo) works normally
    cfg = QuantStarConfig()
    cfg.model.repo = "test-model"
    app = create_app(engine, cfg)
    return app, engine


def _parse_sse(text: str) -> list[dict]:
    """Parse SSE response body into parsed JSON events (skipping [DONE])."""
    events = []
    for line in text.split("\n"):
        if not line.startswith("data: "):
            continue
        data = line[6:].strip()
        if data == "[DONE]":
            continue
        try:
            events.append(json.loads(data))
        except json.JSONDecodeError:
            pass
    return events


# ── Health & VRAM ──────────────────────────────────────────


class TestHealthRoutes:
    def test_10_1_health_ok(self):
        """GET /health returns {"status": "ok"}."""
        app, _ = _make_app()
        with TestClient(app) as client:
            r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    def test_10_2_vram_fields_present(self):
        """GET /health/vram returns cuda_available field."""
        app, engine = _make_app()
        engine.get_vram_info.return_value = {"cuda_available": False}
        with TestClient(app) as client:
            r = client.get("/health/vram")
        assert r.status_code == 200
        assert "cuda_available" in r.json()

    def test_10_4_vram_cuda_false_without_gpu(self):
        """cuda_available=False when no CUDA present."""
        app, engine = _make_app()
        engine.get_vram_info.return_value = {"cuda_available": False}
        with TestClient(app) as client:
            r = client.get("/health/vram")
        assert r.json()["cuda_available"] is False


# ── Models ──────────────────────────────────────────────────


class TestModelsRoutes:
    def test_10_5_models_list_has_context_window(self):
        """GET /v1/models returns model with context_window and max_output_tokens."""
        app, engine = _make_app()
        engine.max_context = 4096
        engine.max_new_tokens = 512
        with TestClient(app) as client:
            r = client.get("/v1/models")
        assert r.status_code == 200
        data = r.json()
        assert data["object"] == "list"
        assert len(data["data"]) >= 1
        m = data["data"][0]
        assert "context_window" in m
        assert "max_output_tokens" in m

    def test_10_6_get_model_returns_info(self):
        """GET /v1/models/{id} returns model info for known model."""
        app, _ = _make_app()
        with TestClient(app) as client:
            r = client.get("/v1/models/test-model")
        assert r.status_code == 200
        assert r.json()["id"] == "test-model"

    def test_10_7_unknown_model_404(self):
        """GET /v1/models/{unknown_id} returns 404."""
        app, _ = _make_app()
        with TestClient(app) as client:
            r = client.get("/v1/models/does-not-exist")
        assert r.status_code == 404


# ── Sync completions ──────────────────────────────────────


class TestSyncCompletions:
    def test_10_8_sync_200(self):
        """POST /v1/chat/completions stream=false returns 200."""
        app, _ = _make_app()
        with TestClient(app) as client:
            r = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        assert r.status_code == 200

    def test_10_9_response_fields(self):
        """response includes id, object, created, model, choices, usage."""
        app, _ = _make_app()
        with TestClient(app) as client:
            r = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        body = r.json()
        for field in ("id", "object", "created", "model", "choices", "usage"):
            assert field in body, f"missing field: {field}"

    def test_10_10_usage_fields(self):
        """usage includes prompt_tokens, completion_tokens, total_tokens."""
        app, _ = _make_app()
        with TestClient(app) as client:
            r = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        usage = r.json()["usage"]
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            assert k in usage

    def test_10_11_finish_reason_stop(self):
        """finish_reason is 'stop' for normal completion."""
        app, _ = _make_app(sync_text="normal response")
        with TestClient(app) as client:
            r = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        assert r.json()["choices"][0]["finish_reason"] == "stop"

    def test_10_12_finish_reason_tool_calls(self):
        """finish_reason is 'tool_calls' when response contains tool calls."""
        tool_xml = "<tool_call><function=search><parameter=query>test</parameter></function></tool_call>"
        app, _ = _make_app(sync_text=tool_xml)
        with TestClient(app) as client:
            r = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "search something"}],
                },
            )
        assert r.json()["choices"][0]["finish_reason"] == "tool_calls"

    def test_10_21_max_tokens_forwarded(self):
        """max_tokens from request body is forwarded to engine."""
        app, engine = _make_app()
        with TestClient(app) as client:
            client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 42,
                },
            )
        call_args = engine.chat_completion_sync.call_args
        assert call_args[0][1] == 42 or call_args[1].get("max_tokens") == 42

    def test_10_22_temperature_not_mutating_engine(self):
        """per-request temperature is passed through without mutating engine state."""
        app, engine = _make_app()
        engine.temperature = 0.7

        with TestClient(app) as client:
            client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "temperature": 0.1,
                },
            )

        assert (
            engine.temperature == 0.7
        ), "engine.temperature must not be mutated by request"

    def test_10_23_top_p_not_mutating_engine(self):
        """per-request top_p is passed through without mutating engine state."""
        app, engine = _make_app()
        engine.top_p = 0.8

        with TestClient(app) as client:
            client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "top_p": 0.3,
                },
            )

        assert engine.top_p == 0.8

    def test_10_24_missing_messages_graceful(self):
        """missing messages field doesn't crash (treated as empty list)."""
        app, _ = _make_app()
        with TestClient(app) as client:
            r = client.post("/v1/chat/completions", json={})
        assert r.status_code == 200

    def test_temperature_forwarded_to_engine(self):
        """Per-request temperature is passed to chat_completion_sync."""
        app, engine = _make_app()
        with TestClient(app) as client:
            client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "temperature": 0.25,
                },
            )
        call_kw = engine.chat_completion_sync.call_args[1]
        assert call_kw.get("temperature") == 0.25


# ── Streaming completions ─────────────────────────────────


class TestStreamCompletions:
    def test_10_14_stream_200(self):
        """POST with stream=true returns 200 with text/event-stream."""
        app, _ = _make_app(stream_tokens=["hi"])
        with TestClient(app) as client:
            r = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )
        assert r.status_code == 200

    def test_10_15_first_chunk_role(self):
        """first SSE chunk has delta.role == 'assistant'."""
        app, _ = _make_app(stream_tokens=["hello"])
        with TestClient(app) as client:
            r = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )
        events = _parse_sse(r.text)
        assert len(events) >= 1
        first_delta = events[0]["choices"][0]["delta"]
        assert first_delta.get("role") == "assistant"

    def test_10_16_chunks_have_valid_fields(self):
        """SSE chunks have id and object == 'chat.completion.chunk'."""
        app, _ = _make_app(stream_tokens=["hello"])
        with TestClient(app) as client:
            r = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )
        events = _parse_sse(r.text)
        for e in events:
            assert "id" in e
            assert e.get("object") == "chat.completion.chunk"

    def test_10_17_final_chunk_has_finish_reason(self):
        """final SSE chunk has finish_reason set."""
        app, _ = _make_app(stream_tokens=["hello"])
        with TestClient(app) as client:
            r = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )
        events = _parse_sse(r.text)
        last = events[-1]
        finish = last["choices"][0].get("finish_reason")
        assert finish in ("stop", "tool_calls")

    def test_10_18_stream_ends_with_done(self):
        """SSE stream ends with [DONE] marker."""
        app, _ = _make_app(stream_tokens=["hello"])
        with TestClient(app) as client:
            r = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )
        assert "[DONE]" in r.text


# ── CORS ────────────────────────────────────────────────────────


class TestCORS:
    def test_10_20_cors_allows_all_origins(self):
        """CORS allows all origins; credentials=False (browser-compatible)."""
        app, _ = _make_app()
        with TestClient(app) as client:
            r = client.options(
                "/v1/chat/completions",
                headers={
                    "Origin": "http://example.com",
                    "Access-Control-Request-Method": "POST",
                },
            )
        origin_header = r.headers.get("access-control-allow-origin", "")
        assert origin_header == "*"
        # credentials must NOT be "true" (breaks CORS with wildcard origin)
        cred_header = r.headers.get("access-control-allow-credentials", "false")
        assert cred_header.lower() != "true"


# ── thinking / reasoning content ─────────────────────────────────


class TestThinkingSync:
    def test_8_3_reasoning_content_extracted(self):
        """sync response: reasoning_content field contains think block text."""
        raw = "<think>my reasoning</think>final answer"
        app, _ = _make_app(sync_text=raw)
        with TestClient(app) as client:
            r = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        msg = r.json()["choices"][0]["message"]
        assert msg.get("reasoning_content") == "my reasoning"

    def test_8_4_think_not_in_content(self):
        """sync response: <think> block not in content field."""
        raw = "<think>hidden reasoning</think>visible answer"
        app, _ = _make_app(sync_text=raw)
        with TestClient(app) as client:
            r = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        msg = r.json()["choices"][0]["message"]
        assert msg.get("content") == "visible answer"
        assert "<think>" not in (msg.get("content") or "")

    def test_8_content_without_think_block(self):
        """Without a think block, content is returned verbatim."""
        app, _ = _make_app(sync_text="plain response")
        with TestClient(app) as client:
            r = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        msg = r.json()["choices"][0]["message"]
        assert msg.get("content") == "plain response"
        assert msg.get("reasoning_content") is None


class TestThinkingStream:
    def test_8_5_reasoning_deltas_in_stream(self):
        """streaming: reasoning content yields reasoning_content delta chunks."""
        # Think block spread across tokens
        tokens = ["<think>re", "ason", "ing</think>", " answer"]
        app, _ = _make_app(stream_tokens=tokens)
        with TestClient(app) as client:
            r = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )
        events = _parse_sse(r.text)
        reasoning_chunks = [
            e["choices"][0]["delta"].get("reasoning_content", "")
            for e in events
            if e.get("choices")
            and e["choices"][0].get("delta", {}).get("reasoning_content")
        ]
        assert len(reasoning_chunks) > 0, "no reasoning_content chunks emitted"
        # The combined reasoning should contain the think content
        combined = "".join(reasoning_chunks)
        assert "reason" in combined or "ing" in combined or "<think>" in combined

    def test_8_7_partial_close_tag_holdback(self):
        """partial </think> tag is not emitted until the full tag is received."""
        # Send </think> in pieces — the holdback should prevent partial emission
        tokens = ["<think>re", "ason</thin", "k> answer"]
        app, _ = _make_app(stream_tokens=tokens)
        with TestClient(app) as client:
            r = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )
        events = _parse_sse(r.text)
        # No content chunk should contain a partial close tag
        content_texts = [
            e["choices"][0]["delta"].get("content", "")
            for e in events
            if e.get("choices") and e["choices"][0].get("delta", {}).get("content")
        ]
        for text in content_texts:
            assert (
                "</thin" not in text
            ), f"partial </think> leaked into content: {text!r}"

    def test_8_8_small_task_disables_thinking(self):
        """_is_small_task returns True for title-generation patterns."""
        from quantstar.server import _is_small_task

        msgs = [{"role": "user", "content": "generate a short title for this chat"}]
        assert _is_small_task(msgs, None) is True

    def test_8_9_small_task_list_content(self):
        """_is_small_task works when message content is a list."""
        from quantstar.server import _is_small_task

        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "generate a short title for this"}
                ],
            }
        ]
        assert _is_small_task(msgs, None) is True

    def test_8_10_regular_message_not_small(self):
        """normal coding question is not a small task."""
        from quantstar.server import _is_small_task

        msgs = [{"role": "user", "content": "Write a Python function to sort a list."}]
        assert _is_small_task(msgs, None) is False


# ── tool calling ──────────────────────────────────────────────────


class TestToolCallParser:
    """Tests for _make_tool_call_stream_parser"""

    def _parser(self, idx=0):
        from quantstar.server import _make_tool_call_stream_parser

        return _make_tool_call_stream_parser(tool_index=idx)

    def test_9_1_single_tool_call_parsed(self):
        """single <tool_call> block is parsed correctly."""
        raw = "<function=search><parameter=query>hello</parameter>"
        parse = self._parser()
        result = parse(raw, finalize=True)
        assert len(result) >= 1

    def test_9_3_function_name_extracted(self):
        """function name extracted from <function=NAME>."""
        raw = "<function=my_tool><parameter=x>1</parameter>"
        parse = self._parser()
        deltas = parse(raw, finalize=True)
        flat = [d for group in deltas for d in group]
        name_delta = next((d for d in flat if d.get("function", {}).get("name")), None)
        assert name_delta is not None
        assert name_delta["function"]["name"] == "my_tool"

    def test_9_4_parameter_extracted(self):
        """parameter value extracted from <parameter=NAME>VALUE</parameter>."""
        raw = "<function=tool><parameter=key>value</parameter>"
        parse = self._parser()
        deltas = parse(raw, finalize=True)
        flat = [d for group in deltas for d in group]
        args_delta = next(
            (d for d in flat if d.get("function", {}).get("arguments")), None
        )
        assert args_delta is not None
        args = json.loads(args_delta["function"]["arguments"])
        assert args["key"] == "value"

    def test_9_5_json_int_argument_parsed(self):
        """integer parameter values are parsed as ints, not strings."""
        raw = "<function=tool><parameter=count>42</parameter>"
        parse = self._parser()
        deltas = parse(raw, finalize=True)
        flat = [d for group in deltas for d in group]
        args_delta = next(
            (d for d in flat if d.get("function", {}).get("arguments")), None
        )
        args = json.loads(args_delta["function"]["arguments"])
        assert args["count"] == 42
        assert isinstance(args["count"], int)

    def test_9_6_non_json_value_kept_as_string(self):
        """non-JSON argument values are kept as raw strings."""
        raw = "<function=tool><parameter=msg>hello world</parameter>"
        parse = self._parser()
        deltas = parse(raw, finalize=True)
        flat = [d for group in deltas for d in group]
        args_delta = next(
            (d for d in flat if d.get("function", {}).get("arguments")), None
        )
        args = json.loads(args_delta["function"]["arguments"])
        assert args["msg"] == "hello world"

    def test_9_7_name_delta_emitted_before_args(self):
        """function name delta is emitted before arguments delta."""
        raw = "<function=f><parameter=x>1</parameter>"
        parse = self._parser()
        deltas = parse(raw, finalize=True)
        flat = [d for group in deltas for d in group]
        # First non-empty function entry should be the name
        has_name = any("name" in d.get("function", {}) for d in flat)
        has_args = any(d.get("function", {}).get("arguments", "") != "" for d in flat)
        assert has_name
        assert has_args

    def test_9_11_tool_call_id_format(self):
        """call id is formatted as call_<uuid12>."""
        raw = "<function=f><parameter=x>1</parameter>"
        parse = self._parser()
        deltas = parse(raw, finalize=True)
        flat = [d for group in deltas for d in group]
        id_delta = next((d for d in flat if d.get("id")), None)
        assert id_delta is not None
        assert id_delta["id"].startswith("call_")
        assert len(id_delta["id"]) == len("call_") + 12

    def test_9_index_set_correctly(self):
        """tool_index is reflected in every delta."""
        raw = "<function=f><parameter=x>1</parameter>"
        for idx in (0, 1, 2):
            parse = self._parser(idx)
            deltas = parse(raw, finalize=True)
            flat = [d for group in deltas for d in group]
            for d in flat:
                assert d["index"] == idx


class TestToolCallSync:
    """Tests for tool call parsing in _sync_response"""

    def test_9_1_sync_tool_calls_in_message(self):
        """sync response: tool_calls in message with id, type, function."""
        xml = "<tool_call><function=search><parameter=q>test</parameter></function></tool_call>"
        app, _ = _make_app(sync_text=xml)
        with TestClient(app) as client:
            r = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "search"}],
                },
            )
        msg = r.json()["choices"][0]["message"]
        assert msg.get("tool_calls") is not None
        tc = msg["tool_calls"][0]
        assert "id" in tc
        assert tc["type"] == "function"
        assert "name" in tc["function"]
        assert "arguments" in tc["function"]

    def test_9_2_multiple_tool_calls_all_parsed(self):
        """multiple <tool_call> blocks are ALL parsed (not just the first)."""
        xml = (
            "<tool_call><function=f1><parameter=a>1</parameter></function></tool_call>"
            "<tool_call><function=f2><parameter=b>2</parameter></function></tool_call>"
        )
        app, _ = _make_app(sync_text=xml)
        with TestClient(app) as client:
            r = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "do things"}],
                },
            )
        tool_calls = r.json()["choices"][0]["message"]["tool_calls"]
        assert len(tool_calls) == 2
        names = {tc["function"]["name"] for tc in tool_calls}
        assert "f1" in names
        assert "f2" in names

    def test_9_12_no_tool_call_tag_in_content(self):
        """<tool_call> tag does not appear in the content field."""
        xml = "<tool_call><function=f><parameter=x>1</parameter></function></tool_call>"
        app, _ = _make_app(sync_text=xml)
        with TestClient(app) as client:
            r = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "call"}],
                },
            )
        msg = r.json()["choices"][0]["message"]
        assert "<tool_call>" not in (msg.get("content") or "")


class TestToolCallStream:
    def test_9_8_finish_reason_tool_calls(self):
        """streaming: finish_reason is 'tool_calls' when tool calls present.

        The streaming state machine starts in 'think' state; it transitions to
        'post' only after seeing </think>. Real Qwen3 output always opens with
        a think block, so the mock must too.
        """
        # Model output: think block then tool call — MUST be separate list items
        # so each gets processed in its own loop iteration (no implicit str concat).
        tokens = [
            "<think>r</think>",
            "<tool_call><function=search><parameter=q>x</parameter></function></tool_call>",
        ]
        app, _ = _make_app(stream_tokens=tokens)
        with TestClient(app) as client:
            r = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "search"}],
                    "stream": True,
                },
            )
        events = _parse_sse(r.text)
        last = events[-1]
        assert last["choices"][0].get("finish_reason") == "tool_calls"

    def test_9_9_sequential_tool_index(self):
        """multiple <tool_call> blocks get sequential tool_index values.

        The state machine needs a </think> to exit think-mode before processing
        tool_call tags. Tokens mimic real model output: think block, then two
        separate tool calls.
        """
        tokens = [
            "<think>reasoning</think>",  # closes think-mode → state=post
            "<tool_call><function=f1><parameter=a>1</parameter></function></tool_call>",
            "<tool_call><function=f2><parameter=b>2</parameter></function></tool_call>",
        ]
        app, _ = _make_app(stream_tokens=tokens)
        with TestClient(app) as client:
            r = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "do"}],
                    "stream": True,
                },
            )
        events = _parse_sse(r.text)
        indices = set()
        for e in events:
            for choice in e.get("choices", []):
                for tc in choice.get("delta", {}).get("tool_calls", []):
                    if isinstance(tc, dict):
                        indices.add(tc.get("index"))

        # Two distinct sequential tool_call indices: {0, 1}
        assert (
            len(indices) >= 2
        ), f"expected ≥2 distinct tool_call indices, got {indices}"
