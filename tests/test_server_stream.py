from __future__ import annotations

"""Server-level streaming tests with a mocked engine (no model / GPU needed).

These drive ``_stream_chat`` / ``_non_stream_chat`` with fake llama-cpp-python
output and assert the emitted SSE shape matches what opencode's AI SDK requires:

  * tool calls arrive as structured ``choices[].delta.tool_calls[]`` with
    ``index`` + ``id`` + ``function.name`` present in the first delta,
  * the literal ``<tool_call>`` text never leaks into a content delta,
  * the terminal chunk carries ``finish_reason: "tool_calls"``,
  * a ``data: [DONE]`` sentinel closes the stream.
"""

import asyncio
import json

from quenstar.server import _stream_chat, _non_stream_chat
from quenstar.toolcall import ToolCallRegistry
from quenstar.types import ChatCompletionRequest


class FakeEngine:
    model_id = "test-model"
    n_tokens = 42

    def __init__(self, chunks):
        self._chunks = chunks

    def chat_completion(self, request):
        for c in self._chunks:
            yield c


class FakeSession:
    def __init__(self):
        self.updated = None

    def update(self, messages):
        self.updated = messages

    def save_checkpoint(self):
        pass


def _drive(agen):
    events = []

    async def run():
        async for ev in agen:
            events.append(ev)

    asyncio.run(run())
    return events


def _parse(events):
    """Return (parsed_chunks, saw_done)."""
    chunks = []
    saw_done = False
    for ev in events:
        data = ev["data"]
        if data == "[DONE]":
            saw_done = True
            continue
        chunks.append(json.loads(data))
    return chunks, saw_done


def _deltas(chunks):
    return [c["choices"][0]["delta"] for c in chunks if c.get("choices")]


def _content(chunks):
    return "".join(d.get("content", "") or "" for d in _deltas(chunks))


def _tool_call_deltas(chunks):
    out = []
    for d in _deltas(chunks):
        out.extend(d.get("tool_calls", []) or [])
    return out


def _finish_reasons(chunks):
    return [c["choices"][0]["finish_reason"] for c in chunks
            if c.get("choices") and c["choices"][0].get("finish_reason")]


# ── streaming a tool call ──────────────────────────────────────────


def test_stream_emits_structured_tool_call():
    engine = FakeEngine([
        {"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "Let me look. "}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": '<tool_call>\n{"name":"bash","arguments":{"command":"ls"}}\n</tool_call>'}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ])
    session = FakeSession()
    registry = ToolCallRegistry()
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "list files"}],
        stream=True,
        tools=[{"type": "function", "function": {"name": "bash", "parameters": {}}}],
    )

    chunks, saw_done = _parse(_drive(_stream_chat(engine, session, request, registry)))

    # content streamed, but the raw tag never leaks
    assert _content(chunks) == "Let me look. "
    assert "<tool_call" not in _content(chunks)

    # exactly one structured tool call, fully formed in its (only) delta
    tcs = _tool_call_deltas(chunks)
    assert len(tcs) == 1
    tc = tcs[0]
    assert tc["index"] == 0
    assert tc["id"].startswith("call_")
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "bash"
    assert json.loads(tc["function"]["arguments"]) == {"command": "ls"}

    # finish reason promoted to tool_calls, stream terminated with [DONE]
    assert _finish_reasons(chunks)[-1] == "tool_calls"
    assert saw_done

    # session stored the assistant turn with the tool call
    assert session.updated is not None
    assistant = session.updated[-1]
    assert assistant["role"] == "assistant"
    assert assistant["tool_calls"][0]["function"]["name"] == "bash"
    # raw bytes registered for exact replay
    assert registry.lookup(tc["id"]) is not None


def test_stream_plain_text_title_path():
    engine = FakeEngine([
        {"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "Fix login bug"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ])
    session = FakeSession()
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "title please"}], stream=True,
    )

    chunks, saw_done = _parse(_drive(_stream_chat(engine, session, request, ToolCallRegistry())))

    assert _content(chunks) == "Fix login bug"
    assert _tool_call_deltas(chunks) == []
    assert _finish_reasons(chunks)[-1] == "stop"
    assert saw_done


def test_stream_tool_call_split_across_chunks():
    # the <tool_call> tag and JSON arrive fragmented across many deltas
    fragments = ["<tool", "_call>", '{"name":"read",', '"arguments":{"path":', '"/x"}}', "</tool_call>"]
    engine = FakeEngine(
        [{"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]}]
        + [{"choices": [{"delta": {"content": f}, "finish_reason": None}]} for f in fragments]
        + [{"choices": [{"delta": {}, "finish_reason": "stop"}]}]
    )
    chunks, saw_done = _parse(
        _drive(_stream_chat(engine, FakeSession(), ChatCompletionRequest(
            messages=[{"role": "user", "content": "read"}], stream=True,
            tools=[{"type": "function", "function": {"name": "read"}}],
        ), ToolCallRegistry()))
    )
    assert "<tool" not in _content(chunks)
    tcs = _tool_call_deltas(chunks)
    assert len(tcs) == 1
    assert tcs[0]["function"]["name"] == "read"
    assert json.loads(tcs[0]["function"]["arguments"]) == {"path": "/x"}
    assert _finish_reasons(chunks)[-1] == "tool_calls"
    assert saw_done


def test_stream_passes_through_native_tool_calls():
    # forced tool_choice path: llama-cpp already produced structured tool_calls
    engine = FakeEngine([
        {"choices": [{"delta": {"tool_calls": [
            {"id": "call_native", "type": "function",
             "function": {"name": "edit", "arguments": '{"file":"a.py"}'}}
        ]}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ])
    chunks, saw_done = _parse(
        _drive(_stream_chat(engine, FakeSession(), ChatCompletionRequest(
            messages=[{"role": "user", "content": "edit"}], stream=True,
            tools=[{"type": "function", "function": {"name": "edit"}}],
        ), ToolCallRegistry()))
    )
    tcs = _tool_call_deltas(chunks)
    assert len(tcs) == 1
    assert tcs[0]["index"] == 0
    assert tcs[0]["function"]["name"] == "edit"
    assert _finish_reasons(chunks)[-1] == "tool_calls"
    assert saw_done


# ── non-streaming ──────────────────────────────────────────────────


def test_non_stream_parses_tool_call():
    engine = FakeEngine([
        {"choices": [{"message": {"role": "assistant",
                                  "content": '<tool_call>{"name":"bash","arguments":{"command":"pwd"}}</tool_call>'},
                      "finish_reason": "stop"}]}
    ])
    session = FakeSession()
    registry = ToolCallRegistry()
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "where am i"}], stream=False,
        tools=[{"type": "function", "function": {"name": "bash"}}],
    )

    resp = asyncio.run(_non_stream_chat(engine, session, request, registry))
    choice = resp["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    tcs = choice["message"]["tool_calls"]
    assert len(tcs) == 1
    assert tcs[0]["function"]["name"] == "bash"
    assert json.loads(tcs[0]["function"]["arguments"]) == {"command": "pwd"}
    # the private _raw key must not leak into the API response
    assert "_raw" not in tcs[0]


def test_non_stream_plain_text():
    engine = FakeEngine([
        {"choices": [{"message": {"role": "assistant", "content": "Hello there."},
                      "finish_reason": "stop"}]}
    ])
    resp = asyncio.run(_non_stream_chat(engine, FakeSession(), ChatCompletionRequest(
        messages=[{"role": "user", "content": "hi"}], stream=False,
    ), ToolCallRegistry()))
    choice = resp["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["content"] == "Hello there."
    assert "tool_calls" not in choice["message"]
