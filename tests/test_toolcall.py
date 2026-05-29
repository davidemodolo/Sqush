from __future__ import annotations

"""Pure-Python tests for tool-call parsing.

These need NO model and NO GPU — they exercise the logic that turns a Qwen/
Hermes-style ``<tool_call>{...}</tool_call>`` token stream into structured
OpenAI tool calls. This is the exact code path that was missing and caused
opencode to "generate a title then nothing": the agent only acts on structured
``tool_calls``, never on raw ``<tool_call>`` text.

Run with:  python -m pytest tests/test_toolcall.py -v
"""

import json

from quenstar.toolcall import (
    StreamingToolCallParser,
    split_content_and_tool_calls,
    _suffix_prefix_overlap,
)


def _feed_all(parser: StreamingToolCallParser, pieces):
    """Feed pieces one at a time; return (joined_content, all_tool_calls)."""
    content = ""
    calls = []
    for p in pieces:
        c, tcs = parser.feed(p)
        content += c
        calls.extend(tcs)
    c, tcs = parser.flush()
    content += c
    calls.extend(tcs)
    return content, calls


def _char_chunks(text):
    return list(text)


# ── plain text (the "title" path) ──────────────────────────────────


def test_plain_text_passes_through():
    content, calls = split_content_and_tool_calls("Hello, this is a normal answer.")
    assert content == "Hello, this is a normal answer."
    assert calls == []


def test_plain_text_char_by_char_passes_through():
    parser = StreamingToolCallParser()
    text = "A short streamed reply."
    content, calls = _feed_all(parser, _char_chunks(text))
    assert content == text
    assert calls == []


# ── single tool call ───────────────────────────────────────────────


SINGLE = '<tool_call>\n{"name": "bash", "arguments": {"command": "ls -la"}}\n</tool_call>'


def test_single_tool_call_whole_string():
    content, calls = split_content_and_tool_calls(SINGLE)
    assert content == ""
    assert len(calls) == 1
    tc = calls[0]
    assert tc["type"] == "function"
    assert tc["id"].startswith("call_")
    assert tc["function"]["name"] == "bash"
    assert json.loads(tc["function"]["arguments"]) == {"command": "ls -la"}
    # the exact bytes are preserved for replay
    assert tc["_raw"].startswith("<tool_call>")
    assert tc["_raw"].endswith("</tool_call>")


def test_single_tool_call_char_by_char():
    parser = StreamingToolCallParser()
    content, calls = _feed_all(parser, _char_chunks(SINGLE))
    assert content == ""
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "bash"
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "ls -la"}


def test_open_tag_split_across_chunk_boundary():
    # The "<tool_call>" tag itself is split mid-tag across feeds.
    pieces = ["<tool_", "call>", '{"name":"read","arguments":{"path":"/etc/hosts"}}', "</tool", "_call>"]
    parser = StreamingToolCallParser()
    content, calls = _feed_all(parser, pieces)
    assert content == ""
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "read"


def test_no_partial_tag_leaks_as_content():
    # While streaming char-by-char, no emitted content chunk may contain the
    # beginning of a tool-call tag — otherwise the agent would render "<tool_call".
    parser = StreamingToolCallParser()
    emitted = []
    for ch in _char_chunks("Sure!" + SINGLE):
        c, _ = parser.feed(ch)
        if c:
            emitted.append(c)
    tail, _ = parser.flush()
    if tail:
        emitted.append(tail)
    joined = "".join(emitted)
    assert joined == "Sure!"
    assert "<tool_call" not in joined


# ── content mixed with tool calls ──────────────────────────────────


def test_content_then_tool_call():
    text = "Let me check that for you.\n" + SINGLE
    content, calls = split_content_and_tool_calls(text)
    assert content == "Let me check that for you.\n"
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "bash"


def test_multiple_tool_calls():
    text = (
        '<tool_call>{"name":"a","arguments":{"x":1}}</tool_call>'
        "\n"
        '<tool_call>{"name":"b","arguments":{"y":2}}</tool_call>'
    )
    content, calls = split_content_and_tool_calls(text)
    assert [c["function"]["name"] for c in calls] == ["a", "b"]
    assert json.loads(calls[0]["function"]["arguments"]) == {"x": 1}
    assert json.loads(calls[1]["function"]["arguments"]) == {"y": 2}
    # ids are unique
    assert calls[0]["id"] != calls[1]["id"]


# ── reasoning / think blocks ───────────────────────────────────────


def test_think_block_is_not_a_tool_call():
    text = "<think>The user wants a listing. I should call bash.</think>\n" + SINGLE
    content, calls = split_content_and_tool_calls(text)
    assert "<think>" in content  # reasoning is passed through as content
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "bash"


def test_think_block_char_by_char():
    text = "<think>reasoning here</think>" + SINGLE
    parser = StreamingToolCallParser()
    content, calls = _feed_all(parser, _char_chunks(text))
    assert content == "<think>reasoning here</think>"
    assert len(calls) == 1


# ── false-positive guards ──────────────────────────────────────────


def test_code_with_angle_brackets_not_mistaken_for_tool_call():
    text = "Use std::vector<int> and Map<String, Object> in your code."
    parser = StreamingToolCallParser()
    content, calls = _feed_all(parser, _char_chunks(text))
    assert content == text
    assert calls == []


def test_lone_less_than_held_then_released():
    parser = StreamingToolCallParser()
    c1, _ = parser.feed("value < ")
    c2, _ = parser.feed("threshold")
    tail, _ = parser.flush()
    assert (c1 + c2 + tail) == "value < threshold"


# ── unterminated tool call (EOS before close tag) ──────────────────


def test_unterminated_tool_call_parsed_on_flush():
    # Model produced a valid JSON tool call body but hit EOS before </tool_call>.
    text = '<tool_call>{"name":"done","arguments":{}}'
    content, calls = split_content_and_tool_calls(text)
    assert content == ""
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "done"


def test_unterminated_garbage_is_dropped_not_leaked():
    # If what follows "<tool_call>" never becomes valid JSON, drop it rather than
    # leaking a broken "<tool_call>{..." fragment to the client.
    text = "<tool_call>{not valid json at all"
    content, calls = split_content_and_tool_calls(text)
    assert content == ""
    assert calls == []


# ── arguments encoding ─────────────────────────────────────────────


def test_arguments_always_json_string():
    text = '<tool_call>{"name":"f","arguments":{"nested":{"a":[1,2,3]},"s":"x"}}</tool_call>'
    _, calls = split_content_and_tool_calls(text)
    args = calls[0]["function"]["arguments"]
    assert isinstance(args, str)
    assert json.loads(args) == {"nested": {"a": [1, 2, 3]}, "s": "x"}


def test_empty_arguments():
    text = '<tool_call>{"name":"noop"}</tool_call>'
    _, calls = split_content_and_tool_calls(text)
    assert calls[0]["function"]["name"] == "noop"
    assert json.loads(calls[0]["function"]["arguments"]) == {}


def test_unicode_arguments_preserved():
    text = '<tool_call>{"name":"echo","arguments":{"msg":"caffè è bòno 日本語"}}</tool_call>'
    _, calls = split_content_and_tool_calls(text)
    assert json.loads(calls[0]["function"]["arguments"]) == {"msg": "caffè è bòno 日本語"}


# ── streaming vs whole-string equivalence ──────────────────────────


def test_streaming_matches_whole_string():
    text = (
        "Thinking about it.\n"
        '<tool_call>{"name":"grep","arguments":{"pattern":"TODO","path":"."}}</tool_call>'
        " done"
    )
    whole_content, whole_calls = split_content_and_tool_calls(text)

    parser = StreamingToolCallParser()
    stream_content, stream_calls = _feed_all(parser, _char_chunks(text))

    assert whole_content == stream_content
    assert [c["function"]["name"] for c in whole_calls] == [
        c["function"]["name"] for c in stream_calls
    ]
    assert [c["function"]["arguments"] for c in whole_calls] == [
        c["function"]["arguments"] for c in stream_calls
    ]


# ── overlap helper ─────────────────────────────────────────────────


def test_suffix_prefix_overlap():
    m = "<tool_call>"
    assert _suffix_prefix_overlap("abc<", m) == 1
    assert _suffix_prefix_overlap("abc<tool", m) == 5  # "<tool" == marker[:5]
    assert _suffix_prefix_overlap("abc<tool_call", m) == 10  # capped at len-1
    assert _suffix_prefix_overlap("hello", m) == 0
    assert _suffix_prefix_overlap("x<d", m) == 0
