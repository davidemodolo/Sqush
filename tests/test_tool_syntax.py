from __future__ import annotations

"""Tests for the DS4-style "greedy only on tool-call syntax" decision logic.

`ToolSyntaxTracker` decides, per character, whether the *next* token should be
forced greedy (tool-call structure / tags / keys / the tool name) or sampled
normally (argument string values, normal content). This is pure stdlib and needs
no model — exactly the logic we can't otherwise test on a machine without a GPU.

`ToolSyntaxGreedyProcessor`'s decode/advance is also tested with a fake llm.

Run with:  python -m pytest tests/test_tool_syntax.py -v
"""

from quenstar.toolcall import ToolSyntaxTracker, ToolSyntaxGreedyProcessor


def classify(s: str) -> list[tuple[str, bool]]:
    """For each char, the greedy decision that applies when generating it
    (i.e. based on everything before it)."""
    t = ToolSyntaxTracker()
    out = []
    for ch in s:
        out.append((ch, t.in_tool_syntax()))
        t.feed(ch)
    return out


def greedy_for_span(s: str, sub: str) -> list[bool]:
    """The greedy decisions for the characters of the first occurrence of `sub`."""
    decisions = classify(s)
    start = s.index(sub)
    return [g for _, g in decisions[start:start + len(sub)]]


CALL = '<tool_call>{"name": "bash", "arguments": {"command": "ls -la /tmp"}}</tool_call>'


# ── outside a tool call: never greedy ──────────────────────────────


def test_plain_content_never_greedy():
    for _, g in classify("Here is a normal answer with no tools."):
        assert g is False


def test_think_block_never_greedy():
    s = "<think>I should call bash to list files.</think>"
    assert all(g is False for _, g in classify(s))


def test_text_before_tool_call_not_greedy():
    # everything in "Sure! " before the tag is sampled normally
    assert greedy_for_span("Sure! " + CALL, "Sure! ") == [False] * len("Sure! ")


# ── inside a tool call ─────────────────────────────────────────────


def test_tool_name_value_is_greedy():
    # the tool NAME must be deterministic even though it's a JSON string value
    assert all(greedy_for_span(CALL, "bash"))


def test_json_keys_are_greedy():
    assert all(greedy_for_span(CALL, "name"))
    assert all(greedy_for_span(CALL, "arguments"))
    assert all(greedy_for_span(CALL, "command"))


def test_structural_chars_are_greedy():
    decisions = dict(classify(CALL))  # last decision per distinct char is fine here
    # sample a few structural characters
    for _, g in classify(CALL):
        pass
    # braces / colon / comma always greedy inside the call
    full = classify(CALL)
    for ch, g in full:
        if ch in "{}[]:,":
            assert g is True, f"structural char {ch!r} should be greedy"


def test_argument_value_is_sampled_normally():
    # the payload "ls -la /tmp" must NOT be greedy
    assert greedy_for_span(CALL, "ls -la /tmp") == [False] * len("ls -la /tmp")


def test_open_tag_commits_to_greedy():
    # The tag only commits to greedy once it is unambiguously "<tool_call>" (>= 3
    # chars, "<to") — so it can't be confused with "<think>".
    decisions = classify(CALL)
    tag = "<tool_call>"
    start = CALL.index(tag)
    span = [g for _, g in decisions[start:start + len(tag)]]
    assert span[:3] == [False, False, False]   # "<", "t", "o" decided before "<to" match
    assert all(span[3:])                        # rest of the tag committed greedily


def test_after_close_tag_back_to_normal():
    s = CALL + "All done!"
    assert greedy_for_span(s, "All done!") == [False] * len("All done!")


# ── trickier payloads ──────────────────────────────────────────────


def test_braces_inside_string_value_do_not_break_structure():
    # code payload containing JSON-significant chars must stay non-greedy and must
    # not pop the structure stack
    payload = 'def f(): return {"a": 1}'
    s = '<tool_call>{"name":"write","arguments":{"code":"' + payload.replace('"', '\\"') + '"}}</tool_call>'
    decisions = classify(s)
    # the 'def f()' part of the payload is inside the value string -> normal
    assert greedy_for_span(s, "def f()") == [False] * len("def f()")
    # and we still detect the closing of the tool call afterwards
    t = ToolSyntaxTracker()
    t.feed(s)
    assert t.in_tool_syntax() is False  # back outside the tool call


def test_escaped_quote_in_value_stays_in_payload():
    s = '<tool_call>{"name":"echo","arguments":{"msg":"say \\"hi\\" now"}}</tool_call>'
    # the text between the escaped quotes is still payload (normal sampling)
    assert greedy_for_span(s, "say ") == [False] * len("say ")
    assert greedy_for_span(s, " now") == [False] * len(" now")


def test_nested_object_argument_values_sampled_normally():
    s = '<tool_call>{"name":"x","arguments":{"o":{"deep":"value-here"}}}</tool_call>'
    assert greedy_for_span(s, "value-here") == [False] * len("value-here")
    # the nested key is still greedy
    assert all(greedy_for_span(s, "deep"))


def test_array_argument_string_values_sampled_normally():
    s = '<tool_call>{"name":"x","arguments":{"items":["alpha","beta"]}}</tool_call>'
    assert greedy_for_span(s, "alpha") == [False] * len("alpha")
    assert greedy_for_span(s, "beta") == [False] * len("beta")


def test_empty_arguments_all_greedy():
    s = '<tool_call>{"name":"noop","arguments":{}}</tool_call>'
    # no payload region at all -> the whole JSON body is greedy
    body = '{"name":"noop","arguments":{}}'
    assert all(greedy_for_span(s, body))


def test_multiple_tool_calls_reset_state():
    s = CALL + "\n" + '<tool_call>{"name":"read","arguments":{"path":"/etc/hosts"}}</tool_call>'
    assert greedy_for_span(s, "/etc/hosts") == [False] * len("/etc/hosts")
    assert all(greedy_for_span(s, "read"))


def test_streaming_in_fragments_matches_charwise():
    # feeding arbitrary fragments must give the same final state as char-by-char
    t1 = ToolSyntaxTracker()
    for ch in CALL:
        t1.feed(ch)
    t2 = ToolSyntaxTracker()
    for frag in [CALL[i:i + 7] for i in range(0, len(CALL), 7)]:
        t2.feed(frag)
    assert t1.in_tool_syntax() == t2.in_tool_syntax() is False


# ── processor decode/advance (fake llm, no numpy needed) ───────────


class FakeLlm:
    """Detokenizes by treating each token id as the index into a char list."""

    def __init__(self, chars):
        self._chars = chars

    def detokenize(self, tokens, prev_tokens=None, special=False):
        return "".join(self._chars[t] for t in tokens).encode("utf-8")


def test_processor_skips_prompt_then_tracks_generation():
    # prompt = 3 tokens (must be ignored), then generate "<tool_call>{..."
    chars = list("PROMPT") + list('<tool_call>{"name":"a"}')
    # token ids are indices into chars; prompt is ids 0..5, generation 6..end
    fake = FakeLlm(chars)
    proc = ToolSyntaxGreedyProcessor(fake)

    prompt_ids = [0, 1, 2, 3, 4, 5]  # "PROMPT"
    # First call: only the prompt is present -> recorded & skipped, tracker empty
    proc._advance(prompt_ids)
    assert proc.force_greedy() is False

    # Now generation arrives token by token (ids continue from 6)
    gen_ids = list(range(6, len(chars)))
    seq = list(prompt_ids)
    for tid in gen_ids:
        seq.append(tid)
        proc._advance(seq)
    # after generating up to '<tool_call>{"name":"a"}' we are inside the call
    # (the '}' closes the object but not the <tool_call> tag) -> still tool syntax
    assert proc.force_greedy() is True


def test_processor_decode_window_does_not_crash_without_prev_support():
    class NoPrevLlm:
        def __init__(self, chars):
            self._chars = chars

        def detokenize(self, tokens, prev_tokens=None, special=False):
            if prev_tokens is not None:
                raise TypeError("prev_tokens unsupported")
            return "".join(self._chars[t] for t in tokens).encode("utf-8")

    chars = list("AB") + list('<tool_call>')
    proc = ToolSyntaxGreedyProcessor(NoPrevLlm(chars))
    proc._advance([0, 1])  # prompt
    seq = [0, 1]
    for tid in range(2, len(chars)):
        seq.append(tid)
        proc._advance(seq)
    # decoded "<tool_call>" -> open tag complete -> inside tool syntax
    assert proc.force_greedy() is True
