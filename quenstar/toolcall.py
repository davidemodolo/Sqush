from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Optional

_log = logging.getLogger(__name__)

TOOL_CALL_PATTERNS = [
    re.compile(r"<tool_call\s*>"),
    re.compile(r"<\|tool\u2581calls\u2581begin\|>"),
    re.compile(r"<\|startoftext\|>\s*<\|tool\u2581calls\u2581begin\|>"),
]

TOOL_CALL_END_PATTERNS = [
    re.compile(r"</tool_call\s*>"),
    re.compile(r"<\|tool▁calls▁end\|>"),
]


class ToolCallDetector:
    def __init__(self):
        self._in_tool_call: bool = False
        self._tool_buffer: str = ""

    def feed(self, text: str) -> tuple[bool, list[dict[str, Any]]]:
        self._tool_buffer += text

        if not self._in_tool_call:
            for pattern in TOOL_CALL_PATTERNS:
                if pattern.search(self._tool_buffer):
                    self._in_tool_call = True
                    _log.debug("Tool call detected in output stream")
                    break

        if self._in_tool_call:
            for pattern in TOOL_CALL_END_PATTERNS:
                if pattern.search(self._tool_buffer):
                    self._in_tool_call = False
                    _log.debug("Tool call end detected")
                    break

        new_tool_calls = self._extract_tool_calls()
        return self._in_tool_call, new_tool_calls

    def is_in_tool_call(self) -> bool:
        return self._in_tool_call

    def _extract_tool_calls(self) -> list[dict[str, Any]]:
        result = []
        text = self._tool_buffer

        for match in re.finditer(
            r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL
        ):
            inner = match.group(1).strip()
            try:
                data = json.loads(inner)
                if "name" in data:
                    tool_call = {
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": data["name"],
                            "arguments": json.dumps(data.get("arguments", {}), ensure_ascii=False),
                        },
                    }
                    result.append(tool_call)
            except json.JSONDecodeError:
                _log.debug("Incomplete tool call JSON, waiting for more tokens")

        for match in re.finditer(
            r'<\|tool\u2581calls\u2581begin\|>(.*?)<\|tool\u2581calls\u2581end\|>', text, re.DOTALL
        ):
            inner = match.group(1).strip()
            parsed = self._parse_qwen_tool_block(inner)
            if parsed:
                result.extend(parsed)

        return result

    def _parse_qwen_tool_block(self, inner: str) -> list[dict[str, Any]]:
        try:
            data = json.loads(inner)
            if isinstance(data, list):
                calls = []
                for item in data:
                    fn = item.get("function", item)
                    calls.append({
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": fn.get("name", ""),
                            "arguments": json.dumps(fn.get("arguments", {}), ensure_ascii=False),
                        },
                    })
                return calls
            elif isinstance(data, dict):
                return [{
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": data.get("name", ""),
                        "arguments": json.dumps(data.get("arguments", {}), ensure_ascii=False),
                    },
                }]
        except json.JSONDecodeError:
            pass
        return []

    def reset(self):
        self._in_tool_call = False
        self._tool_buffer = ""


class ToolCallRegistry:
    def __init__(self, max_entries: int = 100000):
        self._registry: dict[str, str] = {}
        self._max_entries = max_entries

    def register(self, tool_id: str, raw_text: str):
        self._registry[tool_id] = raw_text
        self._trim()

    def lookup(self, tool_id: str) -> Optional[str]:
        return self._registry.get(tool_id)

    def _trim(self):
        while len(self._registry) > self._max_entries:
            self._registry.pop(next(iter(self._registry)))


def replay_tool_calls(
    messages: list[dict[str, Any]],
    registry: ToolCallRegistry,
) -> list[dict[str, Any]]:
    if not registry:
        return messages

    result = []
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            tool_calls = msg["tool_calls"]
            if tool_calls:
                first_id = tool_calls[0].get("id", "")
                raw = registry.lookup(first_id)
                if raw:
                    result.append({"role": "assistant", "content": raw})
                    continue
                canonical = canonicalize_tool_calls(tool_calls)
                if canonical:
                    result.append({"role": "assistant", "content": canonical})
                    continue
        result.append(msg)
    return result


def canonicalize_tool_calls(tool_calls: list[dict[str, Any]]) -> str:
    parts = []
    for tc in tool_calls:
        fn = tc.get("function", {})
        name = fn.get("name", "")
        try:
            args = json.loads(fn.get("arguments", "{}"))
        except (json.JSONDecodeError, TypeError):
            args = {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                args = {}
        canonical = json.dumps(
            {"name": name, "arguments": args},
            separators=(",", ":"),
            sort_keys=True,
            ensure_ascii=False,
        )
        parts.append(f"<tool_call>\n{canonical}\n</tool_call>")
    return "\n".join(parts)


class StreamingToolCallParser:
    """Incrementally separate assistant text from Hermes/Qwen-style
    ``<tool_call>{...}</tool_call>`` blocks in a streamed completion.

    The auto-detected llama-cpp-python chat formatter does **not** parse tool
    calls: it injects the tool schemas into the prompt and returns the model's
    raw ``<tool_call>...`` text as plain ``content``. Agents such as opencode
    only recognise a tool call from a *structured* ``choices[].delta.tool_calls``
    array, so the server must do the parsing itself (this is the core thing
    antirez's ds4 does that a naive passthrough does not).

    Usage::

        parser = StreamingToolCallParser()
        for piece in token_stream:
            content, tool_calls = parser.feed(piece)
            # emit `content` as a content delta, `tool_calls` as tool_call deltas
        content, tool_calls = parser.flush()  # at end of stream

    ``feed`` never leaks a partial open tag as content: any trailing text that
    could be the start of ``<tool_call>`` is held back until the next chunk
    disambiguates it. Each returned tool call is a fully-formed OpenAI tool call
    dict plus a private ``_raw`` key holding the exact bytes the model produced
    (used for byte-exact replay / KV-cache prefix matching).
    """

    OPEN = "<tool_call>"
    CLOSE = "</tool_call>"

    def __init__(self) -> None:
        self._buffer: str = ""
        self._inside: bool = False

    def feed(self, text: str) -> tuple[str, list[dict[str, Any]]]:
        if text:
            self._buffer += text

        out_content: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        while True:
            if not self._inside:
                idx = self._buffer.find(self.OPEN)
                if idx != -1:
                    if idx:
                        out_content.append(self._buffer[:idx])
                    self._buffer = self._buffer[idx + len(self.OPEN):]
                    self._inside = True
                    continue
                # No complete open tag. Emit everything except a trailing
                # fragment that could still grow into "<tool_call>".
                hold = _suffix_prefix_overlap(self._buffer, self.OPEN)
                emit_upto = len(self._buffer) - hold
                if emit_upto > 0:
                    out_content.append(self._buffer[:emit_upto])
                    self._buffer = self._buffer[emit_upto:]
                break
            else:
                jdx = self._buffer.find(self.CLOSE)
                if jdx != -1:
                    inner = self._buffer[:jdx]
                    raw = self.OPEN + inner + self.CLOSE
                    self._buffer = self._buffer[jdx + len(self.CLOSE):]
                    self._inside = False
                    tc = _build_tool_call(inner, raw)
                    if tc:
                        tool_calls.append(tc)
                    continue
                # Tool-call body still arriving; hold all of it.
                break

        return "".join(out_content), tool_calls

    def flush(self) -> tuple[str, list[dict[str, Any]]]:
        """Drain any buffered text at end of stream."""
        content = ""
        tool_calls: list[dict[str, Any]] = []
        if self._inside:
            # Unterminated tool call (model hit EOS before </tool_call>). Try to
            # parse what we have; if it's not valid, drop it rather than leaking
            # a broken "<tool_call>{..." fragment as assistant text.
            tc = _build_tool_call(self._buffer, self.OPEN + self._buffer + self.CLOSE)
            if tc:
                tool_calls.append(tc)
        else:
            content = self._buffer
        self._buffer = ""
        self._inside = False
        return content, tool_calls


def _suffix_prefix_overlap(buffer: str, marker: str) -> int:
    """Length of the longest suffix of ``buffer`` that is a prefix of ``marker``.

    A full occurrence of ``marker`` is handled by the caller (via ``find``); this
    only measures a *partial* trailing match that must be held back.
    """
    max_len = min(len(buffer), len(marker) - 1)
    for length in range(max_len, 0, -1):
        if buffer[-length:] == marker[:length]:
            return length
    return 0


def _build_tool_call(inner: str, raw: str) -> Optional[dict[str, Any]]:
    raw_json = inner.strip()
    if not raw_json:
        return None
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        _log.debug("Tool call block is not valid JSON, skipping: %r", raw_json[:120])
        return None
    if not isinstance(data, dict) or "name" not in data:
        return None
    args = data.get("arguments", {})
    if isinstance(args, str):
        arguments = args
    else:
        arguments = json.dumps(args, ensure_ascii=False)
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {
            "name": data["name"],
            "arguments": arguments,
        },
        "_raw": raw,
    }


class ToolSyntaxTracker:
    """Classify, character by character, whether the model is currently emitting
    tool-call *syntax* (which should be decoded greedily for parseability) or an
    argument *payload* string value (which should be sampled normally).

    This is the DS4 idea: force temperature 0 only on the structural tokens of a
    tool call — the ``<tool_call>``/``</tool_call>`` tags, the JSON braces, keys,
    punctuation and the tool *name* — while letting argument string values (code,
    file contents, commands) be sampled at the normal temperature. Greedy decoding
    over long payloads tends to repeat/degenerate; greedy over structure keeps the
    call parseable.

    Feed it the generated text incrementally; ``in_tool_syntax()`` returns whether
    the *next* token should be forced greedy, given everything seen so far.

    Rule: greedy iff we are inside a ``<tool_call>`` block AND not currently inside
    a string value that lives within the ``arguments`` object (plus greedy while a
    ``<tool_call>`` open tag is being committed). Everything outside a tool call —
    normal content, ``<think>`` reasoning — is sampled normally.
    """

    OPEN = "<tool_call>"
    CLOSE = "</tool_call>"
    # Only commit to greedy once an open tag is unambiguous, so we don't perturb
    # other `<...>` content. "<tool_call>" and "<think>" diverge at index 2, so a
    # match length >= 3 ("<to") can no longer be a `<think>` block.
    _OPEN_COMMIT_AT = 3

    def __init__(self) -> None:
        self._in_tool = False
        self._open_match = 0
        self._close_match = 0
        self._reset_json()

    def _reset_json(self) -> None:
        self._stack: list[str] = []
        self._in_string = False
        self._escape = False
        self._string_is_value = False
        self._capturing_key = False
        self._key_buf = ""
        self._last_key = ""
        self._expecting_value = False
        self._pending_args = False
        self._in_args = False
        self._args_depth = 0
        self._single_string_args = False

    def feed(self, text: str) -> None:
        for ch in text:
            self._feed_char(ch)

    def in_tool_syntax(self) -> bool:
        if self._in_tool:
            payload = self._in_string and self._string_is_value and self._in_args
            return not payload
        # Commit to a tool-call open tag once it is unambiguously started.
        return self._open_match >= self._OPEN_COMMIT_AT

    # ── internals ──────────────────────────────────────────────────

    def _feed_char(self, ch: str) -> None:
        if not self._in_tool:
            self._match_open(ch)
            return
        if self._in_string:
            self._feed_string_char(ch)
            return
        if self._match_close(ch):
            return
        self._feed_struct_char(ch)

    def _match_open(self, ch: str) -> None:
        o = self.OPEN
        if ch == o[self._open_match]:
            self._open_match += 1
            if self._open_match == len(o):
                self._in_tool = True
                self._open_match = 0
                self._reset_json()
        elif ch == o[0]:
            self._open_match = 1
        else:
            self._open_match = 0

    def _match_close(self, ch: str) -> bool:
        c = self.CLOSE
        if ch == c[self._close_match]:
            self._close_match += 1
            if self._close_match == len(c):
                self._in_tool = False
                self._close_match = 0
                self._reset_json()
            return True
        if ch == c[0]:
            self._close_match = 1
            return True
        self._close_match = 0
        return False

    def _feed_string_char(self, ch: str) -> None:
        if self._escape:
            self._escape = False
            if self._capturing_key:
                self._key_buf += ch
            return
        if ch == "\\":
            self._escape = True
            return
        if ch == '"':
            self._in_string = False
            if self._capturing_key:
                self._last_key = self._key_buf
                self._key_buf = ""
                self._capturing_key = False
            elif self._single_string_args:
                self._single_string_args = False
                self._in_args = False
            return
        if self._capturing_key:
            self._key_buf += ch

    def _feed_struct_char(self, ch: str) -> None:
        if ch == "{":
            self._stack.append("{")
            if self._pending_args:
                self._in_args = True
                self._args_depth = len(self._stack)
                self._pending_args = False
            self._expecting_value = False
        elif ch == "[":
            self._stack.append("[")
            if self._pending_args:
                self._in_args = True
                self._args_depth = len(self._stack)
                self._pending_args = False
            self._expecting_value = True
        elif ch in ("}", "]"):
            if self._stack:
                self._stack.pop()
            if self._in_args and len(self._stack) < self._args_depth:
                self._in_args = False
        elif ch == ":":
            self._expecting_value = True
            if len(self._stack) == 1 and self._last_key == "arguments":
                self._pending_args = True
        elif ch == ",":
            self._expecting_value = bool(self._stack) and self._stack[-1] == "["
        elif ch == '"':
            self._in_string = True
            self._escape = False
            self._string_is_value = self._expecting_value
            self._capturing_key = not self._expecting_value
            self._key_buf = ""
            if self._expecting_value and self._pending_args:
                self._single_string_args = True
                self._in_args = True
                self._pending_args = False
        elif self._pending_args and not ch.isspace():
            # arguments value turned out to be a bare scalar (not {/[/"): no
            # payload region — keep it greedy.
            self._pending_args = False


class ToolSyntaxGreedyProcessor:
    """llama-cpp-python ``LogitsProcessor`` that forces greedy (argmax) decoding
    while the model emits tool-call syntax, and leaves logits untouched inside
    argument string values and in normal content.

    Pass an instance wrapped in ``LogitsProcessorList`` to ``create_chat_completion``.
    It runs first in the sampler chain (before temp/top_k/top_p), so collapsing the
    distribution onto its argmax makes those tokens deterministic regardless of the
    configured temperature, while non-syntax tokens are sampled at that temperature.
    """

    # How many preceding token ids to pass to detokenize() for correct spacing of
    # the freshly generated suffix. A small window keeps incremental decode cheap.
    _PREV_WINDOW = 16

    def __init__(self, llm: Any) -> None:
        self._llm = llm
        self._tracker = ToolSyntaxTracker()
        self._consumed: Optional[int] = None  # input_ids already fed to tracker

    def _advance(self, input_ids) -> None:
        # input_ids may be a numpy array (live) or a plain list (tests). Only the
        # newly generated suffix and a small window are materialized, so this stays
        # O(1) amortized rather than O(n) per token.
        n = len(input_ids)
        if self._consumed is None:
            # First call: everything so far is the rendered prompt (which may itself
            # mention tool schemas / the literal "tool_call"). Skip it entirely so
            # the tracker only ever sees generated text.
            self._consumed = n
            return
        if n <= self._consumed:
            return
        new_ids = [int(t) for t in input_ids[self._consumed:n]]
        lo = max(0, self._consumed - self._PREV_WINDOW)
        prev = [int(t) for t in input_ids[lo:self._consumed]]
        frag = self._decode(new_ids, prev)
        self._consumed = n
        if frag:
            self._tracker.feed(frag)

    def _decode(self, new_ids: list[int], prev: list[int]) -> str:
        try:
            raw = self._llm.detokenize(list(new_ids), prev_tokens=list(prev))
        except Exception:
            try:
                raw = self._llm.detokenize(list(new_ids))
            except Exception:
                return ""
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")
        return str(raw)

    def force_greedy(self) -> bool:
        return self._tracker.in_tool_syntax()

    def __call__(self, input_ids, scores):
        import numpy as np

        self._advance(input_ids)
        if self.force_greedy():
            # Collapse the distribution onto its argmax: deterministic regardless
            # of the temperature/top_k/top_p applied later in the sampler chain.
            best = int(np.argmax(scores))
            scores[:] = -np.inf
            scores[best] = 0.0
        return scores


def split_content_and_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Non-streaming convenience: split a full completion into clean content and
    OpenAI tool calls. Equivalent to feeding the whole string then flushing."""
    parser = StreamingToolCallParser()
    content, calls = parser.feed(text)
    tail_content, tail_calls = parser.flush()
    return content + tail_content, calls + tail_calls
