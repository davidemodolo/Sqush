from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Optional

_log = logging.getLogger(__name__)

TOOL_CALL_PATTERNS = [
    re.compile(r"<tool_call\s*>"),
    re.compile(r"<\|tool▁calls▁begin\|>"),
    re.compile(r"<\|tool_calls_section\|>"),
    re.compile(r"<\|startoftext\|>\s*<\|tool_calls▁begin\|>"),
]

TOOL_CALL_END_PATTERNS = [
    re.compile(r"</tool_call\s*>"),
    re.compile(r"<\|tool▁calls▁end\|>"),
]


class ToolCallDetector:
    def __init__(self):
        self._in_tool_call: bool = False
        self._tool_call_depth: int = 0
        self._pending_tool_calls: list[dict[str, Any]] = []
        self._tool_buffer: str = ""

    def feed(self, text: str) -> tuple[bool, list[dict[str, Any]]]:
        self._tool_buffer += text

        if not self._in_tool_call:
            for pattern in TOOL_CALL_PATTERNS:
                if pattern.search(self._tool_buffer):
                    self._in_tool_call = True
                    self._tool_call_depth = 0
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
            r'<\|tool▁calls▁begin\|>(.*?)<\|tool▁calls▁end\|>', text, re.DOTALL
        ):
            inner = match.group(1).strip()
            parsed = self._parse_qwen_tool_block(inner)
            if parsed:
                result.append(parsed)

        return result

    def _parse_qwen_tool_block(self, inner: str) -> Optional[dict[str, Any]]:
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
                return calls[0] if calls else None
            elif isinstance(data, dict):
                return {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": data.get("name", ""),
                        "arguments": json.dumps(data.get("arguments", {}), ensure_ascii=False),
                    },
                }
        except json.JSONDecodeError:
            pass
        return None

    def reset(self):
        self._in_tool_call = False
        self._tool_call_depth = 0
        self._pending_tool_calls = []
        self._tool_buffer = ""


class ToolCallRegistry:
    def __init__(self, max_entries: int = 100000):
        self._registry: dict[str, str] = {}
        self._raw_content: dict[str, str] = {}
        self._max_entries = max_entries

    def register(self, tool_id: str, raw_text: str):
        self._registry[tool_id] = raw_text
        self._trim()

    def register_content(self, key: str, raw_content: str):
        self._raw_content[key] = raw_content
        self._trim()

    def lookup(self, tool_id: str) -> Optional[str]:
        return self._registry.get(tool_id)

    def lookup_content(self, key: str) -> Optional[str]:
        return self._raw_content.get(key)

    def _trim(self):
        while len(self._registry) + len(self._raw_content) > self._max_entries:
            if self._raw_content:
                self._raw_content.pop(next(iter(self._raw_content)))
            elif self._registry:
                self._registry.pop(next(iter(self._registry)))
            else:
                break


def extract_tool_calls_from_text(text: str) -> list[dict[str, Any]]:
    calls = []
    for match in re.finditer(
        r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL
    ):
        inner = match.group(1).strip()
        try:
            data = json.loads(inner)
            if "name" in data:
                calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": data["name"],
                        "arguments": json.dumps(data.get("arguments", {}), ensure_ascii=False),
                    },
                })
        except json.JSONDecodeError:
            pass
    return calls


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
