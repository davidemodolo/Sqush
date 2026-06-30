"""Tests for CLI logic.

Covers: _trim_history (12.7-12.8) and run_cli behavior with mocked I/O.
Interactive-session tests (12.1-12.6) require a real terminal and are skipped.
"""
from __future__ import annotations


from quantstar.cli import _MAX_HISTORY, _trim_history


# ── history trimming ──────────────────────────────────────────

class TestTrimHistory:
    def _msgs(self, n: int, role: str = "user") -> list[dict]:
        return [{"role": role, "content": f"msg {i}"} for i in range(n)]

    def test_short_history_unchanged(self):
        """History shorter than _MAX_HISTORY is returned unchanged."""
        msgs = self._msgs(5, "user") + self._msgs(5, "assistant")
        result = _trim_history(msgs)
        assert len(result) == 10

    def test_12_8_long_history_trimmed_to_max(self):
        """history exceeding _MAX_HISTORY is trimmed to _MAX_HISTORY non-system messages."""
        msgs = self._msgs(_MAX_HISTORY + 10)  # all user messages
        result = _trim_history(msgs)
        non_system = [m for m in result if m["role"] != "system"]
        assert len(non_system) == _MAX_HISTORY

    def test_12_8_most_recent_messages_kept(self):
        """the MOST RECENT messages are kept when trimming."""
        msgs = [{"role": "user", "content": str(i)} for i in range(_MAX_HISTORY + 5)]
        result = _trim_history(msgs)
        non_system = [m for m in result if m["role"] != "system"]
        # Last _MAX_HISTORY messages should be retained
        assert non_system[-1]["content"] == str(_MAX_HISTORY + 4)
        assert non_system[0]["content"] == str(5)  # oldest retained

    def test_system_message_always_kept(self):
        """System message is preserved even when non-system count exceeds max."""
        system = [{"role": "system", "content": "You are helpful."}]
        non_system = self._msgs(_MAX_HISTORY + 5)
        msgs = system + non_system
        result = _trim_history(msgs)
        # System message at front
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are helpful."

    def test_system_message_not_counted_toward_limit(self):
        """System message does not count toward _MAX_HISTORY."""
        system = [{"role": "system", "content": "sys"}]
        non_system = self._msgs(_MAX_HISTORY)
        msgs = system + non_system
        result = _trim_history(msgs)
        non_sys = [m for m in result if m["role"] != "system"]
        assert len(non_sys) == _MAX_HISTORY

    def test_multiple_system_messages_all_kept(self):
        """All system messages are preserved (not counted toward trim limit)."""
        sys1 = {"role": "system", "content": "sys1"}
        sys2 = {"role": "system", "content": "sys2"}
        non_system = self._msgs(3)
        result = _trim_history([sys1, non_system[0], sys2, non_system[1], non_system[2]])
        sys_msgs = [m for m in result if m["role"] == "system"]
        assert len(sys_msgs) == 2

    def test_empty_messages_returns_empty(self):
        """Empty history returns empty list."""
        assert _trim_history([]) == []

    def test_only_system_message_unchanged(self):
        """Only-system history is returned unchanged."""
        msgs = [{"role": "system", "content": "sys"}]
        assert _trim_history(msgs) == msgs


# ── think block stripping ────────────────────────────────────

class TestThinkStripping:
    """Verify that the CLI strips <think>...</think> before displaying output."""

    def _run_cli_one_turn(self, raw_response: str, user_input: str = "hi") -> str:
        """Run one CLI turn with a mocked engine and console; return printed output."""
        from unittest import mock
        from quantstar.engine import InferenceEngine
        from quantstar.config import QuantStarConfig

        captured = []

        class FakeConsole:
            def print(self, text="", **kw):
                captured.append(str(text))

            def input(self, prompt=""):
                raise EOFError  # exit after first turn

        class FakeConsoleFirstTurn:
            _calls = 0

            def print(self, text="", **kw):
                captured.append(str(text))

            def input(self, prompt=""):
                FakeConsoleFirstTurn._calls += 1
                if FakeConsoleFirstTurn._calls == 1:
                    return user_input
                raise EOFError

        engine = mock.MagicMock(spec=InferenceEngine)
        engine.get_vram_info.return_value = {"cuda_available": False}
        engine.chat_completion_stream.side_effect = (
            lambda *a, **kw: iter([raw_response])
        )
        config = QuantStarConfig()

        console = FakeConsoleFirstTurn()

        # Console is imported locally inside run_cli, so patch at rich.console
        with mock.patch("rich.console.Console", return_value=console):
            try:
                from quantstar.cli import run_cli
                run_cli(engine, config)
            except EOFError:
                pass

        return "\n".join(captured)

    def test_12_7_think_block_stripped_from_display(self):
        """<think>...</think> block is stripped before printing."""
        output = self._run_cli_one_turn("<think>hidden</think>visible answer")
        # The printed output must contain the visible part but NOT the think content
        assert "visible answer" in output
        assert "hidden" not in output

    def test_12_7_no_think_block_displayed_verbatim(self):
        """when there is no think block, response is displayed as-is."""
        output = self._run_cli_one_turn("plain answer")
        assert "plain answer" in output
