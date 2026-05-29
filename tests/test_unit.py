from __future__ import annotations

"""Pure unit tests for QuenStar — no GPU or model required."""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quenstar.toolcall import (
    ToolCallDetector,
    ToolCallRegistry,
    canonicalize_tool_calls,
    replay_tool_calls,
)
from quenstar.types import ChatCompletionRequest, KVCacheHeader, KVSaveReason


# ── ToolCallDetector ──────────────────────────────────────────────


class TestToolCallDetector:
    def test_init_state(self):
        d = ToolCallDetector()
        assert d.is_in_tool_call() is False

    def test_detect_start_pattern(self):
        d = ToolCallDetector()
        in_call, calls = d.feed("some text <tool_call>")
        assert in_call is True
        assert calls == []

    def test_detect_qwen_start_pattern(self):
        d = ToolCallDetector()
        in_call, _ = d.feed("analysis <|tool\u2581calls\u2581begin|>")
        assert in_call is True

    def test_detect_end_pattern(self):
        d = ToolCallDetector()
        d.feed("<tool_call>")
        in_call, _ = d.feed('{"name":"test"}</tool_call>')
        assert in_call is False

    def test_detect_qwen_end_pattern(self):
        d = ToolCallDetector()
        d.feed("<|tool_calls_begin|>")
        in_call, _ = d.feed('[{"name":"test"}]<|tool_calls_end|>')
        assert in_call is False

    def test_extract_single_tool_call(self):
        d = ToolCallDetector()
        _, calls = d.feed(
            '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
        )
        assert len(calls) == 1
        assert calls[0]["type"] == "function"
        assert calls[0]["function"]["name"] == "get_weather"
        args = json.loads(calls[0]["function"]["arguments"])
        assert args["city"] == "Paris"

    def test_extract_multiple_tool_calls(self):
        d = ToolCallDetector()
        _, calls = d.feed(
            '<tool_call>\n{"name": "a", "arguments": {}}\n</tool_call>\n'
            '<tool_call>\n{"name": "b", "arguments": {}}\n</tool_call>'
        )
        assert len(calls) == 2
        assert calls[0]["function"]["name"] == "a"
        assert calls[1]["function"]["name"] == "b"

    def test_incomplete_json_waiting_for_tokens(self):
        d = ToolCallDetector()
        _, calls = d.feed("<tool_call>\n{\n</tool_call>")
        assert calls == []

    def test_tool_call_with_newlines_in_json(self):
        d = ToolCallDetector()
        _, calls = d.feed(
            '<tool_call>\n{\n  "name": "run_code",\n  "arguments": {\n    "code": "print(1)"\n  }\n}\n</tool_call>'
        )
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "run_code"

    def test_qwen_tool_block_single_dict(self):
        d = ToolCallDetector()
        _, calls = d.feed(
            '<|tool\u2581calls\u2581begin|>\n{"name": "search", "arguments": {"q": "hello"}}\n<|tool\u2581calls\u2581end|>'
        )
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "search"

    def test_qwen_tool_block_list(self):
        d = ToolCallDetector()
        _, calls = d.feed(
            '<|tool\u2581calls\u2581begin|>\n[{"function": {"name": "f1", "arguments": {}}}, {"function": {"name": "f2", "arguments": {}}}]\n<|tool\u2581calls\u2581end|>'
        )
        assert len(calls) == 2
        assert calls[0]["function"]["name"] == "f1"
        assert calls[1]["function"]["name"] == "f2"

    def test_feed_returns_in_tool_call_state(self):
        d = ToolCallDetector()
        in_call, _ = d.feed("hello <tool_call>")
        assert in_call is True
        in_call, _ = d.feed('{"name":"x"}</tool_call>')
        assert in_call is False

    def test_reset_clears_state(self):
        d = ToolCallDetector()
        d.feed("<tool_call>")
        assert d.is_in_tool_call() is True
        d.reset()
        assert d.is_in_tool_call() is False
        _, calls = d.feed('<tool_call>\n{"name": "x", "arguments": {}}\n</tool_call>')
        assert len(calls) == 1

    def test_not_detect_partial_pattern(self):
        d = ToolCallDetector()
        in_call, _ = d.feed("some <tool text")
        assert in_call is False

    def test_no_false_positive_on_normal_text(self):
        d = ToolCallDetector()
        in_call, _ = d.feed("The weather is nice today. I think we should go out.")
        assert in_call is False
        assert d.is_in_tool_call() is False

    def test_tool_call_id_generated(self):
        d = ToolCallDetector()
        _, calls = d.feed(
            '<tool_call>\n{"name": "test", "arguments": {}}\n</tool_call>'
        )
        assert calls[0]["id"].startswith("call_")
        assert len(calls[0]["id"]) == 13  # "call_" + 8 hex chars

    def test_qwen_block_invalid_json_returns_empty(self):
        d = ToolCallDetector()
        _, calls = d.feed(
            '<|tool\u2581calls\u2581begin|>\nnot valid json{{{</|tool\u2581calls\u2581end|>'
        )
        assert calls == []


# ── ToolCallRegistry ─────────────────────────────────────────────


class TestToolCallRegistry:
    def test_register_and_lookup(self):
        r = ToolCallRegistry(max_entries=10)
        r.register("call_abc", "<tool_call>\nraw text\n</tool_call>")
        assert r.lookup("call_abc") == "<tool_call>\nraw text\n</tool_call>"

    def test_lookup_missing(self):
        r = ToolCallRegistry(max_entries=10)
        assert r.lookup("nonexistent") is None

    def test_lru_eviction(self):
        r = ToolCallRegistry(max_entries=3)
        r.register("a", "1")
        r.register("b", "2")
        r.register("c", "3")
        r.register("d", "4")
        present = sum(1 for k in ["a", "b", "c", "d"] if r.lookup(k) is not None)
        assert present == 3

    def test_lru_eviction_fifo_order(self):
        r = ToolCallRegistry(max_entries=3)
        r.register("a", "1")
        r.register("b", "2")
        r.register("c", "3")
        r.register("d", "4")
        r.register("e", "5")
        present = sum(1 for k in ["a", "b", "c", "d", "e"] if r.lookup(k) is not None)
        assert present == 3


# ── replay_tool_calls ───────────────────────────────────────────


class TestReplayToolCalls:
    def test_replay_exact_raw_text(self):
        r = ToolCallRegistry()
        r.register("call_123", "<tool_call>\n{\"name\":\"f\"}\n</tool_call>")

        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_123", "function": {"name": "f", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "call_123", "content": "result"},
        ]

        result = replay_tool_calls(messages, r)
        assert result[0] == messages[0]
        assert result[1] == {"role": "assistant", "content": "<tool_call>\n{\"name\":\"f\"}\n</tool_call>"}
        assert result[2] == messages[2]

    def test_replay_canonical_fallback(self):
        r = ToolCallRegistry()  # empty registry
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "tool_calls": [{"id": "call_xyz", "function": {"name": "search", "arguments": '{"query":"hello"}'}}],
            },
        ]

        result = replay_tool_calls(messages, r)
        assert result[0] == messages[0]
        # Should have canonicalized form
        assert '<tool_call>' in result[1]["content"]
        assert '"name":"search"' in result[1]["content"]
        assert '"query":"hello"' in result[1]["content"]

    def test_replay_no_tool_calls_passthrough(self):
        r = ToolCallRegistry()
        r.register("x", "y")
        messages = [{"role": "user", "content": "hi"}]
        result = replay_tool_calls(messages, r)
        assert result == messages

    def test_replay_empty_tool_calls_list(self):
        r = ToolCallRegistry()
        r.register("x", "y")
        messages = [
            {"role": "assistant", "tool_calls": []},
            {"role": "user", "content": "next"},
        ]
        result = replay_tool_calls(messages, r)
        assert result == messages


# ── canonicalize_tool_calls ──────────────────────────────────────


class TestCanonicalizeToolCalls:
    def test_sorts_keys(self):
        result = canonicalize_tool_calls([
            {
                "function": {
                    "name": "test",
                    "arguments": '{"z": 1, "a": 2}',
                }
            }
        ])
        # Keys should be sorted: "a" before "z"
        assert '"a":2' in result
        assert '"z":1' in result
        assert result.index('"a"') < result.index('"z"')

    def test_compact_no_spaces(self):
        result = canonicalize_tool_calls([
            {
                "function": {
                    "name": "f",
                    "arguments": '{"key": "value"}',
                }
            }
        ])
        # sort_keys=True: "arguments" sorts before "name"
        assert '{"arguments":{"key":"value"},"name":"f"}' in result

    def test_multiple_tool_calls(self):
        result = canonicalize_tool_calls([
            {"function": {"name": "f1", "arguments": "{}"}},
            {"function": {"name": "f2", "arguments": "{}"}},
        ])
        # Each call: "<tool_call>\n{json}\n</tool_call>", joined by "\n"
        assert result.count("<tool_call>") == 2
        assert result.count("</tool_call>") == 2
        assert '"name":"f1"' in result
        assert '"name":"f2"' in result

    def test_args_are_string(self):
        result = canonicalize_tool_calls([
            {
                "function": {
                    "name": "f",
                    "arguments": '{"x": 1}',
                }
            }
        ])
        # args as a string should be parsed and re-serialized
        assert '"x":1' in result

    def test_invalid_json_args_fallback(self):
        result = canonicalize_tool_calls([
            {
                "function": {
                    "name": "f",
                    "arguments": "not-json",
                }
            }
        ])
        # Should fall back to empty args
        assert '"arguments":{}' in result

    def test_missing_arguments_key(self):
        result = canonicalize_tool_calls([
            {"function": {"name": "f"}}
        ])
        assert '"arguments":{}' in result

    def test_surrounds_with_tool_call_tags(self):
        result = canonicalize_tool_calls([
            {"function": {"name": "f", "arguments": "{}"}}
        ])
        assert result.startswith("<tool_call>")
        assert result.endswith("</tool_call>")


# ── KVCacheHeader ───────────────────────────────────────────────


class TestKVCacheHeader:
    def test_pack_unpack_roundtrip(self):
        import struct
        import time

        h = KVCacheHeader(
            magic=b"QSTK",
            version=1,
            quant_bits=4,
            save_reason=KVSaveReason.COLD,
            flags=0,
            n_tokens=100,
            hit_count=5,
            context_size=262144,
            created_at=time.time(),
            last_used_at=time.time(),
            payload_bytes=4096,
        )
        packed = h.pack()
        expected_size = struct.calcsize(KVCacheHeader._STRUCT_FMT)
        assert len(packed) == expected_size

        unpacked = KVCacheHeader.unpack(packed)
        assert unpacked.magic == b"QSTK"
        assert unpacked.version == 1
        assert unpacked.quant_bits == 4
        assert unpacked.save_reason == KVSaveReason.COLD
        assert unpacked.n_tokens == 100
        assert unpacked.hit_count == 5
        assert unpacked.context_size == 262144
        assert unpacked.payload_bytes == 4096
        assert abs(unpacked.created_at - h.created_at) < 1.0
        assert abs(unpacked.last_used_at - h.last_used_at) < 1.0

    def test_unpack_rejects_partial_data(self):
        with pytest.raises(Exception):
            KVCacheHeader.unpack(b"short")


# ── ChatCompletionRequest ────────────────────────────────────────


class TestChatCompletionRequest:
    def test_from_dict_all_fields(self):
        data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "seed": 42,
            "stream": True,
            "tools": [{"type": "function", "function": {"name": "test"}}],
            "tool_choice": "auto",
            "stop": ["\n"],
            "enable_thinking": False,
        }
        req = ChatCompletionRequest.from_dict(data)
        assert req.model == "test-model"
        assert req.messages == [{"role": "user", "content": "hi"}]
        assert req.max_tokens == 100
        assert req.temperature == 0.7
        assert req.top_p == 0.9
        assert req.top_k == 50
        assert req.seed == 42
        assert req.stream is True
        assert len(req.tools) == 1
        assert req.tool_choice == "auto"
        assert req.stop == ["\n"]
        assert req.enable_thinking is False

    def test_from_dict_defaults(self):
        req = ChatCompletionRequest.from_dict({})
        assert req.model == ""
        assert req.messages == []
        assert req.max_tokens is None
        assert req.temperature is None
        assert req.stream is False
        assert req.tools is None
        assert req.enable_thinking is None

    def test_from_dict_enable_thinking_true(self):
        req = ChatCompletionRequest.from_dict({
            "messages": [{"role": "user", "content": "x"}],
            "enable_thinking": True,
        })
        assert req.enable_thinking is True


# ── _has_image_content ──────────────────────────────────────────


class TestImageDetection:
    def test_detects_image_url(self):
        from quenstar.server import _has_image_content

        msgs = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
                {"type": "text", "text": "describe this"},
            ],
        }]
        assert _has_image_content(msgs) is True

    def test_no_image_in_text_only(self):
        from quenstar.server import _has_image_content

        msgs = [{"role": "user", "content": "hello world"}]
        assert _has_image_content(msgs) is False

    def test_no_image_in_string_content(self):
        from quenstar.server import _has_image_content

        msgs = [{"role": "user", "content": "just text"}]
        assert _has_image_content(msgs) is False

    def test_empty_messages(self):
        from quenstar.server import _has_image_content

        assert _has_image_content([]) is False

    def test_multiple_messages_finds_image(self):
        from quenstar.server import _has_image_content

        msgs = [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": [
                {"type": "text", "text": "look at this"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}},
            ]},
        ]
        assert _has_image_content(msgs) is True


# ── Config loading ──────────────────────────────────────────────


class TestConfig:
    def test_default_config(self):
        from quenstar.config import QuenStarConfig

        c = QuenStarConfig()
        assert c.model.n_ctx == 65536
        assert c.model.n_gpu_layers == -1
        assert c.model.offload_kqv is False
        assert c.model.flash_attn is True
        assert c.server.host == "127.0.0.1"
        assert c.server.port == 8080
        assert c.sampling.default_temperature == 0.8
        assert c.generation.max_tokens == 32768
        assert c.tool_calling.enabled is True
        assert c.tool_calling.manual_token_loop is False

    def test_load_from_yaml_file(self):
        from quenstar.config import QuenStarConfig

        yaml_content = """
model:
  n_ctx: 4096
  n_gpu_layers: 20

server:
  port: 9000

sampling:
  default_temperature: 0.5
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            tmp_path = f.name

        try:
            c = QuenStarConfig.load(tmp_path)
            assert c.model.n_ctx == 4096
            assert c.model.n_gpu_layers == 20
            assert c.server.port == 9000
            assert c.sampling.default_temperature == 0.5
            # Unspecified fields keep defaults
            assert c.server.host == "127.0.0.1"
            assert c.model.offload_kqv is False
        finally:
            os.unlink(tmp_path)

    def test_env_overrides(self):
        from quenstar.config import QuenStarConfig

        yaml_content = """
model:
  n_ctx: 4096
server:
  port: 8080
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            tmp_path = f.name

        try:
            os.environ["QUENSTAR_N_CTX"] = "8192"
            os.environ["QUENSTAR_PORT"] = "9999"
            os.environ["QUENSTAR_LOG_LEVEL"] = "DEBUG"

            c = QuenStarConfig.load(tmp_path)
            assert c.model.n_ctx == 8192
            assert c.server.port == 9999
            assert c.logging.level == "DEBUG"
        finally:
            os.unlink(tmp_path)
            os.environ.pop("QUENSTAR_N_CTX", None)
            os.environ.pop("QUENSTAR_PORT", None)
            os.environ.pop("QUENSTAR_LOG_LEVEL", None)

    def test_bool_env_override(self):
        from quenstar.config import QuenStarConfig

        yaml_content = "model:\n  offload_kqv: false\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            tmp_path = f.name

        try:
            os.environ["QUENSTAR_OFFLOAD_KQV"] = "true"
            c = QuenStarConfig.load(tmp_path)
            assert c.model.offload_kqv is True

            os.environ["QUENSTAR_OFFLOAD_KQV"] = "0"
            c = QuenStarConfig.load(tmp_path)
            assert c.model.offload_kqv is False
        finally:
            os.unlink(tmp_path)
            os.environ.pop("QUENSTAR_OFFLOAD_KQV", None)

    def test_missing_config_file_uses_defaults(self):
        from quenstar.config import QuenStarConfig

        c = QuenStarConfig.load("/nonexistent/path.yaml")
        assert c.model.n_ctx == 65536  # default


# ── _has_llava ──────────────────────────────────────────────────


class TestHasLlava:
    def test_has_llava_imports(self):
        from quenstar.server import _has_llava

        result = _has_llava()
        assert isinstance(result, bool)


# ── KVCacheStore ─────────────────────────────────────────────────


class TestKVCacheStore:
    @pytest.fixture
    def store(self, tmp_path):
        from quenstar.config import KVCacheConfig

        cfg = KVCacheConfig(
            dir=str(tmp_path),
            space_mb=1,
            eviction_half_life_hours=6.0,
        )
        from quenstar.kvstore import KVCacheStore
        return KVCacheStore(config=cfg, model_id="test_model", n_ctx=512)

    def test_compute_key_deterministic(self, store):
        msgs = [{"role": "user", "content": "hello"}]
        k1 = store.compute_key(msgs)
        k2 = store.compute_key(msgs)
        assert k1 == k2
        assert len(k1) == 40  # SHA1 hex

    def test_compute_key_different_messages(self, store):
        k1 = store.compute_key([{"role": "user", "content": "a"}])
        k2 = store.compute_key([{"role": "user", "content": "b"}])
        assert k1 != k2

    def test_store_and_load(self, store):
        from quenstar.types import KVSaveReason

        msgs = [{"role": "user", "content": "test"}]
        key = store.compute_key(msgs)
        state = b"fake_llama_state_bytes"

        store.store(key, state, reason=KVSaveReason.COLD)
        result = store.load(key)
        assert result is not None
        loaded_key, loaded_state, header = result
        assert loaded_key == key
        assert loaded_state == state
        assert header.magic == b"QSTK"
        assert header.save_reason == KVSaveReason.COLD

    def test_load_missing_key(self, store):
        assert store.load("nonexistent_key") is None

    def test_load_wrong_context_size(self, store):
        from quenstar.types import KVSaveReason

        msgs = [{"role": "user", "content": "x"}]
        key = store.compute_key(msgs)
        store.store(key, b"data", reason=KVSaveReason.COLD)

        from quenstar.config import KVCacheConfig
        from quenstar.kvstore import KVCacheStore

        wrong_store = KVCacheStore(
            config=KVCacheConfig(dir=str(store._dir), space_mb=1),
            model_id="test",
            n_ctx=256,
        )
        assert wrong_store.load(key) is None

    def test_load_and_bump_updates_hit_count(self, store):
        from quenstar.types import KVSaveReason

        msgs = [{"role": "user", "content": "bump"}]
        key = store.compute_key(msgs)
        store.store(key, b"data", reason=KVSaveReason.COLD)

        result = store.load_and_bump(key)
        assert result is not None
        _, _, header = result
        assert header.hit_count == 1

    def test_list_files(self, store):
        from quenstar.types import KVSaveReason

        store.store("key_a", b"a", reason=KVSaveReason.COLD)
        store.store("key_b", b"b", reason=KVSaveReason.CONTINUED)

        files = store.list_files()
        assert len(files) == 2
        keys = {f["key"] for f in files}
        assert keys == {"key_a", "key_b"}

    def test_total_size_bytes(self, store):
        from quenstar.types import KVSaveReason
        store.store("key_x", b"hello", reason=KVSaveReason.COLD)
        size = store.total_size_bytes()
        assert size > 0

    def test_delete(self, store):
        from quenstar.types import KVSaveReason
        store.store("del_me", b"data", reason=KVSaveReason.COLD)
        assert store.delete("del_me") is True
        assert store.load("del_me") is None
        assert store.delete("del_me") is False

    def test_eviction_on_space_limit(self, store):
        from quenstar.types import KVSaveReason
        store.config.space_mb = 0.0001
        store.store("a", b"x" * 200, reason=KVSaveReason.COLD)
        store.store("b", b"x" * 200, reason=KVSaveReason.COLD)
        store.store("c", b"x" * 200, reason=KVSaveReason.COLD)
        files = store.list_files()
        assert len(files) <= 2


# ── SessionManager ───────────────────────────────────────────────


class TestSessionManager:
    @pytest.fixture
    def session(self, tmp_path):
        from quenstar.config import QuenStarConfig, KVCacheConfig

        cfg = QuenStarConfig()
        cfg.kv_cache = KVCacheConfig(dir=str(tmp_path), space_mb=1)
        cfg.model.n_ctx = 512

        from quenstar.kvstore import KVCacheStore
        from quenstar.session import SessionManager

        kvstore = KVCacheStore(config=cfg.kv_cache, model_id="test", n_ctx=cfg.model.n_ctx)

        class FakeEngine:
            def __init__(self):
                self._state = b"fake_pickle_state"
                self.reset_called = False

            def save_state(self):
                return self._state

            def load_state(self, data):
                self._state = data

            def reset_context(self):
                self.reset_called = True

        engine = FakeEngine()
        return SessionManager(engine, kvstore, cfg)

    def test_new_session_no_cache_resets_engine(self, session):
        msgs = [{"role": "user", "content": "hi"}]
        resumed = session.new_session(msgs)
        assert resumed is False
        assert session._engine.reset_called
        assert session.session_id is not None
        assert len(session.session_id) == 40

    def test_new_session_continues_on_prefix_match(self, session):
        msgs1 = [{"role": "user", "content": "hi"}]
        session.new_session(msgs1)
        session._engine.reset_called = False

        msgs2 = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        resumed = session.new_session(msgs2)
        assert resumed is False  # prefix match, no re-prefill
        assert not session._engine.reset_called

    def test_new_session_resumes_from_cache(self, session):
        from quenstar.types import KVSaveReason
        msgs = [{"role": "user", "content": "resume_test"}]
        key = session._kvstore.compute_key(msgs)
        session._kvstore.store(key, b"cached_state", reason=KVSaveReason.COLD)

        resumed = session.new_session(msgs)
        assert resumed is True
        assert session.is_resumed
        assert session._engine._state == b"cached_state"

    def test_save_persists_to_disk(self, session):
        msgs = [{"role": "user", "content": "save_test"}]
        session.new_session(msgs)
        session._engine._state = b"after_inference"
        session.save()

        result = session._kvstore.load(session.session_id)
        assert result is not None
        _, loaded_state, _ = result
        assert loaded_state == b"after_inference"

    def test_save_and_reset_clears_state(self, session):
        msgs = [{"role": "user", "content": "reset_test"}]
        session.new_session(msgs)
        session.save_and_reset()
        assert session.session_id is None
        assert session._current_messages == []

    def test_update_changes_current_key(self, session):
        msgs1 = [{"role": "user", "content": "a"}]
        session.new_session(msgs1)
        old_key = session.session_id

        msgs2 = [{"role": "user", "content": "b"}]
        session.update(msgs2)
        assert session.session_id != old_key

    def test_shares_prefix(self):
        from quenstar.session import SessionManager

        assert SessionManager._shares_prefix([], [{"role": "user"}]) is False
        assert SessionManager._shares_prefix([{"a": 1}], [{"a": 1}, {"b": 2}]) is True
        assert SessionManager._shares_prefix([{"a": 1}], [{"a": 2}]) is False
        assert SessionManager._shares_prefix([{"a": 1}], [{"a": 1}]) is True

    def test_list_sessions_returns_kv_files(self, session):
        msgs = [{"role": "user", "content": "list_test"}]
        session.new_session(msgs)
        session.save()

        sessions = session.list_sessions()
        assert len(sessions) >= 1
        assert any(s["key"] == session.session_id for s in sessions)

    def test_delete_session(self, session):
        msgs = [{"role": "user", "content": "delete_test"}]
        session.new_session(msgs)
        session.save()
        key = session.session_id

        assert session.delete_session(key) is True
        assert session._kvstore.load(key) is None
