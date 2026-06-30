"""Tests for __main__ commands.

Covers:
  13.4 — init writes provider config to ~/.config/opencode/opencode.json
  13.5 — init does not modify unrelated existing provider entries
  15.1 — correct provider structure in opencode config
"""
from __future__ import annotations

import json
import os
import tempfile
from unittest import mock



def _make_config(host="127.0.0.1", port=9898, max_context=262144, max_new_tokens=65536):
    from quantstar.config import QuantStarConfig
    cfg = QuantStarConfig()
    cfg.server.host = host
    cfg.server.port = port
    cfg.inference.max_context = max_context
    cfg.inference.max_new_tokens = max_new_tokens
    return cfg


class TestInitOpencode:
    """_init_opencode writes correct config."""

    def _run_init(self, config, existing_cfg: dict | None = None) -> dict:
        """Run _init_opencode with a temp file; return the resulting JSON."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            if existing_cfg is not None:
                json.dump(existing_cfg, f)
            tmp_path = f.name

        try:
            if existing_cfg is None:
                os.unlink(tmp_path)  # simulate file not existing

            from quantstar.__main__ import _init_opencode
            with mock.patch("quantstar.__main__._opencode_config_path", return_value=tmp_path):
                _init_opencode(config)

            with open(tmp_path) as f:
                return json.load(f)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_13_4_creates_file_when_missing(self):
        """init creates config file when it doesn't exist."""
        cfg = _make_config()
        result = self._run_init(cfg, existing_cfg=None)
        assert "provider" in result
        assert "quantstar" in result["provider"]

    def test_15_1_correct_base_url(self):
        """base URL matches configured host:port."""
        cfg = _make_config(host="127.0.0.1", port=9898)
        result = self._run_init(cfg)
        base_url = result["provider"]["quantstar"]["options"]["baseURL"]
        assert "127.0.0.1" in base_url
        assert "9898" in base_url

    def test_15_1_provider_api_key(self):
        """apiKey is set to "local"."""
        cfg = _make_config()
        result = self._run_init(cfg)
        assert result["provider"]["quantstar"]["options"]["apiKey"] == "local"

    def test_15_1_model_has_reasoning_and_tools(self):
        """model entry declares reasoning=True and tools=True."""
        cfg = _make_config()
        result = self._run_init(cfg)
        models = result["provider"]["quantstar"]["models"]
        assert len(models) >= 1
        model = next(iter(models.values()))
        assert model.get("reasoning") is True
        assert model.get("tools") is True

    def test_15_1_model_has_input_output_modalities(self):
        """model declares text+image input and text output."""
        cfg = _make_config()
        result = self._run_init(cfg)
        models = result["provider"]["quantstar"]["models"]
        model = next(iter(models.values()))
        modalities = model.get("modalities", {})
        assert "text" in modalities.get("input", [])
        assert "image" in modalities.get("input", [])
        assert "text" in modalities.get("output", [])

    def test_15_1_context_limit_matches_config(self):
        """context limit in model entry matches inference.max_context."""
        cfg = _make_config(max_context=32768)
        result = self._run_init(cfg)
        models = result["provider"]["quantstar"]["models"]
        model = next(iter(models.values()))
        assert model["limit"]["context"] == 32768

    def test_13_5_does_not_modify_other_providers(self):
        """init leaves unrelated provider entries unchanged."""
        existing = {
            "$schema": "https://opencode.ai/config.json",
            "provider": {
                "openai": {
                    "name": "OpenAI",
                    "options": {"apiKey": "sk-existing"},
                    "models": {"gpt-4": {"name": "GPT-4"}},
                }
            },
        }
        cfg = _make_config()
        result = self._run_init(cfg, existing_cfg=existing)

        # Existing provider is untouched
        assert "openai" in result["provider"]
        openai_cfg = result["provider"]["openai"]
        assert openai_cfg["options"]["apiKey"] == "sk-existing"
        assert "gpt-4" in openai_cfg["models"]

    def test_13_5_overwrites_quantstar_only(self):
        """re-running init updates only the quantstar provider."""
        existing = {
            "provider": {
                "anthropic": {"name": "Anthropic", "options": {"apiKey": "key"}},
                "quantstar": {"name": "OLD", "options": {"baseURL": "http://old"}},
            }
        }
        cfg = _make_config(port=7777)
        result = self._run_init(cfg, existing_cfg=existing)

        # anthropic untouched
        assert result["provider"]["anthropic"]["options"]["apiKey"] == "key"
        # quantstar updated
        assert "7777" in result["provider"]["quantstar"]["options"]["baseURL"]

    def test_schema_key_set(self):
        """Config file gets $schema key pointing to opencode schema."""
        cfg = _make_config()
        result = self._run_init(cfg)
        assert "$schema" in result
        assert "opencode" in result["$schema"]

    def test_agent_entry_written(self):
        """_init_opencode also writes an agent entry for quantstar."""
        cfg = _make_config()
        result = self._run_init(cfg)
        assert "agent" in result
        assert "quantstar" in result["agent"]


class TestOpenCodeConfigPath:
    def test_path_in_xdg_config(self):
        """config path is under ~/.config/opencode/."""
        from quantstar.__main__ import _opencode_config_path
        path = _opencode_config_path()
        assert "opencode" in path
        assert path.endswith(".json")
