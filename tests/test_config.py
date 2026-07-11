"""Tests for config loading.

Covers: load_config() from YAML files, env vars, and defaults.
"""
from __future__ import annotations

import os
import tempfile
import textwrap

import pytest


def _write_yaml(content: str) -> str:
    """Write a temp YAML file and return its path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    f.write(textwrap.dedent(content))
    f.flush()
    f.close()
    return f.name


class TestDefaultConfig:
    """default values used when config keys are missing."""

    def test_18_1_all_sections_present(self):
        """SqushConfig has model, quantization, inference, server, logging."""
        from sqush.config import SqushConfig
        cfg = SqushConfig()
        assert hasattr(cfg, "model")
        assert hasattr(cfg, "quantization")
        assert hasattr(cfg, "inference")
        assert hasattr(cfg, "server")
        assert hasattr(cfg, "logging")

    def test_default_weight_bits_is_4(self):
        from sqush.config import load_config
        cfg = load_config("nonexistent_path.yaml")
        assert cfg.quantization.weight_bits == 4

    def test_default_kv_cache_bits_is_4(self):
        from sqush.config import load_config
        cfg = load_config("nonexistent_path.yaml")
        assert cfg.quantization.kv_cache_bits == 4

    def test_default_host_is_loopback(self):
        """server binds to 127.0.0.1 by default, not 0.0.0.0."""
        from sqush.config import load_config
        cfg = load_config("nonexistent_path.yaml")
        assert cfg.server.host == "127.0.0.1"

    def test_default_log_level(self):
        from sqush.config import load_config
        cfg = load_config("nonexistent_path.yaml")
        assert cfg.logging.level.upper() == "INFO"

    def test_default_temperature(self):
        from sqush.config import load_config
        cfg = load_config("nonexistent_path.yaml")
        assert 0 < cfg.inference.temperature <= 1.0

    def test_default_torch_dtype_bfloat16(self):
        """torch_dtype defaults to 'bfloat16'."""
        from sqush.config import load_config
        cfg = load_config("nonexistent_path.yaml")
        assert cfg.model.torch_dtype == "bfloat16"


class TestYAMLConfig:
    """YAML values override defaults."""

    def test_18_1_model_repo_from_yaml(self):
        path = _write_yaml("""
            model:
              repo: MyOrg/MyModel
        """)
        try:
            from sqush.config import load_config
            cfg = load_config(path)
            assert cfg.model.repo == "MyOrg/MyModel"
        finally:
            os.unlink(path)

    def test_inference_temperature_from_yaml(self):
        path = _write_yaml("""
            inference:
              temperature: 0.3
              top_p: 0.95
        """)
        try:
            from sqush.config import load_config
            cfg = load_config(path)
            assert cfg.inference.temperature == pytest.approx(0.3)
            assert cfg.inference.top_p == pytest.approx(0.95)
        finally:
            os.unlink(path)

    def test_server_host_port_from_yaml(self):
        path = _write_yaml("""
            server:
              host: "0.0.0.0"
              port: 8080
        """)
        try:
            from sqush.config import load_config
            cfg = load_config(path)
            assert cfg.server.host == "0.0.0.0"
            assert cfg.server.port == 8080
        finally:
            os.unlink(path)

    def test_quantization_bits_from_yaml(self):
        path = _write_yaml("""
            quantization:
              weight_bits: 8
              kv_cache_bits: 8
        """)
        try:
            from sqush.config import load_config
            cfg = load_config(path)
            assert cfg.quantization.weight_bits == 8
            assert cfg.quantization.kv_cache_bits == 8
        finally:
            os.unlink(path)

    def test_logging_level_from_yaml(self):
        path = _write_yaml("""
            logging:
              level: DEBUG
        """)
        try:
            from sqush.config import load_config
            cfg = load_config(path)
            assert cfg.logging.level.upper() == "DEBUG"
        finally:
            os.unlink(path)

    def test_partial_yaml_keeps_other_defaults(self):
        """Partial YAML leaves unspecified fields at their defaults."""
        path = _write_yaml("""
            server:
              port: 9999
        """)
        try:
            from sqush.config import load_config
            cfg = load_config(path)
            assert cfg.server.port == 9999
            assert cfg.server.host == "127.0.0.1"  # default unchanged
            assert cfg.quantization.weight_bits == 4  # default unchanged
        finally:
            os.unlink(path)

    def test_empty_yaml_uses_all_defaults(self):
        path = _write_yaml("")
        try:
            from sqush.config import load_config
            cfg = load_config(path)
            assert cfg.quantization.weight_bits == 4
        finally:
            os.unlink(path)


class TestEnvVarConfig:
    """environment variable overrides."""

    def test_18_2_max_context_from_env(self):
        """SQUSH_MAX_CONTEXT overrides inference.max_context."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("SQUSH_MAX_CONTEXT", "32768")
            from sqush.config import load_config
            cfg = load_config("nonexistent.yaml")
        assert cfg.inference.max_context == 32768

    def test_18_2_host_from_env(self):
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("SQUSH_HOST", "0.0.0.0")
            from sqush.config import load_config
            cfg = load_config("nonexistent.yaml")
        assert cfg.server.host == "0.0.0.0"

    def test_18_2_port_from_env(self):
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("SQUSH_PORT", "7777")
            from sqush.config import load_config
            cfg = load_config("nonexistent.yaml")
        assert cfg.server.port == 7777

    def test_18_2_log_level_from_env(self):
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("SQUSH_LOG_LEVEL", "WARNING")
            from sqush.config import load_config
            cfg = load_config("nonexistent.yaml")
        assert cfg.logging.level == "WARNING"

    def test_18_2_env_overrides_yaml(self):
        """Env var takes precedence over YAML value."""
        path = _write_yaml("""
            server:
              port: 9898
        """)
        try:
            with pytest.MonkeyPatch.context() as mp:
                mp.setenv("SQUSH_PORT", "1234")
                from sqush.config import load_config
                cfg = load_config(path)
            assert cfg.server.port == 1234
        finally:
            os.unlink(path)


class TestVramTier:
    """VRAM tier classification and profile application."""

    def test_8gb_is_low(self):
        from sqush.config import classify_vram, VramTier
        assert classify_vram(8) == VramTier.LOW

    def test_16gb_is_low(self):
        from sqush.config import classify_vram, VramTier
        assert classify_vram(16) == VramTier.LOW

    def test_24gb_is_high(self):
        from sqush.config import classify_vram, VramTier
        assert classify_vram(24) == VramTier.HIGH

    def test_boundary_11gb_is_low(self):
        from sqush.config import classify_vram, VramTier
        assert classify_vram(11) == VramTier.LOW

    def test_boundary_12gb_is_low(self):
        from sqush.config import classify_vram, VramTier
        assert classify_vram(12) == VramTier.LOW

    def test_boundary_19gb_is_low(self):
        from sqush.config import classify_vram, VramTier
        assert classify_vram(19) == VramTier.LOW

    def test_boundary_20gb_is_high(self):
        from sqush.config import classify_vram, VramTier
        assert classify_vram(20) == VramTier.HIGH

    def test_vram8_picks_9b_repo(self):
        """8 GB profile selects the pre-quantized 9B model."""
        from sqush.config import load_config
        cfg = load_config("nonexistent.yaml", vram_gb=8)
        assert "9B" in cfg.model.repo or "9b" in cfg.model.repo.lower()

    def test_vram8_sets_256k_context(self):
        from sqush.config import load_config
        cfg = load_config("nonexistent.yaml", vram_gb=8)
        assert cfg.inference.max_context == 262144

    def test_vram24_picks_27b_repo(self):
        from sqush.config import load_config
        cfg = load_config("nonexistent.yaml", vram_gb=24)
        assert "27B" in cfg.model.repo

    def test_vram24_sets_256k_context(self):
        from sqush.config import load_config
        cfg = load_config("nonexistent.yaml", vram_gb=24)
        assert cfg.inference.max_context == 262144

    def test_vram_tier_stored_on_config(self):
        from sqush.config import load_config, VramTier
        cfg = load_config("nonexistent.yaml", vram_gb=8)
        assert cfg.vram_tier == VramTier.LOW

    def test_vram8_sets_max_image_pixels(self):
        from sqush.config import load_config
        cfg = load_config("nonexistent.yaml", vram_gb=8)
        assert cfg.inference.max_image_pixels is not None
        assert cfg.inference.max_image_pixels <= 262144
        # min_pixels must also be set — processor ignores max_pixels without min_pixels
        assert cfg.inference.min_image_pixels is not None

    def test_vram24_no_image_pixel_cap(self):
        from sqush.config import load_config
        cfg = load_config("nonexistent.yaml", vram_gb=24)
        assert cfg.inference.max_image_pixels is None
        assert cfg.inference.min_image_pixels is None

