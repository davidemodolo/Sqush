from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field
from enum import IntEnum

import yaml

log = logging.getLogger(__name__)


class VramTier(IntEnum):
    LOW = 8
    MEDIUM = 16
    HIGH = 24


VRAM_PROFILES: dict[VramTier, dict] = {
    VramTier.LOW: {
        "model": {"repo": "Qwen/Qwen3.5-9B"},
        "quantization": {"weight_bits": 4, "kv_cache_bits": 4},
        "inference": {"max_context": 131072},
    },
    VramTier.MEDIUM: {
        "model": {"repo": None},
        "quantization": {"weight_bits": 4, "kv_cache_bits": 4},
        "inference": {"max_context": None},
    },
    VramTier.HIGH: {
        "model": {"repo": "Qwen/Qwen3.6-27B"},
        "quantization": {"weight_bits": 4, "kv_cache_bits": 4},
        "inference": {"max_context": 262144},
    },
}


@dataclass
class ModelConfig:
    repo: str = "Qwen/Qwen3.6-27B"
    cache_dir: str = "./models"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"


@dataclass
class QuantizationConfig:
    weight_bits: int = 4
    kv_cache_bits: int = 4


@dataclass
class InferenceConfig:
    max_context: int = 262144
    max_new_tokens: int = 65536
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    presence_penalty: float = 1.5


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 9898


@dataclass
class LoggingConfig:
    level: str = "INFO"


@dataclass
class QuantStarConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    vram_tier: VramTier | None = None


def detect_vram() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True,
        )
        return int(out.strip().split("\n")[0]) // 1024
    except Exception:
        return 0


def classify_vram(raw_gb: int) -> VramTier:
    if raw_gb >= 20:
        return VramTier.HIGH
    if raw_gb >= 12:
        return VramTier.MEDIUM
    return VramTier.LOW


def load_config(path: str = "config.yaml", vram_gb: int | None = None) -> QuantStarConfig:
    cfg = QuantStarConfig()

    if os.path.exists(path):
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        if "model" in raw:
            for k, v in raw["model"].items():
                if hasattr(cfg.model, k):
                    setattr(cfg.model, k, v)
        if "quantization" in raw:
            for k, v in raw["quantization"].items():
                if hasattr(cfg.quantization, k):
                    setattr(cfg.quantization, k, v)
        if "inference" in raw:
            for k, v in raw["inference"].items():
                if hasattr(cfg.inference, k):
                    setattr(cfg.inference, k, v)
        if "server" in raw:
            for k, v in raw["server"].items():
                if hasattr(cfg.server, k):
                    setattr(cfg.server, k, v)
        if "logging" in raw:
            for k, v in raw["logging"].items():
                if hasattr(cfg.logging, k):
                    setattr(cfg.logging, k, v)

    if vram_gb is None:
        vram_gb = detect_vram()

    tier = classify_vram(vram_gb)
    cfg.vram_tier = tier
    profile = VRAM_PROFILES.get(tier)
    if profile is not None:
        for section, overrides in profile.items():
            target = getattr(cfg, section)
            for k, v in overrides.items():
                if v is not None and hasattr(target, k):
                    setattr(target, k, v)

    for key, value in os.environ.items():
        if key == "QUANTSTAR_MODEL_REPO":
            cfg.model.repo = value
        elif key == "QUANTSTAR_MODEL_CACHE":
            cfg.model.cache_dir = value
        elif key == "QUANTSTAR_WEIGHT_BITS":
            cfg.quantization.weight_bits = int(value)
        elif key == "QUANTSTAR_KV_BITS":
            cfg.quantization.kv_cache_bits = int(value)
        elif key == "QUANTSTAR_MAX_CONTEXT":
            cfg.inference.max_context = int(value)
        elif key == "QUANTSTAR_HOST":
            cfg.server.host = value
        elif key == "QUANTSTAR_PORT":
            cfg.server.port = int(value)
        elif key == "QUANTSTAR_LOG_LEVEL":
            cfg.logging.level = value

    return cfg
