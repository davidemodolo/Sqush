from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    repo: str = "Qwen/Qwen3.6-27B"
    cache_dir: str = "./models"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"


@dataclass
class QuantizationConfig:
    weight_bits: int = 3
    kv_cache_bits: int = 4
    turbo: bool = True
    hybrid: bool = False


@dataclass
class InferenceConfig:
    max_context: int = 262144
    max_new_tokens: int = 32768
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
class QuenStarConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(path: str = "config.yaml") -> QuenStarConfig:
    cfg = QuenStarConfig()

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

    for key, value in os.environ.items():
        if key == "QUENSTAR_MODEL_REPO":
            cfg.model.repo = value
        elif key == "QUENSTAR_MODEL_CACHE":
            cfg.model.cache_dir = value
        elif key == "QUENSTAR_WEIGHT_BITS":
            cfg.quantization.weight_bits = int(value)
        elif key == "QUENSTAR_KV_BITS":
            cfg.quantization.kv_cache_bits = int(value)
        elif key == "QUENSTAR_MAX_CONTEXT":
            cfg.inference.max_context = int(value)
        elif key == "QUENSTAR_HOST":
            cfg.server.host = value
        elif key == "QUENSTAR_PORT":
            cfg.server.port = int(value)
        elif key == "QUENSTAR_LOG_LEVEL":
            cfg.logging.level = value

    return cfg
