from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

_log = logging.getLogger(__name__)

CONFIG_ENV_PREFIX = "QUENSTAR_"
DEFAULT_CONFIG_PATHS = [
    "./config.yaml",
    "~/.config/quenstar/config.yaml",
]


@dataclass
class ModelConfig:
    path: str = ""
    n_gpu_layers: int = -1
    n_ctx: int = 65536
    offload_kqv: bool = False
    flash_attn: bool = True
    use_mmap: bool = True


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8080
    cors: bool = False


@dataclass
class KVCacheConfig:
    dir: str = "~/.quenstar/kvcache"
    space_mb: int = 8192
    save_interval_tokens: int = 4096
    tail_trim_tokens: int = 32
    alignment_tokens: int = 2048
    eviction_half_life_hours: float = 6.0
    max_saved_sessions: int = 1000


@dataclass
class SamplingConfig:
    default_temperature: float = 0.8
    default_top_p: float = 0.9
    default_top_k: int = 40
    default_min_p: float = 0.05
    default_repeat_penalty: float = 1.0
    tool_call_temperature: float = 0.0
    tool_call_top_p: float = 1.0
    tool_call_top_k: int = 1


@dataclass
class GenerationConfig:
    max_tokens: int = 32768
    stop_strings: list[str] = field(default_factory=list)


@dataclass
class ToolCallingConfig:
    enabled: bool = True
    exact_replay_cache_size: int = 100000
    syntax_tokens_greedy_until: int = 0


@dataclass
class LoggingConfig:
    level: str = "INFO"
    trace: bool = False


@dataclass
class QuenStarConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    kv_cache: KVCacheConfig = field(default_factory=KVCacheConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    tool_calling: ToolCallingConfig = field(default_factory=ToolCallingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def load(cls, path: Optional[str] = None) -> "QuenStarConfig":
        config = cls()
        file_path = _resolve_config_path(path)
        if file_path:
            _log.info("Loading config from %s", file_path)
            with open(file_path) as f:
                data = yaml.safe_load(f) or {}
            config = _apply_yaml(config, data)
        _apply_env_overrides(config)
        return config


def _resolve_config_path(explicit: Optional[str]) -> Optional[Path]:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        return p if p.exists() else None
    for raw in DEFAULT_CONFIG_PATHS:
        p = Path(raw).expanduser().resolve()
        if p.exists():
            return p
    return None


def _apply_yaml(config: QuenStarConfig, data: dict) -> QuenStarConfig:
    merged = QuenStarConfig(
        model=_merge_dataclass(config.model, data.get("model", {})),
        server=_merge_dataclass(config.server, data.get("server", {})),
        kv_cache=_merge_dataclass(config.kv_cache, data.get("kv_cache", {})),
        sampling=_merge_dataclass(config.sampling, data.get("sampling", {})),
        generation=_merge_dataclass(config.generation, data.get("generation", {})),
        tool_calling=_merge_dataclass(config.tool_calling, data.get("tool_calling", {})),
        logging=_merge_dataclass(config.logging, data.get("logging", {})),
    )
    return merged


def _merge_dataclass(dc, overrides: dict):
    return type(dc)(**{**dc.__dict__, **{k: v for k, v in overrides.items() if k in dc.__dict__}})


def _apply_env_overrides(config: QuenStarConfig):
    mapping = {
        "QUENSTAR_MODEL_PATH": ("model", "path"),
        "QUENSTAR_N_GPU_LAYERS": ("model", "n_gpu_layers", int),
        "QUENSTAR_N_CTX": ("model", "n_ctx", int),
        "QUENSTAR_OFFLOAD_KQV": ("model", "offload_kqv", _bool_env),
        "QUENSTAR_FLASH_ATTN": ("model", "flash_attn", _bool_env),
        "QUENSTAR_HOST": ("server", "host"),
        "QUENSTAR_PORT": ("server", "port", int),
        "QUENSTAR_KV_DIR": ("kv_cache", "dir"),
        "QUENSTAR_KV_SPACE_MB": ("kv_cache", "space_mb", int),
        "QUENSTAR_LOG_LEVEL": ("logging", "level"),
        "QUENSTAR_TRACE": ("logging", "trace", _bool_env),
    }
    for env_key, (section, attr, *cast) in mapping.items():
        val = os.environ.get(env_key)
        if val is not None:
            if cast:
                val = cast[0](val)
            setattr(getattr(config, section), attr, val)


def _bool_env(v: str) -> bool:
    return v.lower() in ("1", "true", "yes", "on")
