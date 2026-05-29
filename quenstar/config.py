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
    n_batch: int = 2048
    n_ubatch: int = 256
    offload_kqv: bool = False
    flash_attn: bool = True
    use_mmap: bool = True
    type_k: int = 7
    type_v: int = 7
    yarn_ext_factor: float = -1.0
    yarn_attn_factor: float = 1.0
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0
    mmproj_path: str = ""


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8080
    cors: bool = False


@dataclass
class KVCacheConfig:
    dir: str = "./kvcache"
    space_mb: int = 8192
    eviction_half_life_hours: float = 6.0


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
    manual_token_loop: bool = False
    greedy_tool_syntax: bool = True
    payload_temperature: float = 0.7


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
        CONFIG_ENV_PREFIX + "MODEL_PATH": ("model", "path"),
        CONFIG_ENV_PREFIX + "N_GPU_LAYERS": ("model", "n_gpu_layers", int),
        CONFIG_ENV_PREFIX + "N_CTX": ("model", "n_ctx", int),
        CONFIG_ENV_PREFIX + "N_BATCH": ("model", "n_batch", int),
        CONFIG_ENV_PREFIX + "N_UBATCH": ("model", "n_ubatch", int),
        CONFIG_ENV_PREFIX + "OFFLOAD_KQV": ("model", "offload_kqv", _bool_env),
        CONFIG_ENV_PREFIX + "FLASH_ATTN": ("model", "flash_attn", _bool_env),
        CONFIG_ENV_PREFIX + "TYPE_K": ("model", "type_k", int),
        CONFIG_ENV_PREFIX + "TYPE_V": ("model", "type_v", int),
        CONFIG_ENV_PREFIX + "HOST": ("server", "host"),
        CONFIG_ENV_PREFIX + "PORT": ("server", "port", int),
        CONFIG_ENV_PREFIX + "KV_DIR": ("kv_cache", "dir"),
        CONFIG_ENV_PREFIX + "KV_SPACE_MB": ("kv_cache", "space_mb", int),
        CONFIG_ENV_PREFIX + "LOG_LEVEL": ("logging", "level"),
        CONFIG_ENV_PREFIX + "TRACE": ("logging", "trace", _bool_env),
        CONFIG_ENV_PREFIX + "MMPROJ_PATH": ("model", "mmproj_path"),
    }
    for env_key, (section, attr, *cast) in mapping.items():
        val = os.environ.get(env_key)
        if val is not None:
            if cast:
                val = cast[0](val)
            setattr(getattr(config, section), attr, val)


def _bool_env(v: str) -> bool:
    return v.lower() in ("1", "true", "yes", "on")


def apply_cli_overrides(config: QuenStarConfig, args: "argparse.Namespace"):
    """Apply common CLI argument overrides to config. Works with both __main__.py and cli.py Namespace objects."""
    if getattr(args, 'model', None):
        config.model.path = args.model
    if getattr(args, 'ctx', None) is not None and args.ctx is not None:
        config.model.n_ctx = args.ctx

    if getattr(args, 'max_tokens', None) is not None:
        config.generation.max_tokens = args.max_tokens

    for arg_name, config_attr in [('temp', 'default_temperature'), ('top_p', 'default_top_p'), ('top_k', 'default_top_k')]:
        val = getattr(args, arg_name, None)
        if val is not None:
            setattr(config.sampling, config_attr, val)

    n_gpu = getattr(args, 'n_gpu_layers', None)
    if n_gpu is not None:
        config.model.n_gpu_layers = n_gpu

    if hasattr(args, 'no_offload_kqv') and args.no_offload_kqv is not None:
        config.model.offload_kqv = not args.no_offload_kqv

    for cli_arg, config_attr in [('n_batch', 'n_batch'), ('n_ubatch', 'n_ubatch')]:
        val = getattr(args, cli_arg, None)
        if val is not None:
            setattr(config.model, config_attr, val)


def add_shared_model_args(parser):
    """Add CLI arguments shared between __main__.py and cli.py."""
    parser.add_argument("-m", "--model", default=None, help="Path to GGUF model file")
    parser.add_argument("--ctx", type=int, default=None, help="Context window size in tokens")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max tokens to generate")
    parser.add_argument("--temp", type=float, default=None, help="Temperature")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p sampling")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling")
    parser.add_argument("--n-gpu-layers", type=int, default=-1, help="GPU layers (-1=all)")
    parser.add_argument("--no-offload-kqv", action="store_true", default=None, help="Keep KV cache in system RAM instead of GPU VRAM")
    parser.add_argument("--n-batch", type=int, default=None, help="Batch size for prompt processing")
    parser.add_argument("--n-ubatch", type=int, default=None, help="Micro-batch size for GPU compute")
