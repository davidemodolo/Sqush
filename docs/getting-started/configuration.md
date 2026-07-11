# Configuration

Configuration is resolved in `sqush/config.py` by `load_config()` in three layers, later layers overriding earlier ones:

1. **Dataclass defaults** (`SqushConfig` and its sections).
2. **VRAM tier profile** — auto‑selected from detected (or `--vram`‑overridden) VRAM.
3. **`config.yaml`** — per‑section overrides.
4. **`SQUSH_*` environment variables** — highest precedence.

## `config.yaml`

```yaml
model:
  repo: "Qwen/Qwen3.6-27B"
  cache_dir: "./models"
  torch_dtype: "bfloat16"
  attn_implementation: "sdpa"

quantization:
  weight_bits: 4
  kv_cache_bits: 4

inference:
  max_context: 262144      # full 256k
  max_new_tokens: 65536
  temperature: 0.7
  top_p: 0.8
  top_k: 20
  presence_penalty: 1.5
  max_image_pixels: null   # cap before the vision encoder; null = no limit
  min_image_pixels: null   # must be set alongside max_image_pixels

server:
  host: "127.0.0.1"
  port: 9898

logging:
  level: "INFO"
  tps_interval_tokens: 50  # log streaming throughput every N tokens
```

Only keys that exist on the corresponding dataclass are applied (`hasattr` guard), so unknown keys are ignored rather than erroring.

## Configuration sections

| Section | Fields |
|---------|--------|
| `model` | `repo`, `cache_dir`, `torch_dtype`, `attn_implementation` |
| `quantization` | `weight_bits`, `kv_cache_bits` |
| `inference` | `max_context`, `max_new_tokens`, `temperature`, `top_p`, `top_k`, `presence_penalty`, `max_image_pixels`, `min_image_pixels` |
| `server` | `host`, `port` |
| `logging` | `level`, `tps_interval_tokens` |

## Environment variables

`SQUSH_*` vars are applied last and win over both YAML and the tier profile:

| Variable | Maps to |
|----------|---------|
| `SQUSH_MODEL_REPO` | `model.repo` |
| `SQUSH_MODEL_CACHE` | `model.cache_dir` |
| `SQUSH_WEIGHT_BITS` | `quantization.weight_bits` (int) |
| `SQUSH_KV_BITS` | `quantization.kv_cache_bits` (int) |
| `SQUSH_MAX_CONTEXT` | `inference.max_context` (int) |
| `SQUSH_HOST` | `server.host` |
| `SQUSH_PORT` | `server.port` (int) |
| `SQUSH_LOG_LEVEL` | `logging.level` |

## VRAM detection & override

`detect_vram()` shells out to `nvidia-smi --query-gpu=memory.total` and returns whole GB. `classify_vram()` maps it to a tier:

- **≥ 20 GB → HIGH** (Qwen3.6‑27B)
- **< 20 GB → LOW** (Qwen3.5‑9B)

Override detection with `--vram <GB>` (on `run.sh` or `python -m sqush`). See [VRAM tiers](../concepts/vram-tiers.md) for what each tier changes.

!!! note "The tier profile only sets non‑null values"
    A profile entry of `null`/`None` is skipped when applied, so it never clobbers a default or a YAML value.
