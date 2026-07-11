# Commands

Both `./run.sh <cmd>` and `python -m sqush <cmd>` expose the same subcommands. `run.sh` additionally bootstraps the environment (venv, CUDA wheels, patches) on first run.

| Command | Description |
|---------|-------------|
| `download` | Download the model from HuggingFace (resume‑aware). |
| `bake` | Quantize and save a compact cooked model, deleting the raw one. Skips if the cooked model already exists. |
| `serve` | Start the OpenAI‑compatible API server (bakes on first run if needed). |
| `chat` | Start the interactive Rich CLI (bakes on first run if needed). |
| `info` | Print the resolved configuration and VRAM. |
| `init` | Register Sqush as a provider in `~/.config/opencode/opencode.json`. |

## Global flags

| Flag | Applies to | Meaning |
|------|-----------|---------|
| `--config <path>` | `python -m sqush` | Config file (default `config.yaml`). |
| `--log-level <lvl>` | `python -m sqush` | Override `logging.level`. |
| `--vram <GB>` | both | Override VRAM detection (selects the tier). |

Examples:

```bash
./run.sh --vram 8 serve            # force the 8 GB profile
python -m sqush --config prod.yaml --log-level DEBUG serve
python -m sqush bake               # explicit one-time bake
```

## `download`

Resolves `models/<repo-with-__>` from `model.cache_dir`/`model.repo`. If the directory is a **complete** checkpoint (index + all shards, or single‑file + config) it's reused; otherwise `snapshot_download(resume_download=True)` runs, ignoring `*.pth/*.bin/*.msgpack/*.h5`.

## `bake`

Computes the cooked path from config first, so a re‑run doesn't re‑download. If cooked exists, does nothing. Otherwise downloads (if needed) and bakes per tier — LOW side‑car, HIGH GPU NF4 — then deletes the raw. See [Baking](../concepts/baking.md).

## `serve` / `chat`

Resolve the cooked path; if present, serve from it; otherwise download + bake first. Then `load_and_quantize_model`, build the `InferenceEngine`, and either start uvicorn (`serve`) or the CLI (`chat`). `serve` warms up the engine before binding.
