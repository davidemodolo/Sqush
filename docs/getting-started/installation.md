# Installation

## Requirements

- **Python** ≥ 3.10 (the project is developed and tested on 3.14).
- **NVIDIA GPU** with CUDA. Sqush targets two tiers: **8 GB** (Qwen3.5‑9B) and **24 GB** (Qwen3.6‑27B). CPU‑only runs load but are not practical for inference.
- **Disk**: enough for the raw download plus the cooked model during baking — see [VRAM tiers](../concepts/vram-tiers.md). For the 27B this is ~70 GB transiently, ~18 GB steady‑state.
- `nvidia-smi` on `PATH` (used for GPU/VRAM/CUDA detection).

## One command: `run.sh`

`run.sh` is the supported entry point. It bootstraps everything and then dispatches to a subcommand:

```bash
./run.sh serve
```

On first run it:

1. **Detects the GPU and CUDA version** via `nvidia-smi`, deriving a wheel tag (e.g. `cu126`).
2. **Creates a virtualenv** in `.venv` if missing.
3. **Installs PyTorch/Torchvision** from the matching CUDA index URL, then the project (`pip install -e .`) and optional `quanto` for KV‑cache quantization.
4. **Exports the bitsandbytes CUDA library path** (`LD_LIBRARY_PATH` from the bundled `nvidia/*/lib` dirs) and disables the Triton autotuner disk cache (`FLA_CACHE_RESULTS=0`).
5. **Applies two upstream patches** (idempotent): a transformers docstring fix for the Qwen3.5 output class on Python 3.14, and a bitsandbytes `_check_is_size` → `_check` replacement to silence a `FutureWarning`.
6. **Auto‑selects the bitsandbytes binary** — symlinks the closest available `libbitsandbytes_cudaXXX.so` when no exact match ships.

Dependencies are only reinstalled when the CUDA tag changes (tracked by `.venv/.deps_installed`).

## Manual install

If you prefer to manage the environment yourself:

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -e .
pip install -e ".[kv-cache]"   # optional: quanto
pip install -e ".[test]"       # optional: pytest + httpx for the test suite
```

Then invoke the module directly:

```bash
python -m sqush serve
```

!!! warning "bitsandbytes + CUDA"
    NF4 weight quantization and the `lm_head`/visual‑encoder quantization steps require CUDA — bitsandbytes cannot quantize on CPU. `run.sh` handles the library discovery; a manual install may need `LD_LIBRARY_PATH` pointed at the bundled NVIDIA libs if bitsandbytes can't find its `.so`.

## Dependencies

Core (from `pyproject.toml`): `torch≥2.5`, `torchvision≥0.20`, `transformers≥5.0`, `accelerate≥1.0`, `bitsandbytes≥0.45`, `huggingface_hub≥0.26`, `fastapi`, `uvicorn[standard]`, `sse-starlette`, `pyyaml`, `tqdm`, `rich`, `pillow`.

Optional extras: `kv-cache` (`quanto`), `test` (`pytest`, `httpx`).
