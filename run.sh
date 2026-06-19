#!/usr/bin/env bash
set -euo pipefail

# QuenStar v2 — Qwen3.6-27B quantized inference in 24GB VRAM
# One-command setup and launch.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

info() { echo -e "${GREEN}[INFO]${NC} $*"; }
warn() { echo -e "${RED}[WARN]${NC} $*"; }

# ── GPU check ──────────────────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
    VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    VRAM_GB=$((VRAM_MB / 1024))
    info "GPU detected: ${VRAM_GB} GB VRAM"
else
    warn "No NVIDIA GPU detected via nvidia-smi. CPU-only mode."
    VRAM_GB=0
fi

# ── CUDA 13 library path (needed for bitsandbytes) ─────────────
if [ -d "$SCRIPT_DIR/.venv/lib/python3.14/site-packages/nvidia/cu13/lib" ]; then
    export LD_LIBRARY_PATH="$SCRIPT_DIR/.venv/lib/python3.14/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
fi

# Disable triton autotuner disk cache (avoids None cache key issue with FLA on py3.14)
export FLA_CACHE_RESULTS=0

# ── Virtual environment ────────────────────────────────────────
if [ ! -d ".venv" ]; then
    info "Creating virtual environment …"
    python3 -m venv .venv
fi
source .venv/bin/activate

# ── Install dependencies ───────────────────────────────────────
if [ ! -f ".deps_installed" ]; then
    info "Installing PyTorch with CUDA 12.6 …"
    pip install torch --index-url https://download.pytorch.org/whl/cu126 -q

    info "Installing core dependencies …"
    pip install pyyaml tqdm rich huggingface_hub fastapi "uvicorn[standard]" sse-starlette pillow accelerate -q

    info "Installing transformers …"
    pip install transformers -q

    info "Installing bitsandbytes …"
    pip install bitsandbytes -q

    info "Installing quanto for KV cache quantization …"
    pip install quanto -q 2>/dev/null || warn "quanto not available — KV cache quantization disabled"

    touch .deps_installed
    info "Dependencies installed."
fi

# ── Launch ─────────────────────────────────────────────────────
MODE="${1:-chat}"

case "$MODE" in
    download)
        info "Downloading model …"
        python -m quenstar download
        ;;
    serve)
        info "Starting server …"
        python -m quenstar serve
        ;;
    chat)
        info "Starting interactive chat …"
        python -m quenstar chat
        ;;
    info)
        python -m quenstar info
        ;;
    *)
        echo "Usage: ./run.sh [download|serve|chat|info]"
        echo ""
        echo "  download  — download Qwen3.6-27B from HuggingFace"
        echo "  serve     — start OpenAI-compatible API server"
        echo "  chat      — start interactive CLI chat"
        echo "  info      — show configuration"
        exit 1
        ;;
esac
