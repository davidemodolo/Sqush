#!/usr/bin/env bash
set -euo pipefail

# QuantStar — Qwen3.6-27B quantized inference in 24GB VRAM
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

# ── Patch transformers docstring warnings ───────────────────────
patch_docstring() {
    local f="$SCRIPT_DIR/.venv/lib/python3.14/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py"
    if [ -f "$f" ] && grep -q "^class Qwen3_5CausalLMOutputWithPast" "$f"; then
        if grep -q "loss (\`torch.FloatTensor\`" "$f" 2>/dev/null; then
            return 0  # already patched
        fi
        info "Patching missing doc entries in Qwen3_5CausalLMOutputWithPast …"
        python3 -c "
f = '$f'
with open(f) as fh:
    content = fh.read()
old = '''class Qwen3_5CausalLMOutputWithPast(CausalLMOutputWithPast):
    r\\\"\\\"\\\"
    rope_deltas'''
new = '''class Qwen3_5CausalLMOutputWithPast(CausalLMOutputWithPast):
    r\\\"\\\"\\\"
    loss (\`torch.FloatTensor\` of shape \`(1,)\`, *optional*):
        Language modeling loss (for training).
    logits (\`torch.FloatTensor\` of shape \`(batch_size, sequence_length, config.vocab_size)\`):
        Prediction scores of the language modeling head.
    rope_deltas'''
if old not in content:
    print('SKIP: docstring format different from expected')
else:
    content = content.replace(old, new)
    with open(f, 'w') as fh:
        fh.write(content)
    print('PATCHED')
"
    fi
}
patch_docstring

# ── Virtual environment ────────────────────────────────────────
if [ ! -d ".venv" ]; then
    info "Creating virtual environment …"
    python3 -m venv .venv
fi
source .venv/bin/activate

# ── Install dependencies ───────────────────────────────────────
DEPS_MARKER=".venv/.deps_installed"
if [ ! -f "$DEPS_MARKER" ]; then
    info "Installing PyTorch and Torchvision with CUDA 12.6 (this may take a while) …"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 -q

    info "Installing project and dependencies from pyproject.toml …"
    pip install -e . -q

    info "Installing quanto for KV cache quantization (optional) …"
    pip install "quanto>=0.2.0" -q 2>/dev/null || warn "quanto not available — KV cache quantization disabled"

    touch "$DEPS_MARKER"
    info "Dependencies installed."
fi

# ── Launch ─────────────────────────────────────────────────────
MODE="${1:-chat}"

case "$MODE" in
    download)
        info "Downloading model …"
        python -m quantstar download
        ;;
    serve)
        info "Starting server …"
        python -m quantstar serve
        ;;
    chat)
        info "Starting interactive chat …"
        python -m quantstar chat
        ;;
    info)
        python -m quantstar info
        ;;
    init)
        info "Registering QuantStar in OpenCode config …"
        python -m quantstar init
        ;;
    *)
        echo "Usage: ./run.sh [download|serve|chat|info|init]"
        echo ""
        echo "  download  — download Qwen3.6-27B from HuggingFace"
        echo "  serve     — start OpenAI-compatible API server"
        echo "  chat      — start interactive CLI chat"
        echo "  info      — show configuration"
        echo "  init      — register QuantStar in OpenCode config"
        exit 1
        ;;
esac
