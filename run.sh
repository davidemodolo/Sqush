#!/usr/bin/env bash
set -euo pipefail

# Sqush — quantized Qwen inference (8 GB and 24 GB VRAM)
# One-command setup and launch.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

info() { echo -e "${GREEN}[INFO]${NC} $*"; }
warn() { echo -e "${RED}[WARN]${NC} $*"; }

# ── GPU + CUDA version detection ───────────────────────────────
CUDA_TAG="cpu"
CUDA_INT=0
if command -v nvidia-smi &>/dev/null; then
    VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    VRAM_GB=$((VRAM_MB / 1024))
    info "GPU detected: ${VRAM_GB} GB VRAM"
    _cuda_ver=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1)
    if [ -n "$_cuda_ver" ]; then
        _major=$(echo "$_cuda_ver" | cut -d. -f1)
        _minor=$(echo "$_cuda_ver" | cut -d. -f2)
        CUDA_TAG="cu${_major}${_minor}"
        CUDA_INT=$((_major * 10 + _minor))
        info "CUDA version: ${_cuda_ver} (${CUDA_TAG})"
    fi
else
    warn "No NVIDIA GPU detected via nvidia-smi. CPU-only mode."
    VRAM_GB=0
fi

# ── CUDA library path (needed for bitsandbytes) ─────────────────
_py_site="$SCRIPT_DIR/.venv/lib/python3.14/site-packages"
for _lib_dir in "$_py_site/nvidia/"*/lib; do
    [ -d "$_lib_dir" ] && export LD_LIBRARY_PATH="$_lib_dir:${LD_LIBRARY_PATH:-}"
done

# Disable triton autotuner disk cache (avoids None cache key issue with FLA on py3.14)
export FLA_CACHE_RESULTS=0

# ── Virtual environment ────────────────────────────────────────
if [ ! -d ".venv" ]; then
    info "Creating virtual environment …"
    python3 -m venv .venv
fi
source .venv/bin/activate

# ── Install dependencies ───────────────────────────────────────
DEPS_MARKER=".venv/.deps_installed"
if [ ! -f "$DEPS_MARKER" ] || [ "$(cat "$DEPS_MARKER" 2>/dev/null)" != "$CUDA_TAG" ]; then
    info "Installing PyTorch and Torchvision with ${CUDA_TAG} (this may take a while) …"
    pip install torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" -q

    info "Installing project and dependencies from pyproject.toml …"
    pip install -e . -q

    info "Installing quanto for KV cache quantization (optional) …"
    pip install "quanto>=0.2.0" -q 2>/dev/null || warn "quanto not available — KV cache quantization disabled"

    echo "$CUDA_TAG" > "$DEPS_MARKER"
    info "Dependencies installed."

    # ── Patch transformers docstring (Qwen3.5 output class) ──────
    _qwen_file="$_py_site/transformers/models/qwen3_5/modeling_qwen3_5.py"
    if [ -f "$_qwen_file" ]; then
        python3 -c "
import re
path = '$_qwen_file'
with open(path) as fh:
    content = fh.read()
# Check if already patched
if re.search(r'loss.*\`torch\.FloatTensor\`', content):
    print('SKIP: already patched')
else:
    # Insert loss + logits entries after the r\"\"\"\n of the class docstring
    needle = r'(class Qwen3_5CausalLMOutputWithPast[^\n]*\n\s+r\"\"\"\n)'
    match = re.search(needle, content)
    if not match:
        print('SKIP: cannot find Qwen3_5CausalLMOutputWithPast docstring')
    else:
        insert_at = match.end()
        indent = '    '
        new_entries = (
            indent + 'loss (\`torch.FloatTensor\` of shape \`(1,)\`, *optional*):\n'
            + indent + '    Language modeling loss (for training).\n'
            + indent + 'logits (\`torch.FloatTensor\` of shape \`(batch_size, sequence_length, config.vocab_size)\`):\n'
            + indent + '    Prediction scores of the language modeling head.\n'
        )
        content = content[:insert_at] + new_entries + content[insert_at:]
        with open(path, 'w') as fh:
            fh.write(content)
        print('PATCHED: added loss and logits doc entries to Qwen3_5CausalLMOutputWithPast')
"
    fi

    # ── Patch bitsandbytes _check_is_size FutureWarning ──────────
    _bnb_ops="$_py_site/bitsandbytes/backends/cuda/ops.py"
    if [ -f "$_bnb_ops" ] && grep -q '_check_is_size' "$_bnb_ops"; then
        python3 -c "
path = '$_bnb_ops'
with open(path) as fh:
    content = fh.read()
if '_check_is_size' not in content:
    print('SKIP: already patched')
else:
    content = content.replace('torch._check_is_size(blocksize)', 'torch._check(blocksize >= 0)')
    with open(path, 'w') as fh:
        fh.write(content)
    print('PATCHED: replaced _check_is_size with _check in bitsandbytes ops.py')
"
    fi
fi

# ── Auto-select bitsandbytes binary ────────────────────────────
# bnb ships pre-compiled binaries only up to a certain CUDA version.
# If the exact version isn't available, symlink the closest match
# so bitsandbytes discovers it naturally (avoids BNB_CUDA_VERSION warning).
if [ "$CUDA_INT" -gt 0 ]; then
    _bnb_dir="$_py_site/bitsandbytes"
    _target_so="$_bnb_dir/libbitsandbytes_cuda${CUDA_INT}.so"
    _best_bnb=0
    for _so in "$_bnb_dir"/libbitsandbytes_cuda*.so; do
        [ -f "$_so" ] || continue
        _ver=$(basename "$_so" | grep -oP 'cuda\K[0-9]+')
        if [ "$_ver" -le "$CUDA_INT" ] && [ "$_ver" -gt "$_best_bnb" ]; then
            _best_bnb="$_ver"
        fi
    done
    if [ "$_best_bnb" -gt 0 ] && [ ! -f "$_target_so" ]; then
        _best_so="$_bnb_dir/libbitsandbytes_cuda${_best_bnb}.so"
        ln -sf "$(basename "$_best_so")" "$_target_so"
        if [ "$_best_bnb" -ne "$CUDA_INT" ]; then
            info "bitsandbytes: no binary for ${CUDA_TAG}, symlinked cuda${_best_bnb}"
        fi
    fi
fi

# ── Parse --vram override ──────────────────────────────────────
VRAM_OVERRIDE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --vram)
            VRAM_OVERRIDE="$2"
            shift 2
            ;;
        --vram=*)
            VRAM_OVERRIDE="${1#*=}"
            shift
            ;;
        *)
            break
            ;;
    esac
done
if [ -n "$VRAM_OVERRIDE" ]; then
    VRAM_GB="$VRAM_OVERRIDE"
    info "VRAM override: ${VRAM_GB} GB"
fi

# ── Launch ─────────────────────────────────────────────────────
MODE="${1:-chat}"

case "$MODE" in
    download)
        info "Downloading model …"
        python -m sqush --vram "$VRAM_GB" download
        ;;
    bake)
        info "Baking model (quantize + save compact cooked model, one-time) …"
        python -m sqush --vram "$VRAM_GB" bake
        ;;
    serve)
        info "Starting server …"
        python -m sqush --vram "$VRAM_GB" serve
        ;;
    chat)
        info "Starting interactive chat …"
        python -m sqush --vram "$VRAM_GB" chat
        ;;
    info)
        python -m sqush --vram "$VRAM_GB" info
        ;;
    init)
        info "Registering Sqush in OpenCode config …"
        python -m sqush --vram "$VRAM_GB" init
        ;;
    *)
        echo "Usage: ./run.sh [download|bake|serve|chat|info|init]"
        echo ""
        echo "  download  — download model from HuggingFace"
        echo "  bake      — quantize and save a compact cooked model, deleting the raw one"
        echo "  serve     — start OpenAI-compatible API server"
        echo "  chat      — start interactive CLI chat"
        echo "  info      — show configuration"
        echo "  init      — register Sqush in OpenCode config"
        exit 1
        ;;
esac
