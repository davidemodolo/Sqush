#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${QUENSTAR_MODEL_DIR:-$SCRIPT_DIR/models}"
KV_DIR="${QUENSTAR_KV_DIR:-$HOME/.quenstar/kv}"
CTX="${QUENSTAR_CTX:-131072}"
PORT="${QUENSTAR_PORT:-8080}"
HOST="${HOST:-127.0.0.1}"
KV_SPACE="${KV_SPACE:-8192}"
HF_TOKEN="${HF_TOKEN:-}"
VENV_DIR=".venv"
MODEL_PATH=""
DOWNLOAD=""
INSTALL_DEPS=false
OFFLOAD_KQV="--no-offload-kqv"
CUDA_INSTALL="${QUENSTAR_CUDA:-1}"
AUTO_DOWNLOAD_FALLBACK="q4_k_m"
MODE="auto"

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[quenstar]${NC} $*"; }
warn()  { echo -e "${RED}[quenstar]${NC} $*"; }
step()  { echo -e "${CYAN}==>${NC} $*"; }
hint()  { echo -e "${YELLOW}  ->${NC} $*"; }

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Options:
  -m, --model PATH      Path to GGUF model file (auto-detect)
  --mode MODE           Force hardware preset: desktop|laaptop (auto-detected by default)
  --download VARIANT    Download a specific quantization (q4_k_m, iq4_xs, q3_k_m, iq3_xxs, iq2_m)
  --hf-token TOKEN      HuggingFace token for authenticated downloads
  --cuda                Install llama-cpp-python with CUDA support (default)
  --cpu                 Install llama-cpp-python without CUDA (CPU only)
  --ctx N               Context window size (default: $CTX, model max: 262144)
  --port N              Server port (default: $PORT)
  --host HOST           Server host (default: $HOST)
  --kv-dir PATH         Disk KV cache directory (default: $KV_DIR)
  --kv-space-mb N       Max disk space for KV cache in MB (default: $KV_SPACE)
  --no-offload-kqv      Keep KV cache in system RAM instead of GPU VRAM
  --cors                Enable CORS headers
  --trace               Enable trace logging
  --install-deps        Install Python dependencies into a venv
  --venv PATH           Path to virtualenv (default: .venv)
  -h, --help            Show this help

Environment:
  HF_TOKEN              HuggingFace authentication token
  QUENSTAR_MODEL_DIR    Directory to store/download models (default: ./models)
  QUENSTAR_KV_DIR       Disk KV cache directory
  QUENSTAR_CTX          Default context size
  QUENSTAR_PORT         Default server port
  QUENSTAR_CUDA         1=install with CUDA, 0=CPU only

Examples:
  $0                                              # auto-download & run
  $0 -m ./models/qwen3.6-35b-a3b-iq4_xs.gguf     # specify model
  $0 --download iq4_xs --hf-token hf_xxx          # download with auth
  $0 --install-deps --download iq4_xs              # install + download
  $0 --ctx 32768 --trace                           # debug mode
EOF
    exit 0
}

# ── argument parsing ──────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model)        MODEL_PATH="$2"; shift 2 ;;
        --download)        DOWNLOAD="$2"; shift 2 ;;
        --hf-token)        HF_TOKEN="$2"; shift 2 ;;
        --cuda)            CUDA_INSTALL=1; shift ;;
        --cpu)             CUDA_INSTALL=0; shift ;;
        --ctx)             CTX="$2"; shift 2 ;;
        --port)            PORT="$2"; shift 2 ;;
        --host)            HOST="$2"; shift 2 ;;
        --kv-dir)          KV_DIR="$2"; shift 2 ;;
        --kv-space-mb)     KV_SPACE="$2"; shift 2 ;;
        --no-offload-kqv)  OFFLOAD_KQV="--no-offload-kqv"; shift ;;
        --cors)            CORS="--cors"; shift ;;
        --trace)           TRACE="--trace"; shift ;;
        --install-deps)    INSTALL_DEPS=true; shift ;;
        --venv)            VENV_DIR="$2"; shift 2 ;;
        --mode)            MODE="$2"; shift 2 ;;
        -h|--help)         usage ;;
        *) warn "Unknown option: $1"; usage ;;
    esac
done

# ── mode presets ───────────────────────────────────────────────────

detect_mode() {
    if [ "$MODE" != "auto" ]; then
        return 0
    fi
    local vram_mb=0
    if command -v nvidia-smi &>/dev/null; then
        vram_mb="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)"
    fi

    if [ "$vram_mb" -ge 22000 ]; then
        MODE="desktop"
    elif [ "$vram_mb" -ge 6000 ]; then
        MODE="laptop"
    else
        MODE="desktop"
        warn "Could not detect GPU VRAM. Defaulting to desktop mode."
        warn "Use --mode laptop to force laptop mode."
    fi
}

detect_mode

apply_mode "${MODE:-desktop}"

# ── helpers ────────────────────────────────────────────────────────

verify_gguf() {
    local file="$1"
    local size
    size="$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null || echo 0)"
    [ "$size" -ge 1048576 ] || return 1
    local magic
    magic="$(head -c4 "$file" 2>/dev/null || true)"
    [ "$magic" = "GGUF" ] || return 1
    return 0
}

find_cuda_libdir() {
    local patterns=(
        "/usr/local/lib/ollama/cuda_v12"
        "/usr/local/lib/ollama/cuda_v13"
        "/usr/local/lib/ollama/mlx_cuda_v12"
        "/usr/local/lib/ollama/mlx_cuda_v13"
        "/usr/local/cuda/lib64"
        "/usr/local/cuda-12/lib64"
        "/opt/cuda/lib64"
    )
    for d in "${patterns[@]}"; do
        if [ -f "$d/libcudart.so" ] || [ -f "$d/libcudart.so.12" ] || [ -f "$d/libcudart.so.13" ]; then
            echo "$d"; return 0
        fi
    done
    return 1
}

model_variants() {
    local variant="${1:-}"
    case "$MODE:$variant" in
        desktop:q4_k_m|desktop:Q4_K_M)
            echo "${MODEL_PREFIX}-Q4_K_M.gguf|22.1 GB|Q4_K_M"
            ;;
        desktop:q4_k_s|desktop:Q4_K_S)
            echo "${MODEL_PREFIX}-Q4_K_S.gguf|20.9 GB|Q4_K_S"
            ;;
        desktop:iq4_xs|desktop:IQ4_XS)
            echo "${MODEL_PREFIX}-IQ4_XS.gguf|17.7 GB|IQ4_XS"
            ;;
        desktop:q3_k_m|desktop:Q3_K_M)
            echo "${MODEL_PREFIX}-Q3_K_M.gguf|16.6 GB|Q3_K_M"
            ;;
        desktop:iq3_xxs|desktop:IQ3_XXS)
            echo "${MODEL_PREFIX}-IQ3_XXS.gguf|13.2 GB|IQ3_XXS"
            ;;
        desktop:iq2_m|desktop:IQ2_M)
            echo "${MODEL_PREFIX}-IQ2_M.gguf|11.5 GB|IQ2_M"
            ;;
        desktop:q5_k_m|desktop:Q5_K_M)
            echo "${MODEL_PREFIX}-Q5_K_M.gguf|26.5 GB|Q5_K_M"
            ;;

        laptop:q4_k_m|laptop:Q4_K_M)
            echo "${MODEL_PREFIX}-Q4_K_M.gguf|8.7 GB|Q4_K_M"
            ;;
        laptop:q4_k_s|laptop:Q4_K_S)
            echo "${MODEL_PREFIX}-Q4_K_S.gguf|8.3 GB|Q4_K_S"
            ;;
        laptop:iq4_xs|laptop:IQ4_XS)
            echo "${MODEL_PREFIX}-IQ4_XS.gguf|7.5 GB|IQ4_XS"
            ;;
        laptop:q3_k_m|laptop:Q3_K_M)
            echo "${MODEL_PREFIX}-Q3_K_M.gguf|6.5 GB|Q3_K_M"
            ;;
        laptop:iq3_xxs|laptop:IQ3_XXS)
            echo "${MODEL_PREFIX}-IQ3_XXS.gguf|5.5 GB|IQ3_XXS"
            ;;
        laptop:iq2_m|laptop:IQ2_M)
            echo "${MODEL_PREFIX}-IQ2_M.gguf|4.8 GB|IQ2_M"
            ;;
        laptop:q5_k_m|laptop:Q5_K_M)
            echo "${MODEL_PREFIX}-Q5_K_M.gguf|10.2 GB|Q5_K_M"
            ;;

        *)
            return 1
            ;;
    esac
}

# ── venv setup ─────────────────────────────────────────────────────

if $INSTALL_DEPS || [ ! -d "$VENV_DIR" ] || [ ! -f "$VENV_DIR/bin/activate" ]; then
    step "Setting up virtualenv: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    step "Installing base dependencies..."
    pip install --upgrade pip -q
    pip install fastapi "uvicorn[standard]" sse-starlette pyyaml -q

    if [ "$CUDA_INSTALL" = "1" ]; then
        cuda_dir="$(find_cuda_libdir || true)"
        if [ -n "$cuda_dir" ] || command -v nvidia-smi &>/dev/null; then
            [ -n "$cuda_dir" ] && info "CUDA libraries found at: $cuda_dir"

            step "Installing llama-cpp-python with CUDA 12 support..."
            if pip install llama-cpp-python \
                --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 \
                --force-reinstall --no-cache-dir -q 2>/dev/null; then
                info "Installed llama-cpp-python (CUDA 12.4 pre-built)"
            else
                warn "Pre-built CUDA wheel failed. Trying source build..."
                if command -v cmake &>/dev/null; then
                    CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python \
                        --force-reinstall --no-cache-dir -q
                    info "Installed llama-cpp-python (CUDA source build)"
                else
                    warn "cmake not found. Install cmake and try again, or use --cpu."
                    pip install llama-cpp-python -q
                    info "Installed llama-cpp-python (CPU only)"
                fi
            fi
        else
            warn "No CUDA detected. Installing CPU-only llama-cpp-python."
            pip install llama-cpp-python -q
        fi
    else
        step "Installing CPU-only llama-cpp-python..."
        pip install llama-cpp-python -q
    fi

    info "Dependencies installed."
else
    if [ -f "$VENV_DIR/bin/activate" ]; then
        source "$VENV_DIR/bin/activate"
    fi
fi

# ── CUDA library path ──────────────────────────────────────────────

if [ "$CUDA_INSTALL" = "1" ]; then
    cuda_dir="$(find_cuda_libdir || true)"
    [ -n "$cuda_dir" ] && export LD_LIBRARY_PATH="$cuda_dir:${LD_LIBRARY_PATH:-}"
fi

# ── auto-detect model ──────────────────────────────────────────────

if [ -n "$MODEL_PATH" ] && [ -f "$MODEL_PATH" ] && verify_gguf "$MODEL_PATH"; then
    : # explicit model is valid
elif [ -z "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH" ]; then
    step "Searching for model in $MODEL_DIR..."
    mkdir -p "$MODEL_DIR"

    # Check if any GGUF exists
    local_model=""
    for candidate in \
        "$MODEL_DIR"/"${MODEL_PREFIX}"-Q4_K_M.gguf \
        "$MODEL_DIR"/"${MODEL_PREFIX}"-IQ4_XS.gguf \
        "$MODEL_DIR"/"${MODEL_PREFIX}"-Q5_K_M.gguf \
        "$MODEL_DIR"/"${MODEL_PREFIX}"-Q4_K_S.gguf \
        "$MODEL_DIR"/"${MODEL_PREFIX}"-Q3_K_M.gguf \
        "$MODEL_DIR"/"${MODEL_PREFIX}"-IQ3_XXS.gguf \
        "$MODEL_DIR"/"${MODEL_PREFIX}"-IQ2_M.gguf \
        "$MODEL_DIR"/*.gguf \
        "$SCRIPT_DIR"/*.gguf \
        ; do
        if [ -f "$candidate" ] && verify_gguf "$candidate"; then
            local_model="$candidate"; break
        fi
    done

    if [ -n "$local_model" ]; then
        MODEL_PATH="$local_model"
        info "Found model: $MODEL_PATH"
    fi
fi

# ── auto-download if no model found ────────────────────────────────

if [ -z "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH" ] || ! verify_gguf "$MODEL_PATH"; then
    [ -f "$MODEL_PATH" ] && { warn "Invalid model, removing: $MODEL_PATH"; rm -f "$MODEL_PATH"; }

    VARIANT="${DOWNLOAD:-$AUTO_DOWNLOAD_FALLBACK}"

    if [ -z "$DOWNLOAD" ]; then
        step "No model found. Auto-downloading ${MODEL_FAMILY} ($VARIANT)..."
        hint "Use --download <variant> to pick a different quant"
        hint "Use --mode laptop to force laptop mode"
        hint "Available quants: q4_k_m, iq4_xs, q3_k_m, iq3_xxs, q2_m"
        echo ""
    else
        step "Downloading ${MODEL_FAMILY} ($VARIANT)..."
    fi

    variant_info="$(model_variants "$VARIANT" || true)"
    if [ -z "$variant_info" ]; then
        warn "Unknown variant: $VARIANT"
        echo "Available: q4_k_m, q4_k_s, iq4_xs, q3_k_m, iq3_xxs, iq2_m"
        exit 1
    fi

    IFS='|' read -r HF_FILE SIZE_HINT HF_QUANT <<< "$variant_info"

    DEST="$MODEL_DIR/${HF_FILE,,}"

    if [ -f "$DEST" ] && verify_gguf "$DEST"; then
        local_size="$(stat -c%s "$DEST" 2>/dev/null || stat -f%z "$DEST" 2>/dev/null || echo 0)"
        local_gb="$(echo "scale=1; $local_size / 1073741824" | bc 2>/dev/null || echo '?')"
        info "Model already downloaded: $DEST (${local_gb}GB)"
        MODEL_PATH="$DEST"
    else
        [ -f "$DEST" ] && { warn "Removing corrupt download: $DEST"; rm -f "$DEST"; }

        URL="https://huggingface.co/${HF_REPO}/resolve/main/${HF_FILE}"
        info "Downloading $HF_FILE ($SIZE_HINT)..."
        hint "Source: $HF_REPO"

        CURL_OPTS=(--location --continue-at - --progress-bar -o "$DEST")
        [ -n "$HF_TOKEN" ] && CURL_OPTS+=(-H "Authorization: Bearer $HF_TOKEN")

        if command -v curl &>/dev/null; then
            curl "${CURL_OPTS[@]}" "$URL"
        elif command -v wget &>/dev/null; then
            WGET_OPTS=(-c -q --show-progress -O "$DEST")
            [ -n "$HF_TOKEN" ] && WGET_OPTS+=(--header "Authorization: Bearer $HF_TOKEN")
            wget "${WGET_OPTS[@]}" "$URL"
        else
            warn "Neither curl nor wget found."; exit 1
        fi

        if verify_gguf "$DEST"; then
            local_size="$(stat -c%s "$DEST" 2>/dev/null || stat -f%z "$DEST" 2>/dev/null || echo 0)"
            local_gb="$(echo "scale=1; $local_size / 1073741824" | bc 2>/dev/null || echo '?')"

            # Check download is complete: file should be within 15% of expected size
            expected_bytes="$(echo "$SIZE_HINT" | sed 's/ GB//' | awk '{printf "%.0f", $1 * 1073741824}')"
            min_bytes="$(echo "$expected_bytes" | awk '{printf "%.0f", $1 * 0.85}')"
            if [ "$local_size" -lt "$min_bytes" ]; then
                warn "Download appears incomplete (${local_gb}GB, expected ~${SIZE_HINT})."
                hint "Run again to resume with curl -C, or delete and retry:"
                hint "  rm '$DEST' && ./run.sh"
                exit 1
            fi

            info "Downloaded: $DEST (${local_gb}GB)"
            MODEL_PATH="$DEST"
        else
            warn "Download failed. The file is not a valid GGUF model."
            echo ""

            local snippet
            snippet="$(head -c 300 "$DEST" 2>/dev/null | cat -v || true)"
            if echo "$snippet" | grep -qi "invalid\|password\|authentication\|login\|authorization"; then
                hint "HuggingFace requires authentication for this model."
                echo ""
            fi

            cat <<'AUTH_HELP'
  Option 1: Login with the 'hf' CLI (recommended)
            pip install huggingface_hub
            hf auth login
            ./run.sh

  Option 2: Pass a token directly
            ./run.sh --hf-token hf_xxxxxxxxxxxxx

  Option 3: Use environment variable
            export HF_TOKEN=hf_xxxxxxxxxxxxx
            ./run.sh

  Option 4: Download manually from your browser
            https://huggingface.co/${HF_REPO}
            Place the file in models/ and run ./run.sh
AUTH_HELP
            rm -f "$DEST"
            exit 1
        fi
    fi
fi

# ── final validation ───────────────────────────────────────────────

verify_gguf "$MODEL_PATH" || {
    warn "Not a valid GGUF model: $MODEL_PATH"
    head -c 200 "$MODEL_PATH" 2>/dev/null | cat -v || true
    echo ""
    hint "Delete this file and re-download."
    exit 1
}

# ── start server ───────────────────────────────────────────────────

local_size="$(stat -c%s "$MODEL_PATH" 2>/dev/null || stat -f%z "$MODEL_PATH" 2>/dev/null || echo 0)"
local_gb="$(echo "scale=1; $local_size / 1073741824" | bc 2>/dev/null || echo '?')"

step "Starting QuenStar..."
info "Model:   $MODEL_PATH (${local_gb}GB)"
info "Context: $CTX tokens"
info "Server:  http://$HOST:$PORT"
info "KV dir:  $KV_DIR (max ${KV_SPACE}MB)"
echo ""

exec python3 -m quenstar \
    -m "$MODEL_PATH" \
    --ctx "$CTX" \
    --host "$HOST" \
    --port "$PORT" \
    --kv-dir "$KV_DIR" \
    --kv-space-mb "$KV_SPACE" \
    $OFFLOAD_KQV \
    ${CORS:-} \
    ${TRACE:-}
