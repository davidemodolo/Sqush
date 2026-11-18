#!/usr/bin/env bash
set -euo pipefail

# Usage: ./serve.sh [--vision] [--1bit] [--kv-static]
#   (default)    text only, full context, dynamic KV quantization (f16 -> q8_0 -> q4_0)
#   --vision     adds the image projector (~23.5 GB VRAM)
#   --1bit       UD-IQ1_S model for ~8 GB VRAM: ~96k context, no MTP (default after a 1-bit install)
#   --kv-static  one KV type for the whole context (KV_TYPE, default q4_0)
#
# Settings come from the environment, then config.env (written by install.sh), then the defaults:
#   HOST=127.0.0.1  PORT=8080  MODEL_DIR=./models  CTX=262144  QUANT=UD-IQ4_XS
#   HF_REPO=unsloth/Qwen3.8-27B-GGUF  N_MAX=2  SPEC_TYPE=draft-mtp  IMAGE_MAX_TOKENS=1024
#   KV_STAGES=f16,q8_0,q4_0  KV_MARGIN=4096  KV_TYPE=q4_0  CACHE_RAM=16384 (MiB)
#   ONEBIT_CTX=98304  ONEBIT_QUANT=UD-IQ1_S  VARIANT=full|1bit  BATCH="-b 2048 -ub 256"
#   ALIAS=<model id>  LLAMA_SERVER=<path>  LV=<log verbosity>

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

load_config() {
    local file="$1" key value
    [[ -f "${file}" ]] || return 0
    while IFS='=' read -r key value; do
        [[ "${key}" =~ ^[A-Z_][A-Z0-9_]*$ ]] || continue
        [[ -n "${!key+x}" ]] || export "${key}=${value}"
    done < "${file}"
}
load_config "${SCRIPT_DIR}/config.env"

VISION=0
ONEBIT=0
[[ "${VARIANT:-}" != "1bit" ]] || ONEBIT=1
DYNAMIC=1
for arg in "$@"; do
    case "${arg}" in
        --vision) VISION=1 ;;
        --1bit) ONEBIT=1 ;;
        --kv-static) DYNAMIC=0 ;;
        *) echo "unknown argument: ${arg} (usage: ./serve.sh [--vision] [--1bit] [--kv-static])" >&2; exit 1 ;;
    esac
done
if [[ "${VISION}" -eq 1 ]] && [[ "${ONEBIT}" -eq 1 ]]; then
    echo "!! --vision and --1bit are mutually exclusive" >&2
    exit 1
fi

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
MODEL_DIR="${MODEL_DIR:-${SCRIPT_DIR}/models}"
CTX="${CTX:-262144}"
QUANT="${QUANT:-UD-IQ4_XS}"
N_MAX="${N_MAX:-2}"
SPEC_TYPE="${SPEC_TYPE:-draft-mtp}"
BATCH="${BATCH:--b 2048 -ub 256}"
HF_REPO="${HF_REPO:-unsloth/Qwen3.8-27B-GGUF}"
IMAGE_MAX_TOKENS="${IMAGE_MAX_TOKENS:-1024}"
ONEBIT_CTX="${ONEBIT_CTX:-98304}"
ONEBIT_QUANT="${ONEBIT_QUANT:-UD-IQ1_S}"
KV_STAGES="${KV_STAGES:-f16,q8_0,q4_0}"
KV_MARGIN="${KV_MARGIN:-4096}"
KV_TYPE="${KV_TYPE:-q4_0}"
CACHE_RAM_USER="${CACHE_RAM:-}"
CACHE_RAM="${CACHE_RAM:-16384}"
LLAMA_SERVER="${LLAMA_SERVER:-${SCRIPT_DIR}/llama.cpp/build/bin/llama-server}"

[[ "${DYNAMIC}" -eq 1 ]] || KV_STAGES=""

if [[ ! -x "${LLAMA_SERVER}" ]]; then
    echo "!! ${LLAMA_SERVER} not found, run ./build.sh first" >&2
    exit 1
fi

export LLAMA_CACHE="${MODEL_DIR}"
export LD_LIBRARY_PATH="$(dirname "${LLAMA_SERVER}"):${LD_LIBRARY_PATH:-}"
mkdir -p "${MODEL_DIR}"

free_vram_mib() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1
}

MMPROJ_ARGS=(--no-mmproj)
if [[ "${VISION}" -eq 1 ]]; then
    ALIAS="${ALIAS:-qwen3.8-27b-vision}"
    MMPROJ="$(find "${MODEL_DIR}" -name 'mmproj-*Q8_0.gguf' -print -quit 2>/dev/null || true)"
    [[ -n "${MMPROJ}" ]] || MMPROJ="$(find "${MODEL_DIR}" -name 'mmproj-*.gguf' -print -quit 2>/dev/null || true)"
    if [[ -z "${MMPROJ}" ]]; then
        echo "!! --vision needs an mmproj file in ${MODEL_DIR}, run ./install.sh first" >&2
        exit 1
    fi
    echo "==> mmproj: ${MMPROJ}"
    MMPROJ_ARGS=(--mmproj "${MMPROJ}" --image-min-tokens "$((IMAGE_MAX_TOKENS < 1024 ? IMAGE_MAX_TOKENS : 1024))" --image-max-tokens "${IMAGE_MAX_TOKENS}")
    FREE_MIB="$(free_vram_mib)"
    if [[ -n "${FREE_MIB}" ]] && [[ "${FREE_MIB}" -lt 24000 ]]; then
        echo "!! WARNING: only ${FREE_MIB} MiB of free VRAM, vision at full context needs ~23.5 GB: close other GPU apps"
    fi
elif [[ "${ONEBIT}" -eq 1 ]]; then
    ALIAS="${ALIAS:-qwen3.8-27b-1bit}"
    CTX="${ONEBIT_CTX}"
    QUANT="${ONEBIT_QUANT}"
    SPEC_TYPE="none"
    BATCH="-b 512 -ub 128"
    CACHE_RAM="${CACHE_RAM_USER:-4096}"
    echo "==> 1-bit mode: ${QUANT}, ctx=${CTX}, no MTP"
    # Measured: model 5497 MiB + recurrent state 150 MiB + ~22.3 KiB per token (q4_0 KV + compute).
    NEED_MIB="$(awk "BEGIN{printf \"%.0f\", 5497 + 150 + (${CTX} * 22.3 / 1024)}")"
    TOTAL_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)"
    if [[ -n "${TOTAL_MIB}" ]] && [[ "${TOTAL_MIB}" -lt "${NEED_MIB}" ]]; then
        echo "!! the GPU has ${TOTAL_MIB} MiB of VRAM but ctx ${CTX} needs ~${NEED_MIB} MiB" >&2
        echo "!! lower it, e.g. ONEBIT_CTX=65536 ./serve.sh --1bit" >&2
        exit 1
    fi
    FREE_MIB="$(free_vram_mib)"
    if [[ -n "${FREE_MIB}" ]] && [[ "${FREE_MIB}" -lt "${NEED_MIB}" ]]; then
        echo "!! WARNING: only ${FREE_MIB} MiB of free VRAM, ~${NEED_MIB} needed: close other GPU apps"
    fi
else
    ALIAS="${ALIAS:-qwen3.8-27b}"
    echo "==> text-only mode, the '-vision' model in opencode needs ./serve.sh --vision"
fi

SPEC_ARGS=()
if [[ "${SPEC_TYPE}" != "none" ]]; then
    SPEC_ARGS=(--spec-type "${SPEC_TYPE}" --spec-draft-n-max "${N_MAX}" -ctkd q4_0 -ctvd q4_0)
fi

if [[ -n "${KV_STAGES}" ]]; then
    # The stage conversion messages are logged at level 4.
    KV_ARGS=(--kv-stages "${KV_STAGES}" --kv-requant-margin "${KV_MARGIN}" -lv "${LV:-4}")
    echo "==> dynamic KV quantization: ${KV_STAGES} (convert when < ${KV_MARGIN} free cells)"
else
    KV_ARGS=(-ctk "${KV_TYPE}" -ctv "${KV_TYPE}")
    [[ -z "${LV:-}" ]] || KV_ARGS+=(-lv "${LV}")
    echo "==> static KV cache: ${KV_TYPE}"
fi

echo "==> Qwen3.8-27B: ctx=${CTX}, quant=${QUANT}, speculative=${SPEC_TYPE}"
echo "==> API: http://${HOST}:${PORT}/v1 (model id: ${ALIAS})"

exec "${LLAMA_SERVER}" \
    -hf "${HF_REPO}:${QUANT}" \
    --alias "${ALIAS}" \
    --host "${HOST}" \
    --port "${PORT}" \
    -c "${CTX}" \
    "${KV_ARGS[@]}" \
    ${SPEC_ARGS[@]+"${SPEC_ARGS[@]}"} \
    -fa on \
    -ngl 99 \
    -np 1 \
    ${BATCH} \
    --reasoning auto \
    --kv-unified \
    --metrics \
    --cache-ram "${CACHE_RAM}" \
    "${MMPROJ_ARGS[@]}"
