#!/usr/bin/env bash
set -euo pipefail

# Downloads the model, builds the patched llama-server and registers the models in opencode.
# Asks for each setting with a detected suggestion; set it in the environment to skip the question
# (non-interactive runs take the suggestion). Saves the settings serve.sh needs in config.env.
#
#   VARIANT=full|1bit  MODEL_DIR=./models  PORT=8080
#   OPENCODE_CONFIG=~/.config/opencode/opencode.json (comma-separated list allowed, "none" to skip)
#   HF_REPO=unsloth/Qwen3.8-27B-GGUF  QUANT=UD-IQ4_XS  ONEBIT_QUANT=UD-IQ1_S  MMPROJ_URL=<Q8_0 projector>  SKIP_BUILD=1
#   PROVIDER=sqush  PROVIDER_NAME=Sqush  CTX=262144  OUT=65536  ONEBIT_CTX=98304  ONEBIT_OUT=16384

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config.env"

ask() {
    local var="$1" prompt="$2" suggestion="$3" reply=""
    [[ -z "${!var:-}" ]] || return 0
    if [[ -t 0 ]]; then
        read -rp "${prompt} [${suggestion}]: " reply
    fi
    printf -v "${var}" '%s' "${reply:-${suggestion}}"
}

# Server settings given in the environment, saved so that serve.sh uses the same ones.
PASSED_SETTINGS=()
for key in HF_REPO QUANT ONEBIT_QUANT CTX ONEBIT_CTX; do
    [[ -z "${!key:-}" ]] || PASSED_SETTINGS+=("${key}")
done

config_get() {
    local value=""
    [[ ! -f "${CONFIG_FILE}" ]] || value="$(sed -n "s/^$1=//p" "${CONFIG_FILE}" | tail -1)"
    echo "${value:-$2}"
}

config_set() {
    touch "${CONFIG_FILE}"
    { grep -v "^$1=" "${CONFIG_FILE}" || true; printf '%s=%s\n' "$1" "$2"; } > "${CONFIG_FILE}.tmp"
    mv "${CONFIG_FILE}.tmp" "${CONFIG_FILE}"
}

VRAM_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || true)"
SUGGESTED_VARIANT="full"
if [[ -n "${VRAM_MIB}" ]] && [[ "${VRAM_MIB}" -lt 20000 ]]; then
    SUGGESTED_VARIANT="1bit"
fi
SUGGESTED_VARIANT="$(config_get VARIANT "${SUGGESTED_VARIANT}")"
echo "==> GPU memory: ${VRAM_MIB:-unknown} MiB"
ask VARIANT "Model variant: full (24 GB GPU) or 1bit (~8 GB GPU)" "${SUGGESTED_VARIANT}"
ask MODEL_DIR "Model directory (~20 GB)" "$(config_get MODEL_DIR "${SCRIPT_DIR}/models")"
ask PORT "API port" "$(config_get PORT 8080)"
ask OPENCODE_CONFIG "opencode config to register the models in" "${HOME}/.config/opencode/opencode.json"

case "${VARIANT}" in
    full|1bit) ;;
    *) echo "!! VARIANT must be full or 1bit" >&2; exit 1 ;;
esac

HF_REPO="${HF_REPO:-unsloth/Qwen3.8-27B-GGUF}"
QUANT="${QUANT:-UD-IQ4_XS}"
ONEBIT_QUANT="${ONEBIT_QUANT:-UD-IQ1_S}"
MMPROJ_URL="${MMPROJ_URL:-https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF/resolve/main/mmproj-Qwen3.8-27B-Q8_0.gguf}"
SKIP_BUILD="${SKIP_BUILD:-0}"
PROVIDER="${PROVIDER:-sqush}"
PROVIDER_NAME="${PROVIDER_NAME:-Sqush}"
CTX="${CTX:-262144}"
OUT="${OUT:-65536}"
ONEBIT_CTX="${ONEBIT_CTX:-98304}"
ONEBIT_OUT="${ONEBIT_OUT:-16384}"
MODEL_DIR="$(mkdir -p "${MODEL_DIR}" && cd "${MODEL_DIR}" && pwd)"

for key in VARIANT MODEL_DIR PORT PROVIDER ${PASSED_SETTINGS[@]+"${PASSED_SETTINGS[@]}"}; do
    config_set "${key}" "${!key}"
done
echo "==> saved ${CONFIG_FILE}"

# The llama.app CLI is only used to download: its prebuilt server lacks sm_86 kernels for vision.
if ! command -v llama >/dev/null 2>&1; then
    echo "==> installing the llama.app CLI (model downloads)"
    curl -LsSf https://llama.app/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi

export LLAMA_CACHE="${MODEL_DIR}"
if [[ "${VARIANT}" == "1bit" ]]; then
    echo "==> downloading ${HF_REPO}:${ONEBIT_QUANT} into ${MODEL_DIR}"
    llama download --no-mmproj -hf "${HF_REPO}:${ONEBIT_QUANT}"
else
    echo "==> downloading ${HF_REPO}:${QUANT} into ${MODEL_DIR}"
    llama download --no-mmproj -hf "${HF_REPO}:${QUANT}"
    # The model repo only ships BF16/F16 projectors; Q8_0 keeps vision within 24 GB at full context.
    MMPROJ_FILE="${MODEL_DIR}/$(basename "${MMPROJ_URL}")"
    if [[ ! -s "${MMPROJ_FILE}" ]]; then
        echo "==> downloading the vision projector into ${MMPROJ_FILE}"
        curl -fL --retry 3 -o "${MMPROJ_FILE}.part" "${MMPROJ_URL}"
        mv "${MMPROJ_FILE}.part" "${MMPROJ_FILE}"
    fi
fi

if [[ "${SKIP_BUILD}" == "1" ]]; then
    echo "==> SKIP_BUILD=1, run ./build.sh before ./serve.sh"
elif ! "${SCRIPT_DIR}/build.sh"; then
    echo "!! build failed: fix the error above and run ./install.sh again" >&2
    exit 1
fi

register_models() {
    local cfg="$1" models
    mkdir -p "$(dirname "${cfg}")"
    [[ -s "${cfg}" ]] || echo '{"$schema": "https://opencode.ai/config.json"}' > "${cfg}"

    if [[ "${VARIANT}" == "1bit" ]]; then
        models='| .provider[$prov].models["qwen3.8-27b-1bit"] = entry("Qwen 3.8 27b 1-bit"; ["text"])'
    else
        models='| .provider[$prov].models["qwen3.8-27b"] = entry("Qwen 3.8 27b"; ["text"])
                | .provider[$prov].models["qwen3.8-27b-vision"] = entry("Qwen 3.8 27b Visual"; ["text", "image"])'
    fi
    if ! jq --arg prov "${PROVIDER}" --arg provname "${PROVIDER_NAME}" --arg base "${API_BASE}" \
       --argjson ctx "${MODEL_CTX}" --argjson out "${MODEL_OUT}" \
       'def entry($n; $m): {
            "name": $n, "reasoning": true, "tools": true,
            "modalities": { "input": $m, "output": ["text"] },
            "limit": { "context": $ctx, "output": $out }
        };
        .provider[$prov] = ((.provider[$prov] // { "name": $provname, "npm": "@ai-sdk/openai-compatible" })
                            | .options.baseURL = $base)
        '"${models}" "${cfg}" > "${cfg}.tmp"; then
        rm -f "${cfg}.tmp"
        echo "!! could not update ${cfg} (not plain JSON?), add the provider by hand as in opencode.json" >&2
        return 1
    fi
    mv "${cfg}.tmp" "${cfg}"
    echo "==> registered the models in ${cfg}"
}

MODEL_CTX="${CTX}"
MODEL_OUT="${OUT}"
if [[ "${VARIANT}" == "1bit" ]]; then
    MODEL_CTX="${ONEBIT_CTX}"
    MODEL_OUT="${ONEBIT_OUT}"
fi
API_BASE="http://127.0.0.1:${PORT}/v1"

if [[ "${OPENCODE_CONFIG}" == "none" ]]; then
    echo "==> opencode registration skipped"
elif ! command -v jq >/dev/null 2>&1; then
    echo "!! jq not found, opencode registration skipped (see opencode.json for an example)" >&2
else
    IFS=',' read -ra CFGS <<< "${OPENCODE_CONFIG}"
    for cfg in "${CFGS[@]}"; do
        register_models "${cfg/#\~/${HOME}}" || REGISTER_FAILED=1
    done
fi

echo
if [[ "${VARIANT}" == "1bit" ]]; then
    echo "Done. Start the server with ./serve.sh (opencode model: ${PROVIDER}/qwen3.8-27b-1bit)"
else
    echo "Done. Start the server with ./serve.sh, or ./serve.sh --vision for images"
    echo "opencode models: ${PROVIDER}/qwen3.8-27b, ${PROVIDER}/qwen3.8-27b-vision"
fi

[[ -z "${REGISTER_FAILED:-}" ]] || exit 1
