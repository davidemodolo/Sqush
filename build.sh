#!/usr/bin/env bash
set -euo pipefail

# Builds llama.cpp/build/bin/llama-server: fetches llama.cpp at LLAMA_COMMIT, applies
# patches/*.patch and compiles for the local GPU. Needs git, curl and nvcc;
# cmake and ninja are downloaded into tools/ when missing.
#   JOBS=$(nproc)  CUDA_ARCH=<detected>  LLAMA_SRC=./llama.cpp  LLAMA_REPO  LLAMA_COMMIT  CMAKE_BIN  NINJA_BIN

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="${LLAMA_SRC:-${SCRIPT_DIR}/llama.cpp}"
PATCH_DIR="${SCRIPT_DIR}/patches"
LLAMA_REPO="${LLAMA_REPO:-https://github.com/ggml-org/llama.cpp}"
LLAMA_COMMIT="${LLAMA_COMMIT:-50f068ffffc3e0e4c9c2e4139281c6075224f429}"
JOBS="${JOBS:-$(nproc)}"

CMAKE_VERSION="4.4.3"
NINJA_VERSION="1.13.2"
TOOLS_DIR="${SCRIPT_DIR}/tools"
CMAKE_PORTABLE="${TOOLS_DIR}/cmake-${CMAKE_VERSION}-linux-x86_64/bin/cmake"
NINJA_PORTABLE="${TOOLS_DIR}/ninja"

# Order: explicit override, copy in tools/, system binary, download into tools/.
if [[ -n "${CMAKE_BIN:-}" ]]; then
    CMAKE="${CMAKE_BIN}"
elif [[ -x "${CMAKE_PORTABLE}" ]]; then
    CMAKE="${CMAKE_PORTABLE}"
elif command -v cmake >/dev/null 2>&1; then
    CMAKE="$(command -v cmake)"
else
    echo "==> cmake not found, downloading portable cmake ${CMAKE_VERSION} into tools/"
    mkdir -p "${TOOLS_DIR}"
    curl -fsSL "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz" \
        | tar -xz -C "${TOOLS_DIR}"
    CMAKE="${CMAKE_PORTABLE}"
fi

if [[ -n "${NINJA_BIN:-}" ]]; then
    NINJA="${NINJA_BIN}"
elif [[ -x "${NINJA_PORTABLE}" ]]; then
    NINJA="${NINJA_PORTABLE}"
elif command -v ninja >/dev/null 2>&1; then
    NINJA="$(command -v ninja)"
else
    echo "==> ninja not found, downloading portable ninja ${NINJA_VERSION} into tools/"
    mkdir -p "${TOOLS_DIR}"
    curl -fsSL -o "${TOOLS_DIR}/ninja.zip" "https://github.com/ninja-build/ninja/releases/download/v${NINJA_VERSION}/ninja-linux.zip"
    python3 -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extract('ninja', sys.argv[2])" "${TOOLS_DIR}/ninja.zip" "${TOOLS_DIR}"
    rm -f "${TOOLS_DIR}/ninja.zip"
    chmod +x "${NINJA_PORTABLE}"
    NINJA="${NINJA_PORTABLE}"
fi

if ! command -v nvcc >/dev/null 2>&1 && [[ ! -x /usr/local/cuda/bin/nvcc ]]; then
    echo "!! nvcc not found, install the CUDA toolkit (e.g. 'sudo apt install nvidia-cuda-toolkit')" >&2
    exit 1
fi

if [[ -z "${CUDA_ARCH:-}" ]]; then
    CUDA_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')"
    CUDA_ARCH="${CUDA_ARCH:-86}"
fi

# Also resumes a first fetch that was interrupted.
if ! git -C "${SRC_DIR}" rev-parse -q --verify HEAD >/dev/null 2>&1; then
    echo "==> fetching llama.cpp ${LLAMA_COMMIT:0:7} into ${SRC_DIR}"
    mkdir -p "${SRC_DIR}"
    git -C "${SRC_DIR}" init -q
    git -C "${SRC_DIR}" remote get-url origin >/dev/null 2>&1 || git -C "${SRC_DIR}" remote add origin "${LLAMA_REPO}"
    git -C "${SRC_DIR}" fetch -q --depth 1 origin "${LLAMA_COMMIT}"
    git -C "${SRC_DIR}" checkout -q FETCH_HEAD
fi

HEAD_COMMIT="$(git -C "${SRC_DIR}" rev-parse HEAD)"
if [[ "${HEAD_COMMIT}" != "${LLAMA_COMMIT}" ]]; then
    echo "!! ${SRC_DIR} is at ${HEAD_COMMIT:0:7}, but the patches are made for ${LLAMA_COMMIT:0:7}" >&2
    echo "!! check out ${LLAMA_COMMIT:0:7} there, or remove the directory to fetch it again" >&2
    exit 1
fi

for patch in "${PATCH_DIR}"/*.patch; do
    name="$(basename "${patch}")"
    if git -C "${SRC_DIR}" apply --reverse --check "${patch}" >/dev/null 2>&1; then
        echo "==> ${name} already applied"
    elif git -C "${SRC_DIR}" apply --check "${patch}" >/dev/null 2>&1; then
        echo "==> applying ${name}"
        git -C "${SRC_DIR}" apply "${patch}"
    else
        echo "!! ${name} does not apply: ${SRC_DIR} has local changes that conflict with it" >&2
        echo "!! to keep them, update the patches (see README); to drop them: git -C ${SRC_DIR} checkout -- ." >&2
        exit 1
    fi
done

# A shallow clone has no build number, so the web UI assets would be fetched for "b1".
if [[ -z "${HF_UI_VERSION:-}" ]]; then
    HF_UI_VERSION="$(git ls-remote --tags "${LLAMA_REPO}" 2>/dev/null | awk -v c="${LLAMA_COMMIT}" '$1 == c {sub("refs/tags/", "", $2); print $2}' || true)"
fi
[[ -z "${HF_UI_VERSION}" ]] || export HF_UI_VERSION
rm -rf "${SRC_DIR}/build/tools/ui/dist"

echo "==> configuring (cuda arch ${CUDA_ARCH}, FA all quants)"
"${CMAKE}" -S "${SRC_DIR}" -B "${SRC_DIR}/build" -G Ninja \
    -DCMAKE_MAKE_PROGRAM="${NINJA}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_TESTS=OFF

echo "==> building llama-server (${JOBS} jobs)"
"${CMAKE}" --build "${SRC_DIR}/build" --target llama-server -j "${JOBS}"

echo "==> done:"
"${SRC_DIR}/build/bin/llama-server" --version | head -2
echo "start it with ./serve.sh"