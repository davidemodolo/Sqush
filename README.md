# Sqush

Runs Qwen3.8-27B (UD-IQ4_XS GGUF) with llama.cpp on a single 24 GB GPU (RTX 3090) at the full
262k context, exposed as an OpenAI-compatible API for opencode. It uses a patched llama.cpp with
**dynamic KV cache quantization**.

## Requirements

- Linux x86_64 with an NVIDIA GPU: 24 GB of VRAM, or ~8 GB for the 1-bit variant.
- NVIDIA driver and CUDA toolkit (`nvcc`, e.g. `sudo apt install nvidia-cuda-toolkit`).
- `git`, `curl`, `python3`, and `jq` to register the models in opencode.
- ~20 GB of disk for the model plus ~2 GB for the build. The prompt cache uses up to 16 GB of
  RAM (4 GB in 1-bit mode, see `CACHE_RAM`).

cmake and ninja are downloaded into `tools/` when not installed. `install.sh` also installs the
[llama.app](https://llama.app) CLI in `~/.local/bin`, used only to download the model. The Q8_0
vision projector comes from `ggml-org/Qwen3.8-27B-GGUF` (`MMPROJ_URL`).

## Setup

```bash
git clone https://github.com/davidemodolo/Sqush.git && cd Sqush
./install.sh
```

It asks for each setting with a suggestion detected from the machine (variant from the GPU memory,
model directory, port, opencode config) and saves the server settings in `config.env`. Set a
variable in the environment to skip its question, e.g. `VARIANT=1bit ./install.sh` for ~8 GB VRAM
laptops. Non-interactive runs take the suggestions. Re-run it anytime to update.

### Setup with a coding agent

To let a coding agent (opencode, Claude Code, Codex, ...) do the setup, give it this prompt:

```text
Set up Sqush (https://github.com/davidemodolo/Sqush) on this machine: a patched llama.cpp
server that runs Qwen3.8-27B locally as an OpenAI-compatible API.

1. Clone the repo (skip if already inside it) and read README.md before doing anything else.
2. Check the requirements: Linux x86_64, an NVIDIA GPU (nvidia-smi), the CUDA toolkit (nvcc),
   git, curl, python3, jq, ~25 GB of free disk. List what is missing and ask me before
   installing anything with sudo or the package manager.
3. Pick the variant from the GPU memory: VARIANT=full with ~24 GB of VRAM, VARIANT=1bit with
   ~8-20 GB. Below that, stop and tell me.
4. Run ./install.sh non-interactively with the variant set in the environment, e.g.
   VARIANT=full ./install.sh < /dev/null. It downloads ~20 GB and builds llama.cpp, so it can
   take a while; if it fails, read the error, fix the cause and run it again.
5. Start ./serve.sh in the background, wait until GET http://127.0.0.1:<PORT>/v1/models answers
   (PORT is in config.env), then send one short chat completion and check the reply.
6. Report the variant, the port, the opencode model ids and how to start and stop the server.

Do not set up remote access (remote/) unless I ask, and do not change the files in patches/.
```

## Run

```bash
./serve.sh            # text only
./serve.sh --vision   # + image input (~23.5 GB VRAM, close other GPU apps)
./serve.sh --1bit     # laptop mode (the default after VARIANT=1bit ./install.sh)
./serve.sh --kv-static  # plain q4_0 KV cache instead of dynamic quantization
KV_TYPE=q8_0 CTX=131072 ./serve.sh --kv-static  # another static KV type
```

API: `http://127.0.0.1:8080/v1` by default.
In opencode the models are `sqush/qwen3.8-27b`, `sqush/qwen3.8-27b-vision` (needs `--vision`) and
`sqush/qwen3.8-27b-1bit`; `opencode.json` is an example of the provider block `install.sh` writes.
All tuning knobs (context, ports, stages, cache sizes) are env vars listed at the top of `serve.sh`;
the environment overrides `config.env`.

## Dynamic KV cache quantization

The KV memory is sized for q4_0 at the full context. The cache starts at full precision and is
converted to a smaller type when it runs out of room:

| Stage | Holds up to (262k ctx) |
|-------|------------------------|
| f16   | ~74k tokens            |
| q8_0  | ~139k tokens           |
| q4_0  | 262k tokens            |

- Each conversion pauses generation once (~3 s for f16 → q8_0 at full size); it runs on the
  CPU, multi-threaded.
- When the cache empties out (new session, subagent) it goes back to f16. Tokens that were
  already converted keep the lower precision.
- Idle sessions are kept in a 16 GB RAM prompt cache (`CACHE_RAM`) together with their stage,
  so switching back to a session (e.g. main agent after a subagent) restores it in a second
  instead of reprocessing it.

## Patches

`build.sh` fetches llama.cpp `50f068f` into `llama.cpp/`, applies every file in `patches/` and builds
`llama.cpp/build/bin/llama-server` for the local GPU.

| Patch | Change |
|-------|--------|
| `dynamic-kv-quant.patch` | the dynamic KV cache quantization above |
| `grammar-repetition-limit.patch` | raises the grammar repetition limit from 2000 to 8192, so grammars with larger repetition counts are accepted |

To change them: edit the sources in `llama.cpp/`, then

```bash
git -C llama.cpp diff -- . ':(exclude)src/llama-grammar.cpp' > patches/dynamic-kv-quant.patch
git -C llama.cpp diff -- src/llama-grammar.cpp > patches/grammar-repetition-limit.patch
./build.sh
```

## Remote access

The setup above serves the model on this machine only. To run it as a headless server reachable
from other devices over Tailscale, see [remote/README.md](remote/README.md).

## Benchmarks

Run against a live server; the port comes from `config.env` (`BENCH_HOST` overrides the URL):

| Script | What it checks |
|--------|----------------|
| `bench/kv_stage_test.py` | prefill/decode speed over a long prompt |
| `bench/needle_test.py` | recall of a key at the start of a long prompt (`--reps 560` ≈ 200k tokens) |
| `bench/vision_needle.py` | recall of an image at the start of a long prompt (needs `--vision`, Pillow) |
| `bench/subagent_test.py` | main agent → subagent → main agent: f16 restore + RAM prompt cache |
| `bench/kv_quality.py` | how far the predictions drift from an f16 KV cache (KL divergence), per server configuration |
