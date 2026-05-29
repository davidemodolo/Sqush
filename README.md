# QuenStar

Local LLM inference server optimized for running large models on limited VRAM.
Inspired by [DwarfStar (DS4)](https://github.com/antirez/ds4), it offloads the KV cache
to system RAM so your GPU can dedicate its full memory to model weights.
Built for opencode and any OpenAI-compatible client.

**Key features:**
- **Model on GPU, context in RAM** — `offload_kqv=False` keeps KV cache in system memory
- **DS4-style disk KV cache** — sessions persisted, resumed without re-prefill
- **Per-token temperature 0 during tool calls** — greedy only for `<tool_call>` syntax, normal sampling for payloads
- **Tool call replay & canonicalization** — exact raw text replay prevents KV cache mismatch; deterministic fallback form
- **Auto-detects GPU VRAM** — picks the right model and context for your hardware
- **OpenAI-compatible API** — `/v1/models`, `/v1/chat/completions` with SSE streaming
- **Single-session design** — one live KV cache at a time, disk for everything else

## Quick Start

### 1. Setup

```bash
git clone <repo> && cd QuenStar
./run.sh --install-deps          # creates venv, installs CUDA + llava if possible
```

On first run, `run.sh --install-deps`:
- Detects GPU VRAM (auto-selects desktop/laptop mode)
- Installs cmake, dependencies, and llama-cpp-python
- **Desktop mode**: auto-runs `sudo apt install nvidia-cuda-toolkit` to get `nvcc`, then builds from source with `-DGGML_CUDA=on -DGGML_LLAVA=on` (enables vision/image input)
- **Laptop mode**: installs pre-built CUDA wheel (no vision — 8GB VRAM too tight)
- Falls back gracefully if CUDA toolkit isn't available

### 2. Authenticate with HuggingFace

```bash
pip install huggingface_hub
hf auth login
```

### 3. Start

```bash
./run.sh                          # auto-detects GPU, downloads model, starts server
```

That's it. `./run.sh` with no arguments:
- Detects your GPU VRAM via `nvidia-smi`
- Downloads the right model (35B MoE for 24GB, 14B for 8GB)
- Starts the server on `http://127.0.0.1:8080`

| VRAM | Mode | Model | Size | Context |
|------|------|-------|------|---------|
| 24 GB | `desktop` | Qwen3.6-35B-A3B Q4_K_M | 22 GB | 262K (1M with YaRN) |
| 8 GB | `laptop` | Qwen3-14B-Instruct IQ4_XS | ~7.5 GB | 64K |

### 4. Configure opencode

`./run.sh` will auto-detect your opencode config and offer to set this up interactively. Or add manually to `~/.config/opencode/opencode.json`:

```json
{
  "provider": {
    "quenstar": {
      "name": "QuenStar (local)",
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://127.0.0.1:8080/v1",
        "apiKey": "local"
      },
      "models": {
        "qwen3.6-35b": {
          "name": "Qwen3.6 35B (local)",
          "limit": { "context": 1048576, "output": 32768 }
        }
      }
    }
  },
  "agent": {
    "quenstar": {
      "description": "Local QuenStar",
      "model": "quenstar/qwen3.6-35b",
      "temperature": 0
    }
  }
}
```

Then `opencode --agent quenstar`.

## Interactive CLI

In addition to the server, QuenStar has a built-in interactive chat mode for
talking to the model directly in the terminal.

```bash
# Using the config-aware entry point (needs CUDA libs)
LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v12 python -m quenstar --chat

# Standalone CLI with interactive mode
python -m quenstar.cli -m ./models/qwen3.6-35b-a3b-ud-q4_k_m.gguf -i

# With a system prompt
python -m quenstar.cli -m ./models/qwen3.6-35b-a3b-ud-q4_k_m.gguf -i -s "You are a helpful assistant."
```

Once loaded, you'll see:

```
Interactive chat mode. Type your message and press Enter.
Commands: /quit, /exit, /clear, /system <prompt>
─────────────────────────────────────────────────────────────

You: Write a Python hello world
Assistant: Here's a simple Python hello world:
...streaming tokens...
  [120 tok, ttft 0.3s, 45 tok/s]

You:
```

### Commands

| Command | Action |
|---------|--------|
| `/quit`, `/exit`, `/q` | Exit the chat |
| `/clear` | Clear conversation (keeps system prompt) |
| `/system <prompt>` | Set or change the system prompt |

Press `Ctrl+C` during a response to interrupt generation without exiting.
Press `Ctrl+C` on an empty prompt to exit.

### CLI flags

```
python -m quenstar.cli --help
  -m, --model PATH       Path to GGUF model file (required)
  -i, --interactive      Interactive chat mode
  -s, --system PROMPT    System prompt (interactive mode only)
  -p, --prompt TEXT      One-shot prompt (default: "Say hello in one sentence.")
  --ctx N                Context size (default: from config.yaml)
  --n-gpu-layers N       GPU layers, -1 = all (default: -1)
  --temp F               Temperature (default: from config.yaml)
  --top-p F              Top-p sampling (default: from config.yaml)
  --top-k N              Top-k sampling (default: from config.yaml)
  --max-tokens N         Max tokens to generate (default: from config.yaml)
  --offload-kqv 0|1      KV cache: 0=RAM, 1=GPU (default: 0)
```

## run.sh Usage

```
./run.sh [OPTIONS]

With no arguments: detects GPU, downloads model, starts server.

  -m, --model PATH      Path to GGUF model file
  --mode MODE           Force desktop|laptop (auto-detected by default)
  --download VARIANT    Pick a specific quantization
  --hf-token TOKEN      HuggingFace token
  --ctx N               Context window size (default varies by mode)
  --port N              Server port (default: 8080)
  --kv-dir PATH         Disk KV cache directory (default: ./kvcache)
  --kv-space-mb N       Max disk space for KV cache in MB (default: 8192)
  --no-offload-kqv      Keep KV cache in system RAM instead of GPU VRAM
  --install-deps        Install Python dependencies into a venv
  --cors                Enable CORS headers
  --trace               Enable trace logging
  -h, --help            Show this help
```

## Models

### Desktop (24GB VRAM)
**Qwen3.6-35B-A3B** — 35B MoE, 3B active per token, fast inference.
[unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF)

| Quant | Size | VRAM |
|-------|------|------|
| **Q4_K_M** | 22.1 GB | 24 GB (default) |
| Q4_K_S | 20.9 GB | 24 GB |
| IQ4_XS | 17.7 GB | 24 GB |

### Laptop (8GB VRAM)
**Qwen3-14B-Instruct** — 14B dense, solid coding/tool-calling.
[unsloth/Qwen3-14B-Instruct-GGUF](https://huggingface.co/unsloth/Qwen3-14B-Instruct-GGUF)

| Quant | Size | VRAM |
|-------|------|------|
| **IQ4_XS** | ~7.5 GB | 8 GB (default) |
| Q4_K_M | ~8.7 GB | 8 GB (tight) |
| Q3_K_M | ~6.5 GB | 8 GB |

Max context: 262K native / ~1M with YaRN (desktop). 64K laptop.

### YaRN RoPE Scaling (up to 1M tokens)

Qwen3.6 supports up to ~1M context via YaRN. Enabled by default in desktop mode:
- `n_ctx: 262144` (native max)
- `yarn_ext_factor: 4.0` (4× native = ~1M)

Quality is native up to 262K; scaling activates past the trained range. Set `yarn_ext_factor: -1` to disable.

### Vision Input

Qwen3.6-35B-A3B is a vision-language model (custom ViT encoder). Vision works out of the box:

1. `run.sh` auto-downloads `mmproj-F16.gguf` (899MB) from unsloth on desktop mode
2. The server loads it via `Llava15ChatHandler` and passes it to llama-cpp-python
3. `create_chat_completion` handles `image_url` content parts automatically

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-35b",
    "messages": [{"role":"user","content":[
      {"type":"image_url","image_url":{"url":"https://example.com/photo.jpg"}},
      {"type":"text","text":"Describe this image"}
    ]}]
  }'
```

```yaml
model:
  mmproj_path: "./models/mmproj-qwen3.6.gguf"
```

## Config Reference

`config.yaml` (all options):

```yaml
model:
  path: "./models/qwen3.6-35b-a3b-ud-q4_k_m.gguf"
  n_gpu_layers: -1          # -1 = all layers on GPU
  n_ctx: 131072             # context window
  offload_kqv: false        # KV cache in system RAM, not GPU
  flash_attn: true
  use_mmap: true
  # YaRN (optional, for >262K context):
  yarn_ext_factor: -1.0     # -1 = disabled, 4.0 = ~1M
  yarn_attn_factor: 1.0
  yarn_beta_fast: 32.0
  yarn_beta_slow: 1.0
  # Vision (optional, requires cmake build):
  mmproj_path: ""

server:
  host: "127.0.0.1"
  port: 8080
  cors: false

kv_cache:
  dir: "./kvcache"
  space_mb: 8192
  eviction_half_life_hours: 6.0

sampling:
  default_temperature: 0.8
  default_top_p: 0.9
  default_top_k: 40
  default_min_p: 0.05
  default_repeat_penalty: 1.0

tool_calling:
  enabled: true
  manual_token_loop: false  # false=off, true=per-token temp=0 for tool syntax
  exact_replay_cache_size: 100000

generation:
  max_tokens: 32768

logging:
  level: "INFO"
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│  QuenStar Server (Python/FastAPI)               │
│                                                 │
│  ┌───────────────┐    ┌──────────────────────┐  │
│  │  OpenAI API   │    │  Session Manager     │  │
│  │  /v1/models   │    │  single live session │  │
│  │  /v1/chat     │    │  save/load/resume    │  │
│  └───────┬───────┘    └────────┬─────────────┘  │
│          │                     │                │
│  ┌───────┴─────────────────────┴─────────────┐  │
│  │  Inference Engine (llama-cpp-python)      │  │
│  │  Model on GPU VRAM, KV cache in sys RAM   │  │
│  │  Per-token temp=0 for tool call syntax    │  │
│  │  ToolCallDetector + ToolCallRegistry      │  │
│  └───────────────────────────────────────────┘  │
│                                                 │
│  ┌───────────────────────────────────────────┐  │
│  │  Disk KV Cache (./kvcache/)               │  │
│  │  SHA1-keyed .kv files, LRU eviction       │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

### Per-Token Temperature 0 During Tool Calls

When tools are present, the engine can use a manual per-token loop: `ToolCallDetector` tracks whether the model is inside `<tool_call>...</tool_call>` syntax and switches to `temp=0` only for those tokens. Normal sampling is used for payload text. Controlled via config:

```yaml
# config.yaml
tool_calling:
  manual_token_loop: true   # true = per-token greedy inside tool syntax, false = entire generation (default)
```

When disabled (default), the old behavior applies: temp=0 for the entire generation via `create_chat_completion`.

### Tool Call Replay & Canonicalization

When a model generates a tool call, the raw text is stored in `ToolCallRegistry` (keyed by tool call ID). On the next turn, if the client sends back a normalized tool result, the exact raw text is replayed into the assistant message — preventing KV cache mismatch from JSON key reordering or whitespace changes.

If exact replay misses (cache evicted or first turn), `canonicalize_tool_calls()` generates a deterministic `<tool_call>` form with sorted keys and no whitespace.

### Thinking Suppression

Qwen3.6 emits `<think>...</think>` blocks by default. Suppress them by passing `enable_thinking: false` in the request body:

```json
{
  "model": "qwen3.6-35b",
  "messages": [{"role": "user", "content": "Hello"}],
  "enable_thinking": false
}
```

### Session Resume

Sessions are persisted to disk automatically. When a request matches a cached
session, the llama.cpp state is restored — **no re-prefill needed**. For consecutive
turns in the same conversation, llama.cpp's internal prefix matching skips the
shared prefix automatically.

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/models` | List models |
| GET | `/v1/models/{id}` | Model info |
| POST | `/v1/chat/completions` | Chat completions (SSE streaming) |
| GET | `/health` | Server health + KV cache stats |
| GET | `/health/vram` | GPU VRAM usage (via nvidia-smi) |
| GET | `/sessions` | List saved disk sessions |

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-35b",
    "messages": [{"role": "user", "content": "Write a Python hello world"}],
    "stream": true
  }'
```

## Running Tests

```bash
source .venv/bin/activate

# All tests (needs GPU + downloaded model)
python -m pytest tests/ -v

# Skip GPU-only tests
python -m pytest tests/ -v -m "not slow"

# Run specific test
python -m pytest tests/test_smoke.py::TestServer::test_chat_stream -v
```

Tests cover:

| Test | Checks |
|------|--------|
| `test_cli_loads_and_generates` | Model loads, produces tokens |
| `test_cli_rejects_bad_file` | Invalid GGUF → exit non-zero |
| `test_cli_rejects_missing_file` | Missing file → exit non-zero |
| `test_interactive_chat_generates_and_quits` | Interactive mode streams response, quits cleanly |
| `test_interactive_rejects_bad_file` | Interactive mode: invalid GGUF → exit non-zero |
| `test_interactive_rejects_missing_file` | Interactive mode: missing file → exit non-zero |
| `test_health_endpoint` | `/health` returns ok |
| `test_models_endpoint` | `/v1/models` lists models |
| `test_chat_non_stream` | Non-streaming completion |
| `test_chat_stream` | SSE streaming with content tokens |
| `test_sessions_endpoint` | `/sessions` returns session list |

## License

MIT
