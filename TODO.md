# TODO

Next steps for QuenStar, roughly ordered by priority.

## Core Features

- [x] **Per-token temperature 0 during tool calls** — ~~currently we force temp=0 for the *entire* generation when tools are present.~~ **Done.** `_do_manual_stream` uses `create_chat_completion(max_tokens=1)` for prefill then per-token `sample()`/`eval()` with `ToolCallDetector` switching temp=0 inside tool call syntax. Enabled via `tool_calling.manual_token_loop` config (defaults to false).

- [ ] **Non-streaming response path** — the streaming endpoint works, but the non-streaming path in `_non_stream_chat` receives a single dict from `create_chat_completion` and wraps it. Needs testing with opencode (which may prefer non-streaming in some modes).

- [x] **Exact DSML/text tool call replay** — ~~when a model generates a tool call, store the exact bytes it produced (in `ToolCallRegistry`). When the client sends the result back, replay those exact bytes instead of re-formatting. Prevents KV cache mismatch when clients normalize JSON keys.~~ **Done.** `toolcall.py:replay_tool_calls()` rewrites messages before inference, `server.py:_register_tool_content()` stores raw text after generation. `ToolCallRegistry` tracks by tool call ID with bound LRU trim.

- [x] **Temperature 0 during thinking/syntax detection** — ~~`Qwen3.6-35B-A3B` emits `<think>...</think>` blocks by default. We should suppress or control this per-request.~~ **Done.** `enable_thinking` parameter added to `ChatCompletionRequest` and passed through to llama-cpp-python's chat template. Set `"enable_thinking": false` in the API request body.

## Model Support

- [ ] **More model presets** — add `Qwen3-8B-Instruct` (tiny mode), `Gemma-3-12B`, `Llama-4-12B` to the mode system, each with tested quant sizes.

- [ ] **iMatrix-quantized GGUF** — unsloth's GGUFs don't use importance matrices. Switch to bartowski quants (`*IQ4_XS*.gguf`) which use imatrix for better quality at the same size. Update `run.sh` download URLs accordingly.

- [ ] **Refusal direction removal** — from the paper "Refusal in language models is mediated by a single direction": extract and subtract the refusal direction vector during inference to prevent the model from refusing tool calls. Requires activation steering at the llama.cpp level (modify hidden states between layers).

- [ ] **Model-agnostic chat template handling** — currently relies entirely on llama-cpp-python's chat template from the GGUF metadata. Some models may need manual template overrides for tool calling.

- [ ] **Download from HF mirrors** — the `run.sh` download uses `unsloth` repos which require authentication. Add fallback mirrors (`bartowski`, `lmstudio-community`, `ggml-org`) and `hf download` command support.

- [x] **Vision/image input (multimodal)** — **Done.** `Llava15ChatHandler` passes `clip_model_path` to Llama constructor via `chat_handler=`. llama-cpp-python 0.3.23 handles images through `libmtmd.so` internally. `create_chat_completion` processes `image_url` content parts automatically. `run.sh` auto-downloads `mmproj-F16.gguf` (899MB) from unsloth on desktop. Error responses: 400/501.

## Performance

- [ ] **Pre-fill chunk tuning** — the default `n_batch=4096` balances prefill speed vs VRAM. Expose as a config option and auto-tune per mode.

- [x] **YaRN RoPE scaling for >262K context** — ~~Qwen3.6 supports up to 1M tokens via YaRN position embedding extension (`rope_type: yarn`, `factor: 4.0`). Quality degrades past native 262K. llama.cpp supports this via `--yarn-ext-factor`. Expose in config.~~ **Done.** Added `yarn_ext_factor`, `yarn_attn_factor`, `yarn_beta_fast`, `yarn_beta_slow` to `ModelConfig` and passed to llama-cpp-python. Set `yarn_ext_factor: 4.0` in config.yaml to activate (default -1 = disabled).

- [ ] **Flash attention verification** — `flash_attn=True` is set but needs benchmarking to confirm it helps on the RTX 3090 (compute 8.6). If not, switch to standard attention.

- [x] **GPU memory monitoring** — ~~add a `/health/vram` endpoint showing actual GPU VRAM usage during inference.~~ **Done.** `server.py:/health/vram` queries `nvidia-smi` for per-GPU memory (total/used/free) and utilization.

## Server

- [ ] **Concurrent session support** — currently only one session at a time (the lock serializes all requests). For multi-client use, implement session switching via disk KV cache without the global lock.

- [x] **Better error responses** — ~~return proper HTTP error codes and JSON error bodies instead of crashing the SSE stream on failure.~~ **Done.** `chat_completions` wrapped in try/except → `JSONResponse(400/500, {"error":{...}})`. `_non_stream_chat` propagates engine errors. SSE error chunks now include error message and type.

- [x] **Tool call canonicalization** — ~~if exact replay fails, generate a deterministic DSML/JSON form from the normalized tool object. Compare with the live sampled token stream and rewrite if needed (DS4-style).~~ **Done.** `toolcall.py:canonicalize_tool_calls()` produces deterministic `<tool_call>{"name":"...","arguments":{...}}</tool_call>` with sorted keys, no spaces, UTF-8 safe. Falls back when `replay_tool_calls()` misses the registry.

- [ ] **`/v1/responses` endpoint** — OpenAI Responses API (used by Codex CLI).

- [ ] **Anthropic `/v1/messages` endpoint** — for Claude Code compatibility.

## UX

- [ ] **Server startup self-test** — after loading the model, run a quick inference test (one token) to verify everything works before accepting client requests.

- [ ] **Download progress with `hf` CLI** — `hf download` has built-in progress bars and resume support. Use it instead of `curl` when available.

- [ ] **Config wizard** — `./run.sh --setup` to interactively configure model path, context size, port, etc.
