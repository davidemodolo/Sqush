# TODO

Next steps for QuenStar, roughly ordered by priority.

## Core Features

- [ ] **Per-token temperature 0 during tool calls** — currently we force temp=0 for the *entire* generation when tools are present. DS4 does better: it uses greedy decoding only for tool call *syntax* (tags, JSON structure) while keeping normal sampling for *payloads* (code, file contents). Requires implementing the manual token generation loop from `engine.py` with the `ToolCallDetector` state machine from `toolcall.py`.

- [ ] **Non-streaming response path** — the streaming endpoint works, but the non-streaming path in `_non_stream_chat` receives a single dict from `create_chat_completion` and wraps it. Needs testing with opencode (which may prefer non-streaming in some modes).

- [ ] **Exact DSML/text tool call replay** — when a model generates a tool call, store the exact bytes it produced (in `ToolCallRegistry`). When the client sends the result back, replay those exact bytes instead of re-formatting. Prevents KV cache mismatch when clients normalize JSON keys.

- [ ] **Temperature 0 during thinking/syntax detection** — `Qwen3.6-35B-A3B` emits `<think>...</think>` blocks by default. We should suppress or control this per-request. The model's chat template may accept a `thinking: false` parameter or an `enable_thinking` flag.

## Model Support

- [ ] **More model presets** — add `Qwen3-8B-Instruct` (tiny mode), `Gemma-3-12B`, `Llama-4-12B` to the mode system, each with tested quant sizes.

- [ ] **Model-agnostic chat template handling** — currently relies entirely on llama-cpp-python's chat template from the GGUF metadata. Some models may need manual template overrides for tool calling.

- [ ] **Download from HF mirrors** — the `run.sh` download uses `unsloth` repos which require authentication. Add fallback mirrors (`bartowski`, `lmstudio-community`, `ggml-org`) and `hf download` command support.

## Performance

- [ ] **Pre-fill chunk tuning** — the default `n_batch=4096` balances prefill speed vs VRAM. Expose as a config option and auto-tune per mode.

- [ ] **Flash attention verification** — `flash_attn=True` is set but needs benchmarking to confirm it helps on the RTX 3090 (compute 8.6). If not, switch to standard attention.

- [ ] **GPU memory monitoring** — add a `/health/vram` endpoint showing actual GPU VRAM usage during inference.

## Server

- [ ] **Concurrent session support** — currently only one session at a time (the lock serializes all requests). For multi-client use, implement session switching via disk KV cache without the global lock.

- [ ] **Better error responses** — return proper HTTP error codes and JSON error bodies instead of crashing the SSE stream on failure.

- [ ] **Tool call canonicalization** — if exact replay fails, generate a deterministic DSML/JSON form from the normalized tool object. Compare with the live sampled token stream and rewrite if needed (DS4-style).

- [ ] **`/v1/responses` endpoint** — OpenAI Responses API (used by Codex CLI).

- [ ] **Anthropic `/v1/messages` endpoint** — for Claude Code compatibility.

## UX

- [ ] **Server startup self-test** — after loading the model, run a quick inference test (one token) to verify everything works before accepting client requests.

- [ ] **Download progress with `hf` CLI** — `hf download` has built-in progress bars and resume support. Use it instead of `curl` when available.

- [ ] **Config wizard** — `./run.sh --setup` to interactively configure model path, context size, port, etc.
