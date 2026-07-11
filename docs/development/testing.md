# Testing

The suite runs **entirely without loading a model** — GPU‑dependent code is mocked, so tests run on CPU in a few seconds and are safe in CI.

```bash
pip install -e ".[test]"
python -m pytest
```

## Coverage

- **KV cache math** — int4 quantization, packed layout, round‑trip accuracy, group‑boundary edges, `SqushKVCache` layer structure.
- **Blockwise GQA attention** — first prefill, cached prefill with offset, decode step, causal masking, GQA grouping, numerical stability, `blockwise_attention_from_cache`.
- **Inference engine** — image extraction, message preprocessing, tokenization paths, `_prepare_generation` kwargs, session KV reuse (text + vision), stream and sync paths.
- **FastAPI server** — health/VRAM, models list, sync + streaming completions, SSE format, CORS, thinking/reasoning content, tool‑call parsing and streaming.
- **Config loading** — YAML overrides, env‑var precedence, VRAM tier classification, defaults.
- **CLI** — history trimming, `</think>` splitting.
- **`__main__`** — OpenCode config init and path resolution, tier‑aware bake dispatch (LOW side‑car vs HIGH NF4).
- **Download** — checkpoint completeness detection (resume partial downloads, reject corrupt/partial snapshots).
- **Bake safetensors** — side‑car embedding bake, visual‑encoder passthrough, `skip_modules` patching.

## Mocking conventions

Streaming tests use a `FakeStreamer` implementing `__iter__` and a no‑op `end()` (mirroring `TextIteratorStreamer`, which the engine now calls in a `finally`). Model/tokenizer/processor are `MagicMock`s; `transformers.*` and `_print_memory_usage` are patched where the bake path is exercised.

## End‑to‑end smoke test

Against a live server (requires a GPU and the downloaded model):

```bash
./test_sqush.sh
```

It starts the server, exercises streaming and non‑streaming requests and concurrent load, and verifies no `<think>` tags leak into content fields.
