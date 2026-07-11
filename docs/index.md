# Sqush

> **Sq**ushed **Qu**wen Under **S**mall **H**ardware — a recursive acronym, like Wine.

Sqush runs large Qwen models **quantized on a single consumer GPU**: Qwen3.6‑27B in 24 GB, Qwen3.5‑9B in 8 GB, both with **4‑bit weights, a 4‑bit KV cache, and the full 256k context window**. It exposes a drop‑in **OpenAI‑compatible API**, supports **text + image** input, streams tokens over SSE with `reasoning_content` and `tool_calls` deltas, and runs entirely local and private.

Inspired by antirez's [DwarfStar4](https://github.com/antirez/dwarfstar4).

## What makes it fit

Three techniques combine to squeeze a 27B‑parameter multimodal model — and a quarter‑million‑token context — into 24 GB:

| Technique | What it does | Where |
|-----------|--------------|-------|
| **NF4 weight quantization** | 4‑bit NormalFloat weights via bitsandbytes; embeddings and `lm_head` get a custom 4‑bit path | [Quantization](internals/quantization.md) |
| **int4 append‑only KV cache** | keys/values stored at 4 bits, quantized once per 64‑token group and never re‑quantized | [KV cache](internals/kv-cache.md) |
| **Blockwise GQA attention** | an online‑softmax attention that dequantizes the KV one 1024‑token block at a time, so the full cache is never materialized | [Attention](internals/attention.md) |

## Feature tour

- **Full 256k context** on consumer hardware — custom blockwise GQA keeps the KV cache at the native 4 KV heads.
- **4‑bit everything** — NF4 weights + int4 append‑only KV cache (never re‑quantizes previous tokens).
- **OpenAI‑compatible API** — point any OpenAI client, agent framework, or tool at `base_url`.
- **Streaming (SSE)** — real token streaming with correct `reasoning_content` and incremental `tool_calls` deltas.
- **Tool calling** — the model emits `<tool_call>` XML; the server parses it incrementally into OpenAI‑format deltas.
- **Image input** — the Qwen3.6‑VL vision encoder processes `image_url` content parts (base64 data URLs).
- **Session KV reuse** — appending to a conversation skips prefill of the shared prefix.
- **Interactive CLI** — a Rich‑based chat with `/clear`, `/vram`, `/system` commands.

## Where to start

<div class="grid cards" markdown>

- :material-rocket-launch: **[Quickstart](getting-started/quickstart.md)** — one command to serve.
- :material-cog: **[Configuration](getting-started/configuration.md)** — `config.yaml`, env vars, VRAM tiers.
- :material-fire: **[Baking](concepts/baking.md)** — how the compact on‑disk model is produced.
- :material-chip: **[Internals](internals/architecture.md)** — quantization, KV cache, attention, engine, server.

</div>

!!! note "Naming"
    The Python package is `sqush` (version 2.0.0). The project was renamed from *QuantStar*; you may still see the old name in a git remote URL or cache directory. The GitHub repository is [`davidemodolo/Sqush`](https://github.com/davidemodolo/Sqush).
