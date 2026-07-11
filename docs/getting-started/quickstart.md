# Quickstart

```bash
./run.sh download    # download the model (one-time)
./run.sh serve       # bakes on first run, then starts the API on 127.0.0.1:9898
```

The first `serve` (or `chat`) **bakes** the model — it quantizes to a compact on‑disk checkpoint and deletes the bulky raw download. For the 27B this needs ~70 GB free transiently and settles at ~18 GB. See [Baking](../concepts/baking.md). You can also bake explicitly first:

```bash
./run.sh bake
```

## Test the server

```bash
curl http://127.0.0.1:9898/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3.6-27B","messages":[{"role":"user","content":"Hi!"}]}'
```

Streaming:

```bash
curl http://127.0.0.1:9898/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3.6-27B","stream":true,"messages":[{"role":"user","content":"Explain KV cache quantization."}]}'
```

## Other commands

```bash
./run.sh chat         # interactive Rich CLI
./run.sh info         # print resolved config + VRAM
./run.sh init         # register Sqush as a provider in OpenCode
./run.sh --vram 8 serve   # force the 8 GB profile regardless of detected VRAM
```

Every subcommand forwards to `python -m sqush <cmd>`, so `./run.sh serve` and `python -m sqush serve` are equivalent.

## Use with an OpenAI client

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:9898/v1", api_key="local")
resp = client.chat.completions.create(
    model="Qwen/Qwen3.6-27B",
    messages=[{"role": "user", "content": "Write a haiku about quantization."}],
)
print(resp.choices[0].message.content)
```

## Use with OpenCode

```bash
./run.sh init
```

This writes a `provider.sqush` entry to `~/.config/opencode/opencode.json` (provider only — no agent block). Then run `/models` in OpenCode and pick `sqush/qwen3.6-27b`. See [Server & API](../internals/server.md) for the manual JSON.
