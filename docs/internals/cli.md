# Interactive CLI

`sqush/cli.py` (`run_cli`) is a Rich‑based REPL launched by `python -m sqush chat`. It drives the same `InferenceEngine` as the server, so session KV reuse applies.

## Commands

| Command | Effect |
|---------|--------|
| `/quit` | Exit (also on Ctrl‑D / Ctrl‑C) |
| `/clear` | Reset the conversation |
| `/system <msg>` | Set/replace the system prompt (inserted at index 0) |
| `/vram` | Print VRAM usage (used / total / tensors / reserved) |

Any other input is a user message.

## Loop

1. Append the user message, then `_trim_history` — keep all system messages plus the last `_MAX_HISTORY = 40` non‑system messages (~20 turns).
2. Stream the reply by joining `engine.chat_completion_stream(messages)`.
3. Split the raw output on `</think>` (the template pre‑opens the block, so there's no leading `<think>`): the part before is `reasoning`, the part after is `content`.
4. Print the content and append the assistant message to history.

## Reasoning is kept in history

The assistant message stores `reasoning_content` when present. This is deliberate: session KV reuse requires the re‑rendered prompt to match the cached tokens exactly, and dropping the reasoning would shift tokens and force a full re‑prefill on the next turn. See [Session KV reuse](../concepts/session-kv-reuse.md).
