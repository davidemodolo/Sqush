#!/usr/bin/env python3
"""CLI test tool: load model via Engine, run one-shot prompt or interactive chat."""
from __future__ import annotations

import argparse
import shutil
import sys
import time

from .config import QuenStarConfig
from .engine import Engine
from .types import ChatCompletionRequest


def main():
    args = _parse_args()
    config = _build_config(args)
    engine = Engine(config)

    if args.interactive:
        _run_interactive(engine, args.system)
    else:
        _run_one_shot(engine, args.prompt)


def run_interactive_chat(config: QuenStarConfig, system_prompt: str | None = None):
    engine = Engine(config)
    _chat_loop(engine, system_prompt)


def _parse_args():
    parser = argparse.ArgumentParser(description="QuenStar CLI — test model inference")
    parser.add_argument("-m", "--model", required=True, help="Path to GGUF model")
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive chat mode")
    parser.add_argument("-s", "--system", default=None, help="System prompt (interactive mode)")
    parser.add_argument("-p", "--prompt", default="Say hello in one sentence.")
    parser.add_argument("--ctx", type=int, default=None, help="Context size (default: from config)")
    parser.add_argument("--n-gpu-layers", type=int, default=-1, help="GPU layers (-1=all)")
    parser.add_argument("--temp", type=float, default=None, help="Temperature (default: from config)")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p (default: from config)")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k (default: from config)")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max tokens (default: from config)")
    parser.add_argument("--offload-kqv", type=int, default=0, help="1=KV on GPU, 0=KV in RAM")
    return parser.parse_args()


def _build_config(args):
    config = QuenStarConfig.load()
    config.model.path = args.model
    if args.ctx is not None:
        config.model.n_ctx = args.ctx
    config.model.n_gpu_layers = args.n_gpu_layers
    config.model.offload_kqv = bool(args.offload_kqv)
    if args.max_tokens is not None:
        config.generation.max_tokens = args.max_tokens
    if args.temp is not None:
        config.sampling.default_temperature = args.temp
    if args.top_p is not None:
        config.sampling.default_top_p = args.top_p
    if args.top_k is not None:
        config.sampling.default_top_k = args.top_k
    return config


def _run_one_shot(engine, prompt):
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    print(f"User: {prompt}")
    print("Assistant: ", end="", flush=True)

    t0 = time.time()
    first_token = True
    total_tokens = 0
    ttft = 0.0

    try:
        for chunk in engine.chat_completion(request):
            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            text = delta.get("content", "")
            if not text:
                continue
            if first_token:
                ttft = time.time() - t0
                first_token = False
            print(text, end="", flush=True)
            total_tokens += 1
    except Exception as exc:
        print(f"\nERROR during inference: {exc}", file=sys.stderr)
        sys.exit(1)

    total_time = time.time() - t0
    print()
    print()
    print("---")
    print(f"  tokens: {total_tokens}")
    print(f"  ttft: {ttft:.2f}s")
    print(f"  total: {total_time:.2f}s")
    if total_tokens > 1:
        print(f"  tok/s: {(total_tokens - 1) / (total_time - ttft):.1f}")


def _run_interactive(engine, system_prompt=None):
    _chat_loop(engine, system_prompt)


def _chat_loop(engine, system_prompt=None):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    terminal_width = shutil.get_terminal_size().columns
    separator = "\u2500" * terminal_width

    print("Interactive chat mode. Type your message and press Enter.")
    print("Commands: /quit, /exit, /clear, /system <prompt>")
    print(separator)

    try:
        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                messages = _handle_command(user_input, messages)
                continue

            messages.append({"role": "user", "content": user_input})

            print("Assistant: ", end="", flush=True)
            t0 = time.time()
            first_token = True
            total_tokens = 0
            ttft = 0.0
            full_response = ""

            try:
                request = ChatCompletionRequest(
                    messages=list(messages),
                    stream=True,
                )
                for chunk in engine.chat_completion(request):
                    try:
                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        text = delta.get("content", "")
                        if not text:
                            continue
                        if first_token:
                            ttft = time.time() - t0
                            first_token = False
                        print(text, end="", flush=True)
                        full_response += text
                        total_tokens += 1
                    except KeyboardInterrupt:
                        break
            except KeyboardInterrupt:
                pass
            except Exception as exc:
                print(f"\nERROR during inference: {exc}", file=sys.stderr)
                continue

            total_time = time.time() - t0
            print()

            if full_response:
                messages.append({"role": "assistant", "content": full_response})

            tok_s = (total_tokens - 1) / (total_time - ttft) if total_tokens > 1 else 0
            print(f"  [{total_tokens} tok, ttft {ttft:.1f}s, {tok_s:.0f} tok/s]")

    except KeyboardInterrupt:
        print("\nGoodbye!")


def _handle_command(cmd, messages):
    parts = cmd.split(maxsplit=1)
    command = parts[0].lower()

    if command in ("/quit", "/exit", "/q"):
        print("Goodbye!")
        sys.exit(0)

    elif command == "/clear":
        messages[:] = [m for m in messages if m["role"] == "system"]
        print("[Conversation cleared (system prompt kept)]")

    elif command == "/system":
        if len(parts) > 1:
            new_system = parts[1]
            messages[:] = [m for m in messages if m["role"] != "system"]
            messages.insert(0, {"role": "system", "content": new_system})
            truncated = new_system[:80] + "..." if len(new_system) > 80 else new_system
            print(f"[System prompt set: {truncated}]")
        else:
            print("[Usage: /system <prompt>]")

    else:
        print(f"[Unknown command: {command}]")

    return messages


if __name__ == "__main__":
    main()
