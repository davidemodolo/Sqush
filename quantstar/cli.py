from __future__ import annotations

import logging

from .config import QuantStarConfig
from .engine import InferenceEngine

log = logging.getLogger(__name__)

_MAX_HISTORY = 40  # non-system messages to keep (20 turns)


def _trim_history(messages: list[dict]) -> list[dict]:
    system = [m for m in messages if m["role"] == "system"]
    non_system = [m for m in messages if m["role"] != "system"]
    if len(non_system) > _MAX_HISTORY:
        non_system = non_system[-_MAX_HISTORY:]
    return system + non_system


def run_cli(engine: InferenceEngine, config: QuantStarConfig):
    from rich.console import Console

    console = Console()
    messages: list[dict[str, str]] = []

    console.print("[bold cyan]QuantStar[/] — [dim]Qwen3.6-27B quantized[/]")
    console.print(f"  weight bits: {config.quantization.weight_bits}")
    console.print(f"  KV cache bits: {config.quantization.kv_cache_bits}")
    console.print(f"  max context: {config.inference.max_context}")
    vram = engine.get_vram_info()
    if vram["cuda_available"]:
        console.print(f"  VRAM: {vram['allocated_gb']:.1f}/{vram['total_gb']:.0f} GB allocated")
    console.print()
    console.print("  Commands: [bold]/quit[/], [bold]/clear[/], [bold]/system <msg>[/], [bold]/vram[/]")
    console.print()

    while True:
        try:
            user_input = console.input("[bold green]>>>[/] ")
        except (EOFError, KeyboardInterrupt):
            console.print("\nGoodbye!")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        if user_input == "/quit":
            break
        elif user_input == "/clear":
            messages = []
            console.print("[dim]Conversation cleared.[/]")
            continue
        elif user_input == "/vram":
            vram = engine.get_vram_info()
            if vram["cuda_available"]:
                console.print(f"  VRAM: {vram['allocated_gb']:.1f} GB allocated / {vram['total_gb']:.0f} GB total")
            continue
        elif user_input.startswith("/system "):
            system_text = user_input[8:]
            messages = [m for m in messages if m["role"] != "system"]
            messages.insert(0, {"role": "system", "content": system_text})
            console.print(f"[dim]System prompt set.[/]")
            continue

        messages.append({"role": "user", "content": user_input})
        messages = _trim_history(messages)

        console.print()
        raw = "".join(engine.chat_completion_stream(messages))

        think_start = raw.find("<think>")
        if think_start != -1:
            think_end = raw.find("</think>", think_start)
            if think_end != -1:
                raw = raw[think_end + len("</think>"):].strip()

        if raw:
            console.print(raw)
            messages.append({"role": "assistant", "content": raw})
        console.print()
