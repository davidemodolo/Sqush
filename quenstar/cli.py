from __future__ import annotations

import logging
import sys
from typing import Optional

from .config import QuenStarConfig
from .engine import InferenceEngine

log = logging.getLogger(__name__)


def run_cli(engine: InferenceEngine, config: QuenStarConfig):
    from rich.console import Console
    from rich.markdown import Markdown

    console = Console()
    messages: list[dict[str, str]] = []

    console.print("[bold cyan]QuenStar[/] v2.0 — [dim]Qwen3.6-27B quantized[/]")
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

        console.print()
        response_text = ""
        for token in engine.chat_completion_stream(messages):
            response_text += token
            console.print(token, end="")
        console.print()
        console.print()

        if response_text.strip():
            messages.append({"role": "assistant", "content": response_text})
