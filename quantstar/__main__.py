"""QuantStar — Qwen3.6-27B quantized inference server.

Usage:
    python -m quantstar download           # Download the model
    python -m quantstar serve              # Start OpenAI-compatible server
    python -m quantstar chat               # Interactive chat
    python -m quantstar info               # Show config and VRAM info
    python -m quantstar init               # Register QuantStar in OpenCode config
"""

from __future__ import annotations

import argparse
import json
import logging
import os


def _opencode_config_path() -> str:
    return os.path.expanduser("~/.config/opencode/opencode.json")


def _warmup_engine(engine) -> None:
    """Run dummy generations to autotune triton kernels before serving.

    Pre-warming the chunked-prefill path at common sequence lengths populates
    the triton kernel cache so subsequent requests skip autotuning (avoiding
    first-request latency). The full-prefill path (used by vision requests)
    is protected by the autotuner monkey-patch in quantize.py and does not
    need explicit warmup.
    """
    import torch as _torch

    log = logging.getLogger(__name__)
    log.info("Warming up model (autotuning triton kernels) …")

    for n_tokens in [128, 1024]:
        filler = "test " * max(1, n_tokens // 2)
        messages = [{"role": "user", "content": filler}]
        try:
            engine.reset_session()
            _torch.cuda.synchronize()
            text, _, _ = engine.chat_completion_sync(messages, max_tokens=1, enable_thinking=False)
            log.info("Warmup %d tokens: %r", n_tokens, text[:60])
        except Exception as exc:
            log.warning("Warmup %d tokens failed (non-fatal): %s", n_tokens, exc)

    engine.reset_session()
    _torch.cuda.synchronize()
    log.info("Warmup complete")


def _init_opencode(config) -> None:
    config_path = _opencode_config_path()
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        print(f"Updating existing OpenCode config: {config_path}")
    else:
        cfg = {}
        print(f"Creating OpenCode config: {config_path}")

    cfg.setdefault("$schema", "https://opencode.ai/config.json")
    cfg.setdefault("provider", {})

    cfg["provider"]["quantstar"] = {
        "name": "QuantStar (local)",
        "npm": "@ai-sdk/openai-compatible",
        "options": {
            "baseURL": f"http://{config.server.host}:{config.server.port}/v1",
            "apiKey": "local",
        },
        "models": {
            "qwen3.6-27b": {
                "name": "Qwen3.6 27B 4-bit (local)",
                "reasoning": True,
                "tools": True,
                "modalities": {
                    "input": ["text", "image"],
                    "output": ["text"],
                },
                "limit": {
                    "context": config.inference.max_context,
                    "output": config.inference.max_new_tokens,
                },
            }
        },
    }

    cfg.setdefault("agent", {})
    cfg["agent"]["quantstar"] = {
        "description": "Local QuantStar — Qwen3.6 27B 4-bit",
        "model": "quantstar/qwen3.6-27b",
        "temperature": 0,
    }

    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")

    print(f"  Provider: quantstar")
    print(f"  Base URL: http://{config.server.host}:{config.server.port}/v1")
    print(f"  Context:  {config.inference.max_context:,} tokens")
    print(f"  Output:   {config.inference.max_new_tokens:,} tokens")
    print(f"  Agent:    quantstar → quantstar/qwen3.6-27b")
    print()
    print("Run '/models' in OpenCode and select 'quantstar/qwen3.6-27b' to use it.")


def main():
    parser = argparse.ArgumentParser(description="QuantStar — Qwen3.6-27B quantized inference")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("download", help="Download the model from HuggingFace")
    sub.add_parser("serve", help="Start the OpenAI-compatible server")
    sub.add_parser("chat", help="Start interactive chat")
    sub.add_parser("info", help="Show configuration")
    sub.add_parser("init", help="Register QuantStar in OpenCode config")

    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--log-level", default=None, help="Logging level")

    args = parser.parse_args()

    from .config import load_config
    config = load_config(args.config)

    log_level = (args.log_level or config.logging.level).upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(levelname)s %(asctime)s %(name)s — %(message)s",
    )
    # Suppress known-harmless upstream warnings
    import warnings
    warnings.filterwarnings("ignore", message=".*tl.make_block_ptr is deprecated.*")
    logging.getLogger("transformers.models.qwen3_5.modeling_qwen3_5").setLevel(logging.ERROR)
    logging.getLogger("torch.utils._pytree").setLevel(logging.ERROR)

    if args.command == "download":
        from .download import download_model
        download_model(config.model.repo, config.model.cache_dir)

    elif args.command == "info":
        print(f"Model: {config.model.repo}")
        print(f"Cache dir: {config.model.cache_dir}")
        print(f"Attn: {config.model.attn_implementation}")
        print(f"Torch dtype: {config.model.torch_dtype}")
        print(f"Weight bits: {config.quantization.weight_bits}")
        print(f"KV cache bits: {config.quantization.kv_cache_bits}")
        print(f"Max context: {config.inference.max_context}")
        print(f"Max output:  {config.inference.max_new_tokens}")
        print(f"Server: {config.server.host}:{config.server.port}")

    elif args.command == "init":
        _init_opencode(config)

    elif args.command in ("serve", "chat"):
        from .download import download_model
        model_path = download_model(config.model.repo, config.model.cache_dir)

        from .quantize import load_and_quantize_model
        model, tokenizer, processor, cache_config = load_and_quantize_model(
            model_path=model_path,
            attn_implementation=config.model.attn_implementation,
            torch_dtype_str=config.model.torch_dtype,
        )

        from .engine import InferenceEngine
        engine = InferenceEngine(
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            cache_config=cache_config,
            max_context=config.inference.max_context,
            max_new_tokens=config.inference.max_new_tokens,
            temperature=config.inference.temperature,
            top_p=config.inference.top_p,
            top_k=config.inference.top_k,
            presence_penalty=config.inference.presence_penalty,
        )

        if args.command == "serve":
            _warmup_engine(engine)

            from .server import create_app
            import uvicorn

            app = create_app(engine, config)
            uvicorn.run(
                app,
                host=config.server.host,
                port=config.server.port,
                log_level=config.logging.level.lower(),
            )
        elif args.command == "chat":
            from .cli import run_cli
            run_cli(engine, config)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
