"""QuenStar v2 — Qwen3.6-27B quantized inference server.

Usage:
    python -m quenstar download           # Download the model
    python -m quenstar serve              # Start OpenAI-compatible server
    python -m quenstar chat               # Interactive chat
    python -m quenstar info               # Show config and VRAM info
"""

from __future__ import annotations

import argparse
import logging
import sys


def main():
    parser = argparse.ArgumentParser(description="QuenStar v2 — Qwen3.6-27B quantized inference")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("download", help="Download the model from HuggingFace")
    sub.add_parser("serve", help="Start the OpenAI-compatible server")
    sub.add_parser("chat", help="Start interactive chat")
    sub.add_parser("info", help="Show configuration")

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
        print(f"Server: {config.server.host}:{config.server.port}")

    elif args.command in ("serve", "chat"):
        from .download import download_model
        model_path = download_model(config.model.repo, config.model.cache_dir)

        from .quantize import load_and_quantize_model
        model, tokenizer, cache_config = load_and_quantize_model(
            model_path=model_path,
            weight_bits=config.quantization.weight_bits,
            kv_cache_bits=config.quantization.kv_cache_bits,
            turbo=config.quantization.turbo,
            attn_implementation=config.model.attn_implementation,
            torch_dtype_str=config.model.torch_dtype,
        )

        from .engine import InferenceEngine
        engine = InferenceEngine(
            model=model,
            tokenizer=tokenizer,
            cache_config=cache_config,
            max_context=config.inference.max_context,
            max_new_tokens=config.inference.max_new_tokens,
            temperature=config.inference.temperature,
            top_p=config.inference.top_p,
            top_k=config.inference.top_k,
            presence_penalty=config.inference.presence_penalty,
        )

        if args.command == "serve":
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
