from __future__ import annotations

import argparse
import logging
import sys


def main():
    parser = argparse.ArgumentParser(
        description="QuenStar - Local LLM inference server with disk KV cache",
        prog="quenstar",
    )
    parser.add_argument(
        "-c", "--config",
        default=None,
        help="Path to config YAML file (default: config.yaml or ~/.config/quenstar/config.yaml)",
    )
    parser.add_argument(
        "-m", "--model",
        default=None,
        help="Path to GGUF model file (overrides config)",
    )
    parser.add_argument(
        "--ctx",
        type=int,
        default=None,
        help="Context window size in tokens (overrides config)",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Server host (overrides config)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Server port (overrides config)",
    )
    parser.add_argument(
        "--kv-dir",
        default=None,
        help="KV cache directory (overrides config)",
    )
    parser.add_argument(
        "--kv-space-mb",
        type=int,
        default=None,
        help="Max KV cache disk space in MB (overrides config)",
    )
    parser.add_argument(
        "--no-offload-kqv",
        action="store_true",
        default=None,
        help="Keep KV cache in system RAM instead of GPU VRAM",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        default=None,
        help="Enable trace logging for debugging",
    )
    parser.add_argument(
        "--cors",
        action="store_true",
        default=None,
        help="Enable CORS headers",
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        default=False,
        help="Run quick CLI inference test instead of server",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        default=False,
        help="Run interactive CLI chat mode instead of server",
    )

    args = parser.parse_args()

    from .config import QuenStarConfig

    config = QuenStarConfig.load(args.config)

    if args.model:
        config.model.path = args.model
    if args.ctx:
        config.model.n_ctx = args.ctx
    if args.host:
        config.server.host = args.host
    if args.port:
        config.server.port = args.port
    if args.kv_dir:
        config.kv_cache.dir = args.kv_dir
    if args.kv_space_mb:
        config.kv_cache.space_mb = args.kv_space_mb
    if args.no_offload_kqv is not None:
        config.model.offload_kqv = not args.no_offload_kqv
    if args.trace is not None:
        config.logging.trace = args.trace
    if args.cors:
        config.server.cors = True

    log_level = logging.DEBUG if config.logging.trace else config.logging.level.upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not config.model.path:
        print("Error: No model path specified. Use -m/--model or set QUENSTAR_MODEL_PATH.", file=sys.stderr)
        sys.exit(1)

    if args.cli:
        from .cli import main as cli_main
        sys.argv = ["quenstar-cli", "-m", config.model.path]
        cli_main()
        return

    if args.chat:
        from .cli import run_interactive_chat
        run_interactive_chat(config)
        return

    from .server import run_server

    run_server(config)


if __name__ == "__main__":
    main()
