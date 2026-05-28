#!/usr/bin/env python3
"""Quick CLI test: load model, run a prompt, print tokens."""
from __future__ import annotations

import argparse
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="QuenStar CLI — test model inference")
    parser.add_argument("-m", "--model", required=True, help="Path to GGUF model")
    parser.add_argument("-p", "--prompt", default="Say hello in one sentence.")
    parser.add_argument("--ctx", type=int, default=2048, help="Context size")
    parser.add_argument("--n-gpu-layers", type=int, default=-1, help="GPU layers (-1=all)")
    parser.add_argument("--temp", type=float, default=0.8, help="Temperature")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens")
    parser.add_argument("--offload-kqv", type=int, default=0, help="1=KV on GPU, 0=KV in RAM")
    args = parser.parse_args()

    import os

    print(f"Loading model: {args.model}")
    print(f"  ctx={args.ctx}, gpu_layers={args.n_gpu_layers}, offload_kqv={'GPU' if args.offload_kqv else 'RAM'}")

    # Check file exists and is valid
    if not os.path.isfile(args.model):
        print(f"ERROR: File not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    size_mb = os.path.getsize(args.model) / (1024**2)
    print(f"  file size: {size_mb:.0f} MB")

    with open(args.model, "rb") as f:
        magic = f.read(4)
    if magic != b"GGUF":
        print(f"ERROR: Not a GGUF file (magic: {magic!r})", file=sys.stderr)
        sys.exit(1)
    print("  GGUF: OK")

    # Try loading
    import llama_cpp

    gpu_ok = llama_cpp.llama_supports_gpu_offload()
    print(f"  GPU offload support: {gpu_ok}")

    t0 = time.time()
    try:
        llm = llama_cpp.Llama(
            model_path=args.model,
            n_gpu_layers=args.n_gpu_layers,
            n_ctx=args.ctx,
            offload_kqv=bool(args.offload_kqv),
            verbose=False,
            seed=42,
            n_batch=512,
        )
    except Exception as exc:
        print(f"ERROR loading model: {exc}", file=sys.stderr)
        sys.exit(1)

    load_time = time.time() - t0
    try:
        model_gb = llm.model_size() / (1024**3)
    except AttributeError:
        model_gb = size_mb / 1024
    print(f"  loaded in {load_time:.1f}s, model size: {model_gb:.1f} GB")
    print()

    # Inference
    print(f"User: {args.prompt}")
    print("Assistant: ", end="", flush=True)

    t0 = time.time()
    first_token = True
    total_tokens = 0

    try:
        stream = llm.create_completion(
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temp,
            stream=True,
        )
        for chunk in stream:
            text = chunk["choices"][0]["text"]
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
    print(f"---")
    print(f"  tokens: {total_tokens}")
    print(f"  ttft: {ttft:.2f}s")
    print(f"  total: {total_time:.2f}s")
    if total_tokens > 1:
        print(f"  tok/s: {(total_tokens - 1) / (total_time - ttft):.1f}")


if __name__ == "__main__":
    main()
