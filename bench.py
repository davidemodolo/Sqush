#!/usr/bin/env python3
"""Benchmark QuantStar prefill and decode speed at various context lengths.

Usage:
    python bench.py                          # default: 24GB profile, bitsandbytes
    python bench.py --vram 8 --quantized models/Qwen3.5-9B-quantized
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from typing import Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantstar.config import load_config, VramTier
from quantstar.download import download_model
from quantstar.engine import InferenceEngine


def _build_prompt(tokenizer, target_tokens: int) -> list[dict]:
    question = "\n\nWhat is 2+2? Reply with just the number."
    filler_words = max(0, target_tokens - 40)
    content = ("test " * filler_words) + question
    return [{"role": "user", "content": content}]


def bench_context(
    engine: InferenceEngine,
    target_tokens: int,
    decode_tokens: int = 50,
) -> Optional[dict]:
    tokenizer = engine.tokenizer
    msgs = _build_prompt(tokenizer, target_tokens)
    input_ids, _, _, _ = engine._tokenize(msgs, enable_thinking=False)
    actual = input_ids.shape[1]

    engine.reset_session()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    try:
        kwargs, gen_input = engine._prepare_generation(input_ids, decode_tokens)
    except torch.cuda.OutOfMemoryError:
        print(f"  OOM during prefill at {actual:,} tokens")
        return None
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    try:
        with torch.no_grad():
            outputs = engine.model.generate(gen_input, **kwargs)
    except torch.cuda.OutOfMemoryError:
        print(f"  OOM during decode at {actual:,} tokens")
        return None
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    decode_s = time.perf_counter() - t0

    n_generated = outputs.shape[1] - actual
    decode_tps = n_generated / decode_s if decode_s > 0 else 0
    peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0

    return {
        "context": actual,
        "prefill_s": prefill_s,
        "decode_tok_s": decode_tps,
        "generated": n_generated,
        "vram_peak_gb": peak_gb,
    }


def main():
    parser = argparse.ArgumentParser(description="QuantStar Benchmark")
    parser.add_argument("--vram", type=int, default=None, help="VRAM budget in GB")
    parser.add_argument("--max-context", type=int, default=None, help="Override max context length")
    args = parser.parse_args()

    config = load_config(vram_gb=args.vram)

    print("=" * 60)
    print("  QuantStar Benchmark")
    if args.vram:
        print(f"  VRAM profile: {args.vram} GB")
    print("=" * 60)

    print("\n[1/2] Loading model …")

    from quantstar.quantize import load_and_quantize_model
    model_path = download_model(config.model.repo, config.model.cache_dir)
    model, tokenizer, processor, cache_config = load_and_quantize_model(
        model_path=model_path,
        attn_implementation=config.model.attn_implementation,
        torch_dtype_str=config.model.torch_dtype,
        quantize_embeddings=(config.vram_tier == VramTier.LOW),
    )

    max_ctx = args.max_context or config.inference.max_context
    engine = InferenceEngine(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        cache_config=cache_config,
        max_context=max_ctx,
        max_new_tokens=config.inference.max_new_tokens,
        temperature=config.inference.temperature,
        top_p=config.inference.top_p,
        top_k=config.inference.top_k,
        presence_penalty=config.inference.presence_penalty,
    )

    print("\n[2/2] Warming up (triton autotune) …")
    wm = _build_prompt(tokenizer, 100)
    engine.chat_completion_sync(wm, max_tokens=8, enable_thinking=False)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    print("  done.\n")

    targets = [3_000, 16_000, 32_000, 64_000, 128_000]
    if args.max_context is None:
        targets.append(256_000)
    decode_tokens = 50

    header = f"{'Target':>8}  {'Actual':>7}  {'Prefill':>9}  {'Decode':>9}  {'Peak VRAM':>10}"
    print(header)
    print("-" * len(header))

    for target in targets:
        result = bench_context(engine, target, decode_tokens)
        if result is None:
            print(f"{target:>8,}  {'OOM':>7}")
            continue
        print(
            f"{target:>8,}  "
            f"{result['context']:>7,}  "
            f"{result['prefill_s']:>8.1f}s  "
            f"{result['decode_tok_s']:>7.1f}t/s  "
            f"{result['vram_peak_gb']:>9.1f} GB"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
