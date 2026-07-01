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
    _torch.cuda.empty_cache()
    log.info("Warmup complete")


def _cooked_model_path(raw_path: str) -> str:
    return raw_path.rstrip("/").rstrip("\\") + "-cooked"


def _bake_safetensors(raw_path: str, cooked_path: str, log) -> None:
    """Process raw model safetensors directly, quantizing visual encoder to NF4.

    Never loads the full model into GPU — processes one tensor at a time.
    GPU peak is < 100 MB (one Linear4bit layer briefly for NF4 quantization).
    """
    import json
    import shutil

    import torch as _torch
    from safetensors.torch import load_file as _sf_load, save_file as _sf_save

    os.makedirs(cooked_path, exist_ok=True)

    # Copy all non-shard config/tokenizer/processor files.
    _shard_suffixes = (".safetensors",)
    _skip = {"model.safetensors.index.json"}
    for fname in os.listdir(raw_path):
        if fname in _skip:
            continue
        src_path = os.path.join(raw_path, fname)
        if os.path.isdir(src_path):
            continue
        if any(fname.endswith(s) for s in _shard_suffixes):
            continue
        shutil.copy2(src_path, os.path.join(cooked_path, fname))

    # Patch config.json: remove visual encoder module names from llm_int8_skip_modules.
    config_dst = os.path.join(cooked_path, "config.json")
    if os.path.exists(config_dst):
        with open(config_dst) as _f:
            cfg = json.load(_f)
        qc = cfg.get("quantization_config", {})
        skip_mods = qc.get("llm_int8_skip_modules", [])
        # Drop the visual encoder so bitsandbytes quantizes it at load time.
        # lm_head must STAY in the skip list: on a pre-quantized checkpoint,
        # from_pretrained expects Linear4bit weights to be packed 4-bit in the
        # shard — without the skip entry it loads the raw bf16 weight into a
        # Linear4bit with no quant_state and crashes on the first forward.
        # lm_head is instead NF4-quantized post-load (_quantize_lm_head).
        _visual_prefixes = {"visual", "vision_model", "vision_tower", "img_processor"}
        skip_mods = [m for m in skip_mods if m not in _visual_prefixes]
        qc["llm_int8_skip_modules"] = skip_mods
        cfg["quantization_config"] = qc
        with open(config_dst, "w") as _f:
            json.dump(cfg, _f, indent=2)
            _f.write("\n")

    # Locate shard files.
    index_path = os.path.join(raw_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as _f:
            index = json.load(_f)
        weight_map = index["weight_map"]
        shard_files = sorted(set(weight_map.values()))
    else:
        shard_files = ["model.safetensors"]
        weight_map = None

    _EMBED_KEY = "model.language_model.embed_tokens.weight"
    _EMBED_GROUP_SIZE = 128  # must match quantize.py

    new_weight_map: dict = {}
    embed_quant: dict = {}

    for shard_fname in shard_files:
        src = os.path.join(raw_path, shard_fname)
        dst = os.path.join(cooked_path, shard_fname)
        log.info("  Processing shard %s …", shard_fname)

        tensors = _sf_load(src, device="cpu")
        out: dict[str, _torch.Tensor] = {}

        for key, tensor in tensors.items():
            is_embed_tokens = (
                key == _EMBED_KEY
                and tensor.ndim == 2
                and tensor.dtype in (_torch.bfloat16, _torch.float16)
            )
            if is_embed_tokens:
                # Quantize embed_tokens to 4-bit asymmetric format (same math as
                # _quantize_embeddings in quantize.py) entirely on CPU — no GPU needed.
                # The result is saved as a side-car file; the key is intentionally
                # omitted from the cooked shard so from_pretrained never allocates the
                # 1.93 GB bfloat16 tensor on GPU, preventing CUDA allocator fragmentation.
                out_f, in_f = tensor.shape
                w = tensor.to(_torch.float32)
                num_groups = (in_f + _EMBED_GROUP_SIZE - 1) // _EMBED_GROUP_SIZE
                pad = num_groups * _EMBED_GROUP_SIZE - in_f
                if pad:
                    w = _torch.nn.functional.pad(w, (0, pad))
                w_f = w.reshape(out_f, num_groups, _EMBED_GROUP_SIZE)
                w_min = w_f.amin(-1)
                w_max = w_f.amax(-1)
                scale = (w_max - w_min).clamp(min=1e-9) / 15.0
                zp = (-w_min / scale).round().clamp(0, 15).to(_torch.int32)
                q = ((w_f / scale.unsqueeze(-1)).round() + zp.unsqueeze(-1)).clamp(0, 15).to(_torch.int32)
                gs = _EMBED_GROUP_SIZE
                q = _torch.nn.functional.pad(q, (0, (8 - gs % 8) % 8))
                q = q.reshape(out_f, num_groups, -1, 8)
                packed = _torch.zeros(out_f, num_groups, q.shape[2], dtype=_torch.int32)
                for i in range(8):
                    packed |= (q[..., i] & 0xF) << (i * 4)
                embed_quant = {
                    "_qw":     packed,
                    "_sc":     scale.to(_torch.bfloat16),
                    "_zp":     zp,
                    "_vocab":  _torch.tensor([out_f]),
                    "_hidden": _torch.tensor([in_f]),
                }
                del w, w_f, w_min, w_max, scale, zp, q, packed
                log.info(
                    "  Quantized embed_tokens to 4-bit (%d × %d) — replaced by side-car",
                    out_f, in_f,
                )
                # Save a tiny placeholder so transformers doesn't flag the key as
                # MISSING (which would print a noisy LOAD REPORT). The placeholder
                # shape deliberately differs from the model's expected [vocab, hidden]
                # so transformers skips it (via ignore_mismatched_sizes=True) rather
                # than loading random data into GPU. All actual embedding data comes
                # from quantized_embeddings.safetensors via _load_pre_baked_embeddings.
                out[key] = _torch.zeros(1, in_f, dtype=tensor.dtype)
                if new_weight_map is not None:
                    new_weight_map[key] = shard_fname
            else:
                out[key] = tensor
                if new_weight_map is not None:
                    new_weight_map[key] = shard_fname

        _sf_save(out, dst)
        del tensors, out

    # Save pre-quantized embedding side-car and flag config so serving code can
    # skip the 1.93 GB GPU allocation for embed_tokens during from_pretrained.
    if embed_quant:
        _sf_save(embed_quant, os.path.join(cooked_path, "quantized_embeddings.safetensors"))
        log.info("  Saved quantized_embeddings.safetensors (%.0f MB)", embed_quant["_qw"].nbytes / 1e6)
        with open(config_dst) as _f:
            cfg = json.load(_f)
        cfg["qs_pre_baked_embeddings"] = True
        with open(config_dst, "w") as _f:
            json.dump(cfg, _f, indent=2)
            _f.write("\n")

    # Write updated index if the raw model was sharded.
    if weight_map is not None:
        new_index = dict(index)
        new_index["weight_map"] = new_weight_map
        with open(os.path.join(cooked_path, "model.safetensors.index.json"), "w") as _f:
            json.dump(new_index, _f, indent=2)
            _f.write("\n")


def _bake_model(model_path: str, config, log) -> str:
    """Process raw model: quantize embeddings to 4-bit side-car, save cooked copy, delete raw.

    One-time operation. GPU peak < 100 MB. Returns the path to the saved cooked model.
    """
    import shutil

    cooked_path = _cooked_model_path(model_path)

    log.info("First-time bake: processing model (this runs once) …")
    log.info("  raw    → %s", model_path)
    log.info("  cooked → %s", cooked_path)

    _bake_safetensors(model_path, cooked_path, log)

    log.info("Deleting raw model at %s", model_path)
    shutil.rmtree(model_path)

    log.info("Bake complete.")
    return cooked_path


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
            },
            "qwen3.5-9b": {
                "name": "Qwen3.5 9B 4-bit (local, 8GB)",
                "reasoning": True,
                "tools": True,
                "modalities": {
                    "input": ["text", "image"],
                    "output": ["text"],
                },
                "limit": {"context": 131072, "output": 32768},
            },
        },
    }

    cfg.setdefault("agent", {})
    cfg["agent"]["quantstar"] = {
        "description": "Local QuantStar — Qwen3.6 27B 4-bit",
        "model": "quantstar/qwen3.6-27b",
        "temperature": 0,
    }
    cfg["agent"]["quantstar-8gb"] = {
        "description": "Local QuantStar — Qwen3.5 9B 4-bit (8GB tier)",
        "model": "quantstar/qwen3.5-9b",
        "temperature": 0,
    }

    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")

    print(f"  Provider: quantstar")
    print(f"  Base URL: http://{config.server.host}:{config.server.port}/v1")
    print(f"  Agents:   quantstar → qwen3.6-27b (24GB), quantstar-8gb → qwen3.5-9b (8GB)")
    print()
    print("Run '/models' in OpenCode and select a quantstar model to use it.")


def main():
    parser = argparse.ArgumentParser(description="QuantStar — Qwen3.6-27B quantized inference")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("download", help="Download the model from HuggingFace")
    sub.add_parser("bake", help="Quantize visual encoder and save cooked model (8GB tier only)")
    sub.add_parser("serve", help="Start the OpenAI-compatible server")
    sub.add_parser("chat", help="Start interactive chat")
    sub.add_parser("info", help="Show configuration")
    sub.add_parser("init", help="Register QuantStar in OpenCode config")

    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--log-level", default=None, help="Logging level")
    parser.add_argument("--vram", default=None, type=int, help="VRAM budget in GB (auto-detected if not set)")

    args = parser.parse_args()

    from .config import load_config, VramTier
    config = load_config(args.config, vram_gb=args.vram)

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
    logging.getLogger("accelerate.big_modeling").setLevel(logging.ERROR)

    log = logging.getLogger(__name__)

    if args.command == "download":
        from .download import download_model
        download_model(config.model.repo, config.model.cache_dir)

    elif args.command == "bake":
        from .download import download_model
        raw_path = download_model(config.model.repo, config.model.cache_dir)
        cooked = _cooked_model_path(raw_path)
        if os.path.exists(cooked):
            log.info("Cooked model already exists at %s — nothing to do.", cooked)
        elif config.vram_tier != VramTier.LOW:
            log.info("Bake is only needed for the 8 GB (LOW) tier — skipping.")
        else:
            _bake_model(raw_path, config, log)

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
        from pathlib import Path

        is_low = config.vram_tier == VramTier.LOW

        if is_low:
            raw_path = str(
                Path(config.model.cache_dir).resolve()
                / config.model.repo.replace("/", "__")
            )
            cooked = _cooked_model_path(raw_path)
            if os.path.exists(cooked):
                model_path = cooked
            else:
                model_path = download_model(config.model.repo, config.model.cache_dir)
                model_path = _bake_model(model_path, config, log)
        else:
            model_path = download_model(config.model.repo, config.model.cache_dir)

        from .quantize import load_and_quantize_model
        model, tokenizer, processor, cache_config = load_and_quantize_model(
            model_path=model_path,
            attn_implementation=config.model.attn_implementation,
            torch_dtype_str=config.model.torch_dtype,
            quantize_embeddings=is_low,
            quantize_vision_encoder=False,
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
            max_image_pixels=config.inference.max_image_pixels,
            min_image_pixels=config.inference.min_image_pixels,
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
