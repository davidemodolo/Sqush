from __future__ import annotations

import gc
import logging
from typing import Optional

import torch

log = logging.getLogger(__name__)


def _ensure_cuda_libs():
    import os
    cu13_paths = [
        os.path.expanduser("~/.local/lib/python3.14/site-packages/nvidia/cu13/lib"),
        "/home/davide/Documents/Dev/genai_img_audio/.venv/lib/python3.14/site-packages/nvidia/cu13/lib",
        "/home/davide/Documents/Dev/alphamon/.venv/lib/python3.14/site-packages/nvidia/cu13/lib",
    ]
    for p in cu13_paths:
        if os.path.isdir(p) and p not in os.environ.get("LD_LIBRARY_PATH", ""):
            os.environ["LD_LIBRARY_PATH"] = p + ":" + os.environ.get("LD_LIBRARY_PATH", "")
            break


_ensure_cuda_libs()


def _patch_quantized_cache():
    """Monkey-patch QuantizedCache to support mixed layer types (DeltaNet + Attention).

    Qwen3.6 has 48 linear_attention (DeltaNet) layers + 16 full_attention layers.
    The stock QuantizedCache creates the same layer type for all layers, which breaks
    for the DeltaNet layers that need LinearAttentionCacheLayerMixin.

    This patch creates a hybrid QuantoQuantizedLinearAttentionLayer that inherits from
    both QuantoQuantizedLayer and LinearAttentionCacheLayerMixin, and modifies
    QuantizedCache.__init__ to use the model's layer_types for per-layer dispatch.
    """
    try:
        from transformers.cache_utils import (
            Cache,
            QuantizedCache,
            QuantizedLayer,
            QuantoQuantizedLayer,
            HQQQuantizedLayer,
            LinearAttentionLayer,
            DynamicLayer,
        )

        if getattr(QuantizedCache, "_quenstar_patched", False):
            return

        class QuantoQuantizedLinearAttentionLayer(QuantoQuantizedLayer, LinearAttentionLayer):
            """Hybrid layer: linear attention state + quantized KV cache.

            QuantoQuantizedLayer comes first in MRO so DynamicLayer.lazy_initialization(key_states, value_states)
            is found before LinearAttentionLayer.lazy_initialization(conv_states, recurrent_states).
            We override lazy_initialization to dispatch correctly for both callers:
            - update_conv_state calls with conv_states=
            - update_recurrent_state calls with recurrent_states=
            - QuantizedLayer.update() calls with positional (key_states, value_states)
            """

            def __init__(self, nbits=4, axis_key=0, axis_value=0, q_group_size=64, residual_length=128):
                QuantoQuantizedLayer.__init__(self, nbits, axis_key, axis_value, q_group_size, residual_length)
                LinearAttentionLayer.__init__(self, config=None)

            def lazy_initialization(self, *args, **kwargs):
                if args or 'key_states' in kwargs or 'value_states' in kwargs:
                    # Called by QuantizedLayer.update → DynamicLayer.lazy_initialization
                    DynamicLayer.lazy_initialization(self, *args, **kwargs)
                elif 'conv_states' in kwargs or 'recurrent_states' in kwargs:
                    # Called by update_conv_state / update_recurrent_state
                    LinearAttentionLayer.lazy_initialization(self, **kwargs)

        class HQQQuantizedLinearAttentionLayer(HQQQuantizedLayer, LinearAttentionLayer):
            def __init__(self, nbits=4, axis_key=0, axis_value=0, q_group_size=64, residual_length=128):
                HQQQuantizedLayer.__init__(self, nbits, axis_key, axis_value, q_group_size, residual_length)
                LinearAttentionLayer.__init__(self, config=None)

            def lazy_initialization(self, *args, **kwargs):
                if args or 'key_states' in kwargs or 'value_states' in kwargs:
                    DynamicLayer.lazy_initialization(self, *args, **kwargs)
                elif 'conv_states' in kwargs or 'recurrent_states' in kwargs:
                    LinearAttentionLayer.lazy_initialization(self, **kwargs)

        _original_init = QuantizedCache.__init__

        def _patched_init(self, backend, config, nbits=4, axis_key=0, axis_value=0,
                         q_group_size=64, residual_length=128):
            if backend == "quanto":
                attn_class = QuantoQuantizedLayer
                linear_class = QuantoQuantizedLinearAttentionLayer
            elif backend == "hqq":
                attn_class = HQQQuantizedLayer
                linear_class = HQQQuantizedLinearAttentionLayer
            else:
                raise ValueError(f"Unknown quantization backend `{backend}`")

            text_config = config.get_text_config(decoder=True)
            layer_types = getattr(text_config, "layer_types", None)

            if layer_types is not None and len(layer_types) == text_config.num_hidden_layers:
                layers = []
                for lt in layer_types:
                    if lt in ("linear_attention", "conv", "mamba", "moe", "hybrid"):
                        layers.append(linear_class(nbits, axis_key, axis_value, q_group_size, residual_length))
                    else:
                        layers.append(attn_class(nbits, axis_key, axis_value, q_group_size, residual_length))
            else:
                layers = [
                    attn_class(nbits, axis_key, axis_value, q_group_size, residual_length)
                    for _ in range(text_config.num_hidden_layers)
                ]

            Cache.__init__(self, layers=layers)

        QuantizedCache.__init__ = _patched_init
        QuantizedCache._quenstar_patched = True
        log.info("Patched QuantizedCache for mixed layer types (DeltaNet + Attention)")

    except ImportError as e:
        log.warning(f"Cannot patch QuantizedCache: {e}")
    except Exception as e:
        log.warning(f"Failed to patch QuantizedCache: {e}")


def _print_memory_usage(prefix: str = "") -> None:
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    log.info(f"{prefix} GPU memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")


def _create_quantized_kv_cache(model):
    try:
        from transformers import QuantizedCache
        _patch_quantized_cache()
        cache = QuantizedCache(
            backend="quanto",
            config=model.config,
            nbits=4,
            axis_key=0,
            axis_value=0,
            q_group_size=64,
            residual_length=128,
        )
        log.info("Created quantized KV cache: 4-bit, backend=quanto (patched for DeltaNet)")
        return cache
    except ImportError as e:
        log.warning(f"Quantized KV cache not available ({e}) — using dynamic cache")
        return None
    except Exception as e:
        log.warning(f"Failed to create quantized KV cache ({e}) — using dynamic cache")
        return None


def load_and_quantize_model(
    model_path: str,
    weight_bits: int = 4,
    kv_cache_bits: int = 4,
    turbo: bool = False,
    attn_implementation: str = "sdpa",
    torch_dtype_str: str = "bfloat16",
) -> tuple[torch.nn.Module, object, Optional[object]]:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    dtype = getattr(torch, torch_dtype_str) if torch_dtype_str != "auto" else torch.bfloat16

    _print_memory_usage("before model load")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="cuda:0",
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )
    log.info("Loaded with bitsandbytes 4-bit NF4 quantization")

    _print_memory_usage("after model load")

    gc.collect()
    torch.cuda.empty_cache()
    _print_memory_usage("after gc")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cache = _create_quantized_kv_cache(model)

    model.eval()
    return model, tokenizer, cache
