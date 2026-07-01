"""Tests for _bake_safetensors in quantstar.__main__."""
from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock, patch, call

import torch


def _make_shard(tensors: dict) -> str:
    """Write tensors to a temporary safetensors file, return its path."""
    import tempfile
    from safetensors.torch import save_file

    tmp = tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False)
    tmp.close()
    save_file(tensors, tmp.name)
    return tmp.name


def _run_bake(raw_tensors: dict, config_extra: dict | None = None):
    """
    Run _bake_safetensors against an in-memory raw model directory.
    Returns (cooked_dir, log_mock).
    """
    from safetensors.torch import save_file
    from quantstar.__main__ import _bake_safetensors

    with tempfile.TemporaryDirectory() as raw_dir, \
         tempfile.TemporaryDirectory() as cooked_dir:

        # Write model.safetensors
        save_file(raw_tensors, os.path.join(raw_dir, "model.safetensors"))

        # Write config.json with quantization_config
        cfg = {
            "quantization_config": {
                "quant_method": "bitsandbytes",
                "llm_int8_skip_modules": ["visual", "vision_model", "lm_head"],
            }
        }
        if config_extra:
            cfg.update(config_extra)
        with open(os.path.join(raw_dir, "config.json"), "w") as f:
            json.dump(cfg, f)

        log = MagicMock()

        # Patch bnb Linear4bit so we don't need a real GPU
        with patch("bitsandbytes.nn.Linear4bit") as MockLinear4bit:
            fake_layer = MagicMock()
            MockLinear4bit.return_value = fake_layer
            fake_layer.cuda.return_value = fake_layer
            fake_layer.cpu.return_value = fake_layer

            # Simulate _save_to_state_dict filling destination with NF4 tensors
            def fake_save_to_state_dict(destination, prefix, keep_vars):
                destination[prefix + "weight"] = torch.zeros(4, dtype=torch.uint8)
                destination[prefix + "weight.absmax"] = torch.zeros(2, dtype=torch.uint8)
                destination[prefix + "weight.quant_map"] = torch.zeros(16, dtype=torch.float32)
                destination[prefix + "weight.quant_state.bitsandbytes__nf4"] = torch.zeros(8, dtype=torch.uint8)

            fake_layer._save_to_state_dict.side_effect = fake_save_to_state_dict

            _bake_safetensors(raw_dir, cooked_dir, log)

        # Read cooked model.safetensors
        from safetensors.torch import load_file
        cooked_tensors = load_file(os.path.join(cooked_dir, "model.safetensors"))
        with open(os.path.join(cooked_dir, "config.json")) as f:
            cooked_cfg = json.load(f)

        return cooked_tensors, cooked_cfg, log


class TestBakeSafetensors:
    def test_copies_non_visual_tensors_unchanged(self):
        raw = {
            "model.language_model.layers.0.self_attn.q_proj.weight":
                torch.zeros(8, 8, dtype=torch.uint8),
            # embed_tokens is intentionally excluded from cooked shard (baked to side-car)
            "model.language_model.embed_tokens.weight":
                torch.zeros(16, 8, dtype=torch.bfloat16),
        }
        cooked, _, _ = _run_bake(raw)
        assert "model.language_model.layers.0.self_attn.q_proj.weight" in cooked
        # embed_tokens.weight is a tiny [1, hidden] placeholder in the shard (real data in side-car)
        assert cooked["model.language_model.embed_tokens.weight"].shape[0] == 1

    def test_visual_bfloat16_weights_pass_through_unchanged(self):
        """Visual encoder bfloat16 weights must NOT be pre-baked to NF4 in the cooked shard.

        Bitsandbytes cannot reconstruct a valid quant_state from NF4 weights saved via
        _save_to_state_dict + safetensors round-trip — quant_state comes back None and
        the forward pass crashes with 'BFloat16 and Byte dtype mismatch'. The correct
        approach is to leave visual weights as bfloat16 so bitsandbytes quantizes them
        at load time via from_pretrained, the same way it handles the language model.
        """
        raw = {
            "model.visual.blocks.0.attn.qkv.weight": torch.randn(8, 8, dtype=torch.bfloat16),
            "model.visual.blocks.0.attn.proj.weight": torch.randn(8, 8, dtype=torch.bfloat16),
        }
        cooked, _, _ = _run_bake(raw)
        assert cooked["model.visual.blocks.0.attn.qkv.weight"].dtype == torch.bfloat16
        assert cooked["model.visual.blocks.0.attn.proj.weight"].dtype == torch.bfloat16
        # No NF4 companion keys should appear — those would indicate pre-baking,
        # which breaks at inference time because the quant_state blob is unreadable.
        assert not any(".absmax" in k for k in cooked), \
            "Visual encoder must not be pre-baked to NF4: companion keys found in cooked shard"

    def test_skips_visual_1d_tensors(self):
        """Norm weights and biases (1D) are passed through unchanged."""
        raw = {
            "model.visual.blocks.0.norm1.weight": torch.ones(8, dtype=torch.bfloat16),
            "model.visual.blocks.0.attn.proj.bias": torch.zeros(8, dtype=torch.bfloat16),
        }
        cooked, _, _ = _run_bake(raw)
        assert "model.visual.blocks.0.norm1.weight" in cooked
        assert "model.visual.blocks.0.attn.proj.bias" in cooked
        # No NF4 companion keys
        assert not any(".absmax" in k for k in cooked)

    def test_skips_already_quantized_uint8_tensors(self):
        """uint8 tensors under model.visual (pre-quantized) are not re-quantized."""
        raw = {
            "model.visual.blocks.0.attn.proj.weight":
                torch.zeros(8, 4, dtype=torch.uint8),
        }
        cooked, _, _ = _run_bake(raw)
        assert "model.visual.blocks.0.attn.proj.weight" in cooked
        assert not any(".absmax" in k for k in cooked)

    def test_bakes_embed_tokens_to_side_car(self):
        """embed_tokens.weight must be excluded from cooked shard, saved to side-car, flag set."""
        raw = {
            "model.language_model.embed_tokens.weight":
                torch.randn(16, 8, dtype=torch.bfloat16),
            "model.language_model.layers.0.self_attn.q_proj.weight":
                torch.zeros(8, 8, dtype=torch.uint8),
        }

        with tempfile.TemporaryDirectory() as raw_dir, \
             tempfile.TemporaryDirectory() as cooked_dir:
            from safetensors.torch import save_file, load_file
            save_file(raw, os.path.join(raw_dir, "model.safetensors"))

            cfg = {
                "quantization_config": {
                    "quant_method": "bitsandbytes",
                    "llm_int8_skip_modules": ["lm_head"],
                }
            }
            with open(os.path.join(raw_dir, "config.json"), "w") as f:
                json.dump(cfg, f)

            log = MagicMock()
            with patch("bitsandbytes.nn.Linear4bit"):
                from quantstar.__main__ import _bake_safetensors
                _bake_safetensors(raw_dir, cooked_dir, log)

            cooked = load_file(os.path.join(cooked_dir, "model.safetensors"))
            # embed_tokens.weight must be a tiny [1, hidden] placeholder in the cooked shard
            assert "model.language_model.embed_tokens.weight" in cooked
            assert cooked["model.language_model.embed_tokens.weight"].shape[0] == 1
            # Other weights pass through
            assert "model.language_model.layers.0.self_attn.q_proj.weight" in cooked

            # Side-car must exist with required keys
            sidecar_path = os.path.join(cooked_dir, "quantized_embeddings.safetensors")
            assert os.path.exists(sidecar_path), "quantized_embeddings.safetensors missing"
            sidecar = load_file(sidecar_path)
            for key in ("_qw", "_sc", "_zp", "_vocab", "_hidden"):
                assert key in sidecar, f"side-car missing key {key!r}"
            assert int(sidecar["_vocab"].item()) == 16
            assert int(sidecar["_hidden"].item()) == 8

            # Config must have qs_pre_baked_embeddings flag
            with open(os.path.join(cooked_dir, "config.json")) as f:
                cooked_cfg = json.load(f)
            assert cooked_cfg.get("qs_pre_baked_embeddings") is True

    def test_removes_visual_but_keeps_lm_head_in_skip_modules(self):
        """visual encoder entries are removed (bitsandbytes quantizes them at load),
        but lm_head MUST stay: on a pre-quantized checkpoint, from_pretrained expects
        any Linear4bit module's weights to be packed 4-bit + quant_state in the shard.
        Removing lm_head from the skip list loads its raw bf16 weight into a
        Linear4bit with no quant_state → bitsandbytes AssertionError on the first
        forward. lm_head is NF4-quantized post-load instead (_quantize_lm_head)."""
        raw = {
            "model.language_model.layers.0.self_attn.q_proj.weight":
                torch.zeros(4, 4, dtype=torch.uint8),
        }
        _, cfg, _ = _run_bake(raw)
        skip = cfg["quantization_config"]["llm_int8_skip_modules"]
        assert "visual" not in skip
        assert "vision_model" not in skip
        assert "lm_head" in skip
