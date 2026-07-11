"""Tests for the tier-aware bake dispatch and the NF4 checkpoint bake.

Covers:
  - _bake_model routes LOW → CPU side-car bake, HIGH → GPU NF4 bake
  - _bake_model deletes the raw model and returns the cooked path
  - _cooked_model_path naming
  - bake_nf4_checkpoint loads 4-bit and serializes model + tokenizer + processor
"""
from __future__ import annotations

from unittest import mock

import pytest


def _config(tier):
    from sqush.config import SqushConfig, VramTier
    cfg = SqushConfig()
    cfg.vram_tier = tier
    return cfg


class TestCookedModelPath:
    def test_appends_cooked_suffix(self):
        from sqush.__main__ import _cooked_model_path
        assert _cooked_model_path("/models/Qwen__Qwen3.6-27B") == "/models/Qwen__Qwen3.6-27B-cooked"

    def test_strips_trailing_slash(self):
        from sqush.__main__ import _cooked_model_path
        assert _cooked_model_path("/models/foo/") == "/models/foo-cooked"


class TestBakeModelDispatch:
    def test_low_tier_uses_cpu_side_car_bake(self):
        from sqush.config import VramTier
        cfg = _config(VramTier.LOW)
        log = mock.MagicMock()

        with mock.patch("sqush.__main__._bake_safetensors") as side_car, \
                mock.patch("sqush.quantize.bake_nf4_checkpoint") as nf4, \
                mock.patch("shutil.rmtree") as rmtree:
            from sqush.__main__ import _bake_model, _cooked_model_path
            result = _bake_model("/models/raw", cfg, log)

        side_car.assert_called_once()
        nf4.assert_not_called()
        rmtree.assert_called_once_with("/models/raw")
        assert result == _cooked_model_path("/models/raw")

    def test_high_tier_uses_gpu_nf4_bake(self):
        from sqush.config import VramTier
        cfg = _config(VramTier.HIGH)
        log = mock.MagicMock()

        with mock.patch("sqush.__main__._bake_safetensors") as side_car, \
                mock.patch("sqush.quantize.bake_nf4_checkpoint") as nf4, \
                mock.patch("shutil.rmtree") as rmtree:
            from sqush.__main__ import _bake_model, _cooked_model_path
            result = _bake_model("/models/raw", cfg, log)

        nf4.assert_called_once()
        side_car.assert_not_called()
        # cooked path passed through, and raw removed after baking
        args, kwargs = nf4.call_args
        assert args[0] == "/models/raw"
        assert args[1] == _cooked_model_path("/models/raw")
        rmtree.assert_called_once_with("/models/raw")
        assert result == _cooked_model_path("/models/raw")

    def test_raw_not_deleted_if_bake_raises(self):
        from sqush.config import VramTier
        cfg = _config(VramTier.HIGH)
        log = mock.MagicMock()

        with mock.patch("sqush.quantize.bake_nf4_checkpoint", side_effect=RuntimeError("boom")), \
                mock.patch("shutil.rmtree") as rmtree:
            from sqush.__main__ import _bake_model
            with pytest.raises(RuntimeError):
                _bake_model("/models/raw", cfg, log)

        rmtree.assert_not_called()


class TestBakeNf4Checkpoint:
    def _run(self):
        model = mock.MagicMock()
        tok = mock.MagicMock()
        proc = mock.MagicMock()

        AutoModel = mock.MagicMock()
        AutoModel.from_pretrained.return_value = model
        AutoTok = mock.MagicMock()
        AutoTok.from_pretrained.return_value = tok
        Proc = mock.MagicMock()
        Proc.from_pretrained.return_value = proc
        bnb_cfg = mock.MagicMock()
        BnbConfig = mock.MagicMock(return_value=bnb_cfg)

        with mock.patch("transformers.AutoModelForImageTextToText", AutoModel), \
                mock.patch("transformers.AutoTokenizer", AutoTok), \
                mock.patch("transformers.BitsAndBytesConfig", BnbConfig), \
                mock.patch("transformers.models.qwen3_vl.Qwen3VLProcessor", Proc), \
                mock.patch("sqush.quantize._print_memory_usage"):
            from sqush.quantize import bake_nf4_checkpoint
            bake_nf4_checkpoint("/models/raw", "/models/raw-cooked")

        return dict(model=model, tok=tok, proc=proc, AutoModel=AutoModel,
                    BnbConfig=BnbConfig, bnb_cfg=bnb_cfg)

    def test_loads_with_4bit_nf4_config(self):
        r = self._run()
        _, kwargs = r["BnbConfig"].call_args
        assert kwargs["load_in_4bit"] is True
        assert kwargs["bnb_4bit_quant_type"] == "nf4"
        assert kwargs["bnb_4bit_use_double_quant"] is True

    def test_loads_raw_on_pure_gpu_device_map(self):
        r = self._run()
        args, kwargs = r["AutoModel"].from_pretrained.call_args
        assert args[0] == "/models/raw"
        assert kwargs["quantization_config"] is r["bnb_cfg"]
        # serialization requires a pure-GPU map (no CPU offload)
        assert kwargs["device_map"] == "cuda:0"

    def test_saves_model_tokenizer_and_processor_to_cooked(self):
        r = self._run()
        r["model"].save_pretrained.assert_called_once()
        args, kwargs = r["model"].save_pretrained.call_args
        assert args[0] == "/models/raw-cooked"
        r["tok"].save_pretrained.assert_called_once_with("/models/raw-cooked")
        r["proc"].save_pretrained.assert_called_once_with("/models/raw-cooked")
