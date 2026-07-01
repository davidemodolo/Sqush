"""Tests for the bake-time device_map CPU-offload logic in load_and_quantize_model."""
from __future__ import annotations

import types
from unittest.mock import MagicMock, patch, call


def _make_mock_config(has_quant_config: bool = True, pre_baked: bool = False):
    cfg = MagicMock()
    cfg.qs_pre_baked_embeddings = pre_baked
    if has_quant_config:
        cfg.quantization_config = {"quant_method": "bitsandbytes", "llm_int8_enable_fp32_cpu_offload": False}
    else:
        cfg.quantization_config = None
    return cfg


def _make_mock_model():
    model = MagicMock()
    model.config = MagicMock()
    model.config.quantization_config = MagicMock()
    model.dtype = None
    model.device = None
    return model


def _patch_load_deps(mock_config, mock_model, *, is_pre_quantized=True):
    """Return a context-manager stack that patches all from_pretrained calls."""
    import contextlib

    @contextlib.contextmanager
    def ctx():
        with (
            patch("quantstar.quantize._model_is_pre_quantized", return_value=is_pre_quantized),
            patch("transformers.AutoConfig") as mock_ac,
            patch("transformers.AutoModelForImageTextToText") as mock_amt,
            patch("transformers.AutoTokenizer") as mock_at,
        ):
            mock_ac.from_pretrained.return_value = mock_config
            mock_amt.from_pretrained.return_value = mock_model

            tokenizer = MagicMock()
            tokenizer.pad_token = "x"
            mock_at.from_pretrained.return_value = tokenizer

            processor_mock = MagicMock()
            with patch(
                "transformers.models.qwen3_vl.Qwen3VLProcessor",
                **{"from_pretrained.return_value": processor_mock},
            ):
                yield mock_ac, mock_amt

    return ctx()


class TestBakeDeviceMap:
    def test_dict_device_map_sets_fp32_cpu_offload(self):
        """When device_map is a dict, llm_int8_enable_fp32_cpu_offload is set True."""
        from quantstar.quantize import load_and_quantize_model

        mock_cfg = _make_mock_config(has_quant_config=True)
        mock_model = _make_mock_model()

        with _patch_load_deps(mock_cfg, mock_model):
            load_and_quantize_model(
                model_path="/fake",
                device_map={"model.visual": "cpu", "": 0},
            )

        assert mock_cfg.quantization_config["llm_int8_enable_fp32_cpu_offload"] is True

    def test_string_device_map_does_not_set_fp32_cpu_offload(self):
        """When device_map is a string, llm_int8_enable_fp32_cpu_offload is untouched."""
        from quantstar.quantize import load_and_quantize_model

        mock_cfg = _make_mock_config(has_quant_config=True)
        mock_model = _make_mock_model()

        with _patch_load_deps(mock_cfg, mock_model):
            load_and_quantize_model(
                model_path="/fake",
                device_map="cuda:0",
            )

        assert mock_cfg.quantization_config["llm_int8_enable_fp32_cpu_offload"] is False

    def test_dict_device_map_no_quant_config_does_not_raise(self):
        """If quantization_config is absent (non-pre-quantized), no AttributeError."""
        from quantstar.quantize import load_and_quantize_model

        mock_cfg = _make_mock_config(has_quant_config=False)
        mock_model = _make_mock_model()

        with _patch_load_deps(mock_cfg, mock_model, is_pre_quantized=False):
            load_and_quantize_model(
                model_path="/fake",
                device_map={"model.visual": "cpu", "": 0},
            )
        # no exception raised

    def test_dict_device_map_passed_to_from_pretrained(self):
        """The device_map dict is forwarded verbatim to from_pretrained."""
        from quantstar.quantize import load_and_quantize_model

        dm = {"model.visual": "cpu", "": 0}
        mock_cfg = _make_mock_config()
        mock_model = _make_mock_model()

        with _patch_load_deps(mock_cfg, mock_model) as (_, mock_amt):
            load_and_quantize_model(model_path="/fake", device_map=dm)

        _, kwargs = mock_amt.from_pretrained.call_args
        assert kwargs["device_map"] == dm
