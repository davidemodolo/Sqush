"""Shared pytest fixtures for the QuantStar test suite.

GPU-only tests use `@pytest.mark.gpu` and are skipped automatically without CUDA.
"""
from __future__ import annotations

import base64
import io
from unittest import mock

import pytest
import torch
from PIL import Image


# ── image helpers ────────────────────────────────────────────────────────────

def _tiny_rgb_bytes(size: tuple[int, int] = (4, 4)) -> bytes:
    img = Image.new("RGB", size, (128, 64, 192))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def make_b64_image(size: tuple[int, int] = (4, 4)) -> str:
    return "data:image/png;base64," + base64.b64encode(_tiny_rgb_bytes(size)).decode()


@pytest.fixture
def tiny_image() -> Image.Image:
    return Image.new("RGB", (4, 4), (128, 64, 192))


@pytest.fixture
def b64_image() -> str:
    return make_b64_image()


# ── engine mock helpers ──────────────────────────────────────────────────────

def make_mock_tokenizer() -> mock.MagicMock:
    tok = mock.MagicMock()
    tok.pad_token_id = None
    tok.eos_token_id = 151645
    tok.apply_chat_template.return_value = (
        "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
    )
    fake_input = mock.MagicMock()
    fake_input.shape = (1, 10)
    fake_input.to = lambda device: fake_input
    tok.return_value = {"input_ids": fake_input}
    tok.encode.return_value = [1, 2, 3, 4, 5]
    tok.decode.return_value = "response text"
    return tok


def make_mock_model() -> mock.MagicMock:
    model = mock.MagicMock()
    model.device = "cpu"
    model.dtype = None
    model.config = mock.MagicMock()
    model.config.image_token_id = None

    # generate() returns an object with .sequences and .past_key_values
    gen_out = mock.MagicMock()
    gen_out.sequences = torch.zeros(1, 15, dtype=torch.long)
    gen_out.past_key_values = None
    model.generate.return_value = gen_out
    return model


def make_engine(processor=None, cache_factory=None) -> "InferenceEngine":
    from quantstar.engine import InferenceEngine
    return InferenceEngine(
        model=make_mock_model(),
        tokenizer=make_mock_tokenizer(),
        processor=processor,
        cache_config=cache_factory,
    )


@pytest.fixture
def mock_tokenizer() -> mock.MagicMock:
    return make_mock_tokenizer()


@pytest.fixture
def mock_model() -> mock.MagicMock:
    return make_mock_model()


@pytest.fixture
def engine() -> "InferenceEngine":
    return make_engine()


@pytest.fixture
def default_config():
    from quantstar.config import QuantStarConfig
    return QuantStarConfig()


# ── markers ──────────────────────────────────────────────────────────────────

def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires CUDA GPU")
    config.addinivalue_line("markers", "integration: requires full model load")


@pytest.fixture(autouse=True)
def skip_gpu_without_cuda(request):
    if request.node.get_closest_marker("gpu"):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
