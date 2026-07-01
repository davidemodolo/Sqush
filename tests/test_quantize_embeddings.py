"""Tests for _quantize_embeddings in quantstar.quantize."""
from __future__ import annotations

import inspect


class TestQuantizeEmbeddingsCpuFirst:
    """Verify that embedding weights are moved to CPU BEFORE dtype conversion."""

    def test_cpu_called_before_float32_conversion(self):
        """weight.data.cpu().to(float32), not .to(float32).cpu() — avoids ~4 GB GPU peak."""
        from quantstar import quantize

        src = inspect.getsource(quantize._quantize_embeddings)

        # The correct pattern: .cpu().to(torch.float32)
        assert ".cpu().to(torch.float32)" in src, (
            "_quantize_embeddings must do .cpu().to(torch.float32) to avoid creating "
            "a float32 copy on GPU before moving to CPU. "
            "Found source:\n" + src
        )

        # The wrong pattern must NOT be present
        assert ".to(torch.float32).cpu()" not in src, (
            "_quantize_embeddings must NOT do .to(torch.float32).cpu() — "
            "this creates a ~4 GB float32 tensor on GPU before the CPU move."
        )
