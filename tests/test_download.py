"""Tests for download completeness detection in sqush.download."""
from __future__ import annotations

import json


def _write(path, data):
    path.write_text(json.dumps(data) if isinstance(data, (dict, list)) else data)


class TestDownloadComplete:
    def test_missing_dir_is_incomplete(self, tmp_path):
        from sqush.download import _download_complete
        assert _download_complete(tmp_path / "nope") is False

    def test_sharded_all_present_is_complete(self, tmp_path):
        from sqush.download import _download_complete
        _write(tmp_path / "model.safetensors.index.json",
               {"weight_map": {"a": "shard-1.safetensors", "b": "shard-2.safetensors"}})
        (tmp_path / "shard-1.safetensors").write_text("x")
        (tmp_path / "shard-2.safetensors").write_text("x")
        assert _download_complete(tmp_path) is True

    def test_sharded_missing_shard_is_incomplete(self, tmp_path):
        from sqush.download import _download_complete
        _write(tmp_path / "model.safetensors.index.json",
               {"weight_map": {"a": "shard-1.safetensors", "b": "shard-2.safetensors"}})
        (tmp_path / "shard-1.safetensors").write_text("x")
        # shard-2 never finished downloading
        assert _download_complete(tmp_path) is False

    def test_nonempty_dir_without_index_is_incomplete(self, tmp_path):
        """A partial snapshot (files present, no index/config) must not short-circuit."""
        from sqush.download import _download_complete
        (tmp_path / "shard-1.safetensors").write_text("x")
        assert _download_complete(tmp_path) is False

    def test_unsharded_with_config_is_complete(self, tmp_path):
        from sqush.download import _download_complete
        (tmp_path / "model.safetensors").write_text("x")
        (tmp_path / "config.json").write_text("{}")
        assert _download_complete(tmp_path) is True

    def test_corrupt_index_is_incomplete(self, tmp_path):
        from sqush.download import _download_complete
        (tmp_path / "model.safetensors.index.json").write_text("{not json")
        assert _download_complete(tmp_path) is False
