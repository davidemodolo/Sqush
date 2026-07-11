from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import snapshot_download
from tqdm import tqdm


def _download_complete(target: Path) -> bool:
    """A directory counts as a finished download only when its weight index is
    present and every shard it references exists. A merely non-empty directory
    may be a partial/interrupted snapshot — returning True there would skip the
    resume and later fail in from_pretrained on missing shards."""
    if not target.exists():
        return False
    index = target / "model.safetensors.index.json"
    if index.exists():
        try:
            weight_map = json.loads(index.read_text()).get("weight_map", {})
        except (json.JSONDecodeError, OSError):
            return False
        shards = set(weight_map.values())
        return bool(shards) and all((target / s).exists() for s in shards)
    # Unsharded model: a single weights file plus config is enough.
    return (target / "model.safetensors").exists() and (target / "config.json").exists()


def download_model(repo_id: str, cache_dir: str) -> str:
    cache_dir = Path(cache_dir).resolve()
    target = cache_dir / repo_id.replace("/", "__")

    if _download_complete(target):
        print(f"Model already downloaded at {target}")
        return str(target)

    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {repo_id} to {target} …")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        local_dir_use_symlinks=False,
        resume_download=True,
        tqdm_class=tqdm,
        ignore_patterns=["*.pth", "*.bin", "*.msgpack", "*.h5"],
    )

    print(f"Download complete: {target}")
    return str(target)
