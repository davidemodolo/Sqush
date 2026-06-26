from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download
from tqdm import tqdm


def download_model(repo_id: str, cache_dir: str) -> str:
    cache_dir = Path(cache_dir).resolve()
    target = cache_dir / repo_id.replace("/", "__")

    if target.exists() and any(target.iterdir()):
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
