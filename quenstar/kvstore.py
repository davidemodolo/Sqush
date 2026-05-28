from __future__ import annotations

import json
import hashlib
import json
import logging
import math
import os
import struct
import time
from pathlib import Path
from typing import Any, Optional

from .config import KVCacheConfig
from .types import KVCacheHeader, KVSaveReason

_log = logging.getLogger(__name__)

FLAG_TOOL_ID_MAP = 1 << 0


class KVCacheStore:
    def __init__(self, config: KVCacheConfig, model_id: str, n_ctx: int, quant_bits: int = 4):
        self.config = config
        self.model_id = model_id
        self.n_ctx = n_ctx
        self.quant_bits = quant_bits
        self._dir = Path(config.dir).expanduser().resolve()
        self._dir.mkdir(parents=True, exist_ok=True)
        _log.info("KV cache store: %s (max %d MB)", self._dir, config.space_mb)

    def compute_key(self, messages: list[dict[str, Any]]) -> str:
        canonical = json.dumps(messages, sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(canonical.encode("utf-8")).hexdigest()

    def store(
        self,
        key: str,
        state_bytes: bytes,
        reason: KVSaveReason = KVSaveReason.COLD,
        model_id: str = "",
    ) -> str:
        filepath = self._dir / f"{key}.kv"

        header = KVCacheHeader(
            magic=b"QSTK",
            version=1,
            quant_bits=self.quant_bits,
            save_reason=reason.value,
            flags=0,
            n_tokens=0,
            hit_count=0,
            context_size=self.n_ctx,
            created_at=time.time(),
            last_used_at=time.time(),
            payload_bytes=len(state_bytes),
        )

        with open(filepath, "wb") as f:
            f.write(header.pack())
            f.write(struct.pack("<I", 0))
            f.write(state_bytes)

        self._evict_if_needed(keep=key)
        _log.debug(
            "Saved KV cache: %s (payload=%d bytes, reason=%s)",
            key,
            len(state_bytes),
            reason.name,
        )
        return key

    def load(self, key: str) -> Optional[tuple[str, bytes, KVCacheHeader]]:
        filepath = self._dir / f"{key}.kv"
        if not filepath.exists():
            return None
        return self._load_file(filepath, key)

    def _load_file(
        self, filepath: Path, key: str
    ) -> Optional[tuple[str, bytes, KVCacheHeader]]:
        try:
            with open(filepath, "rb") as f:
                header_data = f.read(KVCacheHeader.HEADER_SIZE)
                if len(header_data) < KVCacheHeader.HEADER_SIZE:
                    _log.warning("Truncated KV file header: %s", filepath)
                    return None

                header = KVCacheHeader.unpack(header_data)
                if header.magic != b"QSTK":
                    _log.warning("Bad KV file magic: %s", filepath)
                    return None
                if header.version != 1:
                    _log.warning(
                        "Unsupported KV file version %d: %s", header.version, filepath
                    )
                    return None
                if header.context_size != self.n_ctx:
                    _log.warning(
                        "Context size mismatch: file=%d current=%d (%s)",
                        header.context_size,
                        self.n_ctx,
                        filepath,
                    )
                    return None

                _ = f.read(4)
                state_bytes = f.read(header.payload_bytes)

            _log.debug(
                "Loaded KV cache: %s (payload=%d bytes)",
                key,
                len(state_bytes),
            )
            return key, state_bytes, header

        except Exception as exc:
            _log.error("Failed to load KV cache %s: %s", key, exc)
            return None

    def load_and_bump(self, key: str) -> Optional[tuple[str, bytes, KVCacheHeader]]:
        filepath = self._dir / f"{key}.kv"
        if not filepath.exists():
            return None
        result = self._load_file(filepath, key)
        if result:
            key, state_bytes, header = result
            header.hit_count += 1
            header.last_used_at = time.time()
            try:
                with open(filepath, "r+b") as f:
                    f.write(header.pack())
            except OSError:
                pass
        return result

    def delete(self, key: str) -> bool:
        filepath = self._dir / f"{key}.kv"
        if filepath.exists():
            filepath.unlink()
            _log.debug("Deleted KV cache: %s", key)
            return True
        return False

    def list_files(self) -> list[dict]:
        results = []
        for filepath in sorted(
            self._dir.glob("*.kv"), key=lambda p: p.stat().st_mtime, reverse=True
        ):
            try:
                with open(filepath, "rb") as f:
                    header_data = f.read(KVCacheHeader.HEADER_SIZE)
                    if len(header_data) < KVCacheHeader.HEADER_SIZE:
                        continue
                    header = KVCacheHeader.unpack(header_data)
                    if header.magic != b"QSTK":
                        continue
                key = filepath.stem
                results.append({
                    "key": key,
                    "n_tokens": header.n_tokens,
                    "hit_count": header.hit_count,
                    "context_size": header.context_size,
                    "created_at": header.created_at,
                    "last_used_at": header.last_used_at,
                    "save_reason": header.save_reason,
                    "payload_bytes": header.payload_bytes,
                    "file_size": filepath.stat().st_size,
                })
            except Exception as exc:
                _log.warning("Failed to read KV header %s: %s", filepath, exc)
        return results

    def total_size_bytes(self) -> int:
        return sum(
            f.stat().st_size for f in self._dir.glob("*.kv") if f.is_file()
        )

    def _evict_if_needed(self, keep: str = ""):
        space_limit = self.config.space_mb * 1024 * 1024
        current = self.total_size_bytes()
        if current <= space_limit:
            return

        _log.info(
            "KV cache eviction triggered (%.1f MB > %d MB limit)",
            current / 1024**2,
            self.config.space_mb,
        )

        files = self.list_files()
        files.sort(key=lambda f: self._eviction_score(f))

        for finfo in files:
            if finfo["key"] == keep:
                continue
            if self.total_size_bytes() <= space_limit:
                break
            self.delete(finfo["key"])
            _log.debug("Evicted KV cache: %s", finfo["key"])

    def _eviction_score(self, finfo: dict) -> float:
        age_hours = (time.time() - finfo["last_used_at"]) / 3600.0
        half_life = self.config.eviction_half_life_hours
        decay = math.exp(-age_hours / half_life) if half_life > 0 else 1.0
        return (1 + finfo["hit_count"]) * decay
