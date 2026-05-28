from __future__ import annotations

import logging
import time
from typing import Any, Optional

from .config import QuenStarConfig
from .engine import Engine
from .kvstore import KVCacheStore
from .types import KVSaveReason

_log = logging.getLogger(__name__)


class SessionManager:
    def __init__(self, engine: Engine, kvstore: KVCacheStore, config: QuenStarConfig):
        self._engine = engine
        self._kvstore = kvstore
        self._config = config
        self._current_key: Optional[str] = None
        self._current_messages: list[dict[str, Any]] = []
        self._resumed_from_cache: bool = False
        self._created_at: float = time.time()

    @property
    def session_id(self) -> Optional[str]:
        return self._current_key

    @property
    def messages(self) -> list[dict[str, Any]]:
        return list(self._current_messages)

    @property
    def is_resumed(self) -> bool:
        return self._resumed_from_cache

    def new_session(self, messages: list[dict[str, Any]]) -> bool:
        key = self._kvstore.compute_key(messages)

        if key == self._current_key:
            self._current_messages = list(messages)
            return False

        if self._shares_prefix(self._current_messages, messages):
            _log.debug("Continuing conversation (prefix match, no re-prefill)")
            self._current_key = key
            self._current_messages = list(messages)
            return False

        result = self._kvstore.load_and_bump(key)
        if result:
            _, state_bytes, header = result
            try:
                self._engine.load_state(state_bytes)
                self._current_key = key
                self._current_messages = list(messages)
                self._resumed_from_cache = True
                self._created_at = header.created_at
                _log.info(
                    "Session resumed from disk (key=%s, hits=%d)",
                    key,
                    header.hit_count,
                )
                return True
            except Exception as exc:
                _log.warning("Failed to restore session %s: %s", key, exc)

        self._engine.reset_context()
        self._current_key = key
        self._current_messages = list(messages)
        self._resumed_from_cache = False
        self._created_at = time.time()
        _log.info("New session (key=%s, %d messages)", key, len(messages))
        return False

    def save(self):
        if self._current_key is None:
            return
        try:
            state_bytes = self._engine.save_state()
        except Exception as exc:
            _log.warning("Failed to save engine state: %s", exc)
            return

        self._kvstore.store(
            key=self._current_key,
            state_bytes=state_bytes,
            reason=KVSaveReason.CONTINUED,
        )

    def save_and_reset(self):
        self.save()
        self._engine.reset_context()
        self._current_key = None
        self._current_messages = []
        self._resumed_from_cache = False

    def update(self, new_messages: list[dict[str, Any]]):
        self._current_messages = list(new_messages)
        self._current_key = self._kvstore.compute_key(new_messages)

    def save_checkpoint(self):
        try:
            state_bytes = self._engine.save_state()
        except Exception:
            return
        self._kvstore.store(
            key=self._current_key or self._kvstore.compute_key(self._current_messages),
            state_bytes=state_bytes,
            reason=KVSaveReason.COLD if not self._resumed_from_cache else KVSaveReason.CONTINUED,
        )

    def list_sessions(self) -> list[dict]:
        return self._kvstore.list_files()

    def delete_session(self, key: str) -> bool:
        return self._kvstore.delete(key)

    @staticmethod
    def _shares_prefix(
        prev: list[dict[str, Any]],
        current: list[dict[str, Any]],
    ) -> bool:
        if not prev:
            return False
        if len(current) < len(prev):
            return False
        return prev == current[: len(prev)]
