"""Lightweight key-value memory store with atomic file persistence."""

import json
import logging
import os
import threading
from pathlib import Path
from typing import Dict, Optional, Union

from app.config.settings import get_settings

logger = logging.getLogger(__name__)


def normalize_key(key: str) -> str:
    """Normalize keys to lower snake_case for consistent retrieval."""
    return "_".join(key.strip().lower().split())


class MemoryStore:
    """Lightweight structured key-value memory store backed by a local JSON file.

    Features thread-safe in-memory caching and atomic file writes (via tempfile
    rename) to guarantee data integrity across unexpected restarts.
    """

    def __init__(self, file_path: Optional[Union[str, Path]] = None) -> None:
        if file_path is None:
            settings = get_settings()
            self.file_path = Path(settings.memory_storage_path)
        else:
            self.file_path = Path(file_path)

        self._lock = threading.Lock()
        self._cache: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        """Load stored memory from disk into the cache."""
        with self._lock:
            if not self.file_path.exists():
                logger.debug("Memory file does not exist at %s. Starting with empty store.", self.file_path)
                self._cache = {}
                return

            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self._cache = {str(k): str(v) for k, v in data.items()}
                    else:
                        logger.warning("Memory file content is not a dict; initializing empty cache.")
                        self._cache = {}
            except Exception as exc:
                logger.error("Failed to read memory file at %s: %s. Starting empty.", self.file_path, exc)
                self._cache = {}

    def _persist(self) -> None:
        """Atomically persist current in-memory cache to disk."""
        # Caller must hold self._lock
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.file_path.with_suffix(".tmp")

            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2, ensure_ascii=False)

            # Atomic replace
            os.replace(temp_path, self.file_path)
            logger.debug("Persisted %d memory entries to %s", len(self._cache), self.file_path)
        except Exception as exc:
            logger.error("Failed to persist memory store to %s: %s", self.file_path, exc)
            raise

    def get(self, key: str) -> Optional[str]:
        """Retrieve a stored memory value by key."""
        norm_key = normalize_key(key)
        with self._lock:
            return self._cache.get(norm_key)

    def set(self, key: str, value: str) -> None:
        """Store or update a memory value."""
        norm_key = normalize_key(key)
        norm_val = value.strip()
        with self._lock:
            self._cache[norm_key] = norm_val
            self._persist()
        logger.info("Stored memory: '%s' = '%s'", norm_key, norm_val)

    def delete(self, key: str) -> bool:
        """Delete a memory item. Returns True if deleted, False if not found."""
        norm_key = normalize_key(key)
        with self._lock:
            if norm_key in self._cache:
                del self._cache[norm_key]
                self._persist()
                logger.info("Deleted memory key: '%s'", norm_key)
                return True
            return False

    def exists(self, key: str) -> bool:
        """Check if a memory key exists."""
        norm_key = normalize_key(key)
        with self._lock:
            return norm_key in self._cache

    def all(self) -> Dict[str, str]:
        """Return a copy of all stored memories."""
        with self._lock:
            return dict(self._cache)

    def clear(self) -> None:
        """Clear all stored memories."""
        with self._lock:
            self._cache.clear()
            self._persist()
        logger.info("Cleared all memories.")
