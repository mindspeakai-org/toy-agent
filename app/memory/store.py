"""Lightweight key-value memory store with atomic file persistence."""

import json
import logging
import os
from pathlib import Path
import threading
from typing import Dict, Optional, Sequence, Union

from app.config.settings import get_settings
from app.memory.base import BaseMemoryStore

logger = logging.getLogger(__name__)

# Deterministic canonical alias mapping at the storage-key level.
CANONICAL_ALIASES: Dict[str, str] = {
    "name": "child_name",
    "child_name": "name",
}


def normalize_key(key: str) -> str:
    """Normalize keys to lower snake_case for consistent retrieval."""
    return "_".join(key.strip().lower().split())


class MemoryStore(BaseMemoryStore):
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
                        self._cache = {normalize_key(str(k)): str(v) for k, v in data.items()}
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

    def _lookup(self, norm_key: str) -> Optional[str]:
        """Look up value by key or its alias without duplication (caller must hold self._lock)."""
        val = self._cache.get(norm_key)
        if val is not None:
            return val
        alias = CANONICAL_ALIASES.get(norm_key)
        if alias and alias in self._cache:
            return self._cache[alias]
        return None

    def get(self, key: str) -> Optional[str]:
        """Retrieve a stored memory value by key, resolving canonical aliases."""
        norm_key = normalize_key(key)
        with self._lock:
            return self._lookup(norm_key)

    def get_many(self, keys: Sequence[str]) -> Dict[str, Optional[str]]:
        """Retrieve multiple memory values by keys in a single operation.

        Normalizes every key, resolves aliases against storage, and returns
        a dictionary mapping the requested key (normalized) to its stored value
        or None if not found.
        """
        results: Dict[str, Optional[str]] = {}
        with self._lock:
            for raw_key in keys:
                norm_key = normalize_key(raw_key)
                results[norm_key] = self._lookup(norm_key)
        return results

    def set(self, key: str, value: str) -> None:
        """Store or update a memory value. Replaces any existing alias entry to prevent duplicates."""
        norm_key = normalize_key(key)
        norm_val = value.strip()
        alias = CANONICAL_ALIASES.get(norm_key)
        with self._lock:
            # If alias already exists in cache, update that entry to avoid duplicates
            if alias and alias in self._cache and norm_key not in self._cache:
                target_key = alias
            else:
                target_key = norm_key
            self._cache[target_key] = norm_val
            self._persist()
        logger.info("Stored memory: '%s' = '%s'", target_key, norm_val)

    def delete(self, key: str) -> bool:
        """Delete a memory item. Returns True if deleted, False if not found."""
        norm_key = normalize_key(key)
        alias = CANONICAL_ALIASES.get(norm_key)
        with self._lock:
            deleted = False
            if norm_key in self._cache:
                del self._cache[norm_key]
                deleted = True
            elif alias and alias in self._cache:
                del self._cache[alias]
                deleted = True

            if deleted:
                self._persist()
                logger.info("Deleted memory key: '%s'", norm_key)
                return True
            return False

    def exists(self, key: str) -> bool:
        """Check if a memory key exists (or its alias)."""
        norm_key = normalize_key(key)
        with self._lock:
            return self._lookup(norm_key) is not None

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
