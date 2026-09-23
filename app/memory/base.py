"""Abstract base classes and data models for the device-side memory subsystem."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence
from pydantic import BaseModel, Field


class MemoryContext(BaseModel):
    """Structured result of resolving requested memory keys against storage."""

    hits: Dict[str, str] = Field(default_factory=dict, description="Requested keys that were found and their values")
    misses: List[str] = Field(default_factory=list, description="Requested keys that were not found in memory")
    resolved: Dict[str, Optional[str]] = Field(default_factory=dict, description="Every requested key mapped to value or None")
    is_complete: bool = Field(default=False, description="True if and only if all requested keys were found")


class BaseMemoryStore(ABC):
    """Abstract interface for device-side key-value memory storage.

    Decouples higher-level agent routing and retrieval logic from the physical
    storage medium (e.g. desktop JSON file, ESP32 NVS, SPIFFS, or companion sync).
    """

    @abstractmethod
    def get(self, key: str) -> Optional[str]:
        """Retrieve a stored memory value by key, or None if absent."""
        pass

    @abstractmethod
    def set(self, key: str, value: str) -> None:
        """Store or update a memory value."""
        pass

    @abstractmethod
    def get_many(self, keys: Sequence[str]) -> Dict[str, Optional[str]]:
        """Retrieve multiple memory values by keys in a single operation.

        Returns a dictionary mapping every requested key to its stored value,
        or None if not found.
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a memory item by key. Returns True if deleted, False if not found."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if a memory key exists in storage."""
        pass

    @abstractmethod
    def all(self) -> Dict[str, str]:
        """Return a copy of all stored key-value pairs."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Wipe all stored key-value items."""
        pass

    def resolve(self, keys: Sequence[str]) -> MemoryContext:
        """Resolve requested keys into a structured MemoryContext.

        Args:
            keys: Sequence of requested memory keys.

        Returns:
            MemoryContext: Structured container of hits, misses, resolved map, and completeness.
        """
        resolved = self.get_many(keys)
        hits = {k: v for k, v in resolved.items() if v is not None}
        misses = [k for k, v in resolved.items() if v is None]
        return MemoryContext(
            hits=hits,
            misses=misses,
            resolved=resolved,
            is_complete=len(misses) == 0,
        )
