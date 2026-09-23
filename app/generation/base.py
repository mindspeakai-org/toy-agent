"""Abstract base interface for answer generation."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from app.memory.base import MemoryContext


class AnswerGenerator(ABC):
    """Abstract interface for producing child-facing answer text from queries and optional memory."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the generator provider."""

    @abstractmethod
    async def generate(
        self,
        query: str,
        memory_context: Optional[MemoryContext] = None,
    ) -> str:
        """Generate response text for the given user query and optional memory context.

        Args:
            query: The user query text.
            memory_context: Optional device-side MemoryContext retrieved from MemoryStore.

        Returns:
            str: Generated natural language answer text.

        Raises:
            GenerationError: If generation fails.
        """


class MockAnswerGenerator(AnswerGenerator):
    """Deterministic mock answer generator for tests and simulated environments."""

    def __init__(
        self,
        canned_response: str = "Mock answer response.",
        provider_name: str = "MockAnswerGenerator",
    ) -> None:
        self._canned_response = canned_response
        self._provider_name = provider_name
        self.call_history: List[Dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self._provider_name

    def set_response(self, response: str) -> None:
        """Set next canned response."""
        self._canned_response = response

    async def generate(
        self,
        query: str,
        memory_context: Optional[MemoryContext] = None,
    ) -> str:
        """Record invocation and return canned response."""
        self.call_history.append({
            "query": query,
            "memory_context": memory_context,
        })
        return self._canned_response
