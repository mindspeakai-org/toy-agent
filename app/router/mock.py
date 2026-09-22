"""Mock router client for testing without the external slm-router service."""

from typing import Any, Dict, List, Optional

from app.router.models import MemoryRequest, ProcessingType, RoutingDecision


class MockRouterClient:
    """Deterministic mock router conforming to the Phase 2 SLM-Router contract.

    Produces strictly routing decisions without generating answers, retrieving memory,
    or executing commands.
    """

    def __init__(self, default_processing: ProcessingType = ProcessingType.LOCAL) -> None:
        self.default_processing = default_processing
        self._custom_responses: Dict[str, RoutingDecision] = {}
        self._injected_error: Optional[Exception] = None
        self.call_history: List[str] = []

    def inject_error(self, error: Optional[Exception]) -> None:
        """Inject an exception to be raised on subsequent route() calls."""
        self._injected_error = error

    def set_mock_response(self, text: str, decision: RoutingDecision) -> None:
        """Register a canned decision for an exact text match."""
        self._custom_responses[text.strip().lower()] = decision

    async def route(self, text: str) -> RoutingDecision:
        """Return a deterministic routing decision based on the Phase 2 contract."""
        self.call_history.append(text)

        if self._injected_error:
            raise self._injected_error

        cleaned = text.strip()
        lower = cleaned.lower()

        if lower in self._custom_responses:
            return self._custom_responses[lower]

        if not cleaned:
            return RoutingDecision(
                processing=ProcessingType.LOCAL,
                memory_required=False,
                memory_request=None,
                query=text,
            )

        # 1. Multiple memories query pattern
        if ("name" in lower and ("animal" in lower or "color" in lower)) or "who am i and" in lower:
            return RoutingDecision(
                processing=ProcessingType.LOCAL,
                memory_required=True,
                memory_request=MemoryRequest(keys=["child_name", "favorite_animal"]),
                query=cleaned,
            )

        # 2. Single memory query pattern
        if "favorite animal" in lower or "my animal" in lower:
            return RoutingDecision(
                processing=ProcessingType.LOCAL,
                memory_required=True,
                memory_request=MemoryRequest(keys=["favorite_animal"]),
                query=cleaned,
            )

        if "favorite color" in lower or "my color" in lower:
            return RoutingDecision(
                processing=ProcessingType.LOCAL,
                memory_required=True,
                memory_request=MemoryRequest(keys=["favorite_color"]),
                query=cleaned,
            )

        if "what is my name" in lower or "what's my name" in lower or lower in {"who am i", "who am i?"}:
            return RoutingDecision(
                processing=ProcessingType.LOCAL,
                memory_required=True,
                memory_request=MemoryRequest(keys=["child_name"]),
                query=cleaned,
            )

        # 3. Cloud query patterns (complex topics, external information, deep science)
        cloud_triggers = [
            "why does", "why do", "why is", "how does", "how do",
            "weather", "tell me a story", "explain", "analyze", "research",
            "universe", "quantum", "photosynthesis", "solar system",
        ]
        if any(trigger in lower for trigger in cloud_triggers):
            return RoutingDecision(
                processing=ProcessingType.CLOUD,
                memory_required=False,
                memory_request=None,
                query=cleaned,
            )

        # 4. Local fallback (commands, greetings, jokes, simple chit-chat)
        return RoutingDecision(
            processing=self.default_processing,
            memory_required=False,
            memory_request=None,
            query=cleaned,
        )

    async def check_health(self) -> Dict[str, Any]:
        """Mock health check."""
        if self._injected_error:
            raise self._injected_error
        return {"status": "healthy", "model": "Mock-Qwen"}

    async def is_healthy(self) -> bool:
        """Mock health probe."""
        return self._injected_error is None

    async def aclose(self) -> None:
        """No-op for mock client."""
        pass
