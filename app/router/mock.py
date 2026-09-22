"""Mock router client for testing without the external slm-router service."""

import re
from typing import Any, Dict, List, Optional

from app.router.exceptions import RouterError
from app.router.models import RouteType, RoutingDecision


class MockRouterClient:
    """Deterministic mock router for testing and standalone local development."""

    def __init__(self, default_route: RouteType = RouteType.LOCAL) -> None:
        self.default_route = default_route
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
        """Return a deterministic routing decision based on rule-based patterns."""
        self.call_history.append(text)

        if self._injected_error:
            raise self._injected_error

        cleaned = text.strip()
        lower = cleaned.lower()

        # Check explicit custom mocks first
        if lower in self._custom_responses:
            return self._custom_responses[lower]

        if not cleaned:
            return RoutingDecision(route=RouteType.UNKNOWN, intent="EMPTY_QUERY", query=text)

        # 1. Memory Update Pattern: "My <key> is <value>" or "Remember (that) my <key> is <value>"
        mem_update_match = re.search(
            r"(?:remember\s+(?:that\s+)?)?my\s+([a-zA-Z0-9_\s]+?)\s+is\s+(.+)",
            lower,
        )
        if mem_update_match:
            raw_key = mem_update_match.group(1).strip()
            raw_val = mem_update_match.group(2).strip().rstrip(".!?,")
            # Strip leading articles: "a tiger" -> "tiger"
            for article in ("a ", "an ", "the "):
                if raw_val.startswith(article):
                    raw_val = raw_val[len(article):].strip()
                    break
            key = "_".join(raw_key.split())
            return RoutingDecision(
                route=RouteType.MEMORY,
                intent="MEMORY_UPDATE",
                key=key,
                value=raw_val,
                confidence=0.98,
                query=cleaned,
                handler="MockRouter",
            )

        # 2. Memory Query Pattern: "What is my <key>?" or "What's my <key>?"
        mem_query_match = re.search(
            r"what(?:\s+is|\'s)\s+my\s+([a-zA-Z0-9_\s]+?)(?:\?|$)",
            lower,
        )
        if mem_query_match:
            raw_key = mem_query_match.group(1).strip()
            key = "_".join(raw_key.split())
            return RoutingDecision(
                route=RouteType.MEMORY,
                intent="MEMORY_QUERY",
                key=key,
                confidence=0.96,
                query=cleaned,
                handler="MockRouter",
            )

        # Direct name query
        if lower in {"who am i", "who am i?", "what is my name", "what's my name", "what is my name?"}:
            return RoutingDecision(
                route=RouteType.MEMORY,
                intent="MEMORY_QUERY",
                key="name",
                confidence=0.97,
                query=cleaned,
                handler="MockRouter",
            )

        # 3. Command Patterns: Device actions
        command_keywords = [
            "turn on", "turn off", "turn up", "turn down", "volume",
            "stop", "pause", "resume", "play music", "be quiet",
            "sleep", "shut down", "light", "lights",
        ]
        if any(keyword in lower for keyword in command_keywords):
            return RoutingDecision(
                route=RouteType.COMMAND,
                intent="DEVICE_ACTION",
                confidence=0.94,
                query=cleaned,
                handler="MockRouter",
            )

        # 4. Cloud Patterns: Complex queries, deep science, long stories, high complexity
        cloud_triggers = [
            "why does", "why do", "why is", "how does", "how do",
            "tell me a story", "explain", "analyze", "research",
            "universe", "quantum", "photosynthesis", "solar system",
            "black hole", "relativity",
        ]
        if any(trigger in lower for trigger in cloud_triggers):
            return RoutingDecision(
                route=RouteType.CLOUD,
                intent="KNOWLEDGE_QUERY",
                confidence=0.92,
                query=cleaned,
                handler="MockRouter",
            )

        # 5. Local Patterns: Chit-chat, greetings, simple math, small talk
        local_triggers = [
            "hello", "hi", "hey", "good morning", "good evening", "bye",
            "goodbye", "tell me a joke", "how are you", "who are you",
            "what can you do", "what's your name", "what is your name",
            "2 + 2", "help",
        ]
        if any(trigger in lower for trigger in local_triggers):
            return RoutingDecision(
                route=RouteType.LOCAL,
                intent="CONVERSATION",
                confidence=0.95,
                query=cleaned,
                handler="MockRouter",
            )

        # Default fallback
        return RoutingDecision(
            route=self.default_route,
            intent="DEFAULT_DISPATCH",
            confidence=0.50,
            query=cleaned,
            handler="MockRouter",
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
