"""Memory handler for reading and storing child preferences and context."""

import logging
import re
from typing import Optional, Tuple

from app.handlers.base import BaseHandler
from app.memory.store import MemoryStore, normalize_key
from app.models.responses import AgentResponse
from app.router.models import RouteType, RoutingDecision

logger = logging.getLogger(__name__)


class MemoryHandler(BaseHandler):
    """Handles memory queries and updates based on structured router decisions."""

    def __init__(self, memory_store: MemoryStore) -> None:
        self.memory = memory_store

    @property
    def name(self) -> str:
        return "MemoryHandler"

    def _extract_query_key(self, query: str) -> Optional[str]:
        """Heuristic fallback to extract a memory key from query text."""
        lower = query.lower().strip()
        # Pattern: "what is my <key>?"
        match = re.search(r"what(?:\s+is|\'s)\s+my\s+([a-zA-Z0-9_\s]+?)(?:\?|$)", lower)
        if match:
            return normalize_key(match.group(1))
        if lower in {"who am i", "who am i?", "what is my name", "what's my name"}:
            return "name"
        return None

    def _extract_update_key_value(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """Heuristic fallback to extract key and value from an update sentence."""
        lower = query.lower().strip()
        # Pattern: "my <key> is <value>"
        match = re.search(r"(?:remember\s+(?:that\s+)?)?my\s+([a-zA-Z0-9_\s]+?)\s+is\s+(.+)", lower)
        if match:
            raw_key = match.group(1).strip()
            raw_val = match.group(2).strip().rstrip(".!?,")
            for article in ("a ", "an ", "the "):
                if raw_val.startswith(article):
                    raw_val = raw_val[len(article):].strip()
                    break
            return normalize_key(raw_key), raw_val
        return None, None

    def _format_friendly_key(self, key: str) -> str:
        """Convert snake_case key into natural words (e.g. 'favorite_animal' -> 'favorite animal')."""
        return key.replace("_", " ")

    async def handle(self, query: str, decision: RoutingDecision) -> AgentResponse:
        """Process memory read or write."""
        intent = decision.intent or ""
        upper_intent = intent.upper()

        key = decision.key
        value = decision.value

        # Heuristic fallback if router did not populate key/value
        if not key:
            key = self._extract_query_key(query)
        if not key and not value:
            extracted_k, extracted_v = self._extract_update_key_value(query)
            if extracted_k and extracted_v:
                key = extracted_k
                value = extracted_v
                upper_intent = "MEMORY_UPDATE"

        # Determine operation mode: UPDATE vs QUERY
        is_update = "UPDATE" in upper_intent or "SET" in upper_intent or bool(value)

        if is_update:
            if key and value:
                self.memory.set(key, value)
                friendly_key = self._format_friendly_key(key)
                text = f"Got it! I'll remember that your {friendly_key} is {value}."
                return AgentResponse(
                    text=text,
                    route=RouteType.MEMORY,
                    handler=self.name,
                    intent="MEMORY_UPDATE",
                    metadata={"key": key, "value": value},
                    success=True,
                )
            else:
                return AgentResponse(
                    text="I wanted to remember that, but I couldn't quite catch what to remember. Could you tell me again?",
                    route=RouteType.MEMORY,
                    handler=self.name,
                    intent="MEMORY_UPDATE_FAILED",
                    metadata={"key": key},
                    success=False,
                )

        # Query path
        if not key:
            return AgentResponse(
                text="I'm trying to remember, but I'm not sure which memory you're asking about!",
                route=RouteType.MEMORY,
                handler=self.name,
                intent="MEMORY_QUERY_UNKNOWN_KEY",
                metadata={},
                success=False,
            )

        val = self.memory.get(key)
        friendly_key = self._format_friendly_key(key)

        if val is not None:
            # Memory Hit
            if key == "name":
                text = f"Your name is {val}!"
            else:
                text = f"Your {friendly_key} is {val}."
            logger.info("Memory hit: '%s' -> '%s'", key, val)
            return AgentResponse(
                text=text,
                route=RouteType.MEMORY,
                handler=self.name,
                intent="MEMORY_QUERY",
                metadata={"key": key, "value": val, "hit": True},
                success=True,
            )
        else:
            # Memory Miss
            logger.info("Memory miss: key '%s' not found", key)
            if key == "name":
                text = "I don't know your name yet! What should I call you?"
            else:
                text = f"I don't know your {friendly_key} yet! What is it?"
            return AgentResponse(
                text=text,
                route=RouteType.MEMORY,
                handler=self.name,
                intent="MEMORY_QUERY",
                metadata={"key": key, "hit": False},
                success=True,
            )
