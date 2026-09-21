"""Local handler for deterministic, on-device child-friendly responses."""

import logging
import re
from typing import Optional

from app.handlers.base import BaseHandler
from app.models.responses import AgentResponse
from app.router.models import RouteType, RoutingDecision

logger = logging.getLogger(__name__)


class LocalHandler(BaseHandler):
    """Handles basic conversational chit-chat, greetings, goodbyes, and facts locally.

    Deterministic and lightweight. No heavy LLM required for Phase 1.
    """

    @property
    def name(self) -> str:
        return "LocalHandler"

    async def handle(self, query: str, decision: RoutingDecision) -> AgentResponse:
        """Produce a friendly deterministic text response."""
        cleaned = query.strip()
        lower = cleaned.lower()

        # If the external router provided a pre-generated local SLM response, we can honor it
        raw_router_response: Optional[str] = decision.raw_response.get("response")
        if raw_router_response and isinstance(raw_router_response, str) and raw_router_response.strip():
            return AgentResponse(
                text=raw_router_response.strip(),
                route=RouteType.LOCAL,
                handler=self.name,
                intent=decision.intent or "LOCAL_SLM_RESPONSE",
                metadata={"source": "router_payload"},
                success=True,
            )

        # 1. Greetings
        if any(w in lower for w in ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]):
            response_text = "Hello there! I'm your toy friend. What would you like to talk about today?"
            intent = "GREETING"

        # 2. Goodbyes
        elif any(w in lower for w in ["bye", "goodbye", "see you", "good night", "talk to you later"]):
            response_text = "Goodbye! I had fun talking with you. See you soon!"
            intent = "FAREWELL"

        # 3. Identity & capabilities
        elif "who are you" in lower or "what is your name" in lower or "what's your name" in lower:
            response_text = "I'm your AI toy buddy! I can answer questions, play games, and remember your favorite things."
            intent = "IDENTITY"

        elif "how are you" in lower:
            response_text = "I'm doing great and happy to talk with you! How are you doing?"
            intent = "CHIT_CHAT"

        elif "tell me a joke" in lower or "joke" in lower:
            response_text = "Why did the teddy bear say no to dessert? Because she was already stuffed!"
            intent = "JOKE"

        # 4. Simple Math
        elif re.search(r"2\s*\+\s*2", lower):
            response_text = "2 + 2 is 4!"
            intent = "MATH"

        # 5. Default friendly response
        else:
            response_text = f"That's so interesting! I love chatting with you."
            intent = "CONVERSATION"

        logger.debug("LocalHandler generated response for '%s' -> '%s'", cleaned, response_text)

        return AgentResponse(
            text=response_text,
            route=RouteType.LOCAL,
            handler=self.name,
            intent=intent,
            metadata={"rule_intent": intent},
            success=True,
        )
