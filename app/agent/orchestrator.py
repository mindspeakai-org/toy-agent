"""Central agent orchestrator coordinating routing, memory, and handlers."""

import logging
from typing import Optional, Union

from app.cloud.client import CloudClient
from app.handlers.cloud import CloudHandler
from app.handlers.command import CommandHandler
from app.handlers.local import LocalHandler
from app.handlers.memory import MemoryHandler
from app.memory.store import MemoryStore
from app.models.responses import AgentResponse
from app.router.client import RouterClient
from app.router.exceptions import (
    RouterConnectionError,
    RouterError,
    RouterResponseError,
    RouterTimeoutError,
)
from app.router.mock import MockRouterClient
from app.router.models import RouteType, RoutingDecision

logger = logging.getLogger("toy_agent.orchestrator")


class AgentOrchestrator:
    """Central orchestrator for the AI Toy Agent.

    Coordinates the complete conversational pipeline:
    1. Receives user text.
    2. Invokes external RouterClient for query classification.
    3. Selects and executes the appropriate handler (LOCAL, MEMORY, COMMAND, CLOUD).
    4. Enforces robust fallbacks for all network/router failure modes.
    5. Normalizes the output into an AgentResponse for the child.
    """

    def __init__(
        self,
        router_client: Optional[Union[RouterClient, MockRouterClient]] = None,
        memory_store: Optional[MemoryStore] = None,
        local_handler: Optional[LocalHandler] = None,
        memory_handler: Optional[MemoryHandler] = None,
        command_handler: Optional[CommandHandler] = None,
        cloud_handler: Optional[CloudHandler] = None,
    ) -> None:
        self.router = router_client or RouterClient()
        self.memory = memory_store or MemoryStore()

        self.local_handler = local_handler or LocalHandler()
        self.memory_handler = memory_handler or MemoryHandler(self.memory)
        self.command_handler = command_handler or CommandHandler()
        self.cloud_handler = cloud_handler or CloudHandler(CloudClient())

    async def process(self, text: str) -> AgentResponse:
        """Process user text through the routing and handler pipeline.

        Guarantees that a child-safe AgentResponse is returned even in the event
        of network timeouts, router connection drops, or malformed data.
        """
        cleaned_text = text.strip()
        if not cleaned_text:
            return AgentResponse(
                text="I didn't hear anything! What's on your mind?",
                route=RouteType.UNKNOWN,
                handler="Orchestrator",
                intent="EMPTY_INPUT",
                metadata={},
                success=True,
            )

        logger.info("================ PIPELINE START ================")
        logger.info("USER_INPUT: %s", cleaned_text)

        # Step 1: Query Router
        try:
            decision = await self.router.route(cleaned_text)
        except RouterTimeoutError as exc:
            logger.error("ROUTER_ERROR: Timeout occurred - %s", exc)
            return self._build_fallback(
                "I'm taking a little too long to think right now. Could you ask me again?",
                reason="ROUTER_TIMEOUT",
                error=str(exc),
            )
        except RouterConnectionError as exc:
            logger.error("ROUTER_ERROR: Connection failed - %s", exc)
            return self._build_fallback(
                "I'm having trouble connecting right now. Let's try again in a moment!",
                reason="ROUTER_UNAVAILABLE",
                error=str(exc),
            )
        except RouterResponseError as exc:
            logger.error("ROUTER_ERROR: Malformed or error response - %s", exc)
            return self._build_fallback(
                "I had trouble understanding that. Let's try something else!",
                reason="ROUTER_MALFORMED_RESPONSE",
                error=str(exc),
            )
        except Exception as exc:
            logger.exception("ROUTER_ERROR: Unexpected error - %s", exc)
            return self._build_fallback(
                "I'm having trouble understanding that right now.",
                reason="UNEXPECTED_ROUTER_ERROR",
                error=str(exc),
            )

        # Step 2: Log Routing Decision
        logger.info("ROUTER: %s", decision.route.value)
        if decision.intent:
            logger.info("INTENT: %s", decision.intent)
        if decision.confidence is not None:
            logger.info("CONFIDENCE: %.2f", decision.confidence)
        if decision.key:
            logger.info("MEMORY_KEY: %s", decision.key)

        # Step 3: Dispatch to Handler
        try:
            if decision.route == RouteType.MEMORY:
                logger.info("HANDLER: MemoryHandler")
                response = await self.memory_handler.handle(cleaned_text, decision)

            elif decision.route == RouteType.LOCAL:
                logger.info("HANDLER: LocalHandler")
                response = await self.local_handler.handle(cleaned_text, decision)

            elif decision.route == RouteType.COMMAND:
                logger.info("HANDLER: CommandHandler")
                response = await self.command_handler.handle(cleaned_text, decision)

            elif decision.route == RouteType.CLOUD:
                logger.info("HANDLER: CloudHandler")
                response = await self.cloud_handler.handle(cleaned_text, decision)

            else:
                logger.warning("HANDLER: Unhandled route [%s]", decision.route)
                response = AgentResponse(
                    text="I'm not quite sure how to answer that yet, but I'm learning every day!",
                    route=RouteType.UNKNOWN,
                    handler="FallbackHandler",
                    intent="UNHANDLED_ROUTE",
                    metadata={"original_route": decision.route.value},
                    success=False,
                )

        except Exception as exc:
            logger.exception("HANDLER_ERROR: Execution failed in handler for route %s: %s", decision.route, exc)
            return self._build_fallback(
                "Oops, something went a little wobbly on my end. Can you say that again?",
                reason="HANDLER_EXECUTION_FAILURE",
                error=str(exc),
            )

        logger.info("RESPONSE: %s", response.text)
        logger.info("================= PIPELINE END =================")
        return response

    def _build_fallback(self, message: str, reason: str, error: str) -> AgentResponse:
        """Construct child-safe fallback response while preserving debug details in metadata."""
        logger.info("FALLBACK_RESPONSE: %s (Reason: %s)", message, reason)
        logger.info("================= PIPELINE END =================")
        return AgentResponse(
            text=message,
            route=RouteType.UNKNOWN,
            handler="FallbackHandler",
            intent=reason,
            metadata={"reason": reason, "technical_error": error},
            success=False,
        )

    async def aclose(self) -> None:
        """Cleanly close network and resource handles."""
        if hasattr(self.router, "aclose"):
            await self.router.aclose()
