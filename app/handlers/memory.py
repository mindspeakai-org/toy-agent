"""Memory handler for resolving device-side memory based on structured router decisions."""

import logging
from typing import Optional

from app.handlers.base import BaseHandler
from app.memory.base import BaseMemoryStore, MemoryContext
from app.models.responses import AgentResponse
from app.router.models import RouteType, RoutingDecision

logger = logging.getLogger(__name__)


class MemoryHandler(BaseHandler):
    """Retrieves device-side memory context using keys determined by the router.

    Strictly consumes decision.memory_request.keys.
    Performs zero regex parsing, zero query heuristics, and zero natural language extraction.
    """

    def __init__(self, memory_store: BaseMemoryStore) -> None:
        self.memory = memory_store

    @property
    def name(self) -> str:
        return "MemoryHandler"

    def retrieve(self, decision: RoutingDecision) -> Optional[MemoryContext]:
        """Retrieve MemoryContext for a RoutingDecision if memory is required.

        Args:
            decision: Structured routing decision from RouterClient.

        Returns:
            Optional[MemoryContext]: MemoryContext if memory_required is True, None otherwise.
        """
        if not decision.memory_required:
            logger.debug("Memory retrieval bypassed: memory_required is False")
            return None

        if decision.memory_request is None or not decision.memory_request.keys:
            logger.warning("memory_required is True but memory_request has no keys")
            return None

        requested_keys = decision.memory_request.keys
        logger.info("[MEMORY RETRIEVAL] Resolving keys: %s", requested_keys)
        context = self.memory.resolve(requested_keys)
        logger.info(
            "[MEMORY CONTEXT] Hits: %s | Misses: %s | Complete: %s",
            list(context.hits.keys()),
            context.misses,
            context.is_complete,
        )
        return context

    async def handle(self, query: str, decision: RoutingDecision) -> AgentResponse:
        """Process memory retrieval for a routing decision."""
        context = self.retrieve(decision)

        summary_text = (
            f"Routing decision: {decision.processing.value} (memory: {len(context.hits)} hits, {len(context.misses)} misses)"
            if context
            else f"Routing decision: {decision.processing.value}"
        )

        return AgentResponse(
            text=summary_text,
            processing=decision.processing,
            decision=decision,
            memory_context=context,
            route=RouteType.MEMORY if decision.memory_required else RouteType(decision.processing.value),
            handler=self.name,
            metadata={
                "memory_required": decision.memory_required,
                "memory_context": context.model_dump() if context else None,
            },
            success=True,
        )
