"""Base handler abstract interface."""

from abc import ABC, abstractmethod

from app.models.responses import AgentResponse
from app.router.models import RoutingDecision


class BaseHandler(ABC):
    """Abstract base handler for processing routed queries."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the handler for observability and logging."""
        pass

    @abstractmethod
    async def handle(self, query: str, decision: RoutingDecision) -> AgentResponse:
        """Handle a routed query and return an AgentResponse.

        Args:
            query: The original user text.
            decision: The parsed routing decision from the router.

        Returns:
            AgentResponse: Child-facing response object.
        """
        pass
