"""Cloud handler managing offloaded complex questions to CloudClient."""

import logging
from typing import Optional

from app.cloud.client import CloudClient
from app.handlers.base import BaseHandler
from app.models.responses import AgentResponse
from app.router.models import RouteType, RoutingDecision

logger = logging.getLogger(__name__)


class CloudHandler(BaseHandler):
    """Handles complex queries routed to the cloud AI."""

    def __init__(self, cloud_client: Optional[CloudClient] = None) -> None:
        self.cloud_client = cloud_client or CloudClient()

    @property
    def name(self) -> str:
        return "CloudHandler"

    async def handle(self, query: str, decision: RoutingDecision) -> AgentResponse:
        """Forward query to cloud client or return safe Phase 1 stub."""
        # If router already executed cloud LLM and passed a response
        raw_router_response: Optional[str] = decision.raw_response.get("response")
        if raw_router_response and isinstance(raw_router_response, str) and raw_router_response.strip():
            return AgentResponse(
                text=raw_router_response.strip(),
                route=RouteType.CLOUD,
                handler=self.name,
                intent=decision.intent or "CLOUD_LLM_RESPONSE",
                metadata={"source": "router_payload"},
                success=True,
            )

        try:
            cloud_text = await self.cloud_client.generate(query)
            return AgentResponse(
                text=cloud_text,
                route=RouteType.CLOUD,
                handler=self.name,
                intent=decision.intent or "CLOUD_DISPATCH",
                metadata={"stub": self.cloud_client.stub_mode},
                success=True,
            )
        except Exception as exc:
            logger.error("CloudHandler failed while communicating with cloud client: %s", exc)
            return AgentResponse(
                text="I couldn't reach my cloud friend right now. Let's try something else!",
                route=RouteType.CLOUD,
                handler=self.name,
                intent="CLOUD_UNAVAILABLE",
                metadata={"error": str(exc)},
                success=False,
            )
