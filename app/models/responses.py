"""Standardized agent response model."""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from app.router.models import RouteType


class AgentResponse(BaseModel):
    """Normalized response delivered by the toy agent."""

    text: str = Field(description="Child-facing text response")
    route: RouteType = Field(description="Route executed")
    handler: str = Field(description="Name of the executing handler")
    intent: Optional[str] = Field(default=None, description="Intent associated with the response")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Execution metadata for tracing")
    success: bool = Field(default=True, description="Whether execution completed without error")
