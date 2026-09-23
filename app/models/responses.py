"""Standardized agent response model."""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from app.memory.base import MemoryContext
from app.router.models import ProcessingType, RouteType, RoutingDecision


class AgentResponse(BaseModel):
    """Normalized response delivered by the toy agent."""

    text: str = Field(default="", description="Child-facing text response or decision message")
    processing: Optional[ProcessingType] = Field(default=None, description="Target processing destination")
    decision: Optional[RoutingDecision] = Field(default=None, description="Structured routing decision from router")
    memory_context: Optional[MemoryContext] = Field(default=None, description="Resolved device-side memory context if memory was required")
    route: Optional[RouteType] = Field(default=None, description="Legacy route identifier for compatibility")
    handler: str = Field(default="Router", description="Name of the executing handler")
    intent: Optional[str] = Field(default=None, description="Intent associated with the response")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Execution metadata for tracing")
    success: bool = Field(default=True, description="Whether execution completed without error")
