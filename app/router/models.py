"""Data models for routing decisions and router API communication."""

from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, field_validator


class RouteType(str, Enum):
    """Supported routing destinations."""

    LOCAL = "LOCAL"
    MEMORY = "MEMORY"
    COMMAND = "COMMAND"
    CLOUD = "CLOUD"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def _missing_(cls, value: object) -> "RouteType":
        if isinstance(value, str):
            val_upper = value.strip().upper()
            for member in cls:
                if member.value == val_upper:
                    return member
        return cls.UNKNOWN


class RoutingDecision(BaseModel):
    """Structured routing decision returned by RouterClient."""

    route: RouteType = Field(default=RouteType.UNKNOWN, description="Target execution route")
    intent: Optional[str] = Field(default=None, description="Fine-grained intent identifier")
    key: Optional[str] = Field(default=None, description="Key for memory operations")
    value: Optional[str] = Field(default=None, description="Value for memory write operations")
    confidence: Optional[float] = Field(default=None, description="Confidence score between 0.0 and 1.0")
    query: Optional[str] = Field(default=None, description="Original query evaluated")
    handler: Optional[str] = Field(default=None, description="Handler name reported by router")
    model_name: Optional[str] = Field(default=None, description="Model identifier used by router")
    timings: Dict[str, float] = Field(default_factory=dict, description="Execution timings reported by router")
    http_latency: Optional[float] = Field(default=None, description="HTTP roundtrip latency in seconds")
    raw_response: Dict[str, Any] = Field(default_factory=dict, description="Raw payload from router")

    @field_validator("route", mode="before")
    @classmethod
    def normalize_route(cls, v: Any) -> RouteType:
        if isinstance(v, RouteType):
            return v
        if isinstance(v, str):
            try:
                return RouteType(v)
            except ValueError:
                return RouteType.UNKNOWN
        return RouteType.UNKNOWN

    @classmethod
    def from_payload(
        cls,
        data: Dict[str, Any],
        query: Optional[str] = None,
        http_latency: Optional[float] = None,
    ) -> "RoutingDecision":
        """Create a RoutingDecision from arbitrary router API JSON payload."""
        route_raw = data.get("route") or data.get("destination") or "UNKNOWN"
        intent_raw = data.get("intent") or data.get("action") or data.get("processing_type")
        key_raw = data.get("key") or data.get("memory_key")
        value_raw = data.get("value") or data.get("memory_value")
        confidence_raw = data.get("confidence")
        handler_raw = data.get("handler")
        model_raw = data.get("model")
        timings_raw = data.get("timings") or {}

        # Convert confidence safely if present
        confidence_float = None
        if confidence_raw is not None:
            try:
                confidence_float = float(confidence_raw)
            except (ValueError, TypeError):
                confidence_float = None

        # Clean timings
        clean_timings: Dict[str, float] = {}
        if isinstance(timings_raw, dict):
            for k, val in timings_raw.items():
                try:
                    clean_timings[str(k)] = float(val)
                except (ValueError, TypeError):
                    pass

        return cls(
            route=route_raw,
            intent=str(intent_raw) if intent_raw is not None else None,
            key=str(key_raw) if key_raw is not None else None,
            value=str(value_raw) if value_raw is not None else None,
            confidence=confidence_float,
            query=query or data.get("query"),
            handler=str(handler_raw) if handler_raw is not None else None,
            model_name=str(model_raw) if model_raw is not None else None,
            timings=clean_timings,
            http_latency=http_latency,
            raw_response=data,
        )
