"""Data models for routing decisions and router API communication."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class ProcessingType(str, Enum):
    """Supported processing destinations from SLM-Router."""

    LOCAL = "LOCAL"
    CLOUD = "CLOUD"


class MemoryRequest(BaseModel):
    """Structured request for memory fields required to answer query."""

    keys: List[str] = Field(..., description="List of semantic memory key identifiers")

    @field_validator("keys", mode="before")
    @classmethod
    def validate_keys(cls, v: Any) -> List[str]:
        if not isinstance(v, list):
            raise ValueError(f"keys must be a list, got {type(v).__name__}")
        if len(v) == 0:
            raise ValueError("keys list cannot be empty")
        cleaned_keys: List[str] = []
        for item in v:
            if not isinstance(item, str):
                raise ValueError(f"Each key in keys must be a string, got {type(item).__name__}")
            item_clean = item.strip()
            if not item_clean:
                raise ValueError("Key in keys cannot be an empty string")
            cleaned_keys.append(item_clean)
        return cleaned_keys


class RoutingDecision(BaseModel):
    """Structured routing decision returned by RouterClient conforming to the SLM-Router contract."""

    processing: ProcessingType = Field(..., description="Target processing destination (LOCAL or CLOUD)")
    memory_required: bool = Field(..., description="Whether memory retrieval is required")
    memory_request: Optional[MemoryRequest] = Field(default=None, description="Memory keys requested if memory_required is True")
    query: Optional[str] = Field(default=None, description="Original query evaluated")
    model_name: Optional[str] = Field(default=None, description="Model identifier if provided")
    timings: Dict[str, float] = Field(default_factory=dict, description="Execution timings reported by router")
    http_latency: Optional[float] = Field(default=None, description="HTTP roundtrip latency in seconds")
    raw_response: Dict[str, Any] = Field(default_factory=dict, description="Raw payload from router")

    # Legacy attributes retained for backward compatibility with isolated Phase 1 handlers
    intent: Optional[str] = Field(default=None, description="Legacy intent attribute for isolated handlers")
    key: Optional[str] = Field(default=None, description="Legacy key attribute for isolated memory handler")
    value: Optional[str] = Field(default=None, description="Legacy value attribute for isolated memory handler")

    @model_validator(mode="before")
    @classmethod
    def handle_legacy_kwargs(cls, data: Any) -> Any:
        """Adapt legacy route kwargs for isolated Phase 1 handlers if processing is omitted."""
        if isinstance(data, dict):
            if "processing" not in data and "route" in data:
                route_val = data["route"]
                if hasattr(route_val, "value"):
                    route_val = route_val.value
                route_str = str(route_val).upper()
                if route_str == "CLOUD":
                    data["processing"] = ProcessingType.CLOUD
                else:
                    data["processing"] = ProcessingType.LOCAL

                if "memory_required" not in data:
                    if route_str == "MEMORY" and data.get("key"):
                        data["memory_required"] = True
                        data["memory_request"] = {"keys": [str(data["key"])]}
                    else:
                        data["memory_required"] = False
                        data["memory_request"] = None
        return data

    @field_validator("processing", mode="before")
    @classmethod
    def validate_processing(cls, v: Any) -> ProcessingType:
        if isinstance(v, ProcessingType):
            return v
        if isinstance(v, str):
            v_upper = v.strip().upper()
            if v_upper in {ProcessingType.LOCAL.value, ProcessingType.CLOUD.value}:
                return ProcessingType(v_upper)
        raise ValueError(f"Invalid processing type: {v!r}. Must be 'LOCAL' or 'CLOUD'.")

    @field_validator("memory_required", mode="before")
    @classmethod
    def validate_memory_required(cls, v: Any) -> bool:
        if not isinstance(v, bool):
            raise ValueError(f"memory_required must be a strict boolean, got {type(v).__name__}")
        return v

    @model_validator(mode="after")
    def validate_memory_consistency(self) -> "RoutingDecision":
        if self.memory_required:
            if self.memory_request is None:
                raise ValueError("memory_request must be non-null when memory_required is true")
            if not self.memory_request.keys or len(self.memory_request.keys) == 0:
                raise ValueError("memory_request.keys must contain at least one key when memory_required is true")
        else:
            if self.memory_request is not None:
                raise ValueError("memory_request must be null when memory_required is false")
        return self

    @property
    def route(self) -> "RouteType":
        """Backward compatibility property returning equivalent RouteType."""
        if self.processing == ProcessingType.CLOUD:
            return RouteType.CLOUD
        return RouteType.LOCAL

    @classmethod
    def from_payload(
        cls,
        data: Dict[str, Any],
        query: Optional[str] = None,
        http_latency: Optional[float] = None,
    ) -> "RoutingDecision":
        """Create a RoutingDecision from arbitrary router API JSON payload with strict validation."""
        if not isinstance(data, dict):
            raise ValueError("Router payload must be a dictionary")

        timings_raw = data.get("timings") or {}
        clean_timings: Dict[str, float] = {}
        if isinstance(timings_raw, dict):
            for k, val in timings_raw.items():
                try:
                    clean_timings[str(k)] = float(val)
                except (ValueError, TypeError):
                    pass

        return cls(
            processing=data.get("processing"),
            memory_required=data.get("memory_required"),
            memory_request=data.get("memory_request"),
            query=query or data.get("query"),
            model_name=data.get("model") or data.get("model_name"),
            timings=clean_timings,
            http_latency=http_latency,
            raw_response=data,
        )


# Legacy RouteType retained for backwards compatibility with isolated Phase 1 handlers.
class RouteType(str, Enum):
    """Legacy route enum for backwards compatibility with isolated handlers."""

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
