"""Router package: clients, models, and exceptions for slm-router communication."""

from app.router.client import RouterClient
from app.router.exceptions import (
    HandlerError,
    RouterConnectionError,
    RouterError,
    RouterResponseError,
    RouterTimeoutError,
    ToyAgentError,
)
from app.router.mock import MockRouterClient
from app.router.models import RouteType, RoutingDecision

__all__ = [
    "RouterClient",
    "MockRouterClient",
    "RouteType",
    "RoutingDecision",
    "ToyAgentError",
    "RouterError",
    "RouterConnectionError",
    "RouterTimeoutError",
    "RouterResponseError",
    "HandlerError",
]
