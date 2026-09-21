"""Exceptions for router and agent operations."""


class ToyAgentError(Exception):
    """Base exception for all toy agent domain errors."""


class RouterError(ToyAgentError):
    """Base exception for errors communicating with the external router."""


class RouterConnectionError(RouterError):
    """Raised when the router service cannot be reached (e.g. connection refused)."""


class RouterTimeoutError(RouterError):
    """Raised when the router request times out."""


class RouterResponseError(RouterError):
    """Raised when the router returns an invalid status code or malformed JSON."""


class HandlerError(ToyAgentError):
    """Raised when a handler fails to process a routed request."""
