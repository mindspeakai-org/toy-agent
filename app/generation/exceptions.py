"""Domain exceptions for answer generation (Local and Cloud)."""

from app.router.exceptions import ToyAgentError


class GenerationError(ToyAgentError):
    """Base exception for all answer generation failures."""


class LocalGenerationError(GenerationError):
    """Raised when the local model fails to load, tokenize, or execute generation."""


class CloudConfigurationError(GenerationError):
    """Raised when the cloud provider is misconfigured (e.g. missing API key)."""


class CloudGenerationError(GenerationError):
    """Raised when the cloud API returns an error or malformed payload."""


class CloudTimeoutError(CloudGenerationError):
    """Raised when the cloud API request times out."""


class EmptyResponseError(GenerationError):
    """Raised when a generator produces an empty or blank response."""
