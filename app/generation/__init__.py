"""Answer generation package for toy-agent (Phase 4)."""

from app.generation.base import AnswerGenerator, MockAnswerGenerator
from app.generation.cloud import CloudAnswerGenerator, MockCloudAnswerGenerator
from app.generation.exceptions import (
    CloudConfigurationError,
    CloudGenerationError,
    CloudTimeoutError,
    EmptyResponseError,
    GenerationError,
    LocalGenerationError,
)
from app.generation.local import LocalAnswerGenerator, format_memory_fact

__all__ = [
    "AnswerGenerator",
    "LocalAnswerGenerator",
    "CloudAnswerGenerator",
    "MockAnswerGenerator",
    "MockCloudAnswerGenerator",
    "format_memory_fact",
    "GenerationError",
    "LocalGenerationError",
    "CloudConfigurationError",
    "CloudGenerationError",
    "CloudTimeoutError",
    "EmptyResponseError",
]
