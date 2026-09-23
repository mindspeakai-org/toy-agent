"""Unit tests for Phase 4 answer generators (Local and Cloud)."""

import json
from unittest.mock import AsyncMock, MagicMock
import httpx
import pytest

from app.generation.base import MockAnswerGenerator
from app.generation.cloud import CloudAnswerGenerator, MockCloudAnswerGenerator
from app.generation.exceptions import (
    CloudConfigurationError,
    CloudGenerationError,
    CloudTimeoutError,
    EmptyResponseError,
    LocalGenerationError,
)
from app.generation.local import LocalAnswerGenerator
from app.memory.base import MemoryContext


# --- Base & Mock Generator Tests ---

@pytest.mark.asyncio
async def test_mock_answer_generator_records_history() -> None:
    """Verify MockAnswerGenerator records queries and optional memory context."""
    gen = MockAnswerGenerator(canned_response="Hello, friend!")
    ctx = MemoryContext(hits={"name": "Alex"}, misses=[], resolved={"name": "Alex"})

    resp = await gen.generate("Who am I?", memory_context=ctx)
    assert resp == "Hello, friend!"
    assert len(gen.call_history) == 1
    assert gen.call_history[0]["query"] == "Who am I?"
    assert gen.call_history[0]["memory_context"] == ctx


# --- LocalAnswerGenerator Unit Tests & Memory Grounding ---

def test_format_memory_fact_deterministic_conversion() -> None:
    """Verify deterministic conversion of memory keys into natural-language statements."""
    from app.generation.local import format_memory_fact

    assert format_memory_fact("child_name", "Alex") == "The child's name is Alex."
    assert format_memory_fact("name", "Alex") == "The child's name is Alex."
    assert format_memory_fact("favorite_animal", "tiger") == "The child's favorite animal is tiger."
    assert format_memory_fact("favorite_color", "blue") == "The child's favorite color is blue."
    assert format_memory_fact("favorite_song", None) == "The child's favorite song is not known."
    assert format_memory_fact("age", "7") == "The child's age is 7."


def test_local_prompt_formatting_with_memory_hits_and_misses() -> None:
    """Verify LocalAnswerGenerator constructs grounded natural-language facts for hits and misses."""
    gen = LocalAnswerGenerator(model_name="Qwen/Qwen2.5-1.5B-Instruct")
    ctx = MemoryContext(
        hits={"child_name": "Alex", "favorite_animal": "tiger"},
        misses=["favorite_song"],
        resolved={"child_name": "Alex", "favorite_animal": "tiger", "favorite_song": None},
    )

    messages = gen._build_prompt_messages("What is my favorite animal and song?", ctx)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "Known device information:" in messages[0]["content"]
    assert "- The child's name is Alex." in messages[0]["content"]
    assert "- The child's favorite animal is tiger." in messages[0]["content"]
    assert "- The child's favorite song is not known." in messages[0]["content"]
    assert "Never invent personal information." in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "What is my favorite animal and song?"


def test_local_prompt_formatting_without_memory() -> None:
    """Verify LocalAnswerGenerator omits memory block when memory_context is None."""
    gen = LocalAnswerGenerator(model_name="Qwen/Qwen2.5-1.5B-Instruct")
    messages = gen._build_prompt_messages("Tell me a joke", memory_context=None)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "Known device information:" not in messages[0]["content"]
    assert messages[1]["content"] == "Tell me a joke"


def test_memory_isolation_and_prompt_grounding() -> None:
    """Verify LocalAnswerGenerator only grounds on the provided MemoryContext and cannot search memory itself."""
    gen = LocalAnswerGenerator(model_name="Qwen/Qwen2.5-1.5B-Instruct")
    # Verify generator does not have a reference to MemoryStore
    assert not hasattr(gen, "memory_store")
    assert not hasattr(gen, "store")

    # When supplied explicit MemoryContext, it grounds solely on that context
    ctx = MemoryContext(
        hits={"favorite_color": "green"},
        misses=[],
        resolved={"favorite_color": "green"},
    )
    messages = gen._build_prompt_messages("What color do I love?", ctx)
    system_text = messages[0]["content"]
    assert "- The child's favorite color is green." in system_text
    # Verify unrelated memory items do not appear
    assert "tiger" not in system_text
    assert "Alex" not in system_text


@pytest.mark.asyncio
async def test_local_generation_model_load_failure() -> None:
    """Verify LocalGenerationError is raised when model fails to load."""
    gen = LocalAnswerGenerator(model_name="nonexistent/fake-model-path-123")
    with pytest.raises(LocalGenerationError):
        await gen.generate("Hello")


# --- CloudAnswerGenerator Unit Tests (ZERO real Gemini calls) ---

@pytest.mark.asyncio
async def test_cloud_generator_missing_api_key() -> None:
    """Verify CloudConfigurationError is raised when GEMINI_API_KEY is not configured."""
    gen = CloudAnswerGenerator(api_key=None)
    # Ensure api_key is None or empty
    gen.api_key = ""
    with pytest.raises(CloudConfigurationError) as exc_info:
        await gen.generate("What is the weather today?")
    assert "GEMINI_API_KEY is not configured" in str(exc_info.value)


@pytest.mark.asyncio
async def test_cloud_generator_successful_gemini_response() -> None:
    """Verify CloudAnswerGenerator parses standard Gemini JSON candidates correctly."""
    mock_transport = httpx.MockTransport(
        lambda request: httpx.Response(
            status_code=200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "Today's weather is sunny and warm!"}
                            ]
                        }
                    }
                ]
            },
        )
    )
    mock_client = httpx.AsyncClient(transport=mock_transport)
    gen = CloudAnswerGenerator(
        api_key="mock_test_key_12345",
        model="gemini-1.5-flash",
        http_client=mock_client,
    )

    try:
        response_text = await gen.generate("What is the weather today?")
        assert response_text == "Today's weather is sunny and warm!"
    finally:
        await mock_client.aclose()


@pytest.mark.asyncio
async def test_cloud_generator_http_error_handling() -> None:
    """Verify CloudGenerationError is raised when Gemini returns a non-200 status code."""
    mock_transport = httpx.MockTransport(
        lambda request: httpx.Response(
            status_code=500,
            text="Internal Server Error in Gemini API",
        )
    )
    mock_client = httpx.AsyncClient(transport=mock_transport)
    gen = CloudAnswerGenerator(
        api_key="mock_test_key_12345",
        http_client=mock_client,
    )

    try:
        with pytest.raises(CloudGenerationError) as exc_info:
            await gen.generate("Tell me about stars")
        assert "status code 500" in str(exc_info.value)
    finally:
        await mock_client.aclose()


@pytest.mark.asyncio
async def test_cloud_generator_timeout_handling() -> None:
    """Verify CloudTimeoutError is raised when Gemini request times out."""
    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("Connection timed out")

    mock_transport = httpx.MockTransport(timeout_handler)
    mock_client = httpx.AsyncClient(transport=mock_transport)
    gen = CloudAnswerGenerator(
        api_key="mock_test_key_12345",
        timeout_seconds=5.0,
        http_client=mock_client,
    )

    try:
        with pytest.raises(CloudTimeoutError):
            await gen.generate("Complex question")
    finally:
        await mock_client.aclose()


@pytest.mark.asyncio
async def test_cloud_generator_empty_candidates_handling() -> None:
    """Verify EmptyResponseError is raised when Gemini returns empty candidates."""
    mock_transport = httpx.MockTransport(
        lambda request: httpx.Response(
            status_code=200,
            json={"candidates": []},
        )
    )
    mock_client = httpx.AsyncClient(transport=mock_transport)
    gen = CloudAnswerGenerator(
        api_key="mock_test_key_12345",
        http_client=mock_client,
    )

    try:
        with pytest.raises(EmptyResponseError):
            await gen.generate("Hello?")
    finally:
        await mock_client.aclose()


@pytest.mark.asyncio
async def test_mock_cloud_answer_generator() -> None:
    """Verify MockCloudAnswerGenerator is safe for automated regression tests."""
    mock_cloud = MockCloudAnswerGenerator(canned_response="Safe deterministic cloud response")
    resp = await mock_cloud.generate("Why is the sky blue?")
    assert resp == "Safe deterministic cloud response"
    assert mock_cloud.call_count == 1
    assert mock_cloud.last_query == "Why is the sky blue?"
