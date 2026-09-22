"""Unit tests for RoutingDecision validation and RouterClient communication under Phase 2 contract."""

import httpx
import pytest
from pydantic import ValidationError

from app.router.client import RouterClient
from app.router.exceptions import (
    RouterConnectionError,
    RouterResponseError,
    RouterTimeoutError,
)
from app.router.models import MemoryRequest, ProcessingType, RoutingDecision


# =====================================================================
# 1. RoutingDecision Model Validation Tests
# =====================================================================

def test_routing_decision_valid_local_no_memory() -> None:
    """Test valid LOCAL decision without memory requirements."""
    decision = RoutingDecision(
        processing=ProcessingType.LOCAL,
        memory_required=False,
        memory_request=None,
    )
    assert decision.processing == ProcessingType.LOCAL
    assert decision.memory_required is False
    assert decision.memory_request is None


def test_routing_decision_valid_local_with_memory() -> None:
    """Test valid LOCAL decision requiring single memory key."""
    decision = RoutingDecision(
        processing=ProcessingType.LOCAL,
        memory_required=True,
        memory_request=MemoryRequest(keys=["favorite_animal"]),
    )
    assert decision.processing == ProcessingType.LOCAL
    assert decision.memory_required is True
    assert decision.memory_request is not None
    assert decision.memory_request.keys == ["favorite_animal"]


def test_routing_decision_valid_local_with_multiple_memories() -> None:
    """Test valid LOCAL decision requiring multiple memory keys."""
    decision = RoutingDecision(
        processing=ProcessingType.LOCAL,
        memory_required=True,
        memory_request=MemoryRequest(keys=["child_name", "favorite_animal"]),
    )
    assert decision.memory_request is not None
    assert decision.memory_request.keys == ["child_name", "favorite_animal"]


def test_routing_decision_valid_cloud_no_memory() -> None:
    """Test valid CLOUD decision without memory requirements."""
    decision = RoutingDecision(
        processing=ProcessingType.CLOUD,
        memory_required=False,
        memory_request=None,
    )
    assert decision.processing == ProcessingType.CLOUD
    assert decision.memory_required is False
    assert decision.memory_request is None


def test_routing_decision_invalid_processing_type() -> None:
    """Test rejection of unknown processing types like 'UNKNOWN' or 'COMMAND'."""
    with pytest.raises((ValidationError, ValueError)):
        RoutingDecision.from_payload({
            "processing": "UNKNOWN",
            "memory_required": False,
            "memory_request": None,
        })

    with pytest.raises((ValidationError, ValueError)):
        RoutingDecision.from_payload({
            "processing": "COMMAND",
            "memory_required": False,
            "memory_request": None,
        })


def test_routing_decision_missing_processing() -> None:
    """Test rejection when processing field is missing."""
    with pytest.raises((ValidationError, ValueError)):
        RoutingDecision.from_payload({
            "memory_required": False,
            "memory_request": None,
        })


def test_routing_decision_missing_memory_required() -> None:
    """Test rejection when memory_required field is missing."""
    with pytest.raises((ValidationError, ValueError)):
        RoutingDecision.from_payload({
            "processing": "LOCAL",
            "memory_request": None,
        })


def test_routing_decision_memory_required_true_with_null_request() -> None:
    """Test rejection when memory_required=true but memory_request is null."""
    with pytest.raises((ValidationError, ValueError)):
        RoutingDecision.from_payload({
            "processing": "LOCAL",
            "memory_required": True,
            "memory_request": None,
        })


def test_routing_decision_memory_required_false_with_non_null_request() -> None:
    """Test rejection when memory_required=false but memory_request is provided."""
    with pytest.raises((ValidationError, ValueError)):
        RoutingDecision.from_payload({
            "processing": "LOCAL",
            "memory_required": False,
            "memory_request": {"keys": ["favorite_animal"]},
        })


def test_routing_decision_empty_memory_keys() -> None:
    """Test rejection when memory_request.keys is an empty list."""
    with pytest.raises((ValidationError, ValueError)):
        RoutingDecision.from_payload({
            "processing": "LOCAL",
            "memory_required": True,
            "memory_request": {"keys": []},
        })


def test_routing_decision_invalid_key_type() -> None:
    """Test rejection when keys is not a list of strings."""
    # string instead of list
    with pytest.raises((ValidationError, ValueError)):
        RoutingDecision.from_payload({
            "processing": "LOCAL",
            "memory_required": True,
            "memory_request": {"keys": "favorite_animal"},
        })

    # non-string item in list
    with pytest.raises((ValidationError, ValueError)):
        RoutingDecision.from_payload({
            "processing": "LOCAL",
            "memory_required": True,
            "memory_request": {"keys": [123]},
        })

    # strict boolean check for memory_required
    with pytest.raises((ValidationError, ValueError)):
        RoutingDecision.from_payload({
            "processing": "LOCAL",
            "memory_required": "true",
            "memory_request": None,
        })


# =====================================================================
# 2. RouterClient Network & Protocol Tests
# =====================================================================

@pytest.mark.asyncio
async def test_router_client_successful_local() -> None:
    """Test successful LOCAL response from RouterClient."""
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/route"
        return httpx.Response(
            200,
            json={
                "processing": "LOCAL",
                "memory_required": False,
                "memory_request": None,
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = RouterClient(client=http_client)
        decision = await client.route("Tell me a joke")

        assert decision.processing == ProcessingType.LOCAL
        assert decision.memory_required is False
        assert decision.memory_request is None


@pytest.mark.asyncio
async def test_router_client_successful_cloud() -> None:
    """Test successful CLOUD response from RouterClient."""
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        return httpx.Response(
            200,
            json={
                "processing": "CLOUD",
                "memory_required": False,
                "memory_request": None,
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = RouterClient(client=http_client)
        decision = await client.route("What is the weather today?")

        assert decision.processing == ProcessingType.CLOUD
        assert decision.memory_required is False
        assert decision.memory_request is None


@pytest.mark.asyncio
async def test_router_client_memory_metadata_preserved() -> None:
    """Test that single memory key request is preserved."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "processing": "LOCAL",
                "memory_required": True,
                "memory_request": {"keys": ["favorite_animal"]},
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = RouterClient(client=http_client)
        decision = await client.route("What is my favorite animal?")

        assert decision.processing == ProcessingType.LOCAL
        assert decision.memory_required is True
        assert decision.memory_request is not None
        assert decision.memory_request.keys == ["favorite_animal"]


@pytest.mark.asyncio
async def test_router_client_multiple_memory_keys_preserved() -> None:
    """Test that multiple memory keys are preserved."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "processing": "LOCAL",
                "memory_required": True,
                "memory_request": {"keys": ["child_name", "favorite_animal"]},
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = RouterClient(client=http_client)
        decision = await client.route("What is my name and favorite animal?")

        assert decision.processing == ProcessingType.LOCAL
        assert decision.memory_required is True
        assert decision.memory_request is not None
        assert decision.memory_request.keys == ["child_name", "favorite_animal"]


@pytest.mark.asyncio
async def test_router_client_empty_query() -> None:
    """Test that empty or whitespace query returns safe LOCAL decision without network call."""
    client = RouterClient()
    decision = await client.route("   ")
    assert decision.processing == ProcessingType.LOCAL
    assert decision.memory_required is False
    assert decision.memory_request is None


@pytest.mark.asyncio
async def test_router_client_malformed_json() -> None:
    """Test that non-JSON response raises RouterResponseError."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="Not valid JSON")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = RouterClient(client=http_client)
        with pytest.raises(RouterResponseError):
            await client.route("Hello?")


@pytest.mark.asyncio
async def test_router_client_invalid_schema() -> None:
    """Test that valid JSON with invalid decision schema raises RouterResponseError."""
    def handler(request: httpx.Request) -> httpx.Response:
        # Violates schema: memory_required=True with memory_request=None
        return httpx.Response(
            200,
            json={"processing": "LOCAL", "memory_required": True, "memory_request": None},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = RouterClient(client=http_client)
        with pytest.raises(RouterResponseError) as exc_info:
            await client.route("Hello?")
        assert "Invalid decision schema" in str(exc_info.value)


@pytest.mark.asyncio
async def test_router_client_http_500() -> None:
    """Test that 500 status raises RouterResponseError."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = RouterClient(client=http_client)
        with pytest.raises(RouterResponseError) as exc_info:
            await client.route("Hello?")
        assert "500" in str(exc_info.value)


@pytest.mark.asyncio
async def test_router_client_timeout() -> None:
    """Test that timeout raises RouterTimeoutError."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Connection timed out")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = RouterClient(client=http_client)
        with pytest.raises(RouterTimeoutError):
            await client.route("Hello?")


@pytest.mark.asyncio
async def test_router_client_connection_error() -> None:
    """Test that connection error raises RouterConnectionError."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = RouterClient(client=http_client)
        with pytest.raises(RouterConnectionError):
            await client.route("Hello?")


@pytest.mark.asyncio
async def test_router_client_check_health_success() -> None:
    """Test successful health check."""
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/health"
        return httpx.Response(200, json={"status": "healthy", "model": "Qwen/Qwen2.5-1.5B-Instruct"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = RouterClient(client=http_client)
        health = await client.check_health()
        assert health["status"] == "healthy"
        assert health["model"] == "Qwen/Qwen2.5-1.5B-Instruct"

        is_h = await client.is_healthy()
        assert is_h is True


@pytest.mark.asyncio
async def test_router_client_check_health_connection_error() -> None:
    """Test health check when server connection fails."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = RouterClient(client=http_client)
        with pytest.raises(RouterConnectionError):
            await client.check_health()

        is_h = await client.is_healthy()
        assert is_h is False


@pytest.mark.asyncio
async def test_router_client_check_health_500() -> None:
    """Test health check when server returns HTTP 500."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Error")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = RouterClient(client=http_client)
        with pytest.raises(RouterResponseError):
            await client.check_health()
