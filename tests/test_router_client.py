"""Unit tests for RouterClient communication and error handling."""

import httpx
import pytest

from app.router.client import RouterClient
from app.router.exceptions import (
    RouterConnectionError,
    RouterResponseError,
    RouterTimeoutError,
)
from app.router.models import RouteType


@pytest.mark.asyncio
async def test_router_client_success() -> None:
    """Test successful response parsing from router API."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/route"
        return httpx.Response(
            200,
            json={
                "route": "MEMORY",
                "intent": "MEMORY_QUERY",
                "key": "favorite_animal",
                "confidence": 0.95,
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = RouterClient(
            base_url="http://mock-router:8000",
            endpoint="/route",
            client=http_client,
        )
        decision = await client.route("What is my favorite animal?")

        assert decision.route == RouteType.MEMORY
        assert decision.intent == "MEMORY_QUERY"
        assert decision.key == "favorite_animal"
        assert decision.confidence == 0.95


@pytest.mark.asyncio
async def test_router_client_empty_text() -> None:
    """Test that empty or whitespace query short-circuits gracefully."""
    client = RouterClient()
    decision = await client.route("   ")
    assert decision.route == RouteType.UNKNOWN
    assert decision.intent == "EMPTY_QUERY"


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
async def test_router_client_malformed_json() -> None:
    """Test that non-JSON response raises RouterResponseError."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="Not JSON at all")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = RouterClient(client=http_client)
        with pytest.raises(RouterResponseError):
            await client.route("Hello?")


@pytest.mark.asyncio
async def test_router_client_json_array_invalid() -> None:
    """Test that a non-dict JSON response raises RouterResponseError."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["invalid", "array"])

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = RouterClient(client=http_client)
        with pytest.raises(RouterResponseError):
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
