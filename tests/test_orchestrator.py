"""Unit and integration tests for AgentOrchestrator."""

from pathlib import Path
import httpx
import pytest

from app.agent.orchestrator import AgentOrchestrator
from app.handlers.local import LocalHandler
from app.memory.store import MemoryStore
from app.router.client import RouterClient
from app.router.exceptions import (
    RouterConnectionError,
    RouterResponseError,
    RouterTimeoutError,
)
from app.router.mock import MockRouterClient
from app.router.models import RouteType, RoutingDecision


@pytest.mark.asyncio
async def test_orchestrator_memory_flow(tmp_path: Path) -> None:
    """Test full memory write then read flow through orchestrator."""
    memory_file = tmp_path / "memory.json"
    memory_store = MemoryStore(memory_file)
    mock_router = MockRouterClient()
    orchestrator = AgentOrchestrator(
        router_client=mock_router,
        memory_store=memory_store,
    )

    # 1. Update memory
    write_res = await orchestrator.process("My favorite animal is a tiger")
    assert write_res.route == RouteType.MEMORY
    assert write_res.success is True
    assert "favorite animal is tiger" in write_res.text.lower()

    # 2. Query memory
    query_res = await orchestrator.process("What is my favorite animal?")
    assert query_res.route == RouteType.MEMORY
    assert query_res.success is True
    assert "favorite animal is tiger" in query_res.text.lower()


@pytest.mark.asyncio
async def test_orchestrator_local_flow(tmp_path: Path) -> None:
    """Test local chit-chat flow."""
    mock_router = MockRouterClient()
    orchestrator = AgentOrchestrator(
        router_client=mock_router,
        memory_store=MemoryStore(tmp_path / "mem.json"),
    )

    res = await orchestrator.process("Hello!")
    assert res.route == RouteType.LOCAL
    assert res.success is True
    assert "Hello" in res.text


@pytest.mark.asyncio
async def test_orchestrator_command_flow(tmp_path: Path) -> None:
    """Test device command flow."""
    mock_router = MockRouterClient()
    orchestrator = AgentOrchestrator(
        router_client=mock_router,
        memory_store=MemoryStore(tmp_path / "mem.json"),
    )

    res = await orchestrator.process("Turn up the volume")
    assert res.route == RouteType.COMMAND
    assert res.success is True
    assert "volume up" in res.text.lower()


@pytest.mark.asyncio
async def test_orchestrator_cloud_flow(tmp_path: Path) -> None:
    """Test cloud dispatch flow."""
    mock_router = MockRouterClient()
    orchestrator = AgentOrchestrator(
        router_client=mock_router,
        memory_store=MemoryStore(tmp_path / "mem.json"),
    )

    res = await orchestrator.process("Why is the sky blue?")
    assert res.route == RouteType.CLOUD
    assert res.success is True
    assert "cloud" in res.text.lower()


@pytest.mark.asyncio
async def test_orchestrator_empty_input(tmp_path: Path) -> None:
    """Test empty string handling."""
    mock_router = MockRouterClient()
    orchestrator = AgentOrchestrator(
        router_client=mock_router,
        memory_store=MemoryStore(tmp_path / "mem.json"),
    )

    res = await orchestrator.process("   ")
    assert "didn't hear anything" in res.text.lower()


@pytest.mark.asyncio
async def test_orchestrator_unknown_route(tmp_path: Path) -> None:
    """Test handling of unsupported/unknown routes."""
    mock_router = MockRouterClient()
    mock_router.set_mock_response(
        "weird input",
        RoutingDecision(route=RouteType.UNKNOWN, intent="UNKNOWN_INTENT"),
    )
    orchestrator = AgentOrchestrator(
        router_client=mock_router,
        memory_store=MemoryStore(tmp_path / "mem.json"),
    )

    res = await orchestrator.process("weird input")
    assert res.route == RouteType.UNKNOWN
    assert res.success is False
    assert "not quite sure" in res.text.lower()


@pytest.mark.asyncio
async def test_orchestrator_resilience_router_timeout(tmp_path: Path) -> None:
    """Test graceful fallback on router timeout."""
    mock_router = MockRouterClient()
    mock_router.inject_error(RouterTimeoutError("Gateway timed out"))
    orchestrator = AgentOrchestrator(
        router_client=mock_router,
        memory_store=MemoryStore(tmp_path / "mem.json"),
    )

    res = await orchestrator.process("Any query")
    assert res.success is False
    assert res.intent == "ROUTER_TIMEOUT"
    assert "taking a little too long" in res.text.lower()


@pytest.mark.asyncio
async def test_orchestrator_resilience_router_connection_error(tmp_path: Path) -> None:
    """Test graceful fallback on router connection refusal."""
    mock_router = MockRouterClient()
    mock_router.inject_error(RouterConnectionError("Connection refused"))
    orchestrator = AgentOrchestrator(
        router_client=mock_router,
        memory_store=MemoryStore(tmp_path / "mem.json"),
    )

    res = await orchestrator.process("Any query")
    assert res.success is False
    assert res.intent == "ROUTER_UNAVAILABLE"
    assert "trouble connecting" in res.text.lower()


@pytest.mark.asyncio
async def test_orchestrator_resilience_router_response_error(tmp_path: Path) -> None:
    """Test graceful fallback on malformed response."""
    mock_router = MockRouterClient()
    mock_router.inject_error(RouterResponseError("Malformed JSON"))
    orchestrator = AgentOrchestrator(
        router_client=mock_router,
        memory_store=MemoryStore(tmp_path / "mem.json"),
    )

    res = await orchestrator.process("Any query")
    assert res.success is False
    assert res.intent == "ROUTER_MALFORMED_RESPONSE"
    assert "trouble understanding" in res.text.lower()


@pytest.mark.asyncio
async def test_orchestrator_resilience_handler_crash(tmp_path: Path) -> None:
    """Test graceful fallback when a handler raises an unexpected exception."""
    class CrashingLocalHandler(LocalHandler):
        async def handle(self, query: str, decision: RoutingDecision):
            raise ZeroDivisionError("Unexpected crash inside handler")

    mock_router = MockRouterClient()
    orchestrator = AgentOrchestrator(
        router_client=mock_router,
        memory_store=MemoryStore(tmp_path / "mem.json"),
        local_handler=CrashingLocalHandler(),
    )

    res = await orchestrator.process("Hello!")
    assert res.success is False
    assert res.intent == "HANDLER_EXECUTION_FAILURE"
    assert "went a little wobbly" in res.text.lower()


# --- Live Integration Test ---

@pytest.mark.skip(reason="Obsolete in Phase 1: Live SLM-Router updated to Phase 2 contract (processing: LOCAL/CLOUD). Will be updated in Phase 2.")
@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_slm_router_integration() -> None:
    """Integration test connecting to live slm-router if reachable.

    Skips automatically if the service is not currently running.
    """
    from app.config.settings import get_settings
    settings = get_settings()
    health_url = f"{settings.router_base_url.rstrip('/')}/health"
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(health_url)
            if resp.status_code != 200:
                pytest.skip(f"Service at {health_url} is not healthy (HTTP {resp.status_code})")
    except Exception:
        pytest.skip(f"External slm-router service is not reachable at {health_url}")

    real_router = RouterClient(
        base_url=settings.router_base_url,
        endpoint=settings.router_route_endpoint,
        timeout=15.0,
    )
    orchestrator = AgentOrchestrator(router_client=real_router)

    try:
        res = await orchestrator.process("Turn on the light")
        assert res.success is True
        assert res.route == RouteType.COMMAND
        assert "light" in res.text.lower() or "turned on" in res.text.lower()
    finally:
        await orchestrator.aclose()
