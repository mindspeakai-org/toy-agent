"""Unit and integration tests for AgentOrchestrator under the Phase 2 routing boundary."""

from pathlib import Path
import httpx
import pytest

from app.agent.orchestrator import AgentOrchestrator
from app.memory.store import MemoryStore
from app.router.client import RouterClient
from app.router.exceptions import (
    RouterConnectionError,
    RouterResponseError,
    RouterTimeoutError,
)
from app.router.mock import MockRouterClient
from app.router.models import MemoryRequest, ProcessingType, RoutingDecision


@pytest.mark.asyncio
async def test_orchestrator_preserves_routing_decision_local_no_memory(tmp_path: Path) -> None:
    """Test text reaches router and LOCAL decision without memory is preserved."""
    mock_router = MockRouterClient()
    orchestrator = AgentOrchestrator(
        router_client=mock_router,
        memory_store=MemoryStore(tmp_path / "mem.json"),
    )

    res = await orchestrator.process("Tell me a joke")
    assert res.success is True
    assert res.processing == ProcessingType.LOCAL
    assert res.decision is not None
    assert res.decision.processing == ProcessingType.LOCAL
    assert res.decision.memory_required is False
    assert res.decision.memory_request is None
    # Verify no downstream generation was attempted (decision is exposed)
    assert res.handler == "SLMRouter"


@pytest.mark.asyncio
async def test_orchestrator_preserves_routing_decision_local_with_memory(tmp_path: Path) -> None:
    """Test LOCAL decision with required memory key is preserved without retrieving memory."""
    mock_router = MockRouterClient()
    orchestrator = AgentOrchestrator(
        router_client=mock_router,
        memory_store=MemoryStore(tmp_path / "mem.json"),
    )

    res = await orchestrator.process("What is my favorite animal?")
    assert res.success is True
    assert res.processing == ProcessingType.LOCAL
    assert res.decision is not None
    assert res.decision.memory_required is True
    assert res.decision.memory_request is not None
    assert res.decision.memory_request.keys == ["favorite_animal"]


@pytest.mark.asyncio
async def test_orchestrator_preserves_routing_decision_multiple_memories(tmp_path: Path) -> None:
    """Test LOCAL decision with multiple required memory keys is preserved."""
    mock_router = MockRouterClient()
    orchestrator = AgentOrchestrator(
        router_client=mock_router,
        memory_store=MemoryStore(tmp_path / "mem.json"),
    )

    res = await orchestrator.process("What is my name and favorite animal?")
    assert res.success is True
    assert res.decision is not None
    assert res.decision.memory_required is True
    assert res.decision.memory_request is not None
    assert res.decision.memory_request.keys == ["child_name", "favorite_animal"]


@pytest.mark.asyncio
async def test_orchestrator_preserves_routing_decision_cloud(tmp_path: Path) -> None:
    """Test CLOUD decision is preserved without calling external cloud LLM."""
    mock_router = MockRouterClient()
    orchestrator = AgentOrchestrator(
        router_client=mock_router,
        memory_store=MemoryStore(tmp_path / "mem.json"),
    )

    res = await orchestrator.process("What is the weather today?")
    assert res.success is True
    assert res.processing == ProcessingType.CLOUD
    assert res.decision is not None
    assert res.decision.processing == ProcessingType.CLOUD
    assert res.decision.memory_required is False


@pytest.mark.asyncio
async def test_orchestrator_empty_input(tmp_path: Path) -> None:
    """Test empty string handling short-circuits gracefully."""
    mock_router = MockRouterClient()
    orchestrator = AgentOrchestrator(
        router_client=mock_router,
        memory_store=MemoryStore(tmp_path / "mem.json"),
    )

    res = await orchestrator.process("   ")
    assert "didn't hear anything" in res.text.lower()
    assert res.success is True


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


# --- Live Integration Test ---

@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_slm_router_integration() -> None:
    """Live integration test connecting to active slm-router on port 8008.

    Validates that real router output adheres to the Phase 2 contract:
    {
      "processing": "LOCAL" | "CLOUD",
      "memory_required": bool,
      "memory_request": {"keys": [...]} | null
    }
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
        # Test 1: Turn on the lights -> LOCAL, memory_required=False
        res_cmd = await orchestrator.process("Turn on the lights")
        assert res_cmd.success is True
        assert res_cmd.decision is not None
        assert res_cmd.decision.processing == ProcessingType.LOCAL
        assert res_cmd.decision.memory_required is False
        assert res_cmd.decision.memory_request is None

        # Test 2: What is my favorite animal? -> LOCAL, memory_required=True
        res_mem = await orchestrator.process("What is my favorite animal?")
        assert res_mem.success is True
        assert res_mem.decision is not None
        assert res_mem.decision.processing == ProcessingType.LOCAL
        assert res_mem.decision.memory_required is True
        assert res_mem.decision.memory_request is not None
        assert "favorite_animal" in res_mem.decision.memory_request.keys

    finally:
        await orchestrator.aclose()
