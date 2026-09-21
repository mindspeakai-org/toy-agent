"""Unit tests for all specialized domain handlers."""

from pathlib import Path
import pytest

from app.cloud.client import CloudClient
from app.handlers.cloud import CloudHandler
from app.handlers.command import CommandHandler
from app.handlers.local import LocalHandler
from app.handlers.memory import MemoryHandler
from app.memory.store import MemoryStore
from app.router.models import RouteType, RoutingDecision


# --- LocalHandler Tests ---

@pytest.mark.asyncio
async def test_local_handler_greeting() -> None:
    handler = LocalHandler()
    decision = RoutingDecision(route=RouteType.LOCAL, intent="GREETING")
    response = await handler.handle("Hello there!", decision)

    assert response.route == RouteType.LOCAL
    assert response.success is True
    assert "Hello" in response.text


@pytest.mark.asyncio
async def test_local_handler_math() -> None:
    handler = LocalHandler()
    decision = RoutingDecision(route=RouteType.LOCAL)
    response = await handler.handle("What is 2 + 2?", decision)

    assert "4" in response.text


@pytest.mark.asyncio
async def test_local_handler_farewell() -> None:
    handler = LocalHandler()
    decision = RoutingDecision(route=RouteType.LOCAL)
    response = await handler.handle("Goodbye toy friend!", decision)

    assert "Goodbye" in response.text or "See you" in response.text


@pytest.mark.asyncio
async def test_local_handler_uses_router_payload_response() -> None:
    handler = LocalHandler()
    decision = RoutingDecision(
        route=RouteType.LOCAL,
        raw_response={"response": "Custom answer directly from router SLM."},
    )
    response = await handler.handle("Any question", decision)

    assert response.text == "Custom answer directly from router SLM."


# --- MemoryHandler Tests ---

@pytest.mark.asyncio
async def test_memory_handler_query_hit(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "mem.json")
    store.set("favorite_animal", "tiger")

    handler = MemoryHandler(store)
    decision = RoutingDecision(
        route=RouteType.MEMORY,
        intent="MEMORY_QUERY",
        key="favorite_animal",
    )
    response = await handler.handle("What is my favorite animal?", decision)

    assert response.success is True
    assert "favorite animal is tiger" in response.text.lower()


@pytest.mark.asyncio
async def test_memory_handler_query_miss(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "mem.json")
    handler = MemoryHandler(store)
    decision = RoutingDecision(
        route=RouteType.MEMORY,
        intent="MEMORY_QUERY",
        key="favorite_color",
    )
    response = await handler.handle("What is my favorite color?", decision)

    assert response.success is True
    assert "don't know your favorite color yet" in response.text.lower()


@pytest.mark.asyncio
async def test_memory_handler_update(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "mem.json")
    handler = MemoryHandler(store)

    decision = RoutingDecision(
        route=RouteType.MEMORY,
        intent="MEMORY_UPDATE",
        key="favorite_toy",
        value="rocket",
    )
    response = await handler.handle("My favorite toy is a rocket", decision)

    assert response.success is True
    assert store.get("favorite_toy") == "rocket"
    assert "remember that your favorite toy is rocket" in response.text.lower()


@pytest.mark.asyncio
async def test_memory_handler_heuristic_extraction(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "mem.json")
    handler = MemoryHandler(store)

    # Empty decision key/value -> handler extracts from text
    decision = RoutingDecision(route=RouteType.MEMORY)
    update_res = await handler.handle("My favorite snack is cookies", decision)
    assert update_res.success is True
    assert store.get("favorite_snack") == "cookies"

    # Query extraction
    query_res = await handler.handle("What is my favorite snack?", decision)
    assert "favorite snack is cookies" in query_res.text.lower()


# --- CommandHandler Tests ---

@pytest.mark.asyncio
async def test_command_handler_volume() -> None:
    handler = CommandHandler()
    decision = RoutingDecision(route=RouteType.COMMAND, intent="DEVICE_ACTION")

    res_up = await handler.handle("Please turn the volume up", decision)
    assert "volume up" in res_up.text.lower()

    res_down = await handler.handle("Turn volume down", decision)
    assert "volume down" in res_down.text.lower()


@pytest.mark.asyncio
async def test_command_handler_stop() -> None:
    handler = CommandHandler()
    decision = RoutingDecision(route=RouteType.COMMAND)
    res = await handler.handle("Stop please", decision)
    assert "stopped" in res.text.lower()


@pytest.mark.asyncio
async def test_command_handler_invokes_action_executor() -> None:
    executed_events = []

    def mock_executor(query: str, raw: dict) -> None:
        executed_events.append(query)

    handler = CommandHandler(action_executor=mock_executor)
    decision = RoutingDecision(route=RouteType.COMMAND)
    await handler.handle("Turn on the light", decision)

    assert len(executed_events) == 1
    assert executed_events[0] == "Turn on the light"


# --- CloudHandler Tests ---

@pytest.mark.asyncio
async def test_cloud_handler_stub() -> None:
    cloud_client = CloudClient(stub_mode=True)
    handler = CloudHandler(cloud_client=cloud_client)
    decision = RoutingDecision(route=RouteType.CLOUD, intent="KNOWLEDGE_QUERY")

    response = await handler.handle("Why does the moon follow me?", decision)
    assert response.success is True
    assert "cloud AI" in response.text or "cloud" in response.text.lower()


@pytest.mark.asyncio
async def test_cloud_handler_handles_client_failure() -> None:
    class FailingCloudClient(CloudClient):
        async def generate(self, text: str) -> str:
            raise RuntimeError("Cloud provider unreachable")

    handler = CloudHandler(cloud_client=FailingCloudClient())
    decision = RoutingDecision(route=RouteType.CLOUD)

    response = await handler.handle("Big question", decision)
    assert response.success is False
    assert "couldn't reach my cloud friend" in response.text.lower()
