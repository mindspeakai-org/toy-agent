"""Unit tests for the local key-value MemoryStore."""

from pathlib import Path
import pytest

from app.memory.store import MemoryStore, normalize_key


def test_normalize_key() -> None:
    """Test key normalization behavior."""
    assert normalize_key("favorite_animal") == "favorite_animal"
    assert normalize_key("Favorite Animal") == "favorite_animal"
    assert normalize_key("   best   friend   ") == "best_friend"
    assert normalize_key("NAME") == "name"


def test_memory_set_and_get(tmp_path: Path) -> None:
    """Test writing and reading memory items."""
    file_path = tmp_path / "memory.json"
    store = MemoryStore(file_path=file_path)

    store.set("favorite_animal", "tiger")
    store.set("favorite_color", "blue")

    assert store.get("favorite_animal") == "tiger"
    assert store.get("favorite_color") == "blue"
    assert store.get("non_existent") is None


def test_memory_key_case_insensitivity(tmp_path: Path) -> None:
    """Test that key casing and spacing normalize identically."""
    file_path = tmp_path / "memory.json"
    store = MemoryStore(file_path=file_path)

    store.set("Favorite Color", "green")
    assert store.get("favorite_color") == "green"
    assert store.get("FAVORITE COLOR") == "green"


def test_memory_exists_and_delete(tmp_path: Path) -> None:
    """Test checking existence and deleting keys."""
    file_path = tmp_path / "memory.json"
    store = MemoryStore(file_path=file_path)

    store.set("pet_name", "Coco")
    assert store.exists("pet_name") is True
    assert store.exists("unknown") is False

    deleted = store.delete("pet_name")
    assert deleted is True
    assert store.exists("pet_name") is False
    assert store.get("pet_name") is None

    # Deleting non-existent key returns False
    assert store.delete("pet_name") is False


def test_memory_persistence(tmp_path: Path) -> None:
    """Test that data survives process restarts by reloading from disk."""
    file_path = tmp_path / "memory.json"
    store1 = MemoryStore(file_path=file_path)
    store1.set("name", "Alex")
    store1.set("favorite_food", "pizza")

    # Instantiate a second store reading from the same file
    store2 = MemoryStore(file_path=file_path)
    assert store2.get("name") == "Alex"
    assert store2.get("favorite_food") == "pizza"
    assert store2.all() == {"name": "Alex", "favorite_food": "pizza"}


def test_memory_clear(tmp_path: Path) -> None:
    """Test wiping all memory items."""
    file_path = tmp_path / "memory.json"
    store = MemoryStore(file_path=file_path)
    store.set("k1", "v1")
    store.set("k2", "v2")

    assert len(store.all()) == 2
    store.clear()
    assert len(store.all()) == 0
    assert store.get("k1") is None


# --- Phase 3: Device-Side Memory Retrieval Tests ---

def test_base_memory_store_interface() -> None:
    """Test that BaseMemoryStore is an abstract base class that cannot be directly instantiated."""
    import pytest
    from app.memory.base import BaseMemoryStore

    with pytest.raises(TypeError):
        BaseMemoryStore()  # type: ignore[abstract]


def test_memory_store_inherits_base_memory_store() -> None:
    """Test that MemoryStore is a subclass of BaseMemoryStore."""
    from app.memory.base import BaseMemoryStore

    store = MemoryStore()
    assert isinstance(store, BaseMemoryStore)


def test_get_many_single_multiple_and_unknown(tmp_path: Path) -> None:
    """Test get_many() with one key, multiple keys, and unknown keys."""
    file_path = tmp_path / "mem.json"
    store = MemoryStore(file_path=file_path)

    store.set("favorite_animal", "tiger")
    store.set("child_name", "Alex")

    # 1. Single key
    res_single = store.get_many(["favorite_animal"])
    assert res_single == {"favorite_animal": "tiger"}

    # 2. Multiple keys
    res_multi = store.get_many(["child_name", "favorite_animal"])
    assert res_multi == {"child_name": "Alex", "favorite_animal": "tiger"}

    # 3. Unknown key maps to None without throwing an exception
    res_unknown = store.get_many(["child_name", "favorite_animal", "unknown_key"])
    assert res_unknown == {
        "child_name": "Alex",
        "favorite_animal": "tiger",
        "unknown_key": None,
    }


def test_memory_context_full_hit(tmp_path: Path) -> None:
    """Test full memory hit where all requested keys exist."""
    file_path = tmp_path / "mem.json"
    store = MemoryStore(file_path=file_path)

    store.set("child_name", "Alex")
    store.set("favorite_animal", "tiger")

    ctx = store.resolve(["child_name", "favorite_animal"])
    assert ctx.hits == {"child_name": "Alex", "favorite_animal": "tiger"}
    assert ctx.misses == []
    assert ctx.resolved == {"child_name": "Alex", "favorite_animal": "tiger"}
    assert ctx.is_complete is True


def test_memory_context_partial_hit(tmp_path: Path) -> None:
    """Test partial memory hit where some keys are missing."""
    file_path = tmp_path / "mem.json"
    store = MemoryStore(file_path=file_path)

    store.set("child_name", "Alex")

    ctx = store.resolve(["child_name", "favorite_color"])
    assert ctx.hits == {"child_name": "Alex"}
    assert ctx.misses == ["favorite_color"]
    assert ctx.resolved == {"child_name": "Alex", "favorite_color": None}
    assert ctx.is_complete is False


def test_memory_context_full_miss(tmp_path: Path) -> None:
    """Test full memory miss where no requested keys exist."""
    file_path = tmp_path / "mem.json"
    store = MemoryStore(file_path=file_path)

    ctx = store.resolve(["favorite_color", "favorite_song"])
    assert ctx.hits == {}
    assert ctx.misses == ["favorite_color", "favorite_song"]
    assert ctx.resolved == {"favorite_color": None, "favorite_song": None}
    assert ctx.is_complete is False


def test_canonical_aliasing_name_and_child_name(tmp_path: Path) -> None:
    """Test deterministic canonical aliasing: name <-> child_name at storage level."""
    file_path = tmp_path / "mem.json"
    store = MemoryStore(file_path=file_path)

    # Store under canonical key "child_name"
    store.set("child_name", "Alex")

    # Both "child_name" and "name" resolve to "Alex"
    assert store.get("child_name") == "Alex"
    assert store.get("name") == "Alex"

    # get_many preserves the router-requested key while resolving the value
    res = store.get_many(["name", "favorite_animal"])
    assert res == {"name": "Alex", "favorite_animal": None}

    # Ensure no duplicate entries were created in storage
    all_entries = store.all()
    assert len(all_entries) == 1
    assert "child_name" in all_entries or "name" in all_entries


def test_memory_retrieval_bypasses_when_memory_required_false() -> None:
    """Test that MemoryHandler does not access storage when memory_required is False."""
    from app.handlers.memory import MemoryHandler
    from app.memory.base import BaseMemoryStore
    from app.router.models import ProcessingType, RoutingDecision

    class MockAccessTrackerStore(BaseMemoryStore):
        def __init__(self):
            self.accessed = False

        def get(self, key):
            self.accessed = True
            return None

        def set(self, key, value):
            pass

        def get_many(self, keys):
            self.accessed = True
            return {}

        def delete(self, key):
            return False

        def exists(self, key):
            return False

        def all(self):
            return {}

        def clear(self):
            pass

    mock_store = MockAccessTrackerStore()
    handler = MemoryHandler(mock_store)

    decision = RoutingDecision(
        processing=ProcessingType.LOCAL,
        memory_required=False,
        memory_request=None,
    )

    ctx = handler.retrieve(decision)
    assert ctx is None
    assert mock_store.accessed is False  # Zero storage calls


def test_memory_retrieval_consumes_memory_request_keys(tmp_path: Path) -> None:
    """Test that MemoryHandler consumes keys directly from RoutingDecision.memory_request."""
    from app.handlers.memory import MemoryHandler
    from app.router.models import MemoryRequest, ProcessingType, RoutingDecision

    store = MemoryStore(tmp_path / "mem.json")
    store.set("favorite_animal", "tiger")
    store.set("child_name", "Alex")
    handler = MemoryHandler(store)

    decision = RoutingDecision(
        processing=ProcessingType.LOCAL,
        memory_required=True,
        memory_request=MemoryRequest(keys=["child_name", "favorite_animal"]),
    )

    ctx = handler.retrieve(decision)
    assert ctx is not None
    assert ctx.hits == {"child_name": "Alex", "favorite_animal": "tiger"}
    assert ctx.misses == []
    assert ctx.is_complete is True


@pytest.mark.asyncio
async def test_orchestrator_routing_decision_plus_memory_context(tmp_path: Path) -> None:
    """Test AgentOrchestrator delivers RoutingDecision + MemoryContext in Phase 3."""
    from app.agent.orchestrator import AgentOrchestrator
    from app.router.mock import MockRouterClient
    from app.router.models import MemoryRequest, ProcessingType, RoutingDecision

    store = MemoryStore(tmp_path / "mem.json")
    store.set("favorite_animal", "tiger")

    mock_router = MockRouterClient()
    mock_router.set_mock_response(
        "What is my favorite animal?",
        RoutingDecision(
            processing=ProcessingType.LOCAL,
            memory_required=True,
            memory_request=MemoryRequest(keys=["favorite_animal"]),
        ),
    )
    mock_router.set_mock_response(
        "Tell me a joke",
        RoutingDecision(
            processing=ProcessingType.LOCAL,
            memory_required=False,
            memory_request=None,
        ),
    )

    orchestrator = AgentOrchestrator(
        router_client=mock_router,
        memory_store=store,
    )

    # Query with memory_required=True
    res_mem = await orchestrator.process("What is my favorite animal?")
    assert res_mem.success is True
    assert res_mem.decision is not None
    assert res_mem.decision.memory_required is True
    assert res_mem.memory_context is not None
    assert res_mem.memory_context.hits == {"favorite_animal": "tiger"}
    assert res_mem.memory_context.is_complete is True

    # Query with memory_required=False (bypasses memory)
    res_no_mem = await orchestrator.process("Tell me a joke")
    assert res_no_mem.success is True
    assert res_no_mem.decision is not None
    assert res_no_mem.decision.memory_required is False
    assert res_no_mem.memory_context is None

    await orchestrator.aclose()

