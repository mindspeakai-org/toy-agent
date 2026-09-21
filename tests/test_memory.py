"""Unit tests for the local key-value MemoryStore."""

from pathlib import Path

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
