"""Contract tests for PixeltableMemoryStore."""

from __future__ import annotations

import uuid

import pixeltable as pxt
import pytest
from pydantic_ai_harness.memory import (
    MemoryConflictError,
    MemoryOperation,
    MemoryOperationConflictError,
    MemoryStore,
    SearchableMemoryStore,
)

from pydantic_ai_pixeltable import PixeltableMemoryStore


@pytest.fixture()
def store():
    table_name = f"test_pydantic_ai.mem_{uuid.uuid4().hex[:8]}"
    memory = PixeltableMemoryStore(table_name=table_name)
    yield memory
    try:
        pxt.drop_table(table_name, force=True)
    except Exception:
        pass
    try:
        pxt.drop_dir("test_pydantic_ai", force=True)
    except Exception:
        pass


async def test_compare_and_set_contract(store: PixeltableMemoryStore) -> None:
    created = await store.write("notes/main.md", "one", expected_version=None)
    assert created.version is not None
    assert not created.replayed
    assert not created.existed
    file = await store.read("notes/main.md", max_chars=1_000)
    assert file is not None
    assert file.content == "one"
    assert file.version == created.version
    assert file.operation_id is None

    updated = await store.write("notes/main.md", "two", expected_version=file.version)
    assert updated.version is not None
    assert updated.version != created.version
    assert updated.existed
    with pytest.raises(MemoryConflictError):
        await store.write("notes/main.md", "stale", expected_version=file.version)
    with pytest.raises(MemoryConflictError):
        await store.delete("notes/main.md", expected_version=file.version)

    deleted = await store.delete("notes/main.md", expected_version=updated.version)
    assert deleted.version is None
    assert deleted.existed
    assert await store.read("notes/main.md", max_chars=1_000) is None


async def test_versions_do_not_repeat_after_delete_and_recreate(store: PixeltableMemoryStore) -> None:
    first = await store.write("main.md", "same", expected_version=None)
    await store.delete("main.md", expected_version=first.version)
    recreated = await store.write("main.md", "same", expected_version=None)

    assert recreated.version != first.version
    with pytest.raises(MemoryConflictError):
        await store.write("main.md", "stale", expected_version=first.version)
    with pytest.raises(MemoryConflictError):
        await store.delete("main.md", expected_version=first.version)


async def test_read_and_listing_bounds(store: PixeltableMemoryStore) -> None:
    created = await store.write("a.md", "0123456789", expected_version=None)
    await store.write("b.md", "b", expected_version=None)
    await store.write("c.md", "c", expected_version=None)

    bounded = await store.read("a.md", max_chars=4)
    assert bounded is not None
    assert bounded.content == "0123"
    assert bounded.version == created.version
    assert bounded.truncated
    complete = await store.read("a.md", max_chars=20)
    assert complete is not None
    assert complete.version == created.version
    assert not complete.truncated
    assert await store.list_paths(limit=2) == ["a.md", "b.md"]


async def test_list_paths_prefix_isolation(store: PixeltableMemoryStore) -> None:
    await store.write("tenant-a/main.md", "a", expected_version=None)
    await store.write("tenant-b/main.md", "b", expected_version=None)
    await store.write("tenant-a/other.md", "c", expected_version=None)

    assert await store.list_paths("tenant-a/", limit=10) == ["tenant-a/main.md", "tenant-a/other.md"]
    assert await store.list_paths("tenant-b/", limit=10) == ["tenant-b/main.md"]


async def test_operation_receipts(store: PixeltableMemoryStore) -> None:
    operation = MemoryOperation(id="run-1:call-1", fingerprint="write:notes/main.md:one")

    first = await store.write("notes/main.md", "one", expected_version=None, operation=operation)
    replay = await store.write("notes/main.md", "one", expected_version=None, operation=operation)
    assert replay.version == first.version
    assert replay.replayed
    assert not replay.existed
    assert await store.get_operation(operation) == replay
    file = await store.read("notes/main.md", max_chars=1_000)
    assert file is not None
    assert file.operation_id == operation.id

    with pytest.raises(MemoryOperationConflictError):
        await store.get_operation(MemoryOperation(id=operation.id, fingerprint="different"))

    delete_operation = MemoryOperation(id="run-1:call-2", fingerprint="delete:missing.md")
    deleted = await store.delete("missing.md", expected_version=None, operation=delete_operation)
    assert not deleted.existed
    assert not deleted.replayed
    assert (await store.delete("missing.md", expected_version=None, operation=delete_operation)).replayed


async def test_rejects_unsafe_and_reserved_paths(store: PixeltableMemoryStore) -> None:
    for path in ("../escape.md", "/absolute.md", "a//b.md", "a/../../b.md", "a b.md"):
        with pytest.raises(ValueError):
            await store.read(path, max_chars=1_000)
    with pytest.raises(ValueError):
        await store.write("__meta__/generation", "nope", expected_version=None)
    with pytest.raises(ValueError):
        await store.write("__op__/run-1", "nope", expected_version=None)


async def test_table_escape_hatch(store: PixeltableMemoryStore) -> None:
    await store.write("note.md", "hello", expected_version=None)
    t = store.table
    rows = t.where(t.kind == "file").select(t.path, t.content).collect()
    assert [row["path"] for row in rows] == ["note.md"]
    assert rows[0]["content"] == "hello"


def test_implements_public_protocols(store: PixeltableMemoryStore) -> None:
    assert isinstance(store, MemoryStore)
    assert isinstance(store, SearchableMemoryStore)
