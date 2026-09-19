"""Contract tests for PixeltableMemoryStore."""

from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor

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
from pydantic_ai_pixeltable.store import _insert_rows


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


async def test_list_and_search_omit_bookkeeping(store: PixeltableMemoryStore) -> None:
    operation = MemoryOperation(id="run-1:call-1", fingerprint="write:a.md:alpha")
    await store.write("a.md", "alpha", expected_version=None, operation=operation)
    paths = await store.list_paths("", limit=50)
    assert paths == ["a.md"]
    assert all(not path.startswith("__") for path in paths)
    found = await store.search("", "alpha", limit=10, max_files=10, max_chars=400, max_file_chars=1_000)
    assert [match.path for match in found.matches] == ["a.md"]


async def test_table_content_update_leaves_version(store: PixeltableMemoryStore) -> None:
    created = await store.write("note.md", "hello", expected_version=None)
    t = store.table
    t.update({"content": "HACKED"}, where=t.path == "note.md")
    file = await store.read("note.md", max_chars=100)
    assert file is not None
    assert file.content == "HACKED"
    assert file.version == created.version
    overwritten = await store.write("note.md", "restored", expected_version=file.version)
    assert overwritten.existed
    restored = await store.read("note.md", max_chars=100)
    assert restored is not None
    assert restored.content == "restored"


async def test_duplicate_insert_is_conflict(store: PixeltableMemoryStore) -> None:
    await store.write("race.md", "A", expected_version=None)
    other = PixeltableMemoryStore(table_name=store._table_name)
    with pytest.raises(MemoryConflictError):
        await other.write("race.md", "B", expected_version=None)

    with pytest.raises(MemoryConflictError, match="Duplicate primary key"):
        _insert_rows(
            store.table,
            [
                {
                    "path": "race.md",
                    "kind": "file",
                    "content": "C",
                    "version": uuid.uuid4().hex,
                    "last_operation_id": None,
                    "fingerprint": None,
                    "existed": None,
                }
            ],
        )


async def test_concurrent_create_across_instances_is_conflict(store: PixeltableMemoryStore) -> None:
    assert store.table is not None
    other = PixeltableMemoryStore(table_name=store._table_name)

    def create(target: PixeltableMemoryStore, content: str) -> str:
        try:
            target._write_sync("race.md", content, None, None)
            return "ok"
        except MemoryConflictError:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda pair: create(*pair), [(store, "A"), (other, "B")]))
    assert results.count("ok") == 1
    assert results.count("conflict") == 1
    file = await store.read("race.md", max_chars=10)
    assert file is not None
    assert file.content in {"A", "B"}


def test_implements_public_protocols(store: PixeltableMemoryStore) -> None:
    assert isinstance(store, MemoryStore)
    assert isinstance(store, SearchableMemoryStore)
