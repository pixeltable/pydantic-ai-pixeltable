"""Contract tests for PixeltableMemoryStore."""

from __future__ import annotations

import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

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


async def test_concurrent_same_operation_id_replays(store: PixeltableMemoryStore) -> None:
    other = PixeltableMemoryStore(table_name=store._table_name)
    operation = MemoryOperation(id="run-1:call-x", fingerprint="delete:missing.md")

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda s: s._delete_sync("missing.md", None, operation), [store, other]))
    assert {r.replayed for r in results} == {False, True}
    assert all(r.existed is False for r in results)


async def test_rejects_reserved_paths_on_read_and_delete(store: PixeltableMemoryStore) -> None:
    for path in ("__op__/run-1", "__meta__/generation"):
        with pytest.raises(ValueError):
            await store.read(path, max_chars=10)
        with pytest.raises(ValueError):
            await store.delete(path, expected_version=None)


async def test_argument_validation(store: PixeltableMemoryStore) -> None:
    with pytest.raises(ValueError):
        await store.read("a.md", max_chars=0)
    with pytest.raises(ValueError):
        await store.list_paths(limit=0)
    with pytest.raises(ValueError):
        await store.list_paths("../x", limit=10)
    with pytest.raises(ValueError):
        await store.search("bad prefix/", "q", limit=1, max_files=1, max_chars=1, max_file_chars=1)


def test_incompatible_existing_table_rejected() -> None:
    name = f"test_pydantic_ai.legacy_{uuid.uuid4().hex[:8]}"
    pxt.create_dir("test_pydantic_ai", if_exists="ignore")
    memory_columns = {
        "path": pxt.String,
        "kind": pxt.String,
        "content": pxt.String | None,
        "version": pxt.String | None,
        "last_operation_id": pxt.String | None,
        "fingerprint": pxt.String | None,
        "existed": pxt.Bool | None,
    }
    pk_columns = {**memory_columns, "path": pxt.String}
    try:
        # Pre-release revisions used an Int version column.
        pxt.create_table(
            name,
            {"path": pxt.String, "version": pxt.Int | None},
            primary_key="path",
        )
        with pytest.raises(ValueError, match="not a memory table"):
            _ = PixeltableMemoryStore(table_name=name).table
        pxt.drop_table(name, force=True)

        # Correct column types but no primary key: compare-and-set would append duplicates.
        pxt.create_table(name, memory_columns)
        with pytest.raises(ValueError, match="not a memory table"):
            _ = PixeltableMemoryStore(table_name=name).table
        pxt.drop_table(name, force=True)

        # Missing bookkeeping columns.
        pxt.create_table(
            name,
            {"path": pxt.String, "kind": pxt.String},
            primary_key="path",
        )
        with pytest.raises(ValueError, match="column 'content' is missing"):
            _ = PixeltableMemoryStore(table_name=name).table
        pxt.drop_table(name, force=True)

        # `__op__` receipts write None to `content`, so a non-nullable column rejects them.
        pxt.create_table(
            name,
            {**pk_columns, "content": pxt.String},
            primary_key="path",
        )
        with pytest.raises(ValueError, match="not nullable"):
            _ = PixeltableMemoryStore(table_name=name).table
        pxt.drop_table(name, force=True)

        # Inserts never set an extra column, so a non-nullable one rejects them.
        pxt.create_table(
            name,
            {**pk_columns, "extra": pxt.String},
            primary_key="path",
        )
        with pytest.raises(ValueError, match="inserts never set"):
            _ = PixeltableMemoryStore(table_name=name).table
        pxt.drop_table(name, force=True)

        # Inserts write `content`, so a computed column there rejects them.
        t = pxt.create_table(
            name,
            {k: v for k, v in pk_columns.items() if k != "content"},
            primary_key="path",
        )
        t.add_computed_column(content=t.version)
        with pytest.raises(ValueError, match="computed"):
            _ = PixeltableMemoryStore(table_name=name).table
        pxt.drop_table(name, force=True)

        # A view reports the same columns and primary key but rejects writes.
        pxt.create_table(name, pk_columns, primary_key="path")
        view = f"{name}_view"
        try:
            pxt.create_view(view, pxt.get_table(name))
            with pytest.raises(ValueError, match="writable table"):
                _ = PixeltableMemoryStore(table_name=view).table
        finally:
            pxt.drop_table(view, force=True, if_not_exists="ignore")
    finally:
        pxt.drop_table(name, force=True, if_not_exists="ignore")


def test_existing_memory_table_is_reused() -> None:
    name = f"test_pydantic_ai.reuse_{uuid.uuid4().hex[:8]}"
    pxt.create_dir("test_pydantic_ai", if_exists="ignore")
    try:
        first = PixeltableMemoryStore(table_name=name)
        assert first.table is not None
        # A second instance must accept the schema the first one created.
        second = PixeltableMemoryStore(table_name=name)
        assert second.table is not None
    finally:
        pxt.drop_table(name, force=True)

    # A hand-built table with the same schema is also accepted, extra nullable columns included.
    name = f"test_pydantic_ai.manual_{uuid.uuid4().hex[:8]}"
    try:
        pxt.create_table(
            name,
            {
                "path": pxt.String,
                "kind": pxt.String,
                "content": pxt.String | None,
                "version": pxt.String | None,
                "last_operation_id": pxt.String | None,
                "fingerprint": pxt.String | None,
                "existed": pxt.Bool | None,
                "note": pxt.String | None,
            },
            primary_key="path",
        )
        # Tables created without the path index (e.g. by 0.1.0) get it on first open.
        assert "path_lookup_idx" in PixeltableMemoryStore(table_name=name).table.get_metadata()["indexes"]
    finally:
        pxt.drop_table(name, force=True)


def test_create_path_still_validates(monkeypatch: pytest.MonkeyPatch) -> None:
    """if_exists='ignore' can return a concurrently created table; it must be checked too."""
    name = f"test_pydantic_ai.race_{uuid.uuid4().hex[:8]}"
    pxt.create_dir("test_pydantic_ai", if_exists="ignore")
    pxt.create_table(name, {"a": pxt.Int})
    real_get_table = pxt.get_table

    def miss(path: Any, *args: Any, **kwargs: Any) -> Any:
        if path == name:
            # The table did not exist yet at this simulated point in the race.
            real_get_table(f"{name}_missing")
        return real_get_table(path, *args, **kwargs)

    monkeypatch.setattr(pxt, "get_table", miss)
    try:
        with pytest.raises(ValueError, match="not a memory table"):
            _ = PixeltableMemoryStore(table_name=name).table
    finally:
        pxt.drop_table(name, force=True)


def _prepared_receipt(t: pxt.Table, operation: MemoryOperation, intent: dict[str, Any], **cols: Any) -> None:
    """Insert a journaled receipt in the prepared state, as a crashed writer would leave it."""
    _insert_rows(
        t,
        [
            {
                "path": f"__op__/{operation.id}",
                "kind": "op",
                "content": json.dumps(intent),
                "version": cols.get("version"),
                "last_operation_id": None,
                "fingerprint": operation.fingerprint,
                "existed": cols.get("existed", False),
            }
        ],
    )


async def test_prepared_receipt_rolls_forward(store: PixeltableMemoryStore) -> None:
    # Crash shape: the journaled intent exists but the file mutation never landed.
    operation = MemoryOperation(id="run-1:call-9", fingerprint="write:notes/a.md:hello")
    _prepared_receipt(
        store.table,
        operation,
        {"file": "notes/a.md", "op": "write", "expected": None, "new": "hello"},
        version="result-v1",
    )
    replay = await store.get_operation(operation)
    assert replay is not None
    assert replay.replayed
    assert replay.version == "result-v1"
    file = await store.read("notes/a.md", max_chars=100)
    assert file is not None
    assert file.content == "hello"
    assert file.version == "result-v1"
    # The receipt is now completed; a second lookup is a plain replay.
    assert await store.get_operation(operation) == replay


async def test_prepared_receipt_after_apply_does_not_double_write(store: PixeltableMemoryStore) -> None:
    # Crash shape: the file mutation landed but the receipt never completed.
    operation = MemoryOperation(id="run-1:call-10", fingerprint="write:b.md:one")
    first = await store.write("b.md", "one", expected_version=None, operation=operation)
    t = store.table
    t.update(
        {"content": json.dumps({"file": "b.md", "op": "write", "expected": None, "new": "one"})},
        where=t.path == f"__op__/{operation.id}",
    )
    replay = await store.write("b.md", "one", expected_version=None, operation=operation)
    assert replay.replayed
    assert replay.version == first.version
    file = await store.read("b.md", max_chars=100)
    assert file is not None
    assert file.content == "one"


async def test_prepared_receipt_conflicts_when_path_moved(store: PixeltableMemoryStore) -> None:
    # The journaled expectation is stale: the path changed after the receipt was written.
    await store.write("c.md", "v1", expected_version=None)
    operation = MemoryOperation(id="run-1:call-11", fingerprint="write:c.md:x")
    _prepared_receipt(
        store.table,
        operation,
        {"file": "c.md", "op": "write", "expected": None, "new": "x"},
        version="result-x",
    )
    with pytest.raises(MemoryConflictError):
        await store.write("c.md", "x", expected_version=None, operation=operation)


async def test_prepared_delete_rolls_forward(store: PixeltableMemoryStore) -> None:
    created = await store.write("d.md", "x", expected_version=None)
    operation = MemoryOperation(id="run-1:call-12", fingerprint="delete:d.md")
    _prepared_receipt(
        store.table,
        operation,
        {"file": "d.md", "op": "delete", "expected": created.version, "new": None},
        version=None,
        existed=True,
    )
    result = await store.delete("d.md", expected_version=created.version, operation=operation)
    assert result.replayed
    assert result.existed
    assert await store.read("d.md", max_chars=10) is None


def test_column_base_and_nullability() -> None:
    from pydantic_ai_pixeltable._types import column_base, is_nullable_type

    assert column_base("String") == "String"
    assert column_base("String | None") == "String"
    assert column_base("Array[(8,), float32] | None") == "Array"
    assert not is_nullable_type("String")
    assert is_nullable_type("Array[(8,), float32] | None")


def test_vendored_helpers_match_harness() -> None:
    from pydantic_ai_harness.memory._store import lexical_search, validate_store_path

    from pydantic_ai_pixeltable.store import _lexical_search, _validate_store_path

    for bad in ("../x", "a//b", "a b"):
        for fn in (validate_store_path, _validate_store_path):
            with pytest.raises(ValueError):
                fn(bad)
    for good in ("a/b.md", "x"):
        validate_store_path(good)
        _validate_store_path(good)

    files = [("a.md", "alpha beta"), ("b.md", "alpha")]
    ours = _lexical_search(files, "alpha", limit=10, max_files=10, max_chars=1_000)
    theirs = lexical_search(files, "alpha", limit=10, max_files=10, max_chars=1_000)
    assert ours == theirs


async def test_rejects_paths_over_prefix_index_limit(store: PixeltableMemoryStore) -> None:
    # The primary-key index covers left(path, 256): longer paths collide once their
    # first 256 characters match, so the store refuses them outright.
    # Segments stay within the 200-char limit; only the total path length trips the cap.
    long_path = f"{'a' * 200}/{'b' * 200}"
    with pytest.raises(ValueError, match="255"):
        await store.write(long_path, "x", expected_version=None)
    with pytest.raises(ValueError, match="255"):
        await store.read(long_path, max_chars=10)
    with pytest.raises(ValueError, match="255"):
        await store.delete(long_path, expected_version=None)


async def test_conflicted_mutation_adopts_peer_receipt(
    store: PixeltableMemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Race: we journal the intent, a peer sharing the operation id applies and completes
    # it, then our own mutation hits the CAS conflict. We must adopt the peer's receipt
    # rather than retry into a double-apply.
    operation = MemoryOperation(id="run-1:call-20", fingerprint="write:p.md:c2")
    original = PixeltableMemoryStore._prepare_operation

    def prepare(self: PixeltableMemoryStore, t: pxt.Table, op: MemoryOperation, *args: Any) -> Any:
        receipt = original(self, t, op, *args)
        if receipt is None and op.id == operation.id:
            peer = PixeltableMemoryStore(table_name=self._table_name)
            peer._write_sync("p.md", "c2", None, op)
        return receipt

    monkeypatch.setattr(PixeltableMemoryStore, "_prepare_operation", prepare)
    outcome = await store.write("p.md", "c2", expected_version=None, operation=operation)
    assert outcome.replayed
    file = await store.read("p.md", max_chars=100)
    assert file is not None
    assert file.content == "c2"


@pytest.mark.parametrize("mutation", ["write", "delete"])
async def test_conflicted_mutation_withdraws_prepared_receipt(
    store: PixeltableMemoryStore, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    # An unrelated writer (another process) moves the path after we journal. Our mutation
    # never landed, so the intent is withdrawn and the harness retry with the same
    # operation id writes cleanly instead of hitting the stale intent forever.
    base = await store.write("q.md", "base", expected_version=None)
    operation = MemoryOperation(id=f"run-1:call-21-{mutation}", fingerprint=f"{mutation}:q.md")
    original = PixeltableMemoryStore._prepare_operation

    def prepare(self: PixeltableMemoryStore, t: pxt.Table, op: MemoryOperation, *args: Any) -> Any:
        receipt = original(self, t, op, *args)
        if receipt is None and op.id == operation.id:
            peer = PixeltableMemoryStore(table_name=self._table_name)
            peer._write_sync("q.md", "peer", base.version, None)
        return receipt

    monkeypatch.setattr(PixeltableMemoryStore, "_prepare_operation", prepare)
    with pytest.raises(MemoryConflictError):
        if mutation == "write":
            await store.write("q.md", "ours", expected_version=base.version, operation=operation)
        else:
            await store.delete("q.md", expected_version=base.version, operation=operation)
    monkeypatch.undo()
    assert await store.get_operation(operation) is None

    current = await store.read("q.md", max_chars=100)
    assert current is not None and current.content == "peer"
    if mutation == "write":
        retried = await store.write("q.md", "ours", expected_version=current.version, operation=operation)
        assert not retried.replayed
        after = await store.read("q.md", max_chars=100)
        assert after is not None and after.content == "ours"
    else:
        retried = await store.delete("q.md", expected_version=current.version, operation=operation)
        assert not retried.replayed
        assert await store.read("q.md", max_chars=100) is None


@pytest.mark.parametrize("mutation", ["write", "delete"])
async def test_withdraw_never_drops_an_intent_a_peer_applied(
    store: PixeltableMemoryStore, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    # A peer replaying our operation id applies the intent but has not completed it, then an
    # unrelated writer lands on top. Our CAS fails and the file no longer shows our version,
    # yet the intent did land: the peer's claim must stop the withdraw, or a retry applies it twice.
    base = await store.write("w.md", "base\n", expected_version=None)
    operation = MemoryOperation(id=f"run-1:call-30-{mutation}", fingerprint=f"{mutation}:w.md")
    peer = PixeltableMemoryStore(table_name=store._table_name)
    other = PixeltableMemoryStore(table_name=store._table_name)
    deferred: list[tuple[Any, ...]] = []
    monkeypatch.setattr(peer, "_complete_operation", lambda *call: deferred.append(call))
    original = PixeltableMemoryStore._prepare_operation

    def prepare(self: PixeltableMemoryStore, t: pxt.Table, op: MemoryOperation, *args: Any) -> Any:
        receipt = original(self, t, op, *args)
        if receipt is None and self is store:
            assert peer._get_operation_sync(op) is not None  # peer rolls our intent forward
            current = other._read_sync("w.md", 100)
            if mutation == "write":
                other._write_sync("w.md", "other\n", None if current is None else current.version, None)
            else:
                other._write_sync("w.md", "recreated\n", None, None)
        return receipt

    monkeypatch.setattr(PixeltableMemoryStore, "_prepare_operation", prepare)
    with pytest.raises(MemoryConflictError):
        if mutation == "write":
            await store.write("w.md", "base\nfact\n", expected_version=base.version, operation=operation)
        else:
            await store.delete("w.md", expected_version=base.version, operation=operation)
    monkeypatch.setattr(PixeltableMemoryStore, "_prepare_operation", original)
    for call in deferred:
        PixeltableMemoryStore._complete_operation(peer, *call)

    replay = await store.get_operation(operation)
    assert replay is not None and replay.replayed  # the retry replays; it does not re-apply
    current = await store.read("w.md", max_chars=100)
    assert current is not None
    assert current.content == ("other\n" if mutation == "write" else "recreated\n")
