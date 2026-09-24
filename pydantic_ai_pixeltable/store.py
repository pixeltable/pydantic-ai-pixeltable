"""Pixeltable backend for Pydantic AI Harness ``Memory``."""

from __future__ import annotations

import json
import re
import threading
import uuid
from collections.abc import Iterable
from typing import Any, cast

import anyio.to_thread
import pixeltable as pxt
from pydantic_ai_harness.memory import (
    MemoryConflictError,
    MemoryFile,
    MemoryMutation,
    MemoryOperation,
    MemoryOperationConflictError,
    MemorySearchMatch,
    MemorySearchResult,
)

from pydantic_ai_pixeltable._types import column_base, is_nullable_type

_KIND_FILE = "file"
_KIND_OP = "op"
_OP_PREFIX = "__op__/"
# Set in a receipt row's (otherwise unused) last_operation_id by whoever rolls its intent forward.
_CLAIMED = "claimed"
# __op__ receipts are written today; __meta__ stays reserved for future bookkeeping.
_RESERVED_ROOTS = frozenset({"__meta__", "__op__"})


def _reject_reserved_path(path: str) -> None:
    if path.split("/", 1)[0] in _RESERVED_ROOTS:
        raise ValueError(f"memory path {path!r} is reserved for store bookkeeping")


# The primary-key index covers left(path, 256): two longer paths whose first 256 characters
# match collide on insert. Cap paths below the boundary.
_MAX_PATH_CHARS = 255


def _check_store_path(path: str) -> None:
    _validate_store_path(path)
    _reject_reserved_path(path)
    if len(path) > _MAX_PATH_CHARS:
        raise ValueError(f"memory path {path!r} exceeds {_MAX_PATH_CHARS} characters")


_MAX_OPERATION_ID_CHARS = _MAX_PATH_CHARS - len(_OP_PREFIX)


def _receipt_path(operation: MemoryOperation) -> str:
    # Receipts share the path key, so a longer id could collide with another id's receipt.
    if len(operation.id) > _MAX_OPERATION_ID_CHARS:
        raise ValueError(f"operation id exceeds {_MAX_OPERATION_ID_CHARS} characters")
    return f"{_OP_PREFIX}{operation.id}"


# Path validation and lexical search mirror pydantic_ai_harness.memory._store (checked against
# harness 0.32.0). Vendored so this package does not depend on a private harness module; keep
# behavior identical to the originals.

_VALID_SEGMENT_RE = re.compile(r"[A-Za-z0-9_.-]{1,200}")


def _validate_store_path(path: str) -> None:
    r"""Reject path strings that could escape a store's root directory."""
    if not all(_VALID_SEGMENT_RE.fullmatch(segment) and ".." not in segment for segment in path.split("/")):
        raise ValueError(f"invalid memory path: {path!r}")


def _validate_store_prefix(prefix: str) -> None:
    if prefix:
        _validate_store_path(prefix.removesuffix("/"))


def _snippet(content: str, query: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    lower = content.lower()
    positions = [lower.find(term) for term in query.lower().split()]
    found = [position for position in positions if position >= 0]
    center = min(found) if found else 0
    start = max(0, center - max_chars // 3)
    end = min(len(content), start + max_chars)
    start = max(0, end - max_chars)
    snippet = content[start:end]
    if start:
        snippet = f"...{snippet[3:]}" if len(snippet) >= 3 else "." * len(snippet)
    if end < len(content):
        snippet = f"{snippet[:-3]}..." if len(snippet) >= 3 else "." * len(snippet)
    return snippet


def _lexical_search(
    files: Iterable[tuple[str, str]],
    query: str,
    *,
    limit: int,
    max_files: int,
    max_chars: int,
    score_prefix: str = "",
) -> MemorySearchResult:
    """Search a sorted file stream with deterministic scan and output bounds."""
    terms = [term for term in query.lower().split() if term]
    if not terms or limit <= 0 or max_files <= 0 or max_chars <= 0:
        return MemorySearchResult(matches=[], scanned=0, truncated=False)
    scored: list[tuple[float, str, str]] = []
    scanned = 0
    truncated = False
    for path, content in files:
        if scanned >= max_files:
            truncated = True
            break
        scanned += 1
        lower_path = path.removeprefix(score_prefix).lower()
        lower_content = content.lower()
        score = float(sum(lower_content.count(term) + 2 * lower_path.count(term) for term in terms))
        if score:
            scored.append((score, path, content))
    scored.sort(key=lambda item: (-item[0], item[1]))
    matches: list[MemorySearchMatch] = []
    remaining = max_chars
    for score, path, content in scored[:limit]:
        visible_path = path.removeprefix(score_prefix)
        available = remaining - len(visible_path)
        if available <= 0:
            truncated = True
            break
        snippet = _snippet(content, query, available)
        matches.append(MemorySearchMatch(path=path, snippet=snippet, score=score))
        remaining -= len(visible_path) + len(snippet)
    if len(scored) > len(matches):
        truncated = True
    return MemorySearchResult(matches=matches, scanned=scanned, truncated=truncated)


def _insert_rows(t: pxt.Table, rows: list[dict[str, Any]]) -> None:
    try:
        t.insert(rows)
    except pxt.Error as exc:
        if "Duplicate primary key" in str(exc):
            raise MemoryConflictError(str(exc)) from exc
        raise


# Expected type base per column for a memory table; compare-and-set needs all of
# them plus the primary key on `path`.
_SCHEMA_COLUMNS = {
    "path": "String",
    "kind": "String",
    "content": "String",
    "version": "String",
    "last_operation_id": "String",
    "fingerprint": "String",
    "existed": "Bool",
}

# File rows and `__op__` receipts each write None to some of these, so they must be nullable.
_NULLABLE_COLUMNS = frozenset({"content", "version", "last_operation_id", "fingerprint", "existed"})

_PATH_INDEX = "path_lookup_idx"


def _memory_schema() -> dict[str, Any]:
    return {
        # Bare types are non-nullable; `T | None` declares a nullable column.
        "path": pxt.String,
        "kind": pxt.String,
        "content": pxt.String | None,
        "version": pxt.String | None,
        "last_operation_id": pxt.String | None,
        "fingerprint": pxt.String | None,
        "existed": pxt.Bool | None,
    }


def _op_intent(file: str, op: str, expected: str | None, new: str | None) -> str:
    """Journaled mutation payload kept on a prepared ``__op__`` receipt until it completes."""
    return json.dumps({"file": file, "op": op, "expected": expected, "new": new})


class PixeltableMemoryStore:
    """Pydantic AI Harness ``MemoryStore`` persisted in a Pixeltable table.

    Implements ``MemoryStore`` and ``SearchableMemoryStore``. File mutations use
    compare-and-set on the file row. Operation receipts are journaled in the same
    table under ``__op__/``: the intended mutation is recorded before it is
    applied, so a crash between the two is rolled forward or detected as applied
    on the next lookup instead of double-applying. Paths whose first segment is
    ``__meta__`` or ``__op__`` are reserved.

    Args:
        table_name: Pixeltable table path (e.g. ``'harness.memory'``).
    """

    def __init__(self, table_name: str = "harness.memory") -> None:
        self._table_name = table_name
        self._lock = threading.RLock()
        self._table: pxt.Table | None = None

    @property
    def table(self) -> pxt.Table:
        """Underlying Pixeltable table for computed columns and queries."""
        with self._lock:
            return self._ensure_table()

    async def read(self, path: str, *, max_chars: int) -> MemoryFile | None:
        return await anyio.to_thread.run_sync(self._locked, self._read_sync, path, max_chars)

    async def get_operation(self, operation: MemoryOperation) -> MemoryMutation | None:
        return await anyio.to_thread.run_sync(self._locked, self._get_operation_sync, operation)

    async def write(
        self,
        path: str,
        content: str,
        *,
        expected_version: str | None,
        operation: MemoryOperation | None = None,
    ) -> MemoryMutation:
        return await anyio.to_thread.run_sync(
            self._locked, self._write_sync, path, content, expected_version, operation
        )

    async def delete(
        self,
        path: str,
        *,
        expected_version: str | None,
        operation: MemoryOperation | None = None,
    ) -> MemoryMutation:
        return await anyio.to_thread.run_sync(self._locked, self._delete_sync, path, expected_version, operation)

    async def list_paths(self, prefix: str = "", *, limit: int) -> list[str]:
        return await anyio.to_thread.run_sync(self._locked, self._list_paths_sync, prefix, limit)

    async def search(
        self,
        prefix: str,
        query: str,
        *,
        limit: int,
        max_files: int,
        max_chars: int,
        max_file_chars: int,
    ) -> MemorySearchResult:
        return await anyio.to_thread.run_sync(
            self._locked, self._search_sync, prefix, query, limit, max_files, max_chars, max_file_chars
        )

    def _locked(self, fn: Any, *args: Any) -> Any:
        with self._lock:
            try:
                return fn(*args)
            except pxt.NotFoundError:
                # The cached handle goes stale if the table was dropped; recreate and retry once.
                self._table = None
                return fn(*args)

    def _ensure_dirs(self) -> None:
        if "." not in self._table_name:
            return
        parts = self._table_name.rsplit(".", 1)[0].split(".")
        acc: list[str] = []
        for part in parts:
            acc.append(part)
            pxt.create_dir(".".join(acc), if_exists="ignore")

    def _ensure_table(self) -> pxt.Table:
        if self._table is not None:
            return self._table
        try:
            t = pxt.get_table(self._table_name)
        except pxt.NotFoundError:
            self._ensure_dirs()
            # if_exists='ignore' can also return a table a concurrent writer just created; it is checked below.
            t = pxt.create_table(self._table_name, _memory_schema(), primary_key="path", if_exists="ignore")
        assert t is not None  # get_table's default if_not_exists='error' raises instead of returning None
        self._check_compatible_schema(t)
        # The primary key indexes left(path, 256), which cannot serve exact-path lookups. Tables
        # created before this index existed get it on first open.
        t.add_btree_index("path", idx_name=_PATH_INDEX, if_exists="ignore")
        self._table = t
        return t

    def _check_compatible_schema(self, t: pxt.Table) -> None:
        """Reject a pre-existing table whose schema cannot back the MemoryStore protocol."""
        metadata = cast(dict[str, Any], t.get_metadata())
        kind = metadata.get("kind")
        if kind != "table":
            raise ValueError(
                f"{self._table_name!r} is a {kind or 'non-table object'}; the memory store needs a writable table"
            )
        columns = metadata.get("columns") or {}
        problems: list[str] = []
        for name, expected in _SCHEMA_COLUMNS.items():
            info = columns.get(name)
            if info is None:
                problems.append(f"column {name!r} is missing")
                continue
            if info.get("is_computed"):
                problems.append(f"column {name!r} is computed; the store writes it directly")
                continue
            type_ = str(info.get("type_"))
            if column_base(type_) != expected:
                problems.append(f"column {name!r} has type {type_!r}, expected {expected!r}")
            elif name in _NULLABLE_COLUMNS and not is_nullable_type(type_):
                problems.append(f"column {name!r} is not nullable; the store writes None to it")
        for name, info in columns.items():
            if name not in _SCHEMA_COLUMNS and not info.get("is_computed") and not is_nullable_type(info.get("type_")):
                problems.append(f"column {name!r} is not nullable, and inserts never set it")
        primary_key = metadata.get("primary_key")
        if primary_key is not None:
            if list(primary_key) != ["path"]:
                problems.append(f"primary key is {list(primary_key)!r}, expected ['path']")
        elif (columns.get("path") or {}).get("is_primary_key") is not True:
            problems.append("column 'path' has no primary key constraint")
        if problems:
            raise ValueError(
                f"{self._table_name!r} is not a memory table ({'; '.join(problems)}); drop and recreate it"
            )

    def _file_row(self, t: pxt.Table, path: str) -> dict[str, Any] | None:
        rows = (
            t.where((t.path == path) & (t.kind == _KIND_FILE))
            .select(t.content, t.version, t.last_operation_id)
            .collect()
        )
        if len(rows) == 0:
            return None
        return rows[0]

    def _lookup_operation(
        self, t: pxt.Table, operation: MemoryOperation, *, withdraw_unapplied: bool = False
    ) -> MemoryMutation | None:
        rows = (
            t.where(t.path == _receipt_path(operation)).select(t.fingerprint, t.version, t.existed, t.content).collect()
        )
        if len(rows) == 0:
            return None
        row = rows[0]
        if row["fingerprint"] != operation.fingerprint:
            raise MemoryOperationConflictError(f"operation id {operation.id!r} was reused with different arguments")
        mutation = MemoryMutation(
            version=None if row["version"] is None else str(row["version"]),
            replayed=True,
            existed=bool(row["existed"]),
        )
        if row["content"] is not None:
            # Prepared but not completed: the mutation may or may not have landed; settle it first.
            if not self._recover_operation(t, operation, row["content"], mutation, withdraw_unapplied):
                # The intent was withdrawn or replaced before we could claim it; look again.
                return self._lookup_operation(t, operation)
        return mutation

    def _recover_operation(
        self, t: pxt.Table, operation: MemoryOperation, intent: str, receipt: MemoryMutation, withdraw: bool
    ) -> bool:
        """Roll a prepared operation forward or confirm it applied, then mark the receipt complete.

        ``withdraw`` is for the writer whose own mutation just lost the compare-and-set: an
        intent nobody claimed cannot have landed, so it is deleted and a retry starts clean, as
        FileStore (which checks and journals in one transaction) never keeps one. That holds
        even when the file is gone, since another writer's delete leaves the same state. A peer
        claims the receipt before rolling it forward, and a claimed or crash-recovered intent that
        cannot be settled stays, blocking recovery like FileStore. After a crash, a missing file
        settles a delete intent, as in FileStore's recovery. Returns False when the intent was
        gone before it could be claimed.
        """
        receipt_row = (t.path == _receipt_path(operation)) & (t.kind == _KIND_OP) & (t.content == intent)
        recorded = json.loads(intent)
        conflict = MemoryConflictError(f"memory path {recorded['file']!r} changed during operation {operation.id!r}")
        unclaimed = receipt_row & (t.last_operation_id == None)  # noqa: E711 (SQL IS NULL)
        if withdraw and t.delete(where=unclaimed).row_count_stats.del_rows == 1:
            raise conflict
        row = self._file_row(t, recorded["file"])
        current = None if row is None else str(row["version"])
        applied = current is None if recorded["op"] == "delete" else current == receipt.version
        if not applied and current == recorded["expected"]:
            # Claim first: a writer withdrawing this intent deletes it only while unclaimed.
            if t.update({"last_operation_id": _CLAIMED}, where=receipt_row).row_count_stats.upd_rows != 1:
                return False
            try:
                self._apply_intent(t, operation, recorded, receipt)
            except MemoryConflictError:
                # A peer may have applied it concurrently; re-check before declaring a conflict.
                row = self._file_row(t, recorded["file"])
                current = None if row is None else str(row["version"])
                applied = current is None if recorded["op"] == "delete" else current == receipt.version
            else:
                applied = True
        if not applied:
            raise conflict
        self._complete_operation(t, operation, intent)
        return True

    def _apply_intent(
        self, t: pxt.Table, operation: MemoryOperation, recorded: dict[str, Any], receipt: MemoryMutation
    ) -> None:
        """Apply the journaled mutation exactly as the original attempt would have."""
        path = recorded["file"]
        if recorded["op"] == "delete":
            status = t.delete(where=(t.path == path) & (t.kind == _KIND_FILE) & (t.version == recorded["expected"]))
            if status.row_count_stats.del_rows != 1:
                raise MemoryConflictError(f"memory path {path!r} changed before it could be deleted")
        elif recorded["expected"] is None:
            _insert_rows(
                t,
                [
                    {
                        "path": path,
                        "kind": _KIND_FILE,
                        "content": recorded["new"],
                        "version": receipt.version,
                        "last_operation_id": operation.id,
                        "fingerprint": None,
                        "existed": None,
                    }
                ],
            )
        else:
            status = t.update(
                {"content": recorded["new"], "version": receipt.version, "last_operation_id": operation.id},
                where=(t.path == path) & (t.kind == _KIND_FILE) & (t.version == recorded["expected"]),
            )
            if status.row_count_stats.upd_rows != 1:
                raise MemoryConflictError(f"memory path {path!r} changed before it could be written")

    def _prepare_operation(
        self,
        t: pxt.Table,
        operation: MemoryOperation,
        intent: str,
        version: str | None,
        existed: bool,
    ) -> MemoryMutation | None:
        """Journal the intended mutation under ``__op__/<id>`` before applying it."""
        try:
            _insert_rows(
                t,
                [
                    {
                        "path": _receipt_path(operation),
                        "kind": _KIND_OP,
                        "content": intent,
                        "version": version,
                        "last_operation_id": None,
                        "fingerprint": operation.fingerprint,
                        "existed": existed,
                    }
                ],
            )
        except MemoryConflictError:
            # Another writer already journaled this operation id; recover or replay its receipt.
            receipt = self._lookup_operation(t, operation)
            if receipt is None:
                raise MemoryConflictError(f"operation {operation.id!r} receipt vanished mid-flight") from None
            return receipt
        return None

    def _complete_operation(self, t: pxt.Table, operation: MemoryOperation, intent: str) -> None:
        """Drop the journaled intent; the mutation is durable, so the payload is no longer needed.

        Matching on ``intent`` keeps a late peer from completing a newer intent under the same id.
        """
        t.update(
            {"content": None},
            where=(t.path == _receipt_path(operation)) & (t.kind == _KIND_OP) & (t.content == intent),
        )

    def _read_sync(self, path: str, max_chars: int) -> MemoryFile | None:
        _check_store_path(path)
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        row = self._file_row(self._ensure_table(), path)
        if row is None:
            return None
        content = row["content"] or ""
        return MemoryFile(
            content=content[:max_chars],
            version=str(row["version"]),
            operation_id=row["last_operation_id"],
            truncated=len(content) > max_chars,
        )

    def _get_operation_sync(self, operation: MemoryOperation) -> MemoryMutation | None:
        return self._lookup_operation(self._ensure_table(), operation)

    def _write_sync(
        self,
        path: str,
        content: str,
        expected_version: str | None,
        operation: MemoryOperation | None,
    ) -> MemoryMutation:
        _check_store_path(path)
        t = self._ensure_table()
        if operation is not None:
            receipt = self._lookup_operation(t, operation)
            if receipt is not None:
                return receipt
        row = self._file_row(t, path)
        current = None if row is None else str(row["version"])
        if current != expected_version:
            raise MemoryConflictError(f"memory path {path!r} changed before it could be written")
        version = uuid.uuid4().hex
        existed = row is not None
        intent = _op_intent(path, "write", expected_version, content)
        if operation is not None:
            receipt = self._prepare_operation(t, operation, intent, version, existed)
            if receipt is not None:
                return receipt
        try:
            if row is None:
                _insert_rows(
                    t,
                    [
                        {
                            "path": path,
                            "kind": _KIND_FILE,
                            "content": content,
                            "version": version,
                            "last_operation_id": operation.id if operation else None,
                            "fingerprint": None,
                            "existed": None,
                        }
                    ],
                )
            else:
                status = t.update(
                    {"content": content, "version": version, "last_operation_id": operation.id if operation else None},
                    where=(t.path == path) & (t.kind == _KIND_FILE) & (t.version == current),
                )
                if status.row_count_stats.upd_rows != 1:
                    raise MemoryConflictError(f"memory path {path!r} changed before it could be written")
        except MemoryConflictError:
            if operation is not None:
                # A peer sharing the operation id may have applied our intent; adopt it rather
                # than let a retry apply it twice. Otherwise the intent is withdrawn.
                receipt = self._lookup_operation(t, operation, withdraw_unapplied=True)
                if receipt is not None:
                    return receipt
            raise
        if operation is not None:
            self._complete_operation(t, operation, intent)
        return MemoryMutation(version=version, replayed=False, existed=existed)

    def _delete_sync(
        self,
        path: str,
        expected_version: str | None,
        operation: MemoryOperation | None,
    ) -> MemoryMutation:
        _check_store_path(path)
        t = self._ensure_table()
        if operation is not None:
            receipt = self._lookup_operation(t, operation)
            if receipt is not None:
                return receipt
        row = self._file_row(t, path)
        current = None if row is None else str(row["version"])
        if current != expected_version:
            raise MemoryConflictError(f"memory path {path!r} changed before it could be deleted")
        existed = row is not None
        intent = _op_intent(path, "delete", expected_version, None)
        if operation is not None:
            receipt = self._prepare_operation(t, operation, intent, None, existed)
            if receipt is not None:
                return receipt
        try:
            if row is not None:
                status = t.delete(where=(t.path == path) & (t.kind == _KIND_FILE) & (t.version == current))
                if status.row_count_stats.del_rows != 1:
                    raise MemoryConflictError(f"memory path {path!r} changed before it could be deleted")
        except MemoryConflictError:
            if operation is not None:
                # Same as in _write_sync: adopt a peer's receipt or withdraw ours.
                receipt = self._lookup_operation(t, operation, withdraw_unapplied=True)
                if receipt is not None:
                    return receipt
            raise
        if operation is not None:
            self._complete_operation(t, operation, intent)
        return MemoryMutation(version=None, replayed=False, existed=existed)

    def _file_rows(self, t: pxt.Table, prefix: str, content_chars: int = 0) -> list[dict[str, Any]]:
        """Live file rows under ``prefix``, sorted by path, with ``content`` cut to ``content_chars`` when set.

        The prefix match is a substring equality, which is collation-independent; a
        ``[prefix, prefix+'\\uffff')`` range scan or an ORDER BY is not, so sorting happens here.
        """
        pred = t.kind == _KIND_FILE
        if prefix:
            pred = pred & (t.path.slice(0, len(prefix)) == prefix)
        columns = {"content": t.content.slice(0, content_chars)} if content_chars else {}
        rows = list(t.where(pred).select(t.path, **columns).collect())
        rows.sort(key=lambda row: str(row["path"]))
        return rows

    def _list_paths_sync(self, prefix: str, limit: int) -> list[str]:
        _validate_store_prefix(prefix)
        if limit <= 0:
            raise ValueError("limit must be positive")
        t = self._ensure_table()
        return [str(row["path"]) for row in self._file_rows(t, prefix)[:limit]]

    def _search_sync(
        self,
        prefix: str,
        query: str,
        limit: int,
        max_files: int,
        max_chars: int,
        max_file_chars: int,
    ) -> MemorySearchResult:
        _validate_store_prefix(prefix)
        if not query.split() or limit <= 0 or max_files <= 0 or max_chars <= 0 or max_file_chars <= 0:
            return MemorySearchResult(matches=[], scanned=0, truncated=False)
        t = self._ensure_table()
        # One extra character tells a file cut at max_file_chars from one that fits exactly.
        fetched = self._file_rows(t, prefix, content_chars=max_file_chars + 1)
        # One file past max_files lets _lexical_search report the scan bound as truncated.
        files = [(str(row["path"]), row["content"] or "") for row in fetched[: max_files + 1]]
        result = _lexical_search(
            [(path, content[:max_file_chars]) for path, content in files],
            query,
            limit=limit,
            max_files=max_files,
            max_chars=max_chars,
            score_prefix=prefix,
        )
        content_truncated = any(len(content) > max_file_chars for _, content in files[:max_files])
        return MemorySearchResult(
            matches=result.matches, scanned=result.scanned, truncated=result.truncated or content_truncated
        )
