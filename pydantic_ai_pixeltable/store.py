"""Pixeltable backend for Pydantic AI Harness ``Memory``."""

from __future__ import annotations

import re
import threading
import uuid
from collections.abc import Iterable
from typing import Any

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

_KIND_FILE = "file"
_KIND_OP = "op"
_OP_PREFIX = "__op__/"
# __op__ receipts are written today; __meta__ stays reserved for future bookkeeping.
_RESERVED_ROOTS = frozenset({"__meta__", "__op__"})


def _reject_reserved_path(path: str) -> None:
    if path.split("/", 1)[0] in _RESERVED_ROOTS:
        raise ValueError(f"memory path {path!r} is reserved for store bookkeeping")


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


def _column_base(type_: Any) -> str:
    base = str(type_).split(" | ", 1)[0]
    # 0.6.x renders non-nullable columns as 'Required[T]'; 0.7.x renders them as 'T'.
    if base.startswith("Required[") and base.endswith("]"):
        base = base[len("Required[") : -1]
    return base.split("[", 1)[0]


class PixeltableMemoryStore:
    """Pydantic AI Harness ``MemoryStore`` persisted in a Pixeltable table.

    Implements ``MemoryStore`` and ``SearchableMemoryStore``. File mutations use
    compare-and-set on the file row; operation receipts are written afterward.
    That is two-phase, not one SQL transaction, and there is no FileStore crash
    journal. A crash between the file mutation and the ``__op__/`` insert can
    yield ``MemoryConflictError`` on replay instead of ``replayed=True``. Paths
    whose first segment is ``__meta__`` or ``__op__`` are reserved.

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
            t = None
        if t is not None:
            self._check_compatible_schema(t)
            self._table = t
            return t
        self._ensure_dirs()
        schema: dict[str, Any] = {
            # Required: older pixeltable makes columns nullable by default and rejects a nullable primary key.
            "path": pxt.Required[pxt.String],
            "kind": pxt.String,
            "content": pxt.String | None,
            "version": pxt.String | None,
            "last_operation_id": pxt.String | None,
            "fingerprint": pxt.String | None,
            "existed": pxt.Bool | None,
        }
        self._table = pxt.create_table(self._table_name, schema, primary_key="path", if_exists="ignore")
        return self._table

    def _check_compatible_schema(self, t: pxt.Table) -> None:
        """Reject a pre-existing table whose schema cannot back the MemoryStore protocol."""
        metadata = t.get_metadata()
        columns = metadata.get("columns") or {}
        problems: list[str] = []
        for name, expected in _SCHEMA_COLUMNS.items():
            info = columns.get(name)
            if info is None:
                problems.append(f"column {name!r} is missing")
            elif _column_base(info.get("type_")) != expected:
                problems.append(f"column {name!r} has type {info.get('type_')!r}, expected {expected!r}")
        primary_key = metadata.get("primary_key")
        if primary_key is not None:
            if list(primary_key) != ["path"]:
                problems.append(f"primary key is {list(primary_key)!r}, expected ['path']")
        elif (columns.get("path") or {}).get("is_primary_key") is False:
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

    def _lookup_operation(self, t: pxt.Table, operation: MemoryOperation) -> MemoryMutation | None:
        rows = t.where(t.path == f"{_OP_PREFIX}{operation.id}").select(t.fingerprint, t.version, t.existed).collect()
        if len(rows) == 0:
            return None
        row = rows[0]
        if row["fingerprint"] != operation.fingerprint:
            raise MemoryOperationConflictError(f"operation id {operation.id!r} was reused with different arguments")
        version = None if row["version"] is None else str(row["version"])
        return MemoryMutation(version=version, replayed=True, existed=bool(row["existed"]))

    def _record_operation(
        self, t: pxt.Table, operation: MemoryOperation | None, mutation: MemoryMutation
    ) -> MemoryMutation | None:
        if operation is None:
            return None
        try:
            _insert_rows(
                t,
                [
                    {
                        "path": f"{_OP_PREFIX}{operation.id}",
                        "kind": _KIND_OP,
                        "content": None,
                        "version": mutation.version,
                        "last_operation_id": None,
                        "fingerprint": operation.fingerprint,
                        "existed": mutation.existed,
                    }
                ],
            )
        except MemoryConflictError:
            # Another writer already recorded this operation id; return its receipt (or raise
            # MemoryOperationConflictError on a fingerprint mismatch) instead of a bare conflict.
            return self._lookup_operation(t, operation)
        return None

    def _read_sync(self, path: str, max_chars: int) -> MemoryFile | None:
        _validate_store_path(path)
        _reject_reserved_path(path)
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
        _validate_store_path(path)
        _reject_reserved_path(path)
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
        op_id = operation.id if operation else None
        if row is None:
            _insert_rows(
                t,
                [
                    {
                        "path": path,
                        "kind": _KIND_FILE,
                        "content": content,
                        "version": version,
                        "last_operation_id": op_id,
                        "fingerprint": None,
                        "existed": None,
                    }
                ],
            )
        else:
            status = t.update(
                {"content": content, "version": version, "last_operation_id": op_id},
                where=(t.path == path) & (t.kind == _KIND_FILE) & (t.version == current),
            )
            if status.row_count_stats.upd_rows != 1:
                raise MemoryConflictError(f"memory path {path!r} changed before it could be written")
        mutation = MemoryMutation(version=version, replayed=False, existed=row is not None)
        return self._record_operation(t, operation, mutation) or mutation

    def _delete_sync(
        self,
        path: str,
        expected_version: str | None,
        operation: MemoryOperation | None,
    ) -> MemoryMutation:
        _validate_store_path(path)
        _reject_reserved_path(path)
        t = self._ensure_table()
        if operation is not None:
            receipt = self._lookup_operation(t, operation)
            if receipt is not None:
                return receipt
        row = self._file_row(t, path)
        current = None if row is None else str(row["version"])
        if current != expected_version:
            raise MemoryConflictError(f"memory path {path!r} changed before it could be deleted")
        if row is not None:
            status = t.delete(where=(t.path == path) & (t.kind == _KIND_FILE) & (t.version == current))
            if status.row_count_stats.del_rows != 1:
                raise MemoryConflictError(f"memory path {path!r} changed before it could be deleted")
        mutation = MemoryMutation(version=None, replayed=False, existed=row is not None)
        return self._record_operation(t, operation, mutation) or mutation

    def _file_query(self, t: pxt.Table, prefix: str) -> Any:
        pred = t.kind == _KIND_FILE
        if prefix:
            # startswith() is typed Int, not Bool.
            pred = pred & (t.path >= prefix) & (t.path < prefix + "\uffff")
        return t.where(pred)

    def _list_paths_sync(self, prefix: str, limit: int) -> list[str]:
        _validate_store_prefix(prefix)
        if limit <= 0:
            raise ValueError("limit must be positive")
        t = self._ensure_table()
        rows = self._file_query(t, prefix).select(t.path).order_by(t.path).limit(limit).collect()
        return [str(row["path"]) for row in rows]

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
        fetched = list(
            self._file_query(t, prefix).select(t.path, t.content).order_by(t.path).limit(max_files + 1).collect()
        )
        has_more = len(fetched) > max_files
        scanned_rows = fetched[:max_files]
        files = [(str(row["path"]), (row["content"] or "")[:max_file_chars]) for row in scanned_rows]
        content_truncated = any(len(row["content"] or "") > max_file_chars for row in scanned_rows)
        result = _lexical_search(
            files, query, limit=limit, max_files=max_files, max_chars=max_chars, score_prefix=prefix
        )
        return MemorySearchResult(
            matches=result.matches,
            scanned=result.scanned,
            truncated=result.truncated or content_truncated or has_more,
        )
