"""Pixeltable backend for Pydantic AI Harness ``Memory``."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pixeltable as pxt
from pydantic_ai_harness.memory import (
    MemoryConflictError,
    MemoryFile,
    MemoryMutation,
    MemoryOperation,
    MemoryOperationConflictError,
    MemorySearchResult,
)
from pydantic_ai_harness.memory._store import lexical_search, validate_store_path, validate_store_prefix

_KIND_FILE = "file"
_KIND_META = "meta"
_KIND_OP = "op"
_META_PATH = "__meta__/generation"
_OP_PREFIX = "__op__/"
_RESERVED_ROOTS = frozenset({"__meta__", "__op__"})


def _reject_reserved_path(path: str) -> None:
    if path.split("/", 1)[0] in _RESERVED_ROOTS:
        raise ValueError(f"memory path {path!r} is reserved for store bookkeeping")


class PixeltableMemoryStore:
    """Pydantic AI Harness ``MemoryStore`` persisted in a Pixeltable table.

    Implements ``MemoryStore`` and ``SearchableMemoryStore``. File mutations use
    compare-and-set on the file row; operation receipts are written afterward
    (the FileStore two-phase pattern, not one SQL transaction). Paths whose
    first segment is ``__meta__`` or ``__op__`` are reserved.

    Args:
        table_name: Pixeltable table path (e.g. ``'harness.memory'``).
    """

    def __init__(self, table_name: str = "harness.memory") -> None:
        self._table_name = table_name
        self._lock = threading.RLock()

    @property
    def table(self) -> pxt.Table:
        """Underlying Pixeltable table for computed columns and queries."""
        with self._lock:
            return self._ensure_table()

    async def read(self, path: str, *, max_chars: int) -> MemoryFile | None:
        return await asyncio.to_thread(self._locked, self._read_sync, path, max_chars)

    async def get_operation(self, operation: MemoryOperation) -> MemoryMutation | None:
        return await asyncio.to_thread(self._locked, self._get_operation_sync, operation)

    async def write(
        self,
        path: str,
        content: str,
        *,
        expected_version: str | None,
        operation: MemoryOperation | None = None,
    ) -> MemoryMutation:
        return await asyncio.to_thread(self._locked, self._write_sync, path, content, expected_version, operation)

    async def delete(
        self,
        path: str,
        *,
        expected_version: str | None,
        operation: MemoryOperation | None = None,
    ) -> MemoryMutation:
        return await asyncio.to_thread(self._locked, self._delete_sync, path, expected_version, operation)

    async def list_paths(self, prefix: str = "", *, limit: int) -> list[str]:
        return await asyncio.to_thread(self._locked, self._list_paths_sync, prefix, limit)

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
        return await asyncio.to_thread(
            self._locked, self._search_sync, prefix, query, limit, max_files, max_chars, max_file_chars
        )

    def _locked(self, fn: Any, *args: Any) -> Any:
        with self._lock:
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
        try:
            return pxt.get_table(self._table_name)
        except Exception:
            pass
        self._ensure_dirs()
        schema: dict[str, Any] = {
            "path": pxt.String,
            "kind": pxt.String,
            "content": pxt.String | None,
            "version": pxt.Int | None,
            "last_operation_id": pxt.String | None,
            "fingerprint": pxt.String | None,
            "existed": pxt.Bool | None,
        }
        return pxt.create_table(self._table_name, schema, primary_key="path", if_exists="ignore")

    def _file_row(self, t: pxt.Table, path: str) -> dict[str, Any] | None:
        rows = (
            t.where((t.path == path) & (t.kind == _KIND_FILE))
            .select(t.content, t.version, t.last_operation_id)
            .collect()
        )
        if len(rows) == 0:
            return None
        return rows[0]

    def _next_generation(self, t: pxt.Table) -> int:
        rows = t.where(t.path == _META_PATH).select(t.version).collect()
        if len(rows) == 0:
            t.insert(
                [
                    {
                        "path": _META_PATH,
                        "kind": _KIND_META,
                        "content": None,
                        "version": 1,
                        "last_operation_id": None,
                        "fingerprint": None,
                        "existed": None,
                    }
                ]
            )
            return 1
        status = t.update({"version": t.version + 1}, where=t.path == _META_PATH, return_rows=True)
        if not status.rows:
            raise RuntimeError("failed to increment memory generation")
        return int(status.rows[0]["version"])

    def _lookup_operation(self, t: pxt.Table, operation: MemoryOperation) -> MemoryMutation | None:
        rows = t.where(t.path == f"{_OP_PREFIX}{operation.id}").select(t.fingerprint, t.version, t.existed).collect()
        if len(rows) == 0:
            return None
        row = rows[0]
        if row["fingerprint"] != operation.fingerprint:
            raise MemoryOperationConflictError(f"operation id {operation.id!r} was reused with different arguments")
        version = None if row["version"] is None else str(row["version"])
        return MemoryMutation(version=version, replayed=True, existed=bool(row["existed"]))

    def _record_operation(self, t: pxt.Table, operation: MemoryOperation | None, mutation: MemoryMutation) -> None:
        if operation is None:
            return
        t.insert(
            [
                {
                    "path": f"{_OP_PREFIX}{operation.id}",
                    "kind": _KIND_OP,
                    "content": None,
                    "version": int(mutation.version) if mutation.version is not None else None,
                    "last_operation_id": None,
                    "fingerprint": operation.fingerprint,
                    "existed": mutation.existed,
                }
            ]
        )

    def _read_sync(self, path: str, max_chars: int) -> MemoryFile | None:
        validate_store_path(path)
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
        validate_store_path(path)
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
        version = self._next_generation(t)
        op_id = operation.id if operation else None
        if row is None:
            t.insert(
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
                ]
            )
        else:
            status = t.update(
                {"content": content, "version": version, "last_operation_id": op_id},
                where=(t.path == path) & (t.kind == _KIND_FILE) & (t.version == int(current)),
            )
            if status.row_count_stats.upd_rows != 1:
                raise MemoryConflictError(f"memory path {path!r} changed before it could be written")
        mutation = MemoryMutation(version=str(version), replayed=False, existed=row is not None)
        self._record_operation(t, operation, mutation)
        return mutation

    def _delete_sync(
        self,
        path: str,
        expected_version: str | None,
        operation: MemoryOperation | None,
    ) -> MemoryMutation:
        validate_store_path(path)
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
        self._next_generation(t)
        if row is not None:
            status = t.delete(where=(t.path == path) & (t.kind == _KIND_FILE) & (t.version == int(current)))
            if status.row_count_stats.del_rows != 1:
                raise MemoryConflictError(f"memory path {path!r} changed before it could be deleted")
        mutation = MemoryMutation(version=None, replayed=False, existed=row is not None)
        self._record_operation(t, operation, mutation)
        return mutation

    def _list_paths_sync(self, prefix: str, limit: int) -> list[str]:
        validate_store_prefix(prefix)
        if limit <= 0:
            raise ValueError("limit must be positive")
        t = self._ensure_table()
        rows = t.where(t.kind == _KIND_FILE).select(t.path).order_by(t.path).collect()
        return [str(row["path"]) for row in rows if str(row["path"]).startswith(prefix)][:limit]

    def _search_sync(
        self,
        prefix: str,
        query: str,
        limit: int,
        max_files: int,
        max_chars: int,
        max_file_chars: int,
    ) -> MemorySearchResult:
        validate_store_prefix(prefix)
        if not query.split() or limit <= 0 or max_files <= 0 or max_chars <= 0 or max_file_chars <= 0:
            return MemorySearchResult(matches=[], scanned=0, truncated=False)
        t = self._ensure_table()
        rows = t.where(t.kind == _KIND_FILE).select(t.path, t.content).order_by(t.path).collect()
        selected = [row for row in rows if str(row["path"]).startswith(prefix)]
        scanned_rows = selected[:max_files]
        files = [(str(row["path"]), (row["content"] or "")[:max_file_chars]) for row in scanned_rows]
        content_truncated = any(len(row["content"] or "") > max_file_chars for row in scanned_rows)
        result = lexical_search(
            files, query, limit=limit, max_files=max_files, max_chars=max_chars, score_prefix=prefix
        )
        return MemorySearchResult(
            matches=result.matches,
            scanned=result.scanned,
            truncated=result.truncated or content_truncated or len(selected) > max_files,
        )
