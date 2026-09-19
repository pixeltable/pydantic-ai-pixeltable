"""Read-only Pixeltable catalog tools for a Pydantic AI agent."""

from __future__ import annotations

import json
from typing import Any

import pixeltable as pxt
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset

_MEDIA = frozenset({"Image", "Video", "Audio", "Document"})
ALL_TABLES = "*"


def _norm(path: str) -> str:
    return path.replace("/", ".")


def _allowed(path: str, tables: list[str] | None) -> bool:
    if tables is None or tables == [ALL_TABLES]:
        return True
    npath = _norm(path)
    return any(npath == _norm(entry) or npath.startswith(f"{_norm(entry)}.") for entry in tables)


def _type_base(type_: str) -> str:
    return type_.split(" | ", 1)[0].split("[", 1)[0]


def _is_media_type(type_: str) -> bool:
    return _type_base(type_) in _MEDIA


def _is_skipped_type(type_: str) -> bool:
    return _type_base(type_) in {"Array", "Binary"}


def _cell(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _cell(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_cell(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            converted = item()
        except (ValueError, TypeError):
            converted = None
        else:
            if isinstance(converted, (str, int, float, bool)):
                return converted
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return isoformat()
    return str(value)


def _row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: _cell(value) for key, value in row.items()}


def _bounded_payload(table: str, rows: list[dict[str, Any]], *, has_more: bool, max_chars: int) -> dict[str, Any]:
    kept = list(rows)
    truncated = has_more
    while True:
        payload = {"table": table, "rows": kept, "truncated": truncated}
        encoded = json.dumps(payload, ensure_ascii=False)
        if len(encoded) <= max_chars:
            return payload
        if not kept:
            return {"table": table, "rows": [], "truncated": True}
        kept.pop()
        truncated = True


class PixeltableToolset(FunctionToolset[AgentDepsT]):
    """List, describe, query, and similarity-search Pixeltable tables."""

    def __init__(self, *, tables: list[str] | None, max_rows: int, max_chars: int) -> None:
        super().__init__()
        self._tables = tables
        self._handles: dict[str, pxt.Table] = {}
        self._max_rows = max_rows
        self._max_chars = max_chars
        self.add_function(self.list_tables, name="list_tables")
        self.add_function(self.describe_table, name="describe_table")
        self.add_function(self.query_table, name="query_table")
        self.add_function(self.similarity_search, name="similarity_search")

    def list_tables(self) -> dict[str, list[str]]:
        """List Pixeltable tables this agent may use.

        Returns:
            The allowed table paths.
        """
        return {"tables": sorted(_norm(path) for path in pxt.list_tables() if _allowed(path, self._tables))}

    def describe_table(self, table: str) -> dict[str, Any]:
        """Describe a table's columns and indexes.

        Args:
            table: Pixeltable table path (for example ``my_app.doc_chunks``).

        Returns:
            Kind, comment, columns, and indexes.
        """
        t = self._open_table(table)
        try:
            metadata = t.get_metadata()
        except pxt.Error as exc:
            self._handles.pop(table, None)
            raise ModelRetry(str(exc)) from exc
        # pixeltable renamed the metadata key "indices" to "indexes" in 0.7.x.
        indexes = metadata.get("indexes") or metadata.get("indices") or {}
        return {
            "table": _norm(metadata["path"]),
            "kind": metadata["kind"],
            "comment": metadata["comment"],
            "columns": [
                {"name": info["name"], "type": info["type_"], "is_computed": info["is_computed"]}
                for info in metadata["columns"].values()
            ],
            "indexes": [
                {"name": info["name"], "index_type": info["index_type"], "columns": info["columns"]}
                for info in indexes.values()
            ],
        }

    def query_table(
        self,
        table: str,
        columns: list[str] | None = None,
        where: dict[str, Any] | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Select rows from a table. ``where`` is equality only (``{"status": "open"}``).

        Args:
            table: Pixeltable table path.
            columns: Columns to return. Omit to skip media, array, and binary.
                A named media column is returned as a file URL.
            where: Equality filters mapping column name to value.
            limit: Maximum rows to return, capped by the capability.

        Returns:
            Matching rows, with ``truncated`` set when the row or character cap applied.
        """
        t = self._open_table(table)
        try:
            metadata = t.get_metadata()
        except pxt.Error as exc:
            self._handles.pop(table, None)
            raise ModelRetry(str(exc)) from exc
        query = self._project(t, columns, metadata)
        query = self._where(query, t, where, metadata)
        return self._collect(table, query, limit)

    def similarity_search(
        self,
        table: str,
        query: str,
        column: str,
        columns: list[str] | None = None,
        limit: int = 5,
        idx: str | None = None,
    ) -> dict[str, Any]:
        """Nearest-neighbor search on a column that has an embedding index.

        Args:
            table: Pixeltable table path.
            query: Text to embed and search for.
            column: Column with an embedding index.
            columns: Extra columns to return with the match text and score.
            limit: Maximum rows to return, capped by the capability.
            idx: Embedding index name. Required when the column has more than one index.

        Returns:
            Rows ordered by similarity, each with a ``score`` field.
        """
        if not query.strip():
            raise ModelRetry("similarity_search query must be non-empty")
        t = self._open_table(table)
        try:
            metadata = t.get_metadata()
        except pxt.Error as exc:
            self._handles.pop(table, None)
            raise ModelRetry(str(exc)) from exc
        column_md = metadata["columns"]
        if column not in column_md:
            raise ModelRetry(f"Unknown column {column!r} on {_norm(metadata['path'])!r}. Call describe_table.")
        selected = self._default_columns(metadata) if columns is None else list(columns)
        if column not in selected and not _is_skipped_type(column_md[column]["type_"]):
            selected = [column, *[name for name in selected if name != column]]
        try:
            search = (
                t[column].similarity(string=query, idx=idx) if idx is not None else t[column].similarity(string=query)
            )
            query_obj = self._project(t, selected, metadata, score=search).order_by(search, asc=False)
            return self._collect(table, query_obj, limit)
        except pxt.Error as exc:
            self._handles.pop(table, None)
            raise ModelRetry(str(exc)) from exc

    def _open_table(self, table: str) -> pxt.Table:
        if not _allowed(table, self._tables):
            raise ModelRetry(f"Table {table!r} is not in the Pixeltable allowlist. Call list_tables.")
        cached = self._handles.get(table)
        if cached is not None:
            return cached
        try:
            t = pxt.get_table(table)
        except pxt.Error as exc:
            raise ModelRetry(f"Cannot open table {table!r}: {exc}") from exc
        self._handles[table] = t
        return t

    def _default_columns(self, metadata: dict[str, Any]) -> list[str]:
        columns: list[str] = []
        for name, info in metadata["columns"].items():
            type_ = info["type_"]
            if _is_media_type(type_) or _is_skipped_type(type_):
                continue
            columns.append(name)
        return columns

    def _project(self, t: pxt.Table, columns: list[str] | None, metadata: dict[str, Any], **extra: Any) -> Any:
        path = _norm(metadata["path"])
        column_md = metadata["columns"]
        if columns is None:
            names = self._default_columns(metadata)
        else:
            names = list(columns)
            if not names:
                raise ModelRetry("query_table needs at least one column")
        items: list[Any] = []
        named: dict[str, Any] = dict(extra)
        for name in names:
            if name in named:
                continue
            if name not in column_md:
                raise ModelRetry(f"Unknown column {name!r} on {path!r}. Call describe_table.")
            type_ = column_md[name]["type_"]
            if _is_skipped_type(type_):
                continue
            ref = t[name]
            if _is_media_type(type_):
                named[name] = ref.fileurl
            else:
                items.append(ref)
        if not items and not named:
            raise ModelRetry("No selectable columns. Name a non-array column, or describe_table.")
        return t.select(*items, **named)

    def _where(self, query: Any, t: pxt.Table, where: dict[str, Any] | None, metadata: dict[str, Any]) -> Any:
        if not where:
            return query
        column_md = metadata["columns"]
        pred = None
        for name, value in where.items():
            if name not in column_md:
                raise ModelRetry(f"Unknown column {name!r} in where. Call describe_table.")
            clause = t[name] == value
            pred = clause if pred is None else pred & clause
        return query.where(pred)

    def _collect(self, table: str, query: Any, limit: int) -> dict[str, Any]:
        if limit < 1:
            raise ModelRetry("limit must be at least 1")
        n = min(limit, self._max_rows)
        try:
            fetched = list(query.limit(n + 1).collect())
        except pxt.Error as exc:
            self._handles.pop(table, None)
            raise ModelRetry(str(exc)) from exc
        has_more = len(fetched) > n
        rows = [_row(dict(row)) for row in fetched[:n]]
        return _bounded_payload(_norm(table), rows, has_more=has_more, max_chars=self._max_chars)
