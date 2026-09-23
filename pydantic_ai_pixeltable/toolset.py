"""Read-only Pixeltable catalog tools for a Pydantic AI agent."""

from __future__ import annotations

import json
from typing import Any, TypedDict, cast

import pixeltable as pxt
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset

from pydantic_ai_pixeltable._types import column_base

_MEDIA = frozenset({"Image", "Video", "Audio", "Document"})
ALL_TABLES = "*"


class TableNames(TypedDict):
    tables: list[str]


class RowsResult(TypedDict):
    table: str
    rows: list[dict[str, Any]]
    truncated: bool


class TableDescription(TypedDict):
    table: str
    kind: str | None
    comment: str | None
    columns: list[dict[str, Any]]
    indexes: list[dict[str, Any]]


# Accepted Python value types per column base type for `where` equality filters.
_WHERE_VALUE_TYPES: dict[str, type | tuple[type, ...]] = {
    "String": str,
    "Int": int,
    "Float": (int, float),
    "Bool": bool,
    "Timestamp": str,
    "Date": str,
}


def _norm(path: str) -> str:
    return path.replace("/", ".")


def _allowed(path: str, tables: list[str] | None) -> bool:
    if tables is None or tables == [ALL_TABLES]:
        return True
    npath = _norm(path)
    return any(npath == _norm(entry) or npath.startswith(f"{_norm(entry)}.") for entry in tables)


def _is_media_type(type_: str) -> bool:
    return column_base(type_) in _MEDIA


def _is_skipped_type(type_: str) -> bool:
    return column_base(type_) in {"Array", "Binary"}


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


def _bounded_payload(table: str, rows: list[dict[str, Any]], *, has_more: bool, max_chars: int) -> RowsResult:
    kept = [dict(row) for row in rows]
    truncated = has_more
    while True:
        payload = RowsResult(table=table, rows=kept, truncated=truncated)
        encoded = json.dumps(payload, ensure_ascii=False)
        if len(encoded) <= max_chars:
            return payload
        if not kept:
            return RowsResult(table=table, rows=[], truncated=True)
        # Prefer trimming the longest string cell over dropping the whole row, so one
        # oversized value cannot empty the result.
        longest_len = 0
        longest_row = longest_key = None
        for index, row in enumerate(kept):
            for key, value in row.items():
                if isinstance(value, str) and len(value) > longest_len:
                    longest_len, longest_row, longest_key = len(value), index, key
        if longest_len == 0 or longest_row is None or longest_key is None:
            kept.pop()
            truncated = True
            continue
        cut = max(longest_len - (len(encoded) - max_chars) - 3, 0)
        value = kept[longest_row][longest_key]
        kept[longest_row][longest_key] = f"{value[:cut]}..." if cut else ""
        truncated = True


class PixeltableToolset(FunctionToolset[AgentDepsT]):
    """List, describe, query, and similarity-search Pixeltable tables."""

    def __init__(self, *, tables: list[str] | None, max_rows: int, max_chars: int, id: str = "pixeltable") -> None:
        super().__init__(id=id)
        if not tables:
            raise ValueError("tables must be a non-empty allowlist; pass ['*'] to allow the whole catalog")
        self._tables = tables
        self._handles: dict[str, pxt.Table] = {}
        self._max_rows = max_rows
        self._max_chars = max_chars
        self.add_function(self.list_tables, name="list_tables")
        self.add_function(self.describe_table, name="describe_table")
        self.add_function(self.query_table, name="query_table")
        self.add_function(self.similarity_search, name="similarity_search")

    def list_tables(self) -> TableNames:
        """List Pixeltable tables this agent may use.

        Returns:
            The allowed table paths.
        """
        try:
            tables = pxt.list_tables()
        except pxt.Error as exc:
            raise ModelRetry(str(exc)) from exc
        return TableNames(tables=sorted(_norm(path) for path in tables if _allowed(path, self._tables)))

    def describe_table(self, table: str) -> TableDescription:
        """Describe a table's columns and indexes.

        Args:
            table: Pixeltable table path (for example `my_app.doc_chunks`).

        Returns:
            Kind, comment, columns, and indexes.
        """
        t = self._open_table(table)
        metadata = self._metadata(table, t)
        # pixeltable renamed the metadata key "indices" to "indexes" in 0.7.x.
        indexes = metadata.get("indexes") or metadata.get("indices") or {}
        return TableDescription(
            table=_norm(metadata["path"]),
            kind=metadata.get("kind"),
            comment=metadata.get("comment"),
            columns=[
                {"name": info.get("name"), "type": info.get("type_"), "is_computed": info.get("is_computed", False)}
                for info in metadata["columns"].values()
            ],
            indexes=[
                {"name": info["name"], "index_type": info["index_type"], "columns": info["columns"]}
                for info in indexes.values()
            ],
        )

    def query_table(
        self,
        table: str,
        columns: list[str] | None = None,
        where: dict[str, Any] | None = None,
        limit: int = 10,
    ) -> RowsResult:
        """Select rows from a table. `where` is equality only (`{"status": "open"}`).

        Args:
            table: Pixeltable table path.
            columns: Columns to return. Omit to skip media, array, binary, and
                computed columns that are not stored. A named media column is
                returned as a file URL.
            where: Equality filters mapping column name to value. Media, array, and
                binary columns reject non-null filters; `None` matches null rows.
            limit: Maximum rows to return, capped by the capability.

        Returns:
            Matching rows, with `truncated` set when the row or character cap applied.
        """
        t = self._open_table(table)
        metadata = self._metadata(table, t)
        try:
            query = self._project(t, columns, metadata)
        except (pxt.Error, TypeError, ValueError) as exc:
            raise ModelRetry(str(exc)) from exc
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
    ) -> RowsResult:
        """Nearest-neighbor search on a column that has an embedding index.

        Args:
            table: Pixeltable table path.
            query: Text to embed and search for.
            column: Column with an embedding index.
            columns: Extra columns to return with the match text and score.
            limit: Maximum rows to return, capped by the capability.
            idx: Embedding index name. Required when the column has more than one index.

        Returns:
            Rows ordered by similarity, each with a `score` field.
        """
        if not query.strip():
            raise ModelRetry("similarity_search query must be non-empty")
        t = self._open_table(table)
        metadata = self._metadata(table, t)
        column_md = metadata["columns"]
        if column not in column_md:
            raise ModelRetry(f"Unknown column {column!r} on {_norm(metadata['path'])!r}. Call describe_table.")
        selected = self._default_columns(metadata) if columns is None else list(columns)
        if column not in selected and not _is_skipped_type(column_md[column]["type_"]):
            selected = [column, *[name for name in selected if name != column]]
        # A real column named `score` must not be shadowed by the similarity score.
        score_name = "score"
        while score_name in column_md:
            score_name = f"similarity_{score_name}"
        try:
            search = (
                t[column].similarity(string=query, idx=idx) if idx is not None else t[column].similarity(string=query)
            )
            query_obj = self._project(t, selected, metadata, **{score_name: search}).order_by(search, asc=False)
            return self._collect(table, query_obj, limit)
        except pxt.Error as exc:
            self._handles.pop(table, None)
            raise ModelRetry(str(exc)) from exc

    def _metadata(self, table: str, t: pxt.Table) -> dict[str, Any]:
        try:
            metadata = cast(dict[str, Any], t.get_metadata())
        except pxt.Error as exc:
            self._handles.pop(table, None)
            raise ModelRetry(str(exc)) from exc
        if "columns" not in metadata or "path" not in metadata:
            raise ModelRetry(f"unexpected metadata shape for {table!r}")
        return metadata

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
        assert t is not None  # if_not_exists='error' raises instead of returning None
        self._handles[table] = t
        return t

    def _default_columns(self, metadata: dict[str, Any]) -> list[str]:
        columns: list[str] = []
        for name, info in metadata["columns"].items():
            type_ = info["type_"]
            if _is_media_type(type_) or _is_skipped_type(type_):
                continue
            # A computed column that is not stored recomputes at query time; an LLM UDF
            # would spend money on every call.
            if info.get("is_computed") and not info.get("is_stored"):
                continue
            columns.append(name)
        return columns

    def _project(self, t: pxt.Table, columns: list[str] | None, metadata: dict[str, Any], **extra: Any) -> Any:
        path = _norm(metadata["path"])
        column_md = metadata["columns"]
        if columns is None:
            names = self._default_columns(metadata)
        else:
            names = list(dict.fromkeys(columns))
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
            type_ = column_md[name]["type_"]
            if value is not None and (_is_skipped_type(type_) or _is_media_type(type_)):
                raise ModelRetry(f"Column {name!r} does not support equality filters. Call describe_table.")
            if value is not None and not isinstance(value, (str, int, float, bool)):
                raise ModelRetry(f"where value for {name!r} must be a scalar, got {type(value).__name__}")
            base = column_base(type_)
            expected = _WHERE_VALUE_TYPES.get(base)
            if (
                value is not None
                and expected is not None
                and (not isinstance(value, expected) or (isinstance(value, bool) and base != "Bool"))
            ):
                raise ModelRetry(f"where value for {name!r} must be {base}, got {type(value).__name__}")
            try:
                clause = t[name] == value
                pred = clause if pred is None else pred & clause
            except (pxt.Error, TypeError, ValueError) as exc:
                raise ModelRetry(f"Cannot filter {name!r} by {value!r}: {exc}") from exc
        try:
            return query.where(pred)
        except (pxt.Error, TypeError, ValueError) as exc:
            raise ModelRetry(str(exc)) from exc

    def _collect(self, table: str, query: Any, limit: int) -> RowsResult:
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
