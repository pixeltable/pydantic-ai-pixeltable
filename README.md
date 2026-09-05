# pydantic-ai-pixeltable

[![PyPI](https://img.shields.io/pypi/v/pydantic-ai-pixeltable)](https://pypi.org/project/pydantic-ai-pixeltable/)
[![CI](https://github.com/pixeltable/pydantic-ai-pixeltable/actions/workflows/ci.yml/badge.svg)](https://github.com/pixeltable/pydantic-ai-pixeltable/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

[Pydantic AI Harness](https://github.com/pydantic/pydantic-ai-harness) integration for [Pixeltable](https://pixeltable.com/). Two layers:

- `PixeltableMemoryStore` — drop-in `MemoryStore` for the existing `Memory` capability (same slot as `FileStore` / `PostgresMemoryStore`)
- `Pixeltable` — read-only catalog tools (`list_tables`, `describe_table`, `query_table`, `similarity_search`) over tables you already have

Requires **Pixeltable >= 0.6.8** and **pydantic-ai-harness >= 0.29.0**.

This is an interoperability bridge. Native Pixeltable agents still use a `TableModel` and computed columns.

## Installation

```bash
pip install pydantic-ai-pixeltable
```

## Quick Start

```python
from pydantic_ai import Agent
from pydantic_ai_harness import Memory, ToolOutputLimits
from pydantic_ai_pixeltable import Pixeltable, PixeltableMemoryStore

agent = Agent(
    "openai:gpt-4o-mini",
    capabilities=[
        Memory(PixeltableMemoryStore(table_name="harness.memory")),
        Pixeltable(tables=["my_app.doc_chunks"], read_only=True),
        ToolOutputLimits(),
    ],
)
```

`Memory` is the short notebook (`MEMORY.md` and other path-addressed files). `Pixeltable` searches application tables. Pair them; do not store a handbook as Memory files. `ToolOutputLimits` bounds oversized tool returns.

`tables` is an allowlist of table paths or directory prefixes. Omit it only for local use against the whole catalog.

## Persist Harness memory

The store creates the table on first use. You do not run `pxt schema update` for this table.

```python
from pydantic_ai_harness import Memory
from pydantic_ai_pixeltable import PixeltableMemoryStore

Memory(PixeltableMemoryStore(table_name="harness.memory"))
```

`Memory` keeps the notebook tools, injection, and CAS retries. This package persists paths, versions, and operation receipts.

### Escape hatch: `.table`

```python
store = PixeltableMemoryStore(table_name="harness.memory")
t = store.table

# File rows have kind == "file". Bookkeeping rows use reserved paths.
t.where(t.kind == "file").select(t.path, t.content, t.version).collect()
```

Do not write paths whose first segment is `__meta__` or `__op__`.

## Search Pixeltable tables

`Pixeltable` does not create tables. Point it at a chunk view (or any table) that already has an embedding index:

```python
import pixeltable as pxt
from pixeltable.functions.document import document_splitter

docs = pxt.create_table("my_app.docs", {"doc": pxt.Document}, if_exists="ignore")
chunks = pxt.create_view(
    "my_app.doc_chunks",
    docs,
    iterator=document_splitter(docs.doc, separators="paragraph"),
    if_exists="ignore",
)
chunks.add_embedding_index("text", embedding=your_embed_fn, if_exists="ignore")
```

`query_table` filters with equality only (`{"status": "open"}`). `similarity_search` calls `column.similarity(string=query)`. Both cap rows and serialized characters (`max_rows`, `max_chars`). Media columns are returned as file URLs, not blobs. `read_only=False` is not implemented.

## Features

- Persistent, versioned memory via Pixeltable's embedded PostgreSQL
- Compare-and-set writes and deletes (`MemoryConflictError` on a stale version)
- Idempotent operation receipts (`MemoryOperation` replay)
- Bounded lexical `search` (`SearchableMemoryStore`)
- Read-only catalog tools with an allowlist and row/character caps
- `.table` escape hatch for computed columns and arbitrary queries

File mutations use compare-and-set on the file row. Receipts are written after the mutation (the same two-phase pattern as Harness `FileStore`), not as one SQL transaction.

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
ruff check . && ruff format --check .
```

## License

Apache 2.0
