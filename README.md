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

`tables` is a required allowlist of table paths or directory prefixes. Pass `tables=["*"]` to allow the whole catalog.

Two `Pixeltable(...)` instances share `id="pixeltable"` and merge. Give one a different `id` if you need both. Tool names match [pydantic-ai-chdb](https://ai.pydantic.dev/capabilities/third-party/). Wrap one side in [`PrefixTools`](https://ai.pydantic.dev/capabilities/prefix-tools/) when you pair them.

A YAML spec can construct `Pixeltable` (`from_spec`, with `custom_capability_types=[Pixeltable]`). `Memory(PixeltableMemoryStore)` is Python-only. The package talks to the local catalog.

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

`Pixeltable` does not create tables. Point it at a chunk view that already has an embedding index. In `app.py`, declare the handbook and index on a `TableModel` and create them with `pxt schema update`:

```python
import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable.functions.huggingface import sentence_transformer

TableModel = pxt.model_base()
embed_fn = sentence_transformer.using(model_id="intfloat/multilingual-e5-large-instruct")


class Docs(TableModel, name="docs"):
    document: pxt.Document


class Chunks(
    TableModel,
    name="doc_chunks",
    base=Docs,
    iterator=pxtf.document.document_splitter(Docs.document, separators="paragraph"),
):
    __indexes__ = [pxt.EmbeddingIndex(text, embedding=embed_fn, name="chunks_embed")]  # type: ignore[name-defined]
```

A notebook or REPL can still call `create_table`, `create_view`, and `add_embedding_index`. Do not put those calls in `app.py`.

`query_table` filters with equality only (`{"status": "open"}`). `similarity_search` calls `column.similarity(string=query)`. Both cap rows and serialized characters (`max_rows`, `max_chars`). Default columns skip media, array, and binary; a named media column comes back as a file URL, not a blob. `read_only=False` is not implemented.

## Features

- Persistent, versioned memory via Pixeltable's embedded PostgreSQL
- Compare-and-set writes and deletes (`MemoryConflictError` on a stale version)
- Idempotent operation receipts (`MemoryOperation` replay)
- Bounded lexical `search` (`SearchableMemoryStore`)
- Read-only catalog tools with an allowlist and row/character caps
- `.table` escape hatch for computed columns and arbitrary queries

File mutations use compare-and-set on the file row. Receipts are a second write after the mutation, not one SQL transaction and not a FileStore crash journal. A crash between those steps can raise `MemoryConflictError` on replay.

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
ruff check . && ruff format --check .
```

## License

Apache 2.0
