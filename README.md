# pydantic-ai-pixeltable

[![CI](https://github.com/pixeltable/pydantic-ai-pixeltable/actions/workflows/ci.yml/badge.svg)](https://github.com/pixeltable/pydantic-ai-pixeltable/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

[Pydantic AI Harness](https://github.com/pydantic/pydantic-ai-harness) integration for [Pixeltable](https://pixeltable.com/). Two independent layers:

- `PixeltableMemoryStore`: persist Harness `Memory` in a Pixeltable table (same `Memory(store)` slot as `FileStore` / `PostgresMemoryStore`)
- `Pixeltable`: read-only catalog tools (`list_tables`, `describe_table`, `query_table`, `similarity_search`) over tables you already have

Neither import requires the other. You can attach catalog tools without Memory, persist the notebook without catalog tools, or mix FileStore Memory with Pixeltable RAG.

Requires **Pixeltable >= 0.6.8** and **pydantic-ai-harness >= 0.29.0**.

This is an interoperability bridge. Native Pixeltable agents still use a `TableModel` and computed columns. Not on PyPI yet.

## Installation

```bash
git clone https://github.com/pixeltable/pydantic-ai-pixeltable.git
cd pydantic-ai-pixeltable
pip install -e .
```

## Quick Start

`Memory` is the short notebook (`MEMORY.md` and other path-addressed files). Search there is lexical. One path is one Markdown string, the same as FileStore. `Pixeltable` searches application tables (embedding similarity when an index exists). Do not store a handbook as Memory files, and do not split the notebook into search rows. `ToolOutputLimits` bounds oversized tool returns.

The usual pairing is official FileStore Memory plus catalog RAG: a notebook you can open, and a corpus you can search.

```python
from pydantic_ai import Agent
from pydantic_ai_harness import Memory, ToolOutputLimits
from pydantic_ai_harness.memory import FileStore
from pydantic_ai_pixeltable import Pixeltable

agent = Agent(
    "openai:gpt-4o-mini",
    capabilities=[
        Memory(FileStore(".agent-memory")),
        Pixeltable(tables=["my_app.doc_chunks"], read_only=True),
        ToolOutputLimits(),
    ],
)
```

Use `PixeltableMemoryStore` only when the notebook should live in the catalog (queryable via `.table`). That store does not require the `Pixeltable` capability.

```python
from pydantic_ai_harness import Memory
from pydantic_ai_pixeltable import PixeltableMemoryStore

Memory(PixeltableMemoryStore(table_name="harness.memory"))
```

Catalog tools do not require Memory or `PixeltableMemoryStore`:

```python
from pydantic_ai import Agent
from pydantic_ai_pixeltable import Pixeltable

agent = Agent(
    "openai:gpt-4o-mini",
    capabilities=[Pixeltable(tables=["my_app.doc_chunks"], read_only=True)],
)
```

Using both Pixeltable layers together is valid. It is not required.

`tables` is a required allowlist of table paths or directory prefixes. Pass `tables=["*"]` to allow the whole catalog.

Two `Pixeltable(...)` instances share `id="pixeltable"` and merge. Give one a different `id` if you need both. [`pydantic-ai-chdb`](https://ai.pydantic.dev/capabilities/third-party/) also registers `list_tables` and `describe_table`. Wrap one side in [`PrefixTools`](https://ai.pydantic.dev/capabilities/prefix-tools/) when you pair them.

The package talks to the local catalog.

## Persist Harness memory

The store creates the table on first use. You do not run `pxt schema update` for this table. `Memory` keeps the notebook tools, injection, and CAS retries. This package persists each path as one Markdown string, plus versions and operation receipts.

`Memory(PixeltableMemoryStore)` is Python-only. Harness YAML backends are `memory`, `file`, and `sqlite`.

### Escape hatch: `.table`

```python
store = PixeltableMemoryStore(table_name="harness.memory")
t = store.table

# File rows have kind == "file". Bookkeeping rows use reserved paths.
t.where(t.kind == "file").select(t.path, t.content, t.version).collect()
```

Use `.table` to query. A raw `t.update` of `content` is not a Memory write: the version does not change, and the next compare-and-set can overwrite it. Do not write paths whose first segment is `__meta__` or `__op__`.

## Search Pixeltable tables

`Pixeltable` does not create tables or insert rows. Point it at a chunk view that already has an embedding index and data.

In `app.py`, declare the tables and index on a `TableModel`. `pxt schema update` creates empty tables and the index. Insert `pxt.Document` rows yourself.

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

```bash
pxt schema update app.py my_app
```

A notebook or REPL can still call `create_table`, `create_view`, and `add_embedding_index`. Do not put those calls in `app.py`.

`query_table` filters with equality only (`{"status": "open"}`). `similarity_search` calls `column.similarity(string=query)` and needs an embedding index on that column. Both cap rows and serialized characters (`max_rows`, `max_chars`). Default columns skip media, array, and binary; a named media column comes back as a file URL, not a blob. `read_only=False` is not implemented.

### YAML spec

```python
from pydantic_ai import Agent
from pydantic_ai_pixeltable import Pixeltable

agent = Agent.from_spec(
    {
        "model": "openai:gpt-4o-mini",
        "capabilities": [{"Pixeltable": {"tables": ["my_app.doc_chunks"]}}],
    },
    custom_capability_types=[Pixeltable],
)
```

Without `custom_capability_types=[Pixeltable]`, `Agent.from_spec` does not know the class.

## Features

- Persistent, versioned memory via Pixeltable's embedded PostgreSQL
- Compare-and-set writes and deletes (`MemoryConflictError` on a stale version)
- Operation receipts: the same `MemoryOperation` replays when the `__op__/` row exists. A crash between the file write and that row can raise `MemoryConflictError` instead.
- Bounded lexical `search` (`SearchableMemoryStore`)
- Read-only catalog tools with an allowlist and row/character caps
- `.table` for queries. A raw `t.update` of content is not a Memory write.

File mutations use compare-and-set on the file row. Receipts are a second write after the mutation, not one SQL transaction and not a FileStore crash journal. A crash between those steps can raise `MemoryConflictError` on replay.

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
pytest tests/test_stress.py -v -m expensive
ruff check . && ruff format --check .
```

`pytest tests/ -v` skips `@pytest.mark.expensive` (~20k-row Memory and catalog volume).

## License

Apache 2.0
