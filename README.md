# pydantic-ai-pixeltable

[![CI](https://github.com/pixeltable/pydantic-ai-pixeltable/actions/workflows/ci.yml/badge.svg)](https://github.com/pixeltable/pydantic-ai-pixeltable/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/pydantic-ai-pixeltable.svg)](https://pypi.org/project/pydantic-ai-pixeltable/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/pixeltable/pydantic-ai-pixeltable/blob/main/LICENSE)

[Pydantic AI Harness](https://github.com/pydantic/pydantic-ai-harness) integration for [Pixeltable](https://pixeltable.com/). Two independent layers:

- `PixeltableMemoryStore`: persist Harness `Memory` in a Pixeltable table (same `Memory(store)` slot as `FileStore`)
- `Pixeltable`: read-only catalog tools (`list_tables`, `describe_table`, `query_table`, `similarity_search`) over tables you already have

Neither requires the other. Requires **Pixeltable >= 0.7.8** and **pydantic-ai-harness >= 0.29.0**.

This is an interoperability bridge. Native Pixeltable agents still use a `TableModel` and computed columns.

## How it fits together

```text
Native backend (FileStore)                 This package (Pixeltable)

Memory(FileStore)                          Memory(PixeltableMemoryStore)
  └ .agent-memory/                           └ pxt table 'harness.memory'
      ├ main/MEMORY.md                           ├ kind='file' rows  path, content, version
      ├ ... (one .md per memory path)            └ kind='op' rows    '__op__/<id>' receipts
      └ .memory-store.sqlite3                  same catalog as your data; inspect via store.table
          (versions + operation journal)
                                           Pixeltable(tables=[...])  (optional, independent)
                                             └ your existing tables/views, incl. pxt.Document and
                                               embedding indexes -> query_table, similarity_search
```

Unique to this backend: memory rows are ordinary catalog rows. `store.search` stays lexical (the Harness contract), but through `store.table` you can add an embedding index on `content` and run Pixeltable similarity queries over memory, or join it with the rest of the catalog.

## Installation

```bash
pip install pydantic-ai-pixeltable
```

## Quick Start

Memory in the catalog, plus read-only tools over the tables you already have:

```python
from pydantic_ai import Agent
from pydantic_ai_harness import Memory, ToolOutputLimits
from pydantic_ai_pixeltable import Pixeltable, PixeltableMemoryStore

agent = Agent(
    "anthropic:claude-sonnet-4-6",
    capabilities=[
        Memory(PixeltableMemoryStore(table_name="harness.memory")),
        Pixeltable(tables=["my_app.doc_chunks"]),
        ToolOutputLimits(),
    ],
)
```

Prefer `FileStore` (or `PostgresMemoryStore`) when the notebook does not need to live in the catalog; the choice is per project.

`tables` is a required allowlist of table paths or directory prefixes. Pass `tables=["*"]` to allow the whole catalog, including the memory table and its `__op__` receipt payloads.

Two `Pixeltable(...)` instances share `id="pixeltable"` and merge by narrowing: an entry survives only when every capability covers it, and a disjoint merge raises. [`pydantic-ai-chdb`](https://ai.pydantic.dev/capabilities/third-party/) also registers `list_tables` and `describe_table`; wrap one side in [`PrefixTools`](https://ai.pydantic.dev/capabilities/prefix-tools/) when you pair them.

## Catalog tools

`Pixeltable` does not create tables or insert rows. Point it at a table or view that already has data, plus an embedding index for `similarity_search`.

- `query_table` filters with equality only (`{"status": "open"}`); media, array, and binary columns reject non-null filters.
- `similarity_search` calls `column.similarity(string=query)` and needs an embedding index on that column.
- Output is bounded by `max_rows` and `max_chars`; the minimal `{"table", "rows", "truncated"}` envelope is always returned, even when `max_chars` is set below its size. Default columns skip media, array, binary, and computed columns that are not stored (they would recompute per query); a named media column returns a file URL, not a blob.

Declare the tables and index on a `TableModel` in `app.py`; `pxt schema update` creates them:

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

### YAML spec

```python
agent = Agent.from_spec(
    {
        "model": "anthropic:claude-sonnet-4-6",
        "capabilities": [{"Pixeltable": {"tables": ["my_app.doc_chunks"]}}],
    },
    custom_capability_types=[Pixeltable],
)
```

Without `custom_capability_types=[Pixeltable]`, `Agent.from_spec` does not know the class.

## Memory store

- The table is created on first use; you do not run `pxt schema update` for it.
- Each path is one row (`kind == "file"`); operation receipts are rows under `__op__/`. Path roots `__meta__` and `__op__` are reserved.
- Compare-and-set: `expected_version` must equal the row's version. Versions are unique UUID strings, not monotonic. A stale version raises `MemoryConflictError`.
- Operation receipts are journaled `__op__` rows: the intended mutation is recorded before it is applied, so a mid-write crash rolls forward or replays cleanly instead of double-applying.
- `store.compact()` rebuilds the table from live rows. Pixeltable keeps old row versions for updates and deletes, so compaction is the only way to reclaim storage. Run it with writers paused; it refuses tables with user-added columns or indexes.
- `Memory(PixeltableMemoryStore)` is Python-only. Harness YAML backends are `memory`, `file`, and `sqlite`.
- This package emits no telemetry of its own; the `memory.*` spans come from the Harness `Memory` capability.

### Escape hatch: `.table`

```python
store = PixeltableMemoryStore(table_name="harness.memory")
t = store.table
t.where(t.kind == "file").select(t.path, t.content, t.version).collect()
```

Use `.table` to query. A raw `t.update` of `content` is not a Memory write: the version does not change, and the next compare-and-set can overwrite it.

## Measured results

Pixeltable 0.7.8 on embedded PostgreSQL, ~20k rows per test, laptop hardware (`pytest tests/test_stress.py -v -m expensive`):

| Operation | Time |
| --- | --- |
| `list_paths`, limit 50 over 20k files | ~12 ms |
| `search` (lexical), 100-file scan bound | ~8 ms |
| CAS write under contention (200 tasks, 4 paths) | ~5 ms per attempt, exactly 4 winners |
| `query_table` equality filter, limit 20 over 20k rows | ~10 ms |
| `similarity_search` top-3 over 20k indexed rows | ~8 ms |
| Embedding index build over 20k rows | ~2.3 s |

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
