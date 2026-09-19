# pydantic-ai-pixeltable

[![CI](https://github.com/pixeltable/pydantic-ai-pixeltable/actions/workflows/ci.yml/badge.svg)](https://github.com/pixeltable/pydantic-ai-pixeltable/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

[Pydantic AI Harness](https://github.com/pydantic/pydantic-ai-harness) integration for [Pixeltable](https://pixeltable.com/). Two independent layers:

- `PixeltableMemoryStore`: persist Harness `Memory` in a Pixeltable table (same `Memory(store)` slot as `FileStore`)
- `Pixeltable`: read-only catalog tools (`list_tables`, `describe_table`, `query_table`, `similarity_search`) over tables you already have

Neither requires the other. Requires **Pixeltable >= 0.6.8** and **pydantic-ai-harness >= 0.29.0**. Not on PyPI yet.

This is an interoperability bridge. Native Pixeltable agents still use a `TableModel` and computed columns.

## Installation

```bash
pip install git+https://github.com/pixeltable/pydantic-ai-pixeltable.git
```

## Quick Start

The usual pairing is FileStore Memory (a notebook you can open) plus Pixeltable catalog tools (a corpus you can search). Search inside Memory is lexical; `similarity_search` on catalog tables uses the embedding index.

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

Use `PixeltableMemoryStore` when the notebook should live in the catalog:

```python
from pydantic_ai_harness import Memory
from pydantic_ai_pixeltable import PixeltableMemoryStore

Memory(PixeltableMemoryStore(table_name="harness.memory"))
```

`tables` is a required allowlist of table paths or directory prefixes. Pass `tables=["*"]` to allow the whole catalog.

Two `Pixeltable(...)` instances share `id="pixeltable"` and merge. [`pydantic-ai-chdb`](https://ai.pydantic.dev/capabilities/third-party/) also registers `list_tables` and `describe_table`; wrap one side in [`PrefixTools`](https://ai.pydantic.dev/capabilities/prefix-tools/) when you pair them.

## Catalog tools

`Pixeltable` does not create tables or insert rows. Point it at a table or view that already has data, plus an embedding index for `similarity_search`.

- `query_table` filters with equality only (`{"status": "open"}`).
- `similarity_search` calls `column.similarity(string=query)` and needs an embedding index on that column.
- Output is bounded by `max_rows` and `max_chars`. Default columns skip media, array, and binary; a named media column returns a file URL, not a blob.
- `read_only=False` is not implemented.

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
        "model": "openai:gpt-4o-mini",
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
- Receipts are a second write after the file mutation, not one SQL transaction. A crash between them can raise `MemoryConflictError` on replay instead of `replayed=True`.
- `Memory(PixeltableMemoryStore)` is Python-only. Harness YAML backends are `memory`, `file`, and `sqlite`.

### Escape hatch: `.table`

```python
store = PixeltableMemoryStore(table_name="harness.memory")
t = store.table
t.where(t.kind == "file").select(t.path, t.content, t.version).collect()
```

Use `.table` to query. A raw `t.update` of `content` is not a Memory write: the version does not change, and the next compare-and-set can overwrite it.

Tables created by pre-release revisions with an `Int` `version` column are rejected on first use; drop and recreate them.

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

Against the previous revision: cached table handles remove one catalog lookup per call, catalog tools fetch metadata once per call instead of twice, and UUID versions remove the shared generation-row update, cutting contended CAS write time ~20% (1.26 s to 1.02 s for 200 attempts).

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
