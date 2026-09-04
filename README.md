# pydantic-ai-pixeltable

[![PyPI](https://img.shields.io/pypi/v/pydantic-ai-pixeltable)](https://pypi.org/project/pydantic-ai-pixeltable/)
[![CI](https://github.com/pixeltable/pydantic-ai-pixeltable/actions/workflows/ci.yml/badge.svg)](https://github.com/pixeltable/pydantic-ai-pixeltable/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

[Pydantic AI Harness](https://github.com/pydantic/pydantic-ai-harness) `MemoryStore` backed by [Pixeltable](https://pixeltable.com/). Drop it into the existing `Memory` capability the same way you pass `FileStore` or `PostgresMemoryStore`.

Requires **Pixeltable >= 0.6.8** and **pydantic-ai-harness >= 0.29.0**.

This is an interoperability bridge. Native Pixeltable agents still use a `TableModel` and computed columns.

## Installation

```bash
pip install pydantic-ai-pixeltable
```

## Quick Start

```python
from pydantic_ai import Agent
from pydantic_ai_harness import Memory
from pydantic_ai_pixeltable import PixeltableMemoryStore

agent = Agent(
    "openai:gpt-4o",
    capabilities=[Memory(PixeltableMemoryStore(table_name="harness.memory"))],
)
```

`Memory` keeps the notebook tools, injection, and CAS retries. This package only persists paths, versions, and operation receipts.

## Pixeltable Escape Hatch: `.table`

```python
store = PixeltableMemoryStore(table_name="harness.memory")
t = store.table

# File rows have kind == "file". Bookkeeping rows use reserved paths.
t.where(t.kind == "file").select(t.path, t.content, t.version).collect()
```

Use `.table` for Pixeltable-native queries and computed columns. Do not write paths whose first segment is `__meta__` or `__op__`; those rows hold the generation counter and operation receipts.

## Features

- Persistent, versioned memory via Pixeltable's embedded PostgreSQL
- Compare-and-set writes and deletes (`MemoryConflictError` on a stale version)
- Idempotent operation receipts (`MemoryOperation` replay)
- Bounded lexical `search` (`SearchableMemoryStore`)
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
