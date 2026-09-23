# pydantic-ai-pixeltable

[![CI](https://github.com/pixeltable/pydantic-ai-pixeltable/actions/workflows/ci.yml/badge.svg)](https://github.com/pixeltable/pydantic-ai-pixeltable/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/pydantic-ai-pixeltable.svg)](https://pypi.org/project/pydantic-ai-pixeltable/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/pixeltable/pydantic-ai-pixeltable/blob/main/LICENSE)

Give a [Pydantic AI](https://ai.pydantic.dev/) agent read access to your [Pixeltable](https://pixeltable.com/) tables, and optionally keep its [Harness](https://github.com/pydantic/pydantic-ai-harness) `Memory` in the same catalog.

- `Pixeltable`: read-only tools `list_tables`, `describe_table`, `query_table`, and `similarity_search` over tables and views you already have.
- `PixeltableMemoryStore`: a Harness `MemoryStore`, used as `Memory(store)` in place of `FileStore`.

Each works without the other. Requires Python 3.11+, pydantic-ai-slim 2.38+, pydantic-ai-harness 0.29.0+, and Pixeltable 0.7.8+.

## Quick start

```bash
pip install pydantic-ai-pixeltable "pydantic-ai-slim[openai]"
```

```python
import pixeltable as pxt
from pixeltable.functions.openai import embeddings
from pydantic_ai import Agent
from pydantic_ai_harness import Memory
from pydantic_ai_pixeltable import Pixeltable, PixeltableMemoryStore

handbook = pxt.create_table("handbook", {"topic": pxt.String, "text": pxt.String})
handbook.insert([{"topic": "expenses", "text": "Meals during business travel are reimbursed up to 60 EUR per day."}])
handbook.add_embedding_index("text", embedding=embeddings.using(model="text-embedding-3-small"))

agent = Agent(
    "openai:gpt-5.6-sol",
    capabilities=[
        Pixeltable(["handbook"]),  # list_tables, describe_table, query_table, similarity_search
        Memory(PixeltableMemoryStore(table_name="memory")),  # notes kept across runs
    ],
)
print(agent.run_sync("Can I expense a 75 EUR dinner? Remember that I travel monthly.").output)
# The answer cites the 60 EUR daily limit; the note lands in the memory table.
```

[`quickstart.py`](https://github.com/pixeltable/pydantic-ai-pixeltable/blob/main/quickstart.py) is the full runnable version: a two-run HR assistant whose second run answers from what the first one remembered. In an app, declare the table on a `TableModel` and create it with `pxt schema update`.

## Catalog tools

- `tables` is a required allowlist of table paths or directory prefixes. `["*"]` allows the whole catalog, including a memory table. A view inside an allowed directory exposes its base table's columns. Version handles (`tbl:3`) are refused, since old versions keep deleted rows and dropped columns.
- `similarity_search` needs an embedding index on the column. `query_table` filters by equality only; timestamp, date, and UUID values are ISO strings.
- Default columns skip media, array, binary, and unstored computed columns, which recompute on every read (possibly a model call) and also reject filters. A named media column returns a file URL.
- `max_rows` (20) and `max_chars` (8000) bound every result. An oversized string is cut to end in `...` and any other oversized value becomes `null`; the `{"table", "rows", "truncated"}` envelope is always returned.
- Two instances on one agent share `id="pixeltable"` and merge by intersecting their allowlists; a disjoint merge raises. A capability passed to a single run replaces the agent's, so it can widen access.
- [`pydantic-ai-chdb`](https://github.com/chdb-io/pydantic-ai-chdb) registers the same `list_tables` and `describe_table` names; wrap one in [`PrefixTools`](https://ai.pydantic.dev/capabilities/prefix-tools/) to use both.
- From a spec: `Agent.from_spec({"model": ..., "capabilities": [{"Pixeltable": ["handbook"]}]}, custom_capability_types=[Pixeltable])`.

## Memory store

- The table is created on first use. Each memory path is a `kind == "file"` row, and operation receipts are `__op__/` rows, so the roots `__op__` and `__meta__` are reserved. Paths are at most 255 characters.
- Writes are compare-and-set on a UUID version. The intent is journaled before the write, so a crash rolls forward on replay instead of applying twice.
- `store.table` is an ordinary table: query it, join it, or add an embedding index on `content`. `search_memory` stays lexical, as in every Harness store. A direct `t.update` is not a Memory write and leaves the version unchanged.
- Pixeltable keeps every row version and receipts are never pruned, so the table grows with history.
- Python only: Harness YAML backends are `memory`, `file`, and `sqlite`. The package emits no telemetry; `memory.*` spans come from Harness `Memory`.

On Pixeltable 0.7.8 with embedded Postgres, 20k rows, laptop (`pytest tests/test_stress.py -m expensive`): `list_paths` and `search` take ~115 and ~125 ms with 18k files under the prefix, a contended CAS write ~5 ms per attempt, `query_table` ~10 ms, `similarity_search` ~8 ms. `list_paths` and `search` sort in Python because database ordering depends on collation, so they scale with the files under the prefix, not with the table.

## Development

```bash
pip install -e ".[dev]"
pytest tests/            # add -m expensive for the 20k-row volume tests
ruff check . && ruff format --check . && mypy pydantic_ai_pixeltable
```

## License

Apache 2.0
