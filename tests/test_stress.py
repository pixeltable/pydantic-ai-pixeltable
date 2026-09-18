"""Local volume checks: ~20k Memory files and catalog rows."""

from __future__ import annotations

import asyncio
import time
import uuid

import pixeltable as pxt
import pytest
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai_harness.memory import MemoryConflictError

from pydantic_ai_pixeltable import Pixeltable, PixeltableMemoryStore
from tests.embed import tiny_embed

pytestmark = pytest.mark.expensive

N_A = 18_000
N_B = 2_000
N_CHUNKS = 20_000
TARGET_TEXT = "cats sit on mats"


@pytest.fixture()
def root():
    name = f"test_pydantic_ai_stress_{uuid.uuid4().hex[:8]}"
    pxt.create_dir(name)
    yield name
    try:
        pxt.drop_dir(name, force=True)
    except Exception:
        pass


def _file_row(path: str, content: str, version: int) -> dict[str, object]:
    return {
        "path": path,
        "kind": "file",
        "content": content,
        "version": version,
        "last_operation_id": None,
        "fingerprint": None,
        "existed": None,
    }


async def test_memory_prefix_list_search_and_cas(root: str) -> None:
    store = PixeltableMemoryStore(table_name=f"{root}.memory")
    t = store.table
    rows = [_file_row(f"tenant-a/n{i}.md", "alpha note", i + 1) for i in range(N_A)]
    rows.extend(_file_row(f"tenant-b/n{i}.md", "beta private", N_A + i + 1) for i in range(N_B))
    started = time.perf_counter()
    for offset in range(0, len(rows), 5_000):
        t.insert(rows[offset : offset + 5_000])
    insert_s = time.perf_counter() - started
    print(f"memory_insert_rows={len(rows)} elapsed_s={insert_s:.3f}")

    started = time.perf_counter()
    listed = await store.list_paths("tenant-a/", limit=50)
    list_s = time.perf_counter() - started
    print(f"list_paths_limit=50 elapsed_s={list_s:.3f}")
    assert len(listed) == 50
    assert all(path.startswith("tenant-a/") for path in listed)
    assert all(not path.startswith("tenant-b/") for path in listed)

    started = time.perf_counter()
    result = await store.search("tenant-a/", "alpha", limit=10, max_files=100, max_chars=4_000, max_file_chars=200)
    search_s = time.perf_counter() - started
    print(f"search_max_files=100 scanned={result.scanned} elapsed_s={search_s:.3f}")
    assert result.scanned <= 100
    assert result.matches
    assert all(match.path.startswith("tenant-a/") for match in result.matches)
    assert all(not match.path.startswith("tenant-b/") for match in result.matches)

    contended = [f"c{i}.md" for i in range(4)]
    versions: dict[str, str] = {}
    for path in contended:
        mutation = await store.write(path, "init", expected_version=None)
        assert mutation.version is not None
        versions[path] = mutation.version

    async def attempt(path: str, i: int) -> object:
        try:
            return await store.write(path, f"w{i}", expected_version=versions[path])
        except MemoryConflictError:
            return None

    cas_started = time.perf_counter()
    outcomes = await asyncio.gather(*[attempt(contended[i % 4], i) for i in range(200)])
    cas_s = time.perf_counter() - cas_started
    winners = [item for item in outcomes if item is not None]
    print(f"cas_tasks=200 winners={len(winners)} elapsed_s={cas_s:.3f}")
    assert len(winners) == 4


async def test_catalog_query_similarity_and_allowlist(root: str) -> None:
    chunks = pxt.create_table(f"{root}.chunks", {"text": pxt.String, "status": pxt.String})
    secret = pxt.create_table(f"{root}.secret", {"text": pxt.String})
    secret.insert([{"text": "hidden"}])
    payload = [
        {"text": TARGET_TEXT if i == 0 else f"row {i} filler", "status": "open" if i % 3 == 0 else "closed"}
        for i in range(N_CHUNKS)
    ]
    started = time.perf_counter()
    for offset in range(0, len(payload), 5_000):
        chunks.insert(payload[offset : offset + 5_000])
    insert_s = time.perf_counter() - started
    print(f"catalog_insert_rows={N_CHUNKS} elapsed_s={insert_s:.3f}")

    started = time.perf_counter()
    chunks.add_embedding_index("text", string_embed=tiny_embed)
    index_s = time.perf_counter() - started
    print(f"embedding_index_rows={N_CHUNKS} elapsed_s={index_s:.3f}")

    tools = Pixeltable(tables=[f"{root}.chunks"], max_rows=20).get_toolset()
    started = time.perf_counter()
    queried = tools.query_table(f"{root}.chunks", where={"status": "open"}, limit=50)
    query_s = time.perf_counter() - started
    print(f"query_table_open elapsed_s={query_s:.3f} rows={len(queried['rows'])} truncated={queried['truncated']}")
    assert queried["truncated"]
    assert len(queried["rows"]) == 20
    assert all(row["status"] == "open" for row in queried["rows"])

    started = time.perf_counter()
    similar = tools.similarity_search(f"{root}.chunks", TARGET_TEXT, "text", limit=3)
    sim_s = time.perf_counter() - started
    print(f"similarity_search elapsed_s={sim_s:.3f} top={similar['rows'][0]['text']!r}")
    assert similar["rows"][0]["text"] == TARGET_TEXT

    with pytest.raises(ModelRetry, match="allowlist"):
        tools.query_table(f"{root}.secret")
