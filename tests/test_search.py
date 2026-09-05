"""SearchableMemoryStore tests for PixeltableMemoryStore."""

from __future__ import annotations

import uuid

import pixeltable as pxt
import pytest

from pydantic_ai_pixeltable import PixeltableMemoryStore


@pytest.fixture()
def store():
    table_name = f"test_pydantic_ai.search_{uuid.uuid4().hex[:8]}"
    memory = PixeltableMemoryStore(table_name=table_name)
    yield memory
    try:
        pxt.drop_table(table_name, force=True)
    except Exception:
        pass
    try:
        pxt.drop_dir("test_pydantic_ai", force=True)
    except Exception:
        pass


async def test_scoped_bounded_search(store: PixeltableMemoryStore) -> None:
    for path, content in (
        ("tenant-a/main/alpha.md", "alpha alpha"),
        ("tenant-a/main/beta.md", "alpha"),
        ("tenant-a/main/other.md", "unrelated"),
        ("tenant-b/main/private.md", "alpha alpha alpha"),
    ):
        await store.write(path, content, expected_version=None)

    result = await store.search("tenant-a/main/", "alpha", limit=10, max_files=10, max_chars=80, max_file_chars=1_000)
    assert [match.path for match in result.matches] == [
        "tenant-a/main/alpha.md",
        "tenant-a/main/beta.md",
    ]
    assert result.scanned == 3
    assert not result.truncated
    assert sum(len(match.path) + len(match.snippet) for match in result.matches) <= 80

    bounded = await store.search("tenant-a/main/", "alpha", limit=10, max_files=1, max_chars=80, max_file_chars=1_000)
    assert bounded.scanned == 1
    assert bounded.truncated
    tiny = await store.search("tenant-a/main/", "alpha", limit=10, max_files=10, max_chars=1, max_file_chars=1_000)
    assert tiny.matches == []
    assert tiny.truncated


async def test_search_empty_and_invalid_bounds(store: PixeltableMemoryStore) -> None:
    await store.write("a.md", "alpha", expected_version=None)
    for query, limit, max_files, max_chars in (
        ("", 10, 10, 80),
        ("alpha", 0, 10, 80),
        ("alpha", 10, 0, 80),
        ("alpha", 10, 10, 0),
    ):
        empty = await store.search(
            "", query, limit=limit, max_files=max_files, max_chars=max_chars, max_file_chars=1_000
        )
        assert empty.matches == []
        assert empty.scanned == 0
        assert not empty.truncated


async def test_search_snippets_cover_tiny_and_offset_windows(store: PixeltableMemoryStore) -> None:
    await store.write("a", "012345alpha-tail", expected_version=None)

    tiny = await store.search("", "alpha", limit=1, max_files=1, max_chars=3, max_file_chars=1_000)
    assert len(tiny.matches) == 1
    assert len(tiny.matches[0].snippet) == 2
    offset = await store.search("", "alpha", limit=1, max_files=1, max_chars=12, max_file_chars=1_000)
    assert len(offset.matches) == 1
    assert offset.matches[0].snippet.startswith("...")


async def test_list_paths_and_search_bound_prefix_in_table(store: PixeltableMemoryStore) -> None:
    for path, content in (
        ("tenant-a/main/a.md", "alpha"),
        ("tenant-a/main/b.md", "alpha"),
        ("tenant-a/main/c.md", "alpha"),
        ("tenant-b/main/private.md", "alpha alpha alpha"),
    ):
        await store.write(path, content, expected_version=None)

    assert await store.list_paths("tenant-a/main/", limit=2) == ["tenant-a/main/a.md", "tenant-a/main/b.md"]
    result = await store.search("tenant-a/main/", "alpha", limit=10, max_files=2, max_chars=200, max_file_chars=1_000)
    assert [match.path for match in result.matches] == ["tenant-a/main/a.md", "tenant-a/main/b.md"]
    assert result.scanned == 2
    assert result.truncated
    outsider = await store.search(
        "tenant-b/main/", "alpha", limit=10, max_files=10, max_chars=200, max_file_chars=1_000
    )
    assert [match.path for match in outsider.matches] == ["tenant-b/main/private.md"]


async def test_search_bounds_each_file_and_ignores_namespace_prefix(store: PixeltableMemoryStore) -> None:
    namespace = "n" * 180
    prefix = f"{namespace}/main/"
    await store.write(f"{prefix}note.md", "prefix TARGET", expected_version=None)

    bounded = await store.search(prefix, "target", limit=10, max_files=10, max_chars=100, max_file_chars=6)
    assert bounded.matches == []
    namespace_result = await store.search(prefix, namespace, limit=10, max_files=10, max_chars=100, max_file_chars=100)
    assert namespace_result.matches == []
    visible = await store.search(prefix, "target", limit=10, max_files=10, max_chars=20, max_file_chars=100)
    assert [match.path for match in visible.matches] == [f"{prefix}note.md"]
