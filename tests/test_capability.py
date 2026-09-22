"""Real-catalog tests for the Pixeltable RAG capability."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pixeltable as pxt
import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelRetry

from pydantic_ai_pixeltable import Pixeltable, PixeltableToolset
from tests.embed import DIM, tiny_embed


@pytest.fixture()
def catalog():
    root = f"test_pydantic_ai_cap_{uuid.uuid4().hex[:8]}"
    pxt.create_dir(root)
    chunks = pxt.create_table(
        f"{root}.chunks",
        {"text": pxt.String, "pos": pxt.Int, "status": pxt.String, "vec": pxt.Array[(8,), pxt.Float]},
    )
    other = pxt.create_table(f"{root}.other", {"text": pxt.String})
    zero = np.zeros(DIM, dtype=np.float32)
    chunks.insert(
        [
            {"text": "cats sit on mats", "pos": 0, "status": "open", "vec": zero},
            {"text": "dogs run in parks", "pos": 1, "status": "closed", "vec": zero},
            {"text": "birds fly in sky", "pos": 2, "status": "open", "vec": zero},
        ]
    )
    other.insert([{"text": "secret handbook"}])
    yield root
    try:
        pxt.drop_dir(root, force=True)
    except Exception:
        pass


def _tools(root: str, *, max_rows: int = 20, max_chars: int = 8000):
    return Pixeltable(tables=[f"{root}.chunks"], max_rows=max_rows, max_chars=max_chars).get_toolset()


def test_read_only_false_rejected() -> None:
    with pytest.raises(ValueError, match="read_only"):
        Pixeltable(read_only=False)


def test_toolset_requires_explicit_allowlist() -> None:
    # None must not silently mean the whole catalog; that is an opt-in via ['*'].
    for tables in (None, []):
        with pytest.raises(ValueError, match="allowlist"):
            PixeltableToolset(tables=tables, max_rows=10, max_chars=100)
    assert PixeltableToolset(tables=["*"], max_rows=10, max_chars=100) is not None


def test_tables_required_and_star(catalog: str) -> None:
    with pytest.raises(ValueError, match="tables"):
        Pixeltable()
    with pytest.raises(ValueError, match="only entry"):
        Pixeltable(tables=["*", f"{catalog}.chunks"])
    names = Pixeltable(tables=["*"]).get_toolset().list_tables()["tables"]
    assert f"{catalog}.chunks" in names
    assert f"{catalog}.other" in names


def test_tables_rejects_a_string() -> None:
    with pytest.raises(ValueError, match="list of paths"):
        Pixeltable(tables="my_app.doc_chunks")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="list of paths"):
        Pixeltable.from_spec(tables="my_app.doc_chunks")  # type: ignore[arg-type]
    spec = {"model": "test", "capabilities": [{"Pixeltable": {"tables": "my_app.doc_chunks"}}]}
    with pytest.raises(ValueError, match="list of paths"):
        Agent.from_spec(spec, custom_capability_types=[Pixeltable])


def test_from_spec_and_id() -> None:
    cap = Pixeltable.from_spec(tables=["my_app.doc_chunks"], max_rows=3, max_chars=100, defer_loading=True)
    assert cap.id == "pixeltable"
    assert cap.tables == ["my_app.doc_chunks"]
    assert cap.max_rows == 3
    assert cap.defer_loading is True
    assert cap.get_instructions() is not None
    assert "my_app.doc_chunks" in str(cap.get_instructions())
    assert Pixeltable(tables=["my_app.doc_chunks"], guidance="").get_instructions() is None


def test_allowlist_exact_and_prefix(catalog: str) -> None:
    exact = Pixeltable(tables=[f"{catalog}.chunks"]).get_toolset()
    assert exact.list_tables()["tables"] == [f"{catalog}.chunks"]

    prefix = Pixeltable(tables=[catalog]).get_toolset()
    assert set(prefix.list_tables()["tables"]) == {f"{catalog}.chunks", f"{catalog}.other"}

    with pytest.raises(ModelRetry, match="allowlist"):
        exact.query_table(f"{catalog}.other")
    with pytest.raises(ModelRetry, match="allowlist"):
        exact.similarity_search(f"{catalog}.other", "secret", "text")


def test_query_equality_where_and_default_columns(catalog: str) -> None:
    tools = _tools(catalog)
    described = tools.describe_table(f"{catalog}.chunks")
    assert described["kind"] == "table"
    names = {col["name"] for col in described["columns"]}
    assert names == {"text", "pos", "status", "vec"}
    for idx in described["indexes"]:
        assert {"name", "index_type", "columns"} <= set(idx)

    result = tools.query_table(f"{catalog}.chunks", where={"status": "open"})
    assert result["table"] == f"{catalog}.chunks"
    assert not result["truncated"]
    assert {row["text"] for row in result["rows"]} == {"cats sit on mats", "birds fly in sky"}
    for row in result["rows"]:
        assert "vec" not in row
        assert set(row) <= {"text", "pos", "status"}


def test_query_row_and_char_bounds(catalog: str) -> None:
    rows = _tools(catalog, max_rows=2).query_table(f"{catalog}.chunks", limit=10)
    assert len(rows["rows"]) == 2
    assert rows["truncated"]

    tiny = _tools(catalog, max_chars=100).query_table(f"{catalog}.chunks", columns=["text"])
    assert tiny["truncated"]
    assert len(json.dumps(tiny, ensure_ascii=False)) <= 100

    # A cap below the minimal envelope size cannot shrink the payload further.
    micro = _tools(catalog, max_chars=1).query_table(f"{catalog}.chunks", columns=["text"])
    assert micro["rows"] == []
    assert micro["truncated"]


def test_similarity_ranks_exact_text(catalog: str) -> None:
    t = pxt.get_table(f"{catalog}.chunks")
    t.add_embedding_index("text", string_embed=tiny_embed)
    tools = _tools(catalog)
    result = tools.similarity_search(f"{catalog}.chunks", "cats sit on mats", "text", limit=3)
    assert result["rows"][0]["text"] == "cats sit on mats"
    assert "score" in result["rows"][0]
    assert result["rows"][0]["score"] >= result["rows"][-1]["score"]


def test_named_media_column_returns_file_url(catalog: str, tmp_path: Path) -> None:
    docs = pxt.create_table(f"{catalog}.docs", {"doc": pxt.Document, "title": pxt.String})
    note = tmp_path / "note.md"
    note.write_text("# note\n")
    docs.insert([{"doc": str(note), "title": "note"}])
    tools = Pixeltable(tables=[f"{catalog}.docs"]).get_toolset()

    default = tools.query_table(f"{catalog}.docs")
    assert default["rows"]
    assert "doc" not in default["rows"][0]
    assert "title" in default["rows"][0]

    named = tools.query_table(f"{catalog}.docs", columns=["doc", "title"])
    url = named["rows"][0]["doc"]
    assert isinstance(url, str)
    assert url.startswith("file:")


def test_agent_from_spec_requires_custom_capability_types() -> None:
    spec = {"model": "test", "capabilities": [{"Pixeltable": {"tables": ["my_app.doc_chunks"]}}]}
    with pytest.raises(ValueError, match="custom_capability_types"):
        Agent.from_spec(spec)
    agent = Agent.from_spec(spec, custom_capability_types=[Pixeltable])
    loaded = [cap for cap in agent.root_capability.capabilities if isinstance(cap, Pixeltable)]
    assert loaded[0].tables == ["my_app.doc_chunks"]


def test_combine_unions_allowlists_and_tightens_caps() -> None:
    merged = Pixeltable.combine([Pixeltable(tables=["a"], max_rows=50), Pixeltable(tables=["b"], max_rows=5)])
    assert merged.tables == ["a", "b"]
    assert merged.max_rows == 5

    star = Pixeltable.combine([Pixeltable(tables=["a"]), Pixeltable(tables=["*"])])
    assert star.tables == ["*"]

    agent = Agent(
        "test",
        capabilities=[Pixeltable(tables=["x"]), Pixeltable(tables=["y"], max_chars=10)],
    )
    caps = [cap for cap in agent.root_capability.capabilities if isinstance(cap, Pixeltable)]
    assert len(caps) == 1
    assert caps[0].tables == ["x", "y"]
    assert caps[0].max_chars == 10


def test_tables_rejects_bad_entries() -> None:
    for bad in (["my_app."], [" "], ["a..b"], [".hidden"]):
        with pytest.raises(ValueError, match="tables entry"):
            Pixeltable(tables=bad)


def test_views_work_as_catalog_targets(catalog: str) -> None:
    chunks = pxt.get_table(f"{catalog}.chunks")
    pxt.create_view(f"{catalog}.open_chunks", chunks.where(chunks.status == "open"))
    tools = Pixeltable(tables=[f"{catalog}.open_chunks"]).get_toolset()

    described = tools.describe_table(f"{catalog}.open_chunks")
    assert described["kind"] == "view"
    assert f"{catalog}.open_chunks" in tools.list_tables()["tables"]
    result = tools.query_table(f"{catalog}.open_chunks")
    assert {row["status"] for row in result["rows"]} == {"open"}


def test_similarity_idx_disambiguation(catalog: str) -> None:
    t = pxt.get_table(f"{catalog}.chunks")
    t.add_embedding_index("text", string_embed=tiny_embed, idx_name="e_a")
    t.add_embedding_index("text", string_embed=tiny_embed, idx_name="e_b")
    tools = _tools(catalog)

    with pytest.raises(ModelRetry):
        tools.similarity_search(f"{catalog}.chunks", "cats sit on mats", "text")
    result = tools.similarity_search(f"{catalog}.chunks", "cats sit on mats", "text", idx="e_a")
    assert result["rows"][0]["text"] == "cats sit on mats"


def test_tool_error_branches(catalog: str) -> None:
    tools = _tools(catalog)
    with pytest.raises(ModelRetry, match="Unknown column"):
        tools.query_table(f"{catalog}.chunks", columns=["nope"])
    with pytest.raises(ModelRetry, match="Unknown column"):
        tools.query_table(f"{catalog}.chunks", where={"nope": 1})
    with pytest.raises(ModelRetry, match="Unknown column"):
        tools.similarity_search(f"{catalog}.chunks", "x", "nope")
    with pytest.raises(ModelRetry, match="non-empty"):
        tools.similarity_search(f"{catalog}.chunks", "  ", "text")
    with pytest.raises(ModelRetry, match="at least one column"):
        tools.query_table(f"{catalog}.chunks", columns=[])
    with pytest.raises(ModelRetry, match="at least 1"):
        tools.query_table(f"{catalog}.chunks", limit=0)
    with pytest.raises(ModelRetry, match="scalar"):
        tools.query_table(f"{catalog}.chunks", where={"status": ["open", "closed"]})
    with pytest.raises(ModelRetry, match="scalar"):
        tools.query_table(f"{catalog}.chunks", where={"status": {"nested": 1}})
    with pytest.raises(ModelRetry, match="equality filters"):
        tools.query_table(f"{catalog}.chunks", where={"vec": 5})

    # Media columns never match an equality filter; None still selects null rows.
    docs = pxt.create_table(f"{catalog}.docs", {"title": pxt.String, "doc": pxt.Document | None})
    docs.insert([{"title": "a", "doc": None}])
    doc_tools = Pixeltable(tables=[f"{catalog}.docs"]).get_toolset()
    with pytest.raises(ModelRetry, match="equality filters"):
        doc_tools.query_table(f"{catalog}.docs", where={"doc": "x"})
    assert [row["title"] for row in doc_tools.query_table(f"{catalog}.docs", where={"doc": None})["rows"]] == ["a"]

    prefix_tools = Pixeltable(tables=[catalog]).get_toolset()
    with pytest.raises(ModelRetry, match="Cannot open"):
        prefix_tools.query_table(f"{catalog}.missing")


def test_projection_errors_become_retries(catalog: str, monkeypatch: pytest.MonkeyPatch) -> None:
    # Expression construction in _project must not escape the tool as a hard error.
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise pxt.NotFoundError(pxt.ErrorCode.PATH_NOT_FOUND, "synthetic select failure")

    monkeypatch.setattr(pxt.Table, "select", boom)
    with pytest.raises(ModelRetry, match="synthetic"):
        _tools(catalog).query_table(f"{catalog}.chunks")
