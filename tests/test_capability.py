"""Real-catalog tests for the Pixeltable RAG capability."""

from __future__ import annotations

import hashlib
import json
import uuid

import numpy as np
import pixeltable as pxt
import pytest
from pydantic_ai.exceptions import ModelRetry

from pydantic_ai_pixeltable import Pixeltable

DIM = 8


@pxt.udf
def tiny_embed(text: str) -> pxt.Array[(8,), pxt.Float]:
    digest = hashlib.sha256(text.encode()).digest()
    values = np.array([(digest[i % 32] / 127.5) - 1.0 for i in range(DIM)], dtype=np.float32)
    norm = float(np.linalg.norm(values))
    return values / norm if norm else values


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


def test_from_spec_and_id() -> None:
    cap = Pixeltable.from_spec(tables=["my_app.doc_chunks"], max_rows=3, max_chars=100)
    assert cap.id == "pixeltable"
    assert cap.tables == ["my_app.doc_chunks"]
    assert cap.max_rows == 3
    assert cap.get_instructions() is not None
    assert "my_app.doc_chunks" in str(cap.get_instructions())
    assert Pixeltable(guidance="").get_instructions() is None


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


def test_similarity_ranks_exact_text(catalog: str) -> None:
    t = pxt.get_table(f"{catalog}.chunks")
    t.add_embedding_index("text", string_embed=tiny_embed)
    tools = _tools(catalog)
    result = tools.similarity_search(f"{catalog}.chunks", "cats sit on mats", "text", limit=3)
    assert result["rows"][0]["text"] == "cats sit on mats"
    assert "score" in result["rows"][0]
    assert result["rows"][0]["score"] >= result["rows"][-1]["score"]
