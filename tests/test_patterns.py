"""Official Pydantic AI / Harness patterns this package claims to host."""

from __future__ import annotations

import uuid

import pixeltable as pxt
import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness import Memory
from pydantic_ai_harness.memory import MemoryConflictError, MemoryOperation

from pydantic_ai_pixeltable import Pixeltable, PixeltableMemoryStore
from tests.embed import tiny_embed

HANDBOOK = "The release pipeline publishes wheels from a git clone until a GitHub Release exists."
NOTEBOOK = "zzz qqq nnn rrrr"
PARAPHRASE = "How do we publish this package before PyPI?"


@pytest.fixture()
def root():
    name = f"test_pydantic_ai_pat_{uuid.uuid4().hex[:8]}"
    pxt.create_dir(name)
    yield name
    try:
        pxt.drop_dir(name, force=True)
    except Exception:
        pass


async def test_memory_notebook_write_and_inject(root: str) -> None:
    store = PixeltableMemoryStore(table_name=f"{root}.memory")
    calls = 0

    def fake(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        injected = ""
        for message in messages:
            for part in getattr(message, "parts", ()):
                content = getattr(part, "content", None)
                items = content if isinstance(content, list) else []
                for item in items:
                    text = getattr(item, "content", "")
                    if isinstance(text, str) and "<memory>" in text:
                        injected = text
        if "MEMORY.md" in injected and "uv" in injected:
            return ModelResponse(parts=[TextPart("injected-ok")])
        if calls == 1:
            return ModelResponse(
                parts=[ToolCallPart("write_memory", {"content": "- user prefers uv"}, tool_call_id="w1")]
            )
        return ModelResponse(parts=[TextPart("saved")])

    agent = Agent(FunctionModel(fake), capabilities=[Memory(store)])
    await agent.run("Remember that I prefer uv.")
    file = await store.read("main/MEMORY.md", max_chars=1_000)
    assert file is not None
    assert "uv" in file.content
    second = await agent.run("What do I prefer?")
    assert second.output == "injected-ok"


async def test_retrieve_is_catalog_not_memory_notebook(root: str) -> None:
    store = PixeltableMemoryStore(table_name=f"{root}.memory")
    await store.write("main/MEMORY.md", NOTEBOOK, expected_version=None)
    chunks = pxt.create_table(f"{root}.chunks", {"text": pxt.String})
    chunks.insert([{"text": HANDBOOK}])
    chunks.add_embedding_index("text", string_embed=tiny_embed)
    tools = Pixeltable(tables=[f"{root}.chunks"]).get_toolset()

    hit = tools.similarity_search(f"{root}.chunks", HANDBOOK, "text", limit=1)
    assert hit["rows"][0]["text"] == HANDBOOK
    assert hit["rows"][0]["score"] == pytest.approx(1.0)

    lexical = await store.search("", PARAPHRASE, limit=10, max_files=10, max_chars=400, max_file_chars=1_000)
    assert lexical.matches == []


def test_equality_lookup_bank_support_shape(root: str) -> None:
    tickets = pxt.create_table(f"{root}.tickets", {"status": pxt.String, "note": pxt.String})
    tickets.insert(
        [
            {"status": "open", "note": "reset password"},
            {"status": "closed", "note": "old ticket"},
            {"status": "open", "note": "wire delay"},
        ]
    )
    tools = Pixeltable(tables=[f"{root}.tickets"]).get_toolset()
    result = tools.query_table(f"{root}.tickets", where={"status": "open"})
    assert {row["note"] for row in result["rows"]} == {"reset password", "wire delay"}


async def test_receipt_gap_replay_conflicts(root: str) -> None:
    store = PixeltableMemoryStore(table_name=f"{root}.memory")
    operation = MemoryOperation(id="run-1:call-1", fingerprint="write:notes/main.md:one")
    first = await store.write("notes/main.md", "one", expected_version=None, operation=operation)
    assert not first.replayed
    t = store.table
    t.delete(where=t.path == f"__op__/{operation.id}")
    with pytest.raises(MemoryConflictError):
        await store.write("notes/main.md", "one", expected_version=None, operation=operation)


def test_yaml_memory_backend_cannot_be_pixeltable() -> None:
    with pytest.raises(ValueError, match="backend"):
        Memory.from_spec(backend="pixeltable")  # type: ignore[arg-type]
