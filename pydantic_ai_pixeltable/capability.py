"""Pixeltable catalog capability: list, describe, query, and similarity-search tables."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT

from pydantic_ai_pixeltable.toolset import PixeltableToolset

if TYPE_CHECKING:
    from pydantic_ai._instructions import AgentInstructions

_INSTRUCTIONS = (
    "You have Pixeltable catalog tools. Call list_tables and describe_table before querying "
    "an unfamiliar table. Use similarity_search for handbook or embedding questions. Use "
    "query_table for structured equality filters. Do not copy retrieved rows into Memory; "
    "Memory is the short notebook, not the corpus."
)


@dataclass
class Pixeltable(AbstractCapability[AgentDepsT]):
    """Read-only tools over existing Pixeltable tables.

    Pair this with Harness ``Memory`` when the agent also keeps a notebook. This
    capability searches application tables; ``PixeltableMemoryStore`` persists
    ``MEMORY.md``.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import Memory, ToolOutputLimits
    from pydantic_ai_pixeltable import Pixeltable, PixeltableMemoryStore

    agent = Agent(
        'openai:gpt-4o-mini',
        capabilities=[
            Memory(PixeltableMemoryStore(table_name='harness.memory')),
            Pixeltable(tables=['my_app.doc_chunks'], read_only=True),
            ToolOutputLimits(),
        ],
    )
    ```
    """

    tables: list[str] | None = None
    """Table paths or directory prefixes the tools may use. ``None`` is the whole catalog."""

    read_only: bool = True
    """Must be ``True``. Mutations are not implemented."""

    max_rows: int = 20
    """Hard cap on rows returned by ``query_table`` and ``similarity_search``."""

    max_chars: int = 8000
    """Hard cap on serialized JSON characters for those two tools."""

    guidance: str | None = None
    """System-prompt text. ``None`` uses the default; ``''`` adds none."""

    id: str | None = "pixeltable"
    """Capability id. Distinct from Harness ``Memory`` (``id='memory'``)."""

    def __post_init__(self) -> None:
        if not self.read_only:
            raise ValueError("Pixeltable(read_only=False) is not implemented; mutations are a follow-up")
        if self.max_rows < 1:
            raise ValueError(f"max_rows must be at least 1, got {self.max_rows}")
        if self.max_chars < 1:
            raise ValueError(f"max_chars must be at least 1, got {self.max_chars}")
        if self.tables is not None:
            cleaned = [entry for entry in self.tables if entry]
            if len(cleaned) != len(self.tables):
                raise ValueError("tables entries must be non-empty paths")
            self.tables = list(cleaned)

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        if self.guidance is not None:
            return self.guidance or None
        if self.tables:
            allowed = ", ".join(self.tables)
            return f"{_INSTRUCTIONS} You may only use these tables or prefixes: {allowed}."
        return _INSTRUCTIONS

    def get_toolset(self) -> PixeltableToolset[AgentDepsT]:
        return PixeltableToolset[AgentDepsT](
            tables=self.tables,
            max_rows=self.max_rows,
            max_chars=self.max_chars,
        )

    @classmethod
    def from_spec(
        cls,
        *,
        tables: Sequence[str] | None = None,
        read_only: bool = True,
        max_rows: int = 20,
        max_chars: int = 8000,
        guidance: str | None = None,
    ) -> Pixeltable[AgentDepsT]:
        return cls(
            tables=list(tables) if tables is not None else None,
            read_only=read_only,
            max_rows=max_rows,
            max_chars=max_chars,
            guidance=guidance,
        )
