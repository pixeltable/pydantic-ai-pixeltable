"""Pixeltable catalog capability: list, describe, query, and similarity-search tables."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT

from pydantic_ai_pixeltable.toolset import ALL_TABLES, PixeltableToolset

if TYPE_CHECKING:
    from pydantic_ai.agent.abstract import AgentInstructions

_INSTRUCTIONS = (
    "You have Pixeltable catalog tools. Call list_tables and describe_table before querying "
    "an unfamiliar table. Use similarity_search for handbook or embedding questions. Use "
    "query_table for structured equality filters."
)


@dataclass
class Pixeltable(AbstractCapability[AgentDepsT]):
    """Read-only tools over existing Pixeltable tables.

    Pair this with Harness ``Memory`` (any store) when the agent also keeps a notebook.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_pixeltable import Pixeltable

    agent = Agent(
        "openai:gpt-4o-mini",
        capabilities=[Pixeltable(tables=["my_app.doc_chunks"], read_only=True)],
    )
    ```
    """

    tables: list[str] | None = None
    """Allowlist of table paths or directory prefixes. Required. ``['*']`` is the whole catalog."""

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
        if isinstance(self.tables, str):
            raise ValueError("tables must be a list of paths, not a string")
        if self.tables is None:
            raise ValueError(
                "Pixeltable requires tables=... (an allowlist). Pass tables=['*'] to allow the whole catalog."
            )
        cleaned = [entry for entry in self.tables if entry]
        if not cleaned:
            raise ValueError("tables must be a non-empty allowlist, or ['*'] for the whole catalog")
        if ALL_TABLES in cleaned and cleaned != [ALL_TABLES]:
            raise ValueError("tables=['*'] must be the only entry when allowing the whole catalog")
        for entry in cleaned:
            if entry != ALL_TABLES and any(not part or any(ch.isspace() for ch in part) for part in entry.split(".")):
                raise ValueError(f"invalid tables entry {entry!r}: expected dotted paths like 'my_app.doc_chunks'")
        self.tables = list(cleaned)
        if self.description is None:
            self.description = (
                "Read-only Pixeltable catalog tools: list_tables, describe_table, query_table, similarity_search."
            )

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        if self.guidance is not None:
            return self.guidance or None
        if self.tables and self.tables != [ALL_TABLES]:
            allowed = ", ".join(self.tables)
            return f"{_INSTRUCTIONS} You may only use these tables or prefixes: {allowed}."
        return _INSTRUCTIONS

    @classmethod
    def combine(cls, capabilities: Sequence[Pixeltable[AgentDepsT]]) -> Pixeltable[AgentDepsT]:
        """Merge same-id instances: union of allowlists, tightest row/char caps.

        The default field merge would union ``['*']`` with other entries into a list that the
        allowlist check then narrows to just those entries, and would take the later (looser)
        cap rather than the smaller one.
        """
        tables: list[str] = []
        for capability in capabilities:
            tables.extend(entry for entry in capability.tables or [] if entry not in tables)
        latest = capabilities[-1]
        return cls(
            tables=[ALL_TABLES] if ALL_TABLES in tables else tables,
            read_only=all(capability.read_only for capability in capabilities),
            max_rows=min(capability.max_rows for capability in capabilities),
            max_chars=min(capability.max_chars for capability in capabilities),
            guidance=next((c.guidance for c in reversed(capabilities) if c.guidance is not None), None),
            id=latest.id,
            description=latest.description,
            defer_loading=latest.defer_loading,
        )

    def get_toolset(self) -> PixeltableToolset[AgentDepsT]:
        return PixeltableToolset[AgentDepsT](
            tables=self.tables,
            max_rows=self.max_rows,
            max_chars=self.max_chars,
            id=self.id or "pixeltable",
        )

    @classmethod
    def from_spec(
        cls,
        *,
        tables: Sequence[str],
        read_only: bool = True,
        max_rows: int = 20,
        max_chars: int = 8000,
        guidance: str | None = None,
        id: str | None = "pixeltable",
        description: str | None = None,
        defer_loading: bool = False,
    ) -> Pixeltable[AgentDepsT]:
        """Build from a YAML/dict spec; register with ``Agent.from_spec(custom_capability_types=[Pixeltable])``."""
        if isinstance(tables, str):
            raise ValueError("tables must be a list of paths, not a string")
        return cls(
            tables=list(tables),
            read_only=read_only,
            max_rows=max_rows,
            max_chars=max_chars,
            guidance=guidance,
            id=id,
            description=description,
            defer_loading=defer_loading,
        )
