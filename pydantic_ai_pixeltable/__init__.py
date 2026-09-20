"""Pydantic AI Harness MemoryStore and Pixeltable catalog tools."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from pydantic_ai_pixeltable.capability import Pixeltable
from pydantic_ai_pixeltable.store import PixeltableMemoryStore
from pydantic_ai_pixeltable.toolset import PixeltableToolset

__all__ = ["Pixeltable", "PixeltableMemoryStore", "PixeltableToolset"]

try:
    __version__ = _pkg_version("pydantic-ai-pixeltable")
except PackageNotFoundError:  # source checkout without install
    __version__ = "0.0.0"
