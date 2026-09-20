"""Parsing of the ``type_`` strings in ``Table.get_metadata()``.

0.6.x renders ``type_`` schema-style: ``'Required[T]'`` when non-nullable, bare ``'T'``
when nullable. 0.7.x renders the repr: ``'T'`` when non-nullable, ``'T | None'`` when
nullable.
"""

from typing import Any


def column_base(type_: Any) -> str:
    """Base type name, e.g. ``'Array'`` for ``'Array[(8,), Float] | None'`` or ``'Required[Array[(8,), Float]]'``."""
    base = str(type_).split(" | ", 1)[0]
    if base.startswith("Required[") and base.endswith("]"):
        base = base[len("Required[") : -1]
    return base.split("[", 1)[0]


def is_nullable_type(type_: Any, legacy_schema: bool) -> bool:
    """Whether the rendered ``type_`` is nullable; a bare ``'T'`` counts only under ``legacy_schema``."""
    rendered = str(type_)
    if rendered.startswith("Required["):
        return False
    if " | None" in rendered:
        return True
    return legacy_schema
