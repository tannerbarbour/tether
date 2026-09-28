"""Registry of FieldType classes.

Built-in field types register themselves with ``@register_field_type`` at import
time. External packages contribute via the ``ecg_linkage.field_types`` entry-point
group, so new kinds of fields never require touching engine code.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Callable, TypeVar

from ecg_linkage.fields.base import FieldType

_REGISTRY: dict[str, type[FieldType]] = {}
_ENTRY_POINTS_LOADED = False

F = TypeVar("F", bound=type[FieldType])


def register_field_type(cls: F) -> F:
    """Class decorator: register ``cls`` under ``cls.name``."""
    name = getattr(cls, "name", None)
    if not name:
        raise ValueError(f"{cls.__name__} must define a non-empty class attribute 'name'")
    if name in _REGISTRY and _REGISTRY[name] is not cls:
        raise ValueError(f"field type '{name}' already registered by {_REGISTRY[name].__name__}")
    _REGISTRY[name] = cls
    return cls


def _load_entry_points() -> None:
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED:
        return
    _ENTRY_POINTS_LOADED = True
    for ep in entry_points(group="ecg_linkage.field_types"):
        cls = ep.load()
        if ep.name not in _REGISTRY:
            register_field_type(cls)


def get_field_type(name: str, **kwargs) -> FieldType:
    """Instantiate the FieldType registered as ``name``."""
    _load_entry_points()
    try:
        cls = _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown field type '{name}'. Known: {sorted(_REGISTRY)}") from exc
    return cls(**kwargs)


def field_type_class(name: str) -> type[FieldType]:
    """Return the registered class without instantiating it."""
    _load_entry_points()
    return _REGISTRY[name]


def list_field_types() -> dict[str, str]:
    """Return ``{name: description}`` for every registered field type."""
    _load_entry_points()
    return {n: c.description for n, c in sorted(_REGISTRY.items())}


def field_type_factory(name: str) -> Callable[..., FieldType]:
    """Return a callable that instantiates ``name`` (useful for profiles)."""
    return lambda **kw: get_field_type(name, **kw)
