"""EntityProfile: a named bundle of canonical fields, blocking rules and constraints.

A profile is pure declaration. It never touches data; the engine reads it to build
field-type instances, Splink settings, the deterministic pre-pass and the
post-clustering constraints.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any, Literal, Mapping

from ecg_linkage.fields.base import ComparisonOptions, FieldType
from ecg_linkage.fields.registry import field_type_class, get_field_type

FieldRole = Literal["identifier", "primary", "supporting", "report_only"]
"""
* identifier  - validated unique key; used in the deterministic pre-pass and hard constraints
* primary     - core evidence in the probabilistic model
* supporting  - weaker evidence, still in the model
* report_only - shown in review output, excluded from the model (e.g. correlated signals)
"""


@dataclass(frozen=True)
class FieldSpec:
    """How one canonical field of a profile is typed and used."""

    field_type: str
    role: FieldRole = "primary"
    required: bool = False
    term_frequency: bool = False
    include_phonetic: bool = True
    jaro_winkler_thresholds: tuple[float, ...] = (0.92, 0.85)
    description: str = ""

    def comparison_options(self, **overrides: Any) -> ComparisonOptions:
        opts = ComparisonOptions(
            term_frequency=self.term_frequency,
            include_phonetic=self.include_phonetic,
            jaro_winkler_thresholds=self.jaro_winkler_thresholds,
        )
        for k, v in overrides.items():
            setattr(opts, k, v)
        return opts


@dataclass(frozen=True)
class BlockingRule:
    """A blocking rule expressed dialect-agnostically.

    ``columns`` -> Splink ``block_on(*columns)``. ``sql`` is an escape hatch for rules
    ``block_on`` cannot express (e.g. swapped names); keep it to plain equality so it
    runs on both DuckDB and Spark.
    """

    columns: tuple[str, ...] = ()
    sql: str | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if bool(self.columns) == bool(self.sql):
            raise ValueError("BlockingRule needs exactly one of 'columns' or 'sql'")


@dataclass(frozen=True)
class Guard:
    """A non-contradiction check attached to a deterministic rule.

    The pair passes if, for ``column``, either side is null, the values are equal,
    the Jaro-Winkler similarity is >= ``jw_threshold``, or (when ``swap_with`` is set)
    the value equals the other side's ``swap_with`` column.
    """

    column: str
    jw_threshold: float | None = 0.85
    swap_with: str | None = None


@dataclass(frozen=True)
class DeterministicRule:
    """Pairs whose ``match_columns`` are equal (and non-null) and pass all guards link directly."""

    match_columns: tuple[str, ...]
    guards: tuple[Guard, ...] = ()
    description: str = ""


@dataclass(frozen=True)
class HardConstraint:
    """Two records with different non-null values in ``column`` must never share a cluster."""

    column: str
    description: str = ""


@dataclass
class EntityProfile:
    """Declarative description of an entity kind (provider, organization, ...)."""

    name: str
    description: str
    fields: dict[str, FieldSpec]
    blocking_rules: list[BlockingRule] = field(default_factory=list)
    training_blocking_rules: list[BlockingRule] = field(default_factory=list)
    deterministic_rules: list[DeterministicRule] = field(default_factory=list)
    hard_constraints: list[HardConstraint] = field(default_factory=list)
    max_cluster_size: int = 6
    min_edge_density: float = 0.5

    # ----------------------------------------------------------------- columns
    def canonical_columns(self) -> list[str]:
        """Every column a mapped source may populate (whole values and components)."""
        cols: list[str] = []
        for fname, spec in self.fields.items():
            cls = field_type_class(spec.field_type)
            cols.append(fname)
            cols.extend(f"{fname}_{c}" for c in cls.components)
        return cols

    def mapping_targets(self) -> dict[str, str]:
        """``{target column: human description}`` used to brief the LLM schema mapper."""
        out: dict[str, str] = {}
        for fname, spec in self.fields.items():
            cls = field_type_class(spec.field_type)
            out[fname] = f"{spec.description or fname} (whole value; type {spec.field_type})"
            for c in cls.components:
                out[f"{fname}_{c}"] = f"{spec.description or fname}: {c} component only"
        return out

    def model_fields(self) -> dict[str, FieldSpec]:
        """Fields that enter the probabilistic model."""
        return {n: s for n, s in self.fields.items() if s.role != "report_only"}

    # ------------------------------------------------------------ field types
    def instantiate_fields(self, dependencies: Mapping[str, Any] | None = None) -> dict[str, FieldType]:
        """Build a FieldType instance per canonical field, injecting matching dependencies.

        ``dependencies`` is a flat mapping (e.g. ``{"aliases": {...}, "zip_centroids": {...},
        "taxonomy": {...}}``); each FieldType constructor receives only the keys it declares.
        """
        deps = dict(dependencies or {})
        out: dict[str, FieldType] = {}
        for fname, spec in self.fields.items():
            cls = field_type_class(spec.field_type)
            params = inspect.signature(cls.__init__).parameters
            kwargs = {k: v for k, v in deps.items() if k in params}
            out[fname] = get_field_type(spec.field_type, **kwargs)
        return out

    def comparison_schema(self) -> dict[str, Any]:
        """A serializable description of comparison definitions (for the KB schema hash)."""
        return {
            "profile": self.name,
            "fields": {
                n: {
                    "type": s.field_type, "role": s.role, "tf": s.term_frequency,
                    "phonetic": s.include_phonetic, "jw": list(s.jaro_winkler_thresholds),
                }
                for n, s in self.model_fields().items()
            },
        }


# ---------------------------------------------------------------- registry
_PROFILES: dict[str, EntityProfile] = {}
_EP_LOADED = False


def register_profile(profile: EntityProfile) -> EntityProfile:
    """Register a profile under ``profile.name``."""
    if profile.name in _PROFILES and _PROFILES[profile.name] is not profile:
        raise ValueError(f"profile '{profile.name}' already registered")
    _PROFILES[profile.name] = profile
    return profile


def _load_entry_points() -> None:
    global _EP_LOADED
    if _EP_LOADED:
        return
    _EP_LOADED = True
    for ep in entry_points(group="ecg_linkage.profiles"):
        obj = ep.load()
        if isinstance(obj, EntityProfile) and obj.name not in _PROFILES:
            register_profile(obj)


def get_profile(name: str) -> EntityProfile:
    """Look up a registered profile by name."""
    _load_entry_points()
    try:
        return _PROFILES[name]
    except KeyError as exc:
        raise KeyError(f"unknown profile '{name}'. Known: {sorted(_PROFILES)}") from exc


def list_profiles() -> dict[str, str]:
    """``{name: description}`` of registered profiles."""
    _load_entry_points()
    return {n: p.description for n, p in sorted(_PROFILES.items())}
