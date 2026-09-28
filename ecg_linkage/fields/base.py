"""FieldType: the core semantic-field abstraction.

A FieldType bundles three things for one *kind* of field (person name, NPI, ...):

1. ``validate``    - is this value well-formed?  Invalid values are flagged, never dropped.
2. ``standardize`` - derived columns (parsed parts, cleaned forms, phonetic codes).
3. ``comparisons`` - Splink comparison(s) whose *levels* grade raw -> cleaned -> fuzzy
                     -> phonetic within ONE comparison, so evidence is never double counted.

Naming convention for derived columns: ``<field>_<suffix>`` where ``<field>`` is the
canonical field name the profile assigns (e.g. ``name``) and ``<suffix>`` is one of the
FieldType's ``outputs``. Every FieldType also emits ``<field>_valid`` (bool) and
``<field>_invalid_reason`` (str or None).

Canonical inputs follow the same convention: the whole value lives in ``<field>`` and
optional pre-split components in ``<field>_<component>`` (e.g. ``name_first``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Mapping

import pandas as pd

from ecg_linkage.fields._text import to_text

if TYPE_CHECKING:  # Splink is imported lazily so field modules stay cheap to import.
    from splink.internals.comparison_creator import ComparisonCreator

Parts = Mapping[str, Any]
"""Input to a FieldType: ``{"value": <whole>, "<component>": ..., ...}``."""

VALID_SUFFIX = "valid"
INVALID_REASON_SUFFIX = "invalid_reason"


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of validating one value."""

    is_valid: bool
    reason: str | None = None

    @classmethod
    def ok(cls) -> "ValidationResult":
        return cls(True, None)

    @classmethod
    def fail(cls, reason: str) -> "ValidationResult":
        return cls(False, reason)


@dataclass
class ComparisonOptions:
    """Per-field knobs a profile or engagement config can set on a comparison."""

    term_frequency: bool = False
    jaro_winkler_thresholds: tuple[float, ...] = (0.92, 0.85)
    include_phonetic: bool = True
    geo_available: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class FieldType(ABC):
    """Abstract semantic field kind. Subclasses are registered by ``name``.

    Subclasses must be stateless with respect to the data (dependencies such as
    lookup tables are injected at construction and are read-only).
    """

    name: ClassVar[str]
    description: ClassVar[str] = ""
    components: ClassVar[tuple[str, ...]] = ()
    """Optional pre-split input components accepted in addition to ``value``."""
    outputs: ClassVar[tuple[str, ...]] = ()
    """Derived column suffixes produced by ``standardize`` (excluding validity columns)."""

    # ------------------------------------------------------------------ contract
    @abstractmethod
    def validate(self, parts: Parts) -> ValidationResult:
        """Validate one value. Missing values are handled by the caller."""

    @abstractmethod
    def standardize(self, parts: Parts) -> dict[str, Any]:
        """Return derived values keyed by suffix (subset of ``outputs``)."""

    @abstractmethod
    def comparisons(self, field_name: str, options: ComparisonOptions) -> list["ComparisonCreator"]:
        """Return Splink comparison(s) for a canonical field named ``field_name``."""

    # ------------------------------------------------------------------ helpers
    def input_columns(self, field_name: str) -> list[str]:
        """Canonical input columns this field type reads for ``field_name``."""
        return [field_name] + [f"{field_name}_{c}" for c in self.components]

    def output_columns(self, field_name: str) -> list[str]:
        """All columns ``apply`` adds for ``field_name``."""
        return [f"{field_name}_{s}" for s in self.outputs] + [
            f"{field_name}_{VALID_SUFFIX}",
            f"{field_name}_{INVALID_REASON_SUFFIX}",
        ]

    def process(self, parts: Parts) -> dict[str, Any]:
        """Validate + standardize one value, returning all derived values incl. validity.

        A value whose every part is missing yields all-None outputs and
        ``valid=False`` with reason ``missing``. An invalid value is still standardized
        (so it remains visible in review) but flagged.
        """
        clean_parts = {k: to_text(v) for k, v in parts.items()}
        empty = {s: None for s in self.outputs}
        if all(v is None for v in clean_parts.values()):
            return {**empty, VALID_SUFFIX: False, INVALID_REASON_SUFFIX: "missing"}
        result = self.validate(clean_parts)
        derived = {**empty, **self.standardize(clean_parts)}
        derived[VALID_SUFFIX] = result.is_valid
        derived[INVALID_REASON_SUFFIX] = result.reason
        return derived

    def apply(self, df: pd.DataFrame, field_name: str) -> pd.DataFrame:
        """Add derived columns for ``field_name`` to a copy of ``df``.

        Row-wise application keeps standardizers value-level pure functions, which is
        what makes them portable to Spark UDFs. Throughput is adequate for the
        pilot's scale (tens of thousands of rows per source).
        """
        cols = [c for c in self.input_columns(field_name) if c in df.columns]
        if not cols:
            raise KeyError(
                f"none of the input columns {self.input_columns(field_name)} present for "
                f"field '{field_name}' ({self.name})"
            )
        records = df[cols].to_dict("records")
        rows = []
        for rec in records:
            parts = {("value" if c == field_name else c[len(field_name) + 1 :]): v for c, v in rec.items()}
            rows.append(self.process(parts))
        derived = pd.DataFrame(rows, index=df.index)
        derived.columns = [f"{field_name}_{c}" for c in derived.columns]
        out = df.copy()
        for c in derived.columns:
            out[c] = derived[c]
        return out
