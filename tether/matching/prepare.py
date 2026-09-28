"""Cleaning stage: canonical frames -> derived (standardized) frames ready for matching."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import pandas as pd

from tether.config import EngagementConfig
from tether.fields.base import FieldType
from tether.ingestion.loader import CANONICAL_ID_COLUMNS, load_source
from tether.profiles.base import EntityProfile, FieldSpec, get_profile


@dataclass
class PreparedData:
    """Standardized input frames plus everything the matcher needs to interpret them."""

    profile: EntityProfile
    field_specs: dict[str, FieldSpec]
    """Effective field specs after engagement overrides (excluded fields removed)."""
    field_types: dict[str, FieldType]
    frames: list[pd.DataFrame]
    geo_available: bool
    validation_summary: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def all_records(self) -> pd.DataFrame:
        return pd.concat(self.frames, ignore_index=True)


def effective_field_specs(profile: EntityProfile, config: EngagementConfig) -> dict[str, FieldSpec]:
    """Apply ``matching.fields`` overrides (role / tf / phonetic / exclude) to the profile."""
    specs: dict[str, FieldSpec] = {}
    for name, spec in profile.fields.items():
        ov = config.matching.fields.get(name)
        if ov is None:
            specs[name] = spec
            continue
        if ov.exclude:
            continue
        updates: dict[str, Any] = {}
        if ov.role is not None:
            updates["role"] = ov.role
        if ov.term_frequency is not None:
            updates["term_frequency"] = ov.term_frequency
        if ov.include_phonetic is not None:
            updates["include_phonetic"] = ov.include_phonetic
        specs[name] = FieldSpec(**{**spec.__dict__, **updates})
    unknown = set(config.matching.fields) - set(profile.fields)
    if unknown:
        raise ValueError(f"matching.fields overrides reference unknown fields {sorted(unknown)}")
    return specs


def standardize_frame(canonical: pd.DataFrame, field_types: Mapping[str, FieldType]) -> pd.DataFrame:
    """Add derived columns for every field type to a canonical frame."""
    out = canonical
    for fname, ft in field_types.items():
        out = ft.apply(out, fname)
    return out


def validation_summary(frames: list[pd.DataFrame], field_names: list[str]) -> pd.DataFrame:
    """Per source and field: counts of valid / invalid / missing values (invalid values are kept)."""
    rows = []
    for df in frames:
        source = df["source_dataset"].iloc[0] if len(df) else ""
        for f in field_names:
            reasons = df[f"{f}_invalid_reason"]
            rows.append({
                "source": source, "field": f, "n": len(df),
                "valid": int(df[f"{f}_valid"].fillna(False).astype(bool).sum()),
                "missing": int((reasons == "missing").sum()),
                "invalid": int(((reasons != "missing") & reasons.notna()).sum()),
                "invalid_reasons": ", ".join(sorted(set(reasons.dropna()) - {"missing"})),
            })
    return pd.DataFrame(rows)


def prepare(config: EngagementConfig, dependencies: Mapping[str, Any] | None = None) -> PreparedData:
    """Load, map and standardize every configured source.

    ``dependencies`` are injected into field types (org aliases, title taxonomy, ZIP
    centroids). They come from the knowledge base and reference sources when those
    layers are configured; the pilot's matching stage works without them.
    """
    profile = get_profile(config.profile)
    specs = effective_field_specs(profile, config)
    sub_profile = EntityProfile(name=profile.name, description=profile.description, fields=specs)
    field_types = sub_profile.instantiate_fields(dependencies)
    frames = []
    for source in config.sources:
        canonical = load_source(source, config, profile)
        frames.append(standardize_frame(canonical, field_types))
    address_fields = [n for n, s in specs.items() if s.field_type == "address"]
    geo = bool(address_fields) and any(
        df[f"{a}_lat"].notna().any() for df in frames for a in address_fields
    )
    return PreparedData(
        profile=profile, field_specs=specs, field_types=field_types, frames=frames,
        geo_available=geo, validation_summary=validation_summary(frames, list(specs)),
    )


def id_columns() -> tuple[str, ...]:
    return CANONICAL_ID_COLUMNS
