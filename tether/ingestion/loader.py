"""Deterministic source loading and mapping application.

The LLM only *proposes* mappings (Phase 3). This module applies an approved
``SchemaMapping`` to full data with plain pandas, so the path from raw file to
canonical frame is fully deterministic and auditable.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tether.config import EngagementConfig, SchemaMapping, SourceConfig
from tether.profiles.base import EntityProfile

RECORD_KEY_SEP = ":"
CANONICAL_ID_COLUMNS = ("unique_id", "source_dataset", "source_record_id")


def read_table(path: str | Path, fmt: str = "csv", encoding: str = "utf-8") -> pd.DataFrame:
    """Read a CSV or Parquet file with every column as string (identifiers keep leading zeros)."""
    path = Path(path)
    if fmt == "parquet":
        df = pd.read_parquet(path)
        return df.astype("string").astype(object).where(df.notna(), None)
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""], encoding=encoding)
    return df.astype(object).where(df.notna(), None)


def make_unique_id(source: str, record_id: str) -> str:
    return f"{source}{RECORD_KEY_SEP}{record_id}"


def apply_mapping(
    raw: pd.DataFrame,
    mapping: SchemaMapping,
    source_name: str,
    record_id_column: str,
    profile: EntityProfile,
) -> pd.DataFrame:
    """Project a raw frame onto the profile's canonical schema.

    Output columns: ``unique_id``, ``source_dataset``, ``source_record_id`` followed by
    every canonical column of the profile (unmapped ones filled with None so all
    sources share one schema, which Splink requires).
    """
    if mapping.status != "approved":
        raise ValueError(f"mapping for source '{mapping.source}' is '{mapping.status}', not approved")
    allowed = set(profile.canonical_columns())
    bad = [e.target for e in mapping.columns if e.target not in allowed]
    if bad:
        raise ValueError(f"mapping targets not in profile '{profile.name}': {bad}")
    missing = [e.source_column for e in mapping.columns if e.source_column not in raw.columns]
    if missing:
        raise KeyError(f"source '{source_name}' lacks mapped columns: {missing}")
    if record_id_column not in raw.columns:
        raise KeyError(f"record id column '{record_id_column}' not in source '{source_name}'")
    if raw[record_id_column].isna().any() or not raw[record_id_column].is_unique:
        raise ValueError(f"record id column '{record_id_column}' in '{source_name}' must be unique and non-null")

    out = pd.DataFrame(index=raw.index)
    out["unique_id"] = [make_unique_id(source_name, r) for r in raw[record_id_column]]
    out["source_dataset"] = source_name
    out["source_record_id"] = raw[record_id_column].astype(str)
    for col in profile.canonical_columns():
        out[col] = None
    for entry in mapping.columns:
        out[entry.target] = raw[entry.source_column].values
    return out.reset_index(drop=True)


def load_source(source: SourceConfig, config: EngagementConfig, profile: EntityProfile) -> pd.DataFrame:
    """Read a configured source and apply its approved mapping."""
    if source.mapping is None:
        raise ValueError(f"source '{source.name}' has no mapping; run `tether propose-mapping` and approve it")
    mapping = SchemaMapping.load(config.resolve(source.mapping))
    raw = read_table(config.resolve(source.path), source.format, source.encoding)
    return apply_mapping(raw, mapping, source.name, source.record_id_column, profile)
