"""Crosswalk and review-queue tables."""

from __future__ import annotations

import pandas as pd

CROSSWALK_COLUMNS = [
    "source", "source_record_id", "entity_id", "cluster_confidence", "cluster_size",
    "record_attachment", "edge_density", "flags", "unique_id",
]


def build_crosswalk(metrics: pd.DataFrame, records: pd.DataFrame, entity_prefix: str = "ENT") -> pd.DataFrame:
    """One row per input record with a stable, human-readable entity id.

    Entity ids are assigned in order of the cluster's smallest member key so the same
    inputs yield the same ids across runs.
    """
    m = metrics.merge(records[["unique_id", "source_dataset", "source_record_id"]], on="unique_id", how="left")
    first_member = m.groupby("cluster_id")["unique_id"].min().sort_values()
    entity_ids = {cid: f"{entity_prefix}{i + 1:06d}" for i, cid in enumerate(first_member.index)}
    m["entity_id"] = m["cluster_id"].map(entity_ids)
    m = m.rename(columns={"source_dataset": "source"})
    return m[CROSSWALK_COLUMNS].sort_values(["entity_id", "source", "source_record_id"]).reset_index(drop=True)


def _display(df: pd.DataFrame, display_columns: list[str]) -> pd.DataFrame:
    cols = ["unique_id_l", "unique_id_r", "match_probability", "match_weight"]
    for c in display_columns:
        cols += [x for x in (f"{c}_l", f"{c}_r") if x in df.columns]
    return df[[c for c in cols if c in df.columns]]


def build_review_queue(
    predictions: pd.DataFrame, review_lower: float, auto_link: float, display_columns: list[str],
    rejected: pd.DataFrame | None = None, npi_column: str | None = None, audit_sample: int = 0, seed: int = 0,
) -> pd.DataFrame:
    """Analyst review queue with a ``review_reason`` per pair.

    * ``score_band``: accepted-by-constraint pairs scored in ``[review_lower, auto_link)``.
    * ``same_npi_name_conflict``: pairs rejected by the name-agreement constraint although both
      records carry the same valid identifier (the deterministic guard refused them too).
    * ``rejected_audit_sample``: a random sample of the other rejected pairs, so the constraint's
      behaviour is audited every run rather than trusted.
    """
    band = predictions[(predictions["match_probability"] >= review_lower) & (predictions["match_probability"] < auto_link)]
    parts = [_display(band, display_columns).assign(review_reason="score_band")]
    if rejected is not None and len(rejected):
        rej = rejected
        if npi_column and f"{npi_column}_l" in rej.columns:
            same = rej[rej[f"{npi_column}_l"].notna() & (rej[f"{npi_column}_l"] == rej[f"{npi_column}_r"])]
            parts.append(_display(same, display_columns).assign(review_reason="same_npi_name_conflict"))
            rej = rej.drop(same.index)
        if audit_sample and len(rej):
            sample = rej.sample(min(audit_sample, len(rej)), random_state=seed)
            parts.append(_display(sample, display_columns).assign(review_reason="rejected_audit_sample"))
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["review_reason", "match_probability"], ascending=[True, False]).reset_index(drop=True)
