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


def build_review_queue(
    predictions: pd.DataFrame, review_lower: float, auto_link: float, display_columns: list[str]
) -> pd.DataFrame:
    """Pairs scored in ``[review_lower, auto_link)`` with side-by-side display values."""
    band = predictions[(predictions["match_probability"] >= review_lower) & (predictions["match_probability"] < auto_link)]
    cols = ["unique_id_l", "unique_id_r", "match_probability", "match_weight"]
    for c in display_columns:
        cols += [x for x in (f"{c}_l", f"{c}_r") if x in band.columns]
    return band[cols].sort_values("match_probability", ascending=False).reset_index(drop=True)
