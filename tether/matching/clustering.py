"""Post-match stage: edge assembly, pair/hard constraints, clustering, cluster QA metrics.

Clustering itself is delegated to Splink (backend-scalable connected components); the
constraint split step runs in Python on the (rare) violating clusters only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from tether.profiles.base import HardConstraint, PairConstraint

EDGE_COLUMNS = ["unique_id_l", "unique_id_r", "match_probability", "link_source"]


def _ordered(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    swap = a > b
    return a.where(~swap, b), b.where(~swap, a)


def apply_pair_constraints(predictions: pd.DataFrame, constraints: Sequence[PairConstraint]) -> pd.Series:
    """Boolean mask of predictions that satisfy every PairConstraint.

    Splink's ``gamma_<comparison>`` holds the level index: -1 null, 0 else, >0 agreement.
    """
    ok = pd.Series(True, index=predictions.index)
    for c in constraints:
        cols = [f"gamma_{name}" for name in c.comparisons if f"gamma_{name}" in predictions.columns]
        if not cols:
            continue
        ok &= predictions[cols].max(axis=1) > 0
    return ok


def build_edges(
    predictions: pd.DataFrame,
    deterministic_pairs: pd.DataFrame,
    pair_constraints: Sequence[PairConstraint],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge probabilistic predictions with deterministic pairs into one edge table.

    Returns ``(edges, rejected)``. Deterministic pairs get probability 1.0 and are exempt
    from pair constraints. A pair present in both keeps ``link_source='both'``.
    """
    ok = apply_pair_constraints(predictions, pair_constraints)
    rejected = predictions.loc[~ok].copy()
    rejected["rejection_reason"] = "pair_constraint: " + "; ".join(c.description for c in pair_constraints)
    prob = predictions.loc[ok, ["unique_id_l", "unique_id_r", "match_probability"]].copy()
    prob["unique_id_l"], prob["unique_id_r"] = _ordered(prob["unique_id_l"], prob["unique_id_r"])
    prob["link_source"] = "probabilistic"

    det = deterministic_pairs[["unique_id_l", "unique_id_r"]].copy()
    if len(det):
        det["unique_id_l"], det["unique_id_r"] = _ordered(det["unique_id_l"], det["unique_id_r"])
    det["match_probability"] = 1.0
    det["link_source"] = "deterministic"

    edges = pd.concat([prob, det], ignore_index=True)
    if edges.empty:
        return pd.DataFrame(columns=EDGE_COLUMNS), rejected
    agg = edges.groupby(["unique_id_l", "unique_id_r"], as_index=False).agg(
        match_probability=("match_probability", "max"),
        n_sources=("link_source", "nunique"),
        link_source=("link_source", "first"),
    )
    agg.loc[agg["n_sources"] > 1, "link_source"] = "both"
    return agg[EDGE_COLUMNS], rejected


def drop_conflicting_edges(
    edges: pd.DataFrame, records: pd.DataFrame, constraints: Sequence[HardConstraint]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove edges whose endpoints hold different non-null values of a constrained column."""
    if edges.empty or not constraints:
        return edges, edges.iloc[0:0].copy()
    lookup = records.set_index("unique_id")
    conflict = pd.Series(False, index=edges.index)
    for c in constraints:
        vl = edges["unique_id_l"].map(lookup[c.column])
        vr = edges["unique_id_r"].map(lookup[c.column])
        conflict |= vl.notna() & vr.notna() & (vl != vr)
    return edges.loc[~conflict].reset_index(drop=True), edges.loc[conflict].reset_index(drop=True)


def split_violating_clusters(
    membership: pd.DataFrame, edges: pd.DataFrame, records: pd.DataFrame, constraints: Sequence[HardConstraint]
) -> tuple[pd.DataFrame, list[dict]]:
    """Split clusters that (transitively) contain conflicting constrained values.

    For each violating cluster: seed one sub-cluster per distinct constrained value, then
    attach every remaining member to the seed it has the strongest edge into (records
    with no edge to any seed become singletons). Returns (membership, split log).
    """
    if membership.empty or not constraints:
        return membership, []
    lookup = records.set_index("unique_id")
    member = membership.copy()
    log: list[dict] = []
    edge_index = pd.concat([
        edges[["unique_id_l", "unique_id_r", "match_probability"]].rename(columns={"unique_id_l": "a", "unique_id_r": "b"}),
        edges[["unique_id_r", "unique_id_l", "match_probability"]].rename(columns={"unique_id_r": "a", "unique_id_l": "b"}),
    ])
    for c in constraints:
        member["_v"] = member["unique_id"].map(lookup[c.column])
        distinct = member.groupby("cluster_id")["_v"].nunique(dropna=True)
        for cid in distinct[distinct > 1].index:
            members = member.loc[member["cluster_id"] == cid]
            seeds = members.dropna(subset=["_v"]).groupby("_v")["unique_id"].apply(list).to_dict()
            assignment: dict[str, str] = {}
            for value, uids in seeds.items():
                for uid in uids:
                    assignment[uid] = f"{cid}#{value}"
            local_edges = edge_index[edge_index["a"].isin(members["unique_id"]) & edge_index["b"].isin(assignment)]
            for uid in members.loc[members["_v"].isna(), "unique_id"]:
                cand = local_edges[local_edges["a"] == uid]
                if cand.empty:
                    assignment[uid] = f"{cid}#{uid}"
                else:
                    best = cand.sort_values("match_probability", ascending=False).iloc[0]["b"]
                    assignment[uid] = assignment[best]
            member.loc[member["cluster_id"] == cid, "cluster_id"] = member.loc[
                member["cluster_id"] == cid, "unique_id"].map(assignment)
            log.append({"constraint": c.column, "original_cluster": cid, "n_records": len(members),
                        "n_subclusters": len(set(assignment.values()))})
    return member.drop(columns="_v"), log


@dataclass
class ClusterQAConfig:
    max_cluster_size: int
    min_edge_density: float


def cluster_metrics(membership: pd.DataFrame, edges: pd.DataFrame, qa: ClusterQAConfig) -> pd.DataFrame:
    """Per-record cluster metrics.

    * ``cluster_size``
    * ``edge_density``: edges within the cluster / possible pairs (1.0 for singletons)
    * ``cluster_confidence``: weakest attachment - the minimum over members of each
      member's strongest edge into the cluster (NaN for singletons)
    * ``record_attachment``: this record's strongest edge into its cluster
    * ``flags``: ``oversized`` / ``low_density`` / ``singleton``
    """
    m = membership[["unique_id", "cluster_id"]].copy()
    cid = m.set_index("unique_id")["cluster_id"]
    e = edges.copy()
    if len(e):
        e["cl"], e["cr"] = e["unique_id_l"].map(cid), e["unique_id_r"].map(cid)
        e = e[e["cl"] == e["cr"]]
    size = m.groupby("cluster_id")["unique_id"].size().rename("cluster_size")
    n_edges = e.groupby("cl").size().rename("n_edges") if len(e) else pd.Series(dtype=int, name="n_edges")
    attach = pd.concat([
        e[["unique_id_l", "match_probability"]].rename(columns={"unique_id_l": "unique_id"}),
        e[["unique_id_r", "match_probability"]].rename(columns={"unique_id_r": "unique_id"}),
    ]).groupby("unique_id")["match_probability"].max().rename("record_attachment") if len(e) else \
        pd.Series(dtype=float, name="record_attachment")
    m = m.join(size, on="cluster_id").join(n_edges, on="cluster_id").join(attach, on="unique_id")
    m["n_edges"] = m["n_edges"].fillna(0).astype(int)
    possible = m["cluster_size"] * (m["cluster_size"] - 1) / 2
    m["edge_density"] = (m["n_edges"] / possible.where(possible > 0)).fillna(1.0)
    conf = m.groupby("cluster_id")["record_attachment"].min().rename("cluster_confidence")
    m = m.join(conf, on="cluster_id")
    flags = []
    for _, r in m.iterrows():
        f = []
        if r["cluster_size"] == 1:
            f.append("singleton")
        if r["cluster_size"] > qa.max_cluster_size:
            f.append("oversized")
        if r["cluster_size"] > 1 and r["edge_density"] < qa.min_edge_density:
            f.append("low_density")
        flags.append("|".join(f))
    m["flags"] = flags
    return m
