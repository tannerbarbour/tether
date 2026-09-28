"""Threshold tuning on a *separate* synthetic seed, chosen on operating cost, not F1 alone.

Cost model (per threshold ``t`` used as ``auto_link``):

* pairs scored >= t are auto-accepted: false ones cost ``false_link_minutes`` each (a wrong merge in
  a deliverable that has to be found and undone);
* pairs in ``[review_lower, t)`` go to an analyst: ``review_minutes_per_pair`` each, and the review is
  assumed to resolve them correctly;
* true pairs below ``review_lower`` (or never blocked) are missed whatever ``t`` is:
  ``missed_link_minutes`` each. This term is constant across ``t`` and is reported, not optimised.

The cluster threshold is chosen by crosswalk F1 subject to ``cluster >= auto_link``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tether.config import EngagementConfig, Thresholds
from tether.matching.clustering import ClusterQAConfig, cluster_metrics, split_violating_clusters
from tether.matching.pipeline import LinkageResult, run_linkage
from tether.reporting.crosswalk import build_crosswalk
from tether.reporting.evaluation import cluster_metrics_vs_truth, pair_key, pairwise_at_threshold, true_pairs_from_truth


@dataclass(frozen=True)
class CostModel:
    review_minutes_per_pair: float = 1.0
    false_link_minutes: float = 30.0
    missed_link_minutes: float = 10.0


@dataclass
class TuningResult:
    tuning_seed: int
    n_entities: int
    auto_link: float
    cluster: float
    auto_link_sweep: pd.DataFrame
    cluster_sweep: pd.DataFrame
    cost_model: CostModel = field(default_factory=CostModel)
    f1_best_auto_link: float = 0.0

    def thresholds(self, review_lower: float) -> Thresholds:
        return Thresholds(auto_link=self.auto_link, review_lower=review_lower, cluster=self.cluster)

    def as_dict(self) -> dict:
        return {"tuning_seed": self.tuning_seed, "n_entities": self.n_entities,
                "thresholds": {"auto_link": self.auto_link, "cluster": self.cluster},
                "f1_best_auto_link": self.f1_best_auto_link, "cost_model": self.cost_model.__dict__,
                "auto_link_sweep": self.auto_link_sweep.to_dict("records"),
                "cluster_sweep": self.cluster_sweep.to_dict("records")}

    def save(self, path: str | Path) -> Path:
        Path(path).write_text(yaml.safe_dump(self.as_dict(), sort_keys=False), encoding="utf-8")
        return Path(path)


def tuning_config(base: EngagementConfig, seed: int, n_entities: int, workdir: Path) -> tuple[EngagementConfig, pd.DataFrame]:
    """Generate a tuning dataset with ``seed`` and a config pointing at it (same mappings, thresholds, KB)."""
    from tether.synthetic import SyntheticConfig, generate

    ds = generate(SyntheticConfig(n_entities=n_entities, seed=seed))
    data = workdir / "data"
    ds.write(data)
    cfg = base.model_copy(deep=True)
    for src in cfg.sources:
        src.path = data / f"{src.name}.csv"
        src.mapping = base.resolve(src.mapping)
    cfg.reference.nppes_path = data / "nppes_sample.csv" if base.reference.nppes_path else None
    cfg.reference.zip_centroids_path = data / "zip_centroids.csv" if base.reference.zip_centroids_path else None
    cfg.knowledge_base.path = base.resolve(cfg.knowledge_base.path)
    cfg.engagement.id = f"{base.engagement.id}-tune{seed}"
    cfg.config_dir = workdir
    return cfg, ds.truth


def auto_link_sweep(edges: pd.DataFrame, truth_pairs: set[str], review_lower: float, grid: list[float],
                    cost: CostModel) -> pd.DataFrame:
    """P/R/F1, queue size, false auto-links and expected cost for each candidate auto_link."""
    keys = pair_key(edges["unique_id_l"], edges["unique_id_r"])
    is_true = keys.isin(truth_pairs).to_numpy()
    prob = edges["match_probability"].to_numpy()
    candidates_true = int(is_true.sum())
    missed_constant = len(truth_pairs) - int((is_true & (prob >= review_lower)).sum())
    rows = []
    for t in grid:
        if t <= review_lower:
            continue
        acc = prob >= t
        band = (prob >= review_lower) & ~acc
        fp = int((acc & ~is_true).sum())
        m = pairwise_at_threshold(edges, "match_probability", t, truth_pairs)
        queue = int(band.sum())
        cost_min = fp * cost.false_link_minutes + queue * cost.review_minutes_per_pair + missed_constant * cost.missed_link_minutes
        rows.append({**m, "queue_size": queue, "queue_true_pairs": int((band & is_true).sum()),
                     "false_auto_links": fp, "missed_below_review": missed_constant,
                     "expected_cost_minutes": round(cost_min, 1)})
    out = pd.DataFrame(rows)
    out.attrs["candidates_true"] = candidates_true
    return out


def sweep(result: LinkageResult, truth: pd.DataFrame, review_lower: float, cost: CostModel = CostModel()) -> TuningResult:
    """Sweep auto_link (cost) and the cluster threshold (crosswalk F1) on one run."""
    assert result.matcher is not None, "run with keep_matcher=True"
    tp = true_pairs_from_truth(truth)
    sources = set(truth["source"])
    edges = result.edges
    edges = edges[edges["unique_id_l"].str.split(":").str[0].isin(sources) & edges["unique_id_r"].str.split(":").str[0].isin(sources)]
    grid = [round(x, 3) for x in np.arange(0.5, 1.0, 0.025)] + [0.99, 0.995, 0.999]
    al = auto_link_sweep(edges, tp, review_lower, grid, cost)
    best_cost = float(al.sort_values(["expected_cost_minutes", "threshold"], ascending=[True, True]).iloc[0]["threshold"])
    best_f1 = float(al.sort_values(["f1", "threshold"], ascending=[False, True]).iloc[0]["threshold"])
    rows = []
    records = result.prepared.all_records
    profile = result.prepared.profile
    for t in [g for g in grid if g >= best_cost]:
        e = result.edges[result.edges["match_probability"] >= t]
        membership = result.matcher.cluster_edges(e, t)
        membership, _ = split_violating_clusters(membership, e, records, profile.hard_constraints)
        metrics = cluster_metrics(membership, e, ClusterQAConfig(profile.max_cluster_size, profile.min_edge_density))
        cw = build_crosswalk(metrics, records)
        cw = cw[cw["source"].isin(sources)]
        rows.append({"threshold": t, **{k: v for k, v in cluster_metrics_vs_truth(cw, truth).items()
                                       if k in ("precision", "recall", "f1", "entity_exact_match_rate")}})
    cl = pd.DataFrame(rows)
    best_cl = float(cl.sort_values(["f1", "threshold"], ascending=[False, True]).iloc[0]["threshold"])
    return TuningResult(tuning_seed=-1, n_entities=len(records), auto_link=best_cost, cluster=max(best_cl, best_cost),
                        auto_link_sweep=al, cluster_sweep=cl, cost_model=cost, f1_best_auto_link=best_f1)


def tune_thresholds(base: EngagementConfig, seed: int, n_entities: int, workdir: str | Path,
                    cost: CostModel = CostModel()) -> TuningResult:
    """Generate a synthetic tuning set with ``seed``, run the pipeline (KB read-only), pick thresholds."""
    logging.getLogger("splink").setLevel(logging.WARNING)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    cfg, truth = tuning_config(base, seed, n_entities, workdir)
    result = run_linkage(cfg, keep_matcher=True, kb_write=False)
    out = sweep(result, truth, base.thresholds.review_lower, cost)
    out.tuning_seed = seed
    out.n_entities = n_entities
    return out
