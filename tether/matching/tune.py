"""Threshold tuning on a *separate* synthetic seed, so reported metrics are not tuned in-sample."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tether.config import EngagementConfig, Thresholds
from tether.matching.clustering import split_violating_clusters
from tether.matching.pipeline import LinkageResult, run_linkage
from tether.reporting.crosswalk import build_crosswalk
from tether.reporting.evaluation import cluster_metrics_vs_truth, pairwise_at_threshold, true_pairs_from_truth


@dataclass
class TuningResult:
    tuning_seed: int
    n_entities: int
    auto_link: float
    cluster: float
    auto_link_sweep: pd.DataFrame
    cluster_sweep: pd.DataFrame

    def thresholds(self, review_lower: float) -> Thresholds:
        return Thresholds(auto_link=self.auto_link, review_lower=review_lower, cluster=self.cluster)

    def save(self, path: str | Path) -> Path:
        Path(path).write_text(yaml.safe_dump({
            "tuning_seed": self.tuning_seed, "n_entities": self.n_entities,
            "thresholds": {"auto_link": self.auto_link, "cluster": self.cluster},
            "auto_link_sweep": self.auto_link_sweep.to_dict("records"),
            "cluster_sweep": self.cluster_sweep.to_dict("records"),
        }, sort_keys=False), encoding="utf-8")
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
    cfg.matching.save_m_to_knowledge_base = False
    cfg.engagement.id = f"{base.engagement.id}-tune{seed}"
    cfg.config_dir = workdir
    return cfg, ds.truth


def sweep(result: LinkageResult, truth: pd.DataFrame, review_lower: float) -> TuningResult:
    """Sweep auto_link (pairwise F1 on edges) and cluster threshold (crosswalk cluster F1)."""
    assert result.matcher is not None, "run with keep_matcher=True"
    tp = true_pairs_from_truth(truth)
    sources = set(truth["source"])
    edges = result.edges
    edges = edges[edges["unique_id_l"].str.split(":").str[0].isin(sources) & edges["unique_id_r"].str.split(":").str[0].isin(sources)]
    grid = [round(x, 3) for x in np.arange(0.5, 1.0, 0.025)] + [0.99, 0.995]
    al = pd.DataFrame([pairwise_at_threshold(edges, "match_probability", t, tp) for t in grid if t > review_lower])
    best_al = float(al.sort_values(["f1", "threshold"], ascending=[False, True]).iloc[0]["threshold"])
    rows = []
    records = result.prepared.all_records
    profile = result.prepared.profile
    for t in [g for g in grid if g >= best_al]:
        e = result.edges[result.edges["match_probability"] >= t]
        membership = result.matcher.cluster_edges(e, t)
        membership, _ = split_violating_clusters(membership, e, records, profile.hard_constraints)
        from tether.matching.clustering import ClusterQAConfig, cluster_metrics

        metrics = cluster_metrics(membership, e, ClusterQAConfig(profile.max_cluster_size, profile.min_edge_density))
        cw = build_crosswalk(metrics, records)
        cw = cw[cw["source"].isin(sources)]
        rows.append({"threshold": t, **{k: v for k, v in cluster_metrics_vs_truth(cw, truth).items()
                                       if k in ("precision", "recall", "f1", "entity_exact_match_rate")}})
    cl = pd.DataFrame(rows)
    best_cl = float(cl.sort_values(["f1", "threshold"], ascending=[False, True]).iloc[0]["threshold"])
    return TuningResult(tuning_seed=-1, n_entities=len(records), auto_link=best_al, cluster=max(best_cl, best_al),
                        auto_link_sweep=al, cluster_sweep=cl)


def tune_thresholds(base: EngagementConfig, seed: int, n_entities: int, workdir: str | Path) -> TuningResult:
    """Generate a synthetic tuning set with ``seed``, run the pipeline, and pick thresholds."""
    logging.getLogger("splink").setLevel(logging.WARNING)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    cfg, truth = tuning_config(base, seed, n_entities, workdir)
    result = run_linkage(cfg, keep_matcher=True)
    out = sweep(result, truth, base.thresholds.review_lower)
    out.tuning_seed = seed
    out.n_entities = n_entities
    return out
