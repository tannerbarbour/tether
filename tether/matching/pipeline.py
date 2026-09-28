"""End-to-end linkage run: prepare -> deterministic pre-pass -> train -> predict -> cluster -> crosswalk."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from tether.config import EngagementConfig
from tether.matching.clustering import (
    ClusterQAConfig,
    build_edges,
    cluster_metrics,
    drop_conflicting_edges,
    split_violating_clusters,
)
from tether.matching.prepare import PreparedData, prepare
from tether.matching.splink_adapter import MProbabilities, SplinkMatcher, TrainedModel, build_settings
from tether.reporting.crosswalk import build_crosswalk, build_review_queue

DISPLAY_COLUMNS = ["name_full_std", "org_clean", "address_full_std", "npi_std", "phone_std", "title_std"]


@dataclass
class LinkageResult:
    config: EngagementConfig
    prepared: PreparedData
    deterministic_pairs: pd.DataFrame
    predictions: pd.DataFrame
    rejected_pairs: pd.DataFrame
    conflicting_edges: pd.DataFrame
    edges: pd.DataFrame
    membership: pd.DataFrame
    metrics: pd.DataFrame
    crosswalk: pd.DataFrame
    review_queue: pd.DataFrame
    model: TrainedModel
    split_log: list[dict] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    matcher: SplinkMatcher | None = None

    def write(self, out_dir: str | Path) -> dict[str, Path]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        for name, df in {
            "crosswalk": self.crosswalk, "review_queue": self.review_queue, "edges": self.edges,
            "deterministic_pairs": self.deterministic_pairs, "rejected_pairs": self.rejected_pairs,
            "conflicting_edges": self.conflicting_edges, "validation_summary": self.prepared.validation_summary,
        }.items():
            paths[name] = out / f"{name}.csv"
            df.to_csv(paths[name], index=False)
        paths["predictions"] = out / "predictions.parquet"
        self.predictions.to_parquet(paths["predictions"], index=False)
        paths["records"] = out / "records.parquet"
        self.prepared.all_records.to_parquet(paths["records"], index=False)
        paths["model"] = out / "model.json"
        self.model.save(paths["model"])
        paths["stats"] = out / "run_stats.json"
        paths["stats"].write_text(json.dumps({**self.stats, "split_log": self.split_log,
                                              "training_log": self.model.training_log}, indent=2, default=str))
        return paths


def run_linkage(
    config: EngagementConfig,
    dependencies: Mapping[str, Any] | None = None,
    m_probabilities: MProbabilities | None = None,
    keep_matcher: bool = False,
) -> LinkageResult:
    """Execute the full matching pipeline for an engagement config."""
    t0 = time.time()
    deps = dict(dependencies or {})
    if "zip_centroids" not in deps and config.reference.zip_centroids_path:
        from tether.reference.zip_centroids import load_zip_centroids

        deps["zip_centroids"] = load_zip_centroids(config.resolve(config.reference.zip_centroids_path))
    prep = prepare(config, deps)
    profile, matching, th = prep.profile, config.matching, config.thresholds

    settings = build_settings(profile, prep.field_specs, prep.field_types, matching, prep.geo_available,
                              m_probabilities=m_probabilities, retain_columns=["source_record_id"])
    matcher = SplinkMatcher(prep.frames, [s.name for s in config.sources], settings)

    det_rules = profile.deterministic_rules
    det_pairs = matcher.deterministic_pairs(det_rules)
    if matching.probability_two_random_records_match is None and det_rules:
        matcher.estimate_probability_two_random_records_match(det_rules, matching.deterministic_rule_recall)
    matcher.estimate_u(matching.max_pairs_for_u, seed=matching.random_seed)
    training_rules = matching.training_blocking_rules if matching.training_blocking_rules is not None \
        else profile.training_blocking_rules
    matcher.estimate_m(training_rules)
    matcher.apply_probability_floor(matching.probability_floor)

    predictions = matcher.predict(th.review_lower)
    edges, rejected = build_edges(predictions, det_pairs, profile.pair_constraints)
    records = prep.all_records
    edges, conflicting = drop_conflicting_edges(edges, records, profile.hard_constraints)
    membership = matcher.cluster_edges(edges, th.cluster)
    membership, split_log = split_violating_clusters(membership, edges[edges["match_probability"] >= th.cluster],
                                                     records, profile.hard_constraints)
    qa = ClusterQAConfig(
        max_cluster_size=config.cluster_qa.max_cluster_size or profile.max_cluster_size,
        min_edge_density=config.cluster_qa.min_edge_density or profile.min_edge_density,
    )
    metrics = cluster_metrics(membership, edges[edges["match_probability"] >= th.cluster], qa)
    crosswalk = build_crosswalk(metrics, records)
    review = build_review_queue(predictions, th.review_lower, th.auto_link, DISPLAY_COLUMNS)
    model = matcher.trained_model()
    stats = {
        "engagement": config.engagement.id, "profile": profile.name, "link_type": matching.link_type,
        "n_records": int(len(records)), "n_sources": len(config.sources),
        "n_deterministic_pairs": int(len(det_pairs)), "n_predictions_above_review": int(len(predictions)),
        "n_pairs_rejected_by_pair_constraints": int(len(rejected)),
        "n_edges_dropped_by_hard_constraints": int(len(conflicting)),
        "n_edges": int(len(edges)), "n_edges_at_cluster_threshold": int((edges["match_probability"] >= th.cluster).sum()),
        "n_clusters": int(membership["cluster_id"].nunique()), "n_clusters_split": len(split_log),
        "n_review_queue": int(len(review)),
        "n_flagged_records": int((metrics["flags"].str.contains("oversized|low_density")).sum()),
        "thresholds": th.model_dump(), "geo_available": prep.geo_available,
        "runtime_seconds": round(time.time() - t0, 2),
    }
    return LinkageResult(
        config=config, prepared=prep, deterministic_pairs=det_pairs, predictions=predictions,
        rejected_pairs=rejected, conflicting_edges=conflicting, edges=edges, membership=membership,
        metrics=metrics, crosswalk=crosswalk, review_queue=review, model=model, split_log=split_log,
        stats=stats, matcher=matcher if keep_matcher else None,
    )
