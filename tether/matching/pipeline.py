"""End-to-end linkage run: prepare -> deterministic pre-pass -> train -> predict -> cluster -> crosswalk."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from tether.config import EngagementConfig
from tether.knowledge import KnowledgeBase, comparison_schema_hash, open_knowledge_base
from tether.knowledge.base import new_asset_id
from tether.matching.clustering import (
    ClusterQAConfig,
    build_edges,
    cluster_metrics,
    drop_conflicting_edges,
    split_violating_clusters,
)
from tether.matching.prepare import PreparedData, prepare, standardize_frame
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
    hub_source: str | None = None

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


def _lookup_effect(prep: PreparedData, deps: Mapping[str, Any]) -> dict[str, int]:
    """How many standardized values the knowledge-base lookups actually changed.

    Re-standardizes org names and titles with the built-in tables only and counts records
    whose derived value differs. A zero here means the lookups added nothing beyond the
    engine's generic alias/taxonomy tables (e.g. the mock LLM reproduces them exactly).
    """
    from tether.fields import get_field_type

    out: dict[str, int] = {}
    records = prep.all_records
    for fname, spec in prep.field_specs.items():
        if spec.field_type == "org_name" and deps.get("aliases"):
            base = get_field_type("org_name")
            derived = base.apply(records[[fname]], fname)[f"{fname}_clean"]
            out[f"{fname}_clean_changed_by_lookup"] = int((derived.fillna("") != records[f"{fname}_clean"].fillna("")).sum())
            out[f"{fname}_lookup_rows"] = len(deps["aliases"])
        if spec.field_type == "job_title" and deps.get("taxonomy"):
            base = get_field_type("job_title")
            derived = base.apply(records[[fname]], fname)[f"{fname}_canonical"]
            out[f"{fname}_canonical_changed_by_lookup"] = int((derived.fillna("") != records[f"{fname}_canonical"].fillna("")).sum())
            out[f"{fname}_lookup_rows"] = len(deps["taxonomy"])
    return out


def _config_hash(config: EngagementConfig) -> str:
    return hashlib.sha256(json.dumps(config.model_dump(mode="json"), sort_keys=True, default=str).encode()).hexdigest()[:12]


def run_linkage(
    config: EngagementConfig,
    dependencies: Mapping[str, Any] | None = None,
    m_probabilities: MProbabilities | None = None,
    keep_matcher: bool = False,
    knowledge_base: KnowledgeBase | None = None,
) -> LinkageResult:
    """Execute the full matching pipeline for an engagement config.

    Knowledge-base interactions (all optional, governed by config): load m starting values,
    save trained m, store deterministic comparison vectors as labeled pairs, register the
    approved mappings and reference snapshots used, and append a run-history row that
    pins the knowledge-base version and snapshot ids for reproducibility.
    """
    from tether.ingestion.lookups import lookup_dependencies
    from tether.reference import NPPESReference, ZipCentroidReference, hub_entity_ids

    t0 = time.time()
    kb = knowledge_base or open_knowledge_base(config)
    kb_version_before = kb.version()
    run_id = new_asset_id()
    eng = config.engagement.id
    scope = config.engagement.reuse_scope
    snapshots: list[dict] = []

    deps = {**lookup_dependencies(config), **(dependencies or {})}
    if "zip_centroids" not in deps and config.reference.zip_centroids_path:
        zc = ZipCentroidReference(config.resolve(config.reference.zip_centroids_path))
        deps["zip_centroids"] = zc.as_dict()
        snapshots.append(zc.snapshot().as_dict())
    nppes = None
    if config.reference.nppes_path and (config.reference.enrich_from_nppes or config.reference.hub_link_via_nppes):
        nppes = NPPESReference(config.resolve(config.reference.nppes_path))
        snapshots.append(nppes.snapshot().as_dict())

    prep = prepare(config, deps)
    profile, matching, th = prep.profile, config.matching, config.thresholds
    schema_hash = comparison_schema_hash(prep.field_specs, profile.name)
    source_names = [s.name for s in config.sources]
    source_types = sorted({s.source_type for s in config.sources if s.source_type})
    person_profile = profile.name != "organization"
    npi_fields = [n for n, sp in prep.field_specs.items() if sp.field_type == "npi"]

    hub_source = None
    if nppes is not None and config.reference.hub_link_via_nppes and npi_fields:
        hub_source = "nppes"
        hub = nppes.hub_frame(profile.canonical_columns(), entity_type=1 if person_profile else 2, source_name=hub_source)
        prep.frames.append(standardize_frame(hub, prep.field_types))
        source_names.append(hub_source)
    if nppes is not None and config.reference.enrich_from_nppes and npi_fields:
        prep.frames = [nppes.enrich(f, f"{npi_fields[0]}_std", person_profile) for f in prep.frames]

    m_seed = m_probabilities
    if m_seed is None and matching.load_m_from_knowledge_base:
        m_seed = kb.m_probabilities(profile.name, schema_hash, eng, source_types, config.knowledge_base.use_promoted_only)
    retain = ["source_record_id"] + [c for c in DISPLAY_COLUMNS if c in prep.frames[0].columns]
    settings = build_settings(profile, prep.field_specs, prep.field_types, matching, prep.geo_available,
                              m_probabilities=m_seed or None, retain_columns=retain)
    matcher = SplinkMatcher(prep.frames, source_names, settings)

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
    if hub_source:
        crosswalk = hub_entity_ids(crosswalk, hub_source)
    accepted_predictions = predictions.loc[~predictions.index.isin(rejected.index)]
    review = build_review_queue(accepted_predictions, th.review_lower, th.auto_link, DISPLAY_COLUMNS)
    model = matcher.trained_model()
    npi_col = f"{npi_fields[0]}_std" if npi_fields else None
    n_rejected_sharing_npi = int(((rejected[f"{npi_col}_l"].notna()) & (rejected[f"{npi_col}_l"] == rejected[f"{npi_col}_r"])).sum()) \
        if npi_col and f"{npi_col}_l" in rejected.columns and len(rejected) else 0
    lookup_effect = _lookup_effect(prep, deps)

    # ---- knowledge base writes
    n_labeled = 0
    if matching.save_m_to_knowledge_base:
        kb.save_model_params(model.m_probabilities, profile.name, schema_hash, eng, source_types, scope, run_id)
        det_vectors = det_pairs.merge(predictions, on=["unique_id_l", "unique_id_r"], how="inner")
        n_labeled = kb.add_labeled_pairs(det_vectors, 1, "deterministic_rule", profile.name, schema_hash, eng, scope)
    for src in config.sources:
        if src.mapping:
            from tether.config import SchemaMapping

            kb.register_mapping(SchemaMapping.load(config.resolve(src.mapping)), eng, scope)
    for snap in snapshots:
        kb.register_reference_snapshot(snap, eng)

    stats = {
        "run_id": run_id, "engagement": eng, "profile": profile.name, "link_type": matching.link_type,
        "comparison_schema_hash": schema_hash, "knowledge_base_version": kb_version_before,
        "reference_snapshots": {s["source"]: s["snapshot_id"] for s in snapshots},
        "hub_source": hub_source, "nppes_enrichment": bool(nppes is not None and config.reference.enrich_from_nppes),
        "m_seeded_levels": sum(len(v) for v in (m_seed or {}).values()),
        "n_records": int(len(records)), "n_sources": len(source_names),
        "n_deterministic_pairs": int(len(det_pairs)), "n_predictions_above_review": int(len(predictions)),
        "n_pairs_rejected_by_pair_constraints": int(len(rejected)),
        "n_edges_dropped_by_hard_constraints": int(len(conflicting)),
        "n_edges": int(len(edges)), "n_edges_at_cluster_threshold": int((edges["match_probability"] >= th.cluster).sum()),
        "n_clusters": int(membership["cluster_id"].nunique()), "n_clusters_split": len(split_log),
        "n_review_queue": int(len(review)),
        "n_flagged_records": int((metrics["flags"].str.contains("oversized|low_density")).sum()),
        "n_labeled_pairs_stored": int(n_labeled),
        "n_rejected_sharing_valid_npi": n_rejected_sharing_npi,
        "lookup_effect": lookup_effect,
        "thresholds": th.model_dump(), "geo_available": prep.geo_available,
        "em_iterations": [t.get("iterations") for t in model.training_log if t.get("step") == "em"],
        "runtime_seconds": round(time.time() - t0, 2),
    }
    kb.record_run({"run_id": run_id, "engagement": eng, "profile": profile.name, "config_hash": _config_hash(config),
                   "knowledge_base_version": kb_version_before, "reference_snapshots": stats["reference_snapshots"],
                   "comparison_schema_hash": schema_hash, "stats": stats}, eng)
    return LinkageResult(
        config=config, prepared=prep, deterministic_pairs=det_pairs, predictions=predictions,
        rejected_pairs=rejected, conflicting_edges=conflicting, edges=edges, membership=membership,
        metrics=metrics, crosswalk=crosswalk, review_queue=review, model=model, split_log=split_log,
        stats=stats, matcher=matcher if keep_matcher else None, hub_source=hub_source,
    )
