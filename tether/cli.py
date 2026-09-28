"""``tether`` command-line interface.

Subcommands are added phase by phase; each is fully functional when present.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_generate_synthetic(args: argparse.Namespace) -> int:
    import yaml

    from tether.synthetic import SyntheticConfig, generate

    overrides = {}
    if args.settings:
        with open(args.settings, encoding="utf-8") as fh:
            overrides = yaml.safe_load(fh) or {}
    if args.n_entities is not None:
        overrides["n_entities"] = args.n_entities
    if args.seed is not None:
        overrides["seed"] = args.seed
    cfg = SyntheticConfig.model_validate(overrides)
    ds = generate(cfg)
    paths = ds.write(args.out)
    print(f"generated {len(ds.entities)} entities -> {len(ds.source_a)} {cfg.source_a_name} rows, "
          f"{len(ds.source_b)} {cfg.source_b_name} rows, {len(ds.nppes)} NPPES rows")
    for name, p in paths.items():
        print(f"  {name:>14}: {p}")
    return 0


def _quiet_splink() -> None:
    import logging

    logging.getLogger("splink").setLevel(logging.WARNING)


def _cmd_run(args: argparse.Namespace) -> int:
    from tether.config import EngagementConfig
    from tether.matching.pipeline import run_linkage

    _quiet_splink()
    cfg = EngagementConfig.load(args.config)
    result = run_linkage(cfg, keep_matcher=True)
    out_dir = Path(args.out or cfg.resolve(cfg.output.dir))
    paths = result.write(out_dir)
    from tether.reporting.explanation import render_explanation_report

    paths["explanation"] = render_explanation_report(result, out_dir / "explanation_report.html")
    s = result.stats
    print(f"linked {s['n_records']} records from {s['n_sources']} source(s) into {s['n_clusters']} entities "
          f"in {s['runtime_seconds']}s")
    print(f"  deterministic pairs: {s['n_deterministic_pairs']}   probabilistic edges: {s['n_edges']}   "
          f"review queue: {s['n_review_queue']}   flagged records: {s['n_flagged_records']}")
    for name in ("crosswalk", "review_queue", "model", "stats", "explanation"):
        print(f"  {name:>12}: {paths[name]}")
    print(f"  knowledge base version now: {result.stats['knowledge_base_version']} -> "
          f"{__import__('tether.knowledge', fromlist=['open_knowledge_base']).open_knowledge_base(cfg).version()}")
    return 0


def _evaluate_config(cfg, truth, baseline_threshold: float, ablate_nppes: bool, tuning=None, kb_write: bool = False):
    from tether.matching.baseline import baseline_npi_name_scores, baseline_scores
    from tether.matching.pipeline import run_linkage
    from tether.reporting.evaluation import cluster_metrics_vs_truth, evaluate

    result = run_linkage(cfg, kb_write=kb_write)
    records = result.prepared.all_records
    records = records[records["source_dataset"].isin(set(truth["source"]))]
    baselines = {
        "baseline: name+org fuzzy": (baseline_scores(records), baseline_threshold),
        "baseline: NPI exact + name fuzzy": (baseline_npi_name_scores(records), baseline_threshold),
    }
    report = evaluate(truth, result.predictions, result.crosswalk, result.edges, deterministic_pairs=result.deterministic_pairs,
                      baselines=baselines, auto_link=cfg.thresholds.auto_link, cluster_threshold=cfg.thresholds.cluster,
                      review_queue=result.review_queue)
    if tuning is not None:
        report.extras["tuning"] = tuning.as_dict()
    label = "with NPPES Type 2 invalidation" if result.stats["nppes_enrichment"] else "without NPPES enrichment"
    report.ablations.append({"name": f"pipeline crosswalk ({label})", "threshold": cfg.thresholds.cluster,
                             **{k: report.cluster[k] for k in ("precision", "recall", "f1", "tp", "fp", "fn")},
                             "entity_exact_match_rate": report.cluster["entity_exact_match_rate"],
                             "n_deterministic_pairs": result.stats["n_deterministic_pairs"],
                             "n_edges_dropped_by_hard_constraints": result.stats["n_edges_dropped_by_hard_constraints"],
                             "n_clusters_split": result.stats["n_clusters_split"]})
    if ablate_nppes and result.stats["nppes_enrichment"]:
        alt = cfg.model_copy(deep=True)
        alt.reference.enrich_from_nppes = False
        alt_result = run_linkage(alt, kb_write=False)
        cw = alt_result.crosswalk[alt_result.crosswalk["source"].isin(set(truth["source"]))]
        cm = cluster_metrics_vs_truth(cw, truth)
        report.ablations.append({"name": "pipeline crosswalk (without NPPES enrichment)", "threshold": cfg.thresholds.cluster,
                                 **{k: cm[k] for k in ("precision", "recall", "f1", "tp", "fp", "fn")},
                                 "entity_exact_match_rate": cm["entity_exact_match_rate"],
                                 "n_deterministic_pairs": alt_result.stats["n_deterministic_pairs"],
                                 "n_edges_dropped_by_hard_constraints": alt_result.stats["n_edges_dropped_by_hard_constraints"],
                                 "n_clusters_split": alt_result.stats["n_clusters_split"]})
    return result, report


def _cmd_evaluate(args: argparse.Namespace) -> int:
    import json

    import pandas as pd

    from tether.config import EngagementConfig
    from tether.reporting.evaluation_report import render_evaluation_report

    _quiet_splink()
    cfg = EngagementConfig.load(args.config)
    truth = pd.read_csv(args.truth, dtype=str)
    out_dir = Path(args.out or cfg.resolve(cfg.output.dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    tuning = None
    if args.tune_seed is not None:
        from tether.matching.tune import tune_thresholds

        tuning = tune_thresholds(cfg, args.tune_seed, args.tune_entities, out_dir / f"tuning_seed{args.tune_seed}")
        tuning.save(out_dir / "tuned_thresholds.yaml")
        cfg.thresholds = tuning.thresholds(cfg.thresholds.review_lower)
        print(f"tuned on seed {args.tune_seed}: auto_link={tuning.auto_link} cluster={tuning.cluster}")
    result, report = _evaluate_config(cfg, truth, args.baseline_threshold, args.ablate_nppes, tuning, kb_write=args.write_kb)
    result.write(out_dir)
    (out_dir / "evaluation.md").write_text(report.to_markdown(), encoding="utf-8")
    for m in report.methods:
        m.curve.to_csv(out_dir / f"pr_curve_{m.name.replace(' ', '_').replace(':', '').replace('+', 'plus')}.csv", index=False)
    (out_dir / "evaluation.json").write_text(json.dumps({
        "data": "synthetic", "llm_provider": cfg.llm.provider,
        "methods": {m.name: {"at_threshold": m.at_threshold, "best": m.best, "nondeterministic_at_threshold": m.nondeterministic_at_threshold,
                             "nondeterministic_best": m.nondeterministic_best} for m in report.methods},
        "cluster": report.cluster, "blocking": report.blocking, "deterministic": report.deterministic,
        "ablations": report.ablations, "tuning": report.extras.get("tuning"), "run_stats": result.stats,
    }, indent=2, default=str), encoding="utf-8")
    html_path = render_evaluation_report(report, result.stats, out_dir / "evaluation_report.html", cfg.engagement.id, cfg.llm.provider)
    print(report.to_markdown())
    if report.ablations:
        print(pd.DataFrame(report.ablations).to_markdown(index=False))
    print(f"\nwritten: {out_dir / 'evaluation.md'} and {html_path}")
    return 0


def _cmd_tune(args: argparse.Namespace) -> int:
    from tether.config import EngagementConfig
    from tether.matching.tune import tune_thresholds

    _quiet_splink()
    cfg = EngagementConfig.load(args.config)
    out_dir = Path(args.out or cfg.resolve(cfg.output.dir))
    t = tune_thresholds(cfg, args.seed, args.n_entities, out_dir / f"tuning_seed{args.seed}")
    path = t.save(out_dir / "tuned_thresholds.yaml")
    print(f"tuned on synthetic seed {args.seed} ({args.n_entities} entities): auto_link={t.auto_link} (cost-minimising; "
          f"F1-max would be {t.f1_best_auto_link}) cluster={t.cluster}")
    print(t.auto_link_sweep[["threshold", "precision", "recall", "f1", "queue_size", "false_auto_links", "expected_cost_minutes"]].to_string(index=False))
    print(f"written: {path}")
    return 0


def _cmd_kb(args: argparse.Namespace) -> int:
    from tether.config import EngagementConfig
    from tether.knowledge import LocalKnowledgeBase, open_knowledge_base

    kb = open_knowledge_base(EngagementConfig.load(args.config)) if args.config else LocalKnowledgeBase(args.path)
    if args.kb_command == "status":
        print(f"knowledge base: {kb.path}  version {kb.version()}")
        print(kb.status().to_string(index=False))
    elif args.kb_command == "list":
        df = kb.read(args.table)
        cols = [c for c in df.columns if c not in ("mapping_json",)]
        print(df[cols].tail(args.limit).to_string(index=False) if len(df) else "(empty)")
    elif args.kb_command == "promote":
        n = kb.promote(args.table, to_state=args.to, reviewer=args.reviewer, engagement=args.engagement,
                       asset_ids=args.asset_id, reuse_scope=args.scope)
        print(f"{n} asset(s) in {args.table} -> {args.to}" + (f" ({args.scope})" if args.scope else ""))
    return 0


def _cmd_propose_mapping(args: argparse.Namespace) -> int:
    from tether.config import EngagementConfig, SchemaMapping
    from tether.ingestion import LLMCallLogger, make_client, propose_mapping, read_table
    from tether.ingestion.lookups import llm_log_dir
    from tether.profiles import get_profile

    cfg = EngagementConfig.load(args.config)
    profile = get_profile(cfg.profile)
    client = make_client(cfg.llm)
    logger = LLMCallLogger(llm_log_dir(cfg))
    kb = Path(cfg.resolve(cfg.knowledge_base.path))
    examples = []
    for p in sorted((kb / "schema_mappings").glob("*.yaml")) if (kb / "schema_mappings").exists() else []:
        m = SchemaMapping.load(p)
        if m.status == "approved" and m.profile == profile.name:
            examples.append(m)
    wanted = [s for s in cfg.sources if not args.source or s.name in args.source]
    for src in wanted:
        raw = read_table(cfg.resolve(src.path), src.format, src.encoding)
        proposal = propose_mapping(
            raw, src.name, profile, client, record_id_column=src.record_id_column, examples=examples,
            sample_rows=cfg.llm.sample_rows, redact_columns=cfg.llm.redact_columns, logger=logger,
        )
        out = Path(args.out) / f"{src.name}.yaml" if args.out else (
            cfg.resolve(src.mapping) if src.mapping else cfg.config_dir / "mappings" / f"{src.name}.yaml")
        if out.exists() and not args.overwrite:
            out = out.with_name(out.stem + ".proposed.yaml")
        proposal.save(out)
        n_llm = sum(1 for c in proposal.columns if c.rationale and c.rationale.startswith("["))
        print(f"{src.name}: {len(proposal.columns)} columns mapped ({n_llm} via {client.provider}), "
              f"{len(proposal.unmapped)} unmapped -> {out}")
        for c in proposal.columns:
            print(f"    {c.source_column!r:>28} -> {c.target:<16} {c.confidence:.2f}  {c.rationale or ''}")
        if proposal.unmapped:
            print(f"    unmapped: {proposal.unmapped}")
    print("review each file, set status: approved, then run `tether run`.")
    return 0


def _cmd_standardize_values(args: argparse.Namespace) -> int:
    from tether.config import EngagementConfig
    from tether.ingestion import LLMCallLogger, load_source, make_client, standardize_values
    from tether.ingestion.lookups import llm_log_dir, open_lookup
    from tether.profiles import get_profile

    cfg = EngagementConfig.load(args.config)
    profile = get_profile(cfg.profile)
    client = make_client(cfg.llm)
    logger = LLMCallLogger(llm_log_dir(cfg))
    frames = [load_source(s, cfg, profile) for s in cfg.sources]
    kinds = {"job_title": ("job_title", cfg.llm.standardize_titles), "org_alias": ("org_name", cfg.llm.standardize_org_aliases)}
    for kind, (field_type, enabled) in kinds.items():
        if not enabled:
            continue
        fields = [n for n, s in profile.fields.items() if s.field_type == field_type]
        values = [v for df in frames for f in fields for v in df[f].dropna().tolist()]
        cache = open_lookup(cfg, kind)
        stats = standardize_values(values, kind, client, cache, batch_size=cfg.llm.max_distinct_values_per_call, logger=logger)
        path = cache.save()
        print(f"{kind}: {stats['n_distinct']} distinct values, {stats['n_cached']} cached, {stats['n_builtin']} built-in, "
              f"{stats['n_llm']} from {client.provider} in {stats['calls']} call(s), {stats['n_rejected']} rejected -> {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tether", description="Explainable AI-assisted entity resolution")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate-synthetic", help="Generate synthetic two-source provider data with ground truth")
    gen.add_argument("--out", type=Path, required=True, help="Output directory")
    gen.add_argument("--settings", type=Path, help="YAML file of SyntheticConfig overrides")
    gen.add_argument("--n-entities", type=int, dest="n_entities")
    gen.add_argument("--seed", type=int)
    gen.set_defaults(func=_cmd_generate_synthetic)

    pm = sub.add_parser("propose-mapping", help="Propose canonical column mappings for the configured sources")
    pm.add_argument("--config", type=Path, required=True)
    pm.add_argument("--source", action="append", help="Limit to a source name (repeatable)")
    pm.add_argument("--out", type=Path, help="Directory for proposed YAML files (default: the config's mapping paths)")
    pm.add_argument("--overwrite", action="store_true", help="Overwrite an existing mapping file")
    pm.set_defaults(func=_cmd_propose_mapping)

    sv = sub.add_parser("standardize-values", help="Standardize distinct job titles / org aliases into lookup tables")
    sv.add_argument("--config", type=Path, required=True)
    sv.set_defaults(func=_cmd_standardize_values)

    run = sub.add_parser("run", help="Run the linkage pipeline for an engagement config")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--out", type=Path, help="Override output directory")
    run.set_defaults(func=_cmd_run)

    ev = sub.add_parser("evaluate", help="Run the pipeline and the baseline; compare both against ground truth")
    ev.add_argument("--config", type=Path, required=True)
    ev.add_argument("--truth", type=Path, required=True, help="CSV with source, source_record_id, entity_id")
    ev.add_argument("--out", type=Path)
    ev.add_argument("--baseline-threshold", type=float, default=85.0, dest="baseline_threshold")
    ev.add_argument("--tune-seed", type=int, dest="tune_seed", help="Tune thresholds on a separate synthetic seed first")
    ev.add_argument("--tune-entities", type=int, default=1000, dest="tune_entities")
    ev.add_argument("--ablate-nppes", action="store_true", dest="ablate_nppes", help="Also run without NPPES enrichment")
    ev.add_argument("--write-kb", action="store_true", dest="write_kb",
                    help="Allow the evaluation run to write model params / run history to the knowledge base (default: read-only)")
    ev.set_defaults(func=_cmd_evaluate)

    tu = sub.add_parser("tune", help="Tune auto_link / cluster thresholds on a separate synthetic seed")
    tu.add_argument("--config", type=Path, required=True)
    tu.add_argument("--seed", type=int, required=True)
    tu.add_argument("--n-entities", type=int, default=1000, dest="n_entities")
    tu.add_argument("--out", type=Path)
    tu.set_defaults(func=_cmd_tune)

    kb = sub.add_parser("kb", help="Inspect and promote knowledge-base assets")
    loc = argparse.ArgumentParser(add_help=False)
    loc.add_argument("--config", type=Path, help="Engagement config (locates the knowledge base)")
    loc.add_argument("--path", type=Path, default=Path("knowledge_base"), help="Knowledge base directory if no config")
    kbs = kb.add_subparsers(dest="kb_command", required=True)
    kbs.add_parser("status", parents=[loc], help="Row counts by table and promotion state")
    kl = kbs.add_parser("list", parents=[loc], help="Show rows of a table")
    kl.add_argument("--table", required=True)
    kl.add_argument("--limit", type=int, default=20)
    kp = kbs.add_parser("promote", parents=[loc], help="Advance assets to reviewed / promoted (requires --reviewer)")
    kp.add_argument("--table", required=True)
    kp.add_argument("--to", choices=["reviewed", "promoted"], default="reviewed")
    kp.add_argument("--reviewer", required=True)
    kp.add_argument("--engagement", help="Promote every eligible asset of this engagement")
    kp.add_argument("--asset-id", action="append", dest="asset_id", help="Promote specific asset ids (repeatable)")
    kp.add_argument("--scope", choices=["generic", "client_scoped"], help="Also set reuse scope")
    kb.set_defaults(func=_cmd_kb)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
