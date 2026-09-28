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
    result = run_linkage(cfg)
    out_dir = args.out or cfg.resolve(cfg.output.dir)
    paths = result.write(out_dir)
    s = result.stats
    print(f"linked {s['n_records']} records from {s['n_sources']} source(s) into {s['n_clusters']} entities "
          f"in {s['runtime_seconds']}s")
    print(f"  deterministic pairs: {s['n_deterministic_pairs']}   probabilistic edges: {s['n_edges']}   "
          f"review queue: {s['n_review_queue']}   flagged records: {s['n_flagged_records']}")
    for name in ("crosswalk", "review_queue", "model", "stats"):
        print(f"  {name:>12}: {paths[name]}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    import json

    import pandas as pd

    from tether.config import EngagementConfig
    from tether.matching.baseline import baseline_scores
    from tether.matching.pipeline import run_linkage
    from tether.reporting.evaluation import evaluate

    _quiet_splink()
    cfg = EngagementConfig.load(args.config)
    truth = pd.read_csv(args.truth, dtype=str)
    result = run_linkage(cfg)
    out_dir = Path(args.out or cfg.resolve(cfg.output.dir))
    result.write(out_dir)
    baseline = baseline_scores(result.prepared.all_records)
    report = evaluate(truth, result.predictions, result.crosswalk, result.edges,
                      deterministic_pairs=result.deterministic_pairs, baseline=baseline,
                      auto_link=cfg.thresholds.auto_link, cluster_threshold=cfg.thresholds.cluster,
                      baseline_threshold=args.baseline_threshold)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "evaluation.md").write_text(report.to_markdown(), encoding="utf-8")
    report.pipeline_pr_curve.to_csv(out_dir / "pr_curve_pipeline.csv", index=False)
    report.baseline_pr_curve.to_csv(out_dir / "pr_curve_baseline.csv", index=False)
    (out_dir / "evaluation.json").write_text(json.dumps({
        "pipeline_pairwise": report.pipeline_pairwise, "pipeline_cluster": report.pipeline_cluster,
        "baseline_pairwise": report.baseline_pairwise, "baseline_best": report.baseline_best,
        "blocking": report.blocking, "deterministic": report.deterministic, "run_stats": result.stats,
    }, indent=2, default=str), encoding="utf-8")
    print(report.to_markdown())
    print(f"\nwritten: {out_dir / 'evaluation.md'}")
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
    ev.set_defaults(func=_cmd_evaluate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
