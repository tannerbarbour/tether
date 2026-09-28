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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tether", description="Explainable AI-assisted entity resolution")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate-synthetic", help="Generate synthetic two-source provider data with ground truth")
    gen.add_argument("--out", type=Path, required=True, help="Output directory")
    gen.add_argument("--settings", type=Path, help="YAML file of SyntheticConfig overrides")
    gen.add_argument("--n-entities", type=int, dest="n_entities")
    gen.add_argument("--seed", type=int)
    gen.set_defaults(func=_cmd_generate_synthetic)

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
