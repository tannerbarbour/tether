"""``ecg-linkage`` command-line interface.

Subcommands are added phase by phase; each is fully functional when present.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_generate_synthetic(args: argparse.Namespace) -> int:
    import yaml

    from ecg_linkage.synthetic import SyntheticConfig, generate

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ecg-linkage", description="Explainable AI-assisted entity resolution")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate-synthetic", help="Generate synthetic two-source provider data with ground truth")
    gen.add_argument("--out", type=Path, required=True, help="Output directory")
    gen.add_argument("--settings", type=Path, help="YAML file of SyntheticConfig overrides")
    gen.add_argument("--n-entities", type=int, dest="n_entities")
    gen.add_argument("--seed", type=int)
    gen.set_defaults(func=_cmd_generate_synthetic)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
