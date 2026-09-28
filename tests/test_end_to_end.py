"""End-to-end: synthetic data -> mapped -> cleaned -> matched -> clustered -> evaluated."""

import logging
from pathlib import Path

import pandas as pd
import pytest
import yaml

from tether.config import EngagementConfig
from tether.matching.baseline import baseline_npi_name_scores, baseline_scores
from tether.matching.pipeline import run_linkage
from tether.reporting.evaluation import evaluate

EXAMPLE = Path(__file__).resolve().parents[1] / "engagements" / "example"


@pytest.fixture(scope="module")
def engagement(tmp_path_factory, small_dataset):
    logging.getLogger("splink").setLevel(logging.WARNING)
    root = tmp_path_factory.mktemp("eng")
    small_dataset.write(root / "data")
    for name in ("roster", "claims"):
        (root / "mappings").mkdir(exist_ok=True)
        (root / "mappings" / f"{name}.yaml").write_text((EXAMPLE / "mappings" / f"{name}.yaml").read_text())
    raw = yaml.safe_load((EXAMPLE / "config.yaml").read_text())
    raw["knowledge_base"]["path"] = str(root / "kb")
    (root / "config.yaml").write_text(yaml.safe_dump(raw))
    cfg = EngagementConfig.load(root / "config.yaml")
    return cfg, run_linkage(cfg), small_dataset


def test_pipeline_outputs_are_consistent(engagement):
    cfg, res, ds = engagement
    n = len(ds.source_a) + len(ds.source_b)
    assert len(res.crosswalk) == n and res.crosswalk["unique_id"].is_unique
    assert res.stats["n_deterministic_pairs"] > 0 and res.stats["n_edges"] > 0
    # hard constraint: no cluster holds two different valid NPIs
    recs = res.prepared.all_records.set_index("unique_id")["npi_std"]
    per_cluster = res.membership.assign(npi=res.membership["unique_id"].map(recs)).groupby("cluster_id")["npi"].nunique()
    assert per_cluster.max() <= 1
    # deterministic pairs always end up in the same cluster
    cid = res.membership.set_index("unique_id")["cluster_id"]
    assert (res.deterministic_pairs["unique_id_l"].map(cid) == res.deterministic_pairs["unique_id_r"].map(cid)).all()


def test_pipeline_beats_baseline(engagement):
    cfg, res, ds = engagement
    base = baseline_scores(res.prepared.all_records)
    base2 = baseline_npi_name_scores(res.prepared.all_records)
    report = evaluate(ds.truth, res.predictions, res.crosswalk, res.edges, deterministic_pairs=res.deterministic_pairs,
                      baselines={"baseline: name+org fuzzy": (base, 85.0), "baseline: NPI exact + name fuzzy": (base2, 85.0)},
                      auto_link=cfg.thresholds.auto_link, cluster_threshold=cfg.thresholds.cluster)
    assert report.deterministic["precision"] == 1.0
    pipe = report.method("pipeline pairwise")
    assert pipe.at_threshold["f1"] > 0.85
    assert pipe.best["f1"] > report.method("baseline: name+org fuzzy").best["f1"]
    assert pipe.nondeterministic_best["f1"] > report.method("baseline: NPI exact + name fuzzy").nondeterministic_best["f1"]
    assert report.cluster["precision"] > 0.95 and 0 < report.cluster["entity_exact_match_rate"] <= 1
    assert "Residual (non-deterministic)" in report.to_markdown()


def test_write_and_model_roundtrip(engagement, tmp_path):
    cfg, res, _ = engagement
    paths = res.write(tmp_path)
    assert paths["crosswalk"].exists() and paths["model"].exists()
    cw = pd.read_csv(paths["crosswalk"])
    assert {"source", "source_record_id", "entity_id", "cluster_confidence"} <= set(cw.columns)
    assert res.model.m_probabilities["name_last"]["exact last name"] > 0.5


def test_cli_run_and_evaluate(engagement, tmp_path, capsys):
    from tether.cli import main

    cfg, _, ds = engagement
    config_path = cfg.config_dir / "config.yaml"
    assert main(["run", "--config", str(config_path), "--out", str(tmp_path / "run")]) == 0
    assert (tmp_path / "run" / "crosswalk.csv").exists()
    truth_path = cfg.config_dir / "data" / "truth.csv"
    assert main(["evaluate", "--config", str(config_path), "--truth", str(truth_path), "--out", str(tmp_path / "ev"),
                 "--ablate-nppes"]) == 0
    assert (tmp_path / "ev" / "evaluation.md").exists() and (tmp_path / "ev" / "evaluation_report.html").exists()
    assert (tmp_path / "run" / "explanation_report.html").exists()
    html = (tmp_path / "ev" / "evaluation_report.html").read_text()
    assert "SYNTHETIC DATA" in html and "MOCK LLM" in html and "without NPPES enrichment" in html
    assert "same_npi_name_conflict" in html
    ex_html = (tmp_path / "run" / "explanation_report.html").read_text()
    assert "cdn.jsdelivr" not in ex_html and "vegaEmbed" in ex_html and len(ex_html) > 500_000
    assert "baseline" in capsys.readouterr().out


def test_cli_propose_and_standardize(engagement, tmp_path):
    from tether.cli import main

    cfg, _, _ = engagement
    config_path = cfg.config_dir / "config.yaml"
    assert main(["propose-mapping", "--config", str(config_path), "--out", str(tmp_path)]) == 0
    from tether.config import SchemaMapping

    m = SchemaMapping.load(tmp_path / "claims.yaml")
    assert m.status == "proposed" and m.as_dict()["prov_last_name"] == "name_last"
    assert main(["standardize-values", "--config", str(config_path)]) == 0
    assert (cfg.resolve(cfg.knowledge_base.path) / "lookup_org_aliases.csv").exists()
    assert (cfg.resolve(cfg.knowledge_base.path) / "llm_log" / "llm_calls.jsonl").exists()
