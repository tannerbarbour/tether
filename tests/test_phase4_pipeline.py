"""Knowledge-base learning across engagements, hub linking, tuning, ablation, reports."""

import logging
from pathlib import Path

import pytest
import yaml

from tether.config import EngagementConfig
from tether.knowledge import LocalKnowledgeBase
from tether.matching.pipeline import run_linkage
from tether.synthetic import SyntheticConfig, generate

EXAMPLE = Path(__file__).resolve().parents[1] / "engagements" / "example"


def _engagement(root: Path, seed: int, eng_id: str, kb_path: Path, n: int = 120, **reference) -> EngagementConfig:
    logging.getLogger("splink").setLevel(logging.WARNING)
    ds = generate(SyntheticConfig(n_entities=n, n_orgs=10, seed=seed, nppes_distractors=10))
    ds.write(root / "data")
    (root / "mappings").mkdir(exist_ok=True)
    for name in ("roster", "claims"):
        (root / "mappings" / f"{name}.yaml").write_text((EXAMPLE / "mappings" / f"{name}.yaml").read_text())
    raw = yaml.safe_load((EXAMPLE / "config.yaml").read_text())
    raw["engagement"]["id"] = eng_id
    raw["knowledge_base"]["path"] = str(kb_path)
    raw["reference"].update(reference)
    (root / "config.yaml").write_text(yaml.safe_dump(raw))
    return EngagementConfig.load(root / "config.yaml")


def test_learning_across_engagements_and_provenance(tmp_path):
    kb_path = tmp_path / "kb"
    cfg_a = _engagement(tmp_path / "a", 11, "eng_a", kb_path)
    res_a = run_linkage(cfg_a)
    assert res_a.stats["m_seeded_levels"] == 0 and res_a.stats["n_labeled_pairs_stored"] > 0
    kb = LocalKnowledgeBase(kb_path)
    assert res_a.stats["knowledge_base_version"] != kb.version()  # run wrote assets
    assert set(kb.read("reference_snapshots")["source"]) == {"zip_centroids", "nppes"}
    assert len(kb.read("schema_mappings")) == 2
    # engagement B sees nothing until A's params are promoted as generic
    cfg_b = _engagement(tmp_path / "b", 12, "eng_b", kb_path)
    assert run_linkage(cfg_b).stats["m_seeded_levels"] == 0
    kb.promote("model_params", to_state="promoted", reviewer="lead", engagement="eng_a", reuse_scope="generic")
    res_b = run_linkage(cfg_b)
    assert res_b.stats["m_seeded_levels"] > 0
    assert res_b.stats["em_iterations"] and all(isinstance(i, int) for i in res_b.stats["em_iterations"])
    hist = kb.read("run_history")
    assert len(hist) == 3 and set(hist["engagement"]) == {"eng_a", "eng_b"}


def test_hub_linking_assigns_npi_entity_ids(tmp_path):
    cfg = _engagement(tmp_path / "h", 13, "eng_h", tmp_path / "kb", hub_link_via_nppes=True)
    res = run_linkage(cfg)
    assert res.hub_source == "nppes" and res.stats["n_sources"] == 3
    cw = res.crosswalk
    assert cw["is_reference"].any()
    npi_entities = cw[cw["entity_id"].str.startswith("NPI-")]
    assert len(npi_entities) > 0
    # source records with a valid NPI present in NPPES resolve to that NPI's entity
    recs = res.prepared.all_records.set_index("unique_id")
    linked = cw[(cw["source"] != "nppes") & cw["entity_id"].str.startswith("NPI-")]
    ok = [recs.loc[u, "npi_std"] in (None, e[4:]) or recs.loc[u, "npi_std"] != recs.loc[u, "npi_std"]
          for u, e in zip(linked["unique_id"], linked["entity_id"])]
    assert all(ok)


def test_nppes_ablation_recovers_recall(tmp_path):
    from tether.reporting.evaluation import cluster_metrics_vs_truth

    kb = tmp_path / "kb"
    with_cfg = _engagement(tmp_path / "w", 14, "eng_w", kb, n=200)
    without_cfg = _engagement(tmp_path / "wo", 14, "eng_wo", kb, n=200, enrich_from_nppes=False)
    import pandas as pd

    truth = pd.read_csv(tmp_path / "w" / "data" / "truth.csv", dtype=str)
    r_with, r_without = run_linkage(with_cfg), run_linkage(without_cfg)
    m_with = cluster_metrics_vs_truth(r_with.crosswalk, truth)
    m_without = cluster_metrics_vs_truth(r_without.crosswalk, truth)
    assert r_with.stats["n_edges_dropped_by_hard_constraints"] < r_without.stats["n_edges_dropped_by_hard_constraints"]
    assert m_with["recall"] >= m_without["recall"]
    assert "n_rejected_sharing_valid_npi" in r_with.stats and "lookup_effect" in r_with.stats


def test_tuning_on_separate_seed(tmp_path):
    from tether.matching.tune import tune_thresholds

    cfg = _engagement(tmp_path / "t", 15, "eng_t", tmp_path / "kb")
    t = tune_thresholds(cfg, seed=99, n_entities=120, workdir=tmp_path / "tune")
    assert t.tuning_seed == 99 and 0.5 < t.auto_link <= t.cluster < 1
    assert {"queue_size", "false_auto_links", "expected_cost_minutes"} <= set(t.auto_link_sweep.columns)
    th = t.thresholds(cfg.thresholds.review_lower)
    assert th.cluster >= th.auto_link
    assert t.save(tmp_path / "tuned.yaml").exists()
    assert not (tmp_path / "kb" / "model_params.csv").exists() or "eng_t-tune99" not in set(
        LocalKnowledgeBase(tmp_path / "kb").read("model_params")["engagement"])


def test_explanation_report_contains_charts(tmp_path):
    from tether.reporting.explanation import parameter_table, render_explanation_report

    cfg = _engagement(tmp_path / "x", 16, "eng_x", tmp_path / "kb")
    res = run_linkage(cfg, keep_matcher=True)
    out = render_explanation_report(res, tmp_path / "explain.html")
    html = out.read_text()
    assert "vegaEmbed" in html and "Match weights" in html and "Cluster QA" in html and res.stats["run_id"] in html
    pt = parameter_table(res.model.settings_dict)
    assert {"comparison", "level", "m", "u", "match_weight_bits"} <= set(pt.columns) and len(pt) > 10


def test_kb_cli(tmp_path, capsys):
    from tether.cli import main

    cfg = _engagement(tmp_path / "c", 17, "eng_c", tmp_path / "kb")
    run_linkage(cfg)
    config_path = str(cfg.config_dir / "config.yaml")
    assert main(["kb", "status", "--config", config_path]) == 0
    assert "model_params" in capsys.readouterr().out
    assert main(["kb", "promote", "--config", config_path, "--table", "model_params", "--to", "promoted",
                 "--reviewer", "lead", "--engagement", "eng_c", "--scope", "generic"]) == 0
    assert "asset(s)" in capsys.readouterr().out
    assert main(["kb", "list", "--config", config_path, "--table", "run_history", "--limit", "5"]) == 0
    with pytest.raises(SystemExit):
        main(["kb", "promote", "--config", config_path, "--table", "model_params"])
