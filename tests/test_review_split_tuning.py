import logging

import pandas as pd

from tether.matching.clustering import split_violating_clusters
from tether.profiles import HardConstraint
from tether.reporting.crosswalk import build_review_queue
from tether.reporting.evaluation import review_queue_truth


def _recs(vals):
    return pd.DataFrame({"unique_id": list(vals), "npi_std": list(vals.values())})


def test_split_reaches_multi_hop_members_and_breaks_ties_deterministically():
    # a(npi 1) -0.9- b -0.8- c -0.7- e -0.6- d(npi 2). Pass 1: b joins a, e joins d (direct seed edges).
    # Pass 2: c has no seed edge; it compares b (0.8, a's cluster) with e (0.7, d's cluster) -> a's cluster.
    recs = _recs({"a": "1", "b": None, "c": None, "d": "2", "e": None})
    edges = pd.DataFrame({"unique_id_l": ["a", "b", "c", "e"], "unique_id_r": ["b", "c", "e", "d"],
                          "match_probability": [0.9, 0.8, 0.7, 0.6], "link_source": "probabilistic"})
    membership = pd.DataFrame({"unique_id": list("abcde"), "cluster_id": ["k"] * 5})
    out, log = split_violating_clusters(membership, edges, recs, [HardConstraint("npi_std")])
    cid = out.set_index("unique_id")["cluster_id"]
    assert cid["b"] == cid["a"] and cid["c"] == cid["a"] and cid["e"] == cid["d"] and cid["a"] != cid["d"]
    assert log[0]["n_multi_hop_assignments"] == 1 and log[0]["n_singletons"] == 0 and log[0]["passes"] >= 2
    # tie: f connects equally to both seeds -> deterministic toward the seed value that sorts first ("1")
    recs2 = _recs({"a": "1", "d": "2", "f": None})
    edges2 = pd.DataFrame({"unique_id_l": ["a", "d"], "unique_id_r": ["f", "f"], "match_probability": [0.9, 0.9],
                           "link_source": "probabilistic"})
    m2 = pd.DataFrame({"unique_id": ["a", "d", "f"], "cluster_id": ["k"] * 3})
    out2, log2 = split_violating_clusters(m2, edges2, recs2, [HardConstraint("npi_std")])
    c2 = out2.set_index("unique_id")["cluster_id"]
    assert c2["f"] == c2["a"] and log2[0]["n_ties"] == 1
    # unreachable member -> singleton
    recs3 = _recs({"a": "1", "d": "2", "g": None})
    m3 = pd.DataFrame({"unique_id": ["a", "d", "g"], "cluster_id": ["k"] * 3})
    out3, log3 = split_violating_clusters(m3, edges2.iloc[0:0], recs3, [HardConstraint("npi_std")])
    assert out3.set_index("unique_id")["cluster_id"].nunique() == 3 and log3[0]["n_singletons"] == 1


def test_review_queue_reasons_and_truth():
    pred = pd.DataFrame({"unique_id_l": ["a:1", "a:2"], "unique_id_r": ["b:1", "b:2"], "match_probability": [0.7, 0.95],
                         "match_weight": [1.0, 5.0]})
    rejected = pd.DataFrame({"unique_id_l": ["a:3", "a:4", "a:5"], "unique_id_r": ["b:3", "b:4", "b:5"],
                             "match_probability": [0.6, 0.55, 0.52], "match_weight": [0.5, 0.3, 0.1],
                             "npi_std_l": ["1", None, "3"], "npi_std_r": ["1", "2", "4"]})
    q = build_review_queue(pred, 0.5, 0.9, [], rejected=rejected, npi_column="npi_std", audit_sample=1, seed=0)
    assert q["review_reason"].value_counts().to_dict() == {"score_band": 1, "same_npi_name_conflict": 1, "rejected_audit_sample": 1}
    assert q.loc[q.review_reason == "same_npi_name_conflict", "unique_id_l"].iloc[0] == "a:3"
    truth = pd.DataFrame({"source": ["a", "b", "a", "b", "a", "b", "a", "b"], "source_record_id": ["1", "1", "3", "3", "4", "4", "5", "5"],
                          "entity_id": ["E1", "E1", "E3", "E3", "E4", "E5", "E6", "E7"]})
    t = review_queue_truth(q, truth).set_index("review_reason")
    assert t.loc["score_band", "true_share"] == 1.0 and t.loc["same_npi_name_conflict", "true_matches"] == 1


def test_em_iterations_log_and_records_agree(small_dataset, tmp_path):
    import yaml
    from pathlib import Path

    from tether.config import EngagementConfig
    from tether.matching.pipeline import run_linkage

    logging.getLogger("splink").setLevel(logging.INFO)
    ex = Path(__file__).resolve().parents[1] / "engagements" / "example"
    small_dataset.write(tmp_path / "data")
    (tmp_path / "mappings").mkdir()
    for n in ("roster", "claims"):
        (tmp_path / "mappings" / f"{n}.yaml").write_text((ex / "mappings" / f"{n}.yaml").read_text())
    raw = yaml.safe_load((ex / "config.yaml").read_text())
    raw["knowledge_base"]["path"] = str(tmp_path / "kb")
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(raw))
    cfg = EngagementConfig.load(tmp_path / "config.yaml")
    res = run_linkage(cfg, kb_write=False)
    em = [t for t in res.model.training_log if t["step"] == "em"]
    assert em and all(t["converged"] for t in em)
    assert all(t["iterations_from_log"] == t["iterations_from_records"] == t["iterations"] for t in em)
    assert not (tmp_path / "kb" / "run_history.csv").exists() and res.stats["kb_write"] is False


def test_cost_based_auto_link_sweep():
    from tether.matching.tune import CostModel, auto_link_sweep

    edges = pd.DataFrame({"unique_id_l": ["a:1", "a:2", "a:3", "a:4"], "unique_id_r": ["b:1", "b:2", "b:3", "b:4"],
                          "match_probability": [0.99, 0.8, 0.7, 0.6]})
    truth_pairs = {"a:1||b:1", "a:3||b:3", "a:9||b:9"}
    sw = auto_link_sweep(edges, truth_pairs, 0.5, [0.65, 0.75, 0.9], CostModel(1.0, 30.0, 10.0))
    r = sw.set_index("threshold")
    assert r.loc[0.9, "false_auto_links"] == 0 and r.loc[0.9, "queue_size"] == 3
    assert r.loc[0.65, "false_auto_links"] == 1 and r.loc[0.65, "queue_size"] == 1
    assert (r["missed_below_review"] == 1).all()
    assert r.loc[0.9, "expected_cost_minutes"] == 3 + 10 and r.loc[0.65, "expected_cost_minutes"] == 30 + 1 + 10
