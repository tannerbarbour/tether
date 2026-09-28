import pandas as pd

from tether.matching.baseline import baseline_links, baseline_npi_name_scores, baseline_scores
from tether.reporting.evaluation import cluster_metrics_vs_truth, pairwise_at_threshold, true_pairs_from_truth


def test_baseline_scores_all_pairs_and_cross_only():
    recs = pd.DataFrame({
        "unique_id": ["a:1", "a:2", "b:1"], "source_dataset": ["a", "a", "b"],
        "name": ["Jane Doe", None, None], "name_first": [None, "John", "Jane"], "name_last": [None, "Roe", "Doe"],
        "org": ["Mercy Hospital", "Mercy Hospital", "Mercy Hosp"],
    })
    all_pairs = baseline_scores(recs)
    assert len(all_pairs) == 3
    cross = baseline_scores(recs, cross_source_only=True)
    assert len(cross) == 2
    top = all_pairs.sort_values("score", ascending=False).iloc[0]
    assert {top["unique_id_l"], top["unique_id_r"]} == {"a:1", "b:1"}
    assert len(baseline_links(all_pairs, 101)) == 0


def test_metrics_math():
    truth = pd.DataFrame({"source": ["a", "b", "b"], "source_record_id": ["1", "1", "2"], "entity_id": ["E", "E", "F"]})
    tp = true_pairs_from_truth(truth)
    assert tp == {"a:1||b:1"}
    scored = pd.DataFrame({"unique_id_l": ["b:1", "b:2"], "unique_id_r": ["a:1", "a:1"], "s": [0.9, 0.9]})
    m = pairwise_at_threshold(scored, "s", 0.5, tp)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 0) and m["precision"] == 0.5
    cw = pd.DataFrame({"source": ["a", "b", "b"], "source_record_id": ["1", "1", "2"], "entity_id": ["X", "X", "Y"]})
    cm = cluster_metrics_vs_truth(cw, truth)
    assert cm["f1"] == 1.0 and cm["entity_exact_match_rate"] == 1.0 and cm["entities_split"] == 0
    # exclusion of deterministic pairs removes them from truth and predictions alike
    nd = pairwise_at_threshold(scored, "s", 0.5, tp, exclude={"a:1||b:1"})
    assert (nd["tp"], nd["fp"], nd["fn"]) == (0, 1, 0)


def test_npi_name_baseline_trusts_equal_npi():
    recs = pd.DataFrame({"unique_id": ["a:1", "b:1", "b:2"], "source_dataset": ["a", "b", "b"],
                         "name": ["Jane Doe", "Xavier Quinn", "Jane Doe"], "npi_digits": ["1234567893", "1234567893", None]})
    s = baseline_npi_name_scores(recs).set_index(["unique_id_l", "unique_id_r"])["score"]
    assert s[("a:1", "b:1")] == 100.0 and s[("a:1", "b:2")] == 100.0 and s[("b:1", "b:2")] < 50
