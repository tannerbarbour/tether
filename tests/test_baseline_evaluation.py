import pandas as pd

from tether.matching.baseline import baseline_links, baseline_scores
from tether.reporting.evaluation import cluster_pairwise, pairwise_at_threshold, true_pairs_from_truth


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
    assert cluster_pairwise(cw, tp)["f1"] == 1.0
