import pandas as pd

from tether.matching.clustering import (
    ClusterQAConfig,
    apply_pair_constraints,
    build_edges,
    cluster_metrics,
    drop_conflicting_edges,
    split_violating_clusters,
)
from tether.profiles import HardConstraint, PairConstraint


def test_pair_constraint_rejects_no_name_agreement():
    pred = pd.DataFrame({"unique_id_l": ["x", "x", "x"], "unique_id_r": ["y", "z", "w"],
                         "match_probability": [0.99, 0.99, 0.99],
                         "gamma_name_first": [0, 2, -1], "gamma_name_last": [0, 0, 0]})
    ok = apply_pair_constraints(pred, [PairConstraint(("name_first", "name_last"))])
    assert ok.tolist() == [False, True, False]
    edges, rejected = build_edges(pred, pd.DataFrame(columns=["unique_id_l", "unique_id_r"]),
                                  [PairConstraint(("name_first", "name_last"))])
    assert len(edges) == 1 and len(rejected) == 2


def test_build_edges_merges_deterministic_and_probabilistic():
    pred = pd.DataFrame({"unique_id_l": ["b", "a"], "unique_id_r": ["a", "c"], "match_probability": [0.7, 0.8]})
    det = pd.DataFrame({"unique_id_l": ["a"], "unique_id_r": ["b"], "rule": ["npi"]})
    edges, _ = build_edges(pred, det, [])
    e = edges.set_index(["unique_id_l", "unique_id_r"])
    assert e.loc[("a", "b"), "match_probability"] == 1.0 and e.loc[("a", "b"), "link_source"] == "both"
    assert e.loc[("a", "c"), "link_source"] == "probabilistic"


def _records():
    return pd.DataFrame({"unique_id": list("abcde"), "npi_std": ["1", "1", None, "2", None]})


def test_drop_conflicting_edges():
    edges = pd.DataFrame({"unique_id_l": ["a", "a", "c"], "unique_id_r": ["b", "d", "d"],
                          "match_probability": [0.99, 0.99, 0.99], "link_source": "probabilistic"})
    kept, dropped = drop_conflicting_edges(edges, _records(), [HardConstraint("npi_std")])
    assert len(kept) == 2 and dropped.iloc[0]["unique_id_r"] == "d"


def test_split_transitive_conflict():
    # a(npi 1) - c(none) - d(npi 2): no direct conflicting edge, but one cluster holds two NPIs.
    edges = pd.DataFrame({"unique_id_l": ["a", "c"], "unique_id_r": ["c", "d"],
                          "match_probability": [0.99, 0.95], "link_source": "probabilistic"})
    membership = pd.DataFrame({"unique_id": ["a", "c", "d"], "cluster_id": ["k", "k", "k"]})
    out, log = split_violating_clusters(membership, edges, _records(), [HardConstraint("npi_std")])
    cid = out.set_index("unique_id")["cluster_id"]
    assert cid["a"] != cid["d"] and cid["c"] == cid["a"]  # c attaches to its strongest edge (a)
    assert log[0]["n_subclusters"] == 2


def test_cluster_metrics_confidence_density_flags():
    membership = pd.DataFrame({"unique_id": ["a", "b", "c", "d"], "cluster_id": ["k", "k", "k", "s"]})
    edges = pd.DataFrame({"unique_id_l": ["a", "b"], "unique_id_r": ["b", "c"],
                          "match_probability": [0.99, 0.96], "link_source": "probabilistic"})
    m = cluster_metrics(membership, edges, ClusterQAConfig(max_cluster_size=2, min_edge_density=0.9)).set_index("unique_id")
    assert m.loc["a", "cluster_size"] == 3 and abs(m.loc["a", "edge_density"] - 2 / 3) < 1e-9
    assert m.loc["a", "cluster_confidence"] == 0.96  # weakest attachment (c's best edge)
    assert m.loc["a", "flags"] == "oversized|low_density" and m.loc["d", "flags"] == "singleton"
    assert pd.isna(m.loc["d", "cluster_confidence"])
