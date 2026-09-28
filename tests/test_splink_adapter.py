import pandas as pd
import pytest

from tether.config import MatchingConfig
from tether.matching.splink_adapter import (
    SplinkMatcher,
    apply_m_probabilities,
    build_comparisons,
    build_settings,
    deterministic_rule_sql,
    render_guard,
)
from tether.profiles import DeterministicRule, Guard, get_profile


@pytest.mark.parametrize("dialect", ["duckdb", "spark"])
def test_guard_renders_join_predicates(dialect):
    sql = render_guard(Guard("name_last_std", 0.85, swap_with="name_first_std"), dialect)
    assert "l.name_last_std IS NULL" in sql and "l.name_last_std = r.name_first_std" in sql
    assert '"name_last_std_l"' not in sql and "name_last_std_l" not in sql
    assert ">= 0.85" in sql


def test_deterministic_rule_sql():
    rule = DeterministicRule(("npi_std",), (Guard("name_last_std", None),))
    assert deterministic_rule_sql(rule, "duckdb") == \
        "l.npi_std = r.npi_std AND (l.name_last_std IS NULL OR r.name_last_std IS NULL OR l.name_last_std = r.name_last_std)"


def test_m_seeding_only_touches_known_levels():
    p = get_profile("provider")
    fts = p.instantiate_fields()
    comps = build_comparisons(p.model_fields(), fts, geo_available=False)
    n = apply_m_probabilities(comps, {"name_last": {"exact last name": 0.8, "no such level": 0.1}, "nope": {"x": 1}})
    assert n == 1
    d = [c for c in comps if c.create_output_column_name() == "name_last"][0].get_comparison("duckdb").as_dict()
    assert d["comparison_levels"][1]["m_probability"] == 0.8


def test_settings_respect_report_only_and_blocking_override():
    p = get_profile("provider")
    fts = p.instantiate_fields()
    cfg = MatchingConfig(blocking_rules=[["name_last_std"], "l.phone_std = r.phone_std"])
    s = build_settings(p, p.fields, fts, cfg, geo_available=False).create_settings_dict("duckdb")
    names = [c["output_column_name"] for c in s["comparisons"]]
    assert "title" not in names and "ein" not in names and "name_first" in names
    assert len(s["blocking_rules_to_generate_predictions"]) == 2


def test_cluster_edges_singletons_and_components():
    p = get_profile("provider")
    fts = p.instantiate_fields()
    cols = p.canonical_columns()
    base = {c: None for c in cols}
    recs = pd.DataFrame([{**base, "unique_id": f"a:{i}", "source_dataset": "a", "source_record_id": str(i),
                          "name": "Jane Doe"} for i in range(4)])
    from tether.matching.prepare import standardize_frame
    frame = standardize_frame(recs, fts)
    settings = build_settings(p, p.fields, fts, MatchingConfig(link_type="dedupe_only"), geo_available=False)
    m = SplinkMatcher([frame], ["a"], settings)
    edges = pd.DataFrame({"unique_id_l": ["a:0", "a:1"], "unique_id_r": ["a:1", "a:2"], "match_probability": [0.99, 0.5]})
    out = m.cluster_edges(edges, threshold=0.9)
    assert set(out["unique_id"]) == {"a:0", "a:1", "a:2", "a:3"}
    cid = out.set_index("unique_id")["cluster_id"]
    assert cid["a:0"] == cid["a:1"] != cid["a:2"] and cid["a:3"] not in (cid["a:0"], cid["a:2"])
