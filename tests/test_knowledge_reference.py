import pandas as pd
import pytest

from tether.config import SchemaMapping
from tether.knowledge import LocalKnowledgeBase, comparison_schema_hash
from tether.knowledge.base import with_asset_columns
from tether.profiles import get_profile
from tether.reference import NPPESReference, ZipCentroidReference, hub_entity_ids


def test_kb_append_usable_promote_version(tmp_path):
    kb = LocalKnowledgeBase(tmp_path)
    v0 = kb.version()
    rows = with_asset_columns(pd.DataFrame({"value": ["A", "B"], "canonical": ["1", "2"]}), "eng1")
    kb.append("lookup_titles", rows)
    assert kb.version() != v0
    assert len(kb.usable("lookup_titles", "eng1")) == 2 and len(kb.usable("lookup_titles", "other")) == 0
    with pytest.raises(ValueError):
        kb.promote("lookup_titles", to_state="promoted", reviewer="", engagement="eng1")
    n = kb.promote("lookup_titles", to_state="promoted", reviewer="rev", engagement="eng1", reuse_scope="generic")
    assert n == 2 and len(kb.usable("lookup_titles", "other")) == 2
    assert kb.promote("lookup_titles", to_state="reviewed", reviewer="rev", engagement="eng1") == 0  # never backwards
    df = kb.read("lookup_titles")
    assert (df["reviewer"] == "rev").all() and (df["version"].astype(int) == 2).all()
    assert (tmp_path / "manifest.json").exists()


def test_model_params_roundtrip_requires_schema_hash(tmp_path):
    kb = LocalKnowledgeBase(tmp_path)
    p = get_profile("provider")
    h = comparison_schema_hash(p.fields, p.name)
    kb.save_model_params({"name_last": {"exact last name": 0.8}}, "provider", h, "eng1", ["roster"], "client_scoped", "r1")
    assert kb.m_probabilities("provider", h, "eng1") == {"name_last": {"exact last name": 0.8}}
    assert kb.m_probabilities("provider", "otherhash", "eng1") == {}
    assert kb.m_probabilities("provider", h, "eng2") == {}  # not promoted
    kb.promote("model_params", to_state="promoted", reviewer="r", engagement="eng1", reuse_scope="generic")
    assert kb.m_probabilities("provider", h, "eng2")["name_last"]["exact last name"] == 0.8


def test_labeled_pairs_store_vectors_only(tmp_path):
    kb = LocalKnowledgeBase(tmp_path)
    g = pd.DataFrame({"unique_id_l": ["a"], "unique_id_r": ["b"], "name_full_std_l": ["JANE"], "gamma_name_first": [3], "gamma_org": [2]})
    assert kb.add_labeled_pairs(g, 1, "deterministic_rule", "provider", "h", "e", "client_scoped") == 1
    df = kb.read("labeled_pairs")
    assert "name_full_std_l" not in df.columns and "unique_id_l" not in df.columns and int(df["label"].iloc[0]) == 1


def test_mapping_registration_dedup_and_run_history(tmp_path):
    kb = LocalKnowledgeBase(tmp_path)
    m = SchemaMapping(source="s", profile="provider", status="approved", columns=[{"source_column": "x", "target": "npi"}])
    assert kb.register_mapping(m, "e", "client_scoped") and not kb.register_mapping(m, "e", "client_scoped")
    assert kb.approved_mappings("provider", "e")[0].as_dict() == {"x": "npi"}
    run_id = kb.record_run({"engagement": "e", "stats": {"n": 1}}, "e")
    assert run_id in set(kb.read("run_history")["run_id"])
    assert kb.status().set_index("table").loc["run_history", "rows"] == 1


def test_nppes_snapshot_enrich_and_hub(small_dataset, tmp_path):
    small_dataset.write(tmp_path)
    ref = NPPESReference(tmp_path / "nppes_sample.csv")
    snap = ref.snapshot()
    assert len(snap.snapshot_id) == 16 and "Public domain" in snap.license and snap.n_rows == len(small_dataset.nppes)
    org_npi = small_dataset.orgs["org_npi"].iloc[0]
    person_npi = small_dataset.entities["npi"].iloc[0]
    recs = pd.DataFrame({"unique_id": ["a", "b", "c"], "npi_std": [org_npi, person_npi, "1999999991"],
                         "npi_valid": [True, True, True], "npi_invalid_reason": [None, None, None]})
    out = ref.enrich(recs)
    assert pd.isna(out.loc[0, "npi_std"]) and out.loc[0, "npi_invalid_reason"] == "type2_organization_npi"
    assert out.loc[1, "npi_std"] == person_npi and out.loc[1, "nppes_found"] and not out.loc[2, "nppes_found"]
    hub = ref.hub_frame(get_profile("provider").canonical_columns())
    assert (hub["source_dataset"] == "nppes").all() and hub["npi"].is_unique and hub["name_last"].notna().all()
    cw = pd.DataFrame({"source": ["roster", "nppes", "claims"], "source_record_id": ["1", person_npi, "2"], "entity_id": ["E1", "E1", "E2"]})
    relabeled = hub_entity_ids(cw)
    assert (relabeled["entity_id"] == [f"NPI-{person_npi}", f"NPI-{person_npi}", "E2"]).all()
    assert relabeled["is_reference"].tolist() == [False, True, False]
    zc = ZipCentroidReference(tmp_path / "zip_centroids.csv")
    assert "62701" in zc.as_dict() and zc.snapshot().source == "zip_centroids"
