import pandas as pd

from tether.config import SchemaMapping
from tether.ingestion import MockLLMClient, propose_mapping
from tether.ingestion.prompts import MappingProposal
from tether.ingestion.schema_mapper import detect_shape, mask_value, prematch
from tether.profiles import get_profile


def test_mask_value():
    assert mask_value("62701-1234") == "99999-9999" and mask_value("Jane O'Neil") == "Aaaa A'Aaaa"


def test_detect_shape():
    assert detect_shape(["1234567893"] * 5)[0] == "npi"
    assert detect_shape(["12-3456789"] * 5)[0] == "ein"
    assert detect_shape(["a@b.org"] * 5)[0] == "email"
    assert detect_shape(["hello"] * 5) is None and detect_shape(["62701"] * 2) is None


def test_prematch_uses_examples_and_shapes():
    raw = pd.DataFrame({"npi": ["1234567893"] * 6, "Prov Last": ["x"] * 6, "TAXID": ["12-3456789"] * 6, "Notes": ["n"] * 6})
    ex = SchemaMapping(source="old", profile="provider", status="approved",
                       columns=[{"source_column": "prov_last", "target": "name_last"}])
    resolved, unresolved = prematch(raw, get_profile("provider"), [ex])
    assert resolved["npi"].target == "npi" and resolved["Prov Last"].target == "name_last"
    assert resolved["TAXID"].target == "ein" and unresolved == ["Notes"]


def test_propose_mapping_on_synthetic_sources(small_dataset):
    profile = get_profile("provider")
    client = MockLLMClient()
    roster = propose_mapping(small_dataset.source_a, "roster", profile, client, record_id_column="Provider ID")
    claims = propose_mapping(small_dataset.source_b, "claims", profile, client, record_id_column="rendering_prov_id")
    assert roster.status == "proposed" and roster.record_id_column == "Provider ID"
    r = roster.as_dict()
    assert r["Provider Name"] == "name" and r["Practice Address"] == "address" and r["Tax ID"] == "ein"
    assert "Provider ID" not in r and "Provider ID" not in roster.unmapped
    c = claims.as_dict()
    assert c["prov_last_name"] == "name_last" and c["addr_line1"] == "address_street" and c["zip"] == "address_zip"
    assert c["billing_org"] == "org" and c["specialty"] == "title"


def test_redaction_masks_samples_before_llm(small_dataset):
    client = MockLLMClient()
    propose_mapping(small_dataset.source_a, "roster", get_profile("provider"), client,
                    record_id_column="Provider ID", redact_columns=["Practice Address"], sample_rows=5)
    payload = client.calls[0].payload
    addr = next(c for c in payload["columns"] if c["name"] == "Practice Address")
    assert all(s == mask_value(s) for s in addr["samples"]) and len(addr["samples"]) <= 5
    name = next(c for c in payload["columns"] if c["name"] == "Provider Name")
    assert any(ch.isalpha() and ch not in "aA" for s in name["samples"] for ch in s)


def test_llm_guardrails_on_targets():
    raw = pd.DataFrame({"colA": ["x"] * 6, "colB": ["y"] * 6, "colC": ["z"] * 6})
    scripted = MappingProposal.model_validate({"columns": [
        {"source_column": "colA", "target": "ssn", "confidence": 0.9, "rationale": "not allowed"},
        {"source_column": "colB", "target": "org", "confidence": 0.7, "rationale": ""},
        {"source_column": "colC", "target": "org", "confidence": 0.9, "rationale": "wins"},
    ]})
    m = propose_mapping(raw, "s", get_profile("provider"), MockLLMClient(script=[scripted]))
    assert m.as_dict() == {"colC": "org"} and set(m.unmapped) == {"colA", "colB"}
    assert m.provenance["method"] == "prematch+llm"


def test_no_llm_call_when_everything_prematches():
    raw = pd.DataFrame({"npi": ["1234567893"] * 6, "phone": ["217-555-0100"] * 6})
    client = MockLLMClient()
    m = propose_mapping(raw, "s", get_profile("provider"), client)
    assert client.calls == [] and m.provenance["method"] == "prematch" and len(m.columns) == 2
