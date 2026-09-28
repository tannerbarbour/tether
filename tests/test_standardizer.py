import pandas as pd

from tether.ingestion import LookupCache, MockLLMClient, extract_identifiers, standardize_values
from tether.ingestion.guardrails import contains_identifier_like, verify_identifier
from tether.ingestion.prompts import IdentifierExtraction, ValueMappingBatch
from tether.ingestion.standardizer import LOOKUP_COLUMNS


def _cache(tmp_path, rows, engagement="eng1"):
    df = pd.DataFrame(rows, columns=LOOKUP_COLUMNS)
    df.to_csv(tmp_path / "lk.csv", index=False)
    return LookupCache(tmp_path / "lk.csv", engagement)


def test_cache_visibility_rules(tmp_path):
    base = {"confidence": "0.9", "reviewer": "", "created_at": "", "version": "1", "method": "m"}
    rows = [
        {**base, "value": "A", "canonical": "1", "engagement": "eng1", "reuse_scope": "client_scoped", "state": "engagement_local"},
        {**base, "value": "B", "canonical": "2", "engagement": "other", "reuse_scope": "client_scoped", "state": "engagement_local"},
        {**base, "value": "C", "canonical": "3", "engagement": "other", "reuse_scope": "generic", "state": "promoted"},
        {**base, "value": "D", "canonical": "4", "engagement": "other", "reuse_scope": "generic", "state": "reviewed"},
    ]
    assert _cache(tmp_path, rows).lookup() == {"A": "1", "C": "3"}
    loose = _cache(tmp_path, rows)
    loose.use_promoted_only = False
    assert loose.lookup() == {"A": "1", "C": "3", "D": "4"}


def test_standardize_titles_cache_then_llm(tmp_path):
    cache = LookupCache(tmp_path / "titles.csv", "eng1")
    client = MockLLMClient()
    stats = standardize_values(["Staff RN", "Wizard of Oz", "wizard of oz"], "job_title", client, cache)
    assert stats["n_builtin"] == 1 and stats["n_llm"] == 1 and stats["calls"] == 1
    assert stats["mapping"]["STAFF RN"] == "REGISTERED NURSE" and stats["mapping"]["WIZARD OF OZ"] == "OTHER"
    cache.save()
    again = standardize_values(["Wizard of Oz"], "job_title", MockLLMClient(), LookupCache(tmp_path / "titles.csv", "eng1"))
    assert again["calls"] == 0 and again["n_cached"] == 1


def test_standardize_guardrails(tmp_path):
    cache = LookupCache(tmp_path / "t.csv", "e")
    scripted = ValueMappingBatch.model_validate({"items": [
        {"value": "X", "canonical": "ASTRONAUT", "confidence": 0.9},          # not in allowed set
        {"value": "Y", "canonical": "PHYSICIAN 1234567890", "confidence": 0.9},  # identifier leakage
        {"value": "Z", "canonical": "PHYSICIAN", "confidence": 0.9},
        {"value": "NOT ASKED", "canonical": "PHYSICIAN", "confidence": 0.9},
    ]})
    stats = standardize_values(["X", "Y", "Z"], "job_title", MockLLMClient(script=[scripted]), cache)
    assert stats["n_rejected"] == 3 and stats["mapping"] == {"Z": "PHYSICIAN"}


def test_org_alias_keys_strip_suffix(tmp_path):
    cache = LookupCache(tmp_path / "o.csv", "e")
    stats = standardize_values(["St. Mary's Med Ctr, Inc."], "org_alias", MockLLMClient(), cache)
    assert stats["mapping"] == {"ST MARY S MED CTR": "SAINT MARY S MEDICAL CENTER"}


def test_identifier_guardrail():
    assert verify_identifier("1234567893", "npi", "NPI 1234567893 on file").accepted
    assert verify_identifier("1234567890", "npi", "NPI 1234567890").reason == "luhn_failed"
    assert verify_identifier("1234567893", "npi", "no id here").reason == "not_verbatim_in_source"
    assert contains_identifier_like("PHYSICIAN 1234567") and not contains_identifier_like("PHYSICIAN")


def test_extract_identifiers_rejects_fabrication():
    text = "Dr. Roe, NPI 1234567893, EIN 12-3456789."
    fabricated = IdentifierExtraction.model_validate({"candidates": [
        {"kind": "npi", "value": "1234567893"}, {"kind": "npi", "value": "1999999991"}, {"kind": "ein", "value": "12-3456789"}]})
    out = extract_identifiers(text, MockLLMClient(script=[fabricated]))
    assert [c["accepted"] for c in out] == [True, False, True]
    assert extract_identifiers(text, MockLLMClient())[0]["value"] == "1234567893"
