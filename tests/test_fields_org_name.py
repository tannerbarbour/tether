import pytest

from ecg_linkage.fields import ComparisonOptions, get_field_type
from ecg_linkage.fields.org_name import apply_aliases, strip_legal_suffix


@pytest.mark.parametrize(
    "raw, clean, suffix",
    [
        ("Riverside Family Practice LLC", "RIVERSIDE FAMILY PRACTICE", "LLC"),
        ("St. Mary's Med Ctr, P.A., Inc.", "SAINT MARY S MEDICAL CENTER", "P A INC"),
        ("Summit Orthopaedic Assoc PLLC", "SUMMIT ORTHOPEDIC ASSOCIATES", "PLLC"),
        ("The Unity Health Care Corporation", "UNITY HEALTHCARE", "CORPORATION"),
        ("Acme Company", "ACME", "COMPANY"),
    ],
)
def test_cleaning(raw, clean, suffix):
    out = get_field_type("org_name").process({"value": raw})
    assert out["clean"] == clean
    assert out["legal_suffix"] == suffix
    assert out["valid"]


def test_raw_std_preserves_punctuation_but_normalizes_case_and_space():
    out = get_field_type("org_name").process({"value": "  st. mary's   med ctr, inc. "})
    assert out["raw_std"] == "ST. MARY'S MED CTR, INC."


def test_strip_suffix_no_suffix():
    assert strip_legal_suffix("MERCY HOSPITAL") == ("MERCY HOSPITAL", None)


def test_alias_phrases_before_tokens():
    assert apply_aliases("OB GYN ASSOC", {"OB GYN": "OBSTETRICS GYNECOLOGY", "ASSOC": "ASSOCIATES"}) == \
        "OBSTETRICS GYNECOLOGY ASSOCIATES"


def test_custom_aliases_merge_with_defaults():
    ft = get_field_type("org_name", aliases={"ECG": "EXAMPLE CONSULTING GROUP"})
    assert ft.process({"value": "ECG Med Ctr"})["clean"] == "EXAMPLE CONSULTING GROUP MEDICAL CENTER"


def test_invalid():
    assert get_field_type("org_name").process({"value": "---"})["invalid_reason"] == "no_alphanumerics"


def test_comparison_levels_order():
    comp = get_field_type("org_name").comparisons("org", ComparisonOptions(term_frequency=True))[0]
    labels = [lv["label_for_charts"] for lv in comp.get_comparison("duckdb").as_dict()["comparison_levels"]]
    assert labels[1:4] == ["exact raw org name", "exact cleaned org name", "org JW >= 0.92"]
    comp.get_comparison("spark")  # compiles for Spark
