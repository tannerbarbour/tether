import pandas as pd

from tether.fields import ComparisonOptions, get_field_type
from tether.fields.person_name import name_variants, parse_person_name


def test_parse_first_middle_last_with_title_and_credential():
    p = parse_person_name("Dr. Robert J. Smith Jr., MD")
    assert (p["first"], p["middle"], p["last"], p["suffix"], p["credential"]) == ("Robert", "J.", "Smith", "JR", "MD")


def test_parse_last_comma_first():
    p = parse_person_name("Smith, Robert J")
    assert (p["first"], p["last"]) == ("Robert", "Smith")


def test_parse_trailing_credential_without_comma():
    p = parse_person_name("ROBERT SMITH MD")
    assert p["last"] == "SMITH" and p["credential"] == "MD"


def test_standardize_full_value():
    ft = get_field_type("person_name")
    out = ft.process({"value": "Bill O'Brien"})
    assert out["first_std"] == "BILL" and out["last_std"] == "O BRIEN"
    assert "WILLIAM" in out["first_variants"]
    assert out["first_dm"] and out["last_dm"] and out["valid"]


def test_components_override_and_credential_in_last():
    ft = get_field_type("person_name")
    out = ft.process({"value": None, "first": "Bob", "middle": "J", "last": "Smith MD", "suffix": None})
    assert out["last"] == "Smith" and out["credential"] == "MD" and out["middle_initial"] == "J"


def test_nickname_variants_include_formal_name():
    assert "ROBERT" in name_variants("BOB")
    assert name_variants("WILLIAM") == ["WILLIAM"] or "WILLIAM" in name_variants("WILLIAM")
    assert name_variants(None) is None


def test_missing_and_invalid():
    ft = get_field_type("person_name")
    assert ft.process({"value": None}) ["invalid_reason"] == "missing"
    assert ft.process({"value": "12345"})["invalid_reason"] == "no_letters"


def test_apply_adds_columns():
    ft = get_field_type("person_name")
    df = pd.DataFrame({"name": ["Jane Doe", None], "other": [1, 2]})
    out = ft.apply(df, "name")
    assert "name_last_std" in out and out.loc[0, "name_last_std"] == "DOE"
    assert out.loc[1, "name_valid"] is False or out.loc[1, "name_valid"] == False  # noqa: E712
    assert list(df.columns) == ["name", "other"]  # input untouched


def test_comparisons_have_graded_levels_and_compile_on_both_dialects():
    ft = get_field_type("person_name")
    comps = ft.comparisons("name", ComparisonOptions(term_frequency=True))
    assert [c.get_comparison("duckdb").as_dict()["output_column_name"] for c in comps] == ["name_first", "name_last"]
    for dialect in ("duckdb", "spark"):
        first, last = (c.get_comparison(dialect).as_dict() for c in comps)
        labels_first = [lv["label_for_charts"] for lv in first["comparison_levels"]]
        assert "nickname / formal-name equivalent" in labels_first
        assert any("swapped" in lv["label_for_charts"] for lv in last["comparison_levels"])
        assert first["comparison_levels"][1].get("tf_adjustment_column") == "name_first_std"
