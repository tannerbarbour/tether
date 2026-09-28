import pytest

from tether.profiles import BlockingRule, get_profile, list_profiles


def test_profiles_registered():
    assert set(list_profiles()) >= {"provider", "organization"}


def test_provider_profile_shape():
    p = get_profile("provider")
    assert p.fields["npi"].role == "identifier"
    assert p.fields["ein"].role == "supporting"
    assert "title" not in p.model_fields()
    assert "name_first" in p.canonical_columns() and "address_zip" in p.canonical_columns()
    assert p.hard_constraints[0].column == "npi_std"
    assert p.deterministic_rules[0].guards[0].swap_with == "name_first_std"


def test_provider_blocking_and_training_rules_reference_derived_columns():
    p = get_profile("provider")
    fts = p.instantiate_fields()
    derived = {c for f, ft in fts.items() for c in ft.output_columns(f)}
    for rule in p.blocking_rules + p.training_blocking_rules:
        for col in rule.columns:
            assert col in derived, col


def test_all_model_comparisons_compile_both_dialects():
    p = get_profile("provider")
    fts = p.instantiate_fields({"zip_centroids": {"62701": (39.8, -89.6)}})
    for fname, spec in p.model_fields().items():
        for comp in fts[fname].comparisons(fname, spec.comparison_options(geo_available=True)):
            for dialect in ("duckdb", "spark"):
                comp.get_comparison(dialect).as_dict()


def test_blocking_rule_validation():
    with pytest.raises(ValueError):
        BlockingRule()
    with pytest.raises(ValueError):
        BlockingRule(columns=("a",), sql="l.a = r.a")


def test_unknown_profile():
    with pytest.raises(KeyError):
        get_profile("vendor")


def test_comparison_schema_is_serializable():
    import json
    json.dumps(get_profile("provider").comparison_schema())
