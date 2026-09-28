import pytest

from ecg_linkage.fields import ComparisonOptions, get_field_type
from ecg_linkage.fields.identifiers import is_valid_ein, is_valid_npi, npi_check_digit


def test_npi_check_digit_known_example():
    # CMS documentation example: 123456789 -> check digit 3
    assert npi_check_digit("123456789") == 3
    assert is_valid_npi("1234567893") == (True, None)


@pytest.mark.parametrize("value, reason", [
    ("1234567890", "luhn_failed"), ("123456789", "not_10_digits"), ("3234567893", "bad_leading_digit"),
])
def test_npi_invalid(value, reason):
    assert is_valid_npi(value) == (False, reason)


def test_npi_field_flags_not_drops():
    out = get_field_type("npi").process({"value": "1234567890"})
    assert out["valid"] is False and out["std"] is None and out["digits"] == "1234567890"


def test_ein():
    assert is_valid_ein("12-3456789") == (True, None)
    assert is_valid_ein("07-3456789") == (False, "invalid_prefix")
    assert is_valid_ein("1234567") == (False, "not_9_digits")
    assert get_field_type("ein").process({"value": "123456789"})["std"] == "12-3456789"


def test_identifier_comparison_is_null_exact_else():
    for name in ("npi", "ein"):
        comp = get_field_type(name).comparisons(name, ComparisonOptions())[0]
        levels = comp.get_comparison("duckdb").as_dict()["comparison_levels"]
        assert len(levels) == 3 and levels[0]["is_null_level"]
