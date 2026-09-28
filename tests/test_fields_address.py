from tether.fields import ComparisonOptions, get_field_type
from tether.fields.address import normalize_state, normalize_street, parse_address


def test_parse_single_line():
    p = parse_address("123 North Main Street, Suite 400, Springfield, Illinois 62701-1234")
    assert p["street"] == "123 North Main Street" and p["unit"] == "Suite 400"
    assert p["city"] == "Springfield" and p["zip"] == "62701-1234"


def test_normalizers():
    assert normalize_street("123 N. Main St. Ste 400") == "123 N MAIN ST"
    assert normalize_street("P.O. Box 12") == "PO BOX 12"
    assert normalize_state("Illinois") == "IL" and normalize_state("il.") == "IL"


def test_standardize_components_and_geo():
    ft = get_field_type("address", zip_centroids={"62701": (39.8, -89.65)})
    out = ft.process({"value": None, "street": "123 N Main Street", "city": "Springfield", "state": "IL", "zip": "62701"})
    assert out["zip5"] == "62701" and out["lat"] == 39.8 and out["full_std"] == "123 N MAIN ST SPRINGFIELD IL 62701"
    assert out["valid"]


def test_equivalent_formats_standardize_identically():
    ft = get_field_type("address")
    a = ft.process({"value": "6277 river rd, ste 637, Pittsburgh, PA 15222"})
    b = ft.process({"value": "6277 River Road Suite 637, PITTSBURGH, PA 15222-4410"})
    assert a["full_std"] == b["full_std"]


def test_invalid_zip_flagged_not_dropped():
    out = get_field_type("address").process({"value": "1 Main St, Springfield, IL 6270"})
    assert out["valid"] is False and out["street_std"] == "1 MAIN ST"


def test_geo_levels_only_when_available():
    ft = get_field_type("address")
    without = ft.comparisons("address", ComparisonOptions(geo_available=False))[0].get_comparison("duckdb").as_dict()
    with_geo = ft.comparisons("address", ComparisonOptions(geo_available=True))[0].get_comparison("spark").as_dict()
    assert len(with_geo["comparison_levels"]) == len(without["comparison_levels"]) + 2
