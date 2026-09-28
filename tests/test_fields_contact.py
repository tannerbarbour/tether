from ecg_linkage.fields import ComparisonOptions, get_field_type


def test_phone_formats():
    ft = get_field_type("phone")
    for raw in ("(217) 555-0123", "217.555.0123", "1-217-555-0123", "+1 217 555 0123", "2175550123"):
        assert ft.process({"value": raw})["std"] == "2175550123"
    assert ft.process({"value": "555-0123"})["invalid_reason"] == "not_10_digits"
    assert ft.process({"value": "0175550123"})["invalid_reason"] == "invalid_area_or_exchange"


def test_email_domain():
    ft = get_field_type("email_domain")
    assert ft.process({"value": "Bob.Smith@RiversideFP.org"})["domain"] == "riversidefp.org"
    assert ft.process({"value": "www.riversidefp.org"})["domain"] == "riversidefp.org"
    assert ft.process({"value": "not an email"})["invalid_reason"] == "not_email_or_domain"


def test_comparisons_compile():
    for name in ("phone", "email_domain"):
        for dialect in ("duckdb", "spark"):
            get_field_type(name).comparisons(name, ComparisonOptions(term_frequency=True))[0].get_comparison(dialect)
