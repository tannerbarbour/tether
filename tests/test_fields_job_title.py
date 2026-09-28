from tether.fields import get_field_type


def test_taxonomy_mapping():
    ft = get_field_type("job_title")
    assert ft.process({"value": "Family Practice Doctor"})["canonical"] == "PHYSICIAN"
    assert ft.process({"value": "PA-C"})["canonical"] == "PHYSICIAN ASSISTANT"
    assert ft.process({"value": "Astronaut"})["canonical"] is None


def test_custom_taxonomy_extends_default():
    ft = get_field_type("job_title", taxonomy={"Astronaut": "PHYSICIAN"})
    assert ft.process({"value": "astronaut"})["canonical"] == "PHYSICIAN"
    assert ft.process({"value": "RN"})["canonical"] == "REGISTERED NURSE"
