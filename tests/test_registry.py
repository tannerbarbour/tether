import pytest

from ecg_linkage.fields import FieldType, get_field_type, list_field_types, register_field_type
from ecg_linkage.fields.base import ValidationResult


def test_builtins_registered():
    names = set(list_field_types())
    assert {"person_name", "org_name", "address", "npi", "ein", "phone", "email_domain", "job_title"} <= names


def test_unknown_field_type():
    with pytest.raises(KeyError):
        get_field_type("nope")


def test_register_custom_field_type():
    @register_field_type
    class ColorField(FieldType):
        name = "test_color"
        outputs = ("std",)

        def validate(self, parts):
            return ValidationResult.ok()

        def standardize(self, parts):
            return {"std": parts["value"].upper()}

        def comparisons(self, field_name, options):
            import splink.comparison_library as cl

            return [cl.ExactMatch(f"{field_name}_std")]

    assert get_field_type("test_color").process({"value": "red"})["std"] == "RED"


def test_duplicate_name_rejected():
    with pytest.raises(ValueError):
        @register_field_type
        class Dup(FieldType):
            name = "npi"
            def validate(self, parts): ...
            def standardize(self, parts): ...
            def comparisons(self, field_name, options): ...
