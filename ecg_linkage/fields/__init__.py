"""Field types and their registry. Importing this package registers the built-ins."""

from ecg_linkage.fields.base import (
    ComparisonOptions,
    FieldType,
    Parts,
    ValidationResult,
)
from ecg_linkage.fields.registry import (
    field_type_class,
    get_field_type,
    list_field_types,
    register_field_type,
)

# Built-in field types register themselves on import.
from ecg_linkage.fields import address, contact, identifiers, job_title, org_name, person_name  # noqa: E402,F401

__all__ = [
    "ComparisonOptions", "FieldType", "Parts", "ValidationResult",
    "field_type_class", "get_field_type", "list_field_types", "register_field_type",
]
