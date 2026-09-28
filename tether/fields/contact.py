"""Contact field types: phone and email domain."""

from __future__ import annotations

import re
from typing import Any, ClassVar

from tether.fields._text import digits_only
from tether.fields.base import ComparisonOptions, FieldType, Parts, ValidationResult
from tether.fields.registry import register_field_type

_EMAIL_RE = re.compile(r"^[^@\s]+@([^@\s]+\.[^@\s]+)$")
_DOMAIN_RE = re.compile(r"^[a-z0-9.-]+\.[a-z]{2,}$")


def normalize_phone(value: str) -> str | None:
    """Return 10 US digits (leading country code 1 stripped) or None."""
    d = digits_only(value)
    if len(d) == 11 and d[0] == "1":
        d = d[1:]
    return d if len(d) == 10 else None


@register_field_type
class PhoneField(FieldType):
    """US phone number."""

    name: ClassVar[str] = "phone"
    description: ClassVar[str] = "US phone normalized to 10 digits"
    outputs: ClassVar[tuple[str, ...]] = ("std", "last7")

    def validate(self, parts: Parts) -> ValidationResult:
        std = normalize_phone(parts.get("value") or "")
        if std is None:
            return ValidationResult.fail("not_10_digits")
        if std[0] in "01" or std[3] in "01":
            return ValidationResult.fail("invalid_area_or_exchange")
        if len(set(std)) == 1:
            return ValidationResult.fail("repeated_digit")
        return ValidationResult.ok()

    def standardize(self, parts: Parts) -> dict[str, Any]:
        std = normalize_phone(parts.get("value") or "")
        valid = self.validate(parts).is_valid
        return {"std": std if valid else None, "last7": std[3:] if std and valid else None}

    def comparisons(self, field_name: str, options: ComparisonOptions):
        import splink.comparison_level_library as cll
        import splink.comparison_library as cl

        std = f"{field_name}_std"
        levels = [
            cll.NullLevel(std),
            cll.ExactMatchLevel(std).configure(label_for_charts="exact phone"),
            cll.ExactMatchLevel(f"{field_name}_last7").configure(label_for_charts="same 7-digit local number"),
            cll.ElseLevel(),
        ]
        return [cl.CustomComparison(levels, output_column_name=field_name,
                                    comparison_description="Phone (graded)")]


@register_field_type
class EmailDomainField(FieldType):
    """Email address or bare domain; only the domain is used as evidence."""

    name: ClassVar[str] = "email_domain"
    description: ClassVar[str] = "Email domain (organization affinity signal)"
    outputs: ClassVar[tuple[str, ...]] = ("domain", "local")

    @staticmethod
    def _split(value: str) -> tuple[str | None, str | None]:
        v = value.strip().lower()
        m = _EMAIL_RE.match(v)
        if m:
            return v.split("@")[0], m.group(1).removeprefix("www.")
        if _DOMAIN_RE.match(v.removeprefix("www.")):
            return None, v.removeprefix("www.")
        return None, None

    def validate(self, parts: Parts) -> ValidationResult:
        _, domain = self._split(parts.get("value") or "")
        return ValidationResult.ok() if domain else ValidationResult.fail("not_email_or_domain")

    def standardize(self, parts: Parts) -> dict[str, Any]:
        local, domain = self._split(parts.get("value") or "")
        return {"domain": domain, "local": local}

    def comparisons(self, field_name: str, options: ComparisonOptions):
        import splink.comparison_level_library as cll
        import splink.comparison_library as cl

        dom = f"{field_name}_domain"
        levels = [
            cll.NullLevel(dom),
            cll.ExactMatchLevel(dom, term_frequency_adjustments=options.term_frequency)
            .configure(label_for_charts="same email domain"),
            cll.ElseLevel(),
        ]
        return [cl.CustomComparison(levels, output_column_name=field_name,
                                    comparison_description="Email domain")]
