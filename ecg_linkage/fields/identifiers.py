"""Identifier field types: NPI (Luhn with 80840 prefix) and EIN (IRS prefix table).

Invalid identifiers are flagged and kept in ``<field>_digits`` but the ``<field>_std``
column, which is the only one comparisons and the deterministic pre-pass read, is
populated only for valid values.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ecg_linkage.fields._text import digits_only
from ecg_linkage.fields.base import ComparisonOptions, FieldType, Parts, ValidationResult
from ecg_linkage.fields.registry import register_field_type

NPI_PREFIX = "80840"
"""Card-issuer prefix prepended to the 9 NPI digits before the Luhn check."""

EIN_VALID_PREFIXES: frozenset[str] = frozenset(
    f"{p:02d}" for p in (
        list(range(1, 7)) + list(range(10, 17)) + list(range(20, 28)) + list(range(30, 36))
        + list(range(36, 40)) + list(range(40, 49)) + list(range(50, 60)) + list(range(60, 69))
        + list(range(71, 78)) + list(range(80, 89)) + list(range(90, 96)) + [98, 99]
    )
)


def luhn_check_digit(digits: str) -> int:
    """Return the Luhn check digit for a string of digits (without the check digit)."""
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return (10 - total % 10) % 10


def npi_check_digit(first_nine: str) -> int:
    """Check digit for the first 9 NPI digits, using the 80840 prefix."""
    return luhn_check_digit(NPI_PREFIX + first_nine)


def is_valid_npi(value: str) -> tuple[bool, str | None]:
    """Validate an NPI string. Returns (is_valid, failure_reason)."""
    d = digits_only(value)
    if len(d) != 10:
        return False, "not_10_digits"
    if d[0] not in "12":
        return False, "bad_leading_digit"
    if npi_check_digit(d[:9]) != int(d[9]):
        return False, "luhn_failed"
    return True, None


def is_valid_ein(value: str) -> tuple[bool, str | None]:
    """Validate an EIN. Returns (is_valid, failure_reason)."""
    d = digits_only(value)
    if len(d) != 9:
        return False, "not_9_digits"
    if d[:2] not in EIN_VALID_PREFIXES:
        return False, "invalid_prefix"
    if d == "000000000":
        return False, "all_zero"
    return True, None


class _IdentifierField(FieldType):
    outputs: ClassVar[tuple[str, ...]] = ("std", "digits")
    description_for_charts: ClassVar[str] = "identifier"

    def _check(self, value: str) -> tuple[bool, str | None]:  # pragma: no cover - abstract
        raise NotImplementedError

    def _format(self, digits: str) -> str:
        return digits

    def validate(self, parts: Parts) -> ValidationResult:
        ok, reason = self._check(parts.get("value") or "")
        return ValidationResult.ok() if ok else ValidationResult.fail(reason or "invalid")

    def standardize(self, parts: Parts) -> dict[str, Any]:
        digits = digits_only(parts.get("value") or "") or None
        ok, _ = self._check(parts.get("value") or "")
        return {"std": self._format(digits) if ok and digits else None, "digits": digits}

    def comparisons(self, field_name: str, options: ComparisonOptions):
        import splink.comparison_level_library as cll
        import splink.comparison_library as cl

        std = f"{field_name}_std"
        levels = [
            cll.NullLevel(std),
            cll.ExactMatchLevel(std).configure(label_for_charts=f"exact valid {self.name.upper()}"),
            cll.ElseLevel(),
        ]
        return [cl.CustomComparison(levels, output_column_name=field_name,
                                    comparison_description=f"{self.name.upper()} (validated exact)")]


@register_field_type
class NPIField(_IdentifierField):
    """National Provider Identifier: 10 digits, Luhn-valid with the 80840 prefix."""

    name: ClassVar[str] = "npi"
    description: ClassVar[str] = "NPI with Luhn (80840) validation"

    def _check(self, value: str) -> tuple[bool, str | None]:
        return is_valid_npi(value)


@register_field_type
class EINField(_IdentifierField):
    """Employer Identification Number: 9 digits with a valid IRS campus prefix."""

    name: ClassVar[str] = "ein"
    description: ClassVar[str] = "EIN with format/prefix validation"

    def _check(self, value: str) -> tuple[bool, str | None]:
        return is_valid_ein(value)

    def _format(self, digits: str) -> str:
        return f"{digits[:2]}-{digits[2:]}"
