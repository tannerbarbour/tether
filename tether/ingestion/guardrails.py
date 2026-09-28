"""Guardrails on LLM output.

* The LLM never *generates* identifiers: any NPI/EIN it returns from extraction must appear
  verbatim in the source text and pass validation.
* Standardized values must not smuggle identifiers (long digit runs) into lookup tables.
* Proposed mapping targets and canonical categories must come from the allowed sets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from tether.fields.identifiers import is_valid_ein, is_valid_npi

_LONG_DIGIT_RUN = re.compile(r"\d{7,}")

VALIDATORS = {"npi": is_valid_npi, "ein": is_valid_ein}


@dataclass(frozen=True)
class IdentifierCheck:
    value: str
    kind: str
    accepted: bool
    reason: str | None = None


def verify_identifier(value: str, kind: str, source_text: str) -> IdentifierCheck:
    """Accept ``value`` only if it is a verbatim substring of ``source_text`` and validates."""
    if kind not in VALIDATORS:
        return IdentifierCheck(value, kind, False, "unknown_identifier_kind")
    if not value or value not in source_text:
        return IdentifierCheck(value, kind, False, "not_verbatim_in_source")
    ok, reason = VALIDATORS[kind](value)
    return IdentifierCheck(value, kind, ok, None if ok else reason)


def contains_identifier_like(text: str) -> bool:
    """True if a canonical/standardized value carries a 7+ digit run (identifier leakage)."""
    return bool(_LONG_DIGIT_RUN.search(text or ""))


def filter_allowed(values: list[str], allowed: set[str]) -> tuple[list[str], list[str]]:
    """Split values into (allowed, rejected)."""
    ok = [v for v in values if v in allowed]
    bad = [v for v in values if v not in allowed]
    return ok, bad
