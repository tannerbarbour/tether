"""Organization name field type: legal-suffix stripping, alias expansion, comparisons."""

from __future__ import annotations

import re
from typing import Any, ClassVar, Mapping

from tether.fields._text import basic_clean, normalize_whitespace, phonetic, strip_accents
from tether.fields.base import ComparisonOptions, FieldType, Parts, ValidationResult
from tether.fields.registry import register_field_type

LEGAL_SUFFIXES: tuple[str, ...] = (
    "INCORPORATED", "INC", "LLC", "L L C", "PLLC", "P L L C", "CORPORATION", "CORP",
    "PC", "P C", "PA", "P A", "LTD", "LIMITED", "LLP", "L L P", "LP", "L P", "CO",
    "COMPANY", "SC", "S C", "PSC", "PLC",
)
"""Stripped only when they appear as trailing tokens, repeatedly (``... PA, Inc``)."""

DEFAULT_ORG_ALIASES: dict[str, str] = {
    "ST": "SAINT", "MT": "MOUNT", "FT": "FORT",
    "MED": "MEDICAL", "CTR": "CENTER", "CNTR": "CENTER", "CENTRE": "CENTER",
    "HOSP": "HOSPITAL", "ASSOC": "ASSOCIATES", "ASSOCS": "ASSOCIATES", "ASSN": "ASSOCIATION",
    "GRP": "GROUP", "SVCS": "SERVICES", "SVC": "SERVICE", "DEPT": "DEPARTMENT",
    "UNIV": "UNIVERSITY", "HLTH": "HEALTH", "HC": "HEALTHCARE", "HEALTH CARE": "HEALTHCARE",
    "FAM": "FAMILY", "PRAC": "PRACTICE", "PHYS": "PHYSICIANS", "CLIN": "CLINIC",
    "ORTHO": "ORTHOPEDIC", "ORTHOPAEDIC": "ORTHOPEDIC", "PEDS": "PEDIATRICS",
    "PEDIATRIC": "PEDIATRICS", "OB GYN": "OBSTETRICS GYNECOLOGY", "OBGYN": "OBSTETRICS GYNECOLOGY",
    "SURG": "SURGERY", "SURGICAL": "SURGERY", "REHAB": "REHABILITATION", "SYS": "SYSTEM",
    "INTL": "INTERNATIONAL", "NATL": "NATIONAL", "AMER": "AMERICAN", "&": "AND",
    "THE": "",
}
"""Generic, engagement-independent token/phrase expansions applied to the cleaned form.

Engagement- or client-specific aliases live in the knowledge base (``lookup_org_aliases``)
and are merged in at construction time; this table is the ``generic`` baseline.
"""

_SUFFIX_RE = re.compile(
    r"(?:[\s,]+(?:" + "|".join(re.escape(s) for s in LEGAL_SUFFIXES) + r"))+\s*$"
)


def strip_legal_suffix(text: str) -> tuple[str, str | None]:
    """Return (name without trailing legal suffix tokens, suffix tokens removed or None).

    Operates on the cleaned (uppercase, punctuation-free) form.
    """
    m = _SUFFIX_RE.search(" " + text)
    if not m:
        return text, None
    removed = normalize_whitespace(m.group(0).replace(",", " "))
    return normalize_whitespace((" " + text)[: m.start()]), removed or None


def apply_aliases(text: str, aliases: Mapping[str, str]) -> str:
    """Apply phrase aliases (longest first) then token aliases to a cleaned string."""
    out = f" {text} "
    for phrase in sorted((k for k in aliases if " " in k), key=len, reverse=True):
        out = out.replace(f" {phrase} ", f" {aliases[phrase]} ")
    tokens = [aliases.get(t, t) for t in out.split()]
    return normalize_whitespace(" ".join(t for t in tokens if t))


@register_field_type
class OrgNameField(FieldType):
    """Organization / practice / employer name."""

    name: ClassVar[str] = "org_name"
    description: ClassVar[str] = "Organization name with legal-suffix and alias normalization"
    outputs: ClassVar[tuple[str, ...]] = ("raw_std", "clean", "legal_suffix", "dm")

    def __init__(self, aliases: Mapping[str, str] | None = None, use_default_aliases: bool = True):
        merged: dict[str, str] = dict(DEFAULT_ORG_ALIASES) if use_default_aliases else {}
        if aliases:
            merged.update({basic_clean(k): basic_clean(v) for k, v in aliases.items()})
        self.aliases = merged

    def validate(self, parts: Parts) -> ValidationResult:
        value = parts.get("value") or ""
        if not re.search(r"[A-Za-z0-9]", value):
            return ValidationResult.fail("no_alphanumerics")
        if len(basic_clean(value)) < 2:
            return ValidationResult.fail("too_short")
        return ValidationResult.ok()

    def standardize(self, parts: Parts) -> dict[str, Any]:
        value = parts.get("value") or ""
        raw_std = normalize_whitespace(strip_accents(value).upper()) or None
        cleaned = basic_clean(value)
        stripped, suffix = strip_legal_suffix(cleaned)
        clean = apply_aliases(stripped, self.aliases) or stripped or None
        return {
            "raw_std": raw_std,
            "clean": clean,
            "legal_suffix": suffix,
            "dm": phonetic(clean) if clean else None,
        }

    def comparisons(self, field_name: str, options: ComparisonOptions):
        import splink.comparison_level_library as cll
        import splink.comparison_library as cl

        clean = f"{field_name}_clean"
        levels = [
            cll.NullLevel(clean),
            cll.ExactMatchLevel(f"{field_name}_raw_std").configure(label_for_charts="exact raw org name"),
            cll.ExactMatchLevel(clean, term_frequency_adjustments=options.term_frequency)
            .configure(label_for_charts="exact cleaned org name"),
            *[cll.JaroWinklerLevel(clean, t).configure(label_for_charts=f"org JW >= {t}")
              for t in options.jaro_winkler_thresholds],
        ]
        if options.include_phonetic:
            levels.append(cll.ExactMatchLevel(f"{field_name}_dm").configure(label_for_charts="org phonetic match"))
        levels.append(cll.ElseLevel())
        return [cl.CustomComparison(levels, output_column_name=field_name,
                                    comparison_description="Organization name (graded)")]
