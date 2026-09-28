"""Person name field type: parsing, nickname variants, Double Metaphone, comparisons."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, ClassVar

from nameparser import HumanName

from ecg_linkage.fields._text import basic_clean, phonetic
from ecg_linkage.fields.base import ComparisonOptions, FieldType, Parts, ValidationResult
from ecg_linkage.fields.registry import register_field_type

GENERATIONAL_SUFFIXES = {"JR", "SR", "II", "III", "IV", "V"}
CREDENTIALS = {
    "MD", "DO", "PHD", "RN", "NP", "PA", "PAC", "PA-C", "APRN", "ARNP", "CRNA", "DDS", "DMD",
    "DPM", "OD", "PSYD", "LCSW", "MPH", "MBA", "FACP", "FACS", "BSN", "MSN", "DNP", "FNP",
    "CNM", "LPN", "PT", "DPT", "OT", "PHARMD", "RPH",
}
_LETTERS_RE = re.compile(r"[A-Za-z]")


@lru_cache(maxsize=1)
def _nicknamer():
    from nicknames import NickNamer

    return NickNamer()


def name_variants(first: str | None) -> list[str] | None:
    """Sorted list of the cleaned first name plus every canonical form it may abbreviate.

    ``Bill`` -> ``["BILL", "ROBERT", "WILL", "WILLIAM", "WILLIS"]``; ``William`` -> ``["WILLIAM"]``.
    Used by an array-intersection comparison level, so a nickname on either side links to
    the formal name without collapsing to a single (lossy) canonical value.
    """
    if not first:
        return None
    token = first.split(" ")[0]
    variants = {token}
    try:
        variants |= {c.upper() for c in _nicknamer().canonicals_of(token.lower())}
    except Exception:  # pragma: no cover - lookup failures degrade to exact-only
        pass
    return sorted(v for v in variants if v)


def _split_suffixes(raw_suffix: str) -> tuple[str | None, str | None]:
    """Separate generational suffix (Jr, III) from credentials (MD, RN)."""
    tokens = [t for t in basic_clean(raw_suffix).split(" ") if t]
    gen = [t for t in tokens if t in GENERATIONAL_SUFFIXES]
    cred = [t for t in tokens if t not in GENERATIONAL_SUFFIXES]
    return (" ".join(gen) or None, " ".join(cred) or None)


def parse_person_name(value: str) -> dict[str, str | None]:
    """Parse a free-text person name into first/middle/last/suffix/credential.

    Handles ``First M Last``, ``Last, First M``, titles (``Dr.``) and trailing
    credentials (``MD``). Returns raw (not cleaned) parts.
    """
    name = HumanName(value)
    suffix, credential = _split_suffixes(name.suffix or "")
    last_tokens = (name.last or "").split()
    # nameparser leaves unknown trailing credentials attached to the last name.
    trailing_creds: list[str] = []
    while last_tokens and basic_clean(last_tokens[-1]) in CREDENTIALS and len(last_tokens) > 1:
        trailing_creds.insert(0, last_tokens.pop())
    if trailing_creds:
        credential = " ".join(filter(None, [credential, basic_clean(" ".join(trailing_creds))]))
    return {
        "first": name.first or None,
        "middle": name.middle or None,
        "last": " ".join(last_tokens) or None,
        "suffix": suffix,
        "credential": credential or None,
    }


@register_field_type
class PersonNameField(FieldType):
    """A person's name; accepts a whole name or pre-split components."""

    name: ClassVar[str] = "person_name"
    description: ClassVar[str] = "Person name with nickname, phonetic and swapped-order handling"
    components: ClassVar[tuple[str, ...]] = ("first", "middle", "last", "suffix")
    outputs: ClassVar[tuple[str, ...]] = (
        "first", "middle", "last", "suffix", "credential",
        "first_std", "middle_std", "middle_initial", "last_std", "full_std",
        "first_variants", "first_dm", "last_dm",
    )

    def _parts(self, parts: Parts) -> dict[str, str | None]:
        parsed: dict[str, str | None] = {k: None for k in ("first", "middle", "last", "suffix", "credential")}
        if parts.get("value"):
            parsed.update(parse_person_name(parts["value"]))
        for comp in ("first", "middle", "last", "suffix"):
            if parts.get(comp):
                parsed[comp] = parts[comp]
        # A component-supplied "last" may itself carry credentials/suffix ("Smith MD").
        if parts.get("last") and not parts.get("value"):
            sub = parse_person_name(f"{parsed.get('first') or 'X'} {parts['last']}")
            parsed["last"] = sub["last"]
            parsed["suffix"] = parsed["suffix"] or sub["suffix"]
            parsed["credential"] = parsed["credential"] or sub["credential"]
        return parsed

    def validate(self, parts: Parts) -> ValidationResult:
        parsed = self._parts(parts)
        if not parsed["last"] and not parsed["first"]:
            return ValidationResult.fail("no_name_tokens")
        if not any(_LETTERS_RE.search(parsed[k] or "") for k in ("first", "last")):
            return ValidationResult.fail("no_letters")
        return ValidationResult.ok()

    def standardize(self, parts: Parts) -> dict[str, Any]:
        p = self._parts(parts)
        first_std = basic_clean(p["first"]) or None if p["first"] else None
        middle_std = basic_clean(p["middle"]) or None if p["middle"] else None
        last_std = basic_clean(p["last"]) or None if p["last"] else None
        full_std = " ".join(t for t in (first_std, middle_std, last_std) if t) or None
        return {
            "first": p["first"], "middle": p["middle"], "last": p["last"],
            "suffix": p["suffix"], "credential": p["credential"],
            "first_std": first_std, "middle_std": middle_std,
            "middle_initial": middle_std[0] if middle_std else None,
            "last_std": last_std, "full_std": full_std,
            "first_variants": name_variants(first_std),
            "first_dm": phonetic(first_std) if first_std else None,
            "last_dm": phonetic(last_std) if last_std else None,
        }

    def comparisons(self, field_name: str, options: ComparisonOptions):
        import splink.comparison_level_library as cll
        import splink.comparison_library as cl

        f, l_ = f"{field_name}_first_std", f"{field_name}_last_std"
        jw = options.jaro_winkler_thresholds
        first_levels = [
            cll.NullLevel(f),
            cll.ExactMatchLevel(f, term_frequency_adjustments=options.term_frequency)
            .configure(label_for_charts="exact first name"),
            cll.ArrayIntersectLevel(f"{field_name}_first_variants", min_intersection=1)
            .configure(label_for_charts="nickname / formal-name equivalent"),
            *[cll.JaroWinklerLevel(f, t).configure(label_for_charts=f"first JW >= {t}") for t in jw],
        ]
        last_levels = [
            cll.NullLevel(l_),
            cll.ExactMatchLevel(l_, term_frequency_adjustments=options.term_frequency)
            .configure(label_for_charts="exact last name"),
            cll.ColumnsReversedLevel(f, l_).configure(label_for_charts="first/last swapped"),
            *[cll.JaroWinklerLevel(l_, t).configure(label_for_charts=f"last JW >= {t}") for t in jw],
        ]
        if options.include_phonetic:
            first_levels.append(
                cll.ExactMatchLevel(f"{field_name}_first_dm").configure(label_for_charts="first phonetic match")
            )
            last_levels.append(
                cll.ExactMatchLevel(f"{field_name}_last_dm").configure(label_for_charts="last phonetic match")
            )
        first_levels.append(cll.ElseLevel())
        last_levels.append(cll.ElseLevel())
        return [
            cl.CustomComparison(first_levels, output_column_name=f"{field_name}_first",
                                comparison_description="First name (graded)"),
            cl.CustomComparison(last_levels, output_column_name=f"{field_name}_last",
                                comparison_description="Last name (graded, swap-aware)"),
        ]
