"""Shared, dialect-free text helpers used by field standardizers.

Every function here is a pure function of a single value so it can be applied
with ``pandas.Series.map`` today and wrapped in a Spark UDF later without change.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")
_DIGITS_RE = re.compile(r"\D")


def is_missing(value: Any) -> bool:
    """Return True for None, NaN, empty and whitespace-only strings."""
    if value is None:
        return True
    if isinstance(value, float) and value != value:  # NaN
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    try:  # pandas NA / NaT
        import pandas as pd

        if value is pd.NA or value is pd.NaT:
            return True
    except ImportError:  # pragma: no cover
        pass
    return False


def to_text(value: Any) -> str | None:
    """Coerce a scalar to a stripped string, or None if missing."""
    if is_missing(value):
        return None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).strip()


def strip_accents(text: str) -> str:
    """Remove diacritics: ``José`` -> ``Jose``."""
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
    )


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace to a single space and trim."""
    return _WS_RE.sub(" ", text).strip()


def strip_punctuation(text: str) -> str:
    """Replace punctuation with spaces (so ``O'Brien`` -> ``O Brien``), then collapse."""
    return normalize_whitespace(_PUNCT_RE.sub(" ", text))


def basic_clean(text: str) -> str:
    """Uppercase, strip accents and punctuation, collapse whitespace."""
    return strip_punctuation(strip_accents(text).upper())


def digits_only(text: str) -> str:
    """Keep only ASCII digits."""
    return _DIGITS_RE.sub("", text)


def phonetic(text: str) -> str | None:
    """Primary Double Metaphone code for a token or short phrase, or None if empty."""
    from doublemetaphone import doublemetaphone

    cleaned = basic_clean(text).replace(" ", "")
    if not cleaned:
        return None
    primary, _secondary = doublemetaphone(cleaned)
    return primary or None
