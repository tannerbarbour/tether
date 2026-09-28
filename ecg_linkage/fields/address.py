"""US address field type: parsing, USPS-style normalization, ZIP-centroid geo, comparisons."""

from __future__ import annotations

import re
from typing import Any, ClassVar, Mapping

from ecg_linkage.fields._text import basic_clean, digits_only, normalize_whitespace
from ecg_linkage.fields.base import ComparisonOptions, FieldType, Parts, ValidationResult
from ecg_linkage.fields.registry import register_field_type

STATE_CODES: dict[str, str] = {
    "ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR", "CALIFORNIA": "CA",
    "COLORADO": "CO", "CONNECTICUT": "CT", "DELAWARE": "DE", "FLORIDA": "FL", "GEORGIA": "GA",
    "HAWAII": "HI", "IDAHO": "ID", "ILLINOIS": "IL", "INDIANA": "IN", "IOWA": "IA", "KANSAS": "KS",
    "KENTUCKY": "KY", "LOUISIANA": "LA", "MAINE": "ME", "MARYLAND": "MD", "MASSACHUSETTS": "MA",
    "MICHIGAN": "MI", "MINNESOTA": "MN", "MISSISSIPPI": "MS", "MISSOURI": "MO", "MONTANA": "MT",
    "NEBRASKA": "NE", "NEVADA": "NV", "NEW HAMPSHIRE": "NH", "NEW JERSEY": "NJ", "NEW MEXICO": "NM",
    "NEW YORK": "NY", "NORTH CAROLINA": "NC", "NORTH DAKOTA": "ND", "OHIO": "OH", "OKLAHOMA": "OK",
    "OREGON": "OR", "PENNSYLVANIA": "PA", "RHODE ISLAND": "RI", "SOUTH CAROLINA": "SC",
    "SOUTH DAKOTA": "SD", "TENNESSEE": "TN", "TEXAS": "TX", "UTAH": "UT", "VERMONT": "VT",
    "VIRGINIA": "VA", "WASHINGTON": "WA", "WEST VIRGINIA": "WV", "WISCONSIN": "WI", "WYOMING": "WY",
    "DISTRICT OF COLUMBIA": "DC", "PUERTO RICO": "PR",
}
VALID_STATES = set(STATE_CODES.values())

STREET_ABBREVIATIONS: dict[str, str] = {
    "STREET": "ST", "AVENUE": "AVE", "AV": "AVE", "ROAD": "RD", "DRIVE": "DR", "BOULEVARD": "BLVD",
    "BOUL": "BLVD", "LANE": "LN", "COURT": "CT", "PLACE": "PL", "PARKWAY": "PKWY", "PKY": "PKWY",
    "HIGHWAY": "HWY", "CIRCLE": "CIR", "TRAIL": "TRL", "TERRACE": "TER", "SQUARE": "SQ",
    "PLAZA": "PLZ", "EXPRESSWAY": "EXPY", "TURNPIKE": "TPKE", "NORTH": "N", "SOUTH": "S",
    "EAST": "E", "WEST": "W", "NORTHEAST": "NE", "NORTHWEST": "NW", "SOUTHEAST": "SE",
    "SOUTHWEST": "SW", "SUITE": "STE", "APARTMENT": "APT", "UNIT": "UNIT", "FLOOR": "FL",
    "BUILDING": "BLDG", "ROOM": "RM", "DEPARTMENT": "DEPT", "FIRST": "1ST", "SECOND": "2ND",
    "THIRD": "3RD", "FOURTH": "4TH", "FIFTH": "5TH", "POST OFFICE BOX": "PO BOX", "P O BOX": "PO BOX",
}
_UNIT_TOKENS = {"STE", "APT", "UNIT", "FL", "BLDG", "RM", "DEPT", "#"}
_ZIP_RE = re.compile(r"\b(\d{5})(?:-?(\d{4}))?\b")
_STATE_ZIP_RE = re.compile(r"\b([A-Z]{2})\s+\d{5}(?:-\d{4})?\s*$")


def normalize_state(text: str | None) -> str | None:
    """``Illinois`` / ``il`` / ``IL.`` -> ``IL``; unknown values are returned cleaned."""
    if not text:
        return None
    cleaned = basic_clean(text)
    if cleaned in VALID_STATES:
        return cleaned
    return STATE_CODES.get(cleaned, cleaned or None)


def normalize_street(text: str | None) -> str | None:
    """Uppercase, strip punctuation, apply USPS abbreviations, drop unit designators."""
    if not text:
        return None
    cleaned = basic_clean(text)
    for phrase in ("POST OFFICE BOX", "P O BOX"):
        cleaned = cleaned.replace(phrase, "PO BOX")
    tokens = [STREET_ABBREVIATIONS.get(t, t) for t in cleaned.split()]
    # Drop unit designator and everything after it (unit is kept separately).
    for i, t in enumerate(tokens):
        if t in _UNIT_TOKENS and i > 0:
            tokens = tokens[:i]
            break
    return normalize_whitespace(" ".join(tokens)) or None


def parse_address(value: str) -> dict[str, str | None]:
    """Parse a single-line US address into street/unit/city/state/zip using usaddress.

    Falls back to regex extraction of state and ZIP when usaddress cannot label
    the string consistently.
    """
    import usaddress

    out: dict[str, str | None] = {"street": None, "unit": None, "city": None, "state": None, "zip": None}
    try:
        tagged, _ = usaddress.tag(value)
    except usaddress.RepeatedLabelError:
        tagged = {}
    if tagged:
        street_keys = [
            "AddressNumberPrefix", "AddressNumber", "AddressNumberSuffix", "StreetNamePreModifier",
            "StreetNamePreDirectional", "StreetNamePreType", "StreetName", "StreetNamePostType",
            "StreetNamePostDirectional", "USPSBoxType", "USPSBoxID", "USPSBoxGroupType", "USPSBoxGroupID",
        ]
        street = " ".join(tagged[k] for k in street_keys if k in tagged)
        unit = " ".join(tagged[k] for k in ("OccupancyType", "OccupancyIdentifier") if k in tagged)
        out.update(
            street=street or None,
            unit=unit or None,
            city=tagged.get("PlaceName"),
            state=tagged.get("StateName"),
            zip=tagged.get("ZipCode"),
        )
    upper = value.upper()
    if not out["zip"]:
        m = _ZIP_RE.search(upper)
        out["zip"] = m.group(0) if m else None
    if not out["state"]:
        m = _STATE_ZIP_RE.search(upper)
        out["state"] = m.group(1) if m else None
    if not out["street"]:
        out["street"] = value.split(",")[0]
    return out


@register_field_type
class AddressField(FieldType):
    """US postal address; accepts a single line or pre-split components.

    ``zip_centroids`` maps ZIP5 -> (lat, lon) and enables the distance-based
    comparison levels. It is injected (e.g. from a ZIP centroid ReferenceSource).
    """

    name: ClassVar[str] = "address"
    description: ClassVar[str] = "US address with parsing, USPS normalization and ZIP-centroid distance"
    components: ClassVar[tuple[str, ...]] = ("street", "city", "state", "zip")
    outputs: ClassVar[tuple[str, ...]] = (
        "street", "unit", "city", "state", "zip5", "zip4", "street_std", "city_std", "full_std", "lat", "lon",
    )

    def __init__(self, zip_centroids: Mapping[str, tuple[float, float]] | None = None):
        self.zip_centroids = dict(zip_centroids or {})

    def _parts(self, parts: Parts) -> dict[str, str | None]:
        parsed: dict[str, str | None] = {"street": None, "unit": None, "city": None, "state": None, "zip": None}
        if parts.get("value"):
            parsed.update(parse_address(parts["value"]))
        for comp in ("street", "city", "state", "zip"):
            if parts.get(comp):
                parsed[comp] = str(parts[comp])
        return parsed

    def validate(self, parts: Parts) -> ValidationResult:
        p = self._parts(parts)
        if p["zip"]:
            z = digits_only(p["zip"])
            if len(z) not in (5, 9):
                return ValidationResult.fail("zip_not_5_or_9_digits")
        if p["state"] and normalize_state(p["state"]) not in VALID_STATES:
            return ValidationResult.fail("unknown_state")
        if not (p["street"] or p["zip"] or p["city"]):
            return ValidationResult.fail("no_address_parts")
        return ValidationResult.ok()

    def standardize(self, parts: Parts) -> dict[str, Any]:
        p = self._parts(parts)
        zdigits = digits_only(p["zip"]) if p["zip"] else ""
        zip5 = zdigits[:5] if len(zdigits) >= 5 else None
        zip4 = zdigits[5:9] if len(zdigits) == 9 else None
        state = normalize_state(p["state"])
        street_std = normalize_street(p["street"])
        city_std = basic_clean(p["city"]) if p["city"] else None
        full_std = " ".join(t for t in (street_std, city_std, state, zip5) if t) or None
        lat, lon = self.zip_centroids.get(zip5, (None, None)) if zip5 else (None, None)
        return {
            "street": p["street"], "unit": p["unit"], "city": p["city"], "state": p["state"],
            "zip5": zip5, "zip4": zip4, "street_std": street_std, "city_std": city_std,
            "full_std": full_std, "lat": lat, "lon": lon,
        }

    def comparisons(self, field_name: str, options: ComparisonOptions):
        import splink.comparison_level_library as cll
        import splink.comparison_library as cl

        street, zip5 = f"{field_name}_street_std", f"{field_name}_zip5"
        lat, lon = f"{field_name}_lat", f"{field_name}_lon"
        levels = [
            cll.And(cll.NullLevel(street), cll.NullLevel(zip5)).configure(is_null_level=True),
            cll.ExactMatchLevel(f"{field_name}_full_std").configure(label_for_charts="exact full address"),
            cll.And(cll.ExactMatchLevel(street), cll.ExactMatchLevel(zip5))
            .configure(label_for_charts="exact street + ZIP"),
            cll.And(cll.JaroWinklerLevel(street, 0.9), cll.ExactMatchLevel(zip5))
            .configure(label_for_charts="similar street + ZIP"),
            cll.ExactMatchLevel(zip5, term_frequency_adjustments=options.term_frequency)
            .configure(label_for_charts="same ZIP5"),
        ]
        if options.geo_available:
            levels += [
                cll.DistanceInKMLevel(lat, lon, 10).configure(label_for_charts="ZIP centroids within 10 km"),
                cll.DistanceInKMLevel(lat, lon, 50).configure(label_for_charts="ZIP centroids within 50 km"),
            ]
        levels += [
            cll.ExactMatchLevel(f"{field_name}_state").configure(label_for_charts="same state"),
            cll.ElseLevel(),
        ]
        return [cl.CustomComparison(levels, output_column_name=field_name,
                                    comparison_description="Address (graded, geo-aware)")]
