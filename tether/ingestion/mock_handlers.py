"""Offline handlers backing ``MockLLMClient`` (heuristic stand-ins for each purpose)."""

from __future__ import annotations

import re
from typing import Any

from rapidfuzz import fuzz

from tether.fields._text import basic_clean
from tether.fields.job_title import DEFAULT_TITLE_TAXONOMY
from tether.fields.org_name import DEFAULT_ORG_ALIASES, strip_legal_suffix

_HEADER_HINTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bnpi\b"), "npi"),
    (re.compile(r"\b(ein|tin|tax ?id|fein)\b"), "ein"),
    (re.compile(r"\b(e-?mail)\b"), "email"),
    (re.compile(r"\b(phone|tel|telephone|fax)\b"), "phone"),
    (re.compile(r"\b(first|given|fname)\b"), "name_first"),
    (re.compile(r"\b(last|surname|lname|family)\b"), "name_last"),
    (re.compile(r"\b(middle|mi|minitial)\b"), "name_middle"),
    (re.compile(r"\bsuffix\b"), "name_suffix"),
    (re.compile(r"\b(zip|postal)\b"), "address_zip"),
    (re.compile(r"\bcity\b"), "address_city"),
    (re.compile(r"\bstate\b"), "address_state"),
    (re.compile(r"\b(street|addr(ess)? ?(line)? ?1|line ?1|addr1)\b"), "address_street"),
    (re.compile(r"\b(address|addr|location)\b"), "address"),
    (re.compile(r"\b(title|specialty|role|position|job)\b"), "title"),
    (re.compile(r"\b(org|organization|practice|employer|billing|company|group|facility|entity)\b"), "org"),
    (re.compile(r"\b(name|provider|physician|clinician)\b"), "name"),
]


def _norm_header(h: str) -> str:
    return re.sub(r"[_\-./]+", " ", h).lower().strip()


def mock_schema_mapping(payload: dict[str, Any]) -> dict[str, Any]:
    targets = set(payload["targets"])
    used: set[str] = set()
    out = []
    for col in payload["columns"]:
        header = _norm_header(col["name"])
        target, conf, why = "UNMAPPED", 0.0, "no header hint"
        for pat, cand in _HEADER_HINTS:
            if cand in targets and cand not in used and pat.search(header):
                target, conf, why = cand, 0.8, f"header hint '{pat.pattern}'"
                break
        if target == "UNMAPPED":
            best = max(((fuzz.token_set_ratio(header, t.replace("_", " ")), t) for t in targets - used), default=(0, ""))
            if best[0] >= 75:
                target, conf, why = best[1], 0.55, f"fuzzy header match ({best[0]})"
        if target != "UNMAPPED":
            used.add(target)
        out.append({"source_column": col["name"], "target": target, "confidence": conf, "rationale": why})
    return {"columns": out}


def mock_standardize_titles(payload: dict[str, Any]) -> dict[str, Any]:
    allowed = payload["allowed"]
    items = []
    for v in payload["values"]:
        std = basic_clean(v)
        canon = DEFAULT_TITLE_TAXONOMY.get(std)
        conf = 0.95 if canon else 0.0
        if not canon:
            score, best = max(((fuzz.token_set_ratio(std, k), c) for k, c in DEFAULT_TITLE_TAXONOMY.items()), default=(0, "OTHER"))
            canon, conf = (best, 0.6) if score >= 80 else ("OTHER", 0.5)
        if canon not in allowed:
            canon = "OTHER"
        items.append({"value": v, "canonical": canon, "confidence": conf})
    return {"items": items}


def mock_standardize_org_aliases(payload: dict[str, Any]) -> dict[str, Any]:
    items = []
    for v in payload["values"]:
        cleaned, _ = strip_legal_suffix(basic_clean(v))
        tokens = [DEFAULT_ORG_ALIASES.get(t, t) for t in cleaned.split()]
        expanded = " ".join(t for t in tokens if t)
        items.append({"value": v, "canonical": expanded, "confidence": 0.9 if expanded != cleaned else 0.99})
    return {"items": items}


def mock_extract_identifiers(payload: dict[str, Any]) -> dict[str, Any]:
    text = payload["text"]
    cands = [{"kind": "npi", "value": m} for m in re.findall(r"\b\d{10}\b", text)]
    cands += [{"kind": "ein", "value": m} for m in re.findall(r"\b\d{2}-\d{7}\b", text)]
    return {"candidates": cands}


DEFAULT_HANDLERS = {
    "schema_mapping": mock_schema_mapping,
    "standardize_titles": mock_standardize_titles,
    "standardize_org_aliases": mock_standardize_org_aliases,
    "extract_identifiers": mock_extract_identifiers,
}
