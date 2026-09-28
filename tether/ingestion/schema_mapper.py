"""AI-assisted schema mapping: deterministic pre-match, then the LLM for what is left.

Flow: profile headers + previously approved mappings + value-shape detection resolve as many
columns as possible without any LLM call. Only unresolved columns (headers plus a small,
optionally redacted sample) go to the model. The result is written as a *proposed*
``SchemaMapping`` YAML for analyst approval; nothing is applied until approved.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import pandas as pd

from tether.config import ColumnMappingEntry, SchemaMapping
from tether.fields._text import to_text
from tether.fields.contact import normalize_phone
from tether.fields.identifiers import is_valid_ein, is_valid_npi
from tether.ingestion.llm import LLMClient
from tether.ingestion.llm_log import LLMCallLogger
from tether.ingestion.prompts import MappingProposal, schema_mapping_request
from tether.profiles.base import EntityProfile

UNMAPPED = "UNMAPPED"
Redactor = Callable[[str, list[str]], list[str]]
"""``(column name, sample values) -> sample values to send``."""


def _norm(header: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", header.lower()).strip("_")


def mask_value(value: str) -> str:
    """Shape mask: digits -> 9, letters -> a/A, everything else kept (``62701-1234`` -> ``99999-9999``)."""
    return re.sub(r"[a-z]", "a", re.sub(r"[A-Z]", "A", re.sub(r"\d", "9", value)))


def default_redactor(redact_columns: list[str]) -> Redactor:
    cols = {_norm(c) for c in redact_columns}

    def redact(column: str, samples: list[str]) -> list[str]:
        return [mask_value(s) for s in samples] if _norm(column) in cols else samples

    return redact


# ------------------------------------------------------- deterministic stage
_SHAPE_DETECTORS: list[tuple[str, Callable[[str], bool]]] = [
    ("npi", lambda v: is_valid_npi(v)[0]),
    ("ein", lambda v: is_valid_ein(v)[0] and bool(re.fullmatch(r"\d{2}-?\d{7}", v.strip()))),
    ("email", lambda v: bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", v.strip()))),
    ("phone", lambda v: normalize_phone(v) is not None and not v.strip().isdigit()),
    ("address_zip", lambda v: bool(re.fullmatch(r"\d{5}(-\d{4})?", v.strip()))),
]


def detect_shape(values: list[str], min_share: float = 0.9) -> tuple[str, float] | None:
    """Return (target, share) when >= ``min_share`` of non-null values match one identifier shape."""
    vals = [v for v in values if v]
    if len(vals) < 5:
        return None
    for target, fn in _SHAPE_DETECTORS:
        share = sum(fn(v) for v in vals) / len(vals)
        if share >= min_share:
            return target, share
    return None


def prematch(
    raw: pd.DataFrame, profile: EntityProfile, examples: list[SchemaMapping], sample_size: int = 200
) -> tuple[dict[str, ColumnMappingEntry], list[str]]:
    """Resolve columns without the LLM. Returns (resolved entries by source column, unresolved columns)."""
    targets = set(profile.canonical_columns())
    approved: dict[str, tuple[str, int]] = {}
    for ex in examples:
        for e in ex.columns:
            key = _norm(e.source_column)
            approved[key] = (e.target, approved.get(key, ("", 0))[1] + 1)
    resolved: dict[str, ColumnMappingEntry] = {}
    unresolved: list[str] = []
    used: set[str] = set()
    for col in raw.columns:
        key = _norm(col)
        entry = None
        if key in targets and key not in used:
            entry = ColumnMappingEntry(source_column=col, target=key, confidence=0.99,
                                       rationale="header equals canonical column name")
        elif key in approved and approved[key][0] in targets and approved[key][0] not in used:
            target, n = approved[key]
            entry = ColumnMappingEntry(source_column=col, target=target, confidence=0.97,
                                       rationale=f"header previously approved -> {target} ({n} mapping(s))")
        else:
            samples = [to_text(v) for v in raw[col].head(sample_size)]
            hit = detect_shape([s for s in samples if s])
            if hit and hit[0] in targets and hit[0] not in used:
                entry = ColumnMappingEntry(source_column=col, target=hit[0], confidence=round(0.8 + 0.15 * hit[1], 3),
                                           rationale=f"{hit[1]:.0%} of sampled values have {hit[0]} shape")
        if entry:
            resolved[col] = entry
            used.add(entry.target)
        else:
            unresolved.append(col)
    return resolved, unresolved


# ------------------------------------------------------------------ LLM stage
def build_llm_payload(
    raw: pd.DataFrame, columns: list[str], profile: EntityProfile, used_targets: set[str],
    examples: list[SchemaMapping], sample_rows: int, redactor: Redactor,
) -> dict[str, Any]:
    sample = raw.head(sample_rows) if sample_rows else raw.head(0)
    cols = []
    for c in columns:
        values = [to_text(v) for v in sample[c]]
        values = [v for v in values if v]
        cols.append({"name": c, "samples": redactor(c, values), "n_distinct": int(raw[c].nunique()),
                     "share_missing": round(float(raw[c].isna().mean()), 3)})
    targets = {k: v for k, v in profile.mapping_targets().items() if k not in used_targets}
    targets[UNMAPPED] = "no suitable canonical target"
    ex = [{"source": e.source, "columns": [{"source_column": c.source_column, "target": c.target} for c in e.columns]}
          for e in examples[:5]]
    return {"columns": cols, "targets": targets, "examples": ex, "profile": profile.name}


def propose_mapping(
    raw: pd.DataFrame,
    source_name: str,
    profile: EntityProfile,
    client: LLMClient,
    *,
    record_id_column: str | None = None,
    examples: list[SchemaMapping] | None = None,
    sample_rows: int = 20,
    redact_columns: list[str] | None = None,
    redactor: Redactor | None = None,
    logger: LLMCallLogger | None = None,
) -> SchemaMapping:
    """Propose a canonical mapping for ``raw``. The result has ``status='proposed'``."""
    examples = examples or []
    redactor = redactor or default_redactor(redact_columns or [])
    resolved, unresolved = prematch(raw, profile, examples)
    used = {e.target for e in resolved.values()}
    if record_id_column in unresolved:
        unresolved.remove(record_id_column)
    llm_used = False
    if unresolved:
        payload = build_llm_payload(raw, unresolved, profile, used, examples, sample_rows, redactor)
        request = schema_mapping_request(payload)
        proposal, record = client.complete(request, MappingProposal)
        llm_used = True
        if logger:
            logger.write(record, source=source_name, n_columns=len(unresolved))
        allowed = set(profile.canonical_columns())
        for item in sorted(proposal.columns, key=lambda i: -i.confidence):
            if item.source_column not in unresolved or item.target == UNMAPPED:
                continue
            if item.target not in allowed:
                continue  # guardrail: targets outside the schema are ignored
            if item.target in used:
                continue  # each target once; higher-confidence proposal wins
            resolved[item.source_column] = ColumnMappingEntry(
                source_column=item.source_column, target=item.target,
                confidence=round(item.confidence, 3), rationale=f"[{client.provider}] {item.rationale}")
            used.add(item.target)
    ordered = [resolved[c] for c in raw.columns if c in resolved]
    unmapped = [c for c in raw.columns if c not in resolved and c != record_id_column]
    return SchemaMapping(
        source=source_name, profile=profile.name, status="proposed", version=1,
        record_id_column=record_id_column, columns=ordered, unmapped=unmapped,
        provenance={"method": "prematch+llm" if llm_used else "prematch", "llm_provider": client.provider,
                    "llm_model": client.model, "n_examples": len(examples), "sample_rows": sample_rows,
                    "redacted_columns": list(redact_columns or [])},
    )
