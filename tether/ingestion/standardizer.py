"""Distinct-value standardization with a lookup cache in front of the LLM.

The LLM sees distinct values only, in batches, and its output is validated (allowed
category set, no identifier leakage) before it is cached as an *engagement_local* lookup
row. Promotion to reuse across engagements is a separate, reviewed step.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd

from tether.fields._text import basic_clean
from tether.fields.job_title import DEFAULT_TITLE_TAXONOMY
from tether.fields.org_name import strip_legal_suffix
from tether.ingestion.guardrails import contains_identifier_like
from tether.ingestion.llm import LLMClient
from tether.ingestion.llm_log import LLMCallLogger
from tether.ingestion.prompts import ValueMappingBatch, org_alias_request, title_request

LOOKUP_COLUMNS = ["value", "canonical", "confidence", "engagement", "reviewer", "created_at",
                  "version", "reuse_scope", "state", "method"]
Kind = Literal["job_title", "org_alias"]


class LookupCache:
    """A CSV-backed lookup table with provenance columns (the KB's ``lookup_*`` layout).

    Rows usable for an engagement: ``promoted`` + ``generic`` rows from anywhere, plus
    every row created by the engagement itself.
    """

    def __init__(self, path: str | Path, engagement: str, use_promoted_only: bool = True):
        self.path = Path(path)
        self.engagement = engagement
        self.use_promoted_only = use_promoted_only
        if self.path.exists():
            self.df = pd.read_csv(self.path, dtype=str, keep_default_na=False)
            for c in LOOKUP_COLUMNS:
                if c not in self.df.columns:
                    self.df[c] = ""
        else:
            self.df = pd.DataFrame(columns=LOOKUP_COLUMNS)

    @staticmethod
    def key(value: str) -> str:
        return basic_clean(value)

    def usable(self) -> pd.DataFrame:
        df = self.df
        own = df["engagement"] == self.engagement
        shared = (df["state"] == "promoted") & (df["reuse_scope"] == "generic")
        if not self.use_promoted_only:
            shared = shared | (df["state"].isin(["reviewed", "promoted"]))
        return df[own | shared]

    def lookup(self) -> dict[str, str]:
        """``{cleaned value: canonical}`` for usable rows (own engagement rows win)."""
        usable = self.usable().copy()
        usable["_own"] = usable["engagement"] == self.engagement
        usable = usable.sort_values(["_own", "confidence"], ascending=[True, True])
        return dict(zip(usable["value"], usable["canonical"]))

    def add(self, items: Iterable[tuple[str, str, float]], method: str, reviewer: str = "") -> int:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        known = set(self.usable()["value"])
        rows = [{
            "value": self.key(v), "canonical": c, "confidence": f"{conf:.3f}", "engagement": self.engagement,
            "reviewer": reviewer, "created_at": now, "version": "1", "reuse_scope": "client_scoped",
            "state": "engagement_local", "method": method,
        } for v, c, conf in items if self.key(v) not in known]
        if rows:
            self.df = pd.concat([self.df, pd.DataFrame(rows)], ignore_index=True)
        return len(rows)

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(self.path, index=False)
        return self.path


def value_key(value: str, kind: Kind) -> str:
    """Cache key: cleaned value; org names also lose trailing legal suffixes."""
    cleaned = basic_clean(value)
    return strip_legal_suffix(cleaned)[0] if kind == "org_alias" else cleaned


def _batches(values: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(values), size):
        yield values[i:i + size]


def standardize_values(
    values: Iterable[str],
    kind: Kind,
    client: LLMClient,
    cache: LookupCache,
    *,
    batch_size: int = 50,
    allowed_titles: Iterable[str] | None = None,
    logger: LLMCallLogger | None = None,
) -> dict:
    """Map distinct ``values`` to canonical forms, calling the LLM only for cache misses.

    Keys are cleaned values (org names additionally have legal suffixes stripped, matching
    what ``OrgNameField`` looks up). Returns ``{"mapping": {key: canonical}, "n_distinct",
    "n_cached", "n_builtin", "n_llm", "n_rejected", "calls"}``.
    """
    distinct = sorted({value_key(v, kind) for v in values if v and value_key(v, kind)})
    cached = cache.lookup()
    builtin = dict(DEFAULT_TITLE_TAXONOMY) if kind == "job_title" else {}
    mapping = {v: cached[v] for v in distinct if v in cached}
    n_cached = len(mapping)
    mapping.update({v: builtin[v] for v in distinct if v in builtin and v not in mapping})
    misses = [v for v in distinct if v not in mapping]
    allowed = sorted(set(allowed_titles or set(DEFAULT_TITLE_TAXONOMY.values())) | {"OTHER"})
    n_llm = n_rejected = calls = 0
    new_rows: list[tuple[str, str, float]] = []
    for batch in _batches(misses, batch_size):
        request = title_request(batch, allowed) if kind == "job_title" else org_alias_request(batch)
        result, record = client.complete(request, ValueMappingBatch)
        calls += 1
        if logger:
            logger.write(record, kind=kind, n_values=len(batch))
        for item in result.items:
            key = value_key(item.value, kind)
            canonical = basic_clean(item.canonical) if kind == "org_alias" else item.canonical.strip().upper()
            if key not in batch or not canonical:
                n_rejected += 1
                continue
            if kind == "job_title" and canonical not in allowed:
                n_rejected += 1
                continue
            if contains_identifier_like(canonical):
                n_rejected += 1
                continue
            mapping[key] = canonical
            new_rows.append((key, canonical, item.confidence))
            n_llm += 1
    cache.add(new_rows, method=f"llm:{client.provider}:{client.model}")
    return {"mapping": mapping, "n_distinct": len(distinct), "n_cached": n_cached,
            "n_builtin": len(distinct) - len(misses) - n_cached, "n_llm": n_llm, "n_rejected": n_rejected,
            "calls": calls}


def extract_identifiers(text: str, client: LLMClient, logger: LLMCallLogger | None = None) -> list[dict]:
    """LLM-assisted identifier extraction with the verbatim + validation guardrail applied."""
    from tether.ingestion.guardrails import verify_identifier
    from tether.ingestion.prompts import IdentifierExtraction, extraction_request

    result, record = client.complete(extraction_request(text), IdentifierExtraction)
    if logger:
        logger.write(record)
    checks = [verify_identifier(c.value, c.kind, text) for c in result.candidates]
    return [c.__dict__ for c in checks]
