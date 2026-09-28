"""KnowledgeBase interface.

Every asset row carries: ``asset_id``, ``engagement``, ``reviewer``, ``created_at``,
``version``, ``reuse_scope`` (generic | client_scoped) and ``state``
(engagement_local | reviewed | promoted). Cross-engagement reuse is restricted to
``promoted`` + ``generic`` rows; an engagement always sees its own rows.

The pilot backend is local CSV/Parquet; the interface is what the engine uses so a
Fabric Lakehouse (Delta) backend can replace it without engine changes.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

import pandas as pd

from tether.config import PromotionState, ReuseScope, SchemaMapping
from tether.profiles.base import FieldSpec

TABLES = (
    "schema_mappings", "lookup_org_aliases", "lookup_titles", "model_params",
    "labeled_pairs", "run_history", "reference_snapshots",
)
ASSET_COLUMNS = ["asset_id", "engagement", "reviewer", "created_at", "version", "reuse_scope", "state"]
STATE_ORDER: dict[str, int] = {"engagement_local": 0, "reviewed": 1, "promoted": 2}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_asset_id() -> str:
    return uuid.uuid4().hex[:12]


def comparison_schema_hash(field_specs: Mapping[str, FieldSpec], profile_name: str) -> str:
    """Hash of the comparison definitions; m-values and gamma vectors are only reusable under it."""
    spec = {
        "profile": profile_name,
        "fields": {n: {"type": s.field_type, "role": s.role, "tf": s.term_frequency, "phonetic": s.include_phonetic,
                       "jw": list(s.jaro_winkler_thresholds)}
                   for n, s in sorted(field_specs.items()) if s.role != "report_only"},
    }
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]


def with_asset_columns(rows: pd.DataFrame, engagement: str, reuse_scope: ReuseScope = "client_scoped",
                       state: PromotionState = "engagement_local", reviewer: str = "") -> pd.DataFrame:
    """Fill any missing asset metadata columns."""
    df = rows.copy()
    defaults = {"asset_id": None, "engagement": engagement, "reviewer": reviewer, "created_at": now_iso(),
                "version": 1, "reuse_scope": reuse_scope, "state": state}
    for c, v in defaults.items():
        if c not in df.columns:
            df[c] = v
        else:
            df[c] = df[c].where(df[c].notna() & (df[c].astype(str) != ""), v)
    df["asset_id"] = [a if isinstance(a, str) and a else new_asset_id() for a in df["asset_id"]]
    return df[ASSET_COLUMNS + [c for c in df.columns if c not in ASSET_COLUMNS]]


class KnowledgeBase(ABC):
    """Storage-agnostic knowledge base."""

    # ------------------------------------------------------------ primitives
    @abstractmethod
    def read(self, table: str) -> pd.DataFrame:
        """Full table (empty frame with asset columns if absent)."""

    @abstractmethod
    def append(self, table: str, rows: pd.DataFrame) -> int:
        """Append rows (already carrying asset columns). Returns rows written."""

    @abstractmethod
    def replace(self, table: str, df: pd.DataFrame) -> None:
        """Overwrite a table (used by promotion)."""

    @abstractmethod
    def version(self) -> str:
        """Content-derived version identifier of the whole knowledge base."""

    # ------------------------------------------------------------ visibility
    def usable(self, table: str, engagement: str, use_promoted_only: bool = True) -> pd.DataFrame:
        df = self.read(table)
        if df.empty:
            return df
        own = df["engagement"] == engagement
        shared = (df["state"] == "promoted") & (df["reuse_scope"] == "generic")
        if not use_promoted_only:
            shared |= df["state"].isin(["reviewed", "promoted"]) & (df["reuse_scope"] == "generic")
        return df[own | shared]

    # ------------------------------------------------------------- promotion
    def promote(self, table: str, *, to_state: PromotionState, reviewer: str, engagement: str | None = None,
                asset_ids: Iterable[str] | None = None, reuse_scope: ReuseScope | None = None) -> int:
        """Advance assets to ``to_state`` (never backwards). Requires a reviewer.

        Selects by ``asset_ids`` or by ``engagement`` (all of its rows below the target state).
        Promotion to ``promoted`` with ``reuse_scope='generic'`` is what makes an asset
        reusable across engagements.
        """
        if not reviewer:
            raise ValueError("promotion requires a reviewer")
        df = self.read(table)
        if df.empty:
            return 0
        mask = pd.Series(True, index=df.index)
        if asset_ids is not None:
            mask &= df["asset_id"].isin(set(asset_ids))
        if engagement is not None:
            mask &= df["engagement"] == engagement
        if asset_ids is None and engagement is None:
            raise ValueError("select assets by asset_ids or engagement")
        mask &= df["state"].map(STATE_ORDER).fillna(0) < STATE_ORDER[to_state]
        n = int(mask.sum())
        if n:
            df.loc[mask, "state"] = to_state
            df.loc[mask, "reviewer"] = reviewer
            df = df.astype(object)
            df.loc[mask, "version"] = (pd.to_numeric(df.loc[mask, "version"], errors="coerce").fillna(1).astype(int) + 1).astype(str)
            if reuse_scope:
                df.loc[mask, "reuse_scope"] = reuse_scope
            self.replace(table, df)
        return n

    # ------------------------------------------------------ typed conveniences
    def approved_mappings(self, profile: str, engagement: str, use_promoted_only: bool = True) -> list[SchemaMapping]:
        df = self.usable("schema_mappings", engagement, use_promoted_only)
        out = []
        for _, r in df[df.get("profile", pd.Series(dtype=str)) == profile].iterrows():
            try:
                out.append(SchemaMapping.model_validate(json.loads(r["mapping_json"])))
            except Exception:
                continue
        return out

    def register_mapping(self, mapping: SchemaMapping, engagement: str, reuse_scope: ReuseScope) -> bool:
        """Store an approved mapping once (deduplicated by content hash)."""
        if mapping.status != "approved":
            return False
        payload = mapping.model_dump(mode="json")
        content_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
        existing = self.read("schema_mappings")
        if not existing.empty and (existing["content_hash"] == content_hash).any():
            return False
        row = pd.DataFrame([{"source": mapping.source, "profile": mapping.profile, "content_hash": content_hash,
                             "n_columns": len(mapping.columns), "mapping_json": json.dumps(payload)}])
        self.append("schema_mappings", with_asset_columns(row, engagement, reuse_scope))
        return True

    def m_probabilities(self, profile: str, schema_hash: str, engagement: str, source_types: Iterable[str] = (),
                        use_promoted_only: bool = True) -> dict[str, dict[str, float]]:
        """Starting m-values keyed ``{comparison: {level label: m}}``.

        Rows must match the profile and comparison schema hash. Rows for the same source
        types are preferred; otherwise any matching rows are averaged per level.
        """
        df = self.usable("model_params", engagement, use_promoted_only)
        if df.empty:
            return {}
        df = df[(df["profile"] == profile) & (df["comparison_schema_hash"] == schema_hash)]
        if df.empty:
            return {}
        wanted = set(source_types)
        if wanted:
            preferred = df[df["source_type"].isin(wanted)]
            if not preferred.empty:
                df = preferred
        df = df.assign(m=pd.to_numeric(df["m_probability"], errors="coerce")).dropna(subset=["m"])
        out: dict[str, dict[str, float]] = {}
        for (comp, level), grp in df.groupby(["comparison", "level"]):
            out.setdefault(comp, {})[level] = float(grp["m"].mean())
        return out

    def save_model_params(self, m: Mapping[str, Mapping[str, float]], profile: str, schema_hash: str,
                          engagement: str, source_types: Iterable[str], reuse_scope: ReuseScope,
                          run_id: str) -> int:
        rows = [{"profile": profile, "comparison_schema_hash": schema_hash, "source_type": st, "comparison": comp,
                 "level": level, "m_probability": val, "run_id": run_id}
                for st in (list(source_types) or [""]) for comp, levels in m.items() for level, val in levels.items()]
        if not rows:
            return 0
        return self.append("model_params", with_asset_columns(pd.DataFrame(rows), engagement, reuse_scope))

    def add_labeled_pairs(self, gammas: pd.DataFrame, label: int, label_source: str, profile: str,
                          schema_hash: str, engagement: str, reuse_scope: ReuseScope) -> int:
        """Store comparison vectors (gamma_* columns) with a label. Never raw record values."""
        gamma_cols = [c for c in gammas.columns if c.startswith("gamma_")]
        if not gamma_cols or gammas.empty:
            return 0
        df = gammas[gamma_cols].copy()
        df.insert(0, "profile", profile)
        df.insert(1, "comparison_schema_hash", schema_hash)
        df.insert(2, "label", label)
        df.insert(3, "label_source", label_source)
        return self.append("labeled_pairs", with_asset_columns(df, engagement, reuse_scope))

    def labeled_pairs(self, profile: str, schema_hash: str, engagement: str, use_promoted_only: bool = True) -> pd.DataFrame:
        df = self.usable("labeled_pairs", engagement, use_promoted_only)
        if df.empty:
            return df
        return df[(df["profile"] == profile) & (df["comparison_schema_hash"] == schema_hash)]

    def record_run(self, run: Mapping[str, Any], engagement: str) -> str:
        run_id = run.get("run_id") or new_asset_id()
        row = pd.DataFrame([{"run_id": run_id, **{k: (json.dumps(v, default=str) if isinstance(v, (dict, list)) else v)
                                                   for k, v in run.items() if k != "run_id"}}])
        self.append("run_history", with_asset_columns(row, engagement, "client_scoped"))
        return run_id

    def register_reference_snapshot(self, info: Mapping[str, Any], engagement: str) -> bool:
        existing = self.read("reference_snapshots")
        if not existing.empty and (existing["snapshot_id"] == info["snapshot_id"]).any():
            return False
        self.append("reference_snapshots", with_asset_columns(pd.DataFrame([dict(info)]), engagement, "generic",
                                                              state="promoted"))
        return True

    def status(self) -> pd.DataFrame:
        rows = []
        for t in TABLES:
            df = self.read(t)
            counts = df["state"].value_counts().to_dict() if not df.empty else {}
            rows.append({"table": t, "rows": len(df), **{s: counts.get(s, 0) for s in STATE_ORDER}})
        return pd.DataFrame(rows)
