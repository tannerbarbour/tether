"""Local file-backed knowledge base (CSV per table, Parquet for comparison vectors)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from tether.config import EngagementConfig
from tether.knowledge.base import ASSET_COLUMNS, TABLES, KnowledgeBase, now_iso

_PARQUET_TABLES = {"labeled_pairs"}


class LocalKnowledgeBase(KnowledgeBase):
    """Tables live as ``<path>/<table>.csv`` (or ``.parquet``); ``manifest.json`` tracks versions."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def _file(self, table: str) -> Path:
        if table not in TABLES:
            raise KeyError(f"unknown knowledge-base table '{table}'; known: {TABLES}")
        return self.path / (f"{table}.parquet" if table in _PARQUET_TABLES else f"{table}.csv")

    def read(self, table: str) -> pd.DataFrame:
        f = self._file(table)
        if not f.exists():
            return pd.DataFrame(columns=ASSET_COLUMNS)
        df = pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f, dtype=str, keep_default_na=False)
        df = df.astype(object)
        for c in ASSET_COLUMNS:
            if c not in df.columns:
                df[c] = ""
        return df

    def append(self, table: str, rows: pd.DataFrame) -> int:
        if rows.empty:
            return 0
        current = self.read(table)
        merged = pd.concat([current, rows.astype(object)], ignore_index=True) if not current.empty else rows
        self.replace(table, merged)
        return len(rows)

    def replace(self, table: str, df: pd.DataFrame) -> None:
        f = self._file(table)
        if f.suffix == ".parquet":
            df.to_parquet(f, index=False)
        else:
            df.to_csv(f, index=False)
        self._update_manifest()

    def version(self) -> str:
        h = hashlib.sha256()
        for t in TABLES:
            f = self._file(t)
            if f.exists():
                h.update(t.encode())
                h.update(f.read_bytes())
        return h.hexdigest()[:12]

    def _update_manifest(self) -> None:
        manifest = {"version": self.version(), "updated_at": now_iso(),
                    "tables": {t: int(len(self.read(t))) for t in TABLES if self._file(t).exists()}}
        (self.path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def open_knowledge_base(config: EngagementConfig) -> KnowledgeBase:
    """The configured knowledge base (local implementation in the pilot)."""
    return LocalKnowledgeBase(config.resolve(config.knowledge_base.path))
