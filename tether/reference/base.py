"""ReferenceSource: an external dataset with snapshot versioning and license metadata."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class SnapshotInfo:
    source: str
    snapshot_id: str
    path: str
    license: str
    license_url: str
    n_rows: int
    file_bytes: int

    def as_dict(self) -> dict:
        return asdict(self)


def file_snapshot_id(path: Path) -> str:
    """Stable id from file content (first 16 hex of SHA-256)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


class ReferenceSource(ABC):
    """A versioned, licensed external dataset loaded from a local snapshot file."""

    name: str
    license: str
    license_url: str

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"{self.name} snapshot not found: {self.path}")
        self._df: pd.DataFrame | None = None
        self.snapshot_id = file_snapshot_id(self.path)

    @abstractmethod
    def _load(self) -> pd.DataFrame:
        """Read and normalize the snapshot file."""

    def frame(self) -> pd.DataFrame:
        if self._df is None:
            self._df = self._load()
        return self._df

    def snapshot(self) -> SnapshotInfo:
        return SnapshotInfo(source=self.name, snapshot_id=self.snapshot_id, path=str(self.path),
                            license=self.license, license_url=self.license_url, n_rows=len(self.frame()),
                            file_bytes=self.path.stat().st_size)
