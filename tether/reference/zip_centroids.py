"""ZIP centroid reference (Census ZCTA Gazetteer format, offline)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tether.reference.base import ReferenceSource


def load_zip_centroids(path: str | Path) -> dict[str, tuple[float, float]]:
    """Return ``{zip5: (lat, lon)}`` from a CSV or tab-delimited gazetteer.

    Accepts the Census Gazetteer columns (``GEOID``, ``INTPTLAT``, ``INTPTLONG``) with
    either delimiter; column names are matched case-insensitively after stripping.
    """
    path = Path(path)
    sep = "\t" if path.suffix.lower() in {".txt", ".tsv"} else ","
    df = pd.read_csv(path, sep=sep, dtype=str)
    df.columns = [c.strip().upper() for c in df.columns]
    zip_col = next(c for c in ("GEOID", "ZIP", "ZCTA5", "ZIP5") if c in df.columns)
    out: dict[str, tuple[float, float]] = {}
    for z, lat, lon in zip(df[zip_col], df["INTPTLAT"], df["INTPTLONG"]):
        z = str(z).strip().zfill(5)[:5]
        try:
            out[z] = (float(lat), float(lon))
        except (TypeError, ValueError):
            continue
    return out


class ZipCentroidReference(ReferenceSource):
    name = "zip_centroids"
    license = "Public domain (U.S. Census Bureau Gazetteer Files, ZCTA)"
    license_url = "https://www.census.gov/geographies/reference-files/time-series/geo/gazetteer-files.html"

    def _load(self) -> pd.DataFrame:
        items = load_zip_centroids(self.path).items()
        return pd.DataFrame([{"zip5": z, "lat": la, "lon": lo} for z, (la, lo) in items])

    def as_dict(self) -> dict[str, tuple[float, float]]:
        return {r.zip5: (r.lat, r.lon) for r in self.frame().itertuples()}
