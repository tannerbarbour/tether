"""ZIP centroid lookup from a Census ZCTA gazetteer-style file (offline)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


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
