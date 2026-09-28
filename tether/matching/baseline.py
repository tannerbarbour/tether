"""Baseline matcher replicating the pre-pipeline approach: one fuzzy score on name + org."""

from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from tether.fields._text import to_text


def baseline_string(records: pd.DataFrame) -> pd.Series:
    """``"<first> <last> <org>"`` lower-cased from canonical (mapped, uncleaned) columns."""
    def build(row: pd.Series) -> str:
        name = to_text(row.get("name")) or " ".join(
            t for t in (to_text(row.get("name_first")), to_text(row.get("name_last"))) if t
        )
        return f"{name} {to_text(row.get('org')) or ''}".lower().strip()

    return records.apply(build, axis=1)


def baseline_scores(records: pd.DataFrame, cross_source_only: bool = False) -> pd.DataFrame:
    """Score every record pair with ``token_sort_ratio`` (0-100). Returns ``unique_id_l/r, score``.

    All-pairs is what the ad hoc scripts do; it is O(n^2) and fine at pilot scale.
    """
    strings = baseline_string(records).tolist()
    ids = records["unique_id"].tolist()
    sources = records["source_dataset"].tolist()
    mat = process.cdist(strings, strings, scorer=fuzz.token_sort_ratio, workers=-1)
    iu, ju = np.triu_indices(len(ids), k=1)
    out = pd.DataFrame({"unique_id_l": np.array(ids)[iu], "unique_id_r": np.array(ids)[ju], "score": mat[iu, ju]})
    if cross_source_only:
        src = np.array(sources)
        out = out[src[iu] != src[ju]].reset_index(drop=True)
    return out


def baseline_links(scores: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Pairs at/above ``threshold`` (0-100)."""
    return scores[scores["score"] >= threshold].reset_index(drop=True)


def baseline_npi_name_scores(records: pd.DataFrame, npi_column: str = "npi_digits") -> pd.DataFrame:
    """Stronger ad hoc baseline: NPI digits equal -> 100, else ``token_sort_ratio`` on the name only.

    Uses raw NPI digits (no Luhn check, no Type 1/2 distinction), which is how a typical
    one-off script treats the identifier.
    """
    def name(row: pd.Series) -> str:
        n = to_text(row.get("name")) or " ".join(
            t for t in (to_text(row.get("name_first")), to_text(row.get("name_last"))) if t)
        return n.lower().strip()

    strings = records.apply(name, axis=1).tolist()
    ids = np.array(records["unique_id"].tolist())
    npi = records[npi_column].map(to_text).to_numpy(dtype=object) if npi_column in records.columns else np.array([None] * len(ids))
    mat = process.cdist(strings, strings, scorer=fuzz.token_sort_ratio, workers=-1)
    iu, ju = np.triu_indices(len(ids), k=1)
    score = mat[iu, ju].astype(float)
    same_npi = np.array([a is not None and a == b for a, b in zip(npi[iu], npi[ju])])
    score[same_npi] = 100.0
    return pd.DataFrame({"unique_id_l": ids[iu], "unique_id_r": ids[ju], "score": score})
