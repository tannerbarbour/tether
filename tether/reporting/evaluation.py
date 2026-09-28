"""Evaluation against ground truth: pairwise P/R/F1, PR curves, cluster metrics, blocking recall."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


def pair_key(a: pd.Series | Iterable[str], b: pd.Series | Iterable[str]) -> pd.Series:
    a, b = pd.Series(list(a)), pd.Series(list(b))
    lo, hi = a.where(a <= b, b), b.where(a <= b, a)
    return lo + "||" + hi


def true_pairs_from_truth(truth: pd.DataFrame) -> set[str]:
    """All unordered record pairs sharing an entity_id (``source:record_id`` keys)."""
    keys = truth["source"].astype(str) + ":" + truth["source_record_id"].astype(str)
    pairs: set[str] = set()
    for _, grp in keys.groupby(truth["entity_id"].values):
        vals = sorted(grp.tolist())
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                pairs.add(f"{vals[i]}||{vals[j]}")
    return pairs


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "tp": tp, "fp": fp, "fn": fn}


def pairwise_at_threshold(scored: pd.DataFrame, score_col: str, threshold: float, truth_pairs: set[str]) -> dict:
    keys = pair_key(scored["unique_id_l"], scored["unique_id_r"])
    pos = keys[scored[score_col] >= threshold]
    tp = int(pos.isin(truth_pairs).sum())
    return {"threshold": threshold, **prf(tp, int(len(pos) - tp), int(len(truth_pairs) - tp))}


def pr_curve(scored: pd.DataFrame, score_col: str, truth_pairs: set[str], thresholds: Iterable[float]) -> pd.DataFrame:
    return pd.DataFrame([pairwise_at_threshold(scored, score_col, t, truth_pairs) for t in thresholds])


def cluster_pairwise(crosswalk: pd.DataFrame, truth_pairs: set[str]) -> dict:
    """Pairwise P/R/F1 implied by the crosswalk's entity assignment (transitive closure)."""
    keys = crosswalk["source"].astype(str) + ":" + crosswalk["source_record_id"].astype(str)
    predicted: set[str] = set()
    for _, grp in keys.groupby(crosswalk["entity_id"].values):
        vals = sorted(grp.tolist())
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                predicted.add(f"{vals[i]}||{vals[j]}")
    tp = len(predicted & truth_pairs)
    return prf(tp, len(predicted) - tp, len(truth_pairs) - tp)


def blocking_recall(candidates: pd.DataFrame, truth_pairs: set[str]) -> dict:
    """Share of true pairs that survived blocking (present in the candidate set)."""
    keys = set(pair_key(candidates["unique_id_l"], candidates["unique_id_r"]))
    found = len(keys & truth_pairs)
    return {"true_pairs": len(truth_pairs), "candidates": len(keys), "found": found,
            "blocking_recall": round(found / len(truth_pairs), 4) if truth_pairs else 0.0}


def misses_by_noise(missed_keys: Iterable[str], truth: pd.DataFrame) -> pd.DataFrame:
    """Count which noise operators appear on the records of missed true pairs."""
    ops = dict(zip(truth["source"].astype(str) + ":" + truth["source_record_id"].astype(str), truth.get("noise_ops", "")))
    counts: dict[str, int] = {}
    n = 0
    for key in missed_keys:
        n += 1
        seen = set()
        for rec in key.split("||"):
            raw = ops.get(rec, "")
            for op in str(raw if isinstance(raw, str) else "").split("|"):
                if op:
                    seen.add(op)
        for op in seen:
            counts[op] = counts.get(op, 0) + 1
    return pd.DataFrame(sorted(counts.items(), key=lambda kv: -kv[1]), columns=["noise_op", "missed_pairs"]).assign(n_missed=n)


@dataclass
class EvaluationReport:
    pipeline_pairwise: dict
    pipeline_cluster: dict
    pipeline_pr_curve: pd.DataFrame
    baseline_pairwise: dict
    baseline_best: dict
    baseline_pr_curve: pd.DataFrame
    blocking: dict
    deterministic: dict
    misses: pd.DataFrame
    extras: dict = field(default_factory=dict)

    def summary_table(self) -> pd.DataFrame:
        rows = [
            {"method": "baseline (token_sort_ratio @ chosen)", **self.baseline_pairwise},
            {"method": "baseline (best F1 threshold)", **self.baseline_best},
            {"method": "pipeline pairwise @ auto_link", **self.pipeline_pairwise},
            {"method": "pipeline clusters (transitive)", "threshold": self.extras.get("cluster_threshold"), **self.pipeline_cluster},
        ]
        return pd.DataFrame(rows)

    def to_markdown(self) -> str:
        lines = ["# Evaluation: baseline vs pipeline", "", self.summary_table().to_markdown(index=False), ""]
        lines += [f"Blocking recall: {self.blocking['blocking_recall']} "
                  f"({self.blocking['found']} of {self.blocking['true_pairs']} true pairs in {self.blocking['candidates']} candidates)",
                  f"Deterministic pre-pass: {self.deterministic}", ""]
        if len(self.misses):
            lines += ["Noise operators on missed true pairs (pipeline clusters):", "",
                      self.misses.to_markdown(index=False), ""]
        return "\n".join(lines)


def evaluate(
    truth: pd.DataFrame,
    predictions: pd.DataFrame,
    crosswalk: pd.DataFrame,
    edges: pd.DataFrame | None = None,
    *,
    deterministic_pairs: pd.DataFrame,
    baseline: pd.DataFrame,
    auto_link: float,
    cluster_threshold: float,
    baseline_threshold: float = 85.0,
) -> EvaluationReport:
    """Compare pipeline outputs and the baseline against ground truth.

    ``predictions`` are the raw Splink candidates (used for blocking recall); ``edges`` are
    the post-constraint pairs the crosswalk is built from (used for pairwise metrics).
    """
    tp_set = true_pairs_from_truth(truth)
    scored = edges if edges is not None else predictions
    pipe_pw = pairwise_at_threshold(scored, "match_probability", auto_link, tp_set)
    pipe_curve = pr_curve(scored, "match_probability", tp_set, [round(x, 3) for x in np.linspace(0.05, 0.999, 40)])
    base_pw = pairwise_at_threshold(baseline, "score", baseline_threshold, tp_set)
    base_curve = pr_curve(baseline, "score", tp_set, list(range(50, 101, 2)))
    base_best = base_curve.sort_values("f1", ascending=False).iloc[0].to_dict()
    cluster = cluster_pairwise(crosswalk, tp_set)
    keys_cw = crosswalk["source"].astype(str) + ":" + crosswalk["source_record_id"].astype(str)
    ent = dict(zip(keys_cw, crosswalk["entity_id"]))
    missed = [k for k in tp_set if ent.get(k.split("||")[0]) != ent.get(k.split("||")[1])]
    det_keys = pair_key(deterministic_pairs["unique_id_l"], deterministic_pairs["unique_id_r"]) if len(deterministic_pairs) \
        else pd.Series(dtype=str)
    det_tp = int(det_keys.isin(tp_set).sum())
    return EvaluationReport(
        pipeline_pairwise=pipe_pw, pipeline_cluster=cluster, pipeline_pr_curve=pipe_curve,
        baseline_pairwise=base_pw, baseline_best=base_best, baseline_pr_curve=base_curve,
        blocking=blocking_recall(predictions, tp_set),
        deterministic={"pairs": int(len(det_keys)), "precision": round(det_tp / len(det_keys), 4) if len(det_keys) else None,
                       "recall": round(det_tp / len(tp_set), 4) if tp_set else None},
        misses=misses_by_noise(missed, truth),
        extras={"cluster_threshold": cluster_threshold, "baseline_threshold": baseline_threshold},
    )
