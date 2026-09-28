"""Evaluation against ground truth.

Metric definitions
------------------
* **Pairwise P/R/F1 at a threshold**: over unordered record pairs. Positives are pairs
  scored at/above the threshold; truth is "same entity_id".
* **Cluster pairwise P/R/F1**: the same pairwise metric on the *transitive closure* of the
  crosswalk (every pair of records that share a predicted entity_id), i.e. what a consumer
  of the crosswalk actually gets, including links implied by transitivity and losses from
  constraint splits.
* **Entity exact-match rate**: share of true entities whose record set is exactly one
  predicted cluster (neither split nor merged with anything). Complements pairwise F1,
  which over-weights large clusters.
* **Non-deterministic subset**: pairwise metrics after removing every pair the
  deterministic pre-pass links from both truth and predictions, so the probabilistic model
  (and the baselines) are judged only where identifiers do not settle the question.
* **Blocking recall**: share of true pairs that survive blocking into the candidate set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


def pair_key(a: pd.Series | Iterable[str], b: pd.Series | Iterable[str]) -> pd.Series:
    a, b = pd.Series(list(a)), pd.Series(list(b))
    lo, hi = a.where(a <= b, b), b.where(a <= b, a)
    return lo + "||" + hi


def _pairs_from_groups(keys: pd.Series, groups: pd.Series) -> set[str]:
    pairs: set[str] = set()
    for _, grp in keys.groupby(groups.values):
        vals = sorted(grp.tolist())
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                pairs.add(f"{vals[i]}||{vals[j]}")
    return pairs


def true_pairs_from_truth(truth: pd.DataFrame) -> set[str]:
    """All unordered record pairs sharing an entity_id (``source:record_id`` keys)."""
    keys = truth["source"].astype(str) + ":" + truth["source_record_id"].astype(str)
    return _pairs_from_groups(keys, truth["entity_id"])


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "tp": tp, "fp": fp, "fn": fn}


def pairwise_at_threshold(scored: pd.DataFrame, score_col: str, threshold: float, truth_pairs: set[str],
                          exclude: set[str] | None = None) -> dict:
    keys = pair_key(scored["unique_id_l"], scored["unique_id_r"])
    pos = keys[scored[score_col] >= threshold]
    if exclude:
        pos = pos[~pos.isin(exclude)]
        truth_pairs = truth_pairs - exclude
    tp = int(pos.isin(truth_pairs).sum())
    return {"threshold": threshold, **prf(tp, int(len(pos) - tp), int(len(truth_pairs) - tp))}


def pr_curve(scored: pd.DataFrame, score_col: str, truth_pairs: set[str], thresholds: Iterable[float],
             exclude: set[str] | None = None) -> pd.DataFrame:
    return pd.DataFrame([pairwise_at_threshold(scored, score_col, t, truth_pairs, exclude) for t in thresholds])


def cluster_metrics_vs_truth(crosswalk: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """Cluster pairwise P/R/F1 plus entity exact-match rate and split/merge counts."""
    truth_pairs = true_pairs_from_truth(truth)
    keys = crosswalk["source"].astype(str) + ":" + crosswalk["source_record_id"].astype(str)
    predicted = _pairs_from_groups(keys, crosswalk["entity_id"])
    tp = len(predicted & truth_pairs)
    out = {**prf(tp, len(predicted) - tp, len(truth_pairs) - tp)}
    tkeys = truth["source"].astype(str) + ":" + truth["source_record_id"].astype(str)
    true_sets = {e: frozenset(g) for e, g in tkeys.groupby(truth["entity_id"].values)}
    pred_sets = {c: frozenset(g) for c, g in keys.groupby(crosswalk["entity_id"].values)}
    pred_lookup = set(pred_sets.values())
    exact = sum(1 for s in true_sets.values() if s in pred_lookup)
    rec_to_cluster = dict(zip(keys, crosswalk["entity_id"]))
    split = sum(1 for s in true_sets.values() if len({rec_to_cluster.get(r) for r in s}) > 1)
    cluster_to_entities: dict[str, set] = {}
    truth_entity = dict(zip(tkeys, truth["entity_id"]))
    for rec, cid in rec_to_cluster.items():
        cluster_to_entities.setdefault(cid, set()).add(truth_entity.get(rec))
    merged = sum(1 for ents in cluster_to_entities.values() if len(ents) > 1)
    out.update({"n_true_entities": len(true_sets), "entity_exact_match_rate": round(exact / len(true_sets), 4),
                "entities_split": split, "clusters_merging_entities": merged})
    return out


def blocking_recall(candidates: pd.DataFrame, truth_pairs: set[str]) -> dict:
    keys = set(pair_key(candidates["unique_id_l"], candidates["unique_id_r"]))
    found = len(keys & truth_pairs)
    return {"true_pairs": len(truth_pairs), "candidates": len(keys), "found": found,
            "blocking_recall": round(found / len(truth_pairs), 4) if truth_pairs else 0.0}


def misses_by_noise(missed_keys: Iterable[str], truth: pd.DataFrame) -> pd.DataFrame:
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
class MethodResult:
    name: str
    at_threshold: dict
    best: dict
    curve: pd.DataFrame
    nondeterministic_at_threshold: dict
    nondeterministic_best: dict


@dataclass
class EvaluationReport:
    methods: list[MethodResult]
    cluster: dict
    blocking: dict
    deterministic: dict
    misses: pd.DataFrame
    extras: dict = field(default_factory=dict)
    ablations: list[dict] = field(default_factory=list)

    def method(self, name: str) -> MethodResult:
        return next(m for m in self.methods if m.name == name)

    def summary_table(self) -> pd.DataFrame:
        rows = []
        for m in self.methods:
            rows.append({"method": f"{m.name} @ chosen", **m.at_threshold})
            rows.append({"method": f"{m.name} @ best F1", **m.best})
        rows.append({"method": "pipeline crosswalk (cluster pairwise)", "threshold": self.extras.get("cluster_threshold"),
                     **{k: self.cluster[k] for k in ("precision", "recall", "f1", "tp", "fp", "fn")}})
        for a in self.ablations:
            rows.append({"method": a["name"], "threshold": a.get("threshold"), **{k: a[k] for k in ("precision", "recall", "f1", "tp", "fp", "fn")}})
        return pd.DataFrame(rows)

    def nondeterministic_table(self) -> pd.DataFrame:
        rows = []
        for m in self.methods:
            rows.append({"method": f"{m.name} @ chosen", **m.nondeterministic_at_threshold})
            rows.append({"method": f"{m.name} @ best F1", **m.nondeterministic_best})
        return pd.DataFrame(rows)

    def to_markdown(self) -> str:
        lines = ["# Evaluation: baselines vs pipeline (synthetic data)", "", self.summary_table().to_markdown(index=False), "",
                 "## Non-deterministic pairs only (deterministic NPI links removed from truth and predictions)", "",
                 self.nondeterministic_table().to_markdown(index=False), "",
                 f"Cluster metrics: {self.cluster}", f"Blocking: {self.blocking}", f"Deterministic pre-pass: {self.deterministic}", ""]
        if len(self.misses):
            lines += ["Noise operators on missed true pairs (crosswalk):", "", self.misses.to_markdown(index=False), ""]
        return "\n".join(lines)


def _restrict(df: pd.DataFrame, sources: set[str]) -> pd.DataFrame:
    return df[df["unique_id_l"].str.split(":").str[0].isin(sources) & df["unique_id_r"].str.split(":").str[0].isin(sources)]


def evaluate(
    truth: pd.DataFrame,
    predictions: pd.DataFrame,
    crosswalk: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    deterministic_pairs: pd.DataFrame,
    baselines: dict[str, tuple[pd.DataFrame, float]],
    auto_link: float,
    cluster_threshold: float,
) -> EvaluationReport:
    """Compare pipeline outputs and baselines against ground truth.

    ``baselines`` maps a method name to ``(scores frame with 'score', chosen threshold)``.
    ``predictions`` are raw Splink candidates (blocking recall); ``edges`` are the
    post-constraint pairs the crosswalk is built from (pairwise metrics).
    """
    tp_set = true_pairs_from_truth(truth)
    sources = set(truth["source"].astype(str))
    crosswalk = crosswalk[crosswalk["source"].astype(str).isin(sources)]
    predictions, edges = _restrict(predictions, sources), _restrict(edges, sources)
    det_keys = set(pair_key(deterministic_pairs["unique_id_l"], deterministic_pairs["unique_id_r"])) if len(deterministic_pairs) else set()

    def method(name: str, scored: pd.DataFrame, col: str, chosen: float, grid: list[float]) -> MethodResult:
        curve = pr_curve(scored, col, tp_set, grid)
        best = curve.sort_values("f1", ascending=False).iloc[0].to_dict()
        nd_curve = pr_curve(scored, col, tp_set, grid, exclude=det_keys)
        return MethodResult(name, pairwise_at_threshold(scored, col, chosen, tp_set), best, curve,
                            pairwise_at_threshold(scored, col, chosen, tp_set, exclude=det_keys),
                            nd_curve.sort_values("f1", ascending=False).iloc[0].to_dict())

    methods = [method(n, _restrict(df, sources), "score", thr, list(range(50, 101, 2))) for n, (df, thr) in baselines.items()]
    methods.append(method("pipeline pairwise", edges, "match_probability", auto_link,
                          [round(x, 3) for x in np.linspace(0.05, 0.999, 40)]))
    cluster = cluster_metrics_vs_truth(crosswalk, truth)
    keys_cw = crosswalk["source"].astype(str) + ":" + crosswalk["source_record_id"].astype(str)
    ent = dict(zip(keys_cw, crosswalk["entity_id"]))
    missed = [k for k in tp_set if ent.get(k.split("||")[0]) != ent.get(k.split("||")[1])]
    det_tp = len(det_keys & tp_set)
    return EvaluationReport(
        methods=methods, cluster=cluster, blocking=blocking_recall(predictions, tp_set),
        deterministic={"pairs": len(det_keys), "precision": round(det_tp / len(det_keys), 4) if det_keys else None,
                       "recall": round(det_tp / len(tp_set), 4) if tp_set else None},
        misses=misses_by_noise(missed, truth),
        extras={"cluster_threshold": cluster_threshold, "auto_link": auto_link},
    )
