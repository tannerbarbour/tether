"""Match explanation report: model parameters, match-weight chart, waterfall charts, cluster QA."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from tether.matching.pipeline import LinkageResult
from tether.reporting.html import esc, page, table, tiles

VEGA_HEAD = ("<script src='https://cdn.jsdelivr.net/npm/vega@5'></script>"
             "<script src='https://cdn.jsdelivr.net/npm/vega-lite@5'></script>"
             "<script src='https://cdn.jsdelivr.net/npm/vega-embed@6'></script>")


def parameter_table(settings_dict: dict) -> pd.DataFrame:
    rows = []
    for comp in settings_dict.get("comparisons", []):
        for lv in comp["comparison_levels"]:
            m, u = lv.get("m_probability"), lv.get("u_probability")
            w = math.log2(m / u) if m and u and m > 0 and u > 0 else None
            rows.append({"comparison": comp["output_column_name"], "level": lv.get("label_for_charts") or lv["sql_condition"],
                         "m": m, "u": u, "match_weight_bits": w, "tf_adjusted": bool(lv.get("tf_adjustment_column"))})
    return pd.DataFrame(rows)


def _sample_pairs(result: LinkageResult, n: int, seed: int) -> list[tuple[str, pd.DataFrame]]:
    pred, th = result.predictions, result.config.thresholds
    k = max(1, n // 3)
    groups = [
        ("accepted (>= auto_link)", pred[pred["match_probability"] >= th.auto_link]),
        ("review band", pred[(pred["match_probability"] >= th.review_lower) & (pred["match_probability"] < th.auto_link)]),
        ("rejected by pair constraint", pred.loc[pred.index.isin(result.rejected_pairs.index)]),
    ]
    return [(label, g.sample(min(k, len(g)), random_state=seed)) for label, g in groups if len(g)]


def _json(obj) -> str:
    return json.dumps(obj, default=lambda o: None if isinstance(o, float) and o != o else str(o))


def render_explanation_report(result: LinkageResult, out_path: str | Path, seed: int = 0) -> Path:
    """Write the HTML explanation report; charts need ``result.matcher`` (run with keep_matcher)."""
    s, cfg = result.stats, result.config
    body = []
    body.append(tiles([
        ("records", f"{s['n_records']:,}", f"{s['n_sources']} sources"),
        ("entities", f"{s['n_clusters']:,}", f"{s['n_clusters_split']} split by constraints"),
        ("deterministic pairs", f"{s['n_deterministic_pairs']:,}", "validated identifier + guard"),
        ("edges", f"{s['n_edges']:,}", f"{s['n_pairs_rejected_by_pair_constraints']} rejected, {s['n_edges_dropped_by_hard_constraints']} conflicting"),
        ("review queue", f"{s['n_review_queue']:,}", f"[{cfg.thresholds.review_lower}, {cfg.thresholds.auto_link})"),
        ("flagged records", f"{s['n_flagged_records']:,}", "oversized / low density"),
    ]))
    body.append("<h2>Run provenance</h2>")
    prov = pd.DataFrame([
        {"item": "run id", "value": s["run_id"]}, {"item": "profile", "value": s["profile"]},
        {"item": "link type", "value": s["link_type"]}, {"item": "comparison schema hash", "value": s["comparison_schema_hash"]},
        {"item": "knowledge base version (before run)", "value": s["knowledge_base_version"]},
        {"item": "reference snapshots", "value": ", ".join(f"{k}={v}" for k, v in s["reference_snapshots"].items()) or "none"},
        {"item": "NPPES enrichment / hub", "value": f"{s['nppes_enrichment']} / {s['hub_source'] or 'off'}"},
        {"item": "m starting values from knowledge base", "value": f"{s['m_seeded_levels']} levels"},
        {"item": "EM iterations per round", "value": ", ".join(str(i) for i in s["em_iterations"])},
        {"item": "thresholds", "value": ", ".join(f"{k}={v}" for k, v in s["thresholds"].items())},
    ])
    body.append(table(prov))
    body.append("<h2>Input validation</h2><p class='note'>Invalid values are flagged and kept for review; only validated identifiers feed the deterministic pass and constraints.</p>")
    body.append(table(result.prepared.validation_summary))
    body.append("<h2>Model parameters</h2><p class='note'>Match weight = log2(m/u). Levels are graded within one comparison (raw → cleaned → fuzzy → phonetic) so evidence is counted once.</p>")
    body.append(table(parameter_table(result.model.settings_dict)))

    charts: list[tuple[str, dict]] = []
    if result.matcher is not None:
        charts.append(("match-weights", result.matcher.match_weights_chart().chart_dict))
        body.append("<h2>Match weights</h2><div class='chart' id='chart-match-weights'></div>")
        body.append("<h2>Why pairs scored the way they did</h2><p class='note'>Waterfall charts for sampled pairs: each bar is one comparison's contribution in bits.</p>")
        for gi, (label, grp) in enumerate(_sample_pairs(result, cfg.output.explanation_sample_pairs, seed)):
            body.append(f"<h3>{esc(label)} ({len(grp)} sampled)</h3>")
            for i, (_, row) in enumerate(grp.iterrows()):
                rec = {k: (None if isinstance(v, float) and v != v else v) for k, v in row.to_dict().items()}
                spec = result.matcher.waterfall_chart([rec]).chart_dict
                cid = f"wf-{gi}-{i}"
                charts.append((cid, spec))
                body.append(f"<p class='note'>{esc(rec['unique_id_l'])} ↔ {esc(rec['unique_id_r'])} · p = {rec['match_probability']:.4f}</p>"
                            f"<div class='chart' id='chart-{cid}'></div>")
    else:
        body.append("<p class='note'>Charts omitted (matcher not retained).</p>")

    body.append("<h2>Cluster QA</h2>")
    sizes = result.metrics.drop_duplicates("cluster_id")["cluster_size"].value_counts().sort_index()
    body.append(table(pd.DataFrame({"cluster_size": sizes.index, "clusters": sizes.values})))
    flagged = result.metrics[result.metrics["flags"].str.contains("oversized|low_density")]
    body.append("<h3>Flagged clusters</h3>" + table(flagged.drop_duplicates("cluster_id"), max_rows=50))
    if result.split_log:
        body.append("<h3>Clusters split by hard constraints</h3>" + table(pd.DataFrame(result.split_log)))
    body.append("<h2>Review queue (top 25)</h2>" + table(result.review_queue, max_rows=25))

    script = "<script>" + "".join(
        f"vegaEmbed('#chart-{cid}', {_json(spec)}, {{actions:false}});" for cid, spec in charts) + "</script>"
    html_out = page("Match explanation report", f"engagement {cfg.engagement.id} · {cfg.engagement.client}",
                    "".join(body) + script, VEGA_HEAD if charts else "")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_out, encoding="utf-8")
    return out
