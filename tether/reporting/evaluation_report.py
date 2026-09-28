"""Evaluation report (HTML) for stakeholders: pipeline vs baselines, honestly labeled."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tether.reporting.evaluation import EvaluationReport
from tether.reporting.html import PALETTE, esc, line_chart_svg, page, table, tiles

SPLIT_ALGORITHM = """
<ol>
<li>Before clustering, every edge whose two records carry <em>different</em> non-null values of a
constrained column (validated NPI) is removed.</li>
<li>Connected components are computed over the remaining edges at the cluster threshold.</li>
<li>A component can still contain two NPIs through a bridge record with no NPI
(A<sub>npi 1</sub> — B<sub>no npi</sub> — C<sub>npi 2</sub>). For each such component, one
sub-cluster is seeded per distinct NPI value; every remaining member is attached to the seed it has
the strongest edge into (highest match probability); a member with no edge to any seed becomes a
singleton.</li>
<li>Each split is logged (original cluster, record count, sub-clusters) and surfaced in the
explanation report.</li>
</ol>
"""

LABELING_PLAN = """
<ol>
<li><strong>Sample</strong> on the first real engagement, stratified by score band: 150 pairs from the review band,
100 accepted pairs, 100 pairs rejected by the name-agreement constraint, 100 random blocked-out pairs
(for blocking recall), plus every cluster split by a hard constraint.</li>
<li><strong>Annotate</strong> each pair independently by two analysts with a third adjudicating disagreements;
record Cohen's kappa. Labels are stored as comparison vectors plus label in the knowledge base
(<code>labeled_pairs</code>), never as raw records.</li>
<li><strong>Estimate</strong> m directly from labels (Splink <code>estimate_m_from_pairwise_labels</code>) and compare
with the EM estimates; large gaps flag comparisons whose synthetic behaviour does not transfer.</li>
<li><strong>Measure</strong> precision and recall on the labeled sample with bootstrap confidence intervals, set the
auto-link and cluster thresholds from that sample, and re-run this report with the <em>real-data</em> label.</li>
<li><strong>Replace</strong> the mock LLM with the Azure deployment for schema mapping and value standardization, and
review the proposed mappings and lookup rows before promotion.</li>
</ol>
"""


def _pr_points(curve: pd.DataFrame) -> list[tuple[float, float, str]]:
    return [(float(r.recall), float(r.precision), f"threshold {r.threshold}: P={r.precision:.3f} R={r.recall:.3f} F1={r.f1:.3f}")
            for r in curve.itertuples()]


def render_evaluation_report(report: EvaluationReport, run_stats: dict, out_path: str | Path,
                             engagement: str = "", llm_provider: str = "mock") -> Path:
    data_label = "SYNTHETIC DATA"
    llm_label = "MOCK LLM (offline heuristics)" if llm_provider == "mock" else f"LLM: {llm_provider}"
    base = report.method("baseline: name+org fuzzy")
    base2 = report.method("baseline: NPI exact + name fuzzy")
    pipe = report.method("pipeline pairwise")
    c = report.cluster
    tuned = report.extras.get("tuning")
    body = [f"<p class='note'><strong>Data: {data_label}.</strong> Every number on this page comes from generated provider records "
            f"with a known crosswalk; the noise model is configurable and documented in the repository. "
            f"<strong>{llm_label}.</strong> No real client data and no production LLM calls were involved. "
            + (f"Thresholds were tuned on a separate synthetic seed ({tuned['tuning_seed']}) and evaluated on this one." if tuned
               else "Thresholds are the engagement config defaults (not tuned on this data).") + "</p>"]
    body.append(tiles([
        ("current approach F1", f"{base.best['f1']:.3f}", f"name+org fuzzy, best threshold {base.best['threshold']:.0f}"),
        ("stronger baseline F1", f"{base2.best['f1']:.3f}", f"NPI exact + name fuzzy, best threshold {base2.best['threshold']:.0f}"),
        ("pipeline crosswalk F1", f"{c['f1']:.3f}", f"P {c['precision']:.3f} · R {c['recall']:.3f} @ cluster {report.extras['cluster_threshold']}"),
        ("entities exactly right", f"{c['entity_exact_match_rate']:.1%}", f"{c['entities_split']} split · {c['clusters_merging_entities']} merged"),
        ("review queue", f"{run_stats.get('n_review_queue', 0):,}", "pairs for analyst review"),
    ]))
    body.append("<h2>Summary</h2>")
    body.append(f"<p>The current approach scores every pair with one fuzzy ratio on name plus organization; at its best possible "
                f"threshold it reaches F1 {base.best['f1']:.3f}. A stronger one-off script that trusts equal NPIs and otherwise "
                f"compares names reaches {base2.best['f1']:.3f}. The pipeline's crosswalk reaches F1 {c['f1']:.3f} "
                f"(precision {c['precision']:.3f}, recall {c['recall']:.3f}); {c['entity_exact_match_rate']:.1%} of true entities "
                f"come out as exactly one cluster. Validated identifiers link {report.deterministic['pairs']:,} pairs deterministically "
                f"at precision {report.deterministic['precision']}; blocking keeps {report.blocking['blocking_recall']:.1%} of true pairs "
                f"while scoring {report.blocking['candidates']:,} candidates instead of every pair.</p>")
    body.append("<h2>Where identifiers do not decide: non-deterministic pairs only</h2>"
                "<p class='note'>Every pair the deterministic NPI rule links is removed from both truth and predictions. "
                "This isolates the probabilistic model's contribution and is the fair comparison with the baselines, which also benefit from NPI.</p>")
    body.append(table(report.nondeterministic_table()))
    body.append("<h2>Precision and recall at every threshold</h2>")
    body.append("<div class='chart'>" + line_chart_svg([
        {"name": "name+org fuzzy", "color": PALETTE["orange"], "points": _pr_points(base.curve)},
        {"name": "NPI + name fuzzy", "color": PALETTE["aqua"], "points": _pr_points(base2.curve)},
        {"name": "pipeline", "color": PALETTE["blue"], "points": _pr_points(pipe.curve)},
    ], "recall", "precision") + "</div>")
    body.append("<h2>All operating points</h2>" + table(report.summary_table()))
    if report.ablations:
        body.append("<h2>Ablation: NPPES Type 2 invalidation</h2>"
                    "<p class='note'>Same data, same thresholds; the only difference is whether NPIs that NPPES identifies as "
                    "organizations (Type 2) are invalidated for person linking before the deterministic pass and constraints run.</p>")
        body.append(table(pd.DataFrame(report.ablations)))
    body.append("<h2>Metric definitions</h2><ul>"
                "<li><strong>Pairwise P/R/F1</strong>: over unordered record pairs at a score threshold.</li>"
                "<li><strong>Crosswalk (cluster pairwise) P/R/F1</strong>: pairwise metrics on the transitive closure of the "
                "crosswalk's entity ids, i.e. what a consumer of the crosswalk receives, including links implied by transitivity "
                "and losses from constraint splits.</li>"
                "<li><strong>Entity exact-match rate</strong>: share of true entities whose records form exactly one predicted cluster.</li>"
                "<li><strong>Blocking recall</strong>: share of true pairs present in the candidate set.</li></ul>")
    body.append("<h2>Hard-constraint clustering (how conflicting NPIs are kept apart)</h2>" + SPLIT_ALGORITHM)
    le = run_stats.get("lookup_effect") or {}
    body.append("<h2>Knowledge-base lookups</h2>")
    body.append(f"<p class='note'>Records whose cleaned organization changed because of lookup rows: "
                f"{le.get('org_clean_changed_by_lookup', 0)} (lookup rows: {le.get('org_lookup_rows', 0)}); titles changed: "
                f"{le.get('title_canonical_changed_by_lookup', 0)}. With the mock LLM the alias expansions reproduce the engine's "
                f"built-in alias table, so matching results are unchanged by design; on real data the Azure model can add "
                f"client-specific aliases the built-in table lacks.</p>")
    body.append(f"<p class='note'>Pairs rejected by the name-agreement constraint that nevertheless share a valid Type 1 NPI: "
                f"<strong>{run_stats.get('n_rejected_sharing_valid_npi', 0)}</strong> (these are the same-NPI, contradicting-name pairs "
                f"the deterministic guard also refused; they belong in review, not in the crosswalk).</p>")
    body.append("<h2>Where the pipeline still misses</h2><p class='note'>Noise operators present on records of true pairs the crosswalk did not join.</p>")
    body.append(table(report.misses))
    if tuned:
        body.append("<h2>Threshold tuning (separate seed)</h2>"
                    f"<p class='note'>Tuned on synthetic seed {tuned['tuning_seed']} with {tuned['n_entities']} entities; "
                    f"chosen auto_link {tuned['auto_link']}, cluster {tuned['cluster']}.</p>"
                    "<h3>auto_link sweep (pairwise F1)</h3>" + table(pd.DataFrame(tuned["auto_link_sweep"]))
                    + "<h3>cluster threshold sweep (crosswalk F1)</h3>" + table(pd.DataFrame(tuned["cluster_sweep"])))
    body.append("<h2>Next step: real-data labeling plan</h2>" + LABELING_PLAN)
    body.append("<h2>Run facts</h2>")
    facts = pd.DataFrame([{"item": k, "value": (", ".join(f"{a}={b}" for a, b in v.items()) if isinstance(v, dict) else v)}
                          for k, v in run_stats.items() if k in (
                              "run_id", "profile", "link_type", "n_records", "n_sources", "n_deterministic_pairs", "n_edges",
                              "n_clusters", "n_clusters_split", "n_review_queue", "n_rejected_sharing_valid_npi",
                              "nppes_enrichment", "hub_source", "m_seeded_levels", "knowledge_base_version",
                              "reference_snapshots", "thresholds", "runtime_seconds")])
    body.append(table(facts))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page("Evaluation: baselines vs pipeline", f"engagement {esc(engagement)} · {data_label.lower()} · {llm_label.lower()}",
                        "".join(body)), encoding="utf-8")
    return out
