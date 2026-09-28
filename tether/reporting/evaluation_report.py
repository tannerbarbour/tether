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
<li>A component can still contain two NPIs through bridge records with no NPI
(A<sub>npi 1</sub> — B — C — D<sub>npi 2</sub>). For each such component one sub-cluster is
<strong>seeded</strong> per distinct NPI value.</li>
<li><strong>Propagation in passes.</strong> In each pass, every still-unassigned member looks only at its
edges to members assigned in <em>earlier</em> passes and joins the sub-cluster behind its strongest
edge. A member with no direct edge to a seed is reached through its neighbours in a later pass
(B joins A in pass 1; C, whose only edges go to B and D, compares its B-edge and D-edge in pass 2).
Because a pass depends only on the previous pass, member order cannot change the result.</li>
<li><strong>Ties</strong> (equal match probability toward two sub-clusters) are broken toward the sub-cluster
whose seed value sorts first, so the outcome is deterministic and reproducible; the count of ties is logged.</li>
<li>Members with no edge path to any seed become singletons (counted in the log). Every split logs the
original cluster, record count, sub-clusters, multi-hop assignments, ties and singletons; the explanation
report shows the log.</li>
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
            + (f"Thresholds were tuned on a separate synthetic seed ({tuned['tuning_seed']}) by expected analyst cost and evaluated on this one." if tuned
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
    body.append("<h2>Where identifiers do not decide: residual (non-deterministic) pairs only</h2>"
                "<p class='note'>Every pair the deterministic NPI rule links is removed from both truth and predictions. "
                "Baselines are re-tuned on this residual subset (their best threshold on it); the pipeline is shown at the "
                "auto_link threshold tuned out of sample, with its residual-best row as an in-sample reference only.</p>")
    body.append(table(report.nondeterministic_table()))
    body.append("<h2>Precision and recall at every threshold</h2>")
    body.append("<div class='chart'>" + line_chart_svg([
        {"name": "name+org fuzzy", "color": PALETTE["orange"], "points": _pr_points(base.curve)},
        {"name": "NPI + name fuzzy", "color": PALETTE["aqua"], "points": _pr_points(base2.curve)},
        {"name": "pipeline", "color": PALETTE["blue"], "points": _pr_points(pipe.curve)},
    ], "recall", "precision") + "</div>")
    body.append("<h2>All operating points</h2>" + table(report.summary_table()))
    if tuned:
        al = pd.DataFrame(tuned["auto_link_sweep"])
        chosen = tuned["thresholds"]["auto_link"]
        row = al[al["threshold"] == chosen].iloc[0]
        f1row = al[al["threshold"] == tuned["f1_best_auto_link"]].iloc[0]
        cm = tuned["cost_model"]
        body.append("<h2>Choosing the auto-link threshold on operating cost (tuning seed)</h2>")
        body.append("<div class='chart'>" + line_chart_svg([
            {"name": "precision", "color": PALETTE["blue"], "points": [(float(r.threshold), float(r.precision), f"t={r.threshold}: {r.precision:.3f}") for r in al.itertuples()]},
            {"name": "recall", "color": PALETTE["orange"], "points": [(float(r.threshold), float(r.recall), f"t={r.threshold}: {r.recall:.3f}") for r in al.itertuples()]},
            {"name": "F1", "color": PALETTE["aqua"], "points": [(float(r.threshold), float(r.f1), f"t={r.threshold}: {r.f1:.3f}") for r in al.itertuples()]},
        ], "auto_link threshold", "score", x_domain=(0.5, 1.0), marker_x=chosen, marker_label=f"chosen {chosen}") + "</div>")
        qmax = float(max(al["queue_size"].max(), al["false_auto_links"].max(), 1))
        body.append("<div class='chart'>" + line_chart_svg([
            {"name": "review queue", "color": PALETTE["blue"], "points": [(float(r.threshold), float(r.queue_size), f"t={r.threshold}: {r.queue_size} pairs") for r in al.itertuples()]},
            {"name": "false auto-links", "color": PALETTE["orange"], "points": [(float(r.threshold), float(r.false_auto_links), f"t={r.threshold}: {r.false_auto_links}") for r in al.itertuples()]},
        ], "auto_link threshold", "pairs", x_domain=(0.5, 1.0), y_domain=(0.0, qmax), marker_x=chosen, marker_label=f"chosen {chosen}") + "</div>")
        cmax = float(al["expected_cost_minutes"].max())
        body.append("<div class='chart'>" + line_chart_svg([
            {"name": "expected cost (min)", "color": PALETTE["blue"], "points": [(float(r.threshold), float(r.expected_cost_minutes), f"t={r.threshold}: {r.expected_cost_minutes:.0f} min") for r in al.itertuples()]},
        ], "auto_link threshold", "analyst minutes", x_domain=(0.5, 1.0), y_domain=(0.0, cmax), marker_x=chosen, marker_label=f"chosen {chosen}") + "</div>")
        body.append(f"<p>Unit costs assumed: {cm['review_minutes_per_pair']:g} min to review a queued pair, "
                    f"{cm['false_link_minutes']:g} min to find and undo a false auto-link in a deliverable, "
                    f"{cm['missed_link_minutes']:g} min per missed link (constant across thresholds: pairs below the review floor). "
                    f"At the chosen threshold {chosen}: {int(row['false_auto_links'])} false auto-links, "
                    f"{int(row['queue_size'])} pairs queued (of which {int(row['queue_true_pairs'])} true), "
                    f"expected cost {row['expected_cost_minutes']:.0f} min, F1 {row['f1']:.3f}. "
                    f"The F1-maximising threshold {tuned['f1_best_auto_link']} would give {int(f1row['false_auto_links'])} false auto-links, "
                    f"a queue of {int(f1row['queue_size'])} and expected cost {f1row['expected_cost_minutes']:.0f} min, F1 {f1row['f1']:.3f}. "
                    f"The threshold is chosen to minimise expected analyst cost under these unit costs; change them to move it.</p>")
        d_fp = int(f1row["false_auto_links"]) - int(row["false_auto_links"])
        d_q = int(row["queue_size"]) - int(f1row["queue_size"])
        if d_fp > 0 and d_q > 0:
            body.append(f"<p><strong>Break-even.</strong> Moving from the F1-maximising threshold {tuned['f1_best_auto_link']} to {chosen} "
                        f"avoids {d_fp} false auto-links at the price of {d_q} extra reviewed pairs. The stricter threshold pays for itself "
                        f"whenever undoing one false link costs more than {d_q / d_fp:.1f} pair reviews; below that ratio the F1-maximising "
                        f"threshold is the cheaper operating point.</p>")
        elif chosen == tuned["f1_best_auto_link"]:
            body.append("<p><strong>Break-even.</strong> The cost-minimising and F1-maximising thresholds coincide under these unit costs.</p>")
        body.append(table(al[["threshold", "precision", "recall", "f1", "queue_size", "queue_true_pairs", "false_auto_links", "expected_cost_minutes"]]))
    body.append("<h2>Review queue: what the analyst would see, with ground truth</h2>"
                "<p class='note'>score_band: pairs between the review floor and auto_link. same_npi_name_conflict: pairs sharing a valid "
                "Type 1 NPI but rejected by the name-agreement constraint (routed to review instead of silently dropped). "
                "rejected_audit_sample: a random sample of other rejected pairs so the constraint is audited every run.</p>")
    body.append(table(report.review_queue_table()))
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
                f"<strong>{run_stats.get('n_rejected_sharing_valid_npi', 0)}</strong>. They are routed to the review queue as "
                f"same_npi_name_conflict; their ground-truth status is in the review-queue table above.</p>")
    body.append("<h2>Where the pipeline still misses</h2><p class='note'>Noise operators present on records of true pairs the crosswalk did not join.</p>")
    body.append(table(report.misses))
    if tuned:
        body.append("<h2>Cluster threshold sweep (tuning seed, crosswalk F1)</h2>"
                    f"<p class='note'>Tuned on synthetic seed {tuned['tuning_seed']} with {tuned['n_entities']} entities; "
                    f"chosen cluster threshold {tuned['thresholds']['cluster']} (constrained to be at least auto_link).</p>"
                    + table(pd.DataFrame(tuned["cluster_sweep"])))
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
