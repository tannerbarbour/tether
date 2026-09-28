# tether

Explainable, AI-assisted entity resolution built on [Splink](https://moj-analytical-services.github.io/splink/)
(Fellegi-Sunter probabilistic record linkage). Links provider, organization, employee and vendor
records across sources that share no unique identifier (rosters, claims, Form 990, surveys).

**Status: pilot complete (Phases 1-4).** Everything runs offline on synthetic data with a mock LLM;
the Azure OpenAI client is wired but untested against a live deployment.

## Quick start

```bash
pip install -e ".[dev]"
tether generate-synthetic --out engagements/example/data              # two sources + truth + NPPES sample
tether propose-mapping --config engagements/example/config.yaml --out /tmp/proposed   # LLM-assisted (mock)
tether standardize-values --config engagements/example/config.yaml    # distinct titles / org aliases -> lookups
tether run --config engagements/example/config.yaml                   # crosswalk + explanation report
tether evaluate --config engagements/example/config.yaml \
    --truth engagements/example/data/truth.csv --tune-seed 777 --ablate-nppes   # manager report
tether kb status --config engagements/example/config.yaml
pytest
```

Outputs land in `engagements/example/output/`: `crosswalk.csv`, `review_queue.csv`,
`explanation_report.html`, `evaluation_report.html`, `model.json`, `run_stats.json`.

## Architecture: three tiers, kept apart

| Tier | Where | Content |
|---|---|---|
| Engine | `tether/` | domain-agnostic ingestion, cleaning, Splink orchestration, constraints, clustering, reporting |
| Knowledge base | `knowledge_base/` (local CSV/Parquet; interface in `tether/knowledge`) | schema mappings, lookups, m-parameters, labeled comparison vectors, run history, reference snapshots |
| Engagement config | `engagements/<id>/config.yaml` | sources, approved mappings, profile, thresholds, overrides |

### Core abstractions

* **FieldType** (`tether/fields`): validator + standardizer (derived columns) + Splink comparison
  factory with graded levels *within one comparison* (exact raw → exact cleaned → JW 0.92 → JW 0.85
  → phonetic → else), so evidence is never double counted. Built-ins: `person_name`, `org_name`,
  `address`, `npi`, `ein`, `phone`, `email_domain`, `job_title`. Register new ones with
  `@register_field_type` or the `tether.field_types` entry-point group.
* **EntityProfile** (`tether/profiles`): field specs with roles (identifier / primary / supporting /
  report_only), blocking rules, EM training rules, deterministic rules with guards, hard
  constraints, pair constraints. `provider` is the pilot profile; `organization` is a stub.
* **Splink adapter** (`tether/matching/splink_adapter.py`): the only module that imports Splink.
  DuckDB backend in the pilot; comparisons and blocking are dialect-agnostic and guards are
  rendered through Splink's dialect layer, so Spark (Fabric) is a backend swap.

### Pipeline

```
raw sources ─ approved mapping ─▶ canonical frame ─ field types ─▶ standardized frame
   ─ NPPES enrichment (Type 2 NPIs invalidated for person linking)
   ─ deterministic pre-pass (validated NPI + name guard) ──────────────┐
   ─ Splink: λ from deterministic rules, u by sampling, m by EM        │
     (m starting values from the knowledge base when available)        ▼
   ─ predictions ─ pair constraints ─ hard-constraint edge drop ─▶ edges
   ─ Splink connected components at the cluster threshold
   ─ transitive-violation split ─ cluster QA (size, density, confidence)
   ─▶ crosswalk, review queue, explanation report, knowledge-base writes
```

### Why the constraints exist

Employer-derived signals (org name, address, phone, EIN, email domain) are conditionally
dependent: colleagues agree on all of them. Under Fellegi-Sunter they add up and can outweigh
two name disagreements. The pilot handles this in the profile layer: EIN and email domain are
`report_only` for providers, EM never blocks on employer attributes, and a **pair constraint**
rejects any pair with no name agreement at any level (deterministic pairs are exempt).

**Hard-constraint clustering.** Edges between records with different valid NPIs are removed
before clustering. A component may still hold two NPIs through bridge records with no NPI. Each such
component is split by seeding one sub-cluster per NPI value and then propagating in passes: in every
pass each unassigned member joins the sub-cluster behind its strongest edge to a member assigned in
an earlier pass, so members several hops from a seed are reached through their neighbours and member
order cannot change the result. Ties are broken toward the sub-cluster whose seed value sorts first
(deterministic); members with no edge path to any seed become singletons. Each split logs records,
sub-clusters, multi-hop assignments, ties and singletons.

**Cluster confidence** is the weakest attachment: for each member, its strongest edge into the
cluster; the cluster takes the minimum.

## AI-assisted ingestion (`tether/ingestion`)

* `LLMClient` protocol; `AzureOpenAIClient` (env: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`,
  `AZURE_OPENAI_API_KEY` or `AZURE_OPENAI_AD_TOKEN`, optional `AZURE_OPENAI_API_VERSION`) and
  `MockLLMClient` (offline heuristics). Structured JSON output, temperature 0, seeded, every call
  logged to `knowledge_base/llm_log/llm_calls.jsonl` with the prompt hash.
* Schema mapping: canonical headers, previously approved mappings and value-shape detection resolve
  columns first; the LLM sees only unresolved columns plus a small (optionally redacted) sample.
  Proposals are written as `status: proposed` YAML; deterministic code applies approved mappings.
* Value standardization: distinct values only, batched, cache first; results are
  `engagement_local` lookup rows until promoted.
* Guardrails: targets and categories restricted to allowed sets, no identifier-like strings in
  lookups, extracted identifiers must be verbatim in the source text and validate.

## Knowledge base and promotion

Every asset row carries `asset_id, engagement, reviewer, created_at, version, reuse_scope, state`.
States move `engagement_local → reviewed → promoted` only via `tether kb promote --reviewer ...`;
only `promoted` + `generic` rows are visible to other engagements. m-values and labeled comparison
vectors are keyed by a `comparison_schema_hash` so they are reused only under identical comparison
definitions; u is never transferred. Every run records the knowledge-base version and reference
snapshot ids it used (`run_history`).

## Reference data (`tether/reference`)

`ReferenceSource` gives snapshot ids (content hash) and license metadata. `NPPESReference` loads a
local NPPES CSV for **enrichment** (entity type, legal name, location; Type 2 NPIs invalidated for
person linking) and **hub linking** (`reference.hub_link_via_nppes: true` adds NPPES individuals as
a source; clusters containing an NPPES record are labeled `NPI-<npi>`). `ZipCentroidReference`
reads Census Gazetteer files for the distance levels.

## Evaluation

`tether evaluate` (knowledge-base read-only unless `--write-kb`) compares two baselines (name+org
fuzzy ratio; NPI exact + name fuzzy) with the pipeline: pairwise P/R/F1 and PR curves; the same on
the **residual (non-deterministic) subset** with baselines re-tuned on it and the pipeline at its
out-of-sample thresholds; crosswalk (cluster pairwise) metrics, entity exact-match rate, blocking
recall, misses by noise operator; the NPPES ablation (`--ablate-nppes`); ground truth of the review
queue by reason; and thresholds tuned on a **separate synthetic seed** (`--tune-seed`) by expected
analyst cost (review minutes vs false-link minutes, `tether.matching.tune.CostModel`) with a
break-even statement against the F1-maximising threshold. Reports embed Vega from Splink's bundled
files, so they render offline. The report labels its data as synthetic and its LLM as mock, and
ends with the real-data labeling plan.

The review queue carries a `review_reason`: `score_band`, `same_npi_name_conflict` (same valid NPI
but names contradict; routed to review rather than dropped) and `rejected_audit_sample` (random
rejected pairs so the name-agreement constraint is audited every run).

## Out of scope for the pilot (seams exist)

Supervised classifier over comparison vectors (`labeled_pairs` holds the training data), active
learning UI, Fabric deployment (Spark backend via the adapter), full geocoding, licensed reference
data.
