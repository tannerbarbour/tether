# Knowledge base

Reusable, versioned, provenance-tagged assets that accumulate across engagements.
The pilot stores each table as Parquet/CSV in this directory; the `ecg_linkage.knowledge`
interface is what the engine talks to, so the backing store can later point at Fabric
Lakehouse Delta tables without engine changes.

| Table               | Content                                                     |
|---------------------|-------------------------------------------------------------|
| schema_mappings     | approved source-column -> canonical mappings (few-shot pool)|
| lookup_org_aliases  | org token/phrase expansions                                 |
| lookup_titles       | job title -> canonical taxonomy                             |
| model_params        | Splink m-probabilities by profile / source type             |
| labeled_pairs       | comparison vectors + label (never raw records)              |
| run_history         | every run with KB version + reference snapshot IDs used     |
| reference_snapshots | external reference datasets with license + snapshot version |

Every asset carries: `engagement`, `reviewer`, `created_at`, `version`,
`reuse_scope` (generic | client_scoped) and `state` (engagement_local | reviewed | promoted).
Only `promoted` + `generic` assets are used across engagements.
