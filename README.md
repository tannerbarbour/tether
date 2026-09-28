# tether

Explainable, AI-assisted entity resolution built on [Splink](https://moj-analytical-services.github.io/splink/)
(Fellegi-Sunter probabilistic record linkage), for linking provider, organization, employee and
vendor records across sources that share no unique identifier.

**Status: pilot, Phase 3 (LLM-assisted ingestion: schema mapping, value standardization, guardrails).**

```bash
pip install -e ".[dev]"
tether generate-synthetic --out engagements/example/data
tether propose-mapping --config engagements/example/config.yaml --out /tmp/proposed   # LLM (mock by default)
tether standardize-values --config engagements/example/config.yaml
tether run --config engagements/example/config.yaml
tether evaluate --config engagements/example/config.yaml --truth engagements/example/data/truth.csv
pytest
```

See `engagements/example/config.yaml` for the engagement configuration format and
`knowledge_base/README.md` for the knowledge-base layout. A full README lands in Phase 4.
