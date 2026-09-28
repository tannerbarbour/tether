# tether

Explainable, AI-assisted entity resolution built on [Splink](https://moj-analytical-services.github.io/splink/)
(Fellegi-Sunter probabilistic record linkage), for linking provider, organization, employee and
vendor records across sources that share no unique identifier.

**Status: pilot, Phase 1 (skeleton, config, field types, profiles, synthetic data).**

```bash
pip install -e ".[dev]"
tether generate-synthetic --out engagements/example/data
pytest
```

See `engagements/example/config.yaml` for the engagement configuration format and
`knowledge_base/README.md` for the knowledge-base layout. A full README lands in Phase 4.
