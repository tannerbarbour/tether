"""tether: explainable, AI-assisted entity resolution built on Splink.

Three tiers of knowledge are kept strictly separate:

* ENGINE (this package): domain-agnostic ingestion, cleaning, matching, reporting.
* KNOWLEDGE BASE (``knowledge_base/``): versioned, provenance-tagged assets that
  accumulate across engagements.
* ENGAGEMENT CONFIG (``engagements/<name>/config.yaml``): per-project sources,
  mappings, thresholds and overrides.
"""

__version__ = "0.1.0"
