"""Engagement configuration models (pydantic) and loaders.

The engagement config is the only per-project input. Everything it references is
either engine code (profile, field types) or knowledge-base assets; it never carries
secrets. Azure settings are read from environment variables (see ``ingestion``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ReuseScope = Literal["generic", "client_scoped"]
PromotionState = Literal["engagement_local", "reviewed", "promoted"]
LinkType = Literal["link_only", "dedupe_only", "link_and_dedupe"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EngagementMeta(_Strict):
    id: str = Field(pattern=r"^[a-z0-9_\-]+$", description="Short slug used in output and KB provenance")
    client: str = ""
    description: str = ""
    analyst: str = ""
    reuse_scope: ReuseScope = "client_scoped"


# --------------------------------------------------------------------- mappings
class ColumnMappingEntry(_Strict):
    """One source column -> one canonical target (``name``, ``name_first``, ``npi`` ...)."""

    source_column: str
    target: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    rationale: str | None = None


class SchemaMapping(_Strict):
    """An approved (or proposed) mapping from a source's raw columns to the canonical schema."""

    source: str
    profile: str
    status: Literal["proposed", "approved", "rejected"] = "proposed"
    version: int = 1
    record_id_column: str | None = None
    columns: list[ColumnMappingEntry] = Field(default_factory=list)
    unmapped: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)

    @field_validator("columns")
    @classmethod
    def _no_duplicate_targets(cls, v: list[ColumnMappingEntry]) -> list[ColumnMappingEntry]:
        seen: set[str] = set()
        for entry in v:
            if entry.target in seen:
                raise ValueError(f"target '{entry.target}' mapped more than once")
            seen.add(entry.target)
        return v

    def as_dict(self) -> dict[str, str]:
        return {e.source_column: e.target for e in self.columns}

    @classmethod
    def load(cls, path: str | Path) -> "SchemaMapping":
        with open(path, encoding="utf-8") as fh:
            return cls.model_validate(yaml.safe_load(fh))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.model_dump(mode="json"), fh, sort_keys=False)


# ---------------------------------------------------------------------- sources
class SourceConfig(_Strict):
    name: str = Field(pattern=r"^[A-Za-z0-9_]+$")
    path: Path
    format: Literal["csv", "parquet"] = "csv"
    record_id_column: str
    source_type: str = Field(default="", description="e.g. roster, claims, form990 - keys KB model params")
    mapping: Path | None = Field(default=None, description="Approved SchemaMapping YAML; required for `run`")
    encoding: str = "utf-8"


# ------------------------------------------------------------------- thresholds
class Thresholds(_Strict):
    auto_link: float = Field(0.90, gt=0, lt=1, description="Pairs at/above are accepted without review")
    review_lower: float = Field(0.50, gt=0, lt=1, description="Pairs in [review_lower, auto_link) go to review")
    cluster: float = Field(0.95, gt=0, le=1, description="Edge threshold for clustering (stricter)")

    @model_validator(mode="after")
    def _ordered(self) -> "Thresholds":
        if not (self.review_lower < self.auto_link <= self.cluster):
            raise ValueError("require review_lower < auto_link <= cluster")
        return self


class ClusterQA(_Strict):
    max_cluster_size: int | None = Field(None, ge=2, description="Override profile default")
    min_edge_density: float | None = Field(None, ge=0, le=1, description="Override profile default")


class FieldOverride(_Strict):
    role: Literal["identifier", "primary", "supporting", "report_only"] | None = None
    term_frequency: bool | None = None
    include_phonetic: bool | None = None
    exclude: bool = False


class MatchingConfig(_Strict):
    link_type: LinkType = "link_and_dedupe"
    blocking_rules: list[list[str] | str] | None = Field(
        None, description="Override profile blocking rules: column lists (block_on) or SQL strings"
    )
    training_blocking_rules: list[list[str] | str] | None = None
    max_pairs_for_u: float = Field(1e6, gt=0)
    em_max_iterations: int = Field(25, ge=1)
    probability_two_random_records_match: float | None = Field(
        None, gt=0, lt=1, description="If unset, estimated from the deterministic rules and their assumed recall"
    )
    deterministic_rule_recall: float = Field(0.6, gt=0, le=1, description="Assumed recall of deterministic rules")
    probability_floor: float = Field(
        1e-3, ge=0, lt=0.5,
        description="Floor for trained m/u so a level never observed in training cannot get an infinite weight",
    )
    load_m_from_knowledge_base: bool = True
    save_m_to_knowledge_base: bool = True
    fields: dict[str, FieldOverride] = Field(default_factory=dict)
    random_seed: int = 42


# -------------------------------------------------------------------------- llm
class LLMConfig(_Strict):
    provider: Literal["mock", "azure"] = "mock"
    sample_rows: int = Field(20, ge=0, le=200)
    temperature: float = 0.0
    redact_columns: list[str] = Field(
        default_factory=list, description="Source columns whose sample values are masked before leaving the tenant"
    )
    standardize_titles: bool = True
    standardize_org_aliases: bool = True
    max_distinct_values_per_call: int = Field(50, ge=1, le=500)
    log_dir: Path | None = None


class KnowledgeBaseConfig(_Strict):
    path: Path = Path("knowledge_base")
    use_promoted_only: bool = True


class ReferenceConfig(_Strict):
    nppes_path: Path | None = None
    zip_centroids_path: Path | None = None
    enrich_from_nppes: bool = True
    hub_link_via_nppes: bool = False


class OutputConfig(_Strict):
    dir: Path = Path("output")
    explanation_sample_pairs: int = Field(25, ge=0)
    rejected_audit_sample: int = Field(25, ge=0, description="Random rejected pairs routed to review as an audit sample")


class EngagementConfig(_Strict):
    engagement: EngagementMeta
    profile: str = "provider"
    sources: list[SourceConfig] = Field(min_length=1)
    thresholds: Thresholds = Field(default_factory=Thresholds)
    cluster_qa: ClusterQA = Field(default_factory=ClusterQA)
    matching: MatchingConfig = Field(default_factory=MatchingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    knowledge_base: KnowledgeBaseConfig = Field(default_factory=KnowledgeBaseConfig)
    reference: ReferenceConfig = Field(default_factory=ReferenceConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    config_dir: Path | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _consistency(self) -> "EngagementConfig":
        names = [s.name for s in self.sources]
        if len(set(names)) != len(names):
            raise ValueError("source names must be unique")
        if self.matching.link_type == "dedupe_only" and len(self.sources) != 1:
            raise ValueError("dedupe_only requires exactly one source")
        if self.matching.link_type == "link_only" and len(self.sources) < 2:
            raise ValueError("link_only requires at least two sources")
        return self

    def resolve(self, path: Path | None) -> Path | None:
        """Resolve a path relative to the config file's directory."""
        if path is None or path.is_absolute() or self.config_dir is None:
            return path
        return (self.config_dir / path).resolve()

    @classmethod
    def load(cls, path: str | Path) -> "EngagementConfig":
        """Load and validate a YAML engagement config; relative paths resolve against its directory."""
        path = Path(path)
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        cfg = cls.model_validate(raw)
        cfg.config_dir = path.parent.resolve()
        return cfg
