"""Glue: where lookup tables live for an engagement and how they feed field types."""

from __future__ import annotations

from pathlib import Path

from tether.config import EngagementConfig
from tether.ingestion.standardizer import LookupCache

LOOKUP_FILES = {"job_title": "lookup_titles.csv", "org_alias": "lookup_org_aliases.csv"}


def lookup_path(config: EngagementConfig, kind: str) -> Path:
    return Path(config.resolve(config.knowledge_base.path)) / LOOKUP_FILES[kind]


def open_lookup(config: EngagementConfig, kind: str) -> LookupCache:
    return LookupCache(lookup_path(config, kind), config.engagement.id, config.knowledge_base.use_promoted_only)


def llm_log_dir(config: EngagementConfig) -> Path:
    return Path(config.resolve(config.llm.log_dir) if config.llm.log_dir
                else Path(config.resolve(config.knowledge_base.path)) / "llm_log")


def lookup_dependencies(config: EngagementConfig) -> dict:
    """Field-type dependencies (``taxonomy``, ``aliases``) from the engagement's usable lookups."""
    return {
        "taxonomy": open_lookup(config, "job_title").lookup(),
        "aliases": open_lookup(config, "org_alias").lookup(),
    }
