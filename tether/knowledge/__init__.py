"""Knowledge base: versioned, provenance-tagged assets that grow across engagements."""

from tether.knowledge.base import (
    ASSET_COLUMNS,
    TABLES,
    KnowledgeBase,
    comparison_schema_hash,
)
from tether.knowledge.local import LocalKnowledgeBase, open_knowledge_base

__all__ = ["ASSET_COLUMNS", "TABLES", "KnowledgeBase", "LocalKnowledgeBase", "comparison_schema_hash",
           "open_knowledge_base"]
