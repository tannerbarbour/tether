"""Prompt templates and structured response models for the ingestion LLM tasks."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from tether.ingestion.llm import LLMRequest, dumps_payload


# ------------------------------------------------------------ schema mapping
class MappingProposalItem(BaseModel):
    source_column: str
    target: str = Field(description="One of the allowed canonical targets, or UNMAPPED")
    confidence: float = Field(ge=0, le=1)
    rationale: str = ""


class MappingProposal(BaseModel):
    columns: list[MappingProposalItem]


SCHEMA_MAPPING_SYSTEM = """You map columns of a tabular data source onto a fixed canonical schema for
entity resolution. Rules:
- Choose each target from the allowed list only; use UNMAPPED when nothing fits.
- Map a column to a whole-value target (e.g. "name") only if it holds the whole value; map to a
  component target (e.g. "name_last") when the source splits values across columns.
- Never invent, infer or output data values; output only column names, targets, confidence and a
  short rationale.
- Each target may be used at most once."""


def schema_mapping_request(payload: dict[str, Any]) -> LLMRequest:
    user = (
        "Allowed targets (name: description):\n" + dumps_payload(payload["targets"]) + "\n\n"
        + "Columns to map, with sample values (may be masked):\n" + dumps_payload(payload["columns"]) + "\n\n"
        + ("Previously approved mappings from similar sources (few-shot examples):\n" + dumps_payload(payload["examples"]) + "\n\n"
           if payload.get("examples") else "")
        + "Return the mapping."
    )
    return LLMRequest(purpose="schema_mapping", system=SCHEMA_MAPPING_SYSTEM, user=user, payload=payload)


# -------------------------------------------------------- standardization
class ValueMapping(BaseModel):
    value: str
    canonical: str
    confidence: float = Field(ge=0, le=1)


class ValueMappingBatch(BaseModel):
    items: list[ValueMapping]


TITLE_SYSTEM = """You classify job titles into a fixed canonical taxonomy for healthcare workforce
analytics. Choose exactly one canonical category from the allowed list for every input value;
use OTHER when none fits. Output only the mapping."""

ORG_ALIAS_SYSTEM = """You expand abbreviations and aliases in healthcare organization names into their
full, conventional form (e.g. "St. Mary's Med Ctr" -> "SAINT MARYS MEDICAL CENTER"). Preserve the
organization's identity: never add, remove or guess location words, numbers or legal suffixes.
Output uppercase without punctuation. If a value needs no change, return it unchanged."""


def title_request(values: list[str], allowed: list[str]) -> LLMRequest:
    payload = {"values": values, "allowed": allowed}
    user = "Allowed categories:\n" + dumps_payload(allowed) + "\n\nValues:\n" + dumps_payload(values)
    return LLMRequest(purpose="standardize_titles", system=TITLE_SYSTEM, user=user, payload=payload)


def org_alias_request(values: list[str]) -> LLMRequest:
    payload = {"values": values}
    return LLMRequest(purpose="standardize_org_aliases", system=ORG_ALIAS_SYSTEM,
                      user="Values:\n" + dumps_payload(values), payload=payload)


# --------------------------------------------------------------- extraction
class IdentifierCandidate(BaseModel):
    kind: str = Field(description="npi or ein")
    value: str = Field(description="verbatim as it appears in the text")


class IdentifierExtraction(BaseModel):
    candidates: list[IdentifierCandidate]


EXTRACTION_SYSTEM = """Extract healthcare identifiers (NPI: 10 digits; EIN: 9 digits, often NN-NNNNNNN)
that appear VERBATIM in the text. Copy characters exactly; never complete, correct or infer digits."""


def extraction_request(text: str) -> LLMRequest:
    return LLMRequest(purpose="extract_identifiers", system=EXTRACTION_SYSTEM, user="Text:\n" + text,
                      payload={"text": text})
