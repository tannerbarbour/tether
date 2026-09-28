"""Job title field type: canonical taxonomy mapping (low-weight supporting signal)."""

from __future__ import annotations

from typing import Any, ClassVar, Mapping

from ecg_linkage.fields._text import basic_clean
from ecg_linkage.fields.base import ComparisonOptions, FieldType, Parts, ValidationResult
from ecg_linkage.fields.registry import register_field_type

DEFAULT_TITLE_TAXONOMY: dict[str, str] = {
    # canonical -> canonical
    "PHYSICIAN": "PHYSICIAN", "NURSE PRACTITIONER": "NURSE PRACTITIONER",
    "PHYSICIAN ASSISTANT": "PHYSICIAN ASSISTANT", "REGISTERED NURSE": "REGISTERED NURSE",
    "MEDICAL DIRECTOR": "MEDICAL DIRECTOR", "SURGEON": "SURGEON", "ADMINISTRATOR": "ADMINISTRATOR",
    "PHARMACIST": "PHARMACIST", "THERAPIST": "THERAPIST",
    # synonyms / variants
    "MD": "PHYSICIAN", "DO": "PHYSICIAN", "DOCTOR": "PHYSICIAN", "STAFF PHYSICIAN": "PHYSICIAN",
    "ATTENDING PHYSICIAN": "PHYSICIAN", "HOSPITALIST": "PHYSICIAN", "INTERNIST": "PHYSICIAN",
    "FAMILY MEDICINE PHYSICIAN": "PHYSICIAN", "FAMILY PRACTICE DOCTOR": "PHYSICIAN",
    "FAMILY PHYSICIAN": "PHYSICIAN", "PEDIATRICIAN": "PHYSICIAN", "CARDIOLOGIST": "PHYSICIAN",
    "PRIMARY CARE PHYSICIAN": "PHYSICIAN", "PCP": "PHYSICIAN",
    "NP": "NURSE PRACTITIONER", "ARNP": "NURSE PRACTITIONER", "APRN": "NURSE PRACTITIONER",
    "FNP": "NURSE PRACTITIONER", "FAMILY NURSE PRACTITIONER": "NURSE PRACTITIONER",
    "ADVANCED PRACTICE NURSE": "NURSE PRACTITIONER", "NURSE PRACTIONER": "NURSE PRACTITIONER",
    "PA": "PHYSICIAN ASSISTANT", "PA C": "PHYSICIAN ASSISTANT", "PAC": "PHYSICIAN ASSISTANT",
    "PHYSICIAN ASSOCIATE": "PHYSICIAN ASSISTANT", "PHYSICIANS ASSISTANT": "PHYSICIAN ASSISTANT",
    "RN": "REGISTERED NURSE", "STAFF RN": "REGISTERED NURSE", "STAFF NURSE": "REGISTERED NURSE",
    "NURSE": "REGISTERED NURSE", "CLINICAL NURSE": "REGISTERED NURSE", "CHARGE NURSE": "REGISTERED NURSE",
    "DIRECTOR OF MEDICINE": "MEDICAL DIRECTOR", "DIR OF MEDICINE": "MEDICAL DIRECTOR",
    "MED DIRECTOR": "MEDICAL DIRECTOR", "CHIEF MEDICAL OFFICER": "MEDICAL DIRECTOR", "CMO": "MEDICAL DIRECTOR",
    "ORTHOPEDIC SURGEON": "SURGEON", "GENERAL SURGEON": "SURGEON", "ORTHOPAEDIC SURGEON": "SURGEON",
    "PRACTICE ADMINISTRATOR": "ADMINISTRATOR", "PRACTICE MANAGER": "ADMINISTRATOR",
    "OFFICE MANAGER": "ADMINISTRATOR", "CLINIC MANAGER": "ADMINISTRATOR",
    "RPH": "PHARMACIST", "PHARMD": "PHARMACIST", "CLINICAL PHARMACIST": "PHARMACIST",
    "PHYSICAL THERAPIST": "THERAPIST", "PT": "THERAPIST", "OCCUPATIONAL THERAPIST": "THERAPIST",
}
"""Generic seed taxonomy. The knowledge base's ``lookup_titles`` table (LLM-proposed,
analyst-reviewed) extends it per engagement and, once promoted, across engagements."""


@register_field_type
class JobTitleField(FieldType):
    """Job title mapped to a canonical taxonomy. Supporting signal only."""

    name: ClassVar[str] = "job_title"
    description: ClassVar[str] = "Job title with canonical taxonomy mapping"
    outputs: ClassVar[tuple[str, ...]] = ("std", "canonical")

    def __init__(self, taxonomy: Mapping[str, str] | None = None, use_default_taxonomy: bool = True):
        merged: dict[str, str] = dict(DEFAULT_TITLE_TAXONOMY) if use_default_taxonomy else {}
        if taxonomy:
            merged.update({basic_clean(k): basic_clean(v) for k, v in taxonomy.items()})
        self.taxonomy = merged

    def validate(self, parts: Parts) -> ValidationResult:
        if not basic_clean(parts.get("value") or ""):
            return ValidationResult.fail("empty_after_cleaning")
        return ValidationResult.ok()

    def standardize(self, parts: Parts) -> dict[str, Any]:
        std = basic_clean(parts.get("value") or "") or None
        return {"std": std, "canonical": self.taxonomy.get(std) if std else None}

    def comparisons(self, field_name: str, options: ComparisonOptions):
        import splink.comparison_level_library as cll
        import splink.comparison_library as cl

        std = f"{field_name}_std"
        levels = [
            cll.NullLevel(std),
            cll.ExactMatchLevel(std).configure(label_for_charts="exact title"),
            cll.ExactMatchLevel(f"{field_name}_canonical").configure(label_for_charts="same canonical title"),
            cll.JaroWinklerLevel(std, 0.9).configure(label_for_charts="title JW >= 0.9"),
            cll.ElseLevel(),
        ]
        return [cl.CustomComparison(levels, output_column_name=field_name,
                                    comparison_description="Job title (supporting)")]
