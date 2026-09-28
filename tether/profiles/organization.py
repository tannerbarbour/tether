"""Organization entity profile (stub: minimal but runnable)."""

from __future__ import annotations

from tether.profiles.base import (
    BlockingRule,
    DeterministicRule,
    EntityProfile,
    FieldSpec,
    HardConstraint,
    register_profile,
)

ORGANIZATION = register_profile(
    EntityProfile(
        name="organization",
        description="Legal entity / practice / facility linked across rosters, Form 990 filings, vendor lists",
        fields={
            "name": FieldSpec("org_name", role="primary", required=True, term_frequency=True,
                              description="Organization legal or trade name"),
            "address": FieldSpec("address", role="primary", description="Organization address"),
            "ein": FieldSpec("ein", role="identifier", description="Employer Identification Number"),
            "npi": FieldSpec("npi", role="identifier", description="Organizational (Type 2) NPI"),
            "phone": FieldSpec("phone", role="supporting", description="Main phone"),
            "email": FieldSpec("email_domain", role="supporting", description="Web/email domain"),
        },
        blocking_rules=[
            BlockingRule(columns=("ein_std",), description="same EIN"),
            BlockingRule(columns=("npi_std",), description="same NPI"),
            BlockingRule(columns=("name_clean",), description="same cleaned name"),
            BlockingRule(columns=("name_dm", "address_zip5"), description="phonetic name + ZIP"),
            BlockingRule(columns=("phone_std",), description="same phone"),
        ],
        training_blocking_rules=[
            BlockingRule(columns=("name_clean",), description="EM: exact cleaned name"),
            BlockingRule(columns=("address_zip5", "address_street_std"), description="EM: exact address"),
        ],
        deterministic_rules=[
            DeterministicRule(match_columns=("ein_std",), description="same valid EIN"),
            DeterministicRule(match_columns=("npi_std",), description="same valid organizational NPI"),
        ],
        hard_constraints=[
            HardConstraint("ein_std", "conflicting valid EINs never share a cluster"),
        ],
    )
)
