"""Provider (individual clinician) entity profile - the pilot profile."""

from __future__ import annotations

from ecg_linkage.profiles.base import (
    BlockingRule,
    DeterministicRule,
    EntityProfile,
    FieldSpec,
    Guard,
    HardConstraint,
    register_profile,
)

PROVIDER = register_profile(
    EntityProfile(
        name="provider",
        description="Individual healthcare provider (clinician) linked across rosters, claims, surveys",
        fields={
            "name": FieldSpec("person_name", role="primary", required=True, term_frequency=True,
                              description="Provider's personal name"),
            "org": FieldSpec("org_name", role="primary", term_frequency=True,
                             description="Employer / practice / billing organization name"),
            "address": FieldSpec("address", role="primary", description="Practice location address"),
            "npi": FieldSpec("npi", role="identifier", description="National Provider Identifier"),
            # EIN identifies the employer, not the person: supporting evidence only.
            "ein": FieldSpec("ein", role="supporting", description="Employer / billing tax ID (EIN)"),
            "phone": FieldSpec("phone", role="supporting", description="Practice phone number"),
            "email": FieldSpec("email_domain", role="supporting", term_frequency=True,
                               description="Work email address (domain used)"),
            # Title is correlated with org/address (same practice => same title); kept out of
            # the model to respect conditional independence, shown in review output.
            "title": FieldSpec("job_title", role="report_only", description="Job title / role"),
        },
        blocking_rules=[
            BlockingRule(columns=("npi_std",), description="same valid NPI"),
            BlockingRule(columns=("name_last_dm", "name_first_dm"), description="phonetic first+last"),
            BlockingRule(columns=("name_last_std", "address_zip5"), description="last name + ZIP"),
            BlockingRule(columns=("name_first_std", "org_clean"), description="first name + org"),
            BlockingRule(columns=("name_last_std", "org_clean"), description="last name + org"),
            BlockingRule(columns=("phone_std",), description="same phone"),
            BlockingRule(
                sql="l.name_first_std = r.name_last_std and l.name_last_std = r.name_first_std",
                description="first/last swapped",
            ),
        ],
        training_blocking_rules=[
            # Round 1 estimates every comparison except the name comparisons ...
            BlockingRule(columns=("name_last_std", "name_first_std"), description="EM: exact name"),
            # ... round 2 estimates the name comparisons.
            BlockingRule(columns=("org_clean", "address_zip5"), description="EM: org + ZIP"),
        ],
        deterministic_rules=[
            DeterministicRule(
                match_columns=("npi_std",),
                # A shared Type-2 (group) NPI on two different clinicians must not link:
                # require the last names not to contradict (swap-aware).
                guards=(Guard(column="name_last_std", jw_threshold=0.85, swap_with="name_first_std"),),
                description="same valid NPI and non-contradicting last name",
            ),
        ],
        hard_constraints=[HardConstraint("npi_std", "conflicting valid NPIs never share a cluster")],
        max_cluster_size=6,
        min_edge_density=0.5,
    )
)
