"""Synthetic two-source provider data with known ground truth and configurable noise.

Produces:

* ``source_a`` - a *roster*-style file (single "Provider Name" column, one-line address)
* ``source_b`` - a *claims*-style file (split name columns, split address columns)
* ``truth``    - (source, source_record_id, entity_id, noise_ops)
* ``nppes``    - a small NPPES-format reference file covering most entities plus distractors
* ``zip_centroids`` - Census-gazetteer-style ZIP centroid table

Both sources deliberately use different raw schemas so the LLM schema mapper has real
work to do, and both contain within-source duplicates.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from ecg_linkage.fields.identifiers import npi_check_digit
from ecg_linkage.synthetic import pools
from ecg_linkage.synthetic.noise import abbreviate_org, reformat_phone, reformat_street, typo, vary_case


class NoiseConfig(BaseModel):
    """Per-record probabilities for each noise operator (applied independently per source)."""

    model_config = ConfigDict(extra="forbid")

    typo_name: float = Field(0.10, ge=0, le=1, description="one typo in first or last name")
    nickname: float = Field(0.15, ge=0, le=1, description="first name replaced by a nickname")
    swap_name_order: float = Field(0.03, ge=0, le=1)
    missing_middle: float = Field(0.35, ge=0, le=1)
    middle_initial_only: float = Field(0.40, ge=0, le=1, description="given the middle is present")
    legal_suffix_variation: float = Field(0.50, ge=0, le=1)
    org_abbreviation: float = Field(0.20, ge=0, le=1)
    typo_org: float = Field(0.05, ge=0, le=1)
    address_format: float = Field(0.60, ge=0, le=1)
    moved_address: float = Field(0.08, ge=0, le=1, description="different address for the same person")
    zip_plus4: float = Field(0.25, ge=0, le=1)
    missing_npi: float = Field(0.15, ge=0, le=1)
    invalid_npi: float = Field(0.03, ge=0, le=1, description="NPI with a corrupted digit")
    group_npi: float = Field(0.05, ge=0, le=1, description="individual row carries the org's Type-2 NPI")
    missing_phone: float = Field(0.20, ge=0, le=1)
    missing_email: float = Field(0.30, ge=0, le=1)
    missing_title: float = Field(0.10, ge=0, le=1)
    title_variation: float = Field(0.50, ge=0, le=1)
    missing_ein: float = Field(0.30, ge=0, le=1)
    within_source_duplicate: float = Field(0.05, ge=0, le=1)


class SyntheticConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_entities: int = Field(1000, ge=10)
    n_orgs: int = Field(60, ge=2)
    p_in_source_a: float = Field(0.85, gt=0, le=1)
    p_in_source_b: float = Field(0.85, gt=0, le=1)
    source_a_name: str = "roster"
    source_b_name: str = "claims"
    nppes_coverage: float = Field(0.90, ge=0, le=1)
    nppes_distractors: int = Field(200, ge=0)
    own_address_rate: float = Field(0.20, ge=0, le=1, description="entity address differs from org's")
    seed: int = 20240901
    noise: NoiseConfig = Field(default_factory=NoiseConfig)


@dataclass
class SyntheticDataset:
    """All generated frames plus the config that produced them."""

    config: SyntheticConfig
    entities: pd.DataFrame
    orgs: pd.DataFrame
    source_a: pd.DataFrame
    source_b: pd.DataFrame
    truth: pd.DataFrame
    nppes: pd.DataFrame
    zip_centroids: pd.DataFrame

    def write(self, out_dir: str | Path) -> dict[str, Path]:
        """Write every frame as CSV under ``out_dir``; returns ``{name: path}``."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        frames = {
            self.config.source_a_name: self.source_a, self.config.source_b_name: self.source_b,
            "truth": self.truth, "entities": self.entities, "orgs": self.orgs,
            "nppes_sample": self.nppes, "zip_centroids": self.zip_centroids,
        }
        paths: dict[str, Path] = {}
        for name, df in frames.items():
            paths[name] = out / f"{name}.csv"
            df.to_csv(paths[name], index=False)
        (out / "synthetic_config.yaml").write_text(_yaml(self.config.model_dump()), encoding="utf-8")
        paths["config"] = out / "synthetic_config.yaml"
        return paths

    def true_pairs(self, cross_source_only: bool = False) -> set[tuple[str, str]]:
        """Set of unordered record-key pairs (``source:record_id``) that share an entity."""
        keys = self.truth.assign(key=self.truth["source"] + ":" + self.truth["source_record_id"].astype(str))
        pairs: set[tuple[str, str]] = set()
        for _, grp in keys.groupby("entity_id"):
            rows = list(zip(grp["key"], grp["source"]))
            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    if cross_source_only and rows[i][1] == rows[j][1]:
                        continue
                    pairs.add(tuple(sorted((rows[i][0], rows[j][0]))))
        return pairs


def _yaml(obj: Any) -> str:
    import yaml

    return yaml.safe_dump(obj, sort_keys=False)


# ----------------------------------------------------------------- generators
def _random_npi(rng: random.Random, entity_type: int = 1) -> str:
    first_nine = str(entity_type) + "".join(rng.choice("0123456789") for _ in range(8))
    return first_nine + str(npi_check_digit(first_nine))


def _random_ein(rng: random.Random) -> str:
    prefixes = ("31", "34", "36", "37", "38", "41", "42", "43", "47", "52", "58", "61", "62", "75", "91")
    return rng.choice(prefixes) + "".join(rng.choice("0123456789") for _ in range(7))


def _random_phone(rng: random.Random, state: str) -> str:
    area = {"IL": "217", "OH": "614", "IN": "317", "TN": "615", "KY": "502", "MO": "314", "WI": "414",
            "MN": "612", "MI": "313", "PA": "412", "GA": "404", "NC": "704", "TX": "214", "CO": "303",
            "AZ": "602", "WA": "206", "OR": "503"}.get(state, "555")
    return area + "555" + f"{rng.randint(0, 9999):04d}"


def _random_street(rng: random.Random) -> str:
    number = rng.randint(100, 9999)
    direction = rng.choice(pools.DIRECTIONALS)[1] + " " if rng.random() < 0.3 else ""
    name = rng.choice(pools.STREET_NAMES)
    stype = rng.choice(pools.STREET_TYPES)[0]
    unit = f", Suite {rng.randint(100, 900)}" if rng.random() < 0.4 else ""
    return f"{number} {direction}{name} {stype}{unit}"


def _make_orgs(cfg: SyntheticConfig, rng: random.Random) -> list[dict[str, Any]]:
    orgs = []
    seen: set[str] = set()
    while len(orgs) < cfg.n_orgs:
        base = rng.choice(pools.SAINTS) if rng.random() < 0.15 else rng.choice(pools.ORG_ADJECTIVES)
        name = f"{base} {rng.choice(pools.ORG_TYPES)}"
        if name in seen:
            continue
        seen.add(name)
        zip5, city, state, lat, lon = rng.choice(pools.ZIPS)
        domain_base = "".join(ch for ch in name.lower() if ch.isalnum())[:14]
        orgs.append({
            "org_id": f"O{len(orgs) + 1:04d}", "org_name": name,
            "legal_suffix_family": rng.randrange(len(pools.LEGAL_SUFFIX_FORMS)),
            "ein": _random_ein(rng), "org_npi": _random_npi(rng, 2),
            "street": _random_street(rng), "city": city, "state": state, "zip5": zip5,
            "lat": lat, "lon": lon, "phone": _random_phone(rng, state),
            "domain": f"{domain_base}.{rng.choice(pools.EMAIL_TLDS)}",
        })
    return orgs


def _make_entities(cfg: SyntheticConfig, rng: random.Random, orgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entities = []
    for i in range(cfg.n_entities):
        org = rng.choice(orgs)
        role_idx = rng.randrange(len(pools.ROLES))
        credential, canonical_title, _ = pools.ROLES[role_idx]
        first, last = rng.choice(pools.FIRST_NAMES), rng.choice(pools.LAST_NAMES)
        middle = rng.choice(pools.FIRST_NAMES) if rng.random() < 0.8 else None
        if rng.random() < cfg.own_address_rate:
            street, city, state, zip5 = _random_street(rng), org["city"], org["state"], org["zip5"]
        else:
            street, city, state, zip5 = org["street"], org["city"], org["state"], org["zip5"]
        entities.append({
            "entity_id": f"E{i + 1:05d}", "first": first, "middle": middle, "last": last,
            "suffix": rng.choice(pools.GENERATIONAL_SUFFIXES) if rng.random() < 0.05 else None,
            "credential": credential, "role_idx": role_idx, "canonical_title": canonical_title,
            "org_id": org["org_id"], "org_name": org["org_name"], "ein": org["ein"], "org_npi": org["org_npi"],
            "street": street, "city": city, "state": state, "zip5": zip5,
            "npi": _random_npi(rng, 1), "phone": org["phone"] if rng.random() < 0.7 else _random_phone(rng, state),
            "email": f"{first.lower()}.{last.lower().replace(chr(39), '')}@{org['domain']}",
        })
    return entities


# --------------------------------------------------------------------- noise
def _nickname_for(first: str, rng: random.Random) -> str | None:
    from ecg_linkage.fields.person_name import _nicknamer

    options = sorted(_nicknamer().nicknames_of(first.lower()))
    options = [o for o in options if len(o) >= 3]
    return rng.choice(options).capitalize() if options else None


def _noisy_view(entity: dict[str, Any], orgs_by_id: dict[str, dict[str, Any]], noise: NoiseConfig,
                rng: random.Random) -> tuple[dict[str, Any], list[str]]:
    """Apply noise operators to a clean entity, returning (noisy values, applied op names)."""
    ops: list[str] = []
    v: dict[str, Any] = dict(entity)
    org = orgs_by_id[entity["org_id"]]

    # --- name
    if rng.random() < noise.nickname:
        nick = _nickname_for(v["first"], rng)
        if nick:
            v["first"], _ = nick, ops.append("nickname")
    if rng.random() < noise.typo_name:
        if rng.random() < 0.5:
            v["first"] = typo(v["first"], rng)
        else:
            v["last"] = typo(v["last"], rng)
        ops.append("typo_name")
    if v["middle"] is not None:
        if rng.random() < noise.missing_middle:
            v["middle"], _ = None, ops.append("missing_middle")
        elif rng.random() < noise.middle_initial_only:
            v["middle"], _ = v["middle"][0], ops.append("middle_initial")
    if rng.random() < noise.swap_name_order:
        v["first"], v["last"] = v["last"], v["first"]
        ops.append("swap_name_order")

    # --- organization
    forms = pools.LEGAL_SUFFIX_FORMS[org["legal_suffix_family"]]
    suffix = forms[0]
    if rng.random() < noise.legal_suffix_variation:
        suffix, _ = rng.choice(forms), ops.append("legal_suffix_variation")
    org_name = v["org_name"]
    if rng.random() < noise.org_abbreviation:
        org_name, _ = abbreviate_org(org_name, rng), ops.append("org_abbreviation")
    if rng.random() < noise.typo_org:
        org_name, _ = typo(org_name, rng), ops.append("typo_org")
    v["org_name"] = f"{org_name}, {suffix}" if suffix and rng.random() < 0.5 else f"{org_name} {suffix}".strip()

    # --- address
    if rng.random() < noise.moved_address:
        zip5, city, state, _, _ = rng.choice(pools.ZIPS)
        v.update(street=_random_street(rng), city=city, state=state, zip5=zip5)
        ops.append("moved_address")
    if rng.random() < noise.address_format:
        v["street"], _ = reformat_street(v["street"], rng), ops.append("address_format")
    v["zip"] = v["zip5"] + (f"-{rng.randint(1000, 9999)}" if rng.random() < noise.zip_plus4 else "")

    # --- identifiers
    if rng.random() < noise.group_npi:
        v["npi"], _ = v["org_npi"], ops.append("group_npi")
    elif rng.random() < noise.missing_npi:
        v["npi"], _ = None, ops.append("missing_npi")
    elif rng.random() < noise.invalid_npi:
        digits = list(v["npi"])
        i = rng.randrange(1, 10)
        digits[i] = str((int(digits[i]) + rng.randint(1, 9)) % 10)
        v["npi"], _ = "".join(digits), ops.append("invalid_npi")
    if rng.random() < noise.missing_ein:
        v["ein"], _ = None, ops.append("missing_ein")

    # --- contact & title
    if rng.random() < noise.missing_phone:
        v["phone"], _ = None, ops.append("missing_phone")
    else:
        v["phone"] = reformat_phone(v["phone"], rng)
    if rng.random() < noise.missing_email:
        v["email"], _ = None, ops.append("missing_email")
    elif rng.random() < 0.3:
        local, domain = v["email"].split("@")
        f, l_ = local.split(".", 1)
        v["email"] = rng.choice((f"{f[0]}{l_}", f"{f}{l_}", f"{f}_{l_}", local.upper())) + "@" + domain
    variants = pools.ROLES[v["role_idx"]][2]
    if rng.random() < noise.missing_title:
        v["title"], _ = None, ops.append("missing_title")
    elif rng.random() < noise.title_variation:
        v["title"], _ = rng.choice(variants), ops.append("title_variation")
    else:
        v["title"] = variants[0]
    return v, ops


# ------------------------------------------------------------------ renderers
def _render_roster(v: dict[str, Any], record_id: str, rng: random.Random) -> dict[str, Any]:
    middle = f" {v['middle']}" + ("." if v["middle"] and len(v["middle"]) == 1 else "") if v["middle"] else ""
    suffix = f" {v['suffix']}" if v["suffix"] else ""
    style = rng.random()
    if style < 0.5:
        name = f"{v['first']}{middle} {v['last']}{suffix}"
    elif style < 0.8:
        name = f"{v['last']}{suffix}, {v['first']}{middle}"
    else:
        name = f"{v['first']}{middle} {v['last']}{suffix}, {v['credential']}"
    if rng.random() < 0.15:
        name = name.upper()
    unit_free = v["street"]
    address = f"{unit_free}, {v['city']}, {v['state']} {v['zip']}"
    if rng.random() < 0.2:
        address = address.upper()
    return {
        "Provider ID": record_id, "Provider Name": name, "Credentials": v["credential"],
        "NPI": v["npi"], "Practice Name": v["org_name"], "Practice Address": address,
        "Phone": v["phone"], "Email": v["email"], "Job Title": v["title"],
        "Tax ID": (f"{v['ein'][:2]}-{v['ein'][2:]}" if v["ein"] and rng.random() < 0.6 else v["ein"]),
    }


def _render_claims(v: dict[str, Any], record_id: str, rng: random.Random) -> dict[str, Any]:
    upper = rng.random() < 0.6
    fmt = (lambda s: s.upper()) if upper else (lambda s: s)
    last = v["last"] + (f" {v['suffix']}" if v["suffix"] and rng.random() < 0.5 else "")
    return {
        "rendering_prov_id": record_id, "prov_last_name": fmt(last), "prov_first_name": fmt(v["first"]),
        "prov_middle": fmt(v["middle"]) if v["middle"] else None, "rendering_npi": v["npi"],
        "billing_org": vary_case(v["org_name"], rng) if rng.random() < 0.3 else v["org_name"],
        "billing_ein": v["ein"], "addr_line1": fmt(v["street"]), "city": fmt(v["city"]),
        "state": v["state"], "zip": v["zip"], "phone_number": v["phone"], "specialty": v["title"],
    }


# ---------------------------------------------------------------------- nppes
NPPES_COLUMNS = [
    "NPI", "Entity Type Code", "Provider Organization Name (Legal Business Name)",
    "Provider Last Name (Legal Name)", "Provider First Name", "Provider Middle Name",
    "Provider Name Suffix Text", "Provider Credential Text",
    "Provider First Line Business Practice Location Address",
    "Provider Business Practice Location Address City Name",
    "Provider Business Practice Location Address State Name",
    "Provider Business Practice Location Address Postal Code",
    "Provider Business Practice Location Address Telephone Number",
    "Healthcare Provider Taxonomy Code_1", "NPI Deactivation Date",
]
_TAXONOMY_BY_TITLE = {
    "PHYSICIAN": "207Q00000X", "NURSE PRACTITIONER": "363L00000X", "PHYSICIAN ASSISTANT": "363A00000X",
    "REGISTERED NURSE": "163W00000X", "MEDICAL DIRECTOR": "207Q00000X", "SURGEON": "207X00000X",
}


def _nppes_row(e: dict[str, Any]) -> dict[str, Any]:
    return {
        "NPI": e["npi"], "Entity Type Code": 1, "Provider Organization Name (Legal Business Name)": None,
        "Provider Last Name (Legal Name)": e["last"].upper(), "Provider First Name": e["first"].upper(),
        "Provider Middle Name": (e["middle"] or "").upper() or None,
        "Provider Name Suffix Text": (e["suffix"] or "").upper() or None,
        "Provider Credential Text": e["credential"].replace("-", ""),
        "Provider First Line Business Practice Location Address": e["street"].upper(),
        "Provider Business Practice Location Address City Name": e["city"].upper(),
        "Provider Business Practice Location Address State Name": e["state"],
        "Provider Business Practice Location Address Postal Code": e["zip5"],
        "Provider Business Practice Location Address Telephone Number": e["phone"],
        "Healthcare Provider Taxonomy Code_1": _TAXONOMY_BY_TITLE.get(e["canonical_title"], "390200000X"),
        "NPI Deactivation Date": None,
    }


def _nppes_org_row(o: dict[str, Any]) -> dict[str, Any]:
    return {
        "NPI": o["org_npi"], "Entity Type Code": 2,
        "Provider Organization Name (Legal Business Name)":
            f"{o['org_name']} {pools.LEGAL_SUFFIX_FORMS[o['legal_suffix_family']][0]}".strip().upper(),
        "Provider Last Name (Legal Name)": None, "Provider First Name": None, "Provider Middle Name": None,
        "Provider Name Suffix Text": None, "Provider Credential Text": None,
        "Provider First Line Business Practice Location Address": o["street"].upper(),
        "Provider Business Practice Location Address City Name": o["city"].upper(),
        "Provider Business Practice Location Address State Name": o["state"],
        "Provider Business Practice Location Address Postal Code": o["zip5"],
        "Provider Business Practice Location Address Telephone Number": o["phone"],
        "Healthcare Provider Taxonomy Code_1": "261QP2300X", "NPI Deactivation Date": None,
    }


# ------------------------------------------------------------------- driver
def generate(config: SyntheticConfig | None = None) -> SyntheticDataset:
    """Generate a complete synthetic dataset deterministically from ``config.seed``."""
    cfg = config or SyntheticConfig()
    rng = random.Random(cfg.seed)
    orgs = _make_orgs(cfg, rng)
    orgs_by_id = {o["org_id"]: o for o in orgs}
    entities = _make_entities(cfg, rng, orgs)

    source_rows: dict[str, list[dict[str, Any]]] = {cfg.source_a_name: [], cfg.source_b_name: []}
    truth_rows: list[dict[str, Any]] = []
    renderers = {cfg.source_a_name: (_render_roster, "R"), cfg.source_b_name: (_render_claims, "C")}
    counters = {cfg.source_a_name: 0, cfg.source_b_name: 0}

    def emit(source: str, entity: dict[str, Any], extra_ops: list[str]) -> None:
        renderer, prefix = renderers[source]
        counters[source] += 1
        record_id = f"{prefix}{counters[source]:06d}"
        view, ops = _noisy_view(entity, orgs_by_id, cfg.noise, rng)
        source_rows[source].append(renderer(view, record_id, rng))
        truth_rows.append({
            "source": source, "source_record_id": record_id, "entity_id": entity["entity_id"],
            "noise_ops": "|".join(extra_ops + ops),
        })

    for e in entities:
        in_a, in_b = rng.random() < cfg.p_in_source_a, rng.random() < cfg.p_in_source_b
        if not (in_a or in_b):
            in_a = True
        if in_a:
            emit(cfg.source_a_name, e, [])
            if rng.random() < cfg.noise.within_source_duplicate:
                emit(cfg.source_a_name, e, ["within_source_duplicate"])
        if in_b:
            emit(cfg.source_b_name, e, [])
            if rng.random() < cfg.noise.within_source_duplicate:
                emit(cfg.source_b_name, e, ["within_source_duplicate"])

    # Shuffle each source so record order carries no information.
    for source, rows in source_rows.items():
        rng.shuffle(rows)

    nppes_rows = [_nppes_row(e) for e in entities if rng.random() < cfg.nppes_coverage]
    nppes_rows += [_nppes_org_row(o) for o in orgs]
    distractor_cfg = cfg.model_copy(update={"n_entities": cfg.nppes_distractors})
    nppes_rows += [_nppes_row(d) for d in _make_entities(distractor_cfg, rng, orgs)]
    rng.shuffle(nppes_rows)

    zip_df = pd.DataFrame(
        [{"GEOID": z, "INTPTLAT": lat, "INTPTLONG": lon, "CITY": c, "STATE": s} for z, c, s, lat, lon in pools.ZIPS]
    )
    return SyntheticDataset(
        config=cfg,
        entities=pd.DataFrame(entities).drop(columns=["role_idx"]),
        orgs=pd.DataFrame(orgs),
        source_a=pd.DataFrame(source_rows[cfg.source_a_name]),
        source_b=pd.DataFrame(source_rows[cfg.source_b_name]),
        truth=pd.DataFrame(truth_rows),
        nppes=pd.DataFrame(nppes_rows, columns=NPPES_COLUMNS),
        zip_centroids=zip_df,
    )
