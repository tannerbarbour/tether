"""CMS NPPES (National Plan and Provider Enumeration System) reference source.

Works from a locally downloaded NPPES data-dissemination CSV (or a subset with the same
headers). Two uses:

* **enrichment** - attach entity type, legal name and practice location to source
  records by validated NPI. For a person profile, Type 2 (organization) NPIs are
  invalidated for linking: they identify a practice, not a person.
* **hub linking** - expose NPPES individuals as a canonical frame so every source is
  resolved against NPPES and joined through NPI.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from tether.fields.identifiers import is_valid_npi
from tether.reference.base import ReferenceSource

NPPES_COLUMN_MAP = {
    "NPI": "npi",
    "Entity Type Code": "entity_type",
    "Provider Organization Name (Legal Business Name)": "org_name",
    "Provider Last Name (Legal Name)": "last",
    "Provider First Name": "first",
    "Provider Middle Name": "middle",
    "Provider Name Suffix Text": "suffix",
    "Provider Credential Text": "credential",
    "Provider First Line Business Practice Location Address": "street",
    "Provider Business Practice Location Address City Name": "city",
    "Provider Business Practice Location Address State Name": "state",
    "Provider Business Practice Location Address Postal Code": "zip",
    "Provider Business Practice Location Address Telephone Number": "phone",
    "Healthcare Provider Taxonomy Code_1": "taxonomy",
    "NPI Deactivation Date": "deactivation_date",
}
ENRICHMENT_COLUMNS = ["entity_type", "first", "last", "org_name", "state", "zip", "taxonomy", "deactivation_date"]


class NPPESReference(ReferenceSource):
    name = "nppes"
    license = "Public domain (CMS NPPES Data Dissemination)"
    license_url = "https://download.cms.gov/nppes/NPI_Files.html"

    def _load(self) -> pd.DataFrame:
        raw = pd.read_csv(self.path, dtype=str, keep_default_na=False, usecols=lambda c: c in NPPES_COLUMN_MAP)
        df = raw.rename(columns=NPPES_COLUMN_MAP)
        for c in NPPES_COLUMN_MAP.values():
            if c not in df.columns:
                df[c] = ""
        df = df.replace({"": None})
        df["entity_type"] = pd.to_numeric(df["entity_type"], errors="coerce").astype("Int64")
        df = df[df["npi"].map(lambda v: bool(v) and is_valid_npi(v)[0])]
        df["zip"] = df["zip"].map(lambda z: z[:5] if isinstance(z, str) else None)
        return df.drop_duplicates(subset=["npi"]).reset_index(drop=True)

    # ---------------------------------------------------------------- enrich
    def enrich(self, records: pd.DataFrame, npi_column: str = "npi_std", person_profile: bool = True) -> pd.DataFrame:
        """Attach ``nppes_*`` columns by validated NPI; invalidate Type 2 NPIs for person linking.

        Adds ``nppes_found`` (bool). For a person profile, records whose NPI resolves to an
        organization get ``<npi>_std`` cleared and ``<npi>_invalid_reason='type2_organization_npi'``,
        keeping the digits in ``<npi>_digits`` for review. Deactivated NPIs are flagged
        (``nppes_deactivated``) but remain usable.
        """
        ref = self.frame().set_index("npi")[ENRICHMENT_COLUMNS].add_prefix("nppes_")
        out = records.merge(ref, left_on=npi_column, right_index=True, how="left")
        out["nppes_found"] = out["nppes_entity_type"].notna()
        out["nppes_deactivated"] = out["nppes_deactivation_date"].notna()
        if person_profile:
            org_npi = out["nppes_entity_type"] == 2
            field = npi_column.removesuffix("_std")
            out.loc[org_npi, npi_column] = None
            out.loc[org_npi, f"{field}_valid"] = False
            out.loc[org_npi, f"{field}_invalid_reason"] = "type2_organization_npi"
        return out

    # --------------------------------------------------------------- hub frame
    def hub_frame(self, canonical_columns: Sequence[str], entity_type: int = 1,
                  source_name: str = "nppes") -> pd.DataFrame:
        """NPPES records of ``entity_type`` as a canonical frame for hub linking."""
        ref = self.frame()
        ref = ref[ref["entity_type"] == entity_type]
        out = pd.DataFrame({c: None for c in canonical_columns}, index=ref.index)
        out["unique_id"] = source_name + ":" + ref["npi"]
        out["source_dataset"] = source_name
        out["source_record_id"] = ref["npi"]
        mapping = {"npi": "npi", "name_first": "first", "name_middle": "middle", "name_last": "last",
                   "name_suffix": "suffix", "org": "org_name", "address_street": "street", "address_city": "city",
                   "address_state": "state", "address_zip": "zip", "phone": "phone"} if entity_type == 1 else \
            {"npi": "npi", "name": "org_name", "address_street": "street", "address_city": "city",
             "address_state": "state", "address_zip": "zip", "phone": "phone"}
        for target, src in mapping.items():
            if target in out.columns:
                out[target] = ref[src].values
        cols = ["unique_id", "source_dataset", "source_record_id"] + list(canonical_columns)
        return out[cols].reset_index(drop=True)


def hub_entity_ids(crosswalk: pd.DataFrame, hub_source: str = "nppes", prefix: str = "NPI-") -> pd.DataFrame:
    """Relabel entity ids of clusters containing a hub record as ``NPI-<npi>``; flag hub rows."""
    cw = crosswalk.copy()
    hub = cw[cw["source"] == hub_source]
    by_entity = hub.groupby("entity_id")["source_record_id"].agg(lambda s: prefix + sorted(s)[0])
    cw["hub_npi"] = cw["entity_id"].map(hub.groupby("entity_id")["source_record_id"].first())
    cw["entity_id"] = cw["entity_id"].map(by_entity).fillna(cw["entity_id"])
    cw["is_reference"] = cw["source"] == hub_source
    return cw
