import pandas as pd

from ecg_linkage.fields.identifiers import is_valid_npi
from ecg_linkage.synthetic import NoiseConfig, SyntheticConfig, generate


def test_deterministic_for_seed():
    a, b = generate(SyntheticConfig(n_entities=50, seed=3)), generate(SyntheticConfig(n_entities=50, seed=3))
    pd.testing.assert_frame_equal(a.source_a, b.source_a)
    pd.testing.assert_frame_equal(a.truth, b.truth)


def test_truth_covers_every_record_and_schemas_differ(small_dataset):
    ds = small_dataset
    ids_a = set(ds.source_a["Provider ID"])
    ids_b = set(ds.source_b["rendering_prov_id"])
    t = ds.truth
    assert set(t[t.source == "roster"].source_record_id) == ids_a
    assert set(t[t.source == "claims"].source_record_id) == ids_b
    assert set(ds.source_a.columns).isdisjoint(set(ds.source_b.columns))
    assert set(t.entity_id) <= set(ds.entities.entity_id)


def test_every_entity_appears_at_least_once(small_dataset):
    assert set(small_dataset.entities.entity_id) == set(small_dataset.truth.entity_id)


def test_noise_ops_recorded_and_duplicates_present(small_dataset):
    ops = small_dataset.truth["noise_ops"].str.split("|").explode()
    assert "within_source_duplicate" in set(ops)
    assert "nickname" in set(ops) and "swap_name_order" in set(ops) and "group_npi" in set(ops)


def test_zero_noise_yields_clean_records():
    zero = NoiseConfig(**{k: 0.0 for k in NoiseConfig.model_fields})
    ds = generate(SyntheticConfig(n_entities=40, seed=1, noise=zero))
    assert (ds.truth["noise_ops"] == "").all()
    npis = ds.source_b["rendering_npi"].dropna()
    assert all(is_valid_npi(n)[0] for n in npis)


def test_invalid_and_group_npis_when_requested():
    noise = NoiseConfig(invalid_npi=1.0, missing_npi=0.0, group_npi=0.0)
    ds = generate(SyntheticConfig(n_entities=30, seed=2, noise=noise))
    assert not any(is_valid_npi(str(n))[0] for n in ds.source_a["NPI"].dropna())
    noise = NoiseConfig(group_npi=1.0)
    ds = generate(SyntheticConfig(n_entities=30, seed=2, noise=noise))
    assert set(ds.source_a["NPI"]) <= set(ds.orgs["org_npi"])


def test_nppes_and_centroids(small_dataset):
    ds = small_dataset
    assert {"NPI", "Entity Type Code", "Provider Last Name (Legal Name)"} <= set(ds.nppes.columns)
    assert (ds.nppes["Entity Type Code"] == 2).sum() == len(ds.orgs)
    assert ds.nppes["NPI"].is_unique
    assert {"GEOID", "INTPTLAT", "INTPTLONG"} <= set(ds.zip_centroids.columns)


def test_true_pairs(small_dataset):
    all_pairs = small_dataset.true_pairs()
    cross = small_dataset.true_pairs(cross_source_only=True)
    assert cross < all_pairs
    assert all(a.split(":")[0] != b.split(":")[0] for a, b in cross)


def test_write(tmp_path, small_dataset):
    paths = small_dataset.write(tmp_path)
    assert (tmp_path / "roster.csv").exists() and (tmp_path / "truth.csv").exists()
    assert paths["config"].exists()
