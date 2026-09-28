import pytest

from ecg_linkage.synthetic import SyntheticConfig, generate


@pytest.fixture(scope="session")
def small_dataset():
    """A small deterministic synthetic dataset shared across tests."""
    return generate(SyntheticConfig(n_entities=150, n_orgs=12, seed=7, nppes_distractors=20))
