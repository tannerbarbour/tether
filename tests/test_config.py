from pathlib import Path

import pytest
import yaml

from ecg_linkage.config import EngagementConfig, SchemaMapping, Thresholds

EXAMPLE = Path(__file__).resolve().parents[1] / "engagements" / "example" / "config.yaml"


def test_example_config_loads_and_resolves_paths():
    cfg = EngagementConfig.load(EXAMPLE)
    assert cfg.profile == "provider" and cfg.matching.link_type == "link_and_dedupe"
    assert cfg.resolve(cfg.sources[0].path).is_absolute()
    assert cfg.resolve(cfg.knowledge_base.path).name == "knowledge_base"
    assert cfg.matching.fields["title"].role == "report_only"


def test_threshold_ordering_enforced():
    with pytest.raises(ValueError):
        Thresholds(auto_link=0.5, review_lower=0.6, cluster=0.9)
    with pytest.raises(ValueError):
        Thresholds(auto_link=0.9, review_lower=0.5, cluster=0.8)


def test_link_type_source_count_consistency(tmp_path):
    raw = yaml.safe_load(EXAMPLE.read_text())
    raw["matching"]["link_type"] = "dedupe_only"
    with pytest.raises(ValueError):
        EngagementConfig.model_validate(raw)


def test_unknown_keys_rejected():
    raw = yaml.safe_load(EXAMPLE.read_text())
    raw["thresholds"]["typo_key"] = 1
    with pytest.raises(ValueError):
        EngagementConfig.model_validate(raw)


def test_schema_mapping_roundtrip(tmp_path):
    m = SchemaMapping(source="roster", profile="provider", status="approved",
                      columns=[{"source_column": "Provider Name", "target": "name", "confidence": 0.9}],
                      unmapped=["Notes"])
    p = tmp_path / "m.yaml"
    m.save(p)
    back = SchemaMapping.load(p)
    assert back.as_dict() == {"Provider Name": "name"} and back.status == "approved"


def test_schema_mapping_duplicate_target():
    with pytest.raises(ValueError):
        SchemaMapping(source="s", profile="provider", columns=[
            {"source_column": "a", "target": "name"}, {"source_column": "b", "target": "name"}])
