import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from tether.config import LLMConfig
from tether.ingestion import AzureOpenAIClient, LLMCallLogger, LLMRequest, MockLLMClient, make_client
from tether.ingestion.prompts import MappingProposal, ValueMappingBatch


class Out(BaseModel):
    answer: str


def test_mock_scripted_and_handler_and_record():
    client = MockLLMClient(script=[Out(answer="hi")])
    out, rec = client.complete(LLMRequest("x", "s", "u"), Out)
    assert out.answer == "hi" and rec.provider == "mock" and rec.purpose == "x" and len(rec.prompt_hash) == 64
    with pytest.raises(KeyError):
        client.complete(LLMRequest("unknown_purpose", "s", "u"), Out)
    out2, _ = client.complete(LLMRequest("standardize_titles", "s", "u", payload={"values": ["RN"], "allowed": ["REGISTERED NURSE"]}), ValueMappingBatch)
    assert out2.items[0].canonical == "REGISTERED NURSE"


def test_prompt_hash_is_deterministic():
    a, b = LLMRequest("p", "sys", "user"), LLMRequest("p", "sys", "user")
    assert a.prompt_hash() == b.prompt_hash() != LLMRequest("p", "sys", "user2").prompt_hash()


def test_make_client_defaults_to_mock():
    assert make_client(LLMConfig(provider="mock")).provider == "mock"


def test_azure_requires_environment(monkeypatch):
    for v in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_AD_TOKEN"):
        monkeypatch.delenv(v, raising=False)
    with pytest.raises(EnvironmentError):
        AzureOpenAIClient()


def test_azure_client_with_injected_sdk_client():
    captured = {}

    def parse(**kwargs):
        captured.update(kwargs)
        parsed = kwargs["response_format"].model_validate({"columns": [
            {"source_column": "c", "target": "npi", "confidence": 0.9, "rationale": "r"}]})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed, refusal=None))],
                               usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)))
    client = AzureOpenAIClient(client=fake, deployment="gpt-test")
    out, rec = client.complete(LLMRequest("schema_mapping", "sys", "usr", temperature=0.0, seed=7), MappingProposal)
    assert out.columns[0].target == "npi"
    assert captured["temperature"] == 0.0 and captured["seed"] == 7 and captured["model"] == "gpt-test"
    assert rec.prompt_tokens == 10 and rec.model == "gpt-test" and rec.provider == "azure_openai"


def test_logger_roundtrip(tmp_path):
    client = MockLLMClient(script=[Out(answer="a")])
    _, rec = client.complete(LLMRequest("x", "s", "u"), Out)
    log = LLMCallLogger(tmp_path)
    log.write(rec, source="roster")
    rows = log.read()
    assert rows[0]["source"] == "roster" and json.loads(rows[0]["response_json"]) == {"answer": "a"}
