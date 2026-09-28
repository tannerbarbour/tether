"""LLM client interface with an Azure OpenAI implementation and an offline mock.

Design rules (see architecture): the LLM only *proposes*; deterministic code applies.
Every call is structured (pydantic response model), temperature 0, seeded, and logged.
The mock reads the structured ``payload`` of a request instead of the prompt text, so
the full pipeline and test suite run offline with plausible behaviour.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from tether.config import LLMConfig

T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMRequest:
    """A structured request. ``payload`` is the data the prompt was rendered from."""

    purpose: str
    system: str
    user: str
    payload: dict[str, Any] = field(default_factory=dict)
    temperature: float = 0.0
    seed: int = 0

    def prompt_hash(self) -> str:
        return hashlib.sha256((self.system + "\n---\n" + self.user).encode("utf-8")).hexdigest()


@dataclass
class LLMCallRecord:
    """Audit record for one call (persisted by ``LLMCallLogger``)."""

    timestamp: str
    provider: str
    model: str
    purpose: str
    prompt_hash: str
    temperature: float
    seed: int
    response_json: str
    latency_ms: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class LLMClient(Protocol):
    """Any client: returns a validated ``response_model`` instance plus an audit record."""

    provider: str
    model: str

    def complete(self, request: LLMRequest, response_model: type[T]) -> tuple[T, LLMCallRecord]: ...


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------- Azure
class AzureOpenAIClient:
    """Azure OpenAI chat completions with structured (JSON schema) output.

    Configured from environment variables only (no secrets in code or config):

    * ``AZURE_OPENAI_ENDPOINT``      https://<resource>.openai.azure.com
    * ``AZURE_OPENAI_API_KEY``       or ``AZURE_OPENAI_AD_TOKEN`` for Entra ID auth
    * ``AZURE_OPENAI_DEPLOYMENT``    deployment (model) name
    * ``AZURE_OPENAI_API_VERSION``   default ``2024-10-21``
    """

    provider = "azure_openai"

    def __init__(self, client: Any | None = None, deployment: str | None = None):
        self.model = deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT") or ""
        if client is not None:
            self._client = client
            return
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        ad_token = os.environ.get("AZURE_OPENAI_AD_TOKEN")
        missing = [n for n, v in (("AZURE_OPENAI_ENDPOINT", endpoint), ("AZURE_OPENAI_DEPLOYMENT", self.model)) if not v]
        if not (api_key or ad_token):
            missing.append("AZURE_OPENAI_API_KEY (or AZURE_OPENAI_AD_TOKEN)")
        if missing:
            raise EnvironmentError(f"Azure OpenAI not configured; missing environment variables: {missing}")
        from openai import AzureOpenAI

        self._client = AzureOpenAI(
            azure_endpoint=endpoint, api_key=api_key, azure_ad_token=ad_token,
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        )

    def complete(self, request: LLMRequest, response_model: type[T]) -> tuple[T, LLMCallRecord]:
        t0 = time.time()
        completion = self._client.chat.completions.parse(
            model=self.model,
            messages=[{"role": "system", "content": request.system}, {"role": "user", "content": request.user}],
            response_format=response_model,
            temperature=request.temperature,
            seed=request.seed,
        )
        choice = completion.choices[0]
        parsed = choice.message.parsed
        if parsed is None:
            raise ValueError(f"model returned no parsable output (refusal={getattr(choice.message, 'refusal', None)!r})")
        usage = getattr(completion, "usage", None)
        record = LLMCallRecord(
            timestamp=_now(), provider=self.provider, model=self.model, purpose=request.purpose,
            prompt_hash=request.prompt_hash(), temperature=request.temperature, seed=request.seed,
            response_json=parsed.model_dump_json(), latency_ms=int((time.time() - t0) * 1000),
            prompt_tokens=getattr(usage, "prompt_tokens", None), completion_tokens=getattr(usage, "completion_tokens", None),
        )
        return parsed, record


# ---------------------------------------------------------------------- Mock
class MockLLMClient:
    """Deterministic offline stand-in.

    Handlers are registered per ``purpose`` and receive ``request.payload``; the built-in
    handlers (schema mapping, title and org-alias standardization, identifier extraction)
    live next to the code that builds those requests. Scripted responses can be queued
    for tests via ``script``.
    """

    provider = "mock"
    model = "mock-v1"

    def __init__(self, handlers: dict[str, Any] | None = None, script: list[BaseModel] | None = None):
        from tether.ingestion import mock_handlers

        self.handlers: dict[str, Any] = dict(mock_handlers.DEFAULT_HANDLERS)
        self.handlers.update(handlers or {})
        self.script = list(script or [])
        self.calls: list[LLMRequest] = []

    def complete(self, request: LLMRequest, response_model: type[T]) -> tuple[T, LLMCallRecord]:
        t0 = time.time()
        self.calls.append(request)
        if self.script:
            result = self.script.pop(0)
        else:
            handler = self.handlers.get(request.purpose)
            if handler is None:
                raise KeyError(f"mock LLM has no handler for purpose '{request.purpose}'")
            result = handler(request.payload)
        parsed = result if isinstance(result, response_model) else response_model.model_validate(result)
        record = LLMCallRecord(
            timestamp=_now(), provider=self.provider, model=self.model, purpose=request.purpose,
            prompt_hash=request.prompt_hash(), temperature=request.temperature, seed=request.seed,
            response_json=parsed.model_dump_json(), latency_ms=int((time.time() - t0) * 1000),
        )
        return parsed, record


def make_client(config: LLMConfig) -> LLMClient:
    """Build the configured client (``mock`` or ``azure``)."""
    if config.provider == "azure":
        return AzureOpenAIClient()
    return MockLLMClient()


def dumps_payload(payload: dict[str, Any]) -> str:
    """Stable JSON rendering used inside prompts (keeps prompt hashes reproducible)."""
    return json.dumps(payload, indent=2, sort_keys=True, default=str)
