"""Append-only JSONL log of every LLM call (prompt hash, model, response, tokens)."""

from __future__ import annotations

import json
from pathlib import Path

from tether.ingestion.llm import LLMCallRecord


class LLMCallLogger:
    def __init__(self, log_dir: str | Path):
        self.path = Path(log_dir) / "llm_calls.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: LLMCallRecord, **context: object) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({**record.as_dict(), **context}, default=str) + "\n")

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
