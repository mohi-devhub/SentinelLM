"""Offline dataset format for SentinelLM v2 evaluation.

V2 JSONL format (backward-compatible with v1 golden_qa.jsonl):
    {
      "input": "What is the capital of France?",
      "output": "Paris is the capital of France.",   # required for offline mode
      "model": "llama3.2",                           # optional
      "context_documents": ["France is ..."],        # optional, for RAG evaluators
      "expected_output": "Paris.",                   # optional
      "expected_blocked": false,                     # optional
      "tags": ["qa", "geography"],                   # optional
      "prompt_version": "customer-support-v3",       # optional
      "record_id": "stable-deterministic-id"         # optional
    }

The only field required beyond v1 is `output`. Records missing `output` are
skipped with a warning so v1 datasets can be reused without modification — they
will simply score zero records (caller should warn the user explicitly).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class OfflineDatasetRecord:
    """A single record from a v2 offline evaluation dataset."""

    record_index: int
    input_text: str
    output_text: str
    model: str = "llama3.2"
    context_documents: list[str] = field(default_factory=list)
    expected_output: str | None = None
    expected_blocked: bool = False
    tags: list[str] = field(default_factory=list)
    prompt_version: str | None = None
    record_id: str | None = None


def load_offline_dataset(path: Path) -> list[OfflineDatasetRecord]:
    """Parse a v2 JSONL file into OfflineDatasetRecord objects.

    Records missing the required `output` field are skipped and a warning is
    printed to stderr. Blank lines and lines starting with '#' are ignored so
    the file can include comments and visual separators.
    """
    records: list[OfflineDatasetRecord] = []
    with open(path, encoding="utf-8") as fh:
        for raw_idx, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            data = json.loads(line)
            if "output" not in data:
                print(
                    f"[warn] {path}:{raw_idx}: missing 'output' field — skipping record",
                    file=sys.stderr,
                )
                continue
            records.append(
                OfflineDatasetRecord(
                    record_index=len(records),
                    input_text=data["input"],
                    output_text=data["output"],
                    model=data.get("model", "llama3.2"),
                    context_documents=data.get("context_documents", []),
                    expected_output=data.get("expected_output"),
                    expected_blocked=data.get("expected_blocked", False),
                    tags=data.get("tags", []),
                    prompt_version=data.get("prompt_version"),
                    record_id=data.get("record_id"),
                )
            )
    return records
