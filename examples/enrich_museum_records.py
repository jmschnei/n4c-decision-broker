"""Example: enrich a batch of museum catalog records with Jev via the broker.

Run (live, needs a local Ollama with `ollama pull nimble`):
    python examples/enrich_museum_records.py
"""
from __future__ import annotations

import json
from pathlib import Path

from n4c_decision_broker import (
    AuditLog,
    DecisionCache,
    HoldBackend,
    JevClient,
    ModelBroker,
    QuestionPolicy,
    QuestionSpec,
)

HERE = Path(__file__).resolve().parent


def main() -> None:
    questions = [
        QuestionSpec(
            id="period",
            type="choice",
            instructions="Historical period of the object described in the record",
            criteria={
                "antique": "Classical antiquity (before 500 CE)",
                "medieval": "Medieval (500-1400)",
                "early_modern": "Early modern (1400-1800)",
                "modern": "Modern (1800-1945)",
                "contemporary": "Contemporary (1945-present)",
            },
        ),
        QuestionSpec(
            id="open_access",
            type="noul",
            instructions="The record contains no personal data, no rights issues, and no restrictions on reuse",
        ),
    ]

    broker = ModelBroker(
        client=JevClient(model="nimble"),  # local Ollama
        fallback=HoldBackend(),  # escalated -> human review queue
        cache=DecisionCache(path="audit/cache.json"),
        audit=AuditLog("audit/decisions.jsonl"),
        policy=QuestionPolicy(auto_threshold=0.85, review_threshold=0.5),
    )

    records = json.loads((HERE / "records.example.json").read_text(encoding="utf-8"))
    results = broker.batch(records, questions)

    for i, resp in enumerate(results, 1):
        print(f"\nrecord {i}: {records[i - 1]!r}")
        for a in resp.answers:
            value = a.choice or a.noul_label or a.score
            conf = f" conf={a.confidence:.2f}" if a.confidence is not None else ""
            print(f"   {a.question_id:<12} -> {value!r:<16} {a.status}{conf}  (source={a.source})")
    print("\nmetrics:", json.dumps(broker.metrics()))
    broker.close()


if __name__ == "__main__":
    main()
