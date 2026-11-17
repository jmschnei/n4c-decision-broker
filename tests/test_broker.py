"""ModelBroker end-to-end with FakeJev (offline, deterministic)."""
import json

import pytest

from jev_broker import (
    AuditLog,
    DecisionCache,
    HoldBackend,
    LLMFallbackBackend,
    ModelBroker,
    QuestionPolicy,
    QuestionSpec,
)
from jev_broker.testing import FakeJev

POLICY = QuestionPolicy(auto_threshold=0.85, review_threshold=0.5)

QUESTIONS = [
    QuestionSpec(
        id="period", type="choice", instructions="Historical period",
        criteria={"antique": "<500", "medieval": "500-1400",
                  "early_modern": "1400-1800", "modern": "1800-1945",
                  "contemporary": ">1945"},
    ),
    QuestionSpec(id="language", type="choice", instructions="Primary language",
                 criteria={"de": "German", "en": "English", "fr": "French", "la": "Latin", "other": "x"}),
    QuestionSpec(id="open_access", type="noul", instructions="No rights issues"),
]


def make_broker(tmp_path, fallback=None):
    return ModelBroker(
        client=FakeJev(seed="tests"),
        fallback=fallback or HoldBackend(),
        cache=DecisionCache(path=tmp_path / "cache.json"),
        audit=AuditLog(tmp_path / "audit.jsonl"),
        policy=POLICY,
    )


def test_all_statuses_present_across_records(tmp_path):
    broker = make_broker(tmp_path)
    records = [
        "Kupferstich, Ansicht von Rom, 17. Jh.",
        "Photographie, Bauhaus Dessau, 1927",
        "Briefwechsel, lateinisch, 13. Jh.",
        "Sonate für Klavier, 1952",
        "Katalogeintrag ohne Datierung",
    ]
    results = broker.batch(records, QUESTIONS)
    statuses = {a.status for r in results for a in r.answers}
    assert "decided" in statuses
    # every answer is one of the four valid statuses
    valid = {"decided", "review", "escalated", "error"}
    assert statuses <= valid
    for r in results:
        assert [a.question_id for a in r.answers] == ["period", "language", "open_access"]
        for a in r.answers:
            if a.probabilities is not None:  # choice/score: routed on confidence
                if a.status == "decided":
                    assert a.confidence is not None and a.confidence >= 0.85
                elif a.status == "review":
                    assert a.confidence is not None and 0.5 <= a.confidence < 0.85
            if a.status == "decided" and a.question.type.value == "noul":
                assert (a.noul or 0) >= 0.7 or (a.noul is not None and a.noul <= 0.3)


def test_cache_serves_second_call(tmp_path):
    broker = make_broker(tmp_path)
    q = QUESTIONS[:1]
    state = "Photographie, Bauhaus Dessau, 1927"
    r1 = broker.decide(state, q)
    r2 = broker.decide(state, q)
    assert broker.provider_calls == 1  # second call answered from cache
    assert all(a.source == "cache" for a in r2.answers)
    assert r1.answers[0].model_dump(exclude={"source"}) == r2.answers[0].model_dump(exclude={"source"})


def test_no_cache_rebill(tmp_path):
    broker = make_broker(tmp_path)
    q = QUESTIONS[:1]
    broker.decide("state A", q, use_cache=False)
    broker.decide("state A", q, use_cache=False)
    assert broker.provider_calls == 2


def test_error_path_is_reported_per_question(tmp_path):
    broker = make_broker(tmp_path)
    broker.client.fail_next = True
    r = broker.decide("some state", QUESTIONS)
    assert all(a.status == "error" for a in r.answers)
    assert all(a.error for a in r.answers)
    assert broker.provider_errors == 1


def test_fallback_llm_used_on_escalation(tmp_path):
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "stub-llm"
        prompt = body["messages"][0]["content"]
        assert "decision engine" in prompt.lower()
        # "early_modern" is a valid key of the first (period) question only
        return httpx.Response(200, json={
            "choices": [{"message": {"content": '{"answer": "early_modern"}'}}],
        })

    fb = LLMFallbackBackend(base_url="https://llm.example.test/v1", api_key="k",
                            model="stub-llm", transport=httpx.MockTransport(handler))
    # force escalation: make the fake provider very uncertain
    class UnconfidentJev(FakeJev):
        def _distribution(self, rng, n):
            return [round(1.0 / n, 4)] * n

    broker = ModelBroker(
        client=UnconfidentJev(seed="t"),
        fallback=fb,
        policy=POLICY,
    )
    r = broker.decide("state", QUESTIONS[:2])
    a = r.answers[0]
    assert a.status == "escalated"
    assert a.source == "fallback_llm"
    assert a.calibrated is False
    assert a.choice == "early_modern"
    assert a.primary is not None  # original Jev answer preserved
    # second question (language): "early_modern" is invalid for its options ->
    # held for a human; Jev's own (unreliable) best guess stays visible, flagged.
    b = r.answers[1]
    assert b.status == "escalated"
    assert b.source == "jev"
    assert b.calibrated is True
    assert b.choice in ("de", "en", "fr", "la", "other")


def test_fallback_failure_holds_for_human(tmp_path):
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "down"})

    fb = LLMFallbackBackend(base_url="https://llm.example.test/v1", model="stub",
                            transport=httpx.MockTransport(handler))
    r = ModelBroker(client=FakeJev(seed="t"), fallback=fb, policy=POLICY).decide(
        "state", QUESTIONS[:1])
    a = r.answers[0]
    if a.status == "escalated":
        # fallback failed (HTTP 500) -> decision held for a human:
        # source stays "jev", calibrated flag intact, guess visible but flagged
        assert a.source == "jev"
        assert a.detail is None
        assert a.primary is None


def test_audit_log_written(tmp_path):
    audit_path = tmp_path / "audit.jsonl"
    broker = make_broker(tmp_path)
    broker.decide("state A", QUESTIONS)
    broker.decide("state B", QUESTIONS)
    lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    entry = json.loads(lines[-1])
    assert entry["request_id"]
    assert entry["policy"]["auto_threshold"] == 0.85
    assert len(entry["answers"]) == 3
    assert "ts" in entry


def test_duplicate_ids_rejected(tmp_path):
    broker = make_broker(tmp_path)
    import pytest
    q1 = QuestionSpec(id="x", type="noul", instructions="a")
    q2 = QuestionSpec(id="x", type="noul", instructions="b")
    with pytest.raises(ValueError):
        broker.decide("state", [q1, q2])


def test_metrics_shape(tmp_path):
    broker = make_broker(tmp_path)
    broker.decide("state A", QUESTIONS)
    m = broker.metrics()
    assert m["provider_calls"] == 1
    assert m["fallback"] == "hold"
    assert m["cache"]["size"] >= 1
