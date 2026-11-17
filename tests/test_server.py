"""REST server: endpoints, validation and API-key authentication (offline)."""
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from n4c_decision_broker import AuditLog, ModelBroker, QuestionPolicy  # noqa: E402
from n4c_decision_broker.server import create_app, load_api_keys  # noqa: E402
from n4c_decision_broker.testing import FakeJev  # noqa: E402

BODY = {
    "state": "Kupferstich, Ansicht von Rom, 17. Jh.",
    "questions": [
        {"id": "lang", "type": "choice", "instructions": "language",
         "criteria": {"de": "German", "en": "English"}},
        {"id": "open", "type": "noul", "instructions": "no rights issues"},
    ],
}


def make(tmp_path, keys=None, audit=True):
    broker = ModelBroker(
        client=FakeJev(seed="srv"),
        audit=AuditLog(tmp_path / "a.jsonl") if audit else None,
        policy=QuestionPolicy(),
    )
    return TestClient(create_app(broker, api_keys=keys)), broker


def test_health_and_decide_without_auth(tmp_path):
    c, _ = make(tmp_path)
    assert c.get("/v1/broker/health").json()["status"] == "ok"
    r = c.post("/v1/broker/decide", json=BODY)
    assert r.status_code == 200
    data = r.json()
    assert [a["question_id"] for a in data["answers"]] == ["lang", "open"]
    assert data["provider_calls"] == 1


def test_questions_as_dict_and_policy_override(tmp_path):
    c, _ = make(tmp_path)
    body = {"state": "x", "questions": {q["id"]: q for q in BODY["questions"]},
            "policy": {"auto_threshold": 0.99, "review_threshold": 0.98}}
    r = c.post("/v1/broker/decide", json=body)
    assert r.status_code == 200
    assert all(a["status"] != "decided" for a in r.json()["answers"] if a["question_id"] == "lang")


def test_validation_errors(tmp_path):
    c, _ = make(tmp_path)
    assert c.post("/v1/broker/decide", json={"state": "x", "questions": []}).status_code == 422
    dup = {"state": "x", "questions": [BODY["questions"][1], BODY["questions"][1]]}
    r = c.post("/v1/broker/decide", json=dup)
    assert r.status_code == 422 and "duplicate" in r.json()["detail"]
    bad = {"state": "x", "questions": [{"id": "c", "type": "choice", "instructions": "i"}]}
    assert c.post("/v1/broker/decide", json=bad).status_code == 422  # choice needs criteria


def test_batch_metrics_audit(tmp_path):
    c, _ = make(tmp_path)
    r = c.post("/v1/broker/batch", json={"records": ["a", "b", "c"], "questions": BODY["questions"]})
    assert r.status_code == 200 and r.json()["count"] == 3
    assert c.get("/v1/broker/metrics").json()["provider_calls"] == 3
    assert len(c.get("/v1/broker/audit?limit=2").json()) == 2


def test_audit_404_without_log(tmp_path):
    c, _ = make(tmp_path, audit=False)
    assert c.get("/v1/broker/audit").status_code == 404


# -- authentication ------------------------------------------------------------
PROTECTED = [("post", "/v1/broker/decide", BODY),
             ("post", "/v1/broker/batch", {"records": ["a"], "questions": BODY["questions"]}),
             ("get", "/v1/broker/metrics", None),
             ("get", "/v1/broker/audit", None)]


def _call(c, method, path, body, headers=None):
    return getattr(c, method)(path, headers=headers or {}, **({"json": body} if body else {}))


@pytest.mark.parametrize("method,path,body", PROTECTED)
def test_protected_endpoints_reject_missing_and_wrong_key(tmp_path, method, path, body):
    c, broker = make(tmp_path, keys=["s3cret"])
    for headers in ({}, {"Authorization": "Bearer nope"}, {"X-API-Key": "nope"},
                    {"Authorization": "Basic s3cret"}):
        r = _call(c, method, path, body, headers)
        assert r.status_code == 401, headers
        assert r.headers["www-authenticate"] == "Bearer"
    assert broker.provider_calls == 0  # rejected requests never reach the model


@pytest.mark.parametrize("method,path,body", PROTECTED)
def test_protected_endpoints_accept_valid_key(tmp_path, method, path, body):
    c, _ = make(tmp_path, keys=["old-key", "s3cret"])  # rotation: several keys valid
    assert _call(c, method, path, body, {"Authorization": "Bearer s3cret"}).status_code == 200
    assert _call(c, method, path, body, {"X-API-Key": "old-key"}).status_code == 200


def test_health_stays_open_with_auth(tmp_path):
    c, _ = make(tmp_path, keys=["s3cret"])
    assert c.get("/v1/broker/health").status_code == 200


def test_load_api_keys():
    assert load_api_keys({"N4C_BROKER_API_KEYS": " a, b ,,c "}) == ["a", "b", "c"]
    assert load_api_keys({}) == []
