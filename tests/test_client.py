"""JevClient: request building, response normalization, error handling."""
import json

import httpx
import pytest

from jev_broker.client import JevAPIError, JevClient
from jev_broker.models import QuestionSpec


def make_client(handler):
    return JevClient(api_key="test-key", base_url="https://api.example.test",
                     transport=httpx.MockTransport(handler))


RAW_RESPONSE = {
    "choices": {"dept": {"choice": "technical",
                         "probabilities": {"billing": 0.05, "technical": 0.9, "sales": 0.05},
                         "confidence": 0.85}},
    "scores": {"frustration": {"score": 1.2,
                               "probabilities": {"0": 0.2, "1": 0.6, "2": 0.2},
                               "confidence": 0.55}},
    "nouls": {"is_urgent": {"noul": 0.97}},
}


def test_request_payload_shape():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=RAW_RESPONSE)

    client = make_client(handler)
    q = QuestionSpec(id="dept", type="choice", instructions="Which team",
                     criteria={"billing": "b", "technical": "t", "sales": "s"})
    client.system_one("ticket text", {"dept": q})

    assert seen["url"] == "https://api.example.test/v1/systemone"
    assert seen["auth"] == "Bearer test-key"
    body = seen["body"]
    assert body["model"] == "nimble"
    assert body["keep_alive"] == "5m"
    assert body["state"] == "ticket text"
    assert body["questions"]["dept"] == {
        "type": "choice",
        "instructions": "Which team",
        "criteria": {"billing": "b", "technical": "t", "sales": "s"},
    }


def test_extract_answers_maps_sections():
    q_choice = QuestionSpec(id="dept", type="choice", instructions="i", criteria={"billing": "b", "technical": "t"})
    q_score = QuestionSpec(id="frustration", type="score", instructions="i", criteria=["l0", "l1"])
    q_noul = QuestionSpec(id="is_urgent", type="noul", instructions="i")
    out = JevClient.extract_answers(RAW_RESPONSE, [q_choice, q_score, q_noul])
    assert out["dept"]["choice"] == "technical"
    assert out["dept"]["confidence"] == 0.85
    assert out["frustration"]["score"] == 1.2
    assert out["is_urgent"]["noul"] == 0.97
    assert "raw" in out["dept"]


def test_extract_answers_missing_question_is_skipped():
    q = QuestionSpec(id="not_there", type="noul", instructions="i")
    assert JevClient.extract_answers(RAW_RESPONSE, [q]) == {}


def test_error_raises_with_status():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "rate limited"})

    client = make_client(handler)
    with pytest.raises(JevAPIError) as ei:
        client.system_one("x", {})
    assert ei.value.status_code == 429


def test_dict_questions_accepted():
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["questions"]["n"] == {"type": "noul", "instructions": "urgent?"}
        return httpx.Response(200, json={"nouls": {"n": {"noul": 0.8}}})

    client = make_client(handler)
    out = client.system_one("state", {"n": {"type": "noul", "instructions": "urgent?"}})
    assert out["nouls"]["n"]["noul"] == 0.8


OLLAMA_RESPONSE = {
    "model": "nimble",
    "answers": {
        "period": {"type": "choice", "choice": "early_modern",
                   "probabilities": {"early_modern": 0.99, "modern": 0.01}, "confidence": 0.91},
        "detail": {"type": "score", "score": 0.78, "legend": {"0": "a", "1": "b"},
                   "probabilities": {"0": 0.3, "1": 0.7}, "confidence": 0.17},
        "open": {"type": "noul", "noul": 0.96},
    },
    "usage": {"input_tokens": 10, "output_tokens": 0},
}


def test_extract_answers_ollama_flat_layout():
    qs = [
        QuestionSpec(id="period", type="choice", instructions="p", criteria={"early_modern": "a", "modern": "b"}),
        QuestionSpec(id="detail", type="score", instructions="d", criteria=["a", "b"]),
        QuestionSpec(id="open", type="noul", instructions="o"),
    ]
    out = JevClient.extract_answers(OLLAMA_RESPONSE, qs)
    assert out["period"]["choice"] == "early_modern" and out["period"]["confidence"] == 0.91
    assert out["detail"]["score"] == 0.78
    assert out["open"]["noul"] == 0.96


def test_defaults_and_env(monkeypatch):
    monkeypatch.delenv("JEV_BROKER_BASE_URL", raising=False)
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:9999")
    monkeypatch.setenv("JEV_BROKER_MODEL", "other")
    c = JevClient()
    assert str(c._http.base_url).rstrip("/") == "http://127.0.0.1:9999"
    assert c.model == "other" and c.api_key is None


def test_state_object_serialized_and_no_auth_locally():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=OLLAMA_RESPONSE)

    c = JevClient(transport=httpx.MockTransport(handler))
    q = QuestionSpec(id="open", type="noul", instructions="o")
    c.system_one({"title": "x"}, {"open": q})
    assert seen["auth"] is None
    assert seen["body"]["state"] == '{"title": "x"}'


def test_box_rejected():
    with pytest.raises(ValueError, match="box"):
        JevClient.question_payload({"id": "b", "type": "box", "instructions": "where"})
