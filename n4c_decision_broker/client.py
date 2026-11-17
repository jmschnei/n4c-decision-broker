"""HTTP client for System One decision models served by Ollama (e.g. ``nimble``).

Endpoint: POST {base_url}/v1/systemone   (default http://localhost:11434, Ollama >= 0.35)
Request:  {"model": "nimble", "state": <string|object|array>,
           "questions": {id: {"type": "choice|score|noul",
                              "instructions": str,
                              "criteria": <object|array>}},
           "keep_alive": "5m"}
Response: {"model": ..., "answers": {id: {"type", <value>, "probabilities", "confidence"}},
           "usage": {...}}
Auth:     none locally; optional Bearer key (OLLAMA_API_KEY) for Ollama cloud models.

Note: the API is Jev-compatible, but ``box`` questions are not supported, and
``confidence`` is entropy-based (1 - H(p)/ln N), not the Jev formula.
"""
from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Mapping, Optional, Union

from n4c_decision_broker.models import QuestionSpec

__all__ = ["JevAPIError", "OllamaAPIError", "JevClient", "OllamaClient"]

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "nimble"
RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


def default_base_url() -> str:
    """N4C_BROKER_BASE_URL, then OLLAMA_HOST, then the local Ollama default."""
    url = os.environ.get("N4C_BROKER_BASE_URL") or os.environ.get("OLLAMA_HOST") or DEFAULT_BASE_URL
    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"
    return url.rstrip("/")


class OllamaAPIError(RuntimeError):
    """Non-2xx response from the Ollama System One API."""

    def __init__(self, status_code: int, message: str, body: Optional[Any] = None):
        super().__init__(f"Ollama API error {status_code}: {message}")
        self.status_code = status_code
        self.message = message
        self.body = body


class OllamaClient:
    """Thin client over Ollama's /v1/systemone endpoint.

    Accepts any httpx transport, which keeps it testable offline.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 120.0,
        keep_alive: Optional[Union[str, int]] = "5m",
        max_retries: int = 3,
        backoff_base: float = 0.5,
        backoff_max: float = 8.0,
        transport: Optional[Any] = None,
        sleep: Any = time.sleep,
    ):
        import httpx  # local import so the package loads even without httpx

        base_url = base_url or default_base_url()
        self.api_key = api_key or os.environ.get("OLLAMA_API_KEY")  # only for cloud models
        self.model = model or os.environ.get("N4C_BROKER_MODEL") or DEFAULT_MODEL
        self.keep_alive = keep_alive
        self.max_retries = max(0, max_retries)
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self._sleep = sleep
        self._httpx = httpx
        self.timeout = timeout
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=timeout,
            transport=transport,
        )

    # -- request building -------------------------------------------------
    @staticmethod
    def question_payload(question: Union[QuestionSpec, dict]) -> dict:
        if isinstance(question, QuestionSpec):
            q = question.model_dump(mode="json")
        else:
            q = question
        if q["type"] == "box":
            raise ValueError("Ollama System One models do not support 'box' questions")
        payload = {"type": q["type"], "instructions": q["instructions"]}
        if q.get("criteria") is not None:
            payload["criteria"] = q["criteria"]
        return payload

    @staticmethod
    def build_request(
        state: Union[str, dict, list],
        questions: Mapping[str, Union[QuestionSpec, dict]],
        model: str = DEFAULT_MODEL,
        keep_alive: Optional[Union[str, int]] = None,
    ) -> dict:
        # Ollama requires `state` to be a non-empty string; objects/arrays go in as JSON text.
        if not isinstance(state, str):
            state = json.dumps(state, ensure_ascii=False, default=str)
        body = {
            "model": model,
            "state": state,
            "questions": {qid: OllamaClient.question_payload(q) for qid, q in questions.items()},
        }
        if keep_alive is not None:
            body["keep_alive"] = keep_alive
        return body

    # -- response normalization -------------------------------------------
    _SECTIONS = {"choice": "choices", "score": "scores", "noul": "nouls", "box": "boxes"}

    @classmethod
    def extract_answers(cls, raw: Mapping, questions: list) -> dict:
        """Map the grouped Jev response into {question_id: {<type>: value,
        probabilities, confidence, raw}} for the requested questions."""
        out: dict[str, dict] = {}
        for q in questions:
            qtype = q.type.value if isinstance(q, QuestionSpec) else q["type"]
            qid = q.id if isinstance(q, QuestionSpec) else q["id"]
            entry = None
            section = cls._SECTIONS.get(qtype)
            if section:
                entry = (raw.get(section) or {}).get(qid)
            if entry is None:
                # Tolerate flat layouts: {qid: {...}} or {"answers": {qid: {...}}}
                entry = raw.get(qid) or (raw.get("answers") or {}).get(qid)
            if entry is None:
                continue
            norm: dict[str, Any] = {
                qtype: entry.get(qtype),
                "probabilities": entry.get("probabilities"),
                "confidence": entry.get("confidence"),
                "raw": dict(entry),
            }
            out[qid] = norm
        return out

    # -- calls --------------------------------------------------------------
    def system_one(
        self,
        state: Union[str, dict, list],
        questions: Mapping[str, Union[QuestionSpec, dict]],
    ) -> dict:
        """Evaluate all questions against the state in one call (parallel on the server)."""
        payload = self.build_request(state, questions, model=self.model, keep_alive=self.keep_alive)
        transient = (self._httpx.TransportError,)  # connect/read/timeout errors
        for attempt in range(self.max_retries + 1):
            last = attempt == self.max_retries
            try:
                resp = self._http.post("/v1/systemone", json=payload)
            except transient:
                if last:
                    raise
                self._backoff(attempt, None)
                continue
            if resp.status_code in RETRY_STATUS and not last:
                self._backoff(attempt, resp.headers.get("retry-after"))
                continue
            if resp.status_code >= 400:
                try:
                    body = resp.json()
                except Exception:
                    body = resp.text
                raise OllamaAPIError(resp.status_code, str(body), body=body)
            return resp.json()
        raise AssertionError("unreachable")  # pragma: no cover

    def _backoff(self, attempt: int, retry_after: Optional[str]) -> None:
        """Exponential backoff with jitter; honours a numeric Retry-After header."""
        delay = min(self.backoff_max, self.backoff_base * (2 ** attempt))
        delay *= random.uniform(0.5, 1.0)
        if retry_after:
            try:
                delay = max(delay, min(float(retry_after), self.backoff_max))
            except ValueError:
                pass
        self._sleep(delay)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "OllamaClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# Backwards-compatible names (the API is Jev-compatible).
JevClient = OllamaClient
JevAPIError = OllamaAPIError
