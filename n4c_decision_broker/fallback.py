"""Escalation backends for low-confidence decisions.

- HoldBackend (default): do nothing automatically; the broker marks the question
  as `escalated` and the application routes it to a human.
- LLMFallbackBackend: ask an OpenAI-compatible chat endpoint for a second
  opinion. The result is explicitly marked non-calibrated (`calibrated: False`,
  `source: "fallback_llm"`); the original Jev answer is preserved under `primary`.
"""
from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional, Union

from n4c_decision_broker.models import QuestionSpec

__all__ = ["EscalationBackend", "HoldBackend", "LLMFallbackBackend", "extract_json_object"]


class EscalationBackend(ABC):
    name = "abstract"

    @abstractmethod
    def escalate(
        self,
        *,
        question: QuestionSpec,
        state: Union[str, dict, list],
        original: Mapping[str, Any],
    ) -> Optional[dict]:
        """Return {"answer": <typed value>, "detail": str} or None if it cannot help."""


class HoldBackend(EscalationBackend):
    """No automated escalation — decision is held for a human."""

    name = "hold"

    def escalate(self, *, question, state, original) -> Optional[dict]:
        return None


def extract_json_object(text: str) -> dict:
    """Pull the first JSON object out of an LLM reply (tolerates fences/prose)."""
    if text is None:
        return {}
    candidates = re.findall(r"\{.*\}", text, flags=re.DOTALL)
    for cand in reversed(candidates):
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return {}


class LLMFallbackBackend(EscalationBackend):
    """OpenAI-compatible chat completions endpoint used as a non-calibrated fallback."""

    name = "fallback_llm"

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 60.0,
        transport: Optional[Any] = None,
    ):
        import httpx

        self.base_url = (base_url or os.environ.get("FALLBACK_LLM_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("FALLBACK_LLM_API_KEY")
        self.model = model or os.environ.get("FALLBACK_LLM_MODEL") or "gpt-4o-mini"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        self._http = httpx.Client(base_url=self.base_url, headers=headers, timeout=timeout, transport=transport)

    def build_prompt(self, question: QuestionSpec, state: Union[str, dict, list]) -> str:
        state_text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, default=str)
        criteria = json.dumps(question.criteria, ensure_ascii=False) if question.criteria else "none"
        if question.type.value == "choice":
            value_rule = '<value> must be exactly one of the option keys'
        elif question.type.value == "score":
            value_rule = '<value> must be an integer index (0-based) of the selected level'
        elif question.type.value == "noul":
            value_rule = '<value> must be "yes" or "no"'
        else:
            value_rule = '<value> must be an object with numeric top/bottom/left/right'
        return (
            "You are a decision engine. Read the STATE and answer the QUESTION.\n"
            "Respond ONLY with a single JSON object: {\"answer\": <value>}\n"
            f"Rule: {value_rule}\n\n"
            f"STATE:\n{state_text}\n\n"
            f"QUESTION: {question.instructions}\n"
            f"CRITERIA: {criteria}\n"
        )

    def _coerce_answer(self, question: QuestionSpec, raw_answer: Any) -> Any:
        qtype = question.type.value
        if qtype == "choice":
            keys = list(question.criteria.keys()) if isinstance(question.criteria, dict) else []
            value = str(raw_answer).strip()
            return value if value in keys else None
        if qtype == "score":
            try:
                idx = int(raw_answer)
            except (TypeError, ValueError):
                return None
            n = len(question.criteria) if isinstance(question.criteria, list) else 0
            return idx if 0 <= idx < max(n, 1) else None
        if qtype == "noul":
            label = str(raw_answer).strip().lower()
            return label if label in ("yes", "no") else None
        if isinstance(raw_answer, dict):
            return raw_answer
        return None

    def escalate(self, *, question: QuestionSpec, state, original: Mapping[str, Any]) -> Optional[dict]:
        prompt = self.build_prompt(question, state)
        body = {
            "model": self.model,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }
        resp = self._http.post("/chat/completions", json=body)
        if resp.status_code >= 400:
            return None
        data = resp.json()
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return None
        parsed = extract_json_object(text or "")
        answer = self._coerce_answer(question, parsed.get("answer"))
        if answer is None:
            return None
        return {"answer": answer, "detail": (text or "")[:500]}

    def close(self) -> None:
        self._http.close()
