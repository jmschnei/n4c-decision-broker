"""ModelBroker: orchestrates Jev calls, confidence routing, fallback, cache, audit."""
from __future__ import annotations

import time
import uuid
from typing import Any, Optional, Sequence, Union

from n4c_decision_broker.audit import AuditLog
from n4c_decision_broker.cache import DecisionCache, cache_key
from n4c_decision_broker.client import JevClient
from n4c_decision_broker.decision import route_answer
from n4c_decision_broker.fallback import EscalationBackend, HoldBackend
from n4c_decision_broker.models import (
    AnswerResult,
    BrokerResponse,
    QuestionPolicy,
    QuestionSpec,
    QuestionType,
    state_sha256,
)

__all__ = ["ModelBroker"]

# Local Ollama inference has no per-token cost; set >0 to track cloud/amortized cost.
DEFAULT_COST_USD_PER_BTOK = 0.0
TOKENS_PER_CHAR = 0.25  # rough heuristic for cost estimation only


class ModelBroker:
    """Confidence-governed decision broker over the Jev model.

    Typical construction:

        broker = ModelBroker(
            client=JevClient(),
            fallback=HoldBackend(),          # or LLMFallbackBackend(...)
            cache=DecisionCache(path="audit/cache.json"),
            audit=AuditLog("audit/decisions.jsonl"),
        )
    """

    def __init__(
        self,
        client: Optional[JevClient] = None,
        fallback: Optional[EscalationBackend] = None,
        cache: Optional[DecisionCache] = None,
        audit: Optional[AuditLog] = None,
        policy: Optional[QuestionPolicy] = None,
        cost_usd_per_btok: float = DEFAULT_COST_USD_PER_BTOK,
    ):
        self.client = client
        self.fallback = fallback or HoldBackend()
        self.cache = cache
        self.audit = audit
        self.policy = policy or QuestionPolicy()
        self.cost_usd_per_btok = cost_usd_per_btok
        self.provider_calls = 0
        self.provider_errors = 0

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _as_question_list(
        questions: Union[Sequence[QuestionSpec], dict, list],
    ) -> list[QuestionSpec]:
        if isinstance(questions, dict):
            questions = list(questions.values())
        out = [q if isinstance(q, QuestionSpec) else QuestionSpec.model_validate(q) for q in questions]
        ids = [q.id for q in out]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate question ids: {sorted(dupes)}")
        if not out:
            raise ValueError("at least one question is required")
        return out

    def _estimate_cost(self, state: Union[str, dict, list], questions: list[QuestionSpec]) -> float:
        if self.client is None:
            return 0.0
        chars = len(state) if isinstance(state, str) else len(str(state))
        for q in questions:
            chars += len(q.instructions) + 64
        tokens = max(256.0, chars * TOKENS_PER_CHAR)
        return round(tokens / 1e9 * self.cost_usd_per_btok, 9)

    # -- core ---------------------------------------------------------------
    def decide(
        self,
        state: Union[str, dict, list],
        questions: Union[Sequence[QuestionSpec], dict, list],
        policy: Optional[QuestionPolicy] = None,
        request_id: Optional[str] = None,
        use_cache: bool = True,
    ) -> BrokerResponse:
        """Evaluate all questions against the state and return governed answers."""
        policy = policy or self.policy
        question_list = self._as_question_list(questions)
        request_id = request_id or uuid.uuid4().hex
        started = time.perf_counter()
        state_hash = state_sha256(state)

        if self.client is None:
            raise RuntimeError(
                "ModelBroker has no client. Pass a JevClient (or a FakeJev for offline use)."
            )

        results: dict[str, AnswerResult] = {}
        pending: list[QuestionSpec] = []

        # 1) Serve whatever the cache already knows (decision = f(state, question, policy)).
        for q in question_list:
            if use_cache and self.cache is not None:
                key = cache_key(state_hash, q, policy)
                hit = self.cache.get(key)
                if hit is not None:
                    results[q.id] = hit.model_copy(update={"source": "cache"})
                    continue
            pending.append(q)

        # 2) One Jev call evaluates every pending question in parallel.
        raw: Optional[dict] = None
        answers_raw: dict[str, dict] = {}
        if pending:
            self.provider_calls += 1
            try:
                raw = self.client.system_one(
                    state, {q.id: JevClient.question_payload(q) for q in pending}
                )
                answers_raw = JevClient.extract_answers(raw or {}, pending)
            except Exception as exc:  # noqa: BLE001 - surfaced per-question below
                self.provider_errors += 1
                for q in pending:
                    results[q.id] = AnswerResult(
                        question_id=q.id,
                        status="error",
                        source="jev",
                        calibrated=False,
                        question=q,
                        error=str(exc),
                    )
            else:
                for q in pending:
                    a = answers_raw.get(q.id)
                    if a is None:
                        results[q.id] = AnswerResult(
                            question_id=q.id,
                            status="error",
                            source="jev",
                            calibrated=False,
                            question=q,
                            error=f"no answer for question '{q.id}' in Jev response",
                        )
                        continue
                    routing = route_answer(q, a, policy)
                    ar = AnswerResult(
                        question_id=q.id,
                        status=routing.status,
                        source="jev",
                        calibrated=True,
                        choice=a.get("choice"),
                        score=a.get("score"),
                        noul=a.get("noul"),
                        noul_label=routing.noul_label,
                        box=a.get("box"),
                        probabilities=a.get("probabilities"),
                        confidence=routing.confidence_equiv,
                        question=q,
                    )
                    # 3) Escalation path: fallback LLM or hold for a human.
                    if routing.status == "escalated" and self.fallback.name != "hold":
                        try:
                            fb = self.fallback.escalate(question=q, state=state, original=a)
                        except Exception:  # noqa: BLE001 - fallback must never break the flow
                            fb = None
                        if fb and fb.get("answer") is not None:
                            ar = self._apply_fallback(ar, q, fb)
                    results[q.id] = ar

        # 4) Restore input order, cache governed (non-error) decisions, audit.
        ordered = [results[q.id] for q in question_list if q.id in results]
        if self.cache is not None and use_cache:
            for q in question_list:
                ar = results.get(q.id)
                if ar is not None and ar.status != "error":
                    self.cache.put(cache_key(state_hash, q, policy), ar)

        latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
        response = BrokerResponse(
            request_id=request_id,
            state_sha256=state_hash,
            answers=ordered,
            latency_ms=latency_ms,
            usage=(raw or {}).get("usage") if isinstance(raw, dict) else None,
            cost_estimate_usd=self._estimate_cost(state, pending) if pending else 0.0,
            provider_calls=1 if pending else 0,
        )

        if self.audit is not None:
            self.audit.record(
                {
                    "request_id": request_id,
                    "state_sha256": state_hash,
                    "policy": policy.model_dump(mode="json"),
                    "latency_ms": latency_ms,
                    "provider_calls": response.provider_calls,
                    "answers": [
                        {
                            "id": ar.question_id,
                            "status": ar.status.value,
                            "source": ar.source,
                            "calibrated": ar.calibrated,
                            "choice": ar.choice,
                            "score": ar.score,
                            "noul": ar.noul,
                            "noul_label": ar.noul_label,
                            "confidence": ar.confidence,
                            "error": ar.error,
                        }
                        for ar in ordered
                    ],
                }
            )
        return response

    @staticmethod
    def _apply_fallback(ar: AnswerResult, q: QuestionSpec, fb: dict) -> AnswerResult:
        value = fb.get("answer")
        primary = ar.model_dump(mode="json")
        primary.pop("primary", None)
        if q.type is QuestionType.CHOICE:
            ar.choice = str(value)
        elif q.type is QuestionType.SCORE:
            ar.score = int(value)
        elif q.type is QuestionType.NOUL:
            ar.noul_label = "yes" if str(value).lower() == "yes" else "no"
        elif q.type is QuestionType.BOX and isinstance(value, dict):
            ar.box = {k: float(v) for k, v in value.items() if isinstance(v, (int, float))}
        ar.status = "escalated"
        ar.source = "fallback_llm"
        ar.calibrated = False
        ar.detail = fb.get("detail")
        ar.primary = primary
        return ar

    def batch(
        self,
        records: Sequence[Union[str, dict, list]],
        questions: Union[Sequence[QuestionSpec], dict, list],
        policy: Optional[QuestionPolicy] = None,
        use_cache: bool = True,
    ) -> list[BrokerResponse]:
        """Evaluate a collection of records (e.g. a museum catalog export)."""
        question_list = self._as_question_list(questions)
        return [
            self.decide(record, question_list, policy=policy, use_cache=use_cache)
            for record in records
        ]

    def metrics(self) -> dict:
        return {
            "provider_calls": self.provider_calls,
            "provider_errors": self.provider_errors,
            "fallback": self.fallback.name,
            "policy": self.policy.model_dump(mode="json"),
            "cache": self.cache.stats() if self.cache is not None else None,
        }

    def close(self) -> None:
        if self.client is not None and hasattr(self.client, "close"):
            self.client.close()
        if hasattr(self.fallback, "close"):
            self.fallback.close()
