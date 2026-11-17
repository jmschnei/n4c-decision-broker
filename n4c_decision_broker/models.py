"""Broker-level data model.

Question/answer semantics mirror TypeSafe's Jev primitives:
- choice: pick one option        -> answer: choice (option key) + probabilities + confidence
- score:  score on an ordered    -> answer: score (float position) + probabilities + confidence
         rubric
- noul:   is this statement      -> answer: noul (probability 0..1, no separate confidence)
         true?
- box:    where in the image     -> answer: box (top/bottom/left/right) + confidence
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Union

from pydantic import BaseModel, Field, model_validator

__all__ = [
    "QuestionType",
    "QuestionSpec",
    "QuestionPolicy",
    "DecisionStatus",
    "AnswerResult",
    "BrokerRequest",
    "BrokerResponse",
    "state_sha256",
    "stable_json",
]


def stable_json(value: Any) -> str:
    """Canonical JSON string used for hashing."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def state_sha256(state: Union[str, dict, list]) -> str:
    return hashlib.sha256(stable_json(state).encode("utf-8")).hexdigest()


class QuestionType(str, Enum):
    CHOICE = "choice"
    SCORE = "score"
    NOUL = "noul"
    BOX = "box"


class QuestionSpec(BaseModel):
    """One typed question about a state (mirrors a TypeSafe primitive)."""

    id: str = Field(..., min_length=1, description="Question id used to match the answer.")
    type: QuestionType
    instructions: str = Field(..., min_length=1)
    # choice: {option_key: description}
    # score:  [level_0_description, level_1_description, ...]
    # noul/box: None
    criteria: Union[dict[str, str], list[str], None] = None

    @model_validator(mode="after")
    def _check_criteria(self) -> "QuestionSpec":
        if self.type in (QuestionType.CHOICE, QuestionType.SCORE) and not self.criteria:
            raise ValueError(f"{self.type.value} question '{self.id}' requires criteria")
        if self.type is QuestionType.CHOICE and not isinstance(self.criteria, dict):
            raise ValueError(f"choice question '{self.id}' criteria must be an object option->description")
        if self.type is QuestionType.SCORE and not isinstance(self.criteria, list):
            raise ValueError(f"score question '{self.id}' criteria must be an array of level descriptions")
        return self

    def cache_key(self) -> str:
        return hashlib.sha256(stable_json(self.model_dump(mode="json")).encode("utf-8")).hexdigest()


class QuestionPolicy(BaseModel):
    """Confidence routing thresholds (per TypeSafe confidence guidance).

    - confidence >= auto_threshold        -> decided   (act automatically)
    - review_threshold <= confidence < auto_threshold -> review (flag for confirmation)
    - confidence < review_threshold       -> escalated (model is unsure; fallback or human)

    Noul answers carry no confidence: the value itself is the probability that the
    statement is true. Recommended bands (TypeSafe docs): >= noul_yes -> yes,
    <= noul_no -> no, in between -> review band.
    """

    auto_threshold: float = Field(0.85, ge=0.0, le=1.0)
    review_threshold: float = Field(0.5, ge=0.0, le=1.0)
    noul_yes: float = Field(0.7, ge=0.0, le=1.0)
    noul_no: float = Field(0.3, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _ordered(self) -> "QuestionPolicy":
        if self.review_threshold > self.auto_threshold:
            raise ValueError("review_threshold must be <= auto_threshold")
        if self.noul_no > self.noul_yes:
            raise ValueError("noul_no must be <= noul_yes")
        return self


class DecisionStatus(str, Enum):
    DECIDED = "decided"        # calibrated answer above the auto threshold
    REVIEW = "review"          # reasonable answer, below auto threshold — confirm before acting
    ESCALATED = "escalated"    # primary model unsure; fallback answer or held for a human
    ERROR = "error"            # provider failure


class AnswerResult(BaseModel):
    """Governed result for one question."""

    question_id: str
    status: DecisionStatus
    source: str = "jev"  # "jev" | "cache" | "fallback_llm" | "none"
    calibrated: bool = True

    # Typed values (fill in according to the question type)
    choice: Union[str, None] = None
    score: Union[float, int, None] = None
    noul: Union[float, None] = None
    noul_label: Union[str, None] = None  # "yes" | "no" | "unsure"
    box: Union[dict[str, float], None] = None
    probabilities: Union[dict[str, float], None] = None
    confidence: Union[float, None] = None  # for noul: |2p-1| equivalent scale

    question: Union[QuestionSpec, None] = None  # echo for self-contained rendering
    detail: Union[str, None] = None              # fallback LLM detail, etc.
    primary: Union[dict[str, Any], None] = None  # original Jev answer when a fallback was used
    error: Union[str, None] = None


class BrokerRequest(BaseModel):
    """A broker request: a state plus the questions to evaluate against it."""

    request_id: Union[str, None] = None
    state: Union[str, dict, list]
    questions: list[QuestionSpec] = Field(..., min_length=1)
    policy: QuestionPolicy = Field(default_factory=QuestionPolicy)

    @model_validator(mode="after")
    def _unique_ids(self) -> "BrokerRequest":
        ids = [q.id for q in self.questions]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate question ids: {sorted(dupes)}")
        return self


class BrokerResponse(BaseModel):
    request_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    state_sha256: str
    answers: list[AnswerResult]
    latency_ms: float
    usage: Union[dict[str, Any], None] = None
    cost_estimate_usd: Union[float, None] = None
    provider_calls: int = 0
