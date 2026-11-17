"""Confidence routing — the broker's decision engine.

Implements TypeSafe's three-path confidence pattern:
high confidence -> act; medium -> proceed with caution (review);
low -> do not act (escalate).

Formulas (docs.typesafe.ai/confidence):
- choice (n options): (p_max - 1/n) / (1 - 1/n)
- score  (n levels):  1 - weighted_distance_to_most_likely / MAD_uniform, floored at 0
- noul:              the value itself is the probability; equivalent confidence |2p - 1|
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from jev_broker.models import QuestionPolicy, QuestionSpec

__all__ = ["Routing", "choice_confidence", "score_confidence", "route_answer"]


@dataclass(frozen=True)
class Routing:
    """Outcome of routing one answer against a policy."""

    status: str  # DecisionStatus value: decided | review | escalated
    confidence_equiv: Optional[float] = None
    noul_label: Optional[str] = None  # yes | no | unsure


def choice_confidence(probabilities: list[float]) -> float:
    n = len(probabilities)
    if n <= 1:
        return 1.0
    p_max = max(probabilities)
    return (p_max - 1.0 / n) / (1.0 - 1.0 / n)


def score_confidence(probabilities: list[float]) -> float:
    n = len(probabilities)
    if n <= 1:
        return 1.0
    m = max(range(n), key=lambda i: probabilities[i])
    mad_unif = sum(abs(i - (n - 1) / 2.0) for i in range(n)) / n
    if mad_unif == 0:
        return 0.0
    distance = sum(probabilities[i] * abs(i - m) for i in range(n))
    return max(0.0, 1.0 - distance / mad_unif)


def _tier(confidence: float, policy: QuestionPolicy) -> str:
    if confidence >= policy.auto_threshold:
        return "decided"
    if confidence >= policy.review_threshold:
        return "review"
    return "escalated"


def _route_distribution(
    probabilities: dict[str, float],
    confidence: Optional[float],
    question: QuestionSpec,
    policy: QuestionPolicy,
) -> Routing:
    values = list(probabilities.values()) if probabilities else []
    if not values:
        return Routing(status="escalated", confidence_equiv=0.0)
    if confidence is None:
        if question.type.value == "score":
            confidence = score_confidence(values)
        else:
            confidence = choice_confidence(values)
    return Routing(status=_tier(confidence, policy), confidence_equiv=confidence)


def _route_noul(p: float, policy: QuestionPolicy) -> Routing:
    confidence_equiv = abs(2.0 * p - 1.0)
    if p >= policy.noul_yes:
        return Routing(status="decided", confidence_equiv=confidence_equiv, noul_label="yes")
    if p <= policy.noul_no:
        return Routing(status="decided", confidence_equiv=confidence_equiv, noul_label="no")
    # Middle band: closer to 0.5 is less informative -> escalate; otherwise review.
    midpoint = (policy.noul_yes + policy.noul_no) / 2.0
    if abs(p - midpoint) < (policy.noul_yes - policy.noul_no) / 4.0:
        return Routing(status="escalated", confidence_equiv=confidence_equiv, noul_label="unsure")
    return Routing(status="review", confidence_equiv=confidence_equiv, noul_label="unsure")


def route_answer(
    question: QuestionSpec,
    answer: dict,
    policy: QuestionPolicy,
) -> Routing:
    """Route one Jev answer (dict with typed value, probabilities, confidence).

    `answer` is the normalized per-question payload produced by the broker:
    keys are the question-type value (choice/score/noul/box), `probabilities`,
    `confidence`.
    """
    qtype = question.type.value
    if qtype == "noul":
        p = float(answer.get("noul"))
        return _route_noul(p, policy)
    probabilities = answer.get("probabilities") or {}
    confidence = answer.get("confidence")
    if qtype == "box":
        # Box has no ordinal rubric; treat reported confidence as-is.
        if confidence is None:
            return Routing(status="review", confidence_equiv=None)
        return Routing(status=_tier(confidence, policy), confidence_equiv=confidence)
    return _route_distribution(probabilities, confidence, question, policy)
