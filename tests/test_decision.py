"""Decision engine: confidence routing for choice / score / noul."""
from jev_broker.decision import choice_confidence, route_answer, score_confidence
from jev_broker.models import QuestionPolicy, QuestionSpec

POLICY = QuestionPolicy(auto_threshold=0.85, review_threshold=0.5)


def q_choice():
    return QuestionSpec(
        id="dept", type="choice", instructions="Which team",
        criteria={"a": "A", "b": "B", "c": "C"},
    )


def test_choice_confidence_formula():
    # 3 options: even split -> 0; all mass -> 1
    assert choice_confidence([1.0, 0.0, 0.0]) == 1.0
    assert abs(choice_confidence([1 / 3, 1 / 3, 1 / 3]) - 0.0) < 1e-9
    # docs example: (0.6, 0.3, 0.1) -> 0.4
    assert abs(choice_confidence([0.6, 0.3, 0.1]) - 0.4) < 1e-9


def test_score_confidence_formula():
    assert score_confidence([1.0, 0.0, 0.0]) == 1.0
    n3 = [1 / 3, 1 / 3, 1 / 3]
    assert abs(score_confidence(n3) - 0.0) < 1e-9


def test_choice_routing_tiers():
    answer = {"choice": "a", "probabilities": {"a": 0.9, "b": 0.05, "c": 0.05}, "confidence": 0.85}
    assert route_answer(q_choice(), answer, POLICY).status == "decided"

    answer = {"choice": "a", "probabilities": {"a": 0.6, "b": 0.25, "c": 0.15}, "confidence": 0.55}
    assert route_answer(q_choice(), answer, POLICY).status == "review"

    answer = {"choice": "a", "probabilities": {"a": 0.4, "b": 0.3, "c": 0.3}, "confidence": 0.2}
    assert route_answer(q_choice(), answer, POLICY).status == "escalated"


def test_choice_missing_confidence_is_derived():
    # no confidence field -> derived from probabilities:
    # (0.93 - 1/3) / (1 - 1/3) = 0.895  -> decided
    answer = {"choice": "a", "probabilities": {"a": 0.93, "b": 0.035, "c": 0.035}}
    r = route_answer(q_choice(), answer, POLICY)
    assert r.status == "decided"
    assert abs(r.confidence_equiv - 0.895) < 1e-9

    # (0.7 - 1/3) / (2/3) = 0.5 -> review
    answer = {"choice": "a", "probabilities": {"a": 0.7, "b": 0.15, "c": 0.15}}
    assert route_answer(q_choice(), answer, POLICY).status == "review"


def test_noul_routing_bands():
    q = QuestionSpec(id="urgent", type="noul", instructions="Urgent?")
    assert route_answer(q, {"noul": 0.9}, POLICY).noul_label == "yes"
    assert route_answer(q, {"noul": 0.9}, POLICY).status == "decided"
    assert route_answer(q, {"noul": 0.1}, POLICY).noul_label == "no"
    assert route_answer(q, {"noul": 0.1}, POLICY).status == "decided"
    # middle band
    r = route_answer(q, {"noul": 0.55}, POLICY)
    assert r.status in ("review", "escalated")
    assert r.noul_label == "unsure"
    # 0.5 is maximally uninformative -> escalated
    assert route_answer(q, {"noul": 0.5}, POLICY).status == "escalated"


def test_noul_confidence_equivalent_scale():
    r = route_answer(QuestionSpec(id="u", type="noul", instructions="x"), {"noul": 0.9}, POLICY)
    assert abs(r.confidence_equiv - 0.8) < 1e-9  # |2*0.9 - 1|


def test_policy_validation():
    import pytest
    with pytest.raises(ValueError):
        QuestionPolicy(auto_threshold=0.5, review_threshold=0.8)


def test_question_spec_validation():
    import pytest
    with pytest.raises(ValueError):
        QuestionSpec(id="x", type="choice", instructions="i")  # missing criteria
    with pytest.raises(ValueError):
        QuestionSpec(id="x", type="choice", instructions="i", criteria=["a", "b"])
    with pytest.raises(ValueError):
        QuestionSpec(id="x", type="score", instructions="i", criteria={"a": 1})
