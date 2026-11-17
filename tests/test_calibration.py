"""Threshold calibration on synthetic observations (offline)."""
from n4c_decision_broker.calibration import (
    Observation, calibrate, collect, expected_calibration_error,
)
from n4c_decision_broker.models import QuestionSpec
from n4c_decision_broker.testing import FakeJev


def _obs(pairs, qid="q", qtype="choice"):
    return [Observation(qid, qtype, c, ok) for c, ok in pairs]


def test_thresholds_follow_measured_accuracy():
    # >=0.9 always right; 0.6-0.8 mostly right; <0.5 coin flip
    pairs = [(0.95, True)] * 10 + [(0.9, True)] * 10 + [(0.7, True)] * 8 + [(0.7, False)] * 2 \
            + [(0.3, True)] * 3 + [(0.3, False)] * 7
    rep = calibrate(_obs(pairs), target_auto=0.95, target_review=0.80, min_support=5)
    pol = rep["recommended_policy"]
    assert pol["auto_threshold"] == 0.9
    assert pol["review_threshold"] == 0.7
    tiers = rep["choice_score"]["tiers_recommended_policy"]
    assert tiers["decided"]["accuracy"] == 1.0
    assert tiers["escalated"]["n"] == 10


def test_unreachable_target_keeps_default_and_notes_it():
    pairs = [(0.9, False)] * 10
    rep = calibrate(_obs(pairs))
    assert rep["recommended_policy"]["auto_threshold"] == 0.85
    assert any("auto_threshold kept" in n for n in rep["notes"])


def test_min_support_blocks_tiny_cutoffs():
    pairs = [(0.99, True)] * 2 + [(0.6, False)] * 20
    rep = calibrate(_obs(pairs), min_support=5)
    assert rep["recommended_policy"]["auto_threshold"] == 0.85


def test_noul_bands():
    nouls = [Observation("n", "noul", p, None, True) for p in (0.95, 0.9, 0.85, 0.8, 0.75)] \
          + [Observation("n", "noul", p, None, False) for p in (0.05, 0.1, 0.15, 0.2, 0.25)] \
          + [Observation("n", "noul", 0.5, None, True), Observation("n", "noul", 0.55, None, False)]
    rep = calibrate(nouls, min_support=5)
    pol = rep["recommended_policy"]
    assert pol["noul_yes"] == 0.75 and pol["noul_no"] == 0.25
    assert rep["noul"]["tiers_recommended_policy"]["yes"]["accuracy"] == 1.0


def test_ece_perfect_and_overconfident():
    assert expected_calibration_error([(0.95, True)] * 19 + [(0.95, False)]) == 0.0
    assert expected_calibration_error([(0.95, False)] * 10) > 0.9
    assert expected_calibration_error([]) is None


def test_collect_with_fake_model():
    qs = [QuestionSpec(id="lang", type="choice", instructions="l", criteria={"de": "a", "en": "b"}),
          QuestionSpec(id="ok", type="noul", instructions="o")]
    recs = [{"state": f"r{i}", "labels": {"lang": "de", "ok": True}} for i in range(6)]
    obs = collect(FakeJev(seed="c"), recs, qs)
    assert len(obs) == 12
    assert {o.qtype for o in obs} == {"choice", "noul"}


def test_threshold_stops_at_first_drop_not_lucky_tail():
    # precision collapses in the middle but "recovers" with many lucky answers in the tail
    pairs = [(0.9, True)] * 6 + [(0.6, False)] * 6 + [(0.1, True)] * 200
    rep = calibrate(_obs(pairs), target_auto=0.95, min_support=5)
    assert rep["recommended_policy"]["auto_threshold"] == 0.9


def test_observations_roundtrip(tmp_path):
    from n4c_decision_broker.calibration import load_observations, save_observations
    obs = _obs([(0.9, True), (0.2, False)])
    save_observations(obs, tmp_path / "o.json")
    assert load_observations(tmp_path / "o.json") == obs


def test_noul_thresholds_clamped_around_half():
    # sharp model: every "no" at p~0, every "yes" at p>=0.2 -> raw optimum would be yes=0.2
    nouls = [Observation("n", "noul", p, None, True) for p in (0.2, 0.3, 0.6, 0.9, 0.95, 0.99)] \
          + [Observation("n", "noul", 0.001, None, False) for _ in range(6)]
    pol = calibrate(nouls, min_support=5)["recommended_policy"]
    assert pol["noul_yes"] >= 0.5 and pol["noul_no"] <= 0.5
