"""Threshold calibration against labeled records.

Runs the model on records with known answers, then picks the routing thresholds
of :class:`QuestionPolicy` so that the *measured* accuracy of each tier meets a
target:

- ``auto_threshold``   smallest confidence whose answers at/above it are
                       >= ``target_auto`` accurate (these are acted on automatically)
- ``review_threshold`` smallest confidence whose answers at/above it are
                       >= ``target_review`` accurate (below it -> escalated)
- ``noul_yes/no``      smallest p for "yes" / largest p for "no" with precision
                       >= ``target_auto`` (in between -> review/escalated)

If no threshold reaches a target the default is kept and the report says so.
Small label sets give noisy thresholds: ``min_support`` guards against picking a
cut-off backed by only a handful of examples.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from n4c_decision_broker.client import JevClient
from n4c_decision_broker.decision import choice_confidence, score_confidence
from n4c_decision_broker.models import QuestionPolicy, QuestionSpec

__all__ = ["Observation", "collect", "save_observations", "load_observations", "calibrate", "load_labeled", "expected_calibration_error"]


@dataclass
class Observation:
    question_id: str
    qtype: str
    confidence: float        # choice/score: model confidence; noul: p(true)
    correct: Optional[bool]  # choice/score: argmax == label (None for noul)
    truth: Any = None        # noul: the boolean label


def load_labeled(path: Union[str, Path]) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("records", [])
    for i, rec in enumerate(data):
        if "state" not in rec or "labels" not in rec:
            raise ValueError(f"record {i} needs 'state' and 'labels'")
    return data


def _truth_bool(v: Any) -> bool:
    return v if isinstance(v, bool) else str(v).strip().lower() in ("yes", "true", "1")


def save_observations(obs: Sequence[Observation], path: Union[str, Path]) -> None:
    """Persist raw observations so thresholds can be recomputed offline (no model calls)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps([o.__dict__ for o in obs], ensure_ascii=False), encoding="utf-8")


def load_observations(path: Union[str, Path]) -> list[Observation]:
    return [Observation(**o) for o in json.loads(Path(path).read_text(encoding="utf-8"))]


def collect(
    client: Any,
    records: Sequence[dict],
    questions: Sequence[QuestionSpec],
    progress: Any = None,
) -> list[Observation]:
    """Ask the model every question for every labeled record (one call per record)."""
    payload = {q.id: JevClient.question_payload(q) for q in questions}
    obs: list[Observation] = []
    for i, rec in enumerate(records):
        raw = client.system_one(rec["state"], payload)
        answers = JevClient.extract_answers(raw, list(questions))
        for q in questions:
            label = rec["labels"].get(q.id)
            a = answers.get(q.id)
            if label is None or a is None:
                continue
            t = q.type.value
            if t == "noul":
                obs.append(Observation(q.id, t, float(a["noul"]), None, _truth_bool(label)))
                continue
            probs = a.get("probabilities") or {}
            if not probs:
                continue
            pred = max(probs, key=probs.get)
            if t == "score":  # labels are level indices; probability keys are "0","1",...
                label = str(label)
            conf = a.get("confidence")
            if conf is None:
                vals = list(probs.values())
                conf = score_confidence(vals) if t == "score" else choice_confidence(vals)
            obs.append(Observation(q.id, t, float(conf), pred == label))
        if progress:
            progress(i + 1, len(records))
    return obs


def expected_calibration_error(pairs: Sequence[tuple[float, bool]], bins: int = 10) -> Optional[float]:
    """ECE of (confidence, correct) pairs, equal-width bins."""
    if not pairs:
        return None
    total = len(pairs)
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [(c, ok) for c, ok in pairs if lo <= c < hi or (b == bins - 1 and c == 1.0)]
        if sel:
            ece += len(sel) / total * abs(sum(c for c, _ in sel) / len(sel) - sum(ok for _, ok in sel) / len(sel))
    return round(ece, 4)


def _lowest_threshold(pairs: Sequence[tuple[float, bool]], target: float, min_support: int) -> Optional[float]:
    """Smallest confidence t such that answers with conf >= t have accuracy >= target.

    Scans from the most confident answer downwards and stops at the first level
    (with enough support) where the cumulative accuracy drops below the target.
    This is deliberately conservative: a few lucky answers deep in the low-confidence
    tail must not drag the threshold down.
    """
    best = None
    for t in sorted({c for c, _ in pairs}, reverse=True):
        sel = [ok for c, ok in pairs if c >= t]
        if len(sel) < min_support:
            continue
        if sum(sel) / len(sel) >= target:
            best = t
        else:
            break
    return best


def _tier_stats(pairs: Sequence[tuple[float, bool]], policy: QuestionPolicy) -> dict:
    tiers: dict[str, list[bool]] = {"decided": [], "review": [], "escalated": []}
    for c, ok in pairs:
        key = "decided" if c >= policy.auto_threshold else "review" if c >= policy.review_threshold else "escalated"
        tiers[key].append(ok)
    n = max(1, len(pairs))
    return {
        k: {"n": len(v), "share": round(len(v) / n, 3),
            "accuracy": round(sum(v) / len(v), 3) if v else None}
        for k, v in tiers.items()
    }


def calibrate(
    observations: Sequence[Observation],
    target_auto: float = 0.95,
    target_review: float = 0.80,
    min_support: int = 5,
    base: Optional[QuestionPolicy] = None,
) -> dict:
    """Return a report with the recommended policy and measured per-tier accuracy."""
    base = base or QuestionPolicy()
    notes: list[str] = []
    dist = [(o.confidence, bool(o.correct)) for o in observations if o.qtype != "noul"]
    nouls = [o for o in observations if o.qtype == "noul"]

    auto = base.auto_threshold
    review = base.review_threshold
    if dist:
        a = _lowest_threshold(dist, target_auto, min_support)
        r = _lowest_threshold(dist, target_review, min_support)
        if a is None:
            notes.append(f"no confidence level reached {target_auto:.0%} accuracy "
                         f"(min support {min_support}); auto_threshold kept at default {auto}")
        else:
            auto = round(a, 4)
        if r is None:
            notes.append(f"no confidence level reached {target_review:.0%} accuracy; "
                         f"review_threshold kept at default {review}")
        else:
            review = round(r, 4)
        review = min(review, auto)
    else:
        notes.append("no choice/score observations; thresholds unchanged")

    yes_t, no_t = base.noul_yes, base.noul_no
    if nouls:
        pos = [(o.confidence, bool(o.truth)) for o in nouls]
        neg = [(1.0 - o.confidence, not o.truth) for o in nouls]  # symmetric: "no" confidence = 1-p
        y = _lowest_threshold(pos, target_auto, min_support)
        n = _lowest_threshold(neg, target_auto, min_support)
        if y is None:
            notes.append("noul 'yes' band never reached target precision; noul_yes kept")
        else:
            yes_t = round(y, 4)
        if n is None:
            notes.append("noul 'no' band never reached target precision; noul_no kept")
        else:
            no_t = round(1.0 - n, 4)
        # "yes" must mean p > 0.5 and "no" p < 0.5, however sharp the model's tail is;
        # otherwise a small label set can yield an empty review band.
        yes_t = max(yes_t, 0.5)
        no_t = min(no_t, 0.5)

    policy = QuestionPolicy(auto_threshold=auto, review_threshold=min(review, auto),
                            noul_yes=yes_t, noul_no=no_t)
    report: dict[str, Any] = {
        "n_observations": len(observations),
        "targets": {"auto": target_auto, "review": target_review, "min_support": min_support},
        "recommended_policy": policy.model_dump(mode="json"),
        "choice_score": {
            "n": len(dist),
            "accuracy": round(sum(ok for _, ok in dist) / len(dist), 3) if dist else None,
            "ece": expected_calibration_error(dist),
            "tiers_default_policy": _tier_stats(dist, base) if dist else None,
            "tiers_recommended_policy": _tier_stats(dist, policy) if dist else None,
        },
        "per_question": {},
        "notes": notes,
    }
    for qid in sorted({o.question_id for o in observations}):
        qs = [o for o in observations if o.question_id == qid]
        if qs[0].qtype == "noul":
            correct = [(o.confidence >= 0.5) == bool(o.truth) for o in qs]
            report["per_question"][qid] = {"type": "noul", "n": len(qs),
                                           "accuracy@0.5": round(sum(correct) / len(correct), 3)}
        else:
            c = [(o.confidence, bool(o.correct)) for o in qs]
            qa = _lowest_threshold(c, target_auto, min_support)
            qr = _lowest_threshold(c, target_review, min_support)
            report["per_question"][qid] = {"type": qs[0].qtype, "n": len(qs),
                                           "accuracy": round(sum(ok for _, ok in c) / len(c), 3),
                                           "ece": expected_calibration_error(c),
                                           "auto_threshold_if_alone": None if qa is None else round(qa, 4),
                                           "review_threshold_if_alone": None if qr is None else round(qr, 4)}
    if nouls:
        def tier(p: float, pol: QuestionPolicy) -> str:
            return "yes" if p >= pol.noul_yes else "no" if p <= pol.noul_no else "between"
        for name, pol in (("default", base), ("recommended", policy)):
            stats: dict[str, Any] = {}
            for lab in ("yes", "no", "between"):
                sel = [o for o in nouls if tier(o.confidence, pol) == lab]
                ok = [bool(o.truth) == (lab == "yes") for o in sel]
                stats[lab] = {"n": len(sel),
                              "accuracy": round(sum(ok) / len(ok), 3) if ok and lab != "between" else None}
            report.setdefault("noul", {})[f"tiers_{name}_policy"] = stats
    return report
