"""Offline stand-ins for demos and tests (no network, deterministic)."""
from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Union

from n4c_decision_broker.decision import choice_confidence, score_confidence
from n4c_decision_broker.models import QuestionSpec

__all__ = ["FakeJev", "demo_questions", "demo_records"]


class FakeJev:
    """Deterministic JevClient lookalike.

    Distribution per (seed, question, state): ~35% confident, ~40% moderate,
    ~25% near-uniform — so demos and tests exercise all three routing tiers.
    Noul values are drawn uniformly (covers yes/no/review bands).
    """

    def __init__(self, seed: str = "n4c-decision-broker"):
        self.seed = seed
        self.last_request: Union[tuple, None] = None
        self.fail_next: bool = False

    def close(self) -> None:  # JevClient lookalike
        pass

    def _state_text(self, state: Union[str, dict, list]) -> str:
        return state if isinstance(state, str) else json.dumps(state, sort_keys=True, default=str)

    def _distribution(self, rng: random.Random, n: int) -> list[float]:
        kind = rng.random()
        if kind < 0.35:
            top = rng.uniform(0.72, 0.95)
        elif kind < 0.75:
            top = rng.uniform(0.52, 0.68)
        else:
            top = rng.uniform(1.0 / n + 0.01, 1.0 / n + 0.18)
        rest = [rng.random() for _ in range(n - 1)]
        s = sum(rest) or 1.0
        p = [top] + [(1.0 - top) * r / s for r in rest]
        idx = rng.randrange(n)
        p[idx], p[0] = p[0], p[idx]
        return [round(v, 4) for v in p]

    def system_one(self, state: Union[str, dict, list], questions: dict[str, dict]) -> dict:
        self.last_request = (state, dict(questions))
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("FakeJev: simulated provider outage")
        out: dict[str, dict] = {}
        state_text = self._state_text(state)
        for qid, q in questions.items():
            key = f"{self.seed}|{qid}|{q.get('type')}|{state_text}"
            rng = random.Random(hashlib.sha256(key.encode("utf-8")).hexdigest())
            t = q.get("type")
            criteria = q.get("criteria")
            if t == "choice":
                opts = list(criteria.keys()) if isinstance(criteria, dict) else list(criteria)
                p = self._distribution(rng, len(opts))
                probs = dict(zip(opts, p))
                top = max(probs, key=probs.get)
                out.setdefault("choices", {})[qid] = {
                    "choice": top,
                    "probabilities": probs,
                    "confidence": round(choice_confidence(p), 4),
                }
            elif t == "score":
                n = len(criteria) if isinstance(criteria, list) else 3
                p = self._distribution(rng, n)
                probs = {str(i): v for i, v in enumerate(p)}
                out.setdefault("scores", {})[qid] = {
                    "score": round(sum(i * v for i, v in enumerate(p)), 4),
                    "probabilities": probs,
                    "confidence": round(score_confidence(p), 4),
                }
            elif t == "noul":
                out.setdefault("nouls", {})[qid] = {"noul": round(rng.random(), 4)}
            elif t == "box":
                top = rng.random() * 0.6
                left = rng.random() * 0.6
                out.setdefault("boxes", {})[qid] = {
                    "box": {"top": round(top, 3), "bottom": round(top + 0.25, 3),
                            "left": round(left, 3), "right": round(left + 0.3, 3)},
                    "confidence": round(rng.random(), 4),
                }
        return out


def demo_questions() -> list[QuestionSpec]:
    """A realistic NFDI4Culture flavor: enriching museum catalog records."""
    return [
        QuestionSpec(
            id="period",
            type="choice",
            instructions="Historical period of the object described in the record",
            criteria={
                "antique": "Classical antiquity (before 500 CE)",
                "medieval": "Medieval (500-1400)",
                "early_modern": "Early modern (1400-1800)",
                "modern": "Modern (1800-1945)",
                "contemporary": "Contemporary (1945-present)",
            },
        ),
        QuestionSpec(
            id="language",
            type="choice",
            instructions="Primary language of the record text",
            criteria={"de": "German", "en": "English", "fr": "French", "la": "Latin", "other": "Anything else"},
        ),
        QuestionSpec(
            id="open_access",
            type="noul",
            instructions="The record contains no personal data, no rights issues, and no restrictions on reuse",
        ),
    ]


def demo_records() -> list[str]:
    return [
        "Kupferstich, 'Ansicht von Rom mit dem Petersdom', 17. Jahrhundert, Sammlung DFKI",
        "Photographie, 'Bauhaus Dessau, Nordflügel', 1927, Nachlass Gropius",
        "Briefwechsel, lateinisch, 'epistulae ad Ciceronem', 13. Jahrhundert",
        "Digitalisierung, 'Scan des Handschriftencodex 42, Bl. 12r', 2024, DFKI",
        "Musikstück, 'Sonate für Klavier No. 8 in Es-Dur', komponiert 1952",
        "Katalogeintrag ohne Datierung: 'Gemälde, unbestimmte Epoche, unleserliche handschriftliche Notizen'",
    ]
