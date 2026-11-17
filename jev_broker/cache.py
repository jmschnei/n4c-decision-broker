"""Decision cache: identical (state, question, policy) tuples reuse the governed
decision instead of re-billing the provider. Deterministic models make this safe."""
from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Union

from jev_broker.models import AnswerResult, QuestionPolicy, QuestionSpec, state_sha256, stable_json

__all__ = ["DecisionCache", "cache_key"]


def cache_key(state_sha: str, question: QuestionSpec, policy: QuestionPolicy) -> str:
    basis = stable_json(
        {
            "state": state_sha,
            "question": question.cache_key(),
            "policy": policy.model_dump(mode="json"),
        }
    )
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


class DecisionCache:
    """In-memory LRU cache with optional JSON disk persistence."""

    def __init__(self, max_size: int = 10_000, path: Optional[Union[str, Path]] = None):
        self._max = max_size
        self._path = Path(path) if path else None
        self._data: "OrderedDict[str, dict]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        if self._path is not None:
            self._load()

    def _load(self) -> None:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            for k, v in raw.items():
                self._data[k] = v
        except (OSError, json.JSONDecodeError):
            self._data = OrderedDict()

    def get(self, key: str) -> Optional[AnswerResult]:
        with self._lock:
            payload = self._data.get(key)
            if payload is None:
                self.misses += 1
                return None
            self.hits += 1
            self._data.move_to_end(key)
            return AnswerResult.model_validate(payload)

    def put(self, key: str, value: AnswerResult) -> None:
        with self._lock:
            self._data[key] = value.model_dump(mode="json")
            self._data.move_to_end(key)
            while len(self._data) > self._max:
                self._data.popitem(last=False)
            if self._path is not None:
                self._save_unlocked()

    def _save_unlocked(self) -> None:
        assert self._path is not None
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, ensure_ascii=False), encoding="utf-8")

    def stats(self) -> dict:
        with self._lock:
            return {
                "size": len(self._data),
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / max(1, self.hits + self.misses), 4),
            }
