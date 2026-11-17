"""Append-only JSONL audit log (NFDI traceability): one line per broker decision."""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

__all__ = ["AuditLog"]


class AuditLog:
    def __init__(self, path: Optional[Union[str, Path]] = None):
        self._path = Path(path) if path else None
        self._lock = threading.Lock()

    def record(self, entry: dict[str, Any]) -> None:
        if self._path is None:
            return
        if "ts" not in entry:
            entry = {**entry, "ts": datetime.now(timezone.utc).isoformat()}
        line = json.dumps(entry, ensure_ascii=False, default=str)
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def recent(self, limit: int = 20) -> list[dict]:
        if self._path is None or not self._path.exists():
            return []
        lines = self._path.read_text(encoding="utf-8").strip().splitlines()[-limit:]
        out = []
        for line in reversed(lines):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out
