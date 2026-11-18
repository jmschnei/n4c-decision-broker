#!/usr/bin/env python3
"""Normalize public text-to-SPARQL datasets into one JSONL schema.

Usage:  python3 -I tools/normalize_datasets.py RAW_DIR OUT_DIR

RAW_DIR has one sub-directory per dataset (spider4sparql, golem, qald10, qald9plus),
exactly as downloaded. Raw files are treated as untrusted data: they are only parsed
(json/csv), never executed. Nothing outside OUT_DIR is written.

Record schema (one JSON object per line):
  id, source, kg, language, question, sparql, gold_answers (or null),
  group (leakage-safe split unit), split (calibration|test), features{...},
  complexity (easy|medium|hard), license
Splits are decided per *group* (all rows of one group land in the same split):
  spider4sparql -> knowledge graph, golem -> SPARQL skeleton, qald -> question id.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

csv.field_size_limit(10**8)
SEED = 20261118
LICENSES = {
    "spider4sparql": "Apache-2.0",
    "golem": "UNVERIFIED (no license file in repo) - internal evaluation use only",
    "qald10": "MIT",
    "qald9plus": "CC-BY-4.0",
}
SAMPLE_TARGET = {"spider4sparql": 300, "golem": 200, "qald10": 100, "qald9plus": 100}
QALD_LANGS = ("en", "de")


# ---------------------------------------------------------------- SPARQL features
def _strip_strings_and_prefixes(q: str) -> str:
    q = re.sub(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'', '""', q)
    # declarations only: whole queries are often a single line, so never strip by line
    q = re.sub(r"(?i)\bprefix\s+[\w-]*:\s*<[^>]*>", " ", q)
    q = re.sub(r"(?i)\bbase\s*<[^>]*>", " ", q)
    return q


def features(sparql: str) -> dict:
    q = _strip_strings_and_prefixes(sparql)
    low = q.lower()
    body_start = low.find("where")
    body = q[body_start:] if body_start >= 0 else q
    # triple-pattern estimate: statement terminators + predicate-object list separators
    # inside the WHERE body (heuristic, no SPARQL parser dependency)
    n_patterns = max(1, len(re.findall(r"\.(?=\s|$|\})", body)) + body.count(";") +
                     (0 if re.search(r"\.\s*\}?\s*$", body.strip()) else 1))
    return {
        "length_chars": len(sparql),
        "n_patterns_est": n_patterns,
        "n_vars": len(set(re.findall(r"[?$]\w+", q))),
        "aggregation": bool(re.search(r"\b(count|sum|avg|min|max|group_concat|sample)\s*\(", low)),
        "group_by": "group by" in low,
        "order_by": "order by" in low,
        "limit": bool(re.search(r"\blimit\s+\d+", low)),
        "filter": "filter" in low,
        "optional": "optional" in low,
        "union": "union" in low,
        "minus_or_not_exists": bool(re.search(r"\bminus\b|not exists", low)),
        "subquery": len(re.findall(r"\bselect\b", low)) > 1,
        "property_path": bool(re.search(r"[)>\w]\s*[/|^]\s*[<:\w(^]|[>\w]\s*[*+]\s", body)),
        "ask": bool(re.match(r"\s*ask\b", low)),
        "service": "service" in low,
        "values": "values" in low,
    }


def complexity(f: dict) -> str:
    mods = sum(f[k] for k in ("aggregation", "group_by", "filter", "optional", "union",
                              "minus_or_not_exists", "subquery", "property_path", "service"))
    if f["subquery"] or f["n_patterns_est"] >= 6 or mods >= 3:
        return "hard"
    if f["n_patterns_est"] <= 2 and mods == 0:
        return "easy"
    return "medium"


def skeleton(sparql: str) -> str:
    """Query with literals/IRIs/numbers abstracted: identifies a question template."""
    q = _strip_strings_and_prefixes(sparql)
    q = re.sub(r"<[^>\s]*>", "<I>", q)
    q = re.sub(r"\b(wd|wdt|p|ps|pq|psn|wdv|wds|dbr|dbo|dbp|rdfs|rdf|golem|gc|dc):[\w-]+", r"\1:X", q)
    q = re.sub(r"\b\d+(\.\d+)?\b", "N", q)
    q = re.sub(r"\s+", " ", q).strip().lower()
    return hashlib.sha256(q.encode()).hexdigest()[:16]


def split_of(group: str) -> str:
    h = int(hashlib.sha256(f"{SEED}|{group}".encode()).hexdigest(), 16)
    return "calibration" if h % 2 == 0 else "test"


def record(source, rid, kg, lang, question, sparql, group, answers=None) -> dict:
    f = features(sparql)
    return {
        "id": f"{source}:{rid}", "source": source, "kg": kg, "language": lang,
        "question": " ".join(question.split()), "sparql": sparql.strip(),
        "gold_answers": answers, "group": f"{source}:{group}", "split": split_of(f"{source}:{group}"),
        "features": f, "complexity": complexity(f), "license": LICENSES[source],
    }


# ---------------------------------------------------------------- loaders
def load_spider4sparql(raw: Path):
    data = json.loads((raw / "dev.json").read_text(encoding="utf-8"))
    for i, d in enumerate(data):
        yield record("spider4sparql", i, d["kg_name"], "en", d["question"], d["query"], group=d["kg_name"])


def load_golem(raw: Path):
    seen = set()
    for fname in ("instantiated_data", "paraphrased_data"):
        with (raw / f"{fname}.tsv").open(encoding="utf-8", newline="") as fh:
            for i, r in enumerate(csv.DictReader(fh, delimiter="\t")):
                q, s, lang = r["Question"].strip(), r["SPARQL"].strip(), r["Language"].strip().lower()
                if not q or not s or (q, s) in seen:
                    continue
                seen.add((q, s))
                yield record("golem", f"{fname[:4]}-{i}", "golem", lang, q, s, group=skeleton(s))


def _qald(source: str, raw_file: Path, kg: str):
    data = json.loads(raw_file.read_text(encoding="utf-8"))["questions"]
    for d in data:
        sparql = (d.get("query") or {}).get("sparql")
        if not sparql:
            continue
        answers = d.get("answers")
        small = answers if answers and len(json.dumps(answers)) <= 4000 else None
        by_lang = {}
        for q in d["question"]:
            by_lang.setdefault(q["language"], q["string"])  # first variant per language
        for lang in QALD_LANGS:
            if lang in by_lang:
                yield record(source, f"{d['id']}-{lang}", kg, lang, by_lang[lang], sparql,
                             group=str(d["id"]), answers=small)


def load_qald10(raw: Path):
    yield from _qald("qald10", raw / "qald_10.json", "wikidata")


def load_qald9plus(raw: Path):
    yield from _qald("qald9plus", raw / "qald_9_plus_train_wikidata.json", "wikidata")


LOADERS = {"spider4sparql": load_spider4sparql, "golem": load_golem,
           "qald10": load_qald10, "qald9plus": load_qald9plus}


# ---------------------------------------------------------------- sampling
def stratified_sample(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Round-robin over (language, complexity) strata; within a stratum prefer unseen groups."""
    if len(rows) <= n:
        return list(rows)
    strata = defaultdict(list)
    for r in rows:
        strata[(r["language"], r["complexity"])].append(r)
    for v in strata.values():
        rng.shuffle(v)
    picked, used_groups = [], Counter()
    keys = sorted(strata)
    while len(picked) < n and any(strata.values()):
        for k in keys:
            if len(picked) >= n:
                break
            pool = strata[k]
            if not pool:
                continue
            pool.sort(key=lambda r: used_groups[r["group"]])  # least-used group first (stable)
            r = pool.pop(0)
            used_groups[r["group"]] += 1
            picked.append(r)
    return picked


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    raw_dir, out_dir = Path(argv[1]), Path(argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    manifest, sample = {"seed": SEED, "datasets": {}}, []
    for name, loader in LOADERS.items():
        rows = list(loader(raw_dir / name))
        (out_dir / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
        picked = stratified_sample(rows, SAMPLE_TARGET[name], rng)
        sample.extend(picked)
        manifest["datasets"][name] = {
            "license": LICENSES[name], "rows": len(rows), "groups": len({r["group"] for r in rows}),
            "languages": dict(Counter(r["language"] for r in rows)),
            "complexity": dict(Counter(r["complexity"] for r in rows)),
            "split": dict(Counter(r["split"] for r in rows)),
            "sampled": len(picked),
            "raw_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted((raw_dir / name).iterdir()) if p.is_file()},
        }
    (out_dir / "eval_sample.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in sample) + "\n", encoding="utf-8")
    manifest["eval_sample"] = {
        "rows": len(sample), "by_source": dict(Counter(r["source"] for r in sample)),
        "by_split": dict(Counter(r["split"] for r in sample)),
        "by_complexity": dict(Counter(r["complexity"] for r in sample)),
        "by_language": dict(Counter(r["language"] for r in sample)),
    }
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
