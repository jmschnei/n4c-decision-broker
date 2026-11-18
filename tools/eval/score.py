#!/usr/bin/env python3
"""Step 2: score generated SPARQL against the gold queries.

    python3 tools/eval/score.py --data datasets/normalized --gen results/generation \
        --out results/scores [--execute]

Offline metrics (always): syntax_valid (strict W3C SPARQL 1.1; Virtuoso/GOLEM accepts some
non-standard forms such as SELECT COUNT(..) AS ?x, which exec_match still credits), exact_match, skeleton_match, iri_precision/recall/f1,
feature_match. With --execute (needs internet) both queries are run on the dataset's endpoint
and result sets are compared (exec_match, exec_f1); only GOLEM and QALD have an endpoint here,
Spider4SPARQL needs a local triplestore (not set up), so it is scored offline only.

Writes item_scores.jsonl (one row per model x question: the router's training table) and
summary.json / summary.md (aggregates per model, source, complexity, language).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # tools/
from normalize_datasets import _strip_strings_and_prefixes, features, skeleton  # noqa: E402

logging.getLogger("rdflib").setLevel(logging.ERROR)  # URI-serialization warnings are noise here

ENDPOINTS = {  # source -> SPARQL endpoint (third-party servers: be gentle)
    "golem": "http://graph.golemlab.eu:8890/sparql",
    # QALD authors' stable endpoint (https://skynet.coypu.org/wikidata/) returned 503 when tested,
    # so the default is the official Wikidata service. It is live (newer than the 2023 dump the
    # gold answers come from), but gold and generated queries both run on it, so the comparison
    # stays consistent. Override with --qald-endpoint.
    "qald10": "https://query.wikidata.org/sparql",
    "qald9plus": "https://query.wikidata.org/sparql",
}
NS = {
    "wd": "http://www.wikidata.org/entity/", "wdt": "http://www.wikidata.org/prop/direct/",
    "p": "http://www.wikidata.org/prop/", "ps": "http://www.wikidata.org/prop/statement/",
    "pq": "http://www.wikidata.org/prop/qualifier/", "wikibase": "http://wikiba.se/ontology#",
    "bd": "http://www.bigdata.com/rdf#", "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#", "xsd": "http://www.w3.org/2001/XMLSchema#",
    "schema": "http://schema.org/", "dct": "http://purl.org/dc/terms/", "dc": "http://purl.org/dc/terms/",
    "golem": "https://golemlab.eu/graph/", "gc": "https://ontology.golemlab.eu/",
    "owl": "http://www.w3.org/2002/07/owl#", "skos": "http://www.w3.org/2004/02/skos/core#",
    "foaf": "http://xmlns.com/foaf/0.1/", "geo": "http://www.opengis.net/ont/geosparql#",
}


# ------------------------------------------------------------------ offline metrics
def syntax_valid(sparql: str, source: str) -> bool:
    if not sparql.strip():
        return False
    try:
        from rdflib.plugins.sparql import prepareQuery
    except ImportError:  # heuristic fallback
        q = _strip_strings_and_prefixes(sparql)
        return bool(re.search(r"(?i)\b(select|ask|construct|describe)\b", q)) and q.count("{") == q.count("}") > 0
    ns = dict(NS)
    if source == "spider4sparql":
        ns[""] = "http://example.org/kg/"  # Spider KGs use ':' as default namespace
    try:
        prepareQuery(sparql, initNs=ns)
        return True
    except Exception:  # noqa: BLE001 - any parse failure = invalid
        return False


def body_norm(sparql: str) -> str:
    q = re.sub(r"(?i)\bprefix\s+[\w-]*:\s*<[^>]*>", " ", sparql)
    return re.sub(r"\s+", " ", q).strip().lower()


def iri_set(sparql: str) -> set[str]:
    """IRIs and prefixed names used (prefixes expanded so wd:Q1 == <http://...Q1>)."""
    prefixes = {m.group(1): m.group(2) for m in re.finditer(r"(?i)\bprefix\s+([\w-]*):\s*<([^>]*)>", sparql)}
    q = _strip_strings_and_prefixes(sparql)
    out = {m.group(1) for m in re.finditer(r"<([^>\s]+)>", q)}
    for m in re.finditer(r"(?<![\w?$<])([A-Za-z][\w-]*)?:([\w][\w.-]*)", q):
        pre, local = m.group(1) or "", m.group(2).rstrip(".")
        base = prefixes.get(pre) or NS.get(pre)
        out.add(base + local if base else f"{pre}:{local}")
    out -= {"http://www.w3.org/1999/02/22-rdf-syntax-ns#type"}
    return out


def prf(pred: set, gold: set) -> tuple[float, float, float]:
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 1.0
    return round(p, 4), round(r, 4), round(2 * p * r / (p + r), 4) if p + r else 0.0


FEATURE_KEYS = ("aggregation", "group_by", "order_by", "filter", "optional", "union", "subquery", "ask")


def offline_scores(row: dict, gen: str) -> dict:
    gold = row["sparql"]
    p, r, f1 = prf(iri_set(gen), iri_set(gold))
    gf, pf = features(gold), features(gen) if gen.strip() else {}
    return {
        "syntax_valid": syntax_valid(gen, row["source"]),
        "exact_match": bool(gen.strip()) and body_norm(gen) == body_norm(gold),
        "skeleton_match": bool(gen.strip()) and skeleton(gen) == skeleton(gold),
        "iri_precision": p, "iri_recall": r, "iri_f1": f1,
        "feature_match": bool(pf) and all(gf[k] == pf[k] for k in FEATURE_KEYS),
    }


# ------------------------------------------------------------------ execution scoring
class Executor:
    """Runs queries on public endpoints with an on-disk cache, spacing and a timeout."""

    def __init__(self, cache_path: Path, min_interval: float = 0.3, timeout: float = 60.0,
                 user_agent: str = "n4c-decision-broker-eval/0.1"):
        import httpx
        self.http = httpx.Client(timeout=timeout, headers={
            "Accept": "application/sparql-results+json", "User-Agent": user_agent})
        self.path, self.min_interval, self.last = cache_path, min_interval, 0.0
        self.cache: dict[str, dict] = {}
        if cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                try:
                    d = json.loads(line)
                    self.cache[d["key"]] = d["res"]
                except (json.JSONDecodeError, KeyError):
                    pass

    def run(self, endpoint: str, query: str) -> dict:
        key = hashlib.sha256(f"{endpoint}\n{query}".encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        interval = 1.0 if "wikidata.org" in endpoint else self.min_interval  # WDQS usage policy
        wait = interval - (time.time() - self.last)
        if wait > 0:
            time.sleep(wait)
        try:
            r = self.http.post(endpoint, data={"query": query})
            self.last = time.time()
            if r.status_code == 200:
                res = {"ok": True, "rows": normalize_results(r.json())}
            else:
                res = {"ok": False, "error": f"HTTP {r.status_code}"}
        except Exception as exc:  # noqa: BLE001
            self.last = time.time()
            res = {"ok": False, "error": type(exc).__name__}
        if res["ok"] or res["error"] == "HTTP 400":  # cache deterministic outcomes only (not 403/429/5xx)
            self.cache[key] = res
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, "res": res}) + "\n")
        return res


def normalize_results(d: dict):
    if "boolean" in d:
        return {"bool": d["boolean"]}
    vars_ = d.get("head", {}).get("vars", [])
    rows = sorted(json.dumps([b.get(v, {}).get("value") for v in vars_], ensure_ascii=False)
                  for b in d.get("results", {}).get("bindings", []))
    return {"rows": rows}


def exec_scores(ex: Executor, row: dict, gen: str) -> dict:
    ep = ENDPOINTS.get(row["source"])
    if not ep:
        return {}
    g = ex.run(ep, row["sparql"])
    if not g["ok"]:
        return {"exec_gold_failed": True}
    if not gen.strip():
        return {"exec_ok": False, "exec_match": False, "exec_f1": 0.0}
    o = ex.run(ep, gen)
    if not o["ok"]:
        return {"exec_ok": False, "exec_match": False, "exec_f1": 0.0}
    a, b = g["rows"], o["rows"]
    if "bool" in a or "bool" in b:
        same = a == b
        return {"exec_ok": True, "exec_match": same, "exec_f1": 1.0 if same else 0.0}
    sa, sb = set(a["rows"]), set(b["rows"])
    _, _, f1 = prf(sb, sa)
    return {"exec_ok": True, "exec_match": a["rows"] == b["rows"] or sa == sb, "exec_f1": f1}


# ------------------------------------------------------------------ aggregation
def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 3) if xs else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def aggregate(items: list[dict], keys: tuple) -> dict:
    groups = defaultdict(list)
    for it in items:
        groups[tuple(it[k] for k in keys)].append(it)
    out = {}
    for k, rows in sorted(groups.items()):
        ex_rows = [r for r in rows if "exec_match" in r]
        out["|".join(map(str, k))] = {
            "n": len(rows),
            "syntax_valid": mean([r["syntax_valid"] for r in rows]),
            "exact_match": mean([r["exact_match"] for r in rows]),
            "skeleton_match": mean([r["skeleton_match"] for r in rows]),
            "iri_f1": mean([r["iri_f1"] for r in rows]),
            "feature_match": mean([r["feature_match"] for r in rows]),
            "exec_n": len(ex_rows),
            "exec_match": mean([r["exec_match"] for r in ex_rows]),
            "exec_f1": mean([r["exec_f1"] for r in ex_rows]),
            "errors": sum(1 for r in rows if r.get("error")),
            "truncated": sum(1 for r in rows if r.get("truncated")),
            "latency_p50_s": pct([r.get("latency_s") for r in rows], 0.5),
            "latency_p95_s": pct([r.get("latency_s") for r in rows], 0.95),
            "tokens_per_s": mean([r.get("tokens_per_s") for r in rows]),
        }
    return out


def to_markdown(summary: dict) -> str:
    cols = ["n", "syntax_valid", "exact_match", "skeleton_match", "iri_f1", "exec_match", "exec_n",
            "latency_p50_s", "tokens_per_s"]
    lines = []
    for title, key in (("Per model", "by_model"), ("Per model x source", "by_model_source"),
                       ("Per model x complexity", "by_model_complexity")):
        lines += [f"\n### {title}\n", "| group | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
        for g, v in summary[key].items():
            lines.append(f"| {g} | " + " | ".join("" if v[c] is None else str(v[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--gen", required=True, type=Path, help="dir with the generate.py *.jsonl outputs")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--qald-endpoint", help="SPARQL endpoint for QALD-10 / QALD-9-plus")
    ap.add_argument("--user-agent", default=os.environ.get("N4C_EVAL_USER_AGENT", "n4c-decision-broker-eval/0.1"),
                    help="HTTP User-Agent for endpoints. Wikidata's WDQS rejects (403) agents without contact\n"
                         "info per https://w.wiki/4wJS: set e.g. 'n4c-eval/0.1 (mailto:you@institution)'")
    ap.add_argument("--execute", action="store_true", help="also run queries on the endpoints (internet)")
    args = ap.parse_args()

    if args.qald_endpoint:
        ENDPOINTS["qald10"] = ENDPOINTS["qald9plus"] = args.qald_endpoint
    rows = {}
    for line in (args.data / "eval_sample.jsonl").read_text(encoding="utf-8").splitlines():
        if line:
            r = json.loads(line)
            rows[r["id"]] = r
    gens = []
    for f in sorted(args.gen.glob("*.shard*.jsonl")):
        gens += [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l]
    args.out.mkdir(parents=True, exist_ok=True)
    ex = Executor(args.out / "exec_cache.jsonl", user_agent=args.user_agent) if args.execute else None

    items, missing = [], 0
    for g in gens:
        row = rows.get(g["id"])
        if row is None:
            missing += 1
            continue
        item = {"id": g["id"], "model": g["model"], "source": row["source"], "language": row["language"],
                "complexity": row["complexity"], "split": row["split"], "group": row["group"],
                "error": g.get("error"), "truncated": g.get("truncated"), "latency_s": g.get("latency_s"),
                "output_tokens": g.get("output_tokens"), "tokens_per_s": g.get("tokens_per_s"),
                "sparql_generated": g["sparql_generated"]}
        item.update(offline_scores(row, g["sparql_generated"]))
        if ex:
            item.update(exec_scores(ex, row, g["sparql_generated"]))
        items.append(item)
    items.sort(key=lambda i: (i["model"], i["id"]))
    (args.out / "item_scores.jsonl").write_text(
        "\n".join(json.dumps(i, ensure_ascii=False) for i in items) + "\n", encoding="utf-8")
    summary = {
        "n_items": len(items), "ignored_unknown_ids": missing, "execute": bool(args.execute),
        "by_model": aggregate(items, ("model",)),
        "by_model_source": aggregate(items, ("model", "source")),
        "by_model_complexity": aggregate(items, ("model", "complexity")),
        "by_model_language": aggregate(items, ("model", "language")),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out / "summary.md").write_text(to_markdown(summary), encoding="utf-8")
    print(to_markdown(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
