#!/usr/bin/env python3
"""Step 2: score generated SPARQL against the gold queries.

    python3 tools/eval/score.py --data datasets/normalized --gen results/generation \
        --out results/scores [--execute] [--spider-kg datasets/raw/spider4sparql_kg]

Offline metrics (always): syntax_valid (strict W3C SPARQL 1.1; Virtuoso/GOLEM accepts some
non-standard forms such as SELECT COUNT(..) AS ?x, which exec_match still credits), exact_match, skeleton_match, iri_precision/recall/f1,
feature_match. Execution metrics run gold and generated queries and compare result sets
(exec_match, exec_f1; column order ignored): --execute does it for GOLEM and QALD on their public
endpoints (internet); --spider-kg DIR does it for Spider4SPARQL on its materialized KGs in a
local Oxigraph store (needs `pyoxigraph`; offline, so it also works on compute nodes).
Items whose gold query fails or returns no rows are excluded from exec metrics (counted as
exec_excluded): an empty gold would make every empty prediction a "match".

    python3 tools/eval/score.py --data datasets/normalized --gold-only [--execute] \
        --spider-kg datasets/raw/spider4sparql_kg --out results/scores

--gold-only runs just the gold queries (no generation needed): it reports which items are
executable and pre-fills exec_cache.jsonl, which can be copied to the cluster's scores dir.

Writes item_scores.jsonl (one row per model x question: the router's training table) and
summary.json / summary.md (aggregates per model, source, complexity, language).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # tools/
if os.environ.get("N4C_PYLIB"):  # pinned extra packages (pip --target), ahead of the image's own;
    sys.path.insert(0, os.environ["N4C_PYLIB"])  # read explicitly because -I ignores PYTHONPATH
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
SPIDER_NS = "http://valuenet/ontop/"  # default ':' namespace of the Spider4SPARQL materialized KGs
LOCAL = "oxigraph:"                   # pseudo-endpoint prefix: oxigraph:<kg name>


class RobotPolicyError(SystemExit):
    pass


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
        ns[""] = SPIDER_NS  # Spider KGs use ':' as default namespace
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


# Spider4SPARQL predicates look like :singer\#name. Models often drop the backslash, and an
# unescaped '#' starts a SPARQL comment, so the rest of the line is lost and the query breaks.
# The intent is unambiguous, so it is repaired before scoring (counted in `repaired`).
_SPIDER_HASH = re.compile(r"(?<![\\<\"'\w])(:[A-Za-z_][\w-]*)#(?=[A-Za-z_])")


def repair_sparql(sparql: str, source: str) -> str:
    """Fix known, meaning-preserving formatting slips (currently: Spider's unescaped '#')."""
    if source != "spider4sparql":
        return sparql
    return _SPIDER_HASH.sub(r"\1\\#", sparql)


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
def _load_kg(ox, kg_dir: str, kg: str):
    """Load one Spider4SPARQL KG; tolerate a truncated last line (wta_1.ttl ships cut off)."""
    store = ox.Store()
    path = Path(kg_dir) / f"{kg}.ttl"
    if not path.exists():
        return store
    data = path.read_bytes()
    try:
        store.load(data, format=ox.RdfFormat.TURTLE)
    except SyntaxError:
        store = ox.Store()
        store.load(data[: data.rstrip(b"\n").rfind(b"\n") + 1], format=ox.RdfFormat.TURTLE)
    return store


def _kg_worker(kg_dir: str, conn) -> None:
    """Child process: answers (kg, query) requests; the parent kills it on timeout."""
    import pyoxigraph as ox
    stores: dict = {}
    while True:
        msg = conn.recv()
        if msg is None:
            return
        kg, query = msg
        try:
            if kg not in stores:
                stores[kg] = _load_kg(ox, kg_dir, kg)
            if len(stores[kg]) == 0:
                conn.send({"ok": False, "error": "empty KG"})
                continue
            res = stores[kg].query(f"PREFIX : <{SPIDER_NS}>\n{query}")
            if isinstance(res, ox.QueryBoolean) or isinstance(res, bool):
                conn.send({"ok": True, "rows": {"bool": bool(res)}})
                continue
            if isinstance(res, ox.QuerySolutions):
                vs = res.variables
                rows = [[None if s[v] is None else s[v].value for v in vs] for s in res]
            else:  # CONSTRUCT / DESCRIBE
                rows = [[t.subject.value, t.predicate.value, t.object.value] for t in res]
            conn.send({"ok": True, "rows": {"rows": sorted(json.dumps(r, ensure_ascii=False) for r in rows)}})
        except SyntaxError:
            conn.send({"ok": False, "error": "HTTP 400"})  # same label as a remote parse error
        except Exception as exc:  # noqa: BLE001
            conn.send({"ok": False, "error": type(exc).__name__})


class LocalKG:
    """Spider4SPARQL KGs in an in-memory Oxigraph store, in a worker process with a timeout."""

    def __init__(self, kg_dir: Path, timeout: float = 30.0):
        import pyoxigraph  # noqa: F401 - fail early with a clear ImportError
        self.kg_dir, self.timeout = str(kg_dir), timeout
        self.ctx = mp.get_context("spawn")
        self.proc = self.conn = None

    def _start(self):
        self.conn, child = self.ctx.Pipe()
        self.proc = self.ctx.Process(target=_kg_worker, args=(self.kg_dir, child), daemon=True)
        self.proc.start()

    def run(self, kg: str, query: str) -> dict:
        if self.proc is None or not self.proc.is_alive():
            self._start()
        try:
            self.conn.send((kg, query))
            # first request per KG also parses it (world_1: ~38k triples, < 1 s)
            if self.conn.poll(self.timeout):
                return self.conn.recv()
            error = "timeout"
        except (EOFError, OSError):  # worker died (e.g. out of memory on a huge join)
            error = "worker crashed"
        self.proc.kill()
        self.proc.join()
        self.proc = None
        return {"ok": False, "error": error}

    def close(self):
        if self.proc is not None and self.proc.is_alive():
            self.conn.send(None)
            self.proc.join(5)


class Executor:
    """Runs queries on public endpoints (or the local Spider KGs) with an on-disk cache,
    spacing and a timeout."""

    def __init__(self, cache_path: Path, min_interval: float = 0.3, timeout: float = 60.0,
                 user_agent: str = "n4c-decision-broker-eval/0.1", spider_kg: Path | None = None,
                 remote: bool = True):
        import httpx
        self.http = httpx.Client(timeout=timeout, headers={
            "Accept": "application/sparql-results+json", "User-Agent": user_agent})
        self.local = LocalKG(spider_kg) if spider_kg else None
        self.remote = remote  # False: only the local Spider KGs (no internet needed)
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
        if endpoint.startswith(LOCAL):
            res = self.local.run(endpoint[len(LOCAL):], query)  # deterministic: always cached
        else:
            res = self._remote(endpoint, query)
        if endpoint.startswith(LOCAL) or res["ok"] or res["error"] == "HTTP 400":
            # cache deterministic outcomes only (not 403/429/5xx/network errors)
            self.cache[key] = res
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, "res": res}) + "\n")
        return res

    def _remote(self, endpoint: str, query: str) -> dict:
        interval = 1.0 if "wikidata.org" in endpoint else self.min_interval  # WDQS usage policy
        wait = interval - (time.time() - self.last)
        if wait > 0:
            time.sleep(wait)
        try:
            r = self.http.post(endpoint, data={"query": query})
            self.last = time.time()
            if r.status_code == 200:
                return {"ok": True, "rows": normalize_results(r.json())}
            if r.status_code == 403 and "robot policy" in r.text:
                # every Wikidata request would fail: stop instead of silently scoring exec_n=0
                raise RobotPolicyError(
                    f"{endpoint} rejected the User-Agent (403, robot policy https://w.wiki/4wJS).\n"
                    "Set contact info: --user-agent 'n4c-eval/0.1 (mailto:you@institution)' or "
                    "N4C_EVAL_USER_AGENT, or score without --execute.")
            return {"ok": False, "error": f"HTTP {r.status_code}"}
        except RobotPolicyError:
            raise
        except Exception as exc:  # noqa: BLE001
            self.last = time.time()
            return {"ok": False, "error": type(exc).__name__}

    def close(self):
        if self.local:
            self.local.close()


def normalize_results(d: dict):
    if "boolean" in d:
        return {"bool": d["boolean"]}
    vars_ = d.get("head", {}).get("vars", [])
    rows = sorted(json.dumps([b.get(v, {}).get("value") for v in vars_], ensure_ascii=False)
                  for b in d.get("results", {}).get("bindings", []))
    return {"rows": rows}


def endpoint_for(ex: Executor, row: dict) -> str | None:
    if row["source"] == "spider4sparql":
        return f"{LOCAL}{row['group'].split(':', 1)[-1]}" if ex.local else None
    return ENDPOINTS.get(row["source"]) if ex.remote else None


def gold_status(g: dict) -> str:
    if not g["ok"]:
        return "failed"
    return "empty" if g["rows"] == {"rows": []} else "ok"


def exec_scores(ex: Executor, row: dict, gen: str) -> dict:
    ep = endpoint_for(ex, row)
    if not ep:
        return {}
    g = ex.run(ep, row["sparql"])
    status = gold_status(g)
    if status != "ok":  # failed, or no rows: an empty prediction would "match" it
        return {"exec_gold_failed" if status == "failed" else "exec_gold_empty": True}
    if not gen.strip():
        return {"exec_ok": False, "exec_match": False, "exec_f1": 0.0}
    o = ex.run(ep, gen)
    if not o["ok"]:
        return {"exec_ok": False, "exec_match": False, "exec_f1": 0.0}
    a, b = g["rows"], o["rows"]
    if "bool" in a or "bool" in b:
        same = a == b
        return {"exec_ok": True, "exec_match": same, "exec_f1": 1.0 if same else 0.0}
    sa, sb = set(a["rows"]), best_column_order(a["rows"], b["rows"])
    _, _, f1 = prf(sb, sa)
    return {"exec_ok": True, "exec_match": sa == sb, "exec_f1": f1}


def best_column_order(gold_rows: list[str], pred_rows: list[str], max_cols: int = 5) -> set[str]:
    """Prediction rows with their columns permuted to best match the gold ones: a correct answer
    that just SELECTs the variables in another order still counts."""
    pred = set(pred_rows)
    if not gold_rows or not pred_rows or pred & set(gold_rows) == pred | set(gold_rows):
        return pred
    from itertools import permutations
    parsed = [json.loads(r) for r in pred_rows]
    ncol = len(parsed[0])
    if ncol < 2 or ncol > max_cols or ncol != len(json.loads(gold_rows[0])):
        return pred
    gold = set(gold_rows)
    best, best_tp = pred, len(pred & gold)
    for perm in permutations(range(ncol)):
        cand = {json.dumps([r[i] for i in perm], ensure_ascii=False) for r in parsed}
        tp = len(cand & gold)
        if tp > best_tp:
            best, best_tp = cand, tp
    return best


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
            "exec_excluded": sum(1 for r in rows if r.get("exec_gold_failed") or r.get("exec_gold_empty")),
            "exec_match": mean([r["exec_match"] for r in ex_rows]),
            "exec_f1": mean([r["exec_f1"] for r in ex_rows]),
            "errors": sum(1 for r in rows if r.get("error")),
            "truncated": sum(1 for r in rows if r.get("truncated")),
            "repaired": sum(1 for r in rows if r.get("repaired")),
            "latency_p50_s": pct([r.get("latency_s") for r in rows], 0.5),
            "latency_p95_s": pct([r.get("latency_s") for r in rows], 0.95),
            "tokens_per_s": mean([r.get("tokens_per_s") for r in rows]),
        }
    return out


def to_markdown(summary: dict) -> str:
    cols = ["n", "syntax_valid", "exact_match", "skeleton_match", "iri_f1", "exec_match", "exec_n",
            "exec_excluded", "repaired", "latency_p50_s", "tokens_per_s"]
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
    ap.add_argument("--gen", type=Path, help="dir with the generate.py *.jsonl outputs")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--qald-endpoint", help="SPARQL endpoint for QALD-10 / QALD-9-plus")
    ap.add_argument("--user-agent", default=os.environ.get("N4C_EVAL_USER_AGENT", "n4c-decision-broker-eval/0.1"),
                    help="HTTP User-Agent for endpoints. Wikidata's WDQS rejects (403) agents without contact\n"
                         "info per https://w.wiki/4wJS: set e.g. 'n4c-eval/0.1 (mailto:you@institution)'")
    ap.add_argument("--execute", action="store_true",
                    help="also run GOLEM/QALD queries on their public endpoints (internet)")
    ap.add_argument("--spider-kg", type=Path,
                    help="dir with the Spider4SPARQL materialized KGs (<kg>.ttl): execute Spider\n"
                         "queries locally with pyoxigraph (offline; independent of --execute)")
    ap.add_argument("--gold-only", action="store_true",
                    help="only execute the gold queries: report executable items, fill the cache")
    ap.add_argument("--sources", help="comma separated subset, e.g. golem,spider4sparql (default: all)")
    args = ap.parse_args()
    if not args.gold_only and not args.gen:
        ap.error("--gen is required unless --gold-only")
    if args.spider_kg and not args.spider_kg.is_dir():
        ap.error(f"--spider-kg {args.spider_kg} is not a directory")

    if args.qald_endpoint:
        ENDPOINTS["qald10"] = ENDPOINTS["qald9plus"] = args.qald_endpoint
    rows = {}
    for line in (args.data / "eval_sample.jsonl").read_text(encoding="utf-8").splitlines():
        if line:
            r = json.loads(line)
            rows[r["id"]] = r
    if args.sources:
        keep = set(args.sources.split(","))
        rows = {k: r for k, r in rows.items() if r["source"] in keep}
    args.out.mkdir(parents=True, exist_ok=True)
    if args.gold_only and not (args.execute or args.spider_kg):
        ap.error("--gold-only needs --execute and/or --spider-kg")
    ex = Executor(args.out / "exec_cache.jsonl", user_agent=args.user_agent, spider_kg=args.spider_kg,
                  remote=args.execute) if (args.execute or args.spider_kg) else None
    try:
        if args.gold_only:
            return gold_check(ex, rows, args.out)
        return score_all(args, rows, ex)
    finally:
        if ex:
            ex.close()


def gold_check(ex: Executor, rows: dict, out: Path) -> int:
    report: dict = defaultdict(lambda: {"ok": 0, "empty": [], "failed": {}, "not_executable": 0})
    for i, row in enumerate(sorted(rows.values(), key=lambda r: r["id"]), 1):
        rep = report[row["source"]]
        ep = endpoint_for(ex, row)
        if not ep:
            rep["not_executable"] += 1
            continue
        g = ex.run(ep, row["sparql"])
        status = gold_status(g)
        if status == "ok":
            rep["ok"] += 1
        elif status == "empty":
            rep["empty"].append(row["id"])
        else:
            rep["failed"][row["id"]] = g["error"]
        if i % 50 == 0:
            print(f"  {i}/{len(rows)}", flush=True)
    (out / "gold_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n| source | executable (ok) | empty gold | failed gold | no endpoint |\n|---|---|---|---|---|")
    for src, rep in sorted(report.items()):
        print(f"| {src} | {rep['ok']} | {len(rep['empty'])} | {len(rep['failed'])} | {rep['not_executable']} |")
    print(f"\ndetails: {out / 'gold_check.json'}   cache: {ex.path}")
    return 0


def score_all(args, rows: dict, ex: Executor | None) -> int:
    gens = []
    for f in sorted(args.gen.glob("*.shard*.jsonl")):
        gens += [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l]

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
        gen = repair_sparql(g["sparql_generated"], row["source"])
        item["repaired"] = gen != g["sparql_generated"]
        if item["repaired"]:
            item["sparql_scored"] = gen
        item.update(offline_scores(row, gen))
        if ex:
            item.update(exec_scores(ex, row, gen))
        items.append(item)
    items.sort(key=lambda i: (i["model"], i["id"]))
    (args.out / "item_scores.jsonl").write_text(
        "\n".join(json.dumps(i, ensure_ascii=False) for i in items) + "\n", encoding="utf-8")
    summary = {
        "n_items": len(items), "ignored_unknown_ids": missing, "execute": bool(args.execute),
        "spider_kg": bool(args.spider_kg),
        "pyoxigraph": __import__("pyoxigraph").__version__ if args.spider_kg else None,
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
