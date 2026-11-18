#!/usr/bin/env python3
"""Step 1 of the evaluation: ask one Ollama model to write the SPARQL for every question.

    python3 tools/eval/generate.py --model qwen3-coder:30b --data datasets/normalized \
        --out results/generation --base-url http://127.0.0.1:11434 [--shard 0 --num-shards 2]

Zero-shot knowledge of the KG is impossible, so each prompt carries k few-shot examples of
the same source and language, chosen by word overlap with the question. They never include the
item itself, a query identical to its gold query, or (QALD/GOLEM) the same question id/template.
For Spider4SPARQL same-KG examples are allowed: they are the only schema hint available. Output is append-only JSONL, so an interrupted
job resumes where it stopped.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # tools/ (explicit: run with -I)
import httpx  # noqa: E402

SOURCES_NOTE = {
    "spider4sparql": "Write the query for a small knowledge graph from the Spider benchmark "
                     "(prefix ':' is the graph's default namespace).",
    "golem": "The knowledge graph is GOLEM, metadata about fan fiction stories "
             "(prefixes golem:, gc:, dc:).",
    "qald10": "The knowledge graph is Wikidata (entities wd:Q..., properties wdt:P...).",
    "qald9plus": "The knowledge graph is Wikidata (entities wd:Q..., properties wdt:P...).",
}
SYSTEM = ("You translate natural-language questions into SPARQL 1.1 queries. "
          "Answer with exactly one query inside a ```sparql code block and nothing else.")
WORD = re.compile(r"\w+", re.UNICODE)


def compact_prefixes(sparql: str) -> str:
    """Drop PREFIX declarations the query body never uses (keeps few-shot prompts short)."""
    decl = re.compile(r"(?i)\bprefix\s+([\w-]*):\s*<[^>]*>")
    body = decl.sub(" ", sparql)
    kept = [m.group(0) for m in decl.finditer(sparql)
            if re.search(r"(?<![\w-])" + re.escape(m.group(1)) + r":", body)]
    return (" ".join(kept) + "\n" + " ".join(body.split())).strip()


def words(text: str) -> set[str]:
    return {w.lower() for w in WORD.findall(text)}


class ShotPool:
    """Few-shot candidates per source, retrieved by Jaccard overlap, excluding the item's group."""

    def __init__(self, rows: list[dict]):
        self.by_source: dict[str, list[tuple[set[str], dict]]] = defaultdict(list)
        for r in rows:
            self.by_source[r["source"]].append((words(r["question"]), r))

    def shots(self, row: dict, k: int) -> list[dict]:
        qw = words(row["question"])
        own_sparql = compact_prefixes(row["sparql"])
        # group = question id (QALD) or SPARQL skeleton (GOLEM): never reuse it, or the shot
        # is a translation/instance of the answer. For Spider4SPARQL the group is the KG itself:
        # same-KG shots are the only way to show the (unknown) schema, so only the item's own
        # query is excluded.
        exclude_group = row["source"] != "spider4sparql"
        scored = []
        for w, cand in self.by_source[row["source"]]:
            if cand["id"] == row["id"] or cand["language"] != row["language"]:
                continue
            if exclude_group and cand["group"] == row["group"]:
                continue
            if compact_prefixes(cand["sparql"]) == own_sparql:
                continue
            union = len(qw | w) or 1
            scored.append((len(qw & w) / union, cand["id"], cand))
        scored.sort(key=lambda t: (-t[0], t[1]))  # deterministic
        picked, seen_sparql = [], set()
        for _, _, cand in scored:
            key = compact_prefixes(cand["sparql"])
            if key in seen_sparql:
                continue
            seen_sparql.add(key)
            picked.append(cand)
            if len(picked) == k:
                break
        return picked


def build_messages(row: dict, shots: list[dict]) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM + " " + SOURCES_NOTE.get(row["source"], "")}]
    for s in shots:
        msgs.append({"role": "user", "content": s["question"]})
        msgs.append({"role": "assistant", "content": "```sparql\n" + compact_prefixes(s["sparql"]) + "\n```"})
    msgs.append({"role": "user", "content": row["question"]})
    return msgs


def extract_sparql(text: str) -> str:
    """Pull the query out of a model reply (fenced block, or from the first SPARQL keyword)."""
    text = re.sub(r"<think>[\s\S]*?</think>", "", text or "")
    m = re.search(r"```(?:sparql|sql|turtle)?\s*\n?([\s\S]*?)```", text, flags=re.I)
    if m:
        return m.group(1).strip()
    m = re.search(r"(?is)\b(prefix|select|ask|construct|describe)\b[\s\S]*", text)
    return m.group(0).strip() if m else text.strip()


def call_model(client: httpx.Client, model: str, messages: list[dict], args) -> dict:
    body = {"model": model, "messages": messages, "stream": False, "think": args.think,
            "options": {"temperature": 0, "seed": 0, "num_predict": args.max_tokens,
                        "num_ctx": args.num_ctx}}
    last_err = None
    for attempt in range(4):
        t0 = time.perf_counter()
        try:
            r = client.post("/api/chat", json=body)
            if r.status_code >= 500 or r.status_code == 429:
                raise httpx.HTTPStatusError(r.text[:200], request=r.request, response=r)
            r.raise_for_status()
            d = r.json()
            wall = time.perf_counter() - t0
            out_tok = d.get("eval_count") or 0
            dur = (d.get("eval_duration") or 0) / 1e9
            return {
                "raw": d["message"]["content"], "error": None, "attempts": attempt + 1,
                "latency_s": round(wall, 3), "prompt_tokens": d.get("prompt_eval_count"),
                "output_tokens": out_tok, "tokens_per_s": round(out_tok / dur, 2) if dur else None,
                "truncated": d.get("done_reason") == "length",
            }
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            last_err = str(exc)
            time.sleep(min(30, 2 ** attempt))
    return {"raw": "", "error": last_err, "attempts": 4, "latency_s": None, "prompt_tokens": None,
            "output_tokens": None, "tokens_per_s": None, "truncated": False}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, type=Path, help="normalized dir (eval_sample.jsonl + <source>.jsonl)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--base-url", default="http://127.0.0.1:11434")
    ap.add_argument("--shots", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4, help="parallel requests (match OLLAMA_NUM_PARALLEL)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N rows of this shard")
    ap.add_argument("--think", action="store_true", help="enable thinking mode (default: off)")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    sample = [json.loads(l) for l in (args.data / "eval_sample.jsonl").read_text(encoding="utf-8").splitlines() if l]
    random.Random(0).shuffle(sample)  # fixed order: balanced shards, mixed sources under --limit
    pool_rows = []
    for src in sorted({r["source"] for r in sample}):
        pool_rows += [json.loads(l) for l in (args.data / f"{src}.jsonl").read_text(encoding="utf-8").splitlines() if l]
    pool = ShotPool(pool_rows)

    rows = [r for i, r in enumerate(sample) if i % args.num_shards == args.shard]
    if args.limit:
        rows = rows[: args.limit]
    safe = re.sub(r"[^\w.-]+", "_", args.model)
    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / f"{safe}.shard{args.shard}of{args.num_shards}.jsonl"
    done = set()
    if out_file.exists():
        for line in out_file.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                pass
    todo = [r for r in rows if r["id"] not in done]
    print(f"{args.model}: shard {args.shard}/{args.num_shards}: {len(rows)} rows, "
          f"{len(done)} done, {len(todo)} to run -> {out_file}", flush=True)

    client = httpx.Client(base_url=args.base_url.rstrip("/"), timeout=args.timeout)
    try:
        tag = client.get("/api/tags").json()
        digest = next((m["digest"] for m in tag.get("models", []) if m["name"] == args.model), None)
    except Exception:  # noqa: BLE001 - metadata only
        digest = None
    (args.out / f"{safe}.meta.json").write_text(json.dumps({
        "model": args.model, "digest": digest, "think": args.think, "shots": args.shots,
        "max_tokens": args.max_tokens, "num_ctx": args.num_ctx, "temperature": 0, "seed": 0,
    }, indent=2), encoding="utf-8")

    lock, counter, started = threading.Lock(), [0], time.time()

    def work(row: dict) -> None:
        shots = pool.shots(row, args.shots)
        res = call_model(client, args.model, build_messages(row, shots), args)
        rec = {"id": row["id"], "model": args.model, "shot_ids": [s["id"] for s in shots],
               "sparql_generated": extract_sparql(res["raw"]), **res}
        with lock:
            with out_file.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            counter[0] += 1
            if counter[0] % 25 == 0 or counter[0] == len(todo):
                print(f"  {counter[0]}/{len(todo)}  {time.time() - started:.0f}s", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        list(ex.map(work, todo))
    client.close()
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
