"""n4c-decision-broker command line interface."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from n4c_decision_broker import __version__
from n4c_decision_broker.audit import AuditLog
from n4c_decision_broker.broker import ModelBroker
from n4c_decision_broker.cache import DecisionCache
from n4c_decision_broker.client import OllamaClient as JevClient
from n4c_decision_broker.fallback import HoldBackend, LLMFallbackBackend
from n4c_decision_broker.models import QuestionPolicy, QuestionSpec
from n4c_decision_broker.testing import FakeJev, demo_questions, demo_records


def _split_head(spec: str, flag: str) -> tuple[str, str, list[str]]:
    """'id=instruction|rest1|rest2' -> (id, instruction, [rest...])."""
    qid, eq, rest = spec.partition("=")
    if not eq:
        raise SystemExit(f"{flag} expects id=instruction|... (got {spec!r})")
    parts = rest.split("|")
    if not parts[0].strip():
        raise SystemExit(f"{flag} is missing its instruction (got {spec!r})")
    return qid.strip(), parts[0].strip(), [p.strip() for p in parts[1:] if p.strip()]


def _parse_choice(spec: str) -> QuestionSpec:
    qid, instruction, parts = _split_head(spec, "--choice")
    if not parts:
        raise SystemExit(f"--choice expects at least one option: id=instruction|key:desc (got {spec!r})")
    criteria: dict[str, str] = {}
    for part in parts:
        key, _, desc = part.partition(":")
        if not key:
            raise SystemExit(f"malformed choice option {part!r} in {spec!r}")
        criteria[key.strip()] = desc.strip()
    return QuestionSpec(id=qid, type="choice", instructions=instruction, criteria=criteria)


def _parse_score(spec: str) -> QuestionSpec:
    qid, instruction, levels = _split_head(spec, "--score")
    if not levels:
        raise SystemExit(f"--score expects levels: id=instruction|level1|level2 (got {spec!r})")
    return QuestionSpec(id=qid, type="score", instructions=instruction, criteria=levels)


def _parse_noul(spec: str) -> QuestionSpec:
    qid, _, rest = spec.partition("=")
    if not rest:
        raise SystemExit(f"--noul expects id=instruction (got {spec!r})")
    return QuestionSpec(id=qid.strip(), type="noul", instructions=rest.strip())


def _load_questions(args: argparse.Namespace) -> list[QuestionSpec]:
    questions: list[QuestionSpec] = []
    if args.questions_file:
        data = json.loads(Path(args.questions_file).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = list(data.values())
        questions.extend(QuestionSpec.model_validate(q) for q in data)
    for spec in args.choice or []:
        questions.append(_parse_choice(spec))
    for spec in args.score or []:
        questions.append(_parse_score(spec))
    for spec in args.noul or []:
        questions.append(_parse_noul(spec))
    if not questions:
        raise SystemExit("no questions provided (use --questions-file, --choice, --score or --noul)")
    return questions


def _load_policy(args: argparse.Namespace) -> QuestionPolicy:
    policy = QuestionPolicy()
    if getattr(args, "auto_threshold", None) is not None:
        policy = policy.model_copy(update={"auto_threshold": args.auto_threshold})
    if getattr(args, "review_threshold", None) is not None:
        policy = policy.model_copy(update={"review_threshold": args.review_threshold})
    return policy


def _build_broker(args: argparse.Namespace, offline: bool = False) -> ModelBroker:
    audit = AuditLog(Path(args.audit_log))
    cache = DecisionCache(path=Path(args.cache_file)) if not args.no_cache else None
    if offline:
        client: Any = FakeJev(seed="n4c-decision-broker-demo")
        fallback: Any = HoldBackend()
    else:
        client = JevClient(model=args.model)
        fallback = LLMFallbackBackend() if args.llm_fallback else HoldBackend()
    return ModelBroker(
        client=client,
        fallback=fallback,
        cache=cache,
        audit=audit,
        policy=_load_policy(args),
    )


def _print_response(resp: Any, as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(resp.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return
    print(f"request {resp.request_id}  state_sha256={resp.state_sha256[:12]}…  "
          f"{resp.latency_ms} ms  cost≈${resp.cost_estimate_usd or 0}")
    for a in resp.answers:
        value = a.choice or (a.noul_label if a.noul is not None else (a.score if a.score is not None else ""))
        conf = f" conf={a.confidence:.2f}" if a.confidence is not None else ""
        flag = "" if a.status.value == "decided" else "  ← " + a.status.value.upper()
        print(f"  {a.question_id:<16} {str(value):<14} src={a.source:<14}{conf}{flag}")
        if a.error:
            print(f"    error: {a.error}")


def cmd_decide(args: argparse.Namespace) -> int:
    broker = _build_broker(args)
    try:
        resp = broker.decide(state=args.state, questions=_load_questions(args), use_cache=not args.no_cache)
    finally:
        broker.close()
    _print_response(resp, as_json=args.json)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    broker = _build_broker(args, offline=True)
    questions = demo_questions()
    records = demo_records()
    print(f"offline demo — FakeJev, {len(records)} records, {len(questions)} questions each\n")
    results = broker.batch(records, questions)
    for i, resp in enumerate(results, start=1):
        print(f"record {i}: {records[i-1]!r}")
        for a in resp.answers:
            value = a.choice or (a.noul_label if a.noul is not None else (a.score if a.score is not None else ""))
            conf = f"conf={a.confidence:.2f}  " if a.confidence is not None else "conf=-        "
            probs = f"p={a.probabilities}" if a.probabilities else ""
            print(f"   - {a.question_id:<12} -> {str(value):<14} {a.status.value:<10} {conf}{probs}")
        print()
    print("metrics:", json.dumps(broker.metrics()))
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    records = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if isinstance(records, dict):
        records = records.get("records", [records])
    broker = _build_broker(args)
    try:
        results = broker.batch(records, _load_questions(args))
    finally:
        broker.close()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r.model_dump(mode="json"), ensure_ascii=False) + "\n")
    print(f"wrote {len(results)} decisions to {out}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
        from n4c_decision_broker.server import create_app
    except ImportError as exc:
        raise SystemExit("server extras missing — install with: pip install 'n4c-decision-broker[server]'") from exc
    broker = _build_broker(args)
    uvicorn.run(create_app(broker), host=args.host, port=args.port, log_level="info")
    return 0


def cmd_metrics(args: argparse.Namespace) -> int:
    broker = _build_broker(args)
    print(json.dumps(broker.metrics(), ensure_ascii=False, indent=2))
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    log = AuditLog(Path(args.audit_log))
    entries = log.recent(limit=args.limit)
    if not entries:
        print("(no audit entries yet)")
        return 0
    for e in entries:
        print(json.dumps(e, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="n4c-decision-broker", description="Confidence-governed broker over local Ollama System One models (e.g. nimble)")
    parser.add_argument("--version", action="version", version=f"n4c-decision-broker {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--questions-file", type=Path, help="JSON list/dict of QuestionSpec objects")
        p.add_argument("--choice", action="append", metavar="id=instruction|opt:desc|opt:desc", help="add a choice question")
        p.add_argument("--score", action="append", metavar="id=instruction|level1|level2", help="add a score question")
        p.add_argument("--noul", action="append", metavar="id=instruction", help="add a noul question")
        p.add_argument("--auto-threshold", type=float, default=None)
        p.add_argument("--review-threshold", type=float, default=None)
        p.add_argument("--model", default=None, help="Ollama model (default: nimble)")
        p.add_argument("--llm-fallback", action="store_true", help="use FALLBACK_LLM_* env for escalations")
        p.add_argument("--audit-log", default="audit/decisions.jsonl")
        p.add_argument("--cache-file", default="audit/cache.json")
        p.add_argument("--no-cache", action="store_true")

    p_decide = sub.add_parser("decide", help="ask a single governed decision (live Jev call)")
    add_common(p_decide)
    p_decide.add_argument("--state", required=True)
    p_decide.add_argument("--json", action="store_true", help="JSON output")
    p_decide.set_defaults(func=cmd_decide)

    p_demo = sub.add_parser("demo", help="offline demo with a deterministic FakeJev (no API key)")
    add_common(p_demo)
    p_demo.set_defaults(func=cmd_demo)

    p_batch = sub.add_parser("batch", help="enrich a JSON list of records")
    add_common(p_batch)
    p_batch.add_argument("--input", required=True, type=Path)
    p_batch.add_argument("--out", default="audit/batch.jsonl", type=Path)
    p_batch.set_defaults(func=cmd_batch)

    p_serve = sub.add_parser("serve", help="run the REST server")
    add_common(p_serve)
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8787)
    p_serve.set_defaults(func=cmd_serve)

    p_metrics = sub.add_parser("metrics", help="print broker metrics")
    add_common(p_metrics)
    p_metrics.set_defaults(func=cmd_metrics)

    p_audit = sub.add_parser("audit", help="show recent audit entries")
    p_audit.add_argument("--audit-log", default="audit/decisions.jsonl")
    p_audit.add_argument("--limit", type=int, default=20)
    p_audit.set_defaults(func=cmd_audit)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
