"""CLI: argument parsing, commands and the serve safety guard (offline)."""
import json

import pytest

from n4c_decision_broker import cli
from n4c_decision_broker.testing import FakeJev

QUESTIONS = ["--choice", "lang=Language|de:German|en:English", "--noul", "open=No rights issues"]


@pytest.fixture
def fake_model(monkeypatch):
    monkeypatch.setattr(cli, "JevClient", lambda model=None: FakeJev(seed="cli"))


def common(tmp_path):
    return ["--audit-log", str(tmp_path / "audit.jsonl"), "--cache-file", str(tmp_path / "cache.json")]


# -- question parsing ------------------------------------------------------------
def test_parse_choice_score_noul():
    c = cli._parse_choice("p=Period|early:1400-1800|modern:1800-1945")
    assert c.type.value == "choice" and c.criteria == {"early": "1400-1800", "modern": "1800-1945"}
    s = cli._parse_score("q=Quality|bad|ok|good")
    assert s.criteria == ["bad", "ok", "good"]
    n = cli._parse_noul("o=Is open")
    assert n.type.value == "noul" and n.instructions == "Is open"


@pytest.mark.parametrize("fn,spec", [
    (cli._parse_choice, "noequals"), (cli._parse_choice, "p=Period"),
    (cli._parse_score, "q=Quality"), (cli._parse_noul, "o"), (cli._parse_choice, "p=|a:b"),
])
def test_parse_errors_exit(fn, spec):
    with pytest.raises(SystemExit):
        fn(spec)


def test_questions_file_and_flags_combined(tmp_path):
    f = tmp_path / "q.json"
    f.write_text(json.dumps([{"id": "n", "type": "noul", "instructions": "x"}]))
    args = cli.build_parser().parse_args(["decide", "--state", "s", "--questions-file", str(f), *QUESTIONS])
    assert [q.id for q in cli._load_questions(args)] == ["n", "lang", "open"]


def test_no_questions_exits():
    args = cli.build_parser().parse_args(["decide", "--state", "s"])
    with pytest.raises(SystemExit):
        cli._load_questions(args)


# -- commands --------------------------------------------------------------------
def test_demo_runs_offline(tmp_path, capsys):
    assert cli.main(["demo", *common(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "offline demo" in out and "metrics:" in out


def test_decide_text_and_json(tmp_path, capsys, fake_model):
    assert cli.main(["decide", "--state", "Kupferstich", *QUESTIONS, *common(tmp_path)]) == 0
    assert "lang" in capsys.readouterr().out
    assert cli.main(["decide", "--state", "Kupferstich", "--json", "--no-cache", *QUESTIONS, *common(tmp_path)]) == 0
    data = json.loads(capsys.readouterr().out)
    assert [a["question_id"] for a in data["answers"]] == ["lang", "open"]


def test_batch_writes_jsonl_in_order(tmp_path, fake_model):
    inp = tmp_path / "records.json"
    inp.write_text(json.dumps(["a", "b", "c"]))
    out = tmp_path / "out" / "batch.jsonl"
    assert cli.main(["batch", "--input", str(inp), "--out", str(out), "--workers", "3",
                     *QUESTIONS, *common(tmp_path)]) == 0
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(lines) == 3
    from n4c_decision_broker.models import state_sha256
    assert [r["state_sha256"] for r in lines] == [state_sha256(s) for s in "abc"]


def test_batch_accepts_records_object(tmp_path, fake_model):
    inp = tmp_path / "records.json"
    inp.write_text(json.dumps({"records": ["only"]}))
    out = tmp_path / "b.jsonl"
    cli.main(["batch", "--input", str(inp), "--out", str(out), *QUESTIONS, *common(tmp_path)])
    assert len(out.read_text().splitlines()) == 1


def test_audit_and_metrics(tmp_path, capsys, fake_model):
    assert cli.main(["audit", "--audit-log", str(tmp_path / "audit.jsonl")]) == 0
    assert "no audit entries" in capsys.readouterr().out
    cli.main(["decide", "--state", "s", *QUESTIONS, *common(tmp_path)])
    capsys.readouterr()
    cli.main(["audit", "--audit-log", str(tmp_path / "audit.jsonl"), "--limit", "1"])
    assert json.loads(capsys.readouterr().out.strip())["answers"]
    assert cli.main(["metrics", *common(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["fallback"] == "hold"


def test_calibrate_roundtrip_offline(tmp_path, capsys, fake_model):
    labeled = tmp_path / "l.json"
    labeled.write_text(json.dumps(
        [{"state": f"r{i}", "labels": {"lang": "de", "open": True}} for i in range(8)]))
    out = tmp_path / "cal.json"
    obs = tmp_path / "obs.json"
    assert cli.main(["calibrate", "--labeled", str(labeled), "--out", str(out),
                     "--obs-out", str(obs), *QUESTIONS, *common(tmp_path)]) == 0
    assert json.loads(out.read_text())["n_observations"] == 16
    # recompute without any model
    assert cli.main(["calibrate", "--from-observations", str(obs), "--out", str(out)]) == 0
    with pytest.raises(SystemExit):
        cli.main(["calibrate", *QUESTIONS])  # neither --labeled nor --from-observations


# -- serve guard -----------------------------------------------------------------
def test_serve_refuses_public_bind_without_keys(tmp_path, monkeypatch):
    pytest.importorskip("uvicorn")
    monkeypatch.delenv("N4C_BROKER_API_KEYS", raising=False)
    with pytest.raises(SystemExit, match="without authentication"):
        cli.main(["serve", "--host", "0.0.0.0", *common(tmp_path)])


def test_serve_passes_keys_to_app(tmp_path, monkeypatch, fake_model):
    uvicorn = pytest.importorskip("uvicorn")
    monkeypatch.setenv("N4C_BROKER_API_KEYS", "k1,k2")
    seen = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: seen.update(kw=kw, app=app))
    assert cli.main(["serve", "--host", "0.0.0.0", "--port", "9000", *common(tmp_path)]) == 0
    assert seen["kw"]["host"] == "0.0.0.0" and seen["kw"]["port"] == 9000
    from fastapi.testclient import TestClient
    c = TestClient(seen["app"])
    assert c.get("/v1/broker/metrics").status_code == 401
    assert c.get("/v1/broker/metrics", headers={"X-API-Key": "k2"}).status_code == 200
