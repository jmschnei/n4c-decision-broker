"""Evaluation tooling (tools/eval): pure helpers, no model or network."""
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS / "eval"))
generate = pytest.importorskip("generate")
score = pytest.importorskip("score")


def test_extract_sparql_variants():
    assert generate.extract_sparql("Sure:\n```sparql\nSELECT * WHERE {?s ?p ?o}\n```\nbye") == "SELECT * WHERE {?s ?p ?o}"
    assert generate.extract_sparql("<think>hmm</think>SELECT ?x WHERE {?x a ?y}").startswith("SELECT ?x")
    assert generate.extract_sparql("Here you go: ASK {?s ?p ?o}") == "ASK {?s ?p ?o}"
    assert generate.extract_sparql("") == ""


def test_compact_prefixes_keeps_only_used():
    q = "PREFIX wd: <http://w/> PREFIX wdt: <http://t/> SELECT ?x WHERE { wd:Q1 wdt:P1 ?x }"
    out = generate.compact_prefixes(q)
    assert out.count("PREFIX") == 2
    q2 = "PREFIX wd: <http://w/> PREFIX unused: <http://u/> SELECT ?x WHERE { wd:Q1 ?p ?x }"
    assert "unused" not in generate.compact_prefixes(q2)


def _row(i, source, group, q, sparql, lang="en"):
    return {"id": f"{source}:{i}", "source": source, "group": f"{source}:{group}",
            "question": q, "sparql": sparql, "language": lang}


def test_shots_never_leak_own_group_or_query():
    rows = [_row(1, "qald10", "g1", "who wrote hamlet", "SELECT ?a WHERE {wd:Q1 wdt:P50 ?a}"),
            _row(2, "qald10", "g1", "who wrote hamlet again", "SELECT ?a WHERE {wd:Q1 wdt:P50 ?a}"),
            _row(3, "qald10", "g2", "who wrote macbeth", "SELECT ?a WHERE {wd:Q2 wdt:P50 ?a}"),
            _row(4, "qald10", "g3", "capital of france", "SELECT ?c WHERE {wd:Q3 wdt:P36 ?c}"),
            _row(5, "qald10", "g4", "who wrote hamlet", "SELECT ?a WHERE {wd:Q1 wdt:P50 ?a}", lang="de")]
    pool = generate.ShotPool(rows)
    ids = [s["id"] for s in pool.shots(rows[0], 3)]
    assert "qald10:1" not in ids and "qald10:2" not in ids  # same group
    assert "qald10:5" not in ids                            # other language
    assert ids[0] == "qald10:3"                             # most similar first


def test_spider_shots_may_share_the_kg_but_not_the_query():
    rows = [_row(1, "spider4sparql", "kgA", "how many singers", "select (count(*) as ?n) where {?s a :singer}"),
            _row(2, "spider4sparql", "kgA", "list singer names", "select ?n where {?s :singer#name ?n}"),
            _row(3, "spider4sparql", "kgA", "count the singers", "select (count(*) as ?n) where {?s a :singer}")]
    ids = [s["id"] for s in generate.ShotPool(rows).shots(rows[0], 3)]
    assert ids == ["spider4sparql:2"]


def test_prf_and_iri_set():
    assert score.prf({"a", "b"}, {"a", "b"}) == (1.0, 1.0, 1.0)
    p, r, f = score.prf({"a"}, {"a", "b"})
    assert (p, r) == (1.0, 0.5) and f == pytest.approx(0.6667, abs=1e-3)
    a = score.iri_set("PREFIX wd: <http://www.wikidata.org/entity/> SELECT ?x WHERE { wd:Q42 ?p ?x }")
    b = score.iri_set("SELECT ?x WHERE { <http://www.wikidata.org/entity/Q42> ?p ?x }")
    assert a == b == {"http://www.wikidata.org/entity/Q42"}


def test_offline_scores_perfect_and_empty():
    gold = "SELECT DISTINCT ?t WHERE { ?s <http://x/title> ?t . }"
    row = {"source": "golem", "sparql": gold}
    good = score.offline_scores(row, gold)
    assert good["exact_match"] and good["skeleton_match"] and good["iri_f1"] == 1.0
    empty = score.offline_scores(row, "")
    assert not empty["syntax_valid"] and not empty["exact_match"] and empty["iri_f1"] == 0.0


def test_normalize_results_and_aggregate():
    r = score.normalize_results({"head": {"vars": ["x"]}, "results": {"bindings": [
        {"x": {"value": "b"}}, {"x": {"value": "a"}}]}})
    assert r["rows"] == ['["a"]', '["b"]']
    assert score.normalize_results({"boolean": True}) == {"bool": True}
    items = [{"model": "m", "syntax_valid": True, "exact_match": False, "skeleton_match": False,
              "iri_f1": 0.5, "feature_match": True, "latency_s": 1.0, "tokens_per_s": 10.0},
             {"model": "m", "syntax_valid": False, "exact_match": False, "skeleton_match": False,
              "iri_f1": 0.0, "feature_match": False, "latency_s": 3.0, "tokens_per_s": None,
              "error": "x"}]
    agg = score.aggregate(items, ("model",))["m"]
    assert agg["n"] == 2 and agg["syntax_valid"] == 0.5 and agg["errors"] == 1 and agg["exec_n"] == 0
