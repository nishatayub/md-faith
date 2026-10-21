import pytest

from mdfaith.artifacts import DEFAULT_SPECS
from mdfaith.claims import Kind
from mdfaith.demo import seed_demo
from mdfaith.pipeline import build_tasks
from mdfaith.reextract import compare_runs, reextract_run
from mdfaith.store import Store


@pytest.fixture(scope="module")
def demo():
    tasks = build_tasks("adk_dims", [DEFAULT_SPECS[k] for k in ("none", "pbc_split")])
    s = Store()
    return s, seed_demo(s, seeds=2, tasks=tasks)


class _Rule:
    name = "rule"

    def extract(self, text):
        from mdfaith.extract_rules import RuleExtractor

        return RuleExtractor().extract(text)


class _OnlyNumeric:
    name = "numeric-only"

    def extract(self, text):
        from mdfaith.extract_rules import RuleExtractor

        return [c for c in RuleExtractor().extract(text) if c.kind is Kind.NUMERIC]


def test_reextract_copies_explanations_and_reverifies(demo):
    s, rid = demo
    new = reextract_run(s, rid, _Rule())
    a, b = s.get_run(rid), s.get_run(new)
    assert b["n_explanations"] == a["n_explanations"] and b["is_demo"] is True and b["status"] == "done"
    assert b["config"]["source_run"] == rid and b["config"]["extractor"] == "rule"
    assert s.per_explanation_counts(rid) and sorted(
        (r["supported"], r["wrong"]) for r in s.per_explanation_counts(rid)
    ) == sorted((r["supported"], r["wrong"]) for r in s.per_explanation_counts(new))


def test_compare_identical_and_degraded_extractors(demo):
    s, rid = demo
    same = reextract_run(s, rid, _Rule())
    res = compare_runs(s, rid, same)
    assert res["jaccard"] == 1.0 and res["verdict_agreement_on_shared"] == 1.0
    part = reextract_run(s, rid, _OnlyNumeric())
    res2 = compare_runs(s, rid, part)
    assert 0 < res2["jaccard"] < 1 and res2["claims_b"] < res2["claims_a"]
    assert res2["by_kind"]["numeric"]["a"] == res2["by_kind"]["numeric"]["b"]


def test_reextract_unknown_run_and_failure_marks_failed(demo):
    s, rid = demo
    with pytest.raises(KeyError):
        reextract_run(s, "nope", _Rule())

    class Boom:
        name = "boom"

        def extract(self, text):
            raise RuntimeError("x")

    before = len(s.list_runs())
    with pytest.raises(RuntimeError):
        reextract_run(s, rid, Boom())
    assert s.list_runs()[0]["status"] == "failed" and len(s.list_runs()) == before + 1
