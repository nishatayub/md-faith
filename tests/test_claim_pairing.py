"""Regression tests: a model-based extractor can emit duplicate claim ids; verdicts must still follow their claims."""

import pytest

from mdfaith.claims import Claim, Kind, Label, Verdict
from mdfaith.export import export_run, reverify_run
from mdfaith.extract import extract_claims
from mdfaith.llm import ScriptedClient
from mdfaith.pipeline import build_tasks
from mdfaith.report import annotate
from mdfaith.store import Store


def _dup_claims():
    return [
        Claim("1", "The mean backbone RMSD is 6.0 Å.", Kind.NUMERIC, {"quantity": "rmsd_mean", "value": 6.0}),
        Claim("1", "The mean backbone RMSD is 4.4 Å.", Kind.NUMERIC, {"quantity": "rmsd_mean", "value": 4.4}),
    ]


def test_store_keeps_each_verdict_with_its_own_claim_even_with_duplicate_ids():
    s = Store()
    rid = s.create_run("r")
    verdicts = [Verdict("1", Label.CONTRADICTED, "first"), Verdict("1", Label.SUPPORTED, "second")]
    s.add_explanation(rid, "t", "none", "table_only", "m", "text", _dup_claims(), verdicts)
    (e,) = s.explanations(rid, with_claims=True)
    assert [(c["payload"]["value"], c["label"]) for c in e["claims"]] == [(6.0, "contradicted"), (4.4, "supported")]


def test_store_rejects_mismatched_lengths():
    s = Store()
    rid = s.create_run("r")
    with pytest.raises(ValueError):
        s.add_explanation(rid, "t", "none", "table_only", "m", "x", _dup_claims(), [Verdict("1", Label.SUPPORTED)])


def test_extractor_renumbers_duplicate_ids():
    reply = (
        '[{"id":"1","text":"The mean backbone RMSD is 6.0 Å.","kind":"numeric","payload":{"quantity":"rmsd_mean","value":6.0}},'
        '{"id":"1","text":"The mean backbone RMSD is 4.4 Å.","kind":"numeric","payload":{"quantity":"rmsd_mean","value":4.4}}]'
    )
    claims = extract_claims(ScriptedClient([reply]), "x")
    assert [c.id for c in claims] == ["c1", "c2"]


def test_annotate_pairs_by_position_with_duplicate_ids():
    tasks = build_tasks("adk_dims", [{"kind": "none"}])
    gt = tasks[0].ground_truth
    text = "The mean backbone RMSD is 6.0 Å. The mean backbone RMSD is 4.4 Å."
    ann = annotate(text, _dup_claims(), [Verdict("1", Label.CONTRADICTED), Verdict("1", Label.SUPPORTED)], gt)
    assert [a.status for a in ann] == ["contradicted", "supported"]


def test_reverify_repairs_a_corrupted_label_and_export_is_consistent(tmp_path):
    tasks = build_tasks("adk_dims", [{"kind": "none"}])
    s = Store()
    rid = s.create_run("r", {"tasks": ["adk_dims:none"], "backends": ["x"], "conditions": ["table_only"], "seeds": [0]})
    true_mean = tasks[0].ground_truth.scalars["rmsd_mean"]
    good = Claim("c1", "The mean backbone RMSD is right.", Kind.NUMERIC, {"quantity": "rmsd_mean", "value": true_mean})
    s.add_explanation(
        rid, "adk_dims:none", "none", "table_only", "x", "t", [good], [Verdict("c1", Label.CONTRADICTED, "wrong")]
    )
    res = reverify_run(s, rid, {t.task_id: t for t in tasks})
    assert res["changed"] == 1 and res["changes"][0]["now"] == "supported"
    assert reverify_run(s, rid, {t.task_id: t for t in tasks})["changed"] == 0
    summary = export_run(s, rid, tmp_path)
    assert summary["verdicts"] == {"supported": 1} and (tmp_path / "claims.csv").exists()
