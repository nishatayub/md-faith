import pytest

from mdfaith.aggregate import aggregate
from mdfaith.artifacts import DEFAULT_SPECS
from mdfaith.demo import DEMO_NAME, seed_demo
from mdfaith.extract import LLMExtractor
from mdfaith.llm import ScriptedClient
from mdfaith.pipeline import build_tasks
from mdfaith.runner import Backend, make_backend, run_experiment
from mdfaith.store import Store


@pytest.fixture(scope="module")
def tasks():
    return build_tasks("adk_dims", [DEFAULT_SPECS["none"], DEFAULT_SPECS["pbc_split"]])


def test_grid_size_demo_flag_and_progress(tasks):
    s = Store()
    seen = []
    rid = run_experiment(
        s,
        tasks,
        [make_backend("sim-careful"), make_backend("sim-hasty")],
        ("image_only", "tool_agent"),
        (0, 1),
        progress=lambda d, t: seen.append((d, t)),
    )
    run = s.get_run(rid)
    assert run["n_explanations"] == 2 * 2 * 2 * 2 and run["status"] == "done" and run["is_demo"] is True
    assert seen[-1] == (16, 16)


def test_real_backend_run_is_not_flagged_demo(tasks):
    s = Store()
    reply = '[{"id":"1","text":"The mean backbone RMSD is 1.0 Å.","kind":"numeric","payload":{"quantity":"rmsd_mean","value":1.0}}]'
    # explanation call, then extraction call
    client = ScriptedClient(["The mean backbone RMSD is 1.0 Å.", reply])
    backend = Backend("scripted", client=client)
    rid = run_experiment(s, tasks[:1], [backend], ("table_only",), (0,), extractor=LLMExtractor(client))
    assert s.get_run(rid)["is_demo"] is False
    (e,) = s.explanations(rid, with_claims=True)
    assert e["claims"][0]["label"] == "contradicted"


def test_failed_run_is_marked_failed(tasks):
    s = Store()
    bad = Backend("boom", client=ScriptedClient([]))
    with pytest.raises(IndexError):
        run_experiment(s, tasks[:1], [bad], ("table_only",), (0,))
    assert s.list_runs()[0]["status"] == "failed"


def test_unknown_backend():
    with pytest.raises(ValueError):
        make_backend("gpt-whatever")


def test_aggregate_and_demo_story(tasks):
    s = Store()
    rid = seed_demo(s, seeds=6, tasks=tasks)
    assert s.get_run(rid)["name"] == DEMO_NAME
    rows = aggregate(s.per_explanation_counts(rid), by=("condition",), n_boot=200)
    by = {r["condition"]: r for r in rows}
    assert by["image_only"]["hallucination_rate"] > by["qc_gated"]["hallucination_rate"]
    assert 0 <= by["tool_agent"]["ci_lo"] <= by["tool_agent"]["hallucination_rate"] <= by["tool_agent"]["ci_hi"] <= 1
    assert by["qc_gated"]["artifact_misread_share"] < by["image_only"]["artifact_misread_share"]
    two = aggregate(s.per_explanation_counts(rid), by=("condition", "model"), n_boot=50)
    assert len(two) == 4 * 3


def test_resume_skips_stored_cells(tasks):
    s = Store()
    first = run_experiment(s, tasks[:1], [make_backend("sim-careful")], ("table_only",), (0, 1))
    n = s.get_run(first)["n_explanations"]
    again = run_experiment(s, tasks[:1], [make_backend("sim-careful")], ("table_only",), (0, 1, 2), resume=first)
    assert again == first and s.get_run(first)["n_explanations"] == n + 1
