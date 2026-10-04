import pytest

from mdfaith.claims import Claim, Kind, Label, Verdict
from mdfaith.store import Store


@pytest.fixture()
def store():
    s = Store(":memory:")
    yield s
    s.close()


def _add(store, run, label=Label.SUPPORTED, model="m1", condition="tool_agent", seed=0):
    claims = [Claim("1", "mean RMSD is 4 A", Kind.NUMERIC, {"quantity": "rmsd_mean", "value": 4.0})]
    verdicts = [Verdict("1", label, "detail")]
    return store.add_explanation(
        run, "adk_dims:none", "none", condition, model, "text", claims, verdicts, seed=seed, n_tool_calls=2
    )


def test_run_lifecycle(store):
    rid = store.create_run("demo", {"a": 1}, is_demo=True)
    run = store.get_run(rid)
    assert run["is_demo"] is True and run["config"] == {"a": 1} and run["status"] == "running"
    store.finish_run(rid)
    assert store.get_run(rid)["status"] == "done"
    assert [r["id"] for r in store.list_runs()] == [rid]


def test_explanations_roundtrip_with_claims(store):
    rid = store.create_run("r")
    _add(store, rid, Label.CONTRADICTED)
    (e,) = store.explanations(rid, with_claims=True)
    assert e["model"] == "m1" and e["n_tool_calls"] == 2
    assert e["claims"][0]["label"] == "contradicted" and e["claims"][0]["payload"]["quantity"] == "rmsd_mean"
    assert store.get_run(rid)["n_explanations"] == 1


def test_per_explanation_counts_feed_metrics(store):
    rid = store.create_run("r")
    _add(store, rid, Label.SUPPORTED)
    _add(store, rid, Label.ARTIFACT_MISREAD)
    rows = store.per_explanation_counts(rid)
    assert [r["supported"] for r in rows] == [1, 0]
    assert [r["wrong"] for r in rows] == [0, 1]


def test_delete_run_cascades(store):
    rid = store.create_run("r")
    _add(store, rid)
    store.delete_run(rid)
    assert store.list_runs() == []
    assert store._db.execute("SELECT COUNT(*) FROM claims").fetchone()[0] == 0


def test_persists_to_disk(tmp_path):
    p = str(tmp_path / "x.db")
    s = Store(p)
    rid = s.create_run("r")
    _add(s, rid)
    s.close()
    s2 = Store(p)
    assert s2.get_run(rid)["n_explanations"] == 1
    s2.close()
