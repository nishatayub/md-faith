import pytest
from fastapi.testclient import TestClient

from mdfaith.api import create_app


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app(":memory:"))


def test_health_and_meta(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["demo_present"] is False
    m = client.get("/api/meta").json()
    assert {a["kind"] for a in m["artifacts"]} == {"none", "pbc_split", "shuffle_frames", "rigid_jitter"}
    assert [c["name"] for c in m["conditions"]] == ["image_only", "table_only", "tool_agent", "qc_gated"]
    assert m["real_models_enabled"] is False


def test_tasks_and_detail_with_qc(client):
    assert len(client.get("/api/tasks").json()) == 4
    d = client.get("/api/tasks/adk_dims:pbc_split").json()
    assert len(d["rmsd"]) == d["n_frames"] == 98 and d["artifact"]["frames"] == [30, 31, 32]
    assert d["qc"]["passed"] is False and "broken_backbone" in [f["code"] for f in d["qc"]["flags"]]
    clean = client.get("/api/tasks/adk_dims:none/qc").json()
    assert clean["passed"] is True
    assert client.get("/api/tasks/nope:none").status_code == 404
    assert client.get("/api/tasks/garbage").status_code == 404


def test_verify_playground_samples(client):
    samples = client.get("/api/tasks/adk_dims:none/samples").json()
    by = {s["label"]: s["text"] for s in samples}
    ok = client.post("/api/verify", json={"task_id": "adk_dims:none", "text": by["Faithful report"]}).json()
    bad = client.post(
        "/api/verify", json={"task_id": "adk_dims:none", "text": by["Sloppy numbers and residues"]}
    ).json()
    over = client.post("/api/verify", json={"task_id": "adk_dims:none", "text": by["Over-interpretation"]}).json()
    assert ok["summary"]["contradicted"] == 0 and ok["summary"]["supported"] >= 3
    assert bad["summary"]["contradicted"] >= 2
    assert over["summary"]["unsupported"] >= 1
    assert all("status" in s for s in ok["sentences"])


def test_verify_flags_artifact_misread(client):
    text = "RMSD peaks at frame 30, indicating a large conformational change."
    r = client.post("/api/verify", json={"task_id": "adk_dims:pbc_split", "text": text}).json()
    assert r["summary"]["artifact"] == 1


def test_verify_validates_input(client):
    assert client.post("/api/verify", json={"task_id": "adk_dims:none", "text": ""}).status_code == 422


def test_runs_lifecycle_and_summary(client):
    r = client.post(
        "/api/runs",
        json={
            "name": "t",
            "models": ["sim-careful"],
            "conditions": ["image_only", "qc_gated"],
            "artifacts": ["none", "pbc_split"],
            "seeds": 2,
        },
    )
    assert r.status_code == 201
    run = r.json()
    assert run["is_demo"] is True and run["n_explanations"] == 1 * 2 * 2 * 2
    s = client.get(f"/api/runs/{run['id']}/summary?by=condition").json()
    assert {x["condition"] for x in s["rows"]} == {"image_only", "qc_gated"}
    ex = client.get(f"/api/runs/{run['id']}/explanations?condition=image_only&limit=2").json()
    assert ex["total"] == 4 and len(ex["items"]) == 2
    rep = client.get(f"/api/explanations/{ex['items'][0]['explanation_id']}").json()
    assert rep["is_demo"] is True and rep["sentences"]
    assert client.delete(f"/api/runs/{run['id']}").status_code == 204
    assert client.get(f"/api/runs/{run['id']}").status_code == 404


def test_real_models_blocked_by_default(client):
    r = client.post("/api/runs", json={"models": ["anthropic:claude-opus-5-5"], "seeds": 1})
    assert r.status_code == 403


def test_bad_run_requests(client):
    assert client.post("/api/runs", json={"models": ["sim-nope"]}).status_code == 422
    assert client.post("/api/runs", json={"conditions": ["bogus"]}).status_code == 422
    assert client.post("/api/runs", json={"seeds": 999}).status_code == 422
    assert client.get("/api/runs/x/summary").status_code == 404


def test_demo_seed_then_health_reports_it(client):
    r = client.post("/api/demo/seed", json={"seeds": 2})
    assert r.status_code == 201 and r.json()["name"].startswith("DEMO")
    assert client.get("/api/health").json()["demo_present"] is True
