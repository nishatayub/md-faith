"""REST API (FastAPI) behind the MD-Faith web app.

Run with `mdfaith serve`. Optional dependency group: `pip install -e ".[api]"`.

Security notes: simulated runs only by default (`MDFAITH_ALLOW_REAL=1` enables real-model backends and therefore spends
API credit); the API never stores or returns credentials; request sizes are bounded.
"""

from __future__ import annotations

import math
import os
import threading
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__, artifacts
from .aggregate import aggregate
from .claims import Claim, Kind, Label, Verdict
from .conditions.base import Task
from .data import SYSTEMS, load_system
from .demo import seed_demo
from .extract_rules import PATTERNS, RuleExtractor
from .groundtruth import compute
from .qc import run_qc
from .report import annotate, summary, to_dicts
from .runner import ALL_CONDITIONS, make_backend, run_experiment
from .simulated import PERSONAS
from .store import Store
from .verify import verify_all

ARTIFACT_INFO = {
    "none": "Unmodified trajectory.",
    "pbc_split": "A domain (residues 1-60) is shifted by one box length in frames 30-32, as if wrapped across the "
    "periodic boundary. Produces false RMSD spikes and stretched backbone bonds.",
    "shuffle_frames": "Frame order randomly permuted; time-ordered claims become meaningless.",
    "rigid_jitter": "Random rigid-body motion applied to every frame; the trajectory is not centred or fitted.",
}
CONDITION_INFO = {
    "image_only": "Vision model sees RMSD / RMSF / contact-map plots only.",
    "table_only": "Same data as downsampled numeric tables.",
    "tool_agent": "Model calls analysis tools on the trajectory, then explains.",
    "qc_gated": "Tool agent that is shown the automated QC report first.",
}


def clean(x):
    """Make numpy / NaN values JSON-safe."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, float | np.floating):
        return None if not math.isfinite(float(x)) else float(x)
    return x


class VerifyRequest(BaseModel):
    task_id: str = "adk_dims:none"
    text: str = Field(min_length=1, max_length=20000)


class RunRequest(BaseModel):
    name: str = Field("web run", max_length=120)
    models: list[str] = Field(default_factory=lambda: list(PERSONAS))
    conditions: list[str] = Field(default_factory=lambda: list(ALL_CONDITIONS))
    artifacts: list[str] = Field(default_factory=lambda: list(ARTIFACT_INFO))
    seeds: int = Field(4, ge=1, le=20)


class DemoRequest(BaseModel):
    seeds: int = Field(8, ge=1, le=20)


class TaskCache:
    def __init__(self):
        self._tasks: dict[str, Task] = {}
        self._qc: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._bases: dict[str, object] = {}

    def get(self, task_id: str) -> Task:
        with self._lock:
            if task_id in self._tasks:
                return self._tasks[task_id]
            try:
                system, kind = task_id.split(":", 1)
            except ValueError:
                raise HTTPException(404, f"bad task id {task_id!r}") from None
            if system not in SYSTEMS or kind not in artifacts.DEFAULT_SPECS:
                raise HTTPException(404, f"unknown task {task_id!r}")
            base = self._bases.setdefault(system, load_system(system))
            u, rec = artifacts.build_default(base, kind)
            task = Task(task_id, system, u, compute(u, system), rec)
            self._tasks[task_id] = task
            return task

    def qc(self, task_id: str) -> dict:
        task = self.get(task_id)
        with self._lock:
            if task_id not in self._qc:
                self._qc[task_id] = run_qc(task.universe).to_dict()
            return self._qc[task_id]


def _claims_and_verdicts(row_claims: list[dict]):
    claims = [Claim(c["claim_id"], c["text"], Kind(c["kind"]), c["payload"]) for c in row_claims]
    verdicts = [Verdict(c["claim_id"], Label(c["label"]), c["detail"]) for c in row_claims]
    return claims, verdicts


def sample_explanations(task: Task) -> list[dict]:
    """Ready-made explanations for the playground, generated from this task's own ground truth."""
    s = task.ground_truth.scalars
    top = task.ground_truth.top_residues("rmsf", 3)
    low = task.ground_truth.top_residues("rmsf", 3, "lowest")
    peak, pf = s["rmsd_max_frame"], s["rmsd_plateau_frame"]
    faithful = (
        f"The mean backbone RMSD is {s['rmsd_mean']:.1f} Å. Residues {top[0]}, {top[1]} and {top[2]} show the highest RMSF. "
        f"The radius of gyration is {s['rg_mean']:.1f} Å. RMSD peaks at frame {peak}."
    )
    sloppy = (
        f"The mean backbone RMSD is {s['rmsd_mean'] * 2.2:.1f} Å. Residues {low[0]}, {low[1]} and {low[2]} show the highest RMSF. "
        f"There are about {round(s['hbond_mean_count'] * 0.5)} hydrogen bonds per frame. RMSD plateaus after frame {max(0, pf - 40)}."
    )
    over = (
        f"The mean backbone RMSD is {s['rmsd_mean']:.1f} Å. RMSD peaks at frame {peak}, indicating a large conformational "
        "change. This is caused by hinge motion between the lid and core domains."
    )
    return [
        {"label": "Faithful report", "text": faithful},
        {"label": "Sloppy numbers and residues", "text": sloppy},
        {"label": "Over-interpretation", "text": over},
    ]


def create_app(db_path: str = "results/mdfaith.db", web_dir: str | Path | None = None) -> FastAPI:
    if db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    store = Store(db_path)
    cache = TaskCache()
    extractor = RuleExtractor()
    allow_real = os.environ.get("MDFAITH_ALLOW_REAL") == "1"

    app = FastAPI(title="MD-Faith API", version=__version__, description="Fact-check AI explanations of MD analyses.")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST", "DELETE"], allow_headers=["*"]
    )
    app.state.store, app.state.cache = store, cache

    # ------------------------------------------------------------------ meta
    @app.get("/api/health")
    def health():
        runs = store.list_runs()
        return {
            "status": "ok",
            "version": __version__,
            "n_runs": len(runs),
            "demo_present": any(r["is_demo"] for r in runs),
        }

    @app.get("/api/meta")
    def meta():
        return {
            "version": __version__,
            "systems": sorted(SYSTEMS),
            "artifacts": [{"kind": k, "description": v} for k, v in ARTIFACT_INFO.items()],
            "conditions": [{"name": k, "description": v} for k, v in CONDITION_INFO.items()],
            "simulated_models": [{"name": p.name, "description": p.description} for p in PERSONAS.values()],
            "claim_kinds": [k.value for k in Kind],
            "verdict_labels": [lab.value for lab in Label],
            "extractor_grammar": PATTERNS,
            "real_models_enabled": allow_real,
        }

    # ------------------------------------------------------------------ tasks
    @app.get("/api/tasks")
    def tasks():
        return [
            {
                "task_id": f"{s}:{k}",
                "system": s,
                "artifact": k,
                "description": ARTIFACT_INFO[k],
            }
            for s in sorted(SYSTEMS)
            for k in ARTIFACT_INFO
        ]

    @app.get("/api/tasks/{task_id}")
    def task_detail(task_id: str):
        t = cache.get(task_id)
        gt = t.ground_truth
        return clean(
            {
                "task_id": t.task_id,
                "system": t.system,
                "n_frames": gt.n_frames,
                "artifact": {
                    "kind": t.artifact.kind,
                    "frames": t.artifact.frames,
                    "description": t.artifact.description,
                },
                "resids": gt.resids,
                "rmsd": gt.rmsd,
                "rmsf": gt.rmsf,
                "rg": gt.rg,
                "hbond_counts": gt.hbond_counts,
                "scalars": gt.scalars,
                "qc": cache.qc(task_id),
            }
        )

    @app.get("/api/tasks/{task_id}/qc")
    def task_qc(task_id: str):
        return clean(cache.qc(task_id))

    @app.get("/api/tasks/{task_id}/samples")
    def task_samples(task_id: str):
        return sample_explanations(cache.get(task_id))

    # ------------------------------------------------------------------ verify
    @app.post("/api/verify")
    def verify(req: VerifyRequest):
        t = cache.get(req.task_id)
        claims = extractor.extract(req.text)
        verdicts = verify_all(claims, t.ground_truth, t.artifact)
        ann = annotate(req.text, claims, verdicts, t.ground_truth, t.artifact)
        return clean(
            {
                "task_id": req.task_id,
                "extractor": extractor.name,
                "n_claims": len(claims),
                "summary": summary(ann),
                "sentences": to_dicts(ann),
                "qc": cache.qc(req.task_id),
            }
        )

    # ------------------------------------------------------------------ runs
    @app.get("/api/runs")
    def runs():
        return store.list_runs()

    @app.post("/api/runs", status_code=201)
    def create_run(req: RunRequest):
        for spec in req.models:
            if not spec.startswith("sim-") and not allow_real:
                raise HTTPException(403, "real-model backends are disabled on this server (set MDFAITH_ALLOW_REAL=1)")
        try:
            backends = [make_backend(m) for m in req.models]
        except (ValueError, KeyError) as e:
            raise HTTPException(422, str(e)) from None
        if not set(req.conditions) <= set(ALL_CONDITIONS) or not set(req.artifacts) <= set(ARTIFACT_INFO):
            raise HTTPException(422, "unknown condition or artifact")
        tasks_ = [cache.get(f"adk_dims:{a}") for a in req.artifacts]
        rid = run_experiment(store, tasks_, backends, req.conditions, range(req.seeds), name=req.name)
        return store.get_run(rid)

    @app.post("/api/demo/seed", status_code=201)
    def demo_seed(req: DemoRequest):
        tasks_ = [cache.get(f"adk_dims:{k}") for k in ("none", "pbc_split", "shuffle_frames", "rigid_jitter")]
        rid = seed_demo(store, seeds=req.seeds, tasks=tasks_)
        return store.get_run(rid)

    def _run_or_404(run_id: str) -> dict:
        r = store.get_run(run_id)
        if not r:
            raise HTTPException(404, "run not found")
        return r

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str):
        return _run_or_404(run_id)

    @app.delete("/api/runs/{run_id}", status_code=204)
    def run_delete(run_id: str):
        _run_or_404(run_id)
        store.delete_run(run_id)

    @app.get("/api/runs/{run_id}/summary")
    def run_summary(run_id: str, by: str = Query("condition,model")):
        run = _run_or_404(run_id)
        keys = tuple(k for k in by.split(",") if k in ("condition", "model", "artifact_kind", "task_id"))
        if not keys:
            raise HTTPException(422, "by must name condition, model, artifact_kind or task_id")
        return clean(
            {"run": run, "by": keys, "rows": aggregate(store.per_explanation_counts(run_id), by=keys, n_boot=500)}
        )

    @app.get("/api/runs/{run_id}/explanations")
    def run_explanations(
        run_id: str,
        condition: str | None = None,
        model: str | None = None,
        artifact: str | None = None,
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ):
        _run_or_404(run_id)
        rows = store.per_explanation_counts(run_id)
        texts = {e["id"]: e for e in store.explanations(run_id)}
        out = []
        for r in rows:
            if (condition and r["condition"] != condition) or (model and r["model"] != model):
                continue
            if artifact and r["artifact_kind"] != artifact:
                continue
            out.append({**r, "n_tool_calls": texts[r["explanation_id"]]["n_tool_calls"]})
        return {"total": len(out), "items": out[offset : offset + limit]}

    @app.get("/api/explanations/{explanation_id}")
    def explanation_report(explanation_id: int):
        row = store._db.execute("SELECT * FROM explanations WHERE id=?", (explanation_id,)).fetchone()
        if not row:
            raise HTTPException(404, "explanation not found")
        e = next(x for x in store.explanations(row["run_id"], with_claims=True) if x["id"] == explanation_id)
        t = cache.get(e["task_id"])
        claims, verdicts = _claims_and_verdicts(e["claims"])
        ann = annotate(e["text"], claims, verdicts, t.ground_truth, t.artifact)
        run = store.get_run(row["run_id"])
        return clean(
            {
                "explanation": {
                    k: e[k] for k in ("id", "run_id", "task_id", "artifact_kind", "condition", "model", "seed")
                },
                "is_demo": run["is_demo"],
                "summary": summary(ann),
                "sentences": to_dicts(ann),
            }
        )

    # ------------------------------------------------------------------ static web app
    if web_dir is not None and Path(web_dir).exists():
        web = Path(web_dir)

        @app.get("/app", include_in_schema=False)
        def app_index():
            return FileResponse(web / "app" / "index.html")

        @app.get("/app/", include_in_schema=False)
        def app_index_slash():
            return RedirectResponse("/app")

        app.mount("/app/static", StaticFiles(directory=web / "app"), name="app-static")
        if (web / "assets").exists():
            app.mount("/assets", StaticFiles(directory=web / "assets"), name="assets")
        if not (web / "site" / "index.html").exists():

            @app.get("/", include_in_schema=False)
            def root_redirect():
                return RedirectResponse("/app")

        if (web / "site" / "index.html").exists():

            @app.get("/", include_in_schema=False)
            def site_index():
                return FileResponse(web / "site" / "index.html")

            app.mount("/site", StaticFiles(directory=web / "site"), name="site")
    return app


def app_factory() -> FastAPI:
    """Factory for `uvicorn --factory`; configured through MDFAITH_DB and MDFAITH_WEB."""
    here = Path(__file__).resolve().parent
    web = os.environ.get("MDFAITH_WEB") or (here / "web")
    return create_app(os.environ.get("MDFAITH_DB", "results/mdfaith.db"), web)
