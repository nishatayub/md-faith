"""mdfaith command line.

mdfaith groundtruth --system adk_dims --out results/gt_adk.json
mdfaith selftest                       # offline end-to-end check with scripted (fake) model replies
"""

from __future__ import annotations

import argparse
import json
import pathlib

import yaml

from . import groundtruth
from .claims import Label
from .data import load_system
from .llm import Reply, ScriptedClient
from .pipeline import build_tasks, score_explanation


def cmd_groundtruth(a) -> None:
    gt = groundtruth.compute(load_system(a.system), a.system)
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(gt.to_dict(), indent=2))
    print(json.dumps(gt.scalars, indent=2))


def cmd_qc(a) -> None:
    from . import artifacts
    from .qc import run_qc

    u, rec = artifacts.build_default(load_system(a.system), a.artifact)
    rep = run_qc(u)
    print(f"artifact={rec.kind}  passed={rep.passed}")
    print(rep.to_prompt())


def cmd_demo(a) -> None:
    from .aggregate import aggregate
    from .demo import seed_demo
    from .store import Store

    store = Store(a.db)
    rid = seed_demo(store, seeds=a.seeds)
    print(f"seeded DEMO run {rid} into {a.db} (simulated explainers, not real models)")
    for r in aggregate(store.per_explanation_counts(rid), by=("condition",), n_boot=300):
        print(
            f"  {r['condition']:<11} hallucination rate {r['hallucination_rate']:.2f} "
            f"[{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]  n_claims={r['n_claims']}"
        )


def cmd_run(a) -> None:
    from .artifacts import DEFAULT_SPECS
    from .extract import LLMExtractor
    from .pipeline import build_tasks
    from .runner import make_backend, run_experiment, supports_images
    from .store import Store

    specs = a.models.split(",")
    backends = [make_backend(m, temperature=a.temperature) for m in specs]
    conditions = a.conditions.split(",") if a.conditions else ["image_only", "table_only", "tool_agent", "qc_gated"]
    if not a.conditions and not all(supports_images(m) for m in specs):
        conditions.remove("image_only")
        print("note: dropping image_only (a model in --models has no vision support); pass --conditions to override")
    extractor = None
    if any(not b.is_simulated for b in backends):
        spec = a.extractor_model
        if spec is None:  # default: free local extractor when only local models are run, else Claude
            spec = "claude-opus-5-5" if any(b.name.startswith("anthropic:") for b in backends) else None
            spec = spec or next(b.name for b in backends if not b.is_simulated)
        if spec.startswith("ollama:"):
            from .llm_ollama import OllamaClient

            client = OllamaClient(
                model=spec.split(":", 1)[1], temperature=0.0, max_tokens=1200
            )  # extractor must be deterministic
        else:
            from .llm_anthropic import AnthropicClient

            client = AnthropicClient(model=spec.removeprefix("anthropic:"))
        extractor = LLMExtractor(client)
    tasks = build_tasks(a.system, [DEFAULT_SPECS[k] for k in a.artifacts.split(",")])
    store = Store(a.db)
    rid = run_experiment(
        store,
        tasks,
        backends,
        conditions,
        range(a.seeds),
        extractor=extractor,
        name=a.name,
        progress=lambda d, t: print(f"\r{d}/{t}", end="", flush=True),
        resume=a.resume,
    )
    print(f"\nrun {rid} stored in {a.db}")


def cmd_serve(a) -> None:
    import os

    try:
        import uvicorn
    except ImportError as e:  # pragma: no cover
        raise SystemExit("serve needs the api extra: pip install -e '.[api]'") from e
    os.environ["MDFAITH_DB"] = a.db
    uvicorn.run("mdfaith.api:app_factory", factory=True, host=a.host, port=a.port, reload=a.reload)


def cmd_figures(a) -> None:
    from .figures import make_figures
    from .store import Store

    store = Store(a.db)
    run_id = a.run or (store.list_runs() or [{}])[0].get("id")
    if not run_id:
        raise SystemExit("no runs in the database; run `mdfaith demo` first")
    for p in make_figures(store, run_id, a.out, with_trajectory=not a.no_trajectory):
        print("wrote", p)


def cmd_selftest(a) -> None:
    """A fake 'agent' that misreads a planted wrapping artifact must be caught; an honest one must pass."""
    cfg = yaml.safe_load(open(a.config))
    specs = [s for s in cfg["artifacts"] if s["kind"] in ("none", "pbc_split")]
    tasks = build_tasks(cfg["systems"][0], specs)
    for t in tasks:
        peak = t.ground_truth.scalars["rmsd_max_frame"]
        claims = json.dumps(
            [
                {
                    "id": "1",
                    "text": "Large conformational change",
                    "kind": "temporal",
                    "payload": {"event": "rmsd_peak", "frame": peak, "physical": True},
                },
                {
                    "id": "2",
                    "text": "Mean RMSD value",
                    "kind": "numeric",
                    "payload": {"quantity": "rmsd_mean", "value": t.ground_truth.scalars["rmsd_mean"]},
                },
                {"id": "3", "text": "Because of hinge motion", "kind": "causal", "payload": {}},
            ]
        )
        res = score_explanation(t, "fake explanation", ScriptedClient([Reply(text=claims)]))
        print(t.task_id, {v.claim_id: v.label.value for v in res["verdicts"]})
        want = Label.ARTIFACT_MISREAD if t.artifact.kind == "pbc_split" else Label.SUPPORTED
        assert res["verdicts"][0].label is want, (t.task_id, res["verdicts"][0])
    print("selftest ok")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="mdfaith")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("groundtruth")
    g.add_argument("--system", default="adk_dims")
    g.add_argument("--out", default="results/groundtruth.json")
    g.set_defaults(fn=cmd_groundtruth)
    q = sub.add_parser("qc", help="run trajectory QC on a system, optionally with a planted artifact")
    q.add_argument("--system", default="adk_dims")
    q.add_argument("--artifact", default="none", choices=["none", "pbc_split", "shuffle_frames", "rigid_jitter"])
    q.set_defaults(fn=cmd_qc)
    d = sub.add_parser("demo", help="seed a DEMO run from simulated explainers")
    d.add_argument("--db", default="results/mdfaith.db")
    d.add_argument("--seeds", type=int, default=8)
    d.set_defaults(fn=cmd_demo)
    r = sub.add_parser("run", help="run an experiment grid (simulated or real models)")
    r.add_argument(
        "--models",
        default="sim-careful,sim-hasty,sim-confident",
        help="sim-<persona> | anthropic:<model> | ollama:<model>",
    )
    r.add_argument(
        "--conditions", default=None, help="comma list (default: all four; image_only dropped for text-only models)"
    )
    r.add_argument(
        "--temperature", type=float, default=None, help="sampling temperature for Ollama explainers (default 0.7)"
    )
    r.add_argument("--artifacts", default="none,pbc_split,shuffle_frames,rigid_jitter")
    r.add_argument("--system", default="adk_dims")
    r.add_argument("--seeds", type=int, default=3)
    r.add_argument(
        "--extractor-model",
        default=None,
        help="claim extractor: ollama:<model> | anthropic:<model> (default: see docs)",
    )
    r.add_argument("--name", default="cli run")
    r.add_argument("--resume", default=None, help="run id to continue, skipping cells already stored")
    r.add_argument("--db", default="results/mdfaith.db")
    r.set_defaults(fn=cmd_run)
    sv = sub.add_parser("serve", help="start the web app and API")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--db", default="results/mdfaith.db")
    sv.add_argument("--reload", action="store_true")
    sv.set_defaults(fn=cmd_serve)
    f = sub.add_parser("figures", help="write figures and tables for a stored run")
    f.add_argument("--db", default="results/mdfaith.db")
    f.add_argument("--run", default=None, help="run id (default: most recent)")
    f.add_argument("--out", default="docs/figures")
    f.add_argument("--no-trajectory", action="store_true", help="skip the trajectory-based figure")
    f.set_defaults(fn=cmd_figures)
    s = sub.add_parser("selftest")
    s.add_argument("--config", default="configs/default.yaml")
    s.set_defaults(fn=cmd_selftest)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
