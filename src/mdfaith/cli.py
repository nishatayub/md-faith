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
    from .runner import make_backend, run_experiment
    from .store import Store

    backends = [make_backend(m) for m in a.models.split(",")]
    extractor = None
    if any(not b.is_simulated for b in backends):
        from .llm_anthropic import AnthropicClient

        extractor = LLMExtractor(AnthropicClient(model=a.extractor_model))
    tasks = build_tasks(a.system, [DEFAULT_SPECS[k] for k in a.artifacts.split(",")])
    store = Store(a.db)
    rid = run_experiment(
        store,
        tasks,
        backends,
        a.conditions.split(","),
        range(a.seeds),
        extractor=extractor,
        name=a.name,
        progress=lambda d, t: print(f"\r{d}/{t}", end="", flush=True),
    )
    print(f"\nrun {rid} stored in {a.db}")


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
    r.add_argument("--models", default="sim-careful,sim-hasty,sim-confident", help="sim-<persona> | anthropic:<model>")
    r.add_argument("--conditions", default="image_only,table_only,tool_agent,qc_gated")
    r.add_argument("--artifacts", default="none,pbc_split,shuffle_frames,rigid_jitter")
    r.add_argument("--system", default="adk_dims")
    r.add_argument("--seeds", type=int, default=3)
    r.add_argument("--extractor-model", default="claude-opus-5-5")
    r.add_argument("--name", default="cli run")
    r.add_argument("--db", default="results/mdfaith.db")
    r.set_defaults(fn=cmd_run)
    s = sub.add_parser("selftest")
    s.add_argument("--config", default="configs/default.yaml")
    s.set_defaults(fn=cmd_selftest)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
