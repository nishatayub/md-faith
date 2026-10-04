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
    s = sub.add_parser("selftest")
    s.add_argument("--config", default="configs/default.yaml")
    s.set_defaults(fn=cmd_selftest)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
