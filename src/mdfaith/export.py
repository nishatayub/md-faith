"""Reproducible export of a stored run: re-verification, summary, explanations and claims as CSV/JSON.

`reverify_run` recomputes every stored claim's verdict with the current verifier and ground truth, so a verifier fix or a
bug repair can be applied to old runs without re-running any model. `export_run` writes the files committed under
`results/<run>/` so every number in the README can be traced to a file.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from .artifacts import DEFAULT_SPECS
from .claims import Claim, Kind
from .metrics import cluster_bootstrap, hallucination_rate
from .pipeline import build_tasks
from .store import Store
from .verify import verify

COUNT_KEYS = ("n", "checkable", "supported", "wrong", "artifact_misread", "unsupported_mechanism")


def tasks_for_run(store: Store, run_id: str) -> dict:
    run = store.get_run(run_id)
    if run is None:
        raise KeyError(f"unknown run {run_id!r}")
    ids = run["config"]["tasks"]
    system = ids[0].split(":", 1)[0]
    kinds = sorted({t.split(":", 1)[1] for t in ids})
    return {t.task_id: t for t in build_tasks(system, [DEFAULT_SPECS[k] for k in kinds])}


def reverify_run(store: Store, run_id: str, tasks: dict | None = None) -> dict:
    """Recompute verdicts for every stored claim. Returns {'claims': n, 'changed': n_changed, 'changes': [...]}."""
    tasks = tasks or tasks_for_run(store, run_id)
    changes, total = [], 0
    for e in store.explanations(run_id, with_claims=True):
        t = tasks[e["task_id"]]
        for row in e["claims"]:
            total += 1
            claim = Claim(row["claim_id"], row["text"], Kind(row["kind"]), row["payload"])
            v = verify(claim, t.ground_truth, t.artifact)
            if v.label.value != row["label"] or v.detail != row["detail"]:
                changes.append(
                    {"explanation_id": e["id"], "claim_row_id": row["id"], "was": row["label"], "now": v.label.value}
                )
                store.update_claim_verdict(row["id"], v.label.value, v.detail)
    return {"claims": total, "changed": len(changes), "changes": changes}


def export_run(store: Store, run_id: str, out_dir: str | Path) -> dict:
    run = store.get_run(run_id)
    if run is None:
        raise KeyError(f"unknown run {run_id!r}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ex = store.explanations(run_id, with_claims=True)
    with open(out / "explanations.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["explanation_id", "task_id", "artifact", "condition", "seed", "n_tool_calls", "n_claims", "text"])
        for e in ex:
            w.writerow(
                [
                    e["id"],
                    e["task_id"],
                    e["artifact_kind"],
                    e["condition"],
                    e["seed"],
                    e["n_tool_calls"],
                    len(e["claims"]),
                    e["text"],
                ]
            )
    with open(out / "claims.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "explanation_id",
                "condition",
                "artifact",
                "seed",
                "claim_id",
                "kind",
                "payload",
                "label",
                "detail",
                "claim_text",
            ]
        )
        for e in ex:
            for c in e["claims"]:
                w.writerow(
                    [
                        e["id"],
                        e["condition"],
                        e["artifact_kind"],
                        e["seed"],
                        c["claim_id"],
                        c["kind"],
                        json.dumps(c["payload"]),
                        c["label"],
                        c["detail"],
                        c["text"],
                    ]
                )
    rows = store.per_explanation_counts(run_id)
    point, lo, hi = cluster_bootstrap([{k: r[k] for k in COUNT_KEYS} for r in rows], hallucination_rate, n=2000, seed=0)
    labels = Counter(c["label"] for e in ex for c in e["claims"])
    n_claims = sum(labels.values())
    summary = {
        "run_id": run_id,
        "name": run["name"],
        "is_demo": run["is_demo"],
        "backends": run["config"].get("backends"),
        "conditions": run["config"].get("conditions"),
        "seeds": run["config"].get("seeds"),
        "tasks": run["config"].get("tasks"),
        "extractor": run["config"].get("extractor"),
        "n_explanations": len(ex),
        "n_explanations_with_no_claims": sum(1 for e in ex if not e["claims"]),
        "n_claims": n_claims,
        "verdicts": dict(labels),
        "checkable_claims": n_claims - labels["unverifiable"] - labels["unsupported_mechanism"],
        "overall_hallucination_rate": {"value": point, "ci95": [lo, hi]},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
