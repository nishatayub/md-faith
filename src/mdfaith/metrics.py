"""Claim-level metrics with task-clustered bootstrap intervals (claims within a task are not independent)."""
from __future__ import annotations

import numpy as np

from .claims import Label

WRONG = {Label.CONTRADICTED, Label.ARTIFACT_MISREAD}


def counts(verdicts) -> dict:
    labels = [v.label for v in verdicts]
    n_checkable = sum(l not in (Label.UNVERIFIABLE, Label.UNSUPPORTED_MECHANISM) for l in labels)
    return {
        "n": len(labels),
        "checkable": n_checkable,
        "supported": sum(l is Label.SUPPORTED for l in labels),
        "wrong": sum(l in WRONG for l in labels),
        "artifact_misread": sum(l is Label.ARTIFACT_MISREAD for l in labels),
        "unsupported_mechanism": sum(l is Label.UNSUPPORTED_MECHANISM for l in labels),
    }


def hallucination_rate(c: dict) -> float:
    """Wrong or artifact-misread claims among checkable claims."""
    return c["wrong"] / c["checkable"] if c["checkable"] else float("nan")


def unsupported_mechanism_rate(c: dict) -> float:
    return c["unsupported_mechanism"] / c["n"] if c["n"] else float("nan")


def cluster_bootstrap(per_task_counts: list, stat=hallucination_rate, n: int = 2000, seed: int = 0):
    """Resample tasks with replacement; return (point, lo, hi) for the pooled statistic."""
    keys = per_task_counts[0].keys()
    pooled = {k: sum(c[k] for c in per_task_counts) for k in keys}
    rng = np.random.default_rng(seed)
    T = len(per_task_counts)
    vals = []
    for _ in range(n):
        pick = rng.integers(0, T, size=T)
        agg = {k: sum(per_task_counts[i][k] for i in pick) for k in keys}
        vals.append(stat(agg))
    vals = np.array(vals, dtype=float)
    return stat(pooled), float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))
