"""Tolerance sensitivity: how much do the reported rates depend on my choice of tolerances?

The verifier uses three tunable tolerances (defaults in parentheses):

- numeric relative tolerance (10%): how close a stated value must be to the computed one
- temporal frame window (5 frames): how close a stated event frame must be
- ranking minimum hit fraction (60%) and residue window (0): how many claimed residues must be in the computed top-k

`sweep()` re-verifies every stored claim of a run under each setting, one parameter at a time with the others at their
defaults, and reports the hallucination rate overall and per condition. Stored verdicts are never modified.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from .claims import Claim, Kind, Label
from .metrics import counts, hallucination_rate
from .store import Store
from .verify import verify

DEFAULTS = {"num_rel_tol": 0.10, "frame_window": 5, "rank_min_hit": 0.6, "rank_window": 0}
GRID = {
    "num_rel_tol": [0.02, 0.05, 0.10, 0.20, 0.30, 0.50],
    "frame_window": [0, 2, 5, 10, 20, 40],
    "rank_min_hit": [0.2, 0.4, 0.6, 0.8, 1.0],
    "rank_window": [0, 1, 2, 5],
}
LABELS = {
    "num_rel_tol": "numeric relative\ntolerance",
    "frame_window": "temporal window\n(frames)",
    "rank_min_hit": "ranking: min. fraction\nof residues",
    "rank_window": "ranking: residue\nwindow",
}


@dataclass
class _V:
    label: Label


def _apply(c: Claim, overrides: dict) -> Claim:
    """Copy of a claim with only the overridden tolerances written into its payload.

    With no overrides the claim is verified exactly as it was when stored, so the default row of a sweep reproduces the
    stored verdicts.
    """
    p = copy.deepcopy(c.payload)
    if c.kind is Kind.NUMERIC and "num_rel_tol" in overrides:
        p["rel_tol"], p["abs_tol"] = overrides["num_rel_tol"], 0.0
    elif c.kind is Kind.TEMPORAL and "frame_window" in overrides:
        p["window"] = overrides["frame_window"]
    elif c.kind is Kind.RANKING:
        if "rank_min_hit" in overrides:
            p["min_hit_frac"] = overrides["rank_min_hit"]
        if "rank_window" in overrides:
            p["window"] = overrides["rank_window"]
    return Claim(c.id, c.text, c.kind, p)


def load_claims(store: Store, run_id: str) -> list[dict]:
    """Stored explanations with their claims as `Claim` objects (verdicts are recomputed, not read)."""
    out = []
    for e in store.explanations(run_id, with_claims=True):
        e = dict(e)
        e["claim_objs"] = [Claim(c["claim_id"], c["text"], Kind(c["kind"]), c["payload"]) for c in e["claims"]]
        out.append(e)
    return out


def rates(explanations: list[dict], tasks: dict, overrides: dict) -> dict:
    """{condition: (hallucination_rate, n_checkable)} plus an 'all' entry, with the given tolerance overrides."""
    per: dict[str, list] = {}
    for e in explanations:
        t = tasks[e["task_id"]]
        for c in e["claim_objs"]:
            v = verify(_apply(c, overrides), t.ground_truth, t.artifact)
            per.setdefault(e["condition"], []).append(_V(v.label))
            per.setdefault("all", []).append(_V(v.label))
    out = {}
    for cond, vs in per.items():
        c = counts(vs)
        out[cond] = (hallucination_rate(c), c["checkable"])
    return out


def sweep(store: Store, run_id: str, tasks: dict) -> list[dict]:
    """One-at-a-time sweep. `tasks` maps task_id -> Task (ground truth + artifact record)."""
    expl = load_claims(store, run_id)
    rows = []
    for param, values in GRID.items():
        for v in values:
            overrides = {} if v == DEFAULTS[param] else {param: v}
            for cond, (rate, n) in rates(expl, tasks, overrides).items():
                rows.append(
                    {
                        "parameter": param,
                        "value": v,
                        "is_default": v == DEFAULTS[param],
                        "condition": cond,
                        "hallucination_rate": rate,
                        "n_checkable": n,
                    }
                )
    return rows


def spread(rows: list[dict], condition: str = "all") -> dict:
    """Per parameter: min and max rate across the grid, i.e. how far my choice could move the headline number."""
    out = {}
    for param in GRID:
        rs = [r["hallucination_rate"] for r in rows if r["parameter"] == param and r["condition"] == condition]
        rs = [x for x in rs if x == x]  # drop NaN
        if rs:
            out[param] = {"min": min(rs), "max": max(rs), "range": max(rs) - min(rs)}
    return out
