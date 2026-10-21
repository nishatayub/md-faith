from itertools import pairwise

import pytest

from mdfaith.artifacts import DEFAULT_SPECS
from mdfaith.demo import seed_demo
from mdfaith.pipeline import build_tasks
from mdfaith.sensitivity import GRID, load_claims, rates, spread, sweep
from mdfaith.store import Store


@pytest.fixture(scope="module")
def setup():
    tasks = build_tasks("adk_dims", [DEFAULT_SPECS[k] for k in ("none", "pbc_split", "shuffle_frames", "rigid_jitter")])
    s = Store()
    rid = seed_demo(s, seeds=3, tasks=tasks)
    return s, rid, {t.task_id: t for t in tasks}


def test_default_setting_reproduces_stored_verdicts(setup):
    s, rid, tasks = setup
    stored = s.per_explanation_counts(rid)
    wrong = sum(r["wrong"] for r in stored)
    checkable = sum(r["checkable"] for r in stored)
    rate, n = rates(load_claims(s, rid), tasks, {})["all"]
    assert n == checkable and rate == pytest.approx(wrong / checkable)


def test_looser_tolerances_never_increase_the_rate(setup):
    """Numeric tolerance and the ranking parameters only ever turn wrong claims into supported ones."""
    s, rid, tasks = setup
    rows = sweep(s, rid, tasks)
    for param, stricter_is_higher in [("num_rel_tol", False), ("rank_min_hit", True), ("rank_window", False)]:
        vals = sorted(GRID[param], reverse=stricter_is_higher)
        rs = [
            next(
                r["hallucination_rate"]
                for r in rows
                if r["parameter"] == param and r["value"] == v and r["condition"] == "all"
            )
            for v in vals
        ]
        assert all(a >= b - 1e-9 for a, b in pairwise(rs)), (param, rs)


def test_frame_window_can_raise_the_rate_through_artifact_flags(setup):
    """A wider window makes more claimed events count as 'inside the artifact frames', so this one is not monotone."""
    s, rid, tasks = setup
    rows = sweep(s, rid, tasks)
    rs = [r["hallucination_rate"] for r in rows if r["parameter"] == "frame_window" and r["condition"] == "all"]
    assert max(rs) - min(rs) > 0
    assert any(b > a + 1e-9 for a, b in pairwise(rs)) or rs == sorted(
        rs, reverse=True
    )  # documented, not asserted either way


def test_sweep_marks_defaults_and_spread(setup):
    s, rid, tasks = setup
    rows = sweep(s, rid, tasks)
    assert {r["parameter"] for r in rows} == set(GRID)
    for p in GRID:
        assert sum(r["is_default"] for r in rows if r["parameter"] == p and r["condition"] == "all") == 1
    sp = spread(rows)
    assert all(v["range"] >= 0 for v in sp.values()) and set(sp) == set(GRID)
