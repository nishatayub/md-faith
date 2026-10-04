"""Seed the database with a clearly labelled DEMO run built from simulated explainers."""

from __future__ import annotations

from .artifacts import DEFAULT_SPECS
from .pipeline import build_tasks
from .runner import ALL_CONDITIONS, make_backend, run_experiment
from .store import Store

DEMO_NAME = "DEMO (simulated explainers, not real models)"


def seed_demo(store: Store, seeds: int = 8, system: str = "adk_dims", tasks=None) -> str:
    tasks = tasks or build_tasks(
        system, [DEFAULT_SPECS[k] for k in ("none", "pbc_split", "shuffle_frames", "rigid_jitter")]
    )
    backends = [make_backend(p) for p in ("sim-careful", "sim-hasty", "sim-confident")]
    return run_experiment(store, tasks, backends, ALL_CONDITIONS, range(seeds), name=DEMO_NAME)
