"""Build tasks and score one explanation. Providers/models are injected, nothing here touches the network."""
from __future__ import annotations

from . import artifacts, groundtruth
from .conditions.base import Task
from .data import load_system
from .extract import extract_claims
from .metrics import counts
from .verify import verify_all


def build_tasks(system: str, artifact_specs: list) -> list:
    base = load_system(system)
    tasks = []
    for spec in artifact_specs:
        u, rec = artifacts.build(base, spec)
        gt = groundtruth.compute(u, system)
        tasks.append(Task(f"{system}:{rec.kind}", system, u, gt, rec))
    return tasks


def score_explanation(task: Task, text: str, extractor_client) -> dict:
    claims = extract_claims(extractor_client, text)
    verdicts = verify_all(claims, task.ground_truth, task.artifact)
    return {"task_id": task.task_id, "claims": claims, "verdicts": verdicts, "counts": counts(verdicts)}
