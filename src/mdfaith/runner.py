"""Experiment runner: tasks x conditions x models x seeds -> explanations -> claims -> verdicts -> store."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .conditions import CONDITIONS as CONDITION_CLASSES
from .extract_rules import RuleExtractor
from .simulated import SimulatedModel
from .store import Store
from .verify import verify_all

ALL_CONDITIONS = ("image_only", "table_only", "tool_agent", "qc_gated")


@dataclass
class Backend:
    """A named explainer: a simulated persona or a real `LLMClient`."""

    name: str
    simulated: SimulatedModel | None = None
    client: object | None = None

    @property
    def is_simulated(self) -> bool:
        return self.simulated is not None

    def explain(self, task, condition: str, seed: int):
        if self.simulated is not None:
            return self.simulated.explain(task, condition, seed)
        return CONDITION_CLASSES[condition](self.client).run(task)


def make_backend(spec: str) -> Backend:
    """`sim-careful` | `anthropic:claude-opus-5-5`."""
    if spec.startswith("sim-"):
        return Backend(spec, simulated=SimulatedModel(spec))
    if spec.startswith("anthropic:"):
        from .llm_anthropic import AnthropicClient

        return Backend(spec, client=AnthropicClient(model=spec.split(":", 1)[1]))
    raise ValueError(f"unknown backend {spec!r}; use sim-<persona> or anthropic:<model>")


def run_experiment(
    store: Store,
    tasks: list,
    backends: list,
    conditions=ALL_CONDITIONS,
    seeds=(0,),
    extractor=None,
    name: str = "run",
    progress: Callable[[int, int], None] | None = None,
) -> str:
    """Run the full grid and persist everything. A run is flagged demo if ANY backend is simulated."""
    is_demo = any(b.is_simulated for b in backends)
    extractor = extractor or RuleExtractor()
    config = {
        "tasks": [t.task_id for t in tasks],
        "backends": [b.name for b in backends],
        "conditions": list(conditions),
        "seeds": list(seeds),
        "extractor": getattr(extractor, "name", type(extractor).__name__),
    }
    run_id = store.create_run(name, config, is_demo=is_demo)
    total = len(tasks) * len(backends) * len(conditions) * len(seeds)
    done = 0
    try:
        for task in tasks:
            for backend in backends:
                for cond in conditions:
                    for seed in seeds:
                        exp = backend.explain(task, cond, seed)
                        claims = extractor.extract(exp.text)
                        verdicts = verify_all(claims, task.ground_truth, task.artifact)
                        store.add_explanation(
                            run_id,
                            task.task_id,
                            task.artifact.kind,
                            cond,
                            backend.name,
                            exp.text,
                            claims,
                            verdicts,
                            seed=seed,
                            n_tool_calls=len(exp.tool_calls),
                            n_model_calls=exp.n_model_calls,
                        )
                        done += 1
                        if progress:
                            progress(done, total)
        store.finish_run(run_id, "done")
    except Exception:
        store.finish_run(run_id, "failed")
        raise
    return run_id
