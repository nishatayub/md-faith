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
        if hasattr(self.client, "seed"):
            self.client.seed = seed
        return CONDITION_CLASSES[condition](self.client).run(task)


VISION_HINTS = ("llava", "moondream", "vl", "vision", "minicpm-v", "gemma3", "llama3.2-vision")


def supports_images(spec: str) -> bool:
    """Simulated and Anthropic backends take images; an Ollama model only if its name suggests a vision model."""
    if not spec.startswith("ollama:"):
        return True
    return any(h in spec.lower() for h in VISION_HINTS)


def make_backend(spec: str, temperature: float | None = None) -> Backend:
    """`sim-careful` | `anthropic:claude-opus-5-5` | `ollama:qwen2.5:3b`."""
    if spec.startswith("sim-"):
        return Backend(spec, simulated=SimulatedModel(spec))
    if spec.startswith("anthropic:"):
        from .llm_anthropic import AnthropicClient

        return Backend(spec, client=AnthropicClient(model=spec.split(":", 1)[1]))
    if spec.startswith("ollama:"):
        from .llm_ollama import OllamaClient

        kw = {} if temperature is None else {"temperature": temperature}
        return Backend(spec, client=OllamaClient(model=spec.split(":", 1)[1], **kw))
    raise ValueError(f"unknown backend {spec!r}; use sim-<persona>, anthropic:<model> or ollama:<model>")


def run_experiment(
    store: Store,
    tasks: list,
    backends: list,
    conditions=ALL_CONDITIONS,
    seeds=(0,),
    extractor=None,
    name: str = "run",
    progress: Callable[[int, int], None] | None = None,
    resume: str | None = None,
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
    done_cells = set()
    if resume:  # continue an interrupted run: skip cells that already have a stored explanation
        run_id = resume
        done_cells = {(e["task_id"], e["model"], e["condition"], e["seed"]) for e in store.explanations(run_id)}
        store.set_status(run_id, "running")
    else:
        run_id = store.create_run(name, config, is_demo=is_demo)
    total = len(tasks) * len(backends) * len(conditions) * len(seeds)
    done = 0
    try:
        for task in tasks:
            for backend in backends:
                for cond in conditions:
                    for seed in seeds:
                        if (task.task_id, backend.name, cond, seed) in done_cells:
                            done += 1
                            continue
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
