"""Simulated explainers for DEMO purposes only.

These are NOT language models. Each persona generates explanations from the ground truth with a configurable error
rate, so the whole pipeline (explanation -> claim extraction -> verification -> statistics -> UI) can be exercised
offline and deterministically. Results produced with them are flagged `is_demo` everywhere and must never be
presented as findings about real models.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np

from .conditions.base import Explanation, Task

CONDITIONS = ("image_only", "table_only", "tool_agent", "qc_gated")


@dataclass(frozen=True)
class Persona:
    name: str
    description: str
    # per-condition relative error (sd) on numbers, probability of noticing an artifact, probability of a wrong ranking
    num_sd: dict
    notice: dict
    wrong_rank: dict
    frame_sd: dict
    causal_p: float


PERSONAS = {
    "sim-careful": Persona(
        "sim-careful",
        "Reads values accurately, usually notices processing problems when it has the means to.",
        num_sd={"image_only": 0.10, "table_only": 0.03, "tool_agent": 0.01, "qc_gated": 0.01},
        notice={"image_only": 0.15, "table_only": 0.30, "tool_agent": 0.55, "qc_gated": 0.97},
        wrong_rank={"image_only": 0.25, "table_only": 0.08, "tool_agent": 0.02, "qc_gated": 0.02},
        frame_sd={"image_only": 8, "table_only": 3, "tool_agent": 0, "qc_gated": 0},
        causal_p=0.25,
    ),
    "sim-hasty": Persona(
        "sim-hasty",
        "Eyeballs plots, rounds aggressively and rarely questions the data.",
        num_sd={"image_only": 0.25, "table_only": 0.15, "tool_agent": 0.08, "qc_gated": 0.08},
        notice={"image_only": 0.03, "table_only": 0.08, "tool_agent": 0.25, "qc_gated": 0.85},
        wrong_rank={"image_only": 0.55, "table_only": 0.35, "tool_agent": 0.15, "qc_gated": 0.15},
        frame_sd={"image_only": 20, "table_only": 12, "tool_agent": 4, "qc_gated": 4},
        causal_p=0.55,
    ),
    "sim-confident": Persona(
        "sim-confident",
        "Accurate numbers but over-interprets: always attaches a mechanism, rarely flags artifacts.",
        num_sd={"image_only": 0.12, "table_only": 0.04, "tool_agent": 0.02, "qc_gated": 0.02},
        notice={"image_only": 0.05, "table_only": 0.10, "tool_agent": 0.20, "qc_gated": 0.70},
        wrong_rank={"image_only": 0.30, "table_only": 0.10, "tool_agent": 0.04, "qc_gated": 0.04},
        frame_sd={"image_only": 10, "table_only": 4, "tool_agent": 1, "qc_gated": 1},
        causal_p=0.95,
    ),
}


class SimulatedModel:
    """Produces an `Explanation` for a task under a named condition. Deterministic given (seed, task, condition)."""

    is_simulated = True

    def __init__(self, persona: str):
        self.persona = PERSONAS[persona]
        self.name = persona

    def explain(self, task: Task, condition: str, seed: int = 0) -> Explanation:
        p = self.persona
        gt = task.ground_truth
        s = gt.scalars
        rng = np.random.default_rng(zlib.crc32(f"{p.name}|{task.task_id}|{condition}|{seed}".encode()))

        def noisy(v: float) -> float:
            return float(v * (1 + rng.normal(0, p.num_sd[condition])))

        art = task.artifact.kind
        notices = art != "none" and rng.random() < p.notice[condition]
        sents = [f"The mean backbone RMSD is {noisy(s['rmsd_mean']):.1f} Å."]

        if rng.random() < p.wrong_rank[condition]:
            lows = gt.top_residues("rmsf", 3, "lowest")
            sents.append(f"Residues {lows[0]}, {lows[1]} and {lows[2]} show the highest RMSF.")
        else:
            top = gt.top_residues("rmsf", 3, "highest")
            sents.append(f"Residues {top[0]}, {top[1]} and {top[2]} show the highest RMSF.")

        sents.append(f"The radius of gyration is {noisy(s['rg_mean']):.1f} Å.")
        sents.append(f"There are about {max(1, round(noisy(s['hbond_mean_count'])))} hydrogen bonds per frame.")

        peak = int(np.clip(round(s["rmsd_max_frame"] + rng.normal(0, p.frame_sd[condition])), 0, gt.n_frames - 1))
        if art == "shuffle_frames" and notices:
            sents.append("The frame order looks scrambled, so I make no claims about when events happen.")
        elif art == "pbc_split" and notices:
            sents.append(f"RMSD peaks at frame {peak}, which is a periodic-boundary wrapping artifact.")
        else:
            sents.append(
                f"RMSD peaks at frame {peak}, indicating a large conformational change."
                if rng.random() < 0.6
                else f"RMSD peaks at frame {peak}."
            )
            plateau = int(
                np.clip(round(s["rmsd_plateau_frame"] + rng.normal(0, p.frame_sd[condition])), 0, gt.n_frames - 1)
            )
            sents.append(f"RMSD plateaus after frame {plateau}.")
        if art == "rigid_jitter" and notices:
            sents.append("The trajectory is not fitted, so I rely on superposition-based quantities.")
        if rng.random() < p.causal_p:
            sents.append("This is caused by hinge motion between the lid and core domains.")

        n_tools = {"tool_agent": 4, "qc_gated": 5}.get(condition, 0)
        return Explanation(" ".join(sents), [{"name": "simulated"}] * n_tools, 1 + n_tools)
