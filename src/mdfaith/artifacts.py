"""Planted trajectory problems with known consequences.

Each generator returns (new_universe, ArtifactRecord). The record states which frames are affected, so the verifier
can tell a claim about a real physical event from a misreading of an artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import MDAnalysis as mda
import numpy as np
from MDAnalysis.coordinates.memory import MemoryReader

from .groundtruth import frames_of


@dataclass
class ArtifactRecord:
    kind: str
    frames: list = field(default_factory=list)  # frames whose coordinates are artificial
    params: dict = field(default_factory=dict)
    description: str = ""


def _rebuild(u: mda.Universe, pos: np.ndarray) -> mda.Universe:
    u2 = mda.Merge(u.atoms)
    u2.load_new(pos.astype(np.float32), format=MemoryReader, order="fac")
    return u2


def none(u):
    return u, ArtifactRecord("none", description="unmodified trajectory")


def pbc_split(u, frames, resid_range, box_len: float = 80.0, axis: int = 0):
    """Shift residues in `resid_range` by one box length along `axis` in `frames`, as if a domain wrapped
    across the periodic boundary. Produces RMSD spikes that are not physical."""
    pos = frames_of(u, "all")
    lo, hi = resid_range
    mask = np.array([lo <= a.resid <= hi for a in u.atoms])
    for f in frames:
        pos[f, mask, axis] += box_len
    return _rebuild(u, pos), ArtifactRecord(
        "pbc_split",
        list(frames),
        {"resid_range": [lo, hi], "box_len": box_len, "axis": axis},
        f"residues {lo}-{hi} shifted by {box_len} A along axis {axis} in frames {list(frames)}",
    )


def shuffle_frames(u, seed: int = 0):
    """Randomly permute frame order: frame-wise statistics are unchanged, time-ordered claims become meaningless."""
    pos = frames_of(u, "all")
    perm = np.random.default_rng(seed).permutation(len(pos))
    return _rebuild(u, pos[perm]), ArtifactRecord(
        "shuffle_frames",
        list(range(len(pos))),
        {"seed": seed, "perm": perm.tolist()},
        "frame order randomly permuted",
    )


def _random_rotation(rng) -> np.ndarray:
    Q, R = np.linalg.qr(rng.normal(size=(3, 3)))
    Q = Q * np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def rigid_jitter(u, max_shift: float = 30.0, seed: int = 0):
    """Apply a random rigid-body rotation and translation to every frame (trajectory not centred or fitted).
    Superposition-based quantities are unchanged; raw coordinates and un-fitted RMSD are not."""
    rng = np.random.default_rng(seed)
    pos = frames_of(u, "all")
    for t in range(len(pos)):
        c = pos[t].mean(0)
        pos[t] = (pos[t] - c) @ _random_rotation(rng) + c + rng.uniform(-max_shift, max_shift, size=3)
    return _rebuild(u, pos), ArtifactRecord(
        "rigid_jitter",
        list(range(len(pos))),
        {"max_shift": max_shift, "seed": seed},
        "random rigid-body motion applied to every frame; trajectory is not fitted",
    )


BUILDERS = {
    "none": none,
    "pbc_split": pbc_split,
    "shuffle_frames": shuffle_frames,
    "rigid_jitter": rigid_jitter,
}


def build(u, spec: dict):
    spec = dict(spec)
    kind = spec.pop("kind")
    return BUILDERS[kind](u, **spec)


# Sensible defaults so an artifact can be requested by name alone (CLI, API, UI).
DEFAULT_SPECS = {
    "none": {"kind": "none"},
    "pbc_split": {"kind": "pbc_split", "frames": [30, 31, 32], "resid_range": [1, 60], "box_len": 80.0},
    "shuffle_frames": {"kind": "shuffle_frames", "seed": 0},
    "rigid_jitter": {"kind": "rigid_jitter", "max_shift": 30.0, "seed": 0},
}


def build_default(u, kind: str):
    return build(u, DEFAULT_SPECS[kind])
