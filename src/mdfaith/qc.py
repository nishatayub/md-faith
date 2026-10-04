"""Trajectory quality control: detect processing problems *before* anyone (human or LLM) interprets the data.

Each detector looks at coordinates only; none of them knows about the planted artifacts in `artifacts.py`, so the
same code can be run on real trajectories. Thresholds live in `QCConfig` and are deliberately conservative.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.stats import spearmanr

from .groundtruth import frames_of, rmsd_series, superpose

SEVERITIES = ("info", "warning", "critical")


@dataclass
class QCConfig:
    ca_bond_max: float = 4.5  # A; trans CA-CA is ~3.8, cis-Pro ~2.9
    jump_abs: float = 3.0  # A; frame-to-frame fitted backbone RMSD above this is a discontinuity
    jump_rel: float = 5.0  # ... or this many times the median step
    cog_step_median: float = 2.0  # A; median centre-of-geometry step above this means not centred/fitted
    order_rank_min: float = 0.5  # Spearman lag-1 of RMSD-to-reference below this: frame order looks scrambled
    min_frames: int = 20
    late_plateau_frac: float = 0.6  # plateau reached after this fraction of the run
    time_gap_ratio: float = 1.5  # time step larger than this times the median step


@dataclass
class QCFlag:
    code: str
    severity: str
    message: str
    frames: list = field(default_factory=list)
    metric: float | None = None


@dataclass
class QCReport:
    n_frames: int
    flags: list = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(f.severity == "critical" for f in self.flags)

    @property
    def codes(self) -> list:
        return [f.code for f in self.flags]

    def has(self, code: str) -> bool:
        return code in self.codes

    def to_dict(self) -> dict:
        return {"n_frames": self.n_frames, "passed": self.passed, "flags": [asdict(f) for f in self.flags]}

    def to_prompt(self) -> str:
        """Plain-text rendering for inclusion in an LLM prompt (the 'QC gate')."""
        if not self.flags:
            return "Automated trajectory QC: no problems detected."
        lines = ["Automated trajectory QC found the following:"]
        for f in self.flags:
            where = f" (frames {_ranges(f.frames)})" if f.frames else ""
            lines.append(f"- [{f.severity.upper()}] {f.code}: {f.message}{where}")
        return "\n".join(lines)


def _ranges(frames: list, limit: int = 6) -> str:
    fs = sorted(set(int(f) for f in frames))
    out, start = [], None
    for i, f in enumerate(fs):
        if start is None:
            start = prev = f
        elif f == prev + 1:
            prev = f
        else:
            out.append((start, prev))
            start = prev = f
        if i == len(fs) - 1:
            out.append((start, prev))
    parts = [f"{a}" if a == b else f"{a}-{b}" for a, b in out[:limit]]
    return ", ".join(parts) + (", ..." if len(out) > limit else "")


def _frame_to_frame(bb: np.ndarray) -> np.ndarray:
    out = np.zeros(len(bb))
    for t in range(1, len(bb)):
        out[t] = np.sqrt(np.mean(np.sum((superpose(bb[t], bb[t - 1]) - bb[t - 1]) ** 2, axis=1)))
    return out


def run_qc(u, cfg: QCConfig | None = None) -> QCReport:
    cfg = cfg or QCConfig()
    bb = frames_of(u, "backbone")
    ca = frames_of(u, "protein and name CA")
    T = len(bb)
    rep = QCReport(n_frames=T)

    if not (np.isfinite(bb).all() and np.isfinite(ca).all()):
        bad = np.nonzero(~np.isfinite(bb).all(axis=(1, 2)))[0].tolist()
        rep.flags.append(QCFlag("non_finite_coordinates", "critical", "NaN or infinite coordinates.", bad))
        return rep

    if T < cfg.min_frames:
        rep.flags.append(
            QCFlag("too_few_frames", "warning", f"Only {T} frames; statistics will be unreliable.", [], float(T))
        )

    # 1. broken backbone: consecutive CA-CA bonds that are intact in most frames but stretched in some
    d = np.linalg.norm(ca[:, 1:] - ca[:, :-1], axis=2)
    stretched = (d > cfg.ca_bond_max) & (np.median(d, axis=0) <= cfg.ca_bond_max)
    bad = np.nonzero(stretched.any(axis=1))[0].tolist()
    if bad:
        rep.flags.append(
            QCFlag(
                "broken_backbone",
                "critical",
                "Backbone bonds are stretched far beyond normal length in these frames, which usually means part "
                "of the protein was wrapped across the periodic boundary. RMSD/RMSF in these frames are not physical.",
                bad,
                float(d[stretched].max()),
            )
        )

    # 2. discontinuities in fitted structure between consecutive frames
    f2f = _frame_to_frame(bb)
    thr = max(cfg.jump_abs, cfg.jump_rel * float(np.median(f2f[1:]) if T > 1 else 0.0))
    jumps = [int(t) for t in np.nonzero(f2f > thr)[0] if t > 0]
    if jumps:
        rep.flags.append(
            QCFlag(
                "frame_discontinuity",
                "critical",
                f"Structure changes abruptly between consecutive frames (fitted RMSD step above {thr:.1f} A).",
                jumps,
                float(f2f.max()),
            )
        )

    # 3. not centred / not fitted
    cog = bb.mean(axis=1)
    step = np.linalg.norm(np.diff(cog, axis=0), axis=1) if T > 1 else np.zeros(1)
    if float(np.median(step)) > cfg.cog_step_median:
        rep.flags.append(
            QCFlag(
                "not_centred_or_fitted",
                "warning",
                "The molecule's centre moves several Angstrom between frames; the trajectory appears not to be "
                "centred or fitted. Use superposition-based analyses and do not read raw coordinate drift as motion.",
                [],
                float(np.median(step)),
            )
        )

    # 4. frame order
    ref_rmsd = rmsd_series(bb)
    if T >= cfg.min_frames:
        rho = float(spearmanr(ref_rmsd[:-1], ref_rmsd[1:])[0])
        if rho < cfg.order_rank_min:
            rep.flags.append(
                QCFlag(
                    "frame_order_unreliable",
                    "warning",
                    "Neighbouring frames are as different from each other as distant frames (rank autocorrelation "
                    f"{rho:.2f}). Either the frame order was scrambled or the system is fully decorrelated; "
                    "time-ordered claims (when something happens) are not trustworthy.",
                    [],
                    rho,
                )
            )

    # 5. late equilibration (information only)
    from .groundtruth import plateau_frame

    pf = plateau_frame(ref_rmsd)
    if T >= cfg.min_frames and pf > cfg.late_plateau_frac * T:
        rep.flags.append(
            QCFlag(
                "late_or_no_plateau",
                "info",
                f"RMSD only settles at frame {pf} of {T}; early frames are not equilibrated.",
                [],
                float(pf),
            )
        )

    # 6. irregular time spacing
    times = np.array([ts.time for ts in u.trajectory], dtype=float)
    if T > 2:
        dt = np.diff(times)
        med = float(np.median(dt))
        if med > 0:
            gaps = np.nonzero(dt > cfg.time_gap_ratio * med)[0] + 1
            if len(gaps):
                rep.flags.append(
                    QCFlag(
                        "time_gaps",
                        "warning",
                        "Irregular time spacing between frames; frames may be missing.",
                        gaps.tolist(),
                        float(dt.max() / med),
                    )
                )
    return rep
