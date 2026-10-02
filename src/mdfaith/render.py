"""Plots and text tables shown to the image-only and table-only conditions."""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .groundtruth import GroundTruth


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def plots(gt: GroundTruth) -> list:
    """RMSD vs frame, RMSF vs residue, contact-occupancy map. Axes are labelled; no ground-truth annotations."""
    f1, a = plt.subplots(figsize=(6, 3))
    a.plot(gt.rmsd)
    a.set_xlabel("frame")
    a.set_ylabel("backbone RMSD (A)")
    f2, a = plt.subplots(figsize=(6, 3))
    a.plot(gt.resids, gt.rmsf)
    a.set_xlabel("residue")
    a.set_ylabel("CA RMSF (A)")
    f3, a = plt.subplots(figsize=(4.5, 4))
    im = a.imshow(gt.contacts, origin="lower", cmap="viridis", vmin=0, vmax=1)
    a.set_xlabel("residue index")
    a.set_ylabel("residue index")
    f3.colorbar(im, label="contact occupancy")
    return [_png(f1), _png(f2), _png(f3)]


def tables(gt: GroundTruth, max_rows: int = 40) -> str:
    """Compact numeric view of the same data (downsampled so it fits a prompt)."""
    step = max(1, gt.n_frames // max_rows)
    lines = ["frame\tbackbone_RMSD_A"] + [f"{i}\t{gt.rmsd[i]:.2f}" for i in range(0, gt.n_frames, step)]
    lines += ["", "residue\tCA_RMSF_A"]
    rstep = max(1, len(gt.resids) // max_rows)
    lines += [f"{gt.resids[i]}\t{gt.rmsf[i]:.2f}" for i in range(0, len(gt.resids), rstep)]
    lines += ["", f"Rg mean {gt.rg.mean():.2f} A, std {gt.rg.std():.2f} A over {gt.n_frames} frames"]
    return "\n".join(lines)
