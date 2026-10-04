"""Publication-style figures and tables generated from a stored run.

`mdfaith figures --run <id>` writes PNGs and CSVs. Figures from simulated (demo) runs are stamped as such so they
cannot be mistaken for findings; replace them by pointing the command at a real-model run.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .aggregate import aggregate
from .store import Store

COND = ["image_only", "table_only", "tool_agent", "qc_gated"]
COND_LABEL = {
    "image_only": "Image only",
    "table_only": "Table only",
    "tool_agent": "Tool agent",
    "qc_gated": "QC-gated agent",
}
ART = ["none", "pbc_split", "shuffle_frames", "rigid_jitter"]
ART_LABEL = {
    "none": "Clean",
    "pbc_split": "Wrapped domain",
    "shuffle_frames": "Shuffled frames",
    "rigid_jitter": "Unfitted frames",
}
# validated categorical slots 1-4 (light surface), fixed order
COLORS = {"image_only": "#2a78d6", "table_only": "#eb6834", "tool_agent": "#1baf7a", "qc_gated": "#eda100"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DEMO_STAMP = "DEMO DATA: simulated explainers, not findings about real models"


def _style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10.5,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK2,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "text.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
        }
    )


def _stamp(fig, is_demo: bool):
    if is_demo:
        fig.text(0.5, 0.985, DEMO_STAMP, ha="center", va="top", fontsize=9, color="#8a5a00", fontweight="bold")


def _save(fig, path: Path):
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_by_condition(rows, is_demo, out: Path):
    rows = {r["condition"]: r for r in rows}
    conds = [c for c in COND if c in rows]
    fig, ax = plt.subplots(figsize=(7.2, 0.8 * len(conds) + 1.6))
    y = np.arange(len(conds))[::-1]
    for yi, c in zip(y, conds, strict=True):
        r = rows[c]
        ax.barh(yi, r["hallucination_rate"], height=0.55, color=COLORS[c])
        ax.errorbar(
            r["hallucination_rate"],
            yi,
            xerr=[[r["hallucination_rate"] - r["ci_lo"]], [r["ci_hi"] - r["hallucination_rate"]]],
            color=INK,
            capsize=3,
            lw=1.3,
        )
        ax.text(r["ci_hi"] + 0.015, yi, f"{r['hallucination_rate']:.2f}", va="center", fontweight="bold")
    ax.set_yticks(y, [COND_LABEL[c] for c in conds])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Hallucination rate (wrong or artifact-misread claims / checkable claims)")
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.set_title(
        "Hallucination rate by condition (95% bootstrap CI)", loc="left", fontsize=12, fontweight="bold", pad=14
    )
    _stamp(fig, is_demo)
    return _save(fig, out / "fig1_hallucination_by_condition.png")


def _heat(ax, M, rows, cols, title):
    im = ax.imshow(M, cmap=matplotlib.colors.ListedColormap(RAMP), vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = M[i, j]
            ax.text(
                j,
                i,
                "n/a" if np.isnan(v) else f"{v:.2f}",
                ha="center",
                va="center",
                fontweight="bold",
                color="white" if (not np.isnan(v) and v > 0.5) else INK,
            )
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=12)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    return im


def fig_heatmap(rows, is_demo, out: Path):
    conds = [c for c in COND if any(r["condition"] == c for r in rows)]
    arts = [a for a in ART if any(r["artifact_kind"] == a for r in rows)]
    M = np.full((len(conds), len(arts)), np.nan)
    for r in rows:
        if r["condition"] in conds and r["artifact_kind"] in arts:
            M[conds.index(r["condition"]), arts.index(r["artifact_kind"])] = r["hallucination_rate"]
    fig, ax = plt.subplots(figsize=(7.2, 0.7 * len(conds) + 1.8))
    im = _heat(
        ax,
        M,
        [COND_LABEL[c] for c in conds],
        [ART_LABEL[a] for a in arts],
        "Hallucination rate by condition and trajectory problem",
    )
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="rate")
    _stamp(fig, is_demo)
    return _save(fig, out / "fig2_condition_by_artifact_heatmap.png")


def fig_artifact_share(rows, is_demo, out: Path):
    rows = {r["condition"]: r for r in rows if r["artifact_misread_share"] is not None}
    conds = [c for c in COND if c in rows]
    fig, ax = plt.subplots(figsize=(7.2, 0.8 * len(conds) + 1.6))
    y = np.arange(len(conds))[::-1]
    for yi, c in zip(y, conds, strict=True):
        v = rows[c]["artifact_misread_share"]
        ax.barh(yi, v, height=0.55, color=COLORS[c])
        ax.text(v + 0.015, yi, f"{v:.0%}", va="center", fontweight="bold")
    ax.set_yticks(y, [COND_LABEL[c] for c in conds])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Share of explanations on artifact variants with at least one artifact-misread claim")
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.set_title("Explanations that misread a planted artifact", loc="left", fontsize=12, fontweight="bold", pad=14)
    _stamp(fig, is_demo)
    return _save(fig, out / "fig3_artifact_misread_share.png")


def fig_artifact_rmsd(out: Path):
    """RMSD of the bundled trajectory under each planted artifact (computed, not simulated)."""
    from .artifacts import DEFAULT_SPECS, build
    from .data import load_system
    from .groundtruth import frames_of, rmsd_series

    base = load_system("adk_dims")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.8), sharey=False)
    for ax, kind in zip(axes, ["none", "pbc_split", "shuffle_frames"], strict=True):
        u, rec = build(base, DEFAULT_SPECS[kind])
        r = rmsd_series(frames_of(u, "backbone"))
        ax.plot(r, color=COLORS["image_only"], lw=1.8)
        if kind == "pbc_split":
            ax.axvspan(min(rec.frames) - 0.5, max(rec.frames) + 0.5, color="#ec835a", alpha=0.25)
            ax.text(max(rec.frames) + 3, r.max() * 0.9, "wrapped\nframes", fontsize=9, color=INK2)
        ax.set_title(ART_LABEL[kind], loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel("frame")
        ax.yaxis.grid(True, color=GRID)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("backbone RMSD (Å)")
    fig.suptitle(
        "The same trajectory under planted problems (computed ground truth)",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
        y=1.05,
    )
    return _save(fig, out / "fig4_artifact_rmsd.png")


def write_tables(by_cond, by_cond_art, out: Path):
    for name, rows in (("table_by_condition.csv", by_cond), ("table_by_condition_artifact.csv", by_cond_art)):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


def make_figures(store: Store, run_id: str, out_dir: str | Path, with_trajectory: bool = True) -> list[Path]:
    run = store.get_run(run_id)
    if run is None:
        raise KeyError(f"unknown run {run_id!r}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _style()
    rows = store.per_explanation_counts(run_id)
    by_cond = aggregate(rows, by=("condition",), n_boot=1000)
    by_ca = aggregate(rows, by=("condition", "artifact_kind"), n_boot=300)
    paths = [
        fig_by_condition(by_cond, run["is_demo"], out),
        fig_heatmap(by_ca, run["is_demo"], out),
        fig_artifact_share(by_cond, run["is_demo"], out),
    ]
    if with_trajectory:
        paths.append(fig_artifact_rmsd(out))
    write_tables(by_cond, by_ca, out)
    return paths
