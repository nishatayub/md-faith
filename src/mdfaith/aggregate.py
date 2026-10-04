"""Aggregate per-explanation claim counts into the statistics shown in the UI and figures."""

from __future__ import annotations

from collections import defaultdict

from .metrics import cluster_bootstrap, hallucination_rate, unsupported_mechanism_rate

COUNT_KEYS = ("n", "checkable", "supported", "wrong", "artifact_misread", "unsupported_mechanism")


def _sum(rows: list) -> dict:
    return {k: sum(r[k] for r in rows) for k in COUNT_KEYS}


def aggregate(rows: list, by=("condition", "model"), n_boot: int = 1000, seed: int = 0) -> list:
    """Group `Store.per_explanation_counts` rows by the given keys.

    The bootstrap resamples explanations (task x seed replicates) with replacement. With a single system the number of
    distinct tasks is small, so the interval reflects replicate variability, not variability across systems.
    """
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in by)].append(r)
    out = []
    for key, grp in sorted(groups.items()):
        c = _sum(grp)
        counts = [{k: r[k] for k in COUNT_KEYS} for r in grp]
        point, lo, hi = cluster_bootstrap(counts, hallucination_rate, n=n_boot, seed=seed)
        on_art = [r for r in grp if r["artifact_kind"] != "none"]
        out.append(
            {
                **dict(zip(by, key, strict=True)),
                "n_explanations": len(grp),
                "n_claims": c["n"],
                "hallucination_rate": point,
                "ci_lo": lo,
                "ci_hi": hi,
                "support_rate": c["supported"] / c["checkable"] if c["checkable"] else float("nan"),
                "unsupported_mechanism_rate": unsupported_mechanism_rate(c),
                "artifact_misread_share": (
                    sum(r["artifact_misread"] > 0 for r in on_art) / len(on_art) if on_art else None
                ),
            }
        )
    return out
