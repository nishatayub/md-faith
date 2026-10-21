"""Re-extract claims from stored explanations with a different extractor, and compare two extractions.

Why: a model that extracts claims from its own explanations checks itself. Re-extracting the same explanation texts with a
separate (ideally stronger) model removes that, without regenerating any explanation.

`compare_runs` measures how often two extractors agree on which claims an explanation contains. Agreement is a proxy for
extraction quality, not a validation: two extractors can agree and both be wrong. Human validation is `validation.py`.
"""

from __future__ import annotations

from collections import defaultdict

from .export import tasks_for_run
from .store import Store
from .verify import verify_all


def reextract_run(store: Store, run_id: str, extractor, name: str | None = None, progress=None) -> str:
    """Create a new run with the same explanations, claims re-extracted by `extractor` and re-verified."""
    src = store.get_run(run_id)
    if src is None:
        raise KeyError(f"unknown run {run_id!r}")
    tasks = tasks_for_run(store, run_id)
    cfg = {**src["config"], "extractor": getattr(extractor, "name", type(extractor).__name__), "source_run": run_id}
    new_id = store.create_run(name or f"{src['name']} (re-extracted)", cfg, is_demo=src["is_demo"])
    expl = store.explanations(run_id)
    try:
        for i, e in enumerate(expl, 1):
            t = tasks[e["task_id"]]
            claims = extractor.extract(e["text"])
            verdicts = verify_all(claims, t.ground_truth, t.artifact)
            store.add_explanation(
                new_id,
                e["task_id"],
                e["artifact_kind"],
                e["condition"],
                e["model"],
                e["text"],
                claims,
                verdicts,
                seed=e["seed"],
                n_tool_calls=e["n_tool_calls"],
                n_model_calls=e["n_model_calls"],
            )
            if progress:
                progress(i, len(expl))
        store.finish_run(new_id, "done")
    except Exception:
        store.finish_run(new_id, "failed")
        raise
    return new_id


def _key(c: dict):
    """Identity of a claim independent of its id or wording."""
    p = c["payload"]
    if c["kind"] == "numeric":
        return (
            "numeric",
            p.get("quantity"),
            round(float(p["value"]), 3) if isinstance(p.get("value"), (int, float)) else None,
        )
    if c["kind"] == "ranking":
        return ("ranking", p.get("quantity"), tuple(sorted(p.get("residues") or [])))
    if c["kind"] == "temporal":
        return ("temporal", p.get("event"), p.get("frame"))
    return ("causal", c["text"].strip().lower()[:80])


def compare_runs(store: Store, run_a: str, run_b: str) -> dict:
    """Agreement between two extractions of the same explanations (matched on task, model, condition and seed)."""

    def load(rid):
        return {
            (e["task_id"], e["model"], e["condition"], e["seed"]): e for e in store.explanations(rid, with_claims=True)
        }

    A, B = load(run_a), load(run_b)
    shared = sorted(set(A) & set(B))
    n_a = n_b = both = same_verdict = 0
    by_kind: dict = defaultdict(lambda: [0, 0, 0])  # in A, in B, in both
    for k in shared:
        ca, cb = defaultdict(list), defaultdict(list)
        for c in A[k]["claims"]:
            ca[_key(c)].append(c)
        for c in B[k]["claims"]:
            cb[_key(c)].append(c)
        for key in set(ca) | set(cb):
            na, nb = len(ca.get(key, [])), len(cb.get(key, []))
            m = min(na, nb)
            n_a, n_b, both = n_a + na, n_b + nb, both + m
            kind = key[0]
            by_kind[kind][0] += na
            by_kind[kind][1] += nb
            by_kind[kind][2] += m
            for x, y in zip(ca.get(key, [])[:m], cb.get(key, [])[:m], strict=True):
                same_verdict += x["label"] == y["label"]
    union = n_a + n_b - both
    return {
        "explanations_compared": len(shared),
        "claims_a": n_a,
        "claims_b": n_b,
        "claims_in_both": both,
        "jaccard": (both / union) if union else None,
        "verdict_agreement_on_shared": (same_verdict / both) if both else None,
        "by_kind": {k: {"a": a, "b": b, "both": c} for k, (a, b, c) in sorted(by_kind.items())},
    }
