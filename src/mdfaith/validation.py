"""Human validation of claim extraction.

Extracting claims from a model's explanation is itself done by a model, so before any result is reported a person has to
check a sample. The workflow:

1. `make_sample` writes a **blinded** annotation file (claims without their verdicts) and a separate answer key.
2. A person fills `faithful` (and optionally `correct_quantity`) for each claim, and for a sample of explanations counts
   the checkable statements the extractor missed.
3. `score` reports extractor precision (with a Wilson interval), the share of verdicts a human agrees with, estimated
   recall, and the hallucination rate recomputed using only the claims a human confirmed.

Blinding matters: seeing "contradicted" next to a claim nudges a labeller toward finding the extraction wrong.
"""

from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from pathlib import Path

from .store import Store

YES = {"yes", "y", "1", "true", "faithful"}
NO = {"no", "n", "0", "false", "unfaithful"}
PARTLY = {"partly", "partial", "p"}

CLAIM_COLS = ["claim_row_id", "condition", "artifact", "kind", "claim_text", "extracted_as", "faithful", "notes"]
EXPL_COLS = ["explanation_id", "condition", "artifact", "n_extracted", "text", "missed_checkable_statements", "notes"]


def _extracted_as(kind: str, payload: dict) -> str:
    p = {k: v for k, v in payload.items() if not k.startswith("_")}
    if kind == "numeric":
        return f"{p.get('quantity')} = {p.get('value')}"
    if kind == "ranking":
        return f"{p.get('direction', 'highest')} {p.get('quantity')}: residues {p.get('residues')}"
    if kind == "temporal":
        return f"{p.get('event')} at frame {p.get('frame')}" + (
            "" if p.get("physical", True) else " (marked as artifact)"
        )
    return "causal / mechanistic statement" + (
        f", citing {p.get('evidence')}" if p.get("evidence") else ", citing no evidence"
    )


def _write(path: Path, cols: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def make_sample(
    store: Store, run_id: str, out_dir: str | Path, n_claims: int = 100, n_explanations: int = 20, seed: int = 0
) -> dict:
    """Stratified (by condition) random sample. Returns counts; writes three files into `out_dir`."""
    if store.get_run(run_id) is None:
        raise KeyError(f"unknown run {run_id!r}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    ex = store.explanations(run_id, with_claims=True)
    pool = defaultdict(list)
    for e in ex:
        for c in e["claims"]:
            pool[e["condition"]].append((e, c))
    total = sum(len(v) for v in pool.values())
    take = min(n_claims, total)
    chosen = []
    for cond in sorted(pool):  # proportional allocation, at least one per condition
        k = max(1, round(take * len(pool[cond]) / total))
        chosen += rng.sample(pool[cond], min(k, len(pool[cond])))
    rng.shuffle(chosen)
    chosen = chosen[:take]
    blinded, key = [], []
    for e, c in chosen:
        blinded.append(
            {
                "claim_row_id": c["id"],
                "condition": e["condition"],
                "artifact": e["artifact_kind"],
                "kind": c["kind"],
                "claim_text": c["text"],
                "extracted_as": _extracted_as(c["kind"], c["payload"]),
                "faithful": "",
                "notes": "",
            }
        )
        key.append({"claim_row_id": c["id"], "label": c["label"], "detail": c["detail"]})
    _write(out / "claims_to_check.csv", CLAIM_COLS, blinded)
    _write(out / "answer_key.csv", ["claim_row_id", "label", "detail"], key)
    ex_pick = rng.sample(ex, min(n_explanations, len(ex)))
    _write(
        out / "explanations_to_check.csv",
        EXPL_COLS,
        [
            {
                "explanation_id": e["id"],
                "condition": e["condition"],
                "artifact": e["artifact_kind"],
                "n_extracted": len(e["claims"]),
                "text": e["text"],
                "missed_checkable_statements": "",
                "notes": "",
            }
            for e in ex_pick
        ],
    )
    return {"claims_sampled": len(chosen), "claims_total": total, "explanations_sampled": len(ex_pick)}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Point estimate and Wilson score interval for a proportion."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


def _read(path: str | Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _norm(v: str) -> str | None:
    v = (v or "").strip().lower()
    if v in YES:
        return "yes"
    if v in NO:
        return "no"
    if v in PARTLY:
        return "partly"
    return None


def prefill(out_dir: str | Path) -> dict:
    """Copy an AI draft into the annotation files so a person can review and edit it in place.

    Writes the draft label into `faithful` (and the draft count into `missed_checkable_statements`) only where the cell is
    still blank, and puts the draft's reason in `notes` prefixed with 'AI draft:'. Never overwrites a human's entry.
    """
    d = Path(out_dir)
    claims_p, ex_p = d / "claims_to_check.csv", d / "explanations_to_check.csv"
    draft = {r["claim_row_id"]: r for r in _read(d / "claims_ai_draft.csv")}
    rows = _read(claims_p)
    n_claims = 0
    for r in rows:
        dr = draft.get(r["claim_row_id"])
        if dr and not (r["faithful"] or "").strip():
            r["faithful"] = dr["ai_draft_faithful"]
            r["notes"] = f"AI draft: {dr['ai_draft_reason']}" + (f" | {r['notes']}" if r["notes"] else "")
            n_claims += 1
    _write(claims_p, CLAIM_COLS, rows)
    n_ex = 0
    if (d / "explanations_ai_draft.csv").exists():
        edraft = {r["explanation_id"]: r for r in _read(d / "explanations_ai_draft.csv")}
        erows = _read(ex_p)
        for r in erows:
            dr = edraft.get(r["explanation_id"])
            if dr and not (r["missed_checkable_statements"] or "").strip():
                r["missed_checkable_statements"] = dr["ai_draft_missed"]
                r["notes"] = f"AI draft: {dr['ai_draft_reason']}" + (f" | {r['notes']}" if r["notes"] else "")
                n_ex += 1
        _write(ex_p, EXPL_COLS, erows)
    return {"claims_prefilled": n_claims, "explanations_prefilled": n_ex}


def draft_agreement(claims_csv: str | Path, draft_csv: str | Path) -> dict:
    """How far the human's final labels moved from an AI draft: agreement, Cohen's kappa and a confusion table.

    Reported next to any validation that started from a draft, because a person shown a draft tends to accept it
    (anchoring); a very high agreement is a warning sign, not reassurance.
    """
    draft = {r["claim_row_id"]: _norm(r["ai_draft_faithful"]) for r in _read(draft_csv)}
    pairs = [(draft.get(r["claim_row_id"]), _norm(r["faithful"])) for r in _read(claims_csv)]
    pairs = [(d, h) for d, h in pairs if d is not None and h is not None]
    n = len(pairs)
    if n == 0:
        return {"n": 0}
    cats = ["yes", "partly", "no"]
    conf = {d: {h: sum(1 for x, y in pairs if x == d and y == h) for h in cats} for d in cats}
    agree = sum(d == h for d, h in pairs) / n
    pe = sum((sum(1 for d, _ in pairs if d == c) / n) * (sum(1 for _, h in pairs if h == c) / n) for c in cats)
    kappa = (agree - pe) / (1 - pe) if pe < 1 else None
    return {
        "n": n,
        "agreement": agree,
        "changed": n - round(agree * n),
        "cohens_kappa": kappa,
        "confusion_draft_by_human": conf,
    }


def score(
    claims_csv: str | Path,
    key_csv: str | Path,
    explanations_csv: str | Path | None = None,
    draft_csv: str | Path | None = None,
) -> dict:
    """Score a filled-in annotation file against the answer key."""
    rows, key = _read(claims_csv), {r["claim_row_id"]: r for r in _read(key_csv)}
    done = [(r, _norm(r["faithful"])) for r in rows]
    unlabelled = sum(1 for _, v in done if v is None)
    done = [(r, v) for r, v in done if v is not None]
    n = len(done)
    out: dict = {"claims_annotated": n, "claims_unlabelled": unlabelled}
    if n:
        yes = sum(v == "yes" for _, v in done)
        partly = sum(v == "partly" for _, v in done)
        p, lo, hi = wilson(yes, n)
        out["extractor_precision"] = {
            "faithful": yes,
            "partly": partly,
            "unfaithful": n - yes - partly,
            "value": p,
            "ci95": [lo, hi],
        }
        by_kind: dict = defaultdict(lambda: [0, 0])
        for r, v in done:
            by_kind[r["kind"]][1] += 1
            by_kind[r["kind"]][0] += v == "yes"
        out["precision_by_kind"] = {k: {"faithful": a, "n": b} for k, (a, b) in sorted(by_kind.items())}
        confirmed = [r for r, v in done if v == "yes"]
        lab = [key[r["claim_row_id"]]["label"] for r in confirmed if r["claim_row_id"] in key]
        wrong = sum(x in ("contradicted", "artifact_misread") for x in lab)
        checkable = sum(x in ("supported", "contradicted", "artifact_misread") for x in lab)
        out["rate_on_confirmed_claims"] = {
            "wrong": wrong,
            "checkable": checkable,
            "value": (wrong / checkable) if checkable else None,
        }
        all_lab = [key[r["claim_row_id"]]["label"] for r, _ in done if r["claim_row_id"] in key]
        w2 = sum(x in ("contradicted", "artifact_misread") for x in all_lab)
        c2 = sum(x in ("supported", "contradicted", "artifact_misread") for x in all_lab)
        out["rate_on_all_sampled_claims"] = {"wrong": w2, "checkable": c2, "value": (w2 / c2) if c2 else None}
        out["enough_for_a_claim"] = n >= 100
    if explanations_csv:
        ex = [r for r in _read(explanations_csv) if (r.get("missed_checkable_statements") or "").strip() != ""]
        if ex:
            missed = sum(int(float(r["missed_checkable_statements"])) for r in ex)
            got = sum(int(r["n_extracted"]) for r in ex)
            out["estimated_recall"] = {
                "explanations": len(ex),
                "extracted": got,
                "missed": missed,
                "value": got / (got + missed) if got + missed else None,
            }
    if draft_csv and Path(draft_csv).exists():
        out["ai_draft_vs_final"] = draft_agreement(claims_csv, draft_csv)
    return out
