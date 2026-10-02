"""Programmatic verification of claims against ground truth. No LLM judge here."""
from __future__ import annotations

from .artifacts import ArtifactRecord
from .claims import Claim, Kind, Label, Verdict
from .groundtruth import GroundTruth


def _numeric(c: Claim, gt: GroundTruth) -> Verdict:
    p = c.payload
    q = p.get("quantity")
    if q not in gt.scalars or "value" not in p:
        return Verdict(c.id, Label.UNVERIFIABLE, f"unknown quantity {q!r} or missing value")
    truth, claimed = gt.scalars[q], float(p["value"])
    tol = max(float(p.get("abs_tol", 0.0)), float(p.get("rel_tol", 0.10)) * abs(truth))
    ok = abs(claimed - truth) <= tol
    return Verdict(c.id, Label.SUPPORTED if ok else Label.CONTRADICTED,
                   f"{q}: claimed {claimed:.3g}, computed {truth:.3g}, tolerance {tol:.3g}")


def _ranking(c: Claim, gt: GroundTruth) -> Verdict:
    p = c.payload
    q, k = p.get("quantity"), int(p.get("k", 0))
    residues = [int(r) for r in p.get("residues", [])]
    if q != "rmsf" or k <= 0 or not residues:
        return Verdict(c.id, Label.UNVERIFIABLE, "unsupported ranking payload")
    window = int(p.get("window", 0))
    truth = gt.top_residues(q, k, p.get("direction", "highest"))
    hits = sum(any(abs(r - t) <= window for t in truth) for r in residues)
    frac = hits / len(residues)
    need = float(p.get("min_hit_frac", 0.6))
    return Verdict(c.id, Label.SUPPORTED if frac >= need else Label.CONTRADICTED,
                   f"{hits}/{len(residues)} claimed residues in computed top-{k} {truth} (need {need:.0%}, window {window})")


def _temporal(c: Claim, gt: GroundTruth, art: ArtifactRecord | None) -> Verdict:
    p = c.payload
    event, frame = p.get("event"), p.get("frame")
    if event not in ("rmsd_peak", "rmsd_plateau") or frame is None:
        return Verdict(c.id, Label.UNVERIFIABLE, "unsupported temporal payload")
    window = int(p.get("window", 5))
    if art is not None and art.kind != "none" and p.get("physical", True):
        if art.kind == "shuffle_frames":
            return Verdict(c.id, Label.ARTIFACT_MISREAD,
                           "time-ordered claim on a trajectory whose frame order was shuffled")
        if any(abs(int(frame) - f) <= window for f in art.frames):
            return Verdict(c.id, Label.ARTIFACT_MISREAD,
                           f"claimed event at frame {frame} lies in artifact frames ({art.description})")
    truth = gt.scalars["rmsd_max_frame" if event == "rmsd_peak" else "rmsd_plateau_frame"]
    ok = abs(int(frame) - truth) <= window
    return Verdict(c.id, Label.SUPPORTED if ok else Label.CONTRADICTED,
                   f"{event}: claimed frame {frame}, computed {truth}, window {window}")


def _causal(c: Claim) -> Verdict:
    if not c.payload.get("evidence"):
        return Verdict(c.id, Label.UNSUPPORTED_MECHANISM, "causal claim with no cited computed evidence")
    return Verdict(c.id, Label.UNVERIFIABLE, "causal claim cites evidence; needs human review")


def verify(c: Claim, gt: GroundTruth, art: ArtifactRecord | None = None) -> Verdict:
    if c.kind is Kind.NUMERIC:
        return _numeric(c, gt)
    if c.kind is Kind.RANKING:
        return _ranking(c, gt)
    if c.kind is Kind.TEMPORAL:
        return _temporal(c, gt, art)
    return _causal(c)


def verify_all(claims, gt: GroundTruth, art: ArtifactRecord | None = None) -> list:
    return [verify(c, gt, art) for c in claims]
