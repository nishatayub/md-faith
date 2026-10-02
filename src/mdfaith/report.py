"""Evidence-linked reports: annotate every sentence of an explanation with a verdict and the computed evidence.

`annotate()` is the product surface of MD-Faith: given an AI-written explanation, the trajectory's ground truth and the
verifier's verdicts, it returns each sentence tagged `supported`, `contradicted`, `artifact`, `unsupported` or
`unchecked`, with the computed number, how it was computed, and (for artifacts) the planted/QC record.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from .artifacts import ArtifactRecord
from .claims import Claim, Kind, Label, Verdict
from .extract_rules import split_sentences
from .groundtruth import GroundTruth

# quantity -> (tool that reproduces it, definition shown to the reader)
DEFINITIONS = {
    "rmsd_mean": ("rmsd", "Mean over frames of backbone RMSD after superposition on frame 0."),
    "rmsd_max": ("rmsd", "Maximum over frames of backbone RMSD after superposition on frame 0."),
    "rmsd_final": ("rmsd", "Backbone RMSD of the last frame after superposition on frame 0."),
    "rmsf_mean": ("rmsf", "Mean over residues of CA RMSF after fitting."),
    "rmsf_max": ("rmsf", "Largest per-residue CA RMSF after fitting."),
    "rg_mean": ("radius_of_gyration", "Mean protein radius of gyration over frames."),
    "hbond_mean_count": ("hbonds", "Mean hydrogen bonds per frame (donor-acceptor < 3.0 A, angle > 150 deg)."),
    "rmsd_peak": ("rmsd", "Frame with the largest backbone RMSD."),
    "rmsd_plateau": ("rmsd", "First frame after which RMSD stays within 15% of its own later mean."),
    "rmsf_ranking": ("rmsf", "Residues ranked by CA RMSF after fitting."),
}

STATUS_ORDER = ["contradicted", "artifact", "unsupported", "supported", "unchecked"]
LABEL_TO_STATUS = {
    Label.SUPPORTED: "supported",
    Label.CONTRADICTED: "contradicted",
    Label.ARTIFACT_MISREAD: "artifact",
    Label.UNSUPPORTED_MECHANISM: "unsupported",
    Label.UNVERIFIABLE: "unchecked",
}


@dataclass
class Evidence:
    claim_id: str
    kind: str
    label: str
    detail: str
    computed: object = None
    tool: str | None = None
    definition: str | None = None


@dataclass
class AnnotatedSentence:
    index: int
    text: str
    status: str
    evidence: list = field(default_factory=list)


def _tokens(s: str) -> set:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def align(sentences: list[str], claims: list[Claim]) -> dict:
    """claim id -> sentence index. Uses the extractor's `_sentence` when present, else best token overlap."""
    out = {}
    toks = [_tokens(s) for s in sentences]
    for c in claims:
        if "_sentence" in c.payload and 0 <= int(c.payload["_sentence"]) < len(sentences):
            out[c.id] = int(c.payload["_sentence"])
            continue
        ct = _tokens(c.text)
        best, best_j = None, 0.0
        for i, t in enumerate(toks):
            j = len(ct & t) / max(1, len(ct | t))
            if j > best_j:
                best, best_j = i, j
        if best is not None and best_j >= 0.2:
            out[c.id] = best
    return out


def _evidence(c: Claim, v: Verdict, gt: GroundTruth, art: ArtifactRecord | None) -> Evidence:
    ev = Evidence(c.id, c.kind.value, v.label.value, v.detail)
    p = c.payload
    if c.kind is Kind.NUMERIC and p.get("quantity") in gt.scalars:
        ev.computed = round(float(gt.scalars[p["quantity"]]), 3)
        ev.tool, ev.definition = DEFINITIONS.get(p["quantity"], (None, None))
    elif c.kind is Kind.RANKING and p.get("quantity") == "rmsf":
        ev.computed = gt.top_residues("rmsf", int(p.get("k", 5)), p.get("direction", "highest"))
        ev.tool, ev.definition = DEFINITIONS["rmsf_ranking"]
    elif c.kind is Kind.TEMPORAL and p.get("event") in ("rmsd_peak", "rmsd_plateau"):
        key = "rmsd_max_frame" if p["event"] == "rmsd_peak" else "rmsd_plateau_frame"
        ev.computed = int(gt.scalars[key])
        ev.tool, ev.definition = DEFINITIONS[p["event"]]
        if v.label is Label.ARTIFACT_MISREAD and art is not None:
            ev.definition = f"{ev.definition} Artifact: {art.description}"
    return ev


def annotate(
    text: str,
    claims: list[Claim],
    verdicts: list[Verdict],
    gt: GroundTruth,
    art: ArtifactRecord | None = None,
) -> list[AnnotatedSentence]:
    sentences = split_sentences(text)
    amap = align(sentences, claims)
    vmap = {v.claim_id: v for v in verdicts}
    out = [AnnotatedSentence(i, s, "unchecked") for i, s in enumerate(sentences)]
    for c in claims:
        i, v = amap.get(c.id), vmap.get(c.id)
        if i is None or v is None:
            continue
        out[i].evidence.append(_evidence(c, v, gt, art))
    for a in out:
        statuses = [LABEL_TO_STATUS[Label(e.label)] for e in a.evidence]
        a.status = min(statuses, key=STATUS_ORDER.index) if statuses else "unchecked"
    return out


def summary(annotated: list[AnnotatedSentence]) -> dict:
    d = {s: 0 for s in STATUS_ORDER}
    for a in annotated:
        d[a.status] += 1
    d["n_sentences"] = len(annotated)
    return d


def to_dicts(annotated: list[AnnotatedSentence]) -> list[dict]:
    return [asdict(a) for a in annotated]


_MARK = {
    "supported": "OK",
    "contradicted": "WRONG",
    "artifact": "ARTIFACT",
    "unsupported": "UNSUPPORTED",
    "unchecked": "-",
}


def to_markdown(annotated: list[AnnotatedSentence]) -> str:
    """Plain-text report: each sentence prefixed with its status, followed by indented evidence lines."""
    lines = []
    for a in annotated:
        lines.append(f"[{_MARK[a.status]}] {a.text}")
        for e in a.evidence:
            comp = f" computed={e.computed}" if e.computed is not None else ""
            tool = f" (tool: {e.tool})" if e.tool else ""
            lines.append(f"    - {e.label}: {e.detail}{comp}{tool}")
    return "\n".join(lines)
