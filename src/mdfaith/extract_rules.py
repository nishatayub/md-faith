"""Deterministic, rule-based claim extractor for a controlled sentence grammar.

Purpose: offline demos, tests and the simulated models in `simulated.py`. It understands the sentence patterns listed
in `PATTERNS` and nothing else, so it is NOT a replacement for the LLM extractor in `extract.py` on free-form model
output. Every claim carries `payload["_sentence"]`, the index of the sentence it came from.
"""

from __future__ import annotations

import re

from .claims import Claim, Kind

NUM = r"([0-9]+(?:\.[0-9]+)?)"
UNIT = r"\s*(?:Å|A\b|angstroms?)"
ARTIFACT_WORDS = re.compile(r"artifact|artefact|wrapp|periodic|pbc|processing|not physical|unphysical|spurious", re.I)
CAUSAL_MARKERS = re.compile(
    r"\b(because|due to|caused by|driven by|results? from|indicat\w+|suggest\w+|impl\w+|reflect\w+|"
    r"hinge|allosteric|conformational change|unfold\w*|denatur\w+)\b",
    re.I,
)
EVIDENCE_MARKERS = re.compile(r"(as shown by|supported by|see (?:the )?)\s*([A-Za-z\- ]+)", re.I)

NUMERIC_PATTERNS = [
    ("rmsd_mean", re.compile(r"(?:mean|average)\s+(?:backbone\s+)?rmsd[^0-9]{0,30}" + NUM + UNIT, re.I)),
    ("rmsd_max", re.compile(r"(?:max(?:imum)?|peak)\s+(?:backbone\s+)?rmsd[^0-9]{0,30}" + NUM + UNIT, re.I)),
    ("rmsd_final", re.compile(r"final\s+(?:backbone\s+)?rmsd[^0-9]{0,30}" + NUM + UNIT, re.I)),
    ("rmsf_mean", re.compile(r"(?:mean|average)\s+(?:ca\s+)?rmsf[^0-9]{0,30}" + NUM + UNIT, re.I)),
    ("rmsf_max", re.compile(r"max(?:imum)?\s+(?:ca\s+)?rmsf[^0-9]{0,30}" + NUM + UNIT, re.I)),
    ("rg_mean", re.compile(r"radius of gyration[^0-9]{0,40}" + NUM + UNIT, re.I)),
    (
        "hbond_mean_count",
        re.compile(r"(?:about|around|approximately|~)?\s*" + NUM + r"\s+hydrogen bonds?\s+per frame", re.I),
    ),
]
RANKING = re.compile(
    r"residues?\s+((?:\d+\s*(?:,|and|&)?\s*)+)[^.]*?\b(highest|most|lowest|least)\b[^.]*?(?:rmsf|flexib|fluctuat|mobil)",
    re.I,
)
PEAK = re.compile(r"rmsd\s+(?:peaks?|spikes?|jumps?|reaches (?:its )?(?:maximum|peak))[^0-9]{0,25}frame\s+(\d+)", re.I)
PLATEAU = re.compile(
    r"(?:rmsd\s+)?(?:plateaus?|stabili[sz]es|equilibrates|converges|levels off)[^0-9]{0,25}frame\s+(\d+)", re.I
)

PATTERNS = [
    "Mean/average/maximum/final backbone RMSD is X Å.",
    "Mean/maximum CA RMSF is X Å.",
    "The radius of gyration is X Å.",
    "About N hydrogen bonds per frame.",
    "Residues A, B and C show the highest (or lowest) RMSF.",
    "RMSD peaks at frame N. / RMSD plateaus after frame N.",
    "Causal phrases (because, indicating, hinge, ...) become causal claims.",
]


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text.strip())
    return [p.strip() for p in parts if p.strip()]


class RuleExtractor:
    name = "rule_based"

    def extract(self, text: str) -> list[Claim]:
        claims: list[Claim] = []
        n = 0

        def add(sent: str, idx: int, kind: Kind, payload: dict):
            nonlocal n
            n += 1
            claims.append(Claim(f"c{n}", sent, kind, {**payload, "_sentence": idx}))

        for idx, sent in enumerate(split_sentences(text)):
            for quantity, pat in NUMERIC_PATTERNS:
                m = pat.search(sent)
                if m:
                    add(sent, idx, Kind.NUMERIC, {"quantity": quantity, "value": float(m.group(1))})
            m = RANKING.search(sent)
            if m:
                residues = [int(x) for x in re.findall(r"\d+", m.group(1))]
                direction = "highest" if m.group(2).lower() in ("highest", "most") else "lowest"
                add(
                    sent,
                    idx,
                    Kind.RANKING,
                    {"quantity": "rmsf", "direction": direction, "k": max(5, len(residues)), "residues": residues},
                )
            physical = not ARTIFACT_WORDS.search(sent)
            m = PEAK.search(sent)
            if m:
                add(sent, idx, Kind.TEMPORAL, {"event": "rmsd_peak", "frame": int(m.group(1)), "physical": physical})
            m = PLATEAU.search(sent)
            if m:
                add(
                    sent,
                    idx,
                    Kind.TEMPORAL,
                    {"event": "rmsd_plateau", "frame": int(m.group(1)), "physical": physical},
                )
            if CAUSAL_MARKERS.search(sent) and physical:
                ev = EVIDENCE_MARKERS.search(sent)
                add(sent, idx, Kind.CAUSAL, {"evidence": [ev.group(2).strip()] if ev else []})
        return claims
