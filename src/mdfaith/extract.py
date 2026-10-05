"""Claim extraction: an LLM turns free-text explanations into structured atomic claims.

The extractor is itself an LLM, so its output is validated (`claims_from_json` drops malformed items) and a random
subset must be checked by hand before any result is reported (see docs/DESIGN.md, 'Validity threats').
"""

from __future__ import annotations

import re

from .claims import CLAIM_JSON_SCHEMA, Kind, claims_from_json

_NUM = re.compile(r"-?\d+(?:\.\d+)?")
MIN_CLAIM_CHARS = 25


def _nums(text: str) -> set:
    return {float(x) for x in _NUM.findall(text)}


def _grounded(c) -> bool:
    """A claim is kept only if the numbers in its payload literally appear in its source sentence.
    Small extractors invent values (null, 0, wrong residues); those are dropped rather than scored."""
    if len(c.text.strip()) < MIN_CLAIM_CHARS:
        return False
    p, have = c.payload, _nums(c.text)
    try:
        if c.kind is Kind.NUMERIC:
            v = p.get("value")
            return isinstance(v, (int, float)) and not isinstance(v, bool) and any(abs(v - h) < 1e-9 for h in have)
        if c.kind is Kind.RANKING:
            res = [int(r) for r in p.get("residues") or []]
            p["residues"] = res
            p.pop("direction: ", None)
            return bool(res) and all(float(r) in have for r in res) and p.get("quantity") == "rmsf"
        if c.kind is Kind.TEMPORAL:
            f = p.get("frame")
            return isinstance(f, (int, float)) and float(f) in have and p.get("event") in ("rmsd_peak", "rmsd_plateau")
    except (TypeError, ValueError):
        return False
    return True  # causal


EXTRACT_PROMPT = """Split the explanation below into atomic claims about a molecular dynamics trajectory.
Return ONLY a JSON array following this schema: {schema}

Kinds and payloads:
- numeric: {{"quantity": one of rmsd_mean|rmsd_max|rmsd_final|rmsf_mean|rmsf_max|rg_mean|rg_std|hbond_mean_count, "value": number}}
- ranking: {{"quantity": "rmsf", "direction": "highest"|"lowest", "k": int, "residues": [residue numbers]}}
- temporal: {{"event": "rmsd_peak"|"rmsd_plateau", "frame": int, "physical": true|false}}
  (physical=false only if the text says the event is an artifact or processing problem)
- causal: {{"evidence": [names of analyses the text cites for the mechanism, may be empty]}}
Rules: extract a claim ONLY from a full sentence that itself states the value, residues or frame. Copy numbers and
residue/frame numbers exactly as written in that sentence; never infer, round to zero, or use null. Ignore table rows,
column headers, lists of tool names, and sentences that only describe what could be done. Skip statements that make no
checkable or mechanistic assertion. Do not invent values. "text" must be the source sentence.

Explanation:
{text}
"""


def extract_claims(client, explanation_text: str) -> list:
    prompt = EXTRACT_PROMPT.format(schema=CLAIM_JSON_SCHEMA, text=explanation_text)
    raw = client.complete([{"role": "user", "content": prompt}]).text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        claims = claims_from_json(raw)
    except ValueError:
        return []
    return [c for c in claims if _grounded(c)]


class LLMExtractor:
    """Adapter giving an `LLMClient` the same `.extract(text)` interface as `extract_rules.RuleExtractor`."""

    def __init__(self, client):
        self.client = client
        self.name = f"llm:{getattr(client, 'name', 'unknown')}"

    def extract(self, text: str) -> list:
        try:
            return extract_claims(self.client, text)
        except (TimeoutError, OSError):  # one stuck extraction must not kill a long run; it yields no claims
            self.n_failed = getattr(self, "n_failed", 0) + 1
            return []
