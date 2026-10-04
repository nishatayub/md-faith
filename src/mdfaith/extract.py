"""Claim extraction: an LLM turns free-text explanations into structured atomic claims.

The extractor is itself an LLM, so its output is validated (`claims_from_json` drops malformed items) and a random
subset must be checked by hand before any result is reported (see docs/DESIGN.md, 'Validity threats').
"""

from __future__ import annotations

from .claims import CLAIM_JSON_SCHEMA, claims_from_json

EXTRACT_PROMPT = """Split the explanation below into atomic claims about a molecular dynamics trajectory.
Return ONLY a JSON array following this schema: {schema}

Kinds and payloads:
- numeric: {{"quantity": one of rmsd_mean|rmsd_max|rmsd_final|rmsf_mean|rmsf_max|rg_mean|rg_std|hbond_mean_count, "value": number}}
- ranking: {{"quantity": "rmsf", "direction": "highest"|"lowest", "k": int, "residues": [residue numbers]}}
- temporal: {{"event": "rmsd_peak"|"rmsd_plateau", "frame": int, "physical": true|false}}
  (physical=false only if the text says the event is an artifact or processing problem)
- causal: {{"evidence": [names of analyses the text cites for the mechanism, may be empty]}}
Skip statements that make no checkable or mechanistic assertion. Do not invent values.

Explanation:
{text}
"""


def extract_claims(client, explanation_text: str) -> list:
    prompt = EXTRACT_PROMPT.format(schema=CLAIM_JSON_SCHEMA, text=explanation_text)
    raw = client.complete([{"role": "user", "content": prompt}]).text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        return claims_from_json(raw)
    except ValueError:
        return []


class LLMExtractor:
    """Adapter giving an `LLMClient` the same `.extract(text)` interface as `extract_rules.RuleExtractor`."""

    def __init__(self, client):
        self.client = client
        self.name = f"llm:{getattr(client, 'name', 'unknown')}"

    def extract(self, text: str) -> list:
        return extract_claims(self.client, text)
