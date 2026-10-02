"""Atomic-claim schema. An explanation is split into claims; each claim is checked on its own."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum


class Kind(str, Enum):
    NUMERIC = "numeric"      # a value of a scalar quantity
    RANKING = "ranking"      # which residues are highest / lowest in a per-residue quantity
    TEMPORAL = "temporal"    # when something happens (frame index)
    CAUSAL = "causal"        # a mechanistic or causal statement; not checkable from the trajectory alone


class Label(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"                  # checkable and wrong
    ARTIFACT_MISREAD = "artifact_misread"          # artifact frames presented as a physical event
    UNSUPPORTED_MECHANISM = "unsupported_mechanism"  # causal claim with no cited evidence
    UNVERIFIABLE = "unverifiable"                  # cannot be checked (unknown quantity, or needs a human)


@dataclass
class Claim:
    id: str
    text: str
    kind: Kind
    payload: dict = field(default_factory=dict)


@dataclass
class Verdict:
    claim_id: str
    label: Label
    detail: str = ""


CLAIM_JSON_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "required": ["id", "text", "kind", "payload"],
        "properties": {
            "id": {"type": "string"},
            "text": {"type": "string"},
            "kind": {"enum": [k.value for k in Kind]},
            "payload": {"type": "object"},
        },
    },
}


def claims_from_json(s: str) -> list:
    """Parse extractor output. Malformed items are dropped, not guessed at."""
    data = json.loads(s)
    out = []
    for item in data:
        try:
            out.append(Claim(str(item["id"]), str(item["text"]), Kind(item["kind"]), dict(item.get("payload", {}))))
        except (KeyError, ValueError, TypeError):
            continue
    return out
