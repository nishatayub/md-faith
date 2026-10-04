from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..artifacts import ArtifactRecord
from ..groundtruth import GroundTruth

SYSTEM_PROMPT = (
    "You are analysing a molecular dynamics trajectory of a protein. Explain what the analysis shows: structural "
    "stability, flexible regions, and any events. State only what the provided data supports. If something looks "
    "like a data or processing problem rather than physical motion, say so."
)


@dataclass
class Task:
    task_id: str
    system: str
    universe: object  # the (possibly artifact-containing) MDAnalysis Universe
    ground_truth: GroundTruth
    artifact: ArtifactRecord
    question: str = "Summarise the stability and flexibility of this protein over the trajectory."


@dataclass
class Explanation:
    text: str
    tool_calls: list = field(default_factory=list)
    n_model_calls: int = 1


class Condition(ABC):
    name: str

    def __init__(self, client):
        self.client = client

    @abstractmethod
    def run(self, task: Task) -> Explanation: ...
