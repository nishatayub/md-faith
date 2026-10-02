from __future__ import annotations

from ..qc import run_qc
from .base import Task
from .tool_agent import ToolAgent


class QCGated(ToolAgent):
    """Tool agent that is shown the automated QC report before it explains (the 'QC gate')."""

    name = "qc_gated"

    def preamble(self, task: Task) -> str:
        return run_qc(task.universe).to_prompt()
