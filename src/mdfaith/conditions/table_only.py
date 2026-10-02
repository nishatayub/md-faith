from __future__ import annotations

from .. import render
from .base import SYSTEM_PROMPT, Condition, Explanation, Task


class TableOnly(Condition):
    """Same information as ImageOnly, given as downsampled numeric tables."""

    name = "table_only"

    def run(self, task: Task) -> Explanation:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task.question + "\n\n" + render.tables(task.ground_truth)}]
        return Explanation(self.client.complete(msgs).text)
