from __future__ import annotations

from .. import render
from .base import SYSTEM_PROMPT, Condition, Explanation, Task


class ImageOnly(Condition):
    """Vision model sees only the rendered plots (RMSD, RMSF, contact map)."""

    name = "image_only"

    def run(self, task: Task) -> Explanation:
        images = render.plots(task.ground_truth)
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": task.question + " Three figures are attached: backbone RMSD vs frame, "
                "CA RMSF vs residue, and a CA contact-occupancy map.",
            },
        ]
        return Explanation(self.client.complete(msgs, images=images).text)
