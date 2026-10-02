from __future__ import annotations

import json

from ..tools import Toolbox
from .base import SYSTEM_PROMPT, Condition, Explanation, Task


class ToolAgent(Condition):
    """The model calls analysis tools on the trajectory and then explains. Bounded loop; the agent sees tool outputs
    only (no ground-truth scalars, no artifact record)."""

    name = "tool_agent"

    def __init__(self, client, max_steps: int = 8):
        super().__init__(client)
        self.max_steps = max_steps

    def preamble(self, task: Task) -> str:
        """Extra context placed before the question. Empty for the plain tool agent."""
        return ""

    def run(self, task: Task) -> Explanation:
        box = Toolbox(task.universe)
        pre = self.preamble(task)
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (pre + "\n\n" if pre else "") + task.question + " Use the tools to compute what you need.",
            },
        ]
        calls, n = [], 0
        for _ in range(self.max_steps):
            reply = self.client.complete(msgs, tools=Toolbox.SPECS)
            n += 1
            if not reply.tool_calls:
                return Explanation(reply.text, calls, n)
            msgs.append({"role": "assistant", "content": reply.text or json.dumps(reply.tool_calls)})
            for tc in reply.tool_calls:
                out = box.call(tc["name"], tc.get("arguments"))
                calls.append({"name": tc["name"], "arguments": tc.get("arguments", {})})
                msgs.append({"role": "tool", "content": json.dumps(out)[:20000]})
        final = self.client.complete([*msgs, {"role": "user", "content": "Give your final explanation now."}])
        return Explanation(final.text, calls, n + 1)
