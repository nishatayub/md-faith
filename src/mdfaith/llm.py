"""Provider-agnostic LLM interface. Real providers are adapters implementing `complete`; none is bundled so the
repo has no network dependency and no keys. `ScriptedClient` makes every condition testable offline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Reply:
    text: str = ""
    tool_calls: list = field(default_factory=list)  # [{"id": str, "name": str, "arguments": dict}]
    raw: object = None  # provider-specific content (e.g. thinking blocks) to echo back unchanged in tool loops


class LLMClient(Protocol):
    name: str

    def complete(self, messages: list, images: list | None = None, tools: list | None = None) -> Reply:
        """messages: [{"role": "system"|"user"|"assistant"|"tool", "content": str}]; images: PNG bytes."""
        ...


class ScriptedClient:
    """Returns pre-scripted replies in order. For tests and dry runs only."""

    name = "scripted"

    def __init__(self, replies: list):
        self._replies = list(replies)
        self.calls = []

    def complete(self, messages, images=None, tools=None) -> Reply:
        self.calls.append({"messages": messages, "n_images": len(images or []), "tools": tools})
        r = self._replies.pop(0)
        return r if isinstance(r, Reply) else Reply(text=str(r))
