"""Anthropic adapter implementing `LLMClient`.

Requires `pip install anthropic` (extra: `llm`) and credentials resolved by the SDK (ANTHROPIC_API_KEY, or an
`ant auth login` profile). No key is ever stored by this project.

Design notes for an *evaluation* harness:
- Server-side refusal fallbacks are deliberately NOT enabled. A silent fallback would change which model produced an
  explanation and contaminate per-model results. Refusals are returned as text `"[refused]"` and recorded as such.
- Thinking blocks are echoed back unchanged in tool loops via `Reply.raw`, as the API requires.
- Not exercised against the live API in CI; `tests/test_llm_anthropic.py` checks message conversion with a fake client.
"""

from __future__ import annotations

import base64

from .llm import Reply

DEFAULT_MODEL = "claude-opus-5-5"
_JSON_TYPES = {"int": "integer", "float": "number", "bool": "boolean", "str": "string"}


def tools_to_anthropic(specs: list) -> list:
    out = []
    for s in specs:
        props = {k: {"type": _JSON_TYPES.get(v, "string")} for k, v in s.get("parameters", {}).items()}
        out.append(
            {
                "name": s["name"],
                "description": s["description"],
                "input_schema": {"type": "object", "properties": props},
            }
        )
    return out


def messages_to_anthropic(messages: list, images: list | None = None) -> tuple[str, list]:
    """Convert MD-Faith messages to (system, anthropic messages). Images go on the last user message."""
    system_parts, out = [], []
    for m in messages:
        role = m["role"]
        if role == "system":
            system_parts.append(m["content"])
        elif role == "user":
            out.append({"role": "user", "content": m["content"]})
        elif role == "assistant":
            if m.get("raw") is not None:
                out.append({"role": "assistant", "content": m["raw"]})
            elif m.get("tool_calls"):
                blocks = ([{"type": "text", "text": m["content"]}] if m.get("content") else []) + [
                    {"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc.get("arguments") or {}}
                    for tc in m["tool_calls"]
                ]
                out.append({"role": "assistant", "content": blocks})
            else:
                out.append({"role": "assistant", "content": m["content"]})
        elif role == "tool":
            block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)  # parallel results go in ONE user message
            else:
                out.append({"role": "user", "content": [block]})
    if images:
        for msg in reversed(out):
            if msg["role"] == "user":
                text = msg["content"] if isinstance(msg["content"], str) else None
                if text is not None:
                    imgs = [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64.standard_b64encode(b).decode(),
                            },
                        }
                        for b in images
                    ]
                    msg["content"] = [*imgs, {"type": "text", "text": text}]
                break
    return "\n\n".join(system_parts), out


class AnthropicClient:
    def __init__(self, model: str = DEFAULT_MODEL, effort: str = "medium", max_tokens: int = 16000, client=None):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self._client = client
        self.model = model
        self.name = model
        self.effort = effort
        self.max_tokens = max_tokens

    def complete(self, messages: list, images: list | None = None, tools: list | None = None) -> Reply:
        system, msgs = messages_to_anthropic(messages, images)
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": msgs,
            "output_config": {"effort": self.effort},
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools_to_anthropic(tools)
        resp = self._client.messages.create(**kwargs)
        if resp.stop_reason == "refusal":
            return Reply(text="[refused]", raw=resp.content)
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [
            {"id": b.id, "name": b.name, "arguments": dict(b.input or {})} for b in resp.content if b.type == "tool_use"
        ]
        return Reply(text=text, tool_calls=calls, raw=resp.content)
