"""Ollama adapter implementing `LLMClient` for free, local, open-weights models.

Talks to a local Ollama server over HTTP (default http://127.0.0.1:11434), so it needs no extra Python package, no key
and no network beyond localhost. Start the server and pull a model first, e.g. `ollama pull qwen2.5:3b`.

Notes for an *evaluation* harness:
- Use a model with tool-calling support for the tool condition, and a vision model (e.g. `llava`) for image conditions.
- Ollama does not return tool-call ids, so stable ids are generated and matched back on the next turn.
- Not exercised against a live server in CI; `tests/test_llm_ollama.py` checks conversion with a fake transport.
"""

from __future__ import annotations

import base64
import json
import time
import urllib.request

from .llm import Reply

DEFAULT_MODEL = "qwen2.5:3b"
DEFAULT_HOST = "http://127.0.0.1:11434"
_JSON_TYPES = {"int": "integer", "float": "number", "bool": "boolean", "str": "string"}


def tools_to_ollama(specs: list) -> list:
    out = []
    for s in specs:
        props = {k: {"type": _JSON_TYPES.get(v, "string")} for k, v in s.get("parameters", {}).items()}
        out.append(
            {
                "type": "function",
                "function": {
                    "name": s["name"],
                    "description": s["description"],
                    "parameters": {"type": "object", "properties": props},
                },
            }
        )
    return out


def messages_to_ollama(messages: list, images: list | None = None) -> list:
    """Convert MD-Faith messages to Ollama chat messages. Images go on the last user message."""
    id_to_name, out = {}, []
    for m in messages:
        role = m["role"]
        if role == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                id_to_name[tc["id"]] = tc["name"]
            out.append(
                {
                    "role": "assistant",
                    "content": m.get("content") or "",
                    "tool_calls": [
                        {"function": {"name": tc["name"], "arguments": tc.get("arguments") or {}}}
                        for tc in m["tool_calls"]
                    ],
                }
            )
        elif role == "tool":
            out.append({"role": "tool", "tool_name": id_to_name.get(m["tool_call_id"], ""), "content": m["content"]})
        else:
            out.append({"role": role, "content": m["content"]})
    if images:
        for msg in reversed(out):
            if msg["role"] == "user":
                msg["images"] = [base64.standard_b64encode(b).decode() for b in images]
                break
    return out


def _http_post(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


class OllamaClient:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        temperature: float = 0.7,
        timeout: float = 120.0,
        max_tokens: int = 1500,
        transport=None,
    ):
        self.model = model
        self.name = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens  # caps runaway repetition loops that small models fall into
        self.seed = 0  # set per call by the runner so seeds give reproducible but different samples
        self.timeout = timeout
        self._post = transport or _http_post
        self._n_calls = 0

    def _post_with_retry(self, url: str, payload: dict, attempts: int = 3) -> dict:
        """Retry dropped or timed-out requests (e.g. after the laptop slept) instead of failing a long run."""
        for i in range(attempts):
            try:
                return self._post(url, payload, self.timeout)
            except (TimeoutError, OSError):
                if i == attempts - 1:
                    raise
                time.sleep(5)

    def complete(self, messages: list, images: list | None = None, tools: list | None = None) -> Reply:
        payload = {
            "model": self.model,
            "messages": messages_to_ollama(messages, images),
            "stream": False,
            "keep_alive": "1h",
            "options": {"temperature": self.temperature, "seed": self.seed, "num_predict": self.max_tokens},
        }
        if tools:
            payload["tools"] = tools_to_ollama(tools)
        data = self._post_with_retry(f"{self.host}/api/chat", payload)
        msg = data.get("message") or {}
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            self._n_calls += 1
            calls.append(
                {
                    "id": f"ollama-{self._n_calls}",
                    "name": fn.get("name", ""),
                    "arguments": dict(fn.get("arguments") or {}),
                }
            )
        return Reply(text=msg.get("content") or "", tool_calls=calls, raw=data)
