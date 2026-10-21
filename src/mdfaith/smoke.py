"""Live smoke test for a model adapter: three short calls that prove the plumbing works against the real service.

1. a plain completion
2. a request that should trigger a tool call (tests tool-schema conversion and the tool-call parse)
3. the follow-up turn carrying the tool result back (tests tool-call ids and, for hosted models, echoing raw blocks)

Cost: three short calls. It checks that the *adapter* works, not that the model is good.
"""

from __future__ import annotations

import json
import time

from .tools import Toolbox


def _step(name: str, fn) -> dict:
    t0 = time.time()
    try:
        ok, detail = fn()
        return {"step": name, "ok": bool(ok), "detail": detail, "seconds": round(time.time() - t0, 1)}
    except Exception as e:  # report, never raise: a smoke test must say what broke
        return {
            "step": name,
            "ok": False,
            "detail": f"{type(e).__name__}: {str(e)[:200]}",
            "seconds": round(time.time() - t0, 1),
        }


def run_smoke(client, model_name: str | None = None) -> dict:
    msgs: list = [{"role": "system", "content": "You are a terse assistant."}]
    state: dict = {}

    def plain():
        r = client.complete([*msgs, {"role": "user", "content": "Reply with exactly one word: ready"}])
        return "ready" in r.text.lower(), r.text.strip()[:80]

    def tool_call():
        u = {
            "role": "user",
            "content": "Call the rmsd tool with ref_frame 0 to compute backbone RMSD. Do not answer in text.",
        }
        r = client.complete([*msgs, u], tools=Toolbox.SPECS)
        state["u"], state["r"] = u, r
        calls = r.tool_calls
        good = bool(calls) and calls[0]["name"] == "rmsd" and all("id" in c for c in calls)
        return good, json.dumps([{"name": c["name"], "arguments": c["arguments"]} for c in calls])[
            :120
        ] if calls else f"no tool call; text={r.text[:60]!r}"

    def follow_up():
        r, u = state["r"], state["u"]
        for i, c in enumerate(r.tool_calls):
            c.setdefault("id", f"call_{i}")
        convo = [
            *msgs,
            u,
            {
                "role": "assistant",
                "content": r.text or json.dumps(r.tool_calls),
                "tool_calls": r.tool_calls,
                "raw": r.raw,
            },
            *[
                {
                    "role": "tool",
                    "tool_call_id": c["id"],
                    "content": json.dumps({"per_frame": [0.0, 0.4, 0.9], "superposed": True}),
                }
                for c in r.tool_calls
            ],
        ]
        out = client.complete(convo, tools=Toolbox.SPECS)
        return bool(out.text.strip()) or bool(out.tool_calls), (out.text.strip() or "tool call")[:100]

    steps = [_step("plain completion", plain), _step("tool call", tool_call)]
    steps.append(
        _step("tool result round trip", follow_up)
        if steps[1]["ok"]
        else {
            "step": "tool result round trip",
            "ok": False,
            "detail": "skipped: no tool call to answer",
            "seconds": 0.0,
        }
    )
    return {
        "model": model_name or getattr(client, "name", "unknown"),
        "passed": all(s["ok"] for s in steps),
        "steps": steps,
    }
