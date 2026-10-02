import types

import pytest

from mdfaith.extract_rules import RuleExtractor
from mdfaith.llm_anthropic import AnthropicClient, messages_to_anthropic, tools_to_anthropic
from mdfaith.pipeline import build_tasks
from mdfaith.report import annotate
from mdfaith.simulated import CONDITIONS, PERSONAS, SimulatedModel
from mdfaith.tools import Toolbox
from mdfaith.verify import verify_all


@pytest.fixture(scope="module")
def tasks():
    return build_tasks(
        "adk_dims", [{"kind": "none"}, {"kind": "pbc_split", "frames": [30, 31], "resid_range": [1, 60]}]
    )


def test_simulated_is_deterministic_and_parseable(tasks):
    m = SimulatedModel("sim-careful")
    a = m.explain(tasks[0], "tool_agent", seed=1)
    b = m.explain(tasks[0], "tool_agent", seed=1)
    assert a.text == b.text
    claims = RuleExtractor().extract(a.text)
    assert {"numeric", "ranking", "temporal"} <= {c.kind.value for c in claims}


def test_simulated_tool_condition_is_more_faithful_than_image_condition(tasks):
    def wrong_rate(persona, cond):
        wrong = total = 0
        for seed in range(30):
            for t in tasks:
                text = SimulatedModel(persona).explain(t, cond, seed).text
                cs = RuleExtractor().extract(text)
                vs = verify_all(cs, t.ground_truth, t.artifact)
                wrong += sum(v.label.value in ("contradicted", "artifact_misread") for v in vs)
                total += sum(v.label.value != "unverifiable" for v in vs)
        return wrong / total

    assert wrong_rate("sim-hasty", "image_only") > wrong_rate("sim-hasty", "tool_agent") + 0.1
    assert wrong_rate("sim-hasty", "tool_agent") > wrong_rate("sim-hasty", "qc_gated") - 0.01


def test_simulated_qc_gate_catches_artifact_more_often(tasks):
    pbc = tasks[1]

    def misreads(cond):
        n = 0
        for seed in range(40):
            text = SimulatedModel("sim-hasty").explain(pbc, cond, seed).text
            cs = RuleExtractor().extract(text)
            n += any(v.label.value == "artifact_misread" for v in verify_all(cs, pbc.ground_truth, pbc.artifact))
        return n

    assert misreads("qc_gated") < misreads("image_only")


def test_all_personas_and_conditions_run(tasks):
    for p in PERSONAS:
        for c in CONDITIONS:
            text = SimulatedModel(p).explain(tasks[0], c).text
            cs = RuleExtractor().extract(text)
            assert annotate(text, cs, verify_all(cs, tasks[0].ground_truth), tasks[0].ground_truth)


def test_tools_to_anthropic_schema():
    t = {x["name"]: x for x in tools_to_anthropic(Toolbox.SPECS)}
    assert t["rmsd"]["input_schema"]["properties"]["superpose"] == {"type": "boolean"}
    assert t["rmsd"]["input_schema"]["properties"]["ref_frame"] == {"type": "integer"}


def test_message_conversion_tool_loop_and_images():
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "explain"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "a", "name": "rmsd", "arguments": {}}], "raw": None},
        {"role": "tool", "tool_call_id": "a", "content": "{}"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "b", "name": "rmsf", "arguments": {}}], "raw": None},
        {"role": "tool", "tool_call_id": "b", "content": "{}"},
        {"role": "tool", "tool_call_id": "c", "content": "{}"},
    ]
    system, out = messages_to_anthropic(msgs)
    assert system == "SYS"
    assert [m["role"] for m in out] == ["user", "assistant", "user", "assistant", "user"]
    assert len(out[-1]["content"]) == 2  # parallel tool results share one user message
    assert out[1]["content"][0]["type"] == "tool_use"
    _, out2 = messages_to_anthropic([{"role": "user", "content": "q"}], images=[b"\x89PNG"])
    assert out2[0]["content"][0]["type"] == "image" and out2[0]["content"][-1]["text"] == "q"


def test_raw_blocks_are_echoed_unchanged():
    raw = [types.SimpleNamespace(type="thinking")]
    _, out = messages_to_anthropic([{"role": "user", "content": "q"}, {"role": "assistant", "content": "", "raw": raw}])
    assert out[1]["content"] is raw


def test_anthropic_client_with_fake_sdk_client():
    blocks = [
        types.SimpleNamespace(type="text", text="hello"),
        types.SimpleNamespace(type="tool_use", id="t1", name="rmsd", input={"ref_frame": 0}),
    ]
    calls = []

    def create(**kw):
        calls.append(kw)
        return types.SimpleNamespace(stop_reason="tool_use", content=blocks)

    fake = types.SimpleNamespace(messages=types.SimpleNamespace(create=create))
    c = AnthropicClient(client=fake, effort="low")
    r = c.complete([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}], tools=Toolbox.SPECS)
    assert r.text == "hello" and r.tool_calls == [{"id": "t1", "name": "rmsd", "arguments": {"ref_frame": 0}}]
    kw = calls[0]
    assert kw["model"] == "claude-opus-5-5" and kw["output_config"] == {"effort": "low"} and kw["system"] == "S"
    assert "thinking" not in kw and "fallbacks" not in kw and "betas" not in kw


def test_refusal_is_recorded_not_hidden():
    fake = types.SimpleNamespace(
        messages=types.SimpleNamespace(create=lambda **kw: types.SimpleNamespace(stop_reason="refusal", content=[]))
    )
    assert AnthropicClient(client=fake).complete([{"role": "user", "content": "x"}]).text == "[refused]"
