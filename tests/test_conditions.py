import json

import pytest

from mdfaith.conditions import ImageOnly, TableOnly, ToolAgent
from mdfaith.llm import Reply, ScriptedClient
from mdfaith.pipeline import build_tasks, score_explanation


@pytest.fixture(scope="module")
def tasks():
    return build_tasks(
        "adk_dims", [{"kind": "none"}, {"kind": "pbc_split", "frames": [30, 31], "resid_range": [1, 60]}]
    )


def test_image_only_sends_three_images(tasks):
    c = ScriptedClient(["explanation"])
    ImageOnly(c).run(tasks[0])
    assert c.calls[0]["n_images"] == 3


def test_table_only_contains_numbers_not_images(tasks):
    c = ScriptedClient(["explanation"])
    TableOnly(c).run(tasks[0])
    assert c.calls[0]["n_images"] == 0
    assert "backbone_RMSD_A" in c.calls[0]["messages"][-1]["content"]


def test_tool_agent_runs_tools_and_hides_ground_truth(tasks):
    c = ScriptedClient(
        [
            Reply(tool_calls=[{"name": "rmsd", "arguments": {}}]),
            Reply(tool_calls=[{"name": "nonexistent"}]),
            Reply(text="final explanation"),
        ]
    )
    exp = ToolAgent(c).run(tasks[0])
    assert exp.text == "final explanation" and [t["name"] for t in exp.tool_calls] == ["rmsd", "nonexistent"]
    sent = json.dumps(c.calls[-1]["messages"])
    assert "rmsd_plateau_frame" not in sent and "pbc_split" not in sent  # no leakage of GT or artifact record
    assert "unknown tool" in sent


def test_tool_agent_is_bounded(tasks):
    c = ScriptedClient([Reply(tool_calls=[{"name": "rmsf"}])] * 3 + ["forced final"])
    exp = ToolAgent(c, max_steps=3).run(tasks[0])
    assert exp.text == "forced final"


def test_frame_to_frame_tool_exposes_wrapping_artifact(tasks):
    from mdfaith.tools import Toolbox

    clean = Toolbox(tasks[0].universe).frame_to_frame_rmsd()["per_frame"]
    art = Toolbox(tasks[1].universe).frame_to_frame_rmsd()["per_frame"]
    assert max(art) > 5 * max(max(clean), 1.0)


def test_extraction_and_scoring_end_to_end(tasks):
    t = tasks[1]
    claims = json.dumps(
        [
            {
                "id": "1",
                "text": "There is a big change in RMSD at frame 30.",
                "kind": "temporal",
                "payload": {"event": "rmsd_peak", "frame": 30, "physical": True},
            }
        ]
    )
    res = score_explanation(t, "x", ScriptedClient([f"```json\n{claims}\n```"]))
    assert res["counts"]["artifact_misread"] == 1
