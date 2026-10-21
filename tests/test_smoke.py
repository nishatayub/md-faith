from mdfaith.llm import Reply, ScriptedClient
from mdfaith.smoke import run_smoke


def test_smoke_passes_with_a_well_behaved_model():
    c = ScriptedClient(
        [
            Reply(text="ready"),
            Reply(tool_calls=[{"id": "t1", "name": "rmsd", "arguments": {"ref_frame": 0}}]),
            Reply(text="The RMSD rises from 0 to 0.9 Å."),
        ]
    )
    res = run_smoke(c, "scripted")
    assert res["passed"] and [s["ok"] for s in res["steps"]] == [True, True, True]


def test_smoke_reports_each_failure_without_raising():
    c = ScriptedClient([Reply(text="hello there"), Reply(text="I would rather chat")])
    res = run_smoke(c, "scripted")
    assert not res["passed"]
    assert [s["ok"] for s in res["steps"]] == [False, False, False]
    assert "skipped" in res["steps"][2]["detail"]


def test_smoke_survives_a_client_that_raises():
    class Boom:
        name = "boom"

        def complete(self, *a, **k):
            raise ConnectionError("no route")

    res = run_smoke(Boom())
    assert not res["passed"] and "ConnectionError" in res["steps"][0]["detail"]


def test_smoke_tool_call_without_id_fails_the_check():
    c = ScriptedClient([Reply(text="ready"), Reply(tool_calls=[{"name": "rmsd", "arguments": {}}])])
    assert run_smoke(c)["steps"][1]["ok"] is False
