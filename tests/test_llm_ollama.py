from mdfaith.llm_ollama import OllamaClient, messages_to_ollama, tools_to_ollama
from mdfaith.runner import make_backend

TOOLS = [{"name": "rmsd", "description": "d", "parameters": {"start": "int", "label": "str"}}]


def test_tools_conversion():
    t = tools_to_ollama(TOOLS)[0]["function"]
    assert t["name"] == "rmsd"
    assert t["parameters"]["properties"]["start"] == {"type": "integer"}


def test_messages_roundtrip_with_tools_and_images():
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "x1", "name": "rmsd", "arguments": {"start": 1}}]},
        {"role": "tool", "tool_call_id": "x1", "content": "{}"},
    ]
    out = messages_to_ollama(msgs, images=[b"png"])
    assert out[2]["tool_calls"][0]["function"]["name"] == "rmsd"
    assert out[3]["tool_name"] == "rmsd"
    assert out[1]["images"]


def test_complete_parses_reply_and_tool_calls():
    seen = {}

    def transport(url, payload, timeout):
        seen.update(url=url, payload=payload)
        return {"message": {"content": "hi", "tool_calls": [{"function": {"name": "rmsd", "arguments": {"start": 2}}}]}}

    r = OllamaClient("m", transport=transport).complete([{"role": "user", "content": "q"}], tools=TOOLS)
    assert r.text == "hi"
    assert r.tool_calls == [{"id": "ollama-1", "name": "rmsd", "arguments": {"start": 2}}]
    assert seen["url"].endswith("/api/chat") and seen["payload"]["stream"] is False


def test_make_backend_ollama():
    assert make_backend("ollama:qwen2.5:3b").client.model == "qwen2.5:3b"


def test_seed_is_sent_and_extractor_drops_ungrounded_claims():
    import json

    from mdfaith.extract import extract_claims
    from mdfaith.llm import ScriptedClient

    sent = {}
    c = OllamaClient("m", transport=lambda u, p, t: sent.update(p) or {"message": {"content": "x"}})
    c.seed = 7
    c.complete([{"role": "user", "content": "q"}])
    assert sent["options"]["seed"] == 7

    items = [
        {
            "id": "1",
            "text": "The mean RMSD is 4.4 Å over the trajectory.",
            "kind": "numeric",
            "payload": {"quantity": "rmsd_mean", "value": 4.4},
        },
        {
            "id": "2",
            "text": "RMSD plot shows minimal fluctuations here.",
            "kind": "numeric",
            "payload": {"quantity": "rmsd_mean", "value": 0},
        },
        {"id": "3", "text": "Residue 131", "kind": "ranking", "payload": {"quantity": "rmsf", "residues": [131]}},
        {
            "id": "4",
            "text": "Residues 12 and 81 are the most flexible.",
            "kind": "ranking",
            "payload": {"quantity": "rmsf", "k": 2, "residues": ["12", "81"]},
        },
    ]
    kept = extract_claims(ScriptedClient([json.dumps(items)]), "t")
    assert [k.id for k in kept] == ["1", "4"]
    assert kept[1].payload["residues"] == [12, 81]
