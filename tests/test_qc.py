import pytest

from mdfaith import artifacts
from mdfaith.conditions import QCGated
from mdfaith.data import load_system
from mdfaith.llm import Reply, ScriptedClient
from mdfaith.pipeline import build_tasks
from mdfaith.qc import QCConfig, run_qc


@pytest.fixture(scope="module")
def u():
    return load_system("adk_dims")


def test_clean_trajectory_has_no_critical_or_warning_flags(u):
    rep = run_qc(u)
    assert rep.passed
    assert not [f for f in rep.flags if f.severity in ("critical", "warning")], rep.to_prompt()


def test_pbc_wrap_is_detected_with_frames(u):
    u2, rec = artifacts.build_default(u, "pbc_split")
    rep = run_qc(u2)
    assert not rep.passed
    assert rep.has("broken_backbone") and rep.has("frame_discontinuity")
    broken = next(f for f in rep.flags if f.code == "broken_backbone")
    assert set(rec.frames) <= set(broken.frames)
    assert "30-32" in rep.to_prompt()


def test_shuffled_order_is_flagged(u):
    u2, _ = artifacts.build_default(u, "shuffle_frames")
    assert run_qc(u2).has("frame_order_unreliable")


def test_unfitted_trajectory_is_flagged(u):
    u2, _ = artifacts.build_default(u, "rigid_jitter")
    rep = run_qc(u2)
    assert rep.has("not_centred_or_fitted")
    assert rep.passed  # warning, not critical: fitted analyses are still valid


def test_too_few_frames_and_config_override(u):
    rep = run_qc(u, QCConfig(min_frames=1000))
    assert rep.has("too_few_frames")


def test_report_serialises(u):
    d = run_qc(u).to_dict()
    assert set(d) == {"n_frames", "passed", "flags"}


def test_qc_gated_condition_injects_report_into_prompt():
    tasks = build_tasks("adk_dims", [{"kind": "pbc_split", "frames": [30, 31], "resid_range": [1, 60]}])
    c = ScriptedClient([Reply(text="done")])
    QCGated(c).run(tasks[0])
    first_user = c.calls[0]["messages"][1]["content"]
    assert "broken_backbone" in first_user and "CRITICAL" in first_user
