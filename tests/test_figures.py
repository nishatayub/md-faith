import pytest

from mdfaith.artifacts import DEFAULT_SPECS
from mdfaith.demo import seed_demo
from mdfaith.figures import DEMO_STAMP, make_figures
from mdfaith.pipeline import build_tasks
from mdfaith.store import Store


@pytest.fixture(scope="module")
def store_and_run():
    s = Store()
    tasks = build_tasks("adk_dims", [DEFAULT_SPECS[k] for k in ("none", "pbc_split", "shuffle_frames", "rigid_jitter")])
    return s, seed_demo(s, seeds=3, tasks=tasks)


def test_figures_and_tables_are_written(store_and_run, tmp_path):
    s, rid = store_and_run
    paths = make_figures(s, rid, tmp_path, with_trajectory=True)
    assert {p.name for p in paths} == {
        "fig1_hallucination_by_condition.png",
        "fig2_condition_by_artifact_heatmap.png",
        "fig3_artifact_misread_share.png",
        "fig4_artifact_rmsd.png",
    }
    assert all(p.stat().st_size > 5000 for p in paths)
    assert (tmp_path / "table_by_condition.csv").read_text().startswith("condition,")


def test_unknown_run_raises(tmp_path):
    with pytest.raises(KeyError):
        make_figures(Store(), "nope", tmp_path)


def test_demo_stamp_text_exists():
    assert "simulated" in DEMO_STAMP and "not findings" in DEMO_STAMP
