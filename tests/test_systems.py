import pytest

import mdfaith.data as data
from mdfaith.pipeline import tasks_from_ids


def test_adk_dims_is_always_available_and_equilibrium_is_listed_only_when_present(monkeypatch):
    assert "adk_dims" in data.available_systems()
    monkeypatch.setattr(
        data, "_equilibrium_paths", lambda download=False: (_ for _ in ()).throw(OSError("not downloaded"))
    )
    assert data.is_available("adk_equilibrium") is False and data.available_systems() == ["adk_dims"]
    with pytest.raises(FileNotFoundError, match="fetch-data"):
        data.load_system("adk_equilibrium")


def test_missing_optional_package_means_unavailable_not_an_error(monkeypatch):
    def boom(download=False):
        raise ImportError("no MDAnalysisData")

    monkeypatch.setattr(data, "_equilibrium_paths", boom)
    assert data.is_available("adk_equilibrium") is False


def test_load_never_downloads_and_fetch_is_the_only_downloader(monkeypatch):
    calls = []
    monkeypatch.setattr(data, "_equilibrium_paths", lambda download=False: calls.append(download) or ("t", "x"))
    assert data.is_available("adk_equilibrium")
    data.fetch_data("adk_equilibrium")
    assert calls == [False, True]
    with pytest.raises(KeyError):
        data.fetch_data("nope")
    with pytest.raises(KeyError):
        data.load_system("nope")


def test_tasks_from_ids_loads_each_system_once_and_builds_all_variants():
    ts = tasks_from_ids(["adk_dims:none", "adk_dims:pbc_split"])
    assert set(ts) == {"adk_dims:none", "adk_dims:pbc_split"}
    assert ts["adk_dims:pbc_split"].artifact.kind == "pbc_split"


@pytest.mark.skipif(
    not data.is_available("adk_equilibrium"), reason="adk_equilibrium not downloaded (mdfaith fetch-data)"
)
def test_equilibrium_system_loads_subsampled_and_runs_through_qc_and_ground_truth():
    from mdfaith.groundtruth import compute
    from mdfaith.qc import run_qc

    u = data.load_system("adk_equilibrium")
    assert len(u.select_atoms("protein").residues) == 214 and 90 <= len(u.trajectory) <= 120
    assert run_qc(u).passed
    gt = compute(u, "adk_equilibrium")
    assert gt.scalars["rmsd_mean"] > 0 and gt.scalars["hbond_mean_count"] > 50
