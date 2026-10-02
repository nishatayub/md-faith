import numpy as np
import pytest

from mdfaith import artifacts, groundtruth, metrics
from mdfaith.claims import Claim, Kind, Label, claims_from_json
from mdfaith.data import load_system
from mdfaith.verify import verify


@pytest.fixture(scope="module")
def u():
    return load_system("adk_dims")


@pytest.fixture(scope="module")
def gt(u):
    return groundtruth.compute(u, "adk_dims")


def test_groundtruth_shapes_and_sanity(u, gt):
    assert gt.n_frames == len(u.trajectory)
    assert len(gt.rmsd) == gt.n_frames and gt.rmsd[0] == pytest.approx(0.0, abs=1e-6)
    assert len(gt.rmsf) == len(gt.resids) == 214
    assert gt.contacts.shape == (214, 214) and np.allclose(gt.contacts, gt.contacts.T)
    assert 0 < gt.scalars["rg_mean"] < 40
    assert gt.scalars["rmsd_max"] > 0


def test_kabsch_recovers_rigid_motion():
    rng = np.random.default_rng(0)
    P = rng.normal(size=(50, 3))
    R = artifacts._random_rotation(rng)
    Q = P @ R + np.array([3.0, -2.0, 7.0])
    assert np.allclose(groundtruth.superpose(Q, P), P, atol=1e-8)


def test_pbc_split_creates_rmsd_spike_in_affected_frames(u, gt):
    u2, rec = artifacts.pbc_split(u, frames=[30, 31], resid_range=(1, 60), box_len=80.0)
    g2 = groundtruth.compute(u2)
    assert rec.frames == [30, 31]
    assert g2.rmsd[30] > 5 * max(gt.rmsd[30], 1.0)
    assert g2.scalars["rmsd_max_frame"] in (30, 31)
    assert g2.rmsd[10] == pytest.approx(gt.rmsd[10], abs=1e-3)   # untouched frames unchanged


def test_shuffle_preserves_distributional_stats_but_not_order(u, gt):
    u2, rec = artifacts.shuffle_frames(u, seed=1)
    g2 = groundtruth.compute(u2)
    assert sorted(rec.params["perm"]) == list(range(gt.n_frames))
    assert g2.scalars["rg_mean"] == pytest.approx(gt.scalars["rg_mean"], rel=1e-6)
    assert not np.allclose(g2.rg, gt.rg)


def test_rigid_jitter_does_not_change_fitted_quantities(u, gt):
    u2, _ = artifacts.rigid_jitter(u, seed=2)
    g2 = groundtruth.compute(u2)
    assert np.allclose(g2.rmsd, gt.rmsd, atol=2e-2)
    assert np.allclose(g2.rmsf, gt.rmsf, atol=2e-2)
    assert g2.scalars["rg_mean"] == pytest.approx(gt.scalars["rg_mean"], rel=1e-4)


def test_verify_numeric(gt):
    truth = gt.scalars["rmsd_mean"]
    ok = Claim("a", "mean RMSD", Kind.NUMERIC, {"quantity": "rmsd_mean", "value": truth * 1.05})
    bad = Claim("b", "mean RMSD", Kind.NUMERIC, {"quantity": "rmsd_mean", "value": truth * 2.0})
    unk = Claim("c", "??", Kind.NUMERIC, {"quantity": "nonsense", "value": 1})
    assert verify(ok, gt).label is Label.SUPPORTED
    assert verify(bad, gt).label is Label.CONTRADICTED
    assert verify(unk, gt).label is Label.UNVERIFIABLE


def test_verify_ranking(gt):
    top = gt.top_residues("rmsf", 5)
    ok = Claim("a", "", Kind.RANKING, {"quantity": "rmsf", "k": 5, "residues": top[:3]})
    bad = Claim("b", "", Kind.RANKING, {"quantity": "rmsf", "k": 5, "residues": gt.top_residues("rmsf", 5, "lowest")})
    assert verify(ok, gt).label is Label.SUPPORTED
    assert verify(bad, gt).label is Label.CONTRADICTED


def test_verify_temporal_flags_artifact_misread(u):
    u2, rec = artifacts.pbc_split(u, frames=[30, 31], resid_range=(1, 60))
    g2 = groundtruth.compute(u2)
    physical = Claim("a", "large conformational change at frame 30", Kind.TEMPORAL,
                     {"event": "rmsd_peak", "frame": 30, "physical": True})
    flagged = Claim("b", "RMSD spike at frame 30 is a wrapping artifact", Kind.TEMPORAL,
                    {"event": "rmsd_peak", "frame": 30, "physical": False})
    assert verify(physical, g2, rec).label is Label.ARTIFACT_MISREAD
    assert verify(flagged, g2, rec).label is Label.SUPPORTED


def test_verify_temporal_rejects_order_claims_after_shuffle(u):
    u2, rec = artifacts.shuffle_frames(u, seed=0)
    g2 = groundtruth.compute(u2)
    c = Claim("a", "RMSD plateaus after frame 40", Kind.TEMPORAL, {"event": "rmsd_plateau", "frame": 40})
    assert verify(c, g2, rec).label is Label.ARTIFACT_MISREAD


def test_verify_causal(gt):
    assert verify(Claim("a", "because of X", Kind.CAUSAL, {}), gt).label is Label.UNSUPPORTED_MECHANISM
    assert verify(Claim("b", "because of X", Kind.CAUSAL, {"evidence": ["rmsf"]}), gt).label is Label.UNVERIFIABLE


def test_claims_from_json_drops_malformed():
    s = ('[{"id":"1","text":"t","kind":"numeric","payload":{}},'
         '{"id":"2","text":"t","kind":"bogus","payload":{}}, {"nope":1}]')
    assert [c.id for c in claims_from_json(s)] == ["1"]


def test_metrics_and_bootstrap():
    from mdfaith.claims import Verdict
    v1 = [Verdict("a", Label.SUPPORTED), Verdict("b", Label.CONTRADICTED), Verdict("c", Label.UNSUPPORTED_MECHANISM)]
    v2 = [Verdict("a", Label.SUPPORTED), Verdict("b", Label.SUPPORTED)]
    c1, c2 = metrics.counts(v1), metrics.counts(v2)
    assert c1["checkable"] == 2 and metrics.hallucination_rate(c1) == 0.5
    point, lo, hi = metrics.cluster_bootstrap([c1, c2], n=200, seed=0)
    assert point == pytest.approx(1 / 4) and lo <= point <= hi


def test_rmsd_matches_mdanalysis_reference_implementation(u, gt):
    from MDAnalysis.analysis import rms
    ref = rms.RMSD(u, select="backbone", ref_frame=0).run().results.rmsd[:, 2]
    assert np.allclose(gt.rmsd, ref, atol=1e-3)


def test_rmsf_matches_mdanalysis_on_fitted_trajectory(u, gt):
    import MDAnalysis as mda
    from MDAnalysis.analysis import align, rms
    u2 = mda.Merge(u.atoms)
    from MDAnalysis.coordinates.memory import MemoryReader
    u2.load_new(groundtruth.frames_of(u, "all").astype(np.float32), format=MemoryReader, order="fac")
    align.AlignTraj(u2, u2, select="backbone", ref_frame=0, in_memory=True).run()
    ref = rms.RMSF(u2.select_atoms("protein and name CA")).run().results.rmsf
    assert np.allclose(gt.rmsf, ref, atol=5e-2)
