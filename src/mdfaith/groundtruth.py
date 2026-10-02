"""Ground-truth quantities computed directly from a trajectory.

Everything an explanation can be checked against lives in `GroundTruth`. Definitions are fixed here so that
claims are scored against the same numbers regardless of which condition produced them.

Units: Angstrom. Frames are 0-indexed trajectory frames (time axes are not trusted).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.distance import cdist

CONTACT_CUTOFF = 8.0  # CA-CA, Angstrom
CONTACT_MIN_SEP = 4  # ignore |i-j| < 4 residues
POLAR_CUTOFF = 3.5  # heavy-atom N/O distance, Angstrom (not a full H-bond criterion; see DESIGN.md)
POLAR_MIN_SEP = 3
PLATEAU_REL_TOL = 0.15  # RMSD stays within 15% of its own later mean


def frames_of(u, selection: str) -> np.ndarray:
    """Positions of `selection` for every frame, shape (T, N, 3)."""
    ag = u.select_atoms(selection)
    out = np.empty((len(u.trajectory), len(ag), 3), dtype=np.float64)
    for i, _ in enumerate(u.trajectory):
        out[i] = ag.positions
    return out


def _kabsch(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Rotation R (applied as P @ R) minimising ||P R - Q||; P, Q centred (N, 3)."""
    U, _, Vt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(U @ Vt))
    return U @ np.diag([1.0, 1.0, d]) @ Vt


def superpose(P: np.ndarray, ref: np.ndarray) -> np.ndarray:
    Pc, Qc = P - P.mean(0), ref - ref.mean(0)
    return Pc @ _kabsch(Pc, Qc) + ref.mean(0)


def rmsd_series(bb: np.ndarray, ref_frame: int = 0) -> np.ndarray:
    ref = bb[ref_frame]
    out = np.empty(len(bb))
    for t, P in enumerate(bb):
        A = superpose(P, ref)
        out[t] = np.sqrt(np.mean(np.sum((A - ref) ** 2, axis=1)))
    return out


def rmsf_per_residue(ca: np.ndarray, bb: np.ndarray) -> np.ndarray:
    """CA RMSF after superposing every frame on the mean-free backbone of frame 0."""
    ref = bb[0]
    aligned = np.empty_like(ca)
    for t in range(len(ca)):
        Pc, Qc = bb[t] - bb[t].mean(0), ref - ref.mean(0)
        R = _kabsch(Pc, Qc)
        aligned[t] = (ca[t] - bb[t].mean(0)) @ R
    mean = aligned.mean(axis=0)
    return np.sqrt(np.mean(np.sum((aligned - mean) ** 2, axis=2), axis=0))


def plateau_frame(rmsd: np.ndarray, rel_tol: float = PLATEAU_REL_TOL) -> int:
    """First frame f such that every later RMSD value is within rel_tol of the mean of frames f.. end."""
    T = len(rmsd)
    for f in range(T):
        tail = rmsd[f:]
        m = tail.mean()
        if m > 0 and np.all(np.abs(tail - m) <= rel_tol * m):
            return f
    return T - 1


def contact_occupancy(ca: np.ndarray) -> np.ndarray:
    n = ca.shape[1]
    occ = np.zeros((n, n))
    for P in ca:
        occ += cdist(P, P) < CONTACT_CUTOFF
    occ /= len(ca)
    idx = np.arange(n)
    occ[np.abs(idx[:, None] - idx[None, :]) < CONTACT_MIN_SEP] = 0.0
    return occ


def polar_occupancy(pos: np.ndarray, res_index: np.ndarray, n_res: int) -> np.ndarray:
    """Fraction of frames in which any N/O heavy-atom pair of two residues is within POLAR_CUTOFF."""
    occ = np.zeros((n_res, n_res))
    for P in pos:
        close = cdist(P, P) < POLAR_CUTOFF
        ri, rj = np.nonzero(close)
        pair = np.zeros((n_res, n_res), dtype=bool)
        pair[res_index[ri], res_index[rj]] = True
        occ += pair
    occ /= len(pos)
    idx = np.arange(n_res)
    occ[np.abs(idx[:, None] - idx[None, :]) < POLAR_MIN_SEP] = 0.0
    return occ


@dataclass
class GroundTruth:
    system: str
    n_frames: int
    resids: list
    rmsd: np.ndarray
    rmsf: np.ndarray
    rg: np.ndarray
    contacts: np.ndarray
    polar: np.ndarray
    scalars: dict = field(default_factory=dict)

    def to_dict(self, include_matrices: bool = False) -> dict:
        d = {
            "system": self.system,
            "n_frames": self.n_frames,
            "resids": list(map(int, self.resids)),
            "rmsd": self.rmsd.tolist(),
            "rmsf": self.rmsf.tolist(),
            "rg": self.rg.tolist(),
            "scalars": self.scalars,
        }
        if include_matrices:
            d["contacts"] = self.contacts.tolist()
            d["polar"] = self.polar.tolist()
        return d

    def top_residues(self, quantity: str, k: int, direction: str = "highest") -> list:
        vals = {"rmsf": self.rmsf}[quantity]
        order = np.argsort(vals)
        order = order[::-1] if direction == "highest" else order
        return [int(self.resids[i]) for i in order[:k]]


def compute(u, system: str = "unknown") -> GroundTruth:
    prot = u.select_atoms("protein")
    resids = [int(r.resid) for r in prot.residues]
    n_res = len(resids)

    bb = frames_of(u, "backbone")
    ca = frames_of(u, "protein and name CA")
    rmsd = rmsd_series(bb)
    rmsf = rmsf_per_residue(ca, bb)
    rg = np.array([prot.radius_of_gyration() for _ in u.trajectory])

    contacts = contact_occupancy(ca)

    polar_ag = u.select_atoms("protein and (name N or name N* or name O or name O*)")
    res_pos = {int(r.resindex): i for i, r in enumerate(prot.residues)}
    res_index = np.array([res_pos[int(a.resindex)] for a in polar_ag])
    polar_pos = frames_of(u, "protein and (name N or name N* or name O or name O*)")
    polar = polar_occupancy(polar_pos, res_index, n_res)

    s = {
        "rmsd_mean": float(rmsd.mean()),
        "rmsd_max": float(rmsd.max()),
        "rmsd_final": float(rmsd[-1]),
        "rmsd_max_frame": int(rmsd.argmax()),
        "rmsd_plateau_frame": int(plateau_frame(rmsd)),
        "rmsf_mean": float(rmsf.mean()),
        "rmsf_max": float(rmsf.max()),
        "rmsf_max_resid": int(resids[int(rmsf.argmax())]),
        "rg_mean": float(rg.mean()),
        "rg_std": float(rg.std()),
        "contacts_persistent_n": int((np.triu(contacts, 1) > 0.9).sum()),
        "polar_persistent_n": int((np.triu(polar, 1) > 0.5).sum()),
    }
    return GroundTruth(system, len(u.trajectory), resids, rmsd, rmsf, rg, contacts, polar, s)
