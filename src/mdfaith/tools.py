"""Tools exposed to the tool-using condition. They run real analysis on the (possibly artifact-containing)
trajectory; the agent never sees ground-truth scalars or the artifact record."""

from __future__ import annotations

import numpy as np

from . import groundtruth as G


class Toolbox:
    def __init__(self, u):
        self.u = u
        self._bb = None
        self._ca = None

    def _arrays(self):
        if self._bb is None:
            self._bb = G.frames_of(self.u, "backbone")
            self._ca = G.frames_of(self.u, "protein and name CA")
        return self._bb, self._ca

    # ---- tools -------------------------------------------------------------------------------
    def rmsd(self, ref_frame: int = 0, superpose: bool = True) -> dict:
        bb, _ = self._arrays()
        if superpose:
            r = G.rmsd_series(bb, ref_frame)
        else:
            r = np.sqrt(np.mean(np.sum((bb - bb[ref_frame]) ** 2, axis=2), axis=1))
        return {"per_frame": [round(float(x), 3) for x in r], "superposed": superpose}

    def rmsf(self) -> dict:
        bb, ca = self._arrays()
        r = G.rmsf_per_residue(ca, bb)
        return {"per_residue": [round(float(x), 3) for x in r]}

    def radius_of_gyration(self) -> dict:
        prot = self.u.select_atoms("protein")
        return {"per_frame": [round(float(prot.radius_of_gyration()), 3) for _ in self.u.trajectory]}

    def frame_to_frame_rmsd(self) -> dict:
        """RMSD between consecutive frames: large values flag discontinuities (wrapping, reordering)."""
        bb, _ = self._arrays()
        out = [0.0]
        for t in range(1, len(bb)):
            a = G.superpose(bb[t], bb[t - 1])
            out.append(round(float(np.sqrt(np.mean(np.sum((a - bb[t - 1]) ** 2, axis=1)))), 3))
        return {"per_frame": out}

    def hbonds(self) -> dict:
        """Hydrogen bonds (donor-acceptor < 3.0 A, angle > 150 deg): count per frame and persistent residue pairs."""
        prot = self.u.select_atoms("protein")
        res_pos = {int(r.resindex): i for i, r in enumerate(prot.residues)}
        counts, occ = G.hbond_analysis(self.u, res_pos, len(prot.residues))
        return {
            "per_frame_count": [int(c) for c in counts],
            "persistent_pairs_over_50pct": int((occ > 0.5).sum()),
        }

    def contacts(self, min_occupancy: float = 0.9) -> dict:
        _, ca = self._arrays()
        occ = G.contact_occupancy(ca)
        i, j = np.nonzero(np.triu(occ, 1) >= min_occupancy)
        return {
            "n_pairs": len(i),
            "pairs_residue_index": [[int(a), int(b)] for a, b in zip(i[:200], j[:200], strict=True)],
        }

    SPECS = [
        {
            "name": "rmsd",
            "description": "Backbone RMSD per frame to a reference frame (Angstrom).",
            "parameters": {"ref_frame": "int", "superpose": "bool"},
        },
        {"name": "rmsf", "description": "CA RMSF per residue after fitting (Angstrom).", "parameters": {}},
        {
            "name": "radius_of_gyration",
            "description": "Protein radius of gyration per frame.",
            "parameters": {},
        },
        {
            "name": "frame_to_frame_rmsd",
            "description": "RMSD between consecutive frames; flags discontinuities.",
            "parameters": {},
        },
        {
            "name": "hbonds",
            "description": "Hydrogen bonds per frame and number of persistent residue pairs.",
            "parameters": {},
        },
        {
            "name": "contacts",
            "description": "CA-CA contacts persistent above an occupancy threshold.",
            "parameters": {"min_occupancy": "float"},
        },
    ]

    def call(self, name: str, arguments: dict | None = None) -> dict:
        if name not in {s["name"] for s in self.SPECS}:
            return {"error": f"unknown tool {name!r}"}
        try:
            return getattr(self, name)(**(arguments or {}))
        except TypeError as e:
            return {"error": str(e)}
