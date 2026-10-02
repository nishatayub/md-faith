"""Public systems only. Unpublished lab trajectories must not be added without the owner's written permission."""
from __future__ import annotations

import MDAnalysis as mda


def load_adk_dims() -> mda.Universe:
    """AdK trajectory (3341 atoms, 214 residues, 98 frames) bundled with MDAnalysisTests. No unit-cell info."""
    from MDAnalysisTests.datafiles import DCD, PSF

    return mda.Universe(PSF, DCD)


SYSTEMS = {"adk_dims": load_adk_dims}


def load_system(name: str) -> mda.Universe:
    if name not in SYSTEMS:
        raise KeyError(f"unknown system {name!r}; available: {sorted(SYSTEMS)}")
    return SYSTEMS[name]()
