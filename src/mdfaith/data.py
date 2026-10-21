"""Public systems only. Unpublished lab trajectories must not be added without the owner's written permission.

- `adk_dims`: AdK transition trajectory bundled with MDAnalysisTests (98 frames, no unit cell). Always available.
- `adk_equilibrium`: AdK 1 microsecond equilibrium run from MDAnalysisData (4187 frames, subsampled). Optional: needs the
  `systems` extra and an explicit `mdfaith fetch-data adk_equilibrium`; it is never downloaded silently.
"""

from __future__ import annotations

import MDAnalysis as mda

EQUILIBRIUM_STRIDE = 40  # 4187 frames / 40 = 105 frames: similar length to adk_dims, keeps runs fast


def load_adk_dims() -> mda.Universe:
    """AdK trajectory (3341 atoms, 214 residues, 98 frames) bundled with MDAnalysisTests. No unit-cell info."""
    from MDAnalysisTests.datafiles import DCD, PSF

    return mda.Universe(PSF, DCD)


def _equilibrium_paths(download: bool = False) -> tuple[str, str]:
    from MDAnalysisData import datasets

    ds = datasets.fetch_adk_equilibrium(download_if_missing=download)
    return ds.topology, ds.trajectory


def load_adk_equilibrium(stride: int = EQUILIBRIUM_STRIDE) -> mda.Universe:
    """AdK 1 us equilibrium run (apo, CHARMM27, solvent stripped), every `stride`-th frame held in memory."""
    top, trj = _equilibrium_paths(download=False)
    u = mda.Universe(top, trj)
    u.transfer_to_memory(step=stride)
    return u


SYSTEMS = {"adk_dims": load_adk_dims, "adk_equilibrium": load_adk_equilibrium}
DESCRIPTIONS = {
    "adk_dims": "AdK transition (DIMS), 98 frames, bundled with MDAnalysisTests",
    "adk_equilibrium": f"AdK 1 us equilibrium, 4187 frames subsampled every {EQUILIBRIUM_STRIDE} (MDAnalysisData)",
}


def is_available(name: str) -> bool:
    """True if the system can be loaded right now without downloading anything."""
    if name == "adk_dims":
        return True
    if name == "adk_equilibrium":
        try:
            _equilibrium_paths(download=False)
            return True
        except (ImportError, OSError):
            return False
    return False


def available_systems() -> list[str]:
    return [n for n in SYSTEMS if is_available(n)]


def fetch_data(name: str) -> None:
    """Explicitly download an optional system. The only place a download is allowed."""
    if name == "adk_equilibrium":
        _equilibrium_paths(download=True)
    elif name != "adk_dims":
        raise KeyError(f"unknown system {name!r}; available: {sorted(SYSTEMS)}")


def load_system(name: str) -> mda.Universe:
    if name not in SYSTEMS:
        raise KeyError(f"unknown system {name!r}; available: {sorted(SYSTEMS)}")
    if not is_available(name):
        raise FileNotFoundError(f"system {name!r} is not downloaded; run `mdfaith fetch-data {name}` first")
    return SYSTEMS[name]()
