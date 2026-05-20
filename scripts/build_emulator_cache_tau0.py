"""Build the tau0-extended HCD-emulator training cache (Phase 2).

For every fully-processed (sim, snap) pair, rescale the raw fake_spectra
optical-depth grid with the freeze-core recipe at each of N alpha values
and stack the per-class P1D into observables_tau0.h5. The CDDF / dN/dX are
tau0-invariant and stored once per (sim, snap).

A --tier flag selects the rescale recipe:
  B (default) : freeze-core (tau_freeze = 1e6)        -> observables_tau0.h5
  A           : uniform rescale (tau_freeze = inf)    -> observables_tau0_uniform.h5

See docs/superpowers/specs/2026-05-17-phase2-hcd-emulator-design.md.

Usage:
    python3 scripts/build_emulator_cache_tau0.py \
        [--hcd-root /scratch/cavestru_root/cavestru0/mfho/hcd_outputs] \
        [--emu-root /nfs/turbo/umor-yueyingn/mfho/emu_full] \
        [--tier B] [--n-alpha 20] [--limit N] [--offset M] \
        [--n-skewers N] [--output PATH] [--spot-check]
"""
from __future__ import annotations

import argparse
import datetime
import subprocess
import sys
from functools import partial
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_emulator_cache as bec  # noqa: E402
from hcd_analysis.tau0_rescale import (  # noqa: E402
    freeze_core_rescale, make_alpha_grid, tau0_from_mean_flux,
    TAU_FREEZE_DEFAULT,
)

_DEFAULT_HCD_ROOT = Path("/scratch/cavestru_root/cavestru0/mfho/hcd_outputs")
_DEFAULT_EMU_ROOT = Path("/nfs/turbo/umor-yueyingn/mfho/emu_full")


def locate_raw_tau_file(emu_root, sim_name: str, snap: int):
    """Return the raw fake_spectra tau HDF5 for (sim_name, snap), or None.

    Layout: <emu_root>/<sim_name>/output/SPECTRA_<NNN>/
            lya_forest_spectra_grid_480.hdf5  (preferred)
            lya_forest_spectra.hdf5            (fallback)
    """
    spectra_dir = Path(emu_root) / sim_name / "output" / f"SPECTRA_{snap:03d}"
    if not spectra_dir.is_dir():
        return None
    grid = spectra_dir / "lya_forest_spectra_grid_480.hdf5"
    if grid.exists():
        return grid
    fallback = spectra_dir / "lya_forest_spectra.hdf5"
    if fallback.exists():
        return fallback
    return None
