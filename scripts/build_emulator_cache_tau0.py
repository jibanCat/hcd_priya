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


def discover_tau0_pairs(hcd_root, emu_root):
    """Return [(sim_name, snap, snap_dir, raw_tau_path), ...] for every
    (sim, snap) that has Phase-1 outputs, a native catalog.npz, AND a
    locatable raw fake_spectra tau grid."""
    out = []
    for sim, snap, snap_dir in bec.discover_sim_snap_pairs(Path(hcd_root)):
        if not (snap_dir / "catalog.npz").exists():
            continue
        raw = locate_raw_tau_file(emu_root, sim, snap)
        if raw is None:
            continue
        out.append((sim, snap, snap_dir, raw))
    return out


def build_tau0_rows(sim_name, snap, snap_dir, raw_tau_path, alpha_grid,
                    k_target, tau_freeze, n_skewers=None):
    """Build the per-alpha rows and the per-snap CDDF block for one (sim, snap).

    Returns (rows, snap_block):
      rows       : list of dicts, one per alpha (per-class P1D + scalars)
      snap_block : one dict with the tau0-invariant CDDF / dN/dX
    """
    from hcd_analysis.catalog import AbsorberCatalog
    from hcd_analysis.io import read_header
    from hcd_analysis.p1d import compute_p1d_per_class

    params_dict = bec.parse_sim_params(sim_name)
    if params_dict is None:
        raise ValueError(f"cannot parse params from sim folder name: {sim_name!r}")
    params = np.array([params_dict[k] for k in bec.PARAM_ORDER], dtype=np.float64)

    meta = bec.read_meta(snap_dir)
    cddf = bec.read_cddf(snap_dir)
    dndx = bec.compute_dndx_per_class(meta["n_absorbers"], float(cddf["total_path"]))
    catalog = AbsorberCatalog.load_npz(snap_dir / "catalog.npz")

    header = read_header(raw_tau_path)
    nbins = int(header.nbins)
    dv_kms = float(meta["dv_kms"])

    rows = []
    for a_idx, alpha in enumerate(alpha_grid):
        per_class = compute_p1d_per_class(
            raw_tau_path, nbins=nbins, dv_kms=dv_kms, catalog=catalog,
            n_skewers=n_skewers,
            tau_transform=partial(freeze_core_rescale, alpha=float(alpha),
                                  tau_freeze=tau_freeze),
        )
        k_src_angular = 2.0 * np.pi * per_class["k"]
        rows.append({
            "sim_name": sim_name,
            "snap": int(snap),
            "alpha": float(alpha),
            "alpha_idx": int(a_idx),
            "tau0": tau0_from_mean_flux(per_class["mean_F_clean"]),
            "params": params,
            "z": float(meta["z"]),
            "dv_kms": dv_kms,
            "nbins_native": nbins,
            "P_clean":       bec.interp_p1d_loglog(k_src_angular, per_class["P_clean"], k_target),
            "P_LLS_only":    bec.interp_p1d_loglog(k_src_angular, per_class["P_LLS_only"], k_target),
            "P_subDLA_only": bec.interp_p1d_loglog(k_src_angular, per_class["P_subDLA_only"], k_target),
            "P_DLA_only":    bec.interp_p1d_loglog(k_src_angular, per_class["P_DLA_only"], k_target),
            "mean_F_clean":  float(per_class["mean_F_clean"]),
            "mean_F_LLS":    float(per_class["mean_F_LLS"]),
            "mean_F_subDLA": float(per_class["mean_F_subDLA"]),
            "mean_F_DLA":    float(per_class["mean_F_DLA"]),
            "n_sightlines_clean":  int(per_class["n_sightlines_clean"]),
            "n_sightlines_LLS":    int(per_class["n_sightlines_LLS"]),
            "n_sightlines_subDLA": int(per_class["n_sightlines_subDLA"]),
            "n_sightlines_DLA":    int(per_class["n_sightlines_DLA"]),
        })

    snap_block = {
        "sim_name": sim_name,
        "snap": int(snap),
        "f_nhi": np.asarray(cddf["f_nhi"], dtype=np.float64),
        "n_absorbers": np.asarray(cddf["n_absorbers"], dtype=np.int64),
        "log_nhi_centres": np.asarray(cddf["log_nhi_centres"], dtype=np.float64),
        "log_nhi_edges": np.asarray(cddf["log_nhi_edges"], dtype=np.float64),
        "total_path_dX": float(cddf["total_path"]),
        **dndx,
    }
    return rows, snap_block
