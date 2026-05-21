"""Build the tau0-extended HCD-emulator training cache (Phase 2, v2.0).

For every fully-processed (sim, snap) pair, drive fake_spectra directly (via
hcd_analysis.priya_p1d) at each of N alpha-slope values to produce both:
  Tier P : PRIYA-compatible total P1D over all sightlines (bit-identical to
           PRIYA's flux_vectors), after the _filter_single_tau_complex mask.
  Tier C : per-class P1D (clean/LLS/subDLA/DLA) on the unfiltered tau, sharing
           Tier P's mean-flux normalisation (scale + target_F).
The CDDF / dN/dX are tau0-invariant and stored once per (sim, snap).

See docs/superpowers/specs/2026-05-17-phase2-hcd-emulator-design.md.

Usage:
    python3 scripts/build_emulator_cache_tau0.py \
        [--hcd-root /scratch/cavestru_root/cavestru0/mfho/hcd_outputs] \
        [--emu-root /nfs/turbo/umor-yueyingn/mfho/emu_full] \
        [--n-alpha 20] [--limit N] [--offset M] \
        [--n-skewers N] [--output PATH] [--spot-check]
"""
from __future__ import annotations

import argparse
import datetime
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_emulator_cache as bec  # noqa: E402
from hcd_analysis.tau0_rescale import make_alpha_grid_priya_aligned  # noqa: E402

_DEFAULT_HCD_ROOT = Path("/scratch/cavestru_root/cavestru0/mfho/hcd_outputs")
_DEFAULT_EMU_ROOT = Path("/nfs/turbo/umor-yueyingn/mfho/emu_full")


def locate_raw_tau_file(emu_root, sim_name: str, snap: int):
    """Return the raw fake_spectra tau HDF5 for (sim_name, snap), or None.

    Layout: <emu_root>/<sim_name>/output/SPECTRA_<NNN>/
            lya_forest_spectra_grid_480.hdf5  (preferred)
            lya_forest_spectra.hdf5            (fallback)
    """
    found = _grid_in_dir(Path(emu_root) / sim_name / "output" / f"SPECTRA_{snap:03d}")
    if found is not None:
        return found
    # Fallback: the Phase-1 (hcd_outputs) and raw (emu_full) folder names can
    # round the 9 params differently (e.g. the 4th HR sim is alphaq1.57/
    # omegamh20.141 in hcd_outputs but alphaq1.58/omegamh20.142 in emu_full).
    # Match the emu_root folder by parsed params instead of by exact name.
    alt = _match_emu_folder_by_params(emu_root, sim_name)
    if alt is not None:
        return _grid_in_dir(Path(emu_root) / alt / "output" / f"SPECTRA_{snap:03d}")
    return None


def _grid_in_dir(spectra_dir):
    """Return the grid (preferred) or fallback tau HDF5 in a SPECTRA dir, or None."""
    if not spectra_dir.is_dir():
        return None
    grid = spectra_dir / "lya_forest_spectra_grid_480.hdf5"
    if grid.exists():
        return grid
    fallback = spectra_dir / "lya_forest_spectra.hdf5"
    if fallback.exists():
        return fallback
    return None


def _match_emu_folder_by_params(emu_root, sim_name, rtol=0.015):
    """Return the emu_root folder whose parsed 9 params match `sim_name`'s to
    within `rtol` on every param, or None if there is no unique match.

    Used when the exact-name path is absent because hcd_outputs and emu_full
    round the folder-name params differently. rtol=1.5% safely spans the
    <0.7% rounding gap while staying far below the inter-sim separation.
    """
    target = bec.parse_sim_params(sim_name)
    if target is None:
        return None
    match = None
    for d in sorted(Path(emu_root).iterdir()):
        if not d.is_dir():
            continue
        p = bec.parse_sim_params(d.name)
        if p is None:
            continue
        if all(abs(p[k] - target[k]) <= rtol * max(abs(target[k]), 1e-30)
               for k in target):
            if match is not None and match != d.name:
                return None  # ambiguous -> refuse to guess
            match = d.name
    return match


def discover_tau0_pairs(hcd_root, emu_root, include_hires=True):
    """Return [(sim_name, snap, snap_dir, raw_tau_path), ...] for every
    (sim, snap) that has Phase-1 outputs, a native catalog.npz, AND a
    locatable raw fake_spectra tau grid.

    LF sims live directly under `hcd_root`; the 4 HR sims live under
    `hcd_root/hires`. Both have their raw tau (and SimulationICs.json) under
    `emu_root/<sim>` (bare, no hires/ prefix). The list is LF-first then HR,
    each sorted by (sim, snap) via bec.discover_sim_snap_pairs, so --offset/
    --limit sharding is deterministic.
    """
    roots = [Path(hcd_root)]
    if include_hires:
        hires = Path(hcd_root) / "hires"
        if hires.is_dir():
            roots.append(hires)
    out = []
    for root in roots:
        for sim, snap, snap_dir in bec.discover_sim_snap_pairs(root):
            if not (snap_dir / "catalog.npz").exists():
                continue
            raw = locate_raw_tau_file(emu_root, sim, snap)
            if raw is None:
                continue
            out.append((sim, snap, snap_dir, raw))
    return out


# PRIYA's zout grid runs 2.0..5.4 in steps of 0.2; snapshot redshifts can land
# slightly off (e.g. 4.600013). For bit-compatibility with PRIYA's training
# data the mean-flux model must use the GRID z, not the snapshot's exact z
# (obs_mean_tau ∝ (1+z)^3.65; an off-grid z gives a ~1e-5 P1D bias — see
# docs/superpowers/2026-05-20-priya-p1d-consistency-check.md §6c).
def _snap_z_to_priya_grid(z):
    """Round a snapshot redshift to PRIYA's zout grid (nearest multiple of 0.2)."""
    return round(float(z) / 0.2) * 0.2


def _read_tau(raw_tau_path, n_skewers=None):
    """Read the full tau grid into memory as float64 (optionally first n_skewers)."""
    with h5py.File(raw_tau_path, "r") as f:
        ds = f["tau/H/1/1215"]
        if n_skewers is not None:
            return ds[:n_skewers].astype(np.float64)
        return ds[...].astype(np.float64)


# PRIYA's emulator parameter "Ap" is the primordial scalar amplitude at the
# Lya pivot k_p = pi/4 /Mpc, while CAMB's `scalar_amp` (As) is at k_0 = 0.05
# /Mpc. They are the same amplitude at different pivots:
#   Ap = As * (k_p / k_0)^(ns - 1),  with k_p/k_0 = (pi/4)/0.05 = 5*pi.
# Verified to machine precision against PRIYA's params array across the LF
# grid (sim 0/29/44). See docs/SESSION_HANDOVER_2026_05_20.md.
_AP_PIVOT_RATIO = 5.0 * np.pi  # k_p/k_0


def _read_priya_params(raw_tau_path):
    """Return the PRIYA-exact 9-param vector (in bec.PARAM_ORDER) for a sim,
    read from its SimulationICs.json.

    Folder-name parsing (bec.parse_sim_params) rounds the params (off by up to
    ~0.4% from PRIYA's Latin-hypercube values), so we read full precision from
    SimulationICs.json instead. 8 cosmo params map directly; `Ap` is converted
    from CAMB `scalar_amp` (As) to PRIYA's Lya-pivot Ap via _AP_PIVOT_RATIO.

    `raw_tau_path` is <emu_root>/<sim>/output/SPECTRA_NNN/<grid>.hdf5, so the
    SimulationICs.json lives two parents up from the SPECTRA dir.
    """
    import json
    ics_path = Path(raw_tau_path).resolve().parents[2] / "SimulationICs.json"
    with open(ics_path) as f:
        ics = json.load(f)
    ns = float(ics["ns"])
    ap = float(ics["scalar_amp"]) * _AP_PIVOT_RATIO ** (ns - 1.0)
    vals = {
        "ns": ns,
        "Ap": ap,
        "herei": float(ics["here_i"]),
        "heref": float(ics["here_f"]),
        "alphaq": float(ics["alpha_q"]),
        "hub": float(ics["hubble"]),
        "omegamh2": float(ics["omega0"]) * float(ics["hubble"]) ** 2,
        "hireionz": float(ics["hireionz"]),
        "bhfeedback": float(ics["bhfeedback"]),
    }
    return np.array([vals[k] for k in bec.PARAM_ORDER], dtype=np.float64)


def build_tau0_rows(sim_name, snap, snap_dir, raw_tau_path, alpha_slope_grid,
                    k_target, n_skewers=None):
    """Build per-alpha rows (Tier-P total + Tier-C per-class P1D) and the
    per-snap CDDF block for one (sim, snap), driving fake_spectra directly.

    Tier P: PRIYA-compatible total P1D over all sightlines after the
    _filter_single_tau_complex DLA mask. Bit-identical to PRIYA's flux_vectors.
    Tier C: per-class (clean/LLS/subDLA/DLA) P1D on the UNFILTERED tau, sharing
    Tier P's scale + target_F (so they decompose against the total).

    Returns (rows, snap_block).
    """
    from hcd_analysis.catalog import AbsorberCatalog
    from hcd_analysis.io import read_header
    from hcd_analysis.priya_p1d import compute_tier_p_p1d, compute_tier_c_p1d

    # PRIYA-exact params from SimulationICs.json (full precision; folder-name
    # parsing rounds them, and PRIYA's "Ap" is at a different pivot than CAMB As).
    params = _read_priya_params(raw_tau_path)

    meta = bec.read_meta(snap_dir)
    cddf = bec.read_cddf(snap_dir)
    dndx = bec.compute_dndx_per_class(meta["n_absorbers"], float(cddf["total_path"]))
    catalog = AbsorberCatalog.load_npz(snap_dir / "catalog.npz")

    nbins = int(read_header(raw_tau_path).nbins)
    dv_kms = float(meta["dv_kms"])
    vmax = nbins * dv_kms
    z_meta = float(meta["z"])
    z_grid = _snap_z_to_priya_grid(z_meta)

    tau_unfilt = _read_tau(raw_tau_path, n_skewers=n_skewers)

    rows = []
    for a_idx, alpha in enumerate(alpha_slope_grid):
        alpha = float(alpha)
        # Tier P needs a filtered copy (the filter mutates tau in place).
        tau_filt = tau_unfilt.copy()
        kf_p, P_tier_p, target_F, scale = compute_tier_p_p1d(
            tau_filt, vmax, alpha_slope=alpha, z=z_grid)
        del tau_filt
        # Tier C on the unfiltered tau, sharing Tier P's normalisation.
        kf_c, by_class, n_by_class, _, _ = compute_tier_c_p1d(
            tau_unfilt, vmax, alpha_slope=alpha, z=z_grid, catalog=catalog,
            external_scale=scale, external_target_F=target_F)
        rows.append({
            "sim_name": sim_name,
            "snap": int(snap),
            "alpha_slope": alpha,
            "alpha_idx": int(a_idx),
            "z_meta": z_meta,
            "z_grid": float(z_grid),
            "dv_kms": dv_kms,
            "nbins_native": nbins,
            "target_F": float(target_F),
            "scale": float(scale),
            "params": params,
            "P_tier_p": bec.interp_p1d_loglog(kf_p, P_tier_p, k_target),
            "P_clean":  bec.interp_p1d_loglog(kf_c, by_class["clean"], k_target),
            "P_LLS":    bec.interp_p1d_loglog(kf_c, by_class["LLS"], k_target),
            "P_subDLA": bec.interp_p1d_loglog(kf_c, by_class["subDLA"], k_target),
            "P_DLA":    bec.interp_p1d_loglog(kf_c, by_class["DLA"], k_target),
            "n_clean":  int(n_by_class["clean"]),
            "n_LLS":    int(n_by_class["LLS"]),
            "n_subDLA": int(n_by_class["subDLA"]),
            "n_DLA":    int(n_by_class["DLA"]),
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


_ROW_FLOAT_KEYS = ("alpha_slope", "target_F", "scale", "z_meta", "z_grid", "dv_kms")
_ROW_INT_KEYS = (
    "snap", "alpha_idx", "nbins_native", "snap_group_idx",
    "n_clean", "n_LLS", "n_subDLA", "n_DLA",
)
_ROW_P1D_KEYS = ("P_tier_p", "P_clean", "P_LLS", "P_subDLA", "P_DLA")
_SNAP_FLOAT_KEYS = ("total_path_dX", "dNdX_LLS", "dNdX_subDLA", "dNdX_DLA")
_SNAP_2D_KEYS = ("f_nhi", "n_absorbers")


def write_cache_tau0(rows, snap_blocks, k_target, output_path, alpha_range):
    """Stack per-alpha `rows` + per-snap `snap_blocks` into one HDF5 cache.

    Each row carries `snap_group_idx`, an index into the snap_* datasets.
    """
    if not rows:
        raise ValueError("write_cache_tau0 called with no rows; nothing to write.")
    if not snap_blocks:
        raise ValueError("write_cache_tau0 called with no snap_blocks.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log_nhi_centres = snap_blocks[0]["log_nhi_centres"]
    log_nhi_edges = snap_blocks[0]["log_nhi_edges"]
    for b in snap_blocks[1:]:
        assert np.array_equal(b["log_nhi_centres"], log_nhi_centres), \
            "log_nhi_centres mismatch across snap_blocks"

    with h5py.File(output_path, "w") as f:
        f.attrs["created_utc"] = (
            datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z")
        f.attrs["git_sha"] = bec._git_sha(REPO_ROOT)
        f.attrs["n_rows"] = len(rows)
        f.attrs["n_snaps"] = len(snap_blocks)
        f.attrs["cache_version"] = "2.0"
        f.attrs["priya_convention"] = (
            "Kim 2013 slope-alpha (obs_mean_tau=2.3e-3(1+z)^3.65); "
            "fake_spectra _filter_single_tau_complex(tau_thresh=1e6, thresh2=0.25); "
            "flux_power window=False spec_res=0")
        f.attrs["tau_thresh"] = 1.0e6
        f.attrs["alpha_range"] = np.asarray(alpha_range, dtype=np.float64)
        f.attrs["k_convention"] = "angular (rad*s/km), PRIYA convention"

        f.create_dataset("k_target", data=np.asarray(k_target, dtype=np.float64))
        f.create_dataset("param_names",
                         data=np.array(list(bec.PARAM_ORDER), dtype=h5py.string_dtype()))
        f.create_dataset("log_nhi_centres", data=log_nhi_centres)
        f.create_dataset("log_nhi_edges", data=log_nhi_edges)

        # --- per-row (sim, snap, alpha) datasets ---
        f.create_dataset("sim_name",
                         data=np.array([r["sim_name"] for r in rows],
                                       dtype=h5py.string_dtype()))
        f.create_dataset("params", data=np.stack([r["params"] for r in rows], axis=0))
        for key in _ROW_FLOAT_KEYS:
            f.create_dataset(key, data=np.array([r[key] for r in rows], dtype=np.float64))
        for key in _ROW_INT_KEYS:
            f.create_dataset(key, data=np.array([r[key] for r in rows], dtype=np.int32))
        for key in _ROW_P1D_KEYS:
            f.create_dataset(key, data=np.stack([r[key] for r in rows], axis=0))

        # --- per-(sim, snap) tau0-invariant CDDF datasets ---
        f.create_dataset("snap_sim_name",
                         data=np.array([b["sim_name"] for b in snap_blocks],
                                       dtype=h5py.string_dtype()))
        f.create_dataset("snap_snap",
                         data=np.array([b["snap"] for b in snap_blocks], dtype=np.int32))
        for key in _SNAP_FLOAT_KEYS:
            f.create_dataset("snap_" + key,
                             data=np.array([b[key] for b in snap_blocks], dtype=np.float64))
        for key in _SNAP_2D_KEYS:
            f.create_dataset("snap_" + key,
                             data=np.stack([b[key] for b in snap_blocks], axis=0))


def _default_output() -> Path:
    return REPO_ROOT / "hcd_analysis" / "_emulator_data" / "observables_tau0.h5"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the tau0-extended HCD-emulator cache (Phase 2).")
    parser.add_argument("--hcd-root", type=Path, default=_DEFAULT_HCD_ROOT)
    parser.add_argument("--emu-root", type=Path, default=_DEFAULT_EMU_ROOT)
    parser.add_argument("--alpha-refine", type=int, default=2,
                        help="PRIYA-aligned grid: refine*10 alpha containing "
                             "PRIYA's 10 (default 2 -> 20 alpha).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N (sim, snap) pairs.")
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip the first M pairs (for sharded array jobs).")
    parser.add_argument("--n-skewers", type=int, default=None,
                        help="Limit skewers per snap (dry runs only).")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--spot-check", action="store_true",
                        help="After writing, verify row 0 P_tier_p is finite.")
    args = parser.parse_args()

    from hcd_analysis.p1d import _DEFAULT_K_BINS
    k_target = 2.0 * np.pi * _DEFAULT_K_BINS

    alpha_slope_grid = make_alpha_grid_priya_aligned(refine=args.alpha_refine)
    output = args.output or _default_output()

    pairs = discover_tau0_pairs(args.hcd_root, args.emu_root)
    print(f"Found {len(pairs)} tau0-buildable (sim, snap) pairs")
    pairs = pairs[args.offset:]
    if args.limit is not None:
        pairs = pairs[: args.limit]
    print(f"Processing {len(pairs)} pairs (offset={args.offset}, limit={args.limit}); "
          f"n_alpha={len(alpha_slope_grid)} (PRIYA-aligned, refine={args.alpha_refine})")

    all_rows, snap_blocks = [], []
    for i, (sim, snap, snap_dir, raw) in enumerate(pairs):
        rows, block = build_tau0_rows(
            sim, snap, snap_dir, raw, alpha_slope_grid, k_target,
            n_skewers=args.n_skewers)
        gi = len(snap_blocks)
        for r in rows:
            r["snap_group_idx"] = gi
        all_rows.extend(rows)
        snap_blocks.append(block)
        if (i + 1) % 10 == 0 or (i + 1) == len(pairs):
            print(f"  built {i + 1}/{len(pairs)} pairs ({len(all_rows)} rows)")

    write_cache_tau0(all_rows, snap_blocks, k_target, output,
                     alpha_range=(float(alpha_slope_grid[0]),
                                  float(alpha_slope_grid[-1])))
    print(f"Wrote {output}  ({output.stat().st_size / 1e6:.2f} MB, "
          f"{len(all_rows)} rows, {len(snap_blocks)} snaps)")

    if args.spot_check and all_rows:
        with h5py.File(output, "r") as f:
            assert np.isfinite(f["P_tier_p"][0]).any(), \
                "spot-check failed: row 0 P_tier_p has no finite values"
            med_target_F = float(np.median(f["target_F"][...]))
            med_scale = float(np.median(f["scale"][...]))
        print(f"spot-check: row 0 P_tier_p finite OK; "
              f"median target_F={med_target_F:.4f} scale={med_scale:.4f}")


if __name__ == "__main__":
    main()
