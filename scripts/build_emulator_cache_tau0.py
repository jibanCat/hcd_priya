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


_ROW_FLOAT_KEYS = (
    "alpha", "tau0", "z", "dv_kms",
    "mean_F_clean", "mean_F_LLS", "mean_F_subDLA", "mean_F_DLA",
)
_ROW_INT_KEYS = (
    "snap", "alpha_idx", "nbins_native", "snap_group_idx",
    "n_sightlines_clean", "n_sightlines_LLS",
    "n_sightlines_subDLA", "n_sightlines_DLA",
)
_ROW_P1D_KEYS = ("P_clean", "P_LLS_only", "P_subDLA_only", "P_DLA_only")
_SNAP_FLOAT_KEYS = ("total_path_dX", "dNdX_LLS", "dNdX_subDLA", "dNdX_DLA")
_SNAP_2D_KEYS = ("f_nhi", "n_absorbers")


def write_cache_tau0(rows, snap_blocks, k_target, output_path,
                     tier, tau_freeze, alpha_range):
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
        f.attrs["rescale_tier"] = tier
        f.attrs["tau_freeze"] = float(tau_freeze)
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


def tier_to_tau_freeze(tier: str) -> float:
    """Map a --tier letter to its tau_freeze boundary.

    B = freeze-core production (tau_freeze = 1e6).
    A = uniform-rescale twin   (tau_freeze = inf) — a deliberately-wrong
        systematic baseline; see spec sec. 3.
    """
    if tier == "B":
        return TAU_FREEZE_DEFAULT
    if tier == "A":
        return np.inf
    raise ValueError(f"unknown tier {tier!r}; expected 'A' or 'B'")


def _default_output(tier: str) -> Path:
    name = "observables_tau0.h5" if tier == "B" else "observables_tau0_uniform.h5"
    return REPO_ROOT / "hcd_analysis" / "_emulator_data" / name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the tau0-extended HCD-emulator cache (Phase 2).")
    parser.add_argument("--hcd-root", type=Path, default=_DEFAULT_HCD_ROOT)
    parser.add_argument("--emu-root", type=Path, default=_DEFAULT_EMU_ROOT)
    parser.add_argument("--tier", choices=["A", "B"], default="B",
                        help="B = freeze-core (default); A = uniform-rescale twin.")
    parser.add_argument("--n-alpha", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N (sim, snap) pairs.")
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip the first M pairs (for sharded array jobs).")
    parser.add_argument("--n-skewers", type=int, default=None,
                        help="Limit skewers per snap (dry runs only).")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--spot-check", action="store_true",
                        help="After writing, re-verify row 0: tau0 == -ln(mean_F_clean).")
    args = parser.parse_args()

    from hcd_analysis.p1d import _DEFAULT_K_BINS
    k_target = 2.0 * np.pi * _DEFAULT_K_BINS

    tau_freeze = tier_to_tau_freeze(args.tier)
    alpha_grid = make_alpha_grid(n=args.n_alpha)
    output = args.output if args.output is not None else _default_output(args.tier)

    pairs = discover_tau0_pairs(args.hcd_root, args.emu_root)
    print(f"Found {len(pairs)} tau0-buildable (sim, snap) pairs")
    pairs = pairs[args.offset:]
    if args.limit is not None:
        pairs = pairs[: args.limit]
    print(f"Processing {len(pairs)} pairs (offset={args.offset}, limit={args.limit}); "
          f"tier={args.tier} (tau_freeze={tau_freeze}); n_alpha={args.n_alpha}")

    all_rows, snap_blocks = [], []
    for i, (sim, snap, snap_dir, raw) in enumerate(pairs):
        rows, block = build_tau0_rows(
            sim, snap, snap_dir, raw, alpha_grid, k_target,
            tau_freeze=tau_freeze, n_skewers=args.n_skewers)
        gi = len(snap_blocks)
        for r in rows:
            r["snap_group_idx"] = gi
        all_rows.extend(rows)
        snap_blocks.append(block)
        if (i + 1) % 10 == 0 or (i + 1) == len(pairs):
            print(f"  built {i + 1}/{len(pairs)} pairs ({len(all_rows)} rows)")

    write_cache_tau0(all_rows, snap_blocks, k_target, output,
                     tier=args.tier, tau_freeze=tau_freeze,
                     alpha_range=(float(alpha_grid[0]), float(alpha_grid[-1])))
    print(f"Wrote {output}  ({output.stat().st_size / 1e6:.2f} MB, "
          f"{len(all_rows)} rows, {len(snap_blocks)} snaps)")

    if args.spot_check and all_rows:
        with h5py.File(output, "r") as f:
            tau0_0 = float(f["tau0"][0])
            mfc_0 = float(f["mean_F_clean"][0])
        assert np.isclose(tau0_0, -np.log(mfc_0)), \
            f"spot-check failed: tau0={tau0_0} != -ln(mean_F_clean)={-np.log(mfc_0)}"
        print(f"spot-check: row 0 tau0={tau0_0:.4f} == -ln(mean_F_clean) OK")


if __name__ == "__main__":
    main()
