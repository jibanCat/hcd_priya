"""Build the tau0-extended HCD-emulator training cache (Phase 2, v3.1).

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
        [--fidelity {lf,hr}] \
        [--hcd-root /scratch/cavestru_root/cavestru0/mfho/hcd_outputs] \
        [--emu-root /nfs/turbo/umor-yueyingn/mfho/emu_full] \
        [--alpha-refine 2] [--limit N] [--offset M] \
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

_HCD_OUTPUTS = Path("/scratch/cavestru_root/cavestru0/mfho/hcd_outputs")
_LF_EMU_ROOT = Path("/nfs/turbo/umor-yueyingn/mfho/emu_full")
_HR_EMU_ROOT = Path("/scratch/yueyingn_root/yueyingn0/mfho/priya/emu_full_hires_2")
_N_K = {"lf": 172, "hr": 525}


def locate_raw_tau_file(emu_root, sim_name: str, snap: int):
    """Return the raw fake_spectra tau HDF5 for (sim_name, snap), or None.

    Layout: <emu_root>/<sim_name>/output/SPECTRA_<NNN>/
            lya_forest_spectra_grid_480.hdf5  (required; 691200-skewer PRIYA product)

    Only the 691200-skewer grid_480 file is accepted.  The low-res 32k
    lya_forest_spectra.hdf5 is never used (it cannot match PRIYA bit-identity).
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
    """Return the 691200-skewer PRIYA grid tau file in a SPECTRA dir, or None.

    The low-res `lya_forest_spectra.hdf5` fallback (32000 skewers) is
    intentionally NOT used: PRIYA bit-identity requires the grid_480 product.
    A dir with only the fallback (e.g. ns0.907/SPECTRA_015, a 32k z=3.2
    duplicate) is therefore skipped."""
    if not spectra_dir.is_dir():
        return None
    grid = spectra_dir / "lya_forest_spectra_grid_480.hdf5"
    return grid if grid.exists() else None


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


def discover_tau0_pairs(hcd_root, emu_root, fidelity="lf", max_off_grid=0.05):
    """Return [(sim, snap, snap_dir, raw_tau_path), ...] for one fidelity.

    fidelity='lf': sims directly under hcd_root; raw under emu_root/<sim>.
    fidelity='hr': pass hcd_root = the hcd_outputs BASE; /hires is appended
    automatically; raw under emu_root/<sim> (emu_full_hires_2). Pairs are
    returned in the deterministic (sim, snap) order from bec.discover_sim_snap_pairs.

    Deduplication: for each (sim, z_grid) only the snap whose meta z is CLOSEST
    to the PRIYA grid is kept. Any kept snap still more than max_off_grid from
    the grid is skipped with a WARN (protects against ~7% wrong mean-flux
    normalization from off-grid snaps getting the wrong target_F).
    """
    import json
    if fidelity == "hr":
        root = Path(hcd_root) / "hires"
    else:
        root = Path(hcd_root)
    out = []
    for sim, snap, snap_dir in bec.discover_sim_snap_pairs(root):
        if not (snap_dir / "catalog.npz").exists():
            continue
        raw = locate_raw_tau_file(emu_root, sim, snap)
        if raw is None:
            continue
        out.append((sim, snap, snap_dir, raw))

    best = {}   # (sim, round(z_grid,4)) -> (off_grid_distance, pair)
    for sim, snap, snap_dir, raw in out:
        z_meta = float(json.load(open(snap_dir / "meta.json"))["z"])
        z_grid = _snap_z_to_priya_grid(z_meta)
        off = abs(z_meta - z_grid)
        key = (sim, round(z_grid, 4))
        if key not in best or off < best[key][0]:
            best[key] = (off, (sim, snap, snap_dir, raw))
    deduped = []
    for (sim, z_grid), (off, pair) in best.items():
        if off > max_off_grid:
            print(f"  WARN: skip off-grid snap {pair[0][:20]} {pair[1]} "
                  f"(z_grid={z_grid}, off={off:.3f} > {max_off_grid})")
            continue
        deduped.append(pair)
    return sorted(deduped, key=lambda p: (p[0], p[1]))


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
                    n_k, n_skewers=None):
    """Build per-alpha rows (Tier-P total + Tier-C per-fine-bin P1D) and the
    per-snap CDDF block for one (sim, snap), driving fake_spectra directly.

    Tier P: PRIYA-compatible total P1D over all sightlines after the
    _filter_single_tau_complex DLA mask. Bit-identical to PRIYA's flux_vectors.
    Stores the first n_k native FFT bins.
    n_k: number of leading native-FFT k-bins to retain (must be <= the snap's
    native grid length, approximately nbins//2).
    Tier C: per-fine-NHI-bin (N_TIER_C_BINS=15) P1D on the UNFILTERED tau,
    sharing Tier P's scale + target_F (so they decompose against the total).
    Stores P_by_bin[:, :n_k] and n_by_bin counts.

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

    hdr = read_header(raw_tau_path)
    nbins = int(hdr.nbins)
    z_raw = float(hdr.redshift)
    dv_kms = float(meta["dv_kms"])
    vmax = nbins * dv_kms

    z_meta = float(meta["z"])
    z_grid = _snap_z_to_priya_grid(z_meta)
    # Defensive: raw header z must agree with Phase-1 meta z (catches mis-pairing
    # across the cross-sim SPECTRA-numbering differences).
    assert abs(z_raw - z_meta) < 1e-2, \
        f"z mismatch raw={z_raw} meta={z_meta} for {sim_name} snap {snap}"

    tau_unfilt = _read_tau(raw_tau_path, n_skewers=n_skewers)
    rows = []
    for a_idx, alpha in enumerate(alpha_slope_grid):
        alpha = float(alpha)
        tau_filt = tau_unfilt.copy()
        kf_p, P_tier_p, target_F, scale = compute_tier_p_p1d(
            tau_filt, vmax, alpha_slope=alpha, z=z_grid)
        # Filtered Tier C: per-class on PRIYA's whole-array-filtered tau (sums to Tier P).
        _, P_by_bin_filt, n_by_bin_filt, _, _ = compute_tier_c_p1d(
            tau_filt, vmax, alpha_slope=alpha, z=z_grid, catalog=catalog,
            external_scale=scale, external_target_F=target_F)
        del tau_filt
        # Unfiltered Tier C (for the HCD add-back path):
        kf_c, P_by_bin, n_by_bin, _, _ = compute_tier_c_p1d(
            tau_unfilt, vmax, alpha_slope=alpha, z=z_grid, catalog=catalog,
            external_scale=scale, external_target_F=target_F)
        assert np.array_equal(n_by_bin, n_by_bin_filt), \
            "N_HI class counts differ between filtered and unfiltered Tier C"
        if a_idx == 0:
            assert len(kf_p) >= n_k and len(kf_c) >= n_k, \
                f"native grid {len(kf_p)} bins < n_k={n_k} for {sim_name} snap {snap}"
        rows.append({
            "sim_name": sim_name, "snap": int(snap),
            "alpha_slope": alpha, "alpha_idx": int(a_idx),
            "z_meta": z_meta, "z_grid": float(z_grid),
            "dv_kms": dv_kms, "nbins_native": nbins,
            "target_F": float(target_F), "scale": float(scale),
            "params": params,
            "kfkms": kf_p[:n_k].astype(np.float64),
            "P_tier_p": P_tier_p[:n_k].astype(np.float64),
            "P_tier_c": P_by_bin[:, :n_k].astype(np.float64),
            "P_tier_c_filtered": P_by_bin_filt[:, :n_k].astype(np.float64),
            "tier_c_counts": n_by_bin.astype(np.int64),
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
_ROW_INT_KEYS = ("snap", "alpha_idx", "nbins_native", "snap_group_idx")
_ROW_P1D_KEYS = ("kfkms", "P_tier_p")          # 2-D (n_rows, n_k)
_ROW_TIERC_KEYS = ("P_tier_c", "P_tier_c_filtered")  # 3-D (n_rows, N_TIER_C_BINS, n_k)
_ROW_COUNT_KEYS = ("tier_c_counts",)            # 2-D (n_rows, N_TIER_C_BINS)
_SNAP_FLOAT_KEYS = ("total_path_dX", "dNdX_LLS", "dNdX_subDLA", "dNdX_DLA")
_SNAP_2D_KEYS = ("f_nhi", "n_absorbers")


def write_cache_tau0(rows, snap_blocks, output_path, alpha_range, n_k):
    """Stack per-alpha `rows` + per-snap `snap_blocks` into one HDF5 cache (v3.1).

    Each row carries `snap_group_idx`, an index into the snap_* datasets.
    Schema v3.1: per-row kfkms + P_tier_p (2-D), P_tier_c (3-D),
    tier_c_counts (2-D); tier_c_labels/tier_c_nhi_edges descriptors; no k_target.
    """
    from hcd_analysis.priya_p1d import tier_c_labels, FINE_NHI_EDGES, N_TIER_C_BINS
    assert len(tier_c_labels()) == N_TIER_C_BINS == len(FINE_NHI_EDGES) + 1
    if not rows:
        raise ValueError("write_cache_tau0 called with no rows.")
    if not snap_blocks:
        raise ValueError("write_cache_tau0 called with no snap_blocks.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log_nhi_centres = snap_blocks[0]["log_nhi_centres"]
    log_nhi_edges = snap_blocks[0]["log_nhi_edges"]

    with h5py.File(output_path, "w") as f:
        f.attrs["created_utc"] = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
        f.attrs["git_sha"] = bec._git_sha(REPO_ROOT)
        f.attrs["n_rows"] = len(rows)
        f.attrs["n_snaps"] = len(snap_blocks)
        f.attrs["cache_version"] = "3.1"
        f.attrs["n_k"] = int(n_k)
        f.attrs["priya_convention"] = (
            "Kim 2013 slope-alpha (obs_mean_tau=2.3e-3(1+z)^3.65); "
            "fake_spectra _filter_single_tau_complex(tau_thresh=1e6, thresh2=0.25); "
            "flux_power window=False spec_res=0; native k-grid (first n_k FFT bins)")
        f.attrs["tau_thresh"] = 1.0e6
        f.attrs["alpha_range"] = np.asarray(alpha_range, dtype=np.float64)
        f.attrs["k_convention"] = "angular (rad*s/km) native FFT grid, PRIYA convention"
        f.attrs["tier_c_note"] = (
            "P_tier_c = per-class P1D on UNFILTERED tau (HCD add-back); "
            "P_tier_c_filtered = on PRIYA tau=1e6-filtered tau "
            "(count-weighted sum == Tier P).")

        f.create_dataset("tier_c_labels",
                         data=np.array(tier_c_labels(), dtype=h5py.string_dtype()))
        f.create_dataset("tier_c_nhi_edges", data=np.asarray(FINE_NHI_EDGES, dtype=np.float64))
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
        for key in _ROW_TIERC_KEYS:
            f.create_dataset(key, data=np.stack([r[key] for r in rows], axis=0))
        for key in _ROW_COUNT_KEYS:
            f.create_dataset(key, data=np.stack([r[key] for r in rows], axis=0).astype(np.int64))

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


def _default_output(fidelity) -> Path:
    return REPO_ROOT / "hcd_analysis" / "_emulator_data" / f"observables_tau0_{fidelity}.h5"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the tau0-extended HCD-emulator cache (Phase 2).")
    parser.add_argument("--fidelity", choices=("lf", "hr"), default="lf")
    parser.add_argument("--hcd-root", type=Path, default=None)
    parser.add_argument("--emu-root", type=Path, default=None)
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

    fid = args.fidelity
    hcd_root = args.hcd_root or _HCD_OUTPUTS
    emu_root = args.emu_root or (_HR_EMU_ROOT if fid == "hr" else _LF_EMU_ROOT)
    n_k = _N_K[fid]
    alpha_slope_grid = make_alpha_grid_priya_aligned(refine=args.alpha_refine)
    output = args.output or _default_output(fid)

    pairs = discover_tau0_pairs(hcd_root, emu_root, fidelity=fid)
    print(f"[{fid}] Found {len(pairs)} tau0-buildable (sim, snap) pairs (n_k={n_k})")
    pairs = pairs[args.offset:]
    if args.limit is not None:
        pairs = pairs[: args.limit]
    print(f"Processing {len(pairs)} pairs (offset={args.offset}, limit={args.limit}); "
          f"n_alpha={len(alpha_slope_grid)}")

    all_rows, snap_blocks = [], []
    for i, (sim, snap, snap_dir, raw) in enumerate(pairs):
        rows, block = build_tau0_rows(sim, snap, snap_dir, raw, alpha_slope_grid,
                                      n_k=n_k, n_skewers=args.n_skewers)
        gi = len(snap_blocks)
        for r in rows:
            r["snap_group_idx"] = gi
        all_rows.extend(rows)
        snap_blocks.append(block)
        if (i + 1) % 10 == 0 or (i + 1) == len(pairs):
            print(f"  built {i+1}/{len(pairs)} pairs ({len(all_rows)} rows)")

    write_cache_tau0(all_rows, snap_blocks, output,
                     alpha_range=(float(alpha_slope_grid[0]), float(alpha_slope_grid[-1])),
                     n_k=n_k)
    print(f"Wrote {output} ({output.stat().st_size/1e6:.2f} MB, "
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
