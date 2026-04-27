"""
Test 11 — real-PRIYA gate for the clustering pipeline.

Builds the all-HCD-masked δ_F field on a real LF snap at z ≈ 2.3, then
extracts b_F two independent ways:

  1. From the masked-P1D vs the linear-theory template
     (`hcd_analysis.lya_bias.fit_b_F`, the "P1D path").
  2. From the ξ_FF(r_par, r_perp) monopole vs ξ_lin(r) on linear scales
     (`fit_b_F_from_xi_FF`, the "ξ_FF path").

PASS criteria (per `docs/clustering_definitions.md` §8 test 11):
  - b_F from each path lies in [-0.25, -0.12] (Slosar+2011 / du Mas
    des Bourboux+2020 envelope at z ≈ 2.3).
  - The two b_F values agree within 1 σ.

Default sim/snap: `ns0.803Ap2.2e-09…/snap_022` at z = 2.2 (closest
LF emu_full snap to z = 2.3).

Run::

    python3 scripts/run_test11.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hcd_analysis.clustering import (
    build_delta_F_field,
    load_sightline_geometry,
    pixel_to_xyz,
    fold_signed_to_abs,
    xi_auto_lya,
)
from hcd_analysis.lya_bias import (
    find_camb_pk_for_z,
    fit_b_F,
    fit_b_F_from_xi_FF,
    hMpc_to_kms_factor,
    load_camb_pk,
    xi_lin_monopole,
)


SIM = "ns0.803Ap2.2e-09herei4.05heref2.67alphaq2.21hub0.735omegamh20.141hireionz7.17bhfeedback0.056"
DEFAULT_SNAP = "snap_022"      # z ≈ 2.2 (closest to 2.3)
EMU = Path("/nfs/turbo/umor-yueyingn/mfho/emu_full")
SCRATCH = Path("/scratch/cavestru_root/cavestru0/mfho/hcd_outputs")
OUT_DIR = ROOT / "figures" / "analysis" / "06_clustering"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim", default=SIM)
    p.add_argument("--snap", default=DEFAULT_SNAP)
    p.add_argument("--xi-subsample", type=int, default=8000,
                   help="ξ_FF pixel subsample per axis (default 8000 → ~24k total)")
    p.add_argument("--rng-seed", type=int, default=2026)
    args = p.parse_args()

    sim_dir_emu = EMU / args.sim
    sim_dir_scratch = SCRATCH / args.sim
    snap_idx = int(args.snap.split("_")[1])
    spec_path = sim_dir_emu / "output" / f"SPECTRA_{snap_idx:03d}" / "lya_forest_spectra_grid_480.hdf5"
    cat_path = sim_dir_scratch / args.snap / "catalog.npz"

    print(f"=== Test 11 — {args.sim[:40]} {args.snap} ===")
    print(f"spectra: {spec_path}")
    print(f"catalog: {cat_path}")

    # 1) Load geometry, τ, and catalog
    t0 = time.time()
    geom = load_sightline_geometry(spec_path)
    with h5py.File(spec_path, "r") as f:
        tau = f["tau/H/1/1215"][:]
    cat = np.load(str(cat_path), allow_pickle=True)
    print(f"  loaded: {tau.shape} τ, {len(cat['NHI'])} catalog absorbers, "
          f"z={geom.z_snap:.4f}, h={geom.hubble:.3f}, box={geom.box:.1f} Mpc/h, "
          f"dv≈{geom.dx_pix * geom.hubble * 100 * 1e3:.1f} km/s "
          f"(in {time.time() - t0:.1f}s)")
    # Compute dv_kms exactly: from box + Hz
    with h5py.File(spec_path, "r") as f:
        Hz = float(f["Header"].attrs["Hz"])
    F = hMpc_to_kms_factor(geom.z_snap, geom.hubble, Hz)
    box_kms = geom.box / F                # Mpc/h → km/s
    dv_kms = box_kms / geom.n_pix
    print(f"  dv_kms = {dv_kms:.3f}; box_LOS = {box_kms:.0f} km/s")

    # 2) Build δ_F with all-HCD masking
    t0 = time.time()
    df_res = build_delta_F_field(
        tau=tau,
        skewer_idx=cat["skewer_idx"],
        pix_start=cat["pix_start"],
        pix_end=cat["pix_end"],
        NHI=cat["NHI"],
    )
    n_skewers, n_pix = df_res.delta_F.shape
    n_masked = int(df_res.mask.sum())
    print(f"  δ_F built in {time.time() - t0:.1f}s. ⟨F⟩ = {df_res.mean_F:.4f}, "
          f"n_masked = {n_masked} ({100 * n_masked / df_res.mask.size:.2f}% of pixels). "
          f"Per-class: {df_res.n_masked_per_class}")
    n_clean = int(np.sum(~df_res.mask.any(axis=1)))
    print(f"  n_clean_sightlines = {n_clean}/{n_skewers}")

    # 3) Fit b_F via P1D
    print("\n--- Path 1: b_F from P1D vs linear theory ---")
    t0 = time.time()
    camb_path = find_camb_pk_for_z(sim_dir_emu / "output", geom.z_snap)
    print(f"  CAMB P_lin: {camb_path.name}")
    p1d_res = fit_b_F(
        delta_F=df_res.delta_F,
        pixel_mask=df_res.mask,
        dv_kms=dv_kms, z=geom.z_snap, hubble=geom.hubble,
        Hz_kms_per_Mpc=Hz,
        P_lin_camb_path=camb_path,
        beta_F_assume=1.5,
        k_min_kms=5e-4, k_max_kms=5e-3,
    )
    print(f"  b_F (P1D) = {p1d_res.b_F:+.4f} ± {p1d_res.b_F_err:.4f}  "
          f"(β_F={p1d_res.beta_F_assumed} fixed; "
          f"n_clean={p1d_res.n_clean_sightlines}; n_fit_bins={int(p1d_res.fit_mask.sum())}; "
          f"χ²={p1d_res.chi2:.2f}; in {time.time() - t0:.1f}s)")

    # 4) Build pixel arrays for ξ_FF
    print("\n--- Path 2: b_F from ξ_FF monopole ---")
    t0 = time.time()
    # Build all (skewer, pixel) → 3D positions
    all_skewer = np.repeat(np.arange(n_skewers, dtype=np.int64), n_pix)
    all_pixel = np.tile(np.arange(n_pix, dtype=np.int64), n_skewers)
    # Skip masked pixels (their δ_F = 0 contributes nothing anyway,
    # and we'd prefer to keep the array small)
    keep_flat = ~df_res.mask.ravel()
    all_skewer = all_skewer[keep_flat]
    all_pixel = all_pixel[keep_flat]
    df_flat = df_res.delta_F.ravel()[keep_flat]
    print(f"  unmasked pixels: {len(all_skewer):,}")

    # Subsample to keep pair-count tractable
    rng = np.random.default_rng(args.rng_seed)
    if args.xi_subsample > 0 and args.xi_subsample < len(all_skewer):
        idx = rng.choice(len(all_skewer), size=args.xi_subsample, replace=False)
        idx.sort()
        all_skewer = all_skewer[idx]
        all_pixel = all_pixel[idx]
        df_flat = df_flat[idx]
        print(f"  subsampled to {len(all_skewer):,}")
    pixel_xyz = pixel_to_xyz(geom, all_skewer, all_pixel)
    pixel_axis = geom.axis[all_skewer]

    # Pair counter on the subsampled pixels
    r_perp_edges = np.linspace(0.0, 50.0, 26)
    r_par_edges = np.linspace(-50.0, 50.0, 51)
    xi_signed, counts, npairs = xi_auto_lya(
        pixel_xyz=pixel_xyz, pixel_los_axis=pixel_axis, pixel_delta_F=df_flat,
        box=geom.box, r_perp_bins=r_perp_edges, r_par_bins_signed=r_par_edges,
        chunk_size=2048,
    )
    # Fold to |r_par|
    xi_folded, par_edges_abs = fold_signed_to_abs(xi_signed, r_par_edges)
    # Fold npairs by SUM (not nanmean) for the monopole weighting
    n_par = len(r_par_edges) - 1
    half = n_par // 2
    n_pair_folded = npairs[:, half:] + npairs[:, :half][:, ::-1]
    print(f"  ξ_FF computed in {time.time() - t0:.1f}s; "
          f"total pairs = {npairs.sum():,}; "
          f"max bin pairs = {npairs.max()}")

    # 5) Fit b_F from ξ_FF monopole
    perp_centres = 0.5 * (r_perp_edges[:-1] + r_perp_edges[1:])
    par_centres_abs = 0.5 * (par_edges_abs[:-1] + par_edges_abs[1:])
    k_h, P_h = load_camb_pk(camb_path)
    try:
        xi_res = fit_b_F_from_xi_FF(
            xi_2d=xi_folded, npairs_2d=n_pair_folded,
            r_perp_centres=perp_centres, r_par_centres=par_centres_abs,
            k_lin=k_h, P_lin=P_h, beta_F=1.5,
            r_min=10.0, r_max=40.0, n_r_bins=10,
        )
        print(f"  b_F (ξ_FF) = {xi_res.b_F:+.4f} ± {xi_res.b_F_err:.4f}  "
              f"(β_F={xi_res.beta_F_assumed} fixed; n_fit_bins={xi_res.n_fit_bins})")
    except Exception as exc:
        print(f"  ξ_FF fit FAILED: {exc}")
        xi_res = None

    # 6) Verdict
    print("\n--- Verdict ---")
    LIT_LO, LIT_HI = -0.25, -0.12
    p1d_in = LIT_LO <= p1d_res.b_F <= LIT_HI
    xi_in = (xi_res is not None) and (LIT_LO <= xi_res.b_F <= LIT_HI)
    consistent = (
        xi_res is not None
        and abs(p1d_res.b_F - xi_res.b_F) <= 1.0 * np.hypot(p1d_res.b_F_err, xi_res.b_F_err)
    )
    print(f"  H11.a  b_F (P1D)   in [{LIT_LO}, {LIT_HI}] : "
          f"{'PASS' if p1d_in else 'FAIL'} ({p1d_res.b_F:+.4f})")
    if xi_res is not None:
        print(f"  H11.b  b_F (ξ_FF)  in [{LIT_LO}, {LIT_HI}] : "
              f"{'PASS' if xi_in else 'FAIL'} ({xi_res.b_F:+.4f})")
        print(f"  H11.c  P1D vs ξ_FF agree within 1σ        : "
              f"{'PASS' if consistent else 'FAIL'}")

    # 7) Save figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # Left: P1D fit
    axL = axes[0]
    fm = p1d_res.fit_mask
    axL.loglog(p1d_res.k_par_kms, p1d_res.P1D_obs_kms, ".", color="C0", label="P1D obs (clean SLs)")
    axL.loglog(p1d_res.k_par_kms[fm], p1d_res.P1D_template_kms[fm],
               "-", color="C3", label=f"template b_F²·I  (b_F={p1d_res.b_F:+.3f})")
    axL.axvspan(5e-4, 5e-3, color="grey", alpha=0.15, label="fit window")
    axL.set_xlabel("k [s/km]")
    axL.set_ylabel("P1D [km/s]")
    axL.set_title(f"P1D path  (z={geom.z_snap:.2f})")
    axL.grid(True, alpha=0.3, which="both")
    axL.legend(fontsize=8)
    # Right: ξ_FF monopole fit
    axR = axes[1]
    if xi_res is not None:
        axR.plot(xi_res.r_centres, xi_res.xi_obs, "o", color="C0", label="ξ_FF^(0) obs")
        axR.plot(xi_res.r_centres, xi_res.xi_template, "-", color="C3",
                 label=f"b_F²·K(β)·ξ_lin  (b_F={xi_res.b_F:+.3f})")
        axR.set_xlabel("r [Mpc/h]")
        axR.set_ylabel(r"$\xi_{FF}^{(0)}(r)$")
        axR.set_title("ξ_FF path")
        axR.grid(True, alpha=0.3)
        axR.legend(fontsize=8)
    fig.suptitle(f"Test 11 — {args.sim[:30]}…  {args.snap}  z={geom.z_snap:.2f}",
                 fontsize=11)
    fig.tight_layout()
    out_png = OUT_DIR / f"test11_{args.snap}.png"
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"\n  wrote {out_png}")

    # 8) JSON dump for downstream consumers
    out_json = OUT_DIR / f"test11_{args.snap}.json"
    summary = {
        "sim": args.sim, "snap": args.snap, "z": float(geom.z_snap),
        "mean_F": float(df_res.mean_F),
        "n_clean_sightlines": int(p1d_res.n_clean_sightlines),
        "b_F_p1d": float(p1d_res.b_F),
        "b_F_p1d_err": float(p1d_res.b_F_err),
        "b_F_xi": float(xi_res.b_F) if xi_res is not None else None,
        "b_F_xi_err": float(xi_res.b_F_err) if xi_res is not None else None,
        "p1d_in_lit": bool(p1d_in),
        "xi_in_lit": bool(xi_in) if xi_res is not None else None,
        "agree_1sigma": bool(consistent) if xi_res is not None else None,
        "xi_subsample": int(args.xi_subsample),
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  wrote {out_json}")
    return summary


if __name__ == "__main__":
    main()
