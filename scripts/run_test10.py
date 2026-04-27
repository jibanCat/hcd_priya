"""
Test 10 — DLA × Lyα cross-correlation gate (FR+2012 sanity).

Uses the b_F = -0.141 measured in test 11 as the calibrator, then
extracts b_DLA from the ξ_×(r_par, r_perp) cross-correlation monopole:

    ξ_×^(0)(r) = b_DLA · b_F · K_×(β_DLA, β_F) · ξ_lin^(0)(r)

PASS criteria (per docs/clustering_definitions.md §8 test 10):
  - b_DLA from ξ_× lies in FR+2012 envelope [1.7, 2.5].

Default sim/snap: ns0.803Ap2.2e-09…/snap_022 at z = 2.20 (matches
test 11; b_F input is the same-snap measurement, not literature).

Run::

    python3 scripts/run_test10.py

Optional: --b-F to override the bF used (default reads from
test11_snap_022.json if present, falls back to -0.141).
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
    fold_signed_to_abs,
    load_sightline_geometry,
    pixel_to_xyz,
    xi_cross_dla_lya,
)
from hcd_analysis.lya_bias import (
    find_camb_pk_for_z,
    fit_b_DLA_from_xi_cross,
    hMpc_to_kms_factor,
    load_camb_pk,
)


SIM = "ns0.803Ap2.2e-09herei4.05heref2.67alphaq2.21hub0.735omegamh20.141hireionz7.17bhfeedback0.056"
DEFAULT_SNAP = "snap_022"
EMU = Path("/nfs/turbo/umor-yueyingn/mfho/emu_full")
SCRATCH = Path("/scratch/cavestru_root/cavestru0/mfho/hcd_outputs")
OUT_DIR = ROOT / "figures" / "analysis" / "06_clustering"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TEST11_JSON = OUT_DIR / "test11_snap_022.json"


def _load_b_F() -> float:
    if TEST11_JSON.exists():
        with open(TEST11_JSON) as f:
            d = json.load(f)
        if d.get("b_F_xi") is not None:
            return float(d["b_F_xi"])
    return -0.141      # fallback to the value reported in clustering_test11_results.md


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim", default=SIM)
    p.add_argument("--snap", default=DEFAULT_SNAP)
    p.add_argument("--xi-pixel-subsample", type=int, default=200000,
                   help="Lyα-pixel subsample for the cross (default 200000)")
    p.add_argument("--beta-iter", type=int, default=4,
                   help="β_DLA self-consistency iterations (default 4)")
    p.add_argument("--b-F", type=float, default=None,
                   help="b_F to use (default: read from test 11 JSON)")
    p.add_argument("--rng-seed", type=int, default=2026)
    args = p.parse_args()

    sim_dir_emu = EMU / args.sim
    sim_dir_scratch = SCRATCH / args.sim
    snap_idx = int(args.snap.split("_")[1])
    spec_path = sim_dir_emu / "output" / f"SPECTRA_{snap_idx:03d}" / "lya_forest_spectra_grid_480.hdf5"
    cat_path = sim_dir_scratch / args.snap / "catalog.npz"

    b_F = args.b_F if args.b_F is not None else _load_b_F()
    print(f"=== Test 10 — {args.sim[:40]} {args.snap} ===")
    print(f"  using b_F = {b_F:+.4f} (calibrator)")

    # 1) Load geometry, τ, catalog
    t0 = time.time()
    geom = load_sightline_geometry(spec_path)
    with h5py.File(spec_path, "r") as f:
        tau = f["tau/H/1/1215"][:]
        Hz = float(f["Header"].attrs["Hz"])
        Om0_sim = float(f["Header"].attrs["omegam"])         # Copilot #12
    cat = np.load(str(cat_path))                              # Copilot #6: drop allow_pickle
    F_factor = hMpc_to_kms_factor(geom.z_snap, geom.hubble, Hz)
    box_kms = geom.box / F_factor
    dv_kms = box_kms / geom.n_pix
    print(f"  loaded geom + tau + catalog in {time.time() - t0:.1f}s "
          f"(z={geom.z_snap:.3f}, dv={dv_kms:.2f} km/s)")

    # 2) Build δ_F (all-HCD masked) — same convention as test 11
    t0 = time.time()
    df_res = build_delta_F_field(
        tau=tau, skewer_idx=cat["skewer_idx"],
        pix_start=cat["pix_start"], pix_end=cat["pix_end"], NHI=cat["NHI"],
    )
    print(f"  δ_F built in {time.time() - t0:.1f}s. ⟨F⟩ = {df_res.mean_F:.4f}, "
          f"masked: {int(df_res.mask.sum()):,} pixels")

    # 3) DLA point catalog
    log_nhi = np.log10(np.maximum(cat["NHI"].astype(np.float64), 1.0))
    is_dla = log_nhi >= 20.3
    n_dla = int(is_dla.sum())
    print(f"  DLA count (log NHI ≥ 20.3): {n_dla}")
    if n_dla < 100:
        print(f"  WARNING: very few DLAs — cross-corr will be noisy")
    # DLA 3D positions: use the centre pixel of each [pix_start, pix_end] range
    dla_pix_centre = (
        cat["pix_start"][is_dla].astype(np.int64)
        + cat["pix_end"][is_dla].astype(np.int64)
    ) // 2
    dla_skewer = cat["skewer_idx"][is_dla].astype(np.int64)
    dla_xyz = pixel_to_xyz(geom, dla_skewer, dla_pix_centre)

    # 4) Subsampled Lyα-pixel arrays for the cross.
    #
    # Copilot review #11 on PR #7: the previous version did
    #     all_skewer = np.repeat(arange(n_sightlines), n_pix)
    #     all_pixel  = np.tile(arange(n_pix), n_sightlines)
    # which materialises three n_sightlines·n_pix arrays (~3 × 5.6 GB
    # at LF resolution) before subsampling — wasteful and OOM-prone
    # on smaller nodes.  Replace with np.flatnonzero on the unmasked
    # mask (peak ~5.6 GB once) and divmod to recover (skewer, pixel).
    rng = np.random.default_rng(args.rng_seed)
    unmasked_idx = np.flatnonzero(~df_res.mask.ravel())   # one allocation
    if 0 < args.xi_pixel_subsample < unmasked_idx.size:
        chosen_local = rng.choice(unmasked_idx.size, size=args.xi_pixel_subsample,
                                  replace=False)
        chosen_local.sort()
        chosen_idx = unmasked_idx[chosen_local]
    else:
        chosen_idx = unmasked_idx
    n_pix = geom.n_pix
    all_skewer = (chosen_idx // n_pix).astype(np.int64)
    all_pixel = (chosen_idx % n_pix).astype(np.int64)
    df_flat = df_res.delta_F.ravel()[chosen_idx]
    pixel_xyz = pixel_to_xyz(geom, all_skewer, all_pixel)
    pixel_axis = geom.axis[all_skewer]
    print(f"  Lyα-pixel sample size for ξ_×: {len(all_skewer):,}")

    # 5) ξ_× pair count
    t0 = time.time()
    r_perp_edges = np.linspace(0.0, 50.0, 26)
    r_par_edges = np.linspace(-50.0, 50.0, 51)
    xi_signed, counts, npairs = xi_cross_dla_lya(
        pixel_xyz=pixel_xyz, pixel_los_axis=pixel_axis,
        pixel_delta_F=df_flat,
        dla_xyz=dla_xyz,
        box=geom.box,
        r_perp_bins=r_perp_edges, r_par_bins_signed=r_par_edges,
        chunk_size=2048,
    )
    print(f"  ξ_× computed in {time.time() - t0:.1f}s; "
          f"total pairs = {npairs.sum():,}")
    # Copilot #13: pair-count-weighted fold, not nanmean.  Pass counts
    # and npairs so the folded ξ uses (counts_pos+counts_neg) /
    # (npairs_pos+npairs_neg).
    xi_folded, par_edges_abs = fold_signed_to_abs(
        xi_signed, r_par_edges, counts=counts, npairs=npairs,
    )
    n_par = len(r_par_edges) - 1
    half = n_par // 2
    n_pair_folded = npairs[:, half:] + npairs[:, :half][:, ::-1]

    # 6) Fit b_DLA from the monopole, with β_DLA self-consistency iteration.
    perp_centres = 0.5 * (r_perp_edges[:-1] + r_perp_edges[1:])
    par_centres_abs = 0.5 * (par_edges_abs[:-1] + par_edges_abs[1:])
    camb_path = find_camb_pk_for_z(sim_dir_emu / "output", geom.z_snap)
    k_h, P_h = load_camb_pk(camb_path)

    # Compute f(z) ≈ Ω_m(z)^0.55  (Linder 2005 fitting form).
    # Copilot review #12 on PR #7: read Ω_m(z=0) from the spectra
    # HDF5 Header instead of the hardcoded 0.26.  PRIYA varies
    # `omegamh2` and `h` across sims; using a fixed Ω_m would put
    # f(z) off by tens of percent on extreme grid points.
    Om0 = Om0_sim                                # from f["Header"].attrs["omegam"]
    Om_z = Om0 * (1.0 + geom.z_snap) ** 3 / (
        Om0 * (1.0 + geom.z_snap) ** 3 + (1.0 - Om0)
    )
    f_z = float(Om_z ** 0.55)
    print(f"  f(z={geom.z_snap:.2f}) = {f_z:.3f} (Ω_m,0={Om0:.3f}); "
          f"β_DLA iteration starts at 0.5, updates as f/b_DLA")

    res = None
    beta_DLA = 0.5
    history = []
    for it in range(args.beta_iter):
        try:
            r = fit_b_DLA_from_xi_cross(
                xi_2d=xi_folded, npairs_2d=n_pair_folded,
                r_perp_centres=perp_centres, r_par_centres=par_centres_abs,
                k_lin=k_h, P_lin=P_h,
                b_F=b_F, beta_DLA=beta_DLA, beta_F=1.5,
                r_min=10.0, r_max=40.0, n_r_bins=10,
            )
        except Exception as exc:
            print(f"  iter {it}: fit FAILED: {exc}")
            break
        history.append((beta_DLA, r.b_DLA, r.b_DLA_err))
        new_beta = f_z / max(abs(r.b_DLA), 0.1)
        print(f"  iter {it}: β_DLA={beta_DLA:.3f} → b_DLA={r.b_DLA:+.3f} "
              f"± {r.b_DLA_err:.3f}; next β_DLA = f/b = {new_beta:.3f}")
        res = r
        if abs(new_beta - beta_DLA) < 5e-3:
            print(f"  (converged at iter {it})")
            break
        beta_DLA = new_beta

    if res is not None:
        print(f"\n  Final b_DLA = {res.b_DLA:+.3f} ± {res.b_DLA_err:.3f}  "
              f"(β_DLA={res.beta_DLA_assumed:.3f}, K_×={res.K_cross:.3f}, "
              f"n_fit_bins={res.n_fit_bins})")

    # 7) Verdict
    # Comparison envelope spans Bird+2014 hydro-sim linear theory (b ≈ 1.7)
    # and BOSS observations (FR+2012: 2.17, Pérez-Ràfols+2018: 1.99).
    # See docs/clustering_test10_results.md for the literature breakdown.
    print("\n--- Verdict ---")
    LIT_LO, LIT_HI = 1.5, 2.4         # Bird+2014 sim → BOSS obs envelope
    BIRD14_LIN = 1.7                   # Bird+2014 linear-theory hydro prediction
    if res is not None:
        in_range = LIT_LO <= res.b_DLA <= LIT_HI
        delta_bird = abs(res.b_DLA - BIRD14_LIN)
        print(f"  H10.a  b_DLA in [Bird+14 sim, FR+2012 obs] envelope "
              f"[{LIT_LO}, {LIT_HI}] : "
              f"{'PASS' if in_range else 'FAIL'} ({res.b_DLA:+.3f})")
        print(f"  H10.b  |b_DLA − Bird+2014 linear (1.7)| = {delta_bird:.3f} : "
              f"{'PASS' if delta_bird < res.b_DLA_err else 'review'}")

    # 8) Figure
    if res is not None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        ax.plot(res.r_centres, res.xi_obs, "o", color="C0", label="ξ_×^(0) obs")
        ax.plot(res.r_centres, res.xi_template, "-", color="C3",
                label=f"b_DLA·b_F·K(β)·ξ_lin  (b_DLA={res.b_DLA:+.2f})")
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_xlabel("r [Mpc/h]")
        ax.set_ylabel(r"$\xi_{\times}^{(0)}(r)$")
        ax.set_title(f"Test 10 — DLA × Lyα cross  z={geom.z_snap:.2f}  "
                     f"(N_DLA={n_dla}, b_F={b_F:+.3f})")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        out_png = OUT_DIR / f"test10_{args.snap}.png"
        fig.tight_layout()
        fig.savefig(out_png, dpi=130, bbox_inches="tight")
        print(f"\n  wrote {out_png}")

    # 9) JSON dump
    out_json = OUT_DIR / f"test10_{args.snap}.json"
    summary = {
        "sim": args.sim, "snap": args.snap, "z": float(geom.z_snap),
        "n_dla": n_dla,
        "b_F_assumed": float(b_F),
        "b_DLA": float(res.b_DLA) if res is not None else None,
        "b_DLA_err": float(res.b_DLA_err) if res is not None else None,
        "in_lit_envelope": bool(LIT_LO <= res.b_DLA <= LIT_HI) if res is not None else None,
        "delta_to_bird14_linear": (
            float(abs(res.b_DLA - BIRD14_LIN)) if res is not None else None
        ),
        "lyα_pixel_subsample": int(args.xi_pixel_subsample),
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  wrote {out_json}")


if __name__ == "__main__":
    main()
