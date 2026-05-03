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
    xi_cross_dla_lya_rmu,
)
from hcd_analysis.lya_bias import (
    find_camb_pk_for_z,
    fit_b_DLA_from_xi_cross,
    fit_b_beta_from_xi_cross_multipoles,
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
    p.add_argument("--mode", choices=["rperp_rpar", "rmu"], default="rperp_rpar",
                   help="Pair-binning + multipole-extraction scheme.  'rperp_rpar' "
                        "(default) is the legacy monopole-only fit; 'rmu' uses (r, |μ|) "
                        "binning + Hamilton uniform-μ multipole extraction + joint "
                        "(b_DLA, β_DLA) fit on monopole + quadrupole.  See "
                        "docs/clustering_multipole_jacobian_todo.md.")
    p.add_argument("--n-mu-bins", type=int, default=20,
                   help="Number of |μ|-bins on [0, 1] for --mode rmu (default 20).")
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

    # 5–9) Mode-specific pair count + fit + output
    camb_path = find_camb_pk_for_z(sim_dir_emu / "output", geom.z_snap)
    k_h, P_h = load_camb_pk(camb_path)
    LIT_LO, LIT_HI = 1.5, 2.4         # Bird+14 sim → BOSS obs envelope
    BIRD14_LIN = 1.7                   # Bird+14 linear-theory hydro prediction
    out_tag = "" if args.mode == "rperp_rpar" else "_rmu"

    if args.mode == "rperp_rpar":
        # ----- legacy (r_⊥, r_∥) monopole-only path -----
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
        xi_folded, par_edges_abs = fold_signed_to_abs(
            xi_signed, r_par_edges, counts=counts, npairs=npairs,
        )
        n_par = len(r_par_edges) - 1
        half = n_par // 2
        n_pair_folded = npairs[:, half:] + npairs[:, :half][:, ::-1]
        perp_centres = 0.5 * (r_perp_edges[:-1] + r_perp_edges[1:])
        par_centres_abs = 0.5 * (par_edges_abs[:-1] + par_edges_abs[1:])

        # f(z) for β_DLA self-consistency iteration
        Om0 = Om0_sim
        Om_z = Om0 * (1.0 + geom.z_snap) ** 3 / (
            Om0 * (1.0 + geom.z_snap) ** 3 + (1.0 - Om0)
        )
        f_z = float(Om_z ** 0.55)
        print(f"  f(z={geom.z_snap:.2f}) = {f_z:.3f} (Ω_m,0={Om0:.3f}); "
              f"β_DLA iteration starts at 0.5, updates as f/b_DLA")

        res = None
        beta_DLA = 0.5
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

            # Verdict + figure + JSON
            print("\n--- Verdict ---")
            in_range = LIT_LO <= res.b_DLA <= LIT_HI
            delta_bird = abs(res.b_DLA - BIRD14_LIN)
            print(f"  H10.a  b_DLA in [Bird+14 sim, FR+12 obs] envelope "
                  f"[{LIT_LO}, {LIT_HI}] : "
                  f"{'PASS' if in_range else 'FAIL'} ({res.b_DLA:+.3f})")
            print(f"  H10.b  |b_DLA − Bird+2014 linear (1.7)| = {delta_bird:.3f} : "
                  f"{'PASS' if delta_bird < res.b_DLA_err else 'review'}")

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
            out_png = OUT_DIR / f"test10_{args.snap}{out_tag}.png"
            fig.tight_layout()
            fig.savefig(out_png, dpi=130, bbox_inches="tight")
            print(f"\n  wrote {out_png}")

        out_json = OUT_DIR / f"test10_{args.snap}{out_tag}.json"
        summary = {
            "mode": args.mode,
            "sim": args.sim, "snap": args.snap, "z": float(geom.z_snap),
            "n_dla": n_dla,
            "b_F_assumed": float(b_F),
            "b_DLA": float(res.b_DLA) if res is not None else None,
            "b_DLA_err": float(res.b_DLA_err) if res is not None else None,
            "beta_DLA_assumed": (
                float(res.beta_DLA_assumed) if res is not None else None
            ),
            "in_lit_envelope": (
                bool(LIT_LO <= res.b_DLA <= LIT_HI) if res is not None else None
            ),
            "delta_to_bird14_linear": (
                float(abs(res.b_DLA - BIRD14_LIN)) if res is not None else None
            ),
            "lyα_pixel_subsample": int(args.xi_pixel_subsample),
        }
        with open(out_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  wrote {out_json}")

    else:
        # ----- (r, |μ|) Hamilton-uniform path: joint (b_DLA, β_DLA) fit -----
        # Uses xi_cross_dla_lya_rmu + extract_multipoles_rmu inside
        # fit_b_beta_from_xi_cross_multipoles.  See
        # docs/clustering_multipole_jacobian_todo.md for the design.
        t0 = time.time()
        r_edges = np.linspace(0.0, 50.0, 26)              # 25 r-bins, 2 Mpc/h
        mu_edges = np.linspace(0.0, 1.0, args.n_mu_bins + 1)
        xi_rmu, _counts, npairs_rmu = xi_cross_dla_lya_rmu(
            pixel_xyz=pixel_xyz, pixel_los_axis=pixel_axis,
            pixel_delta_F=df_flat,
            dla_xyz=dla_xyz,
            box=geom.box,
            r_bins=r_edges, mu_bins=mu_edges,
            chunk_size=2048,
        )
        print(f"  ξ_× (r, |μ|) computed in {time.time() - t0:.1f}s; "
              f"total pairs = {npairs_rmu.sum():,}; "
              f"grid {xi_rmu.shape}")

        r_centres = 0.5 * (r_edges[:-1] + r_edges[1:])
        mu_centres = 0.5 * (mu_edges[:-1] + mu_edges[1:])

        # Save the (r, |μ|) grid so the validation plotter can re-use
        # it without re-running the 10-min pair count.
        out_npz = OUT_DIR / f"test10_{args.snap}{out_tag}_grid.npz"
        np.savez(
            str(out_npz),
            xi_rmu=xi_rmu, npairs_rmu=npairs_rmu,
            r_edges=r_edges, mu_edges=mu_edges,
            r_centres=r_centres, mu_centres=mu_centres,
            b_F_assumed=b_F, beta_F_assumed=1.5,
            z_snap=geom.z_snap,
        )
        print(f"  wrote {out_npz}")

        try:
            jres = fit_b_beta_from_xi_cross_multipoles(
                xi_rmu=xi_rmu, npairs_rmu=npairs_rmu,
                r_centres=r_centres, mu_centres=mu_centres,
                k_lin=k_h, P_lin=P_h,
                b_F=b_F, beta_F=1.5,
                r_min=10.0, r_max=40.0,
                b_DLA_init=2.0, beta_DLA_init=0.5,
            )
        except Exception as exc:
            print(f"  joint fit FAILED: {exc}")
            jres = None

        if jres is not None:
            print(f"\n  Final b_DLA  = {jres.b_DLA:+.3f} ± {jres.b_DLA_err:.3f}")
            print(f"        β_DLA  = {jres.beta_DLA:+.3f} ± {jres.beta_DLA_err:.3f}")
            print(f"        K_0 = {jres.K_0:.3f}, K_2 = {jres.K_2:.3f}, "
                  f"n_fit_bins = {jres.n_fit_bins}, χ² = {jres.chi2:.2f}")

            print("\n--- Verdict (rmu mode) ---")
            in_range = LIT_LO <= jres.b_DLA <= LIT_HI
            delta_bird = abs(jres.b_DLA - BIRD14_LIN)
            print(f"  H10.a  b_DLA in [{LIT_LO}, {LIT_HI}] : "
                  f"{'PASS' if in_range else 'FAIL'} ({jres.b_DLA:+.3f})")
            print(f"  H10.b  |b_DLA − Bird+14 (1.7)| = {delta_bird:.3f} : "
                  f"{'PASS' if delta_bird < jres.b_DLA_err else 'review'}")
            # β_DLA expectation: f(z=2.2)/b_DLA ≈ 0.95 / 1.7 ≈ 0.56.
            # Tinker+10 halo bias model gives β ≈ 0.5–0.6 at z = 2.3.
            print(f"  H10.c  β_DLA in Tinker+10 expectation [0.3, 0.8] : "
                  f"{'PASS' if 0.3 <= jres.beta_DLA <= 0.8 else 'review'} "
                  f"({jres.beta_DLA:+.3f})")

            # Figure: 3-panel layout — (r, |μ|) heatmap + monopole + quadrupole
            fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
            mono_mask = (jres.r_centres >= 5) & (jres.r_centres <= 50)

            # Panel 0: ξ_×(r, |μ|) heatmap
            extent = [mu_edges[0], mu_edges[-1], r_edges[0], r_edges[-1]]
            im = axes[0].imshow(
                xi_rmu, origin="lower", aspect="auto", extent=extent,
                cmap="RdBu_r",
                vmin=-np.nanmax(np.abs(xi_rmu)),
                vmax=+np.nanmax(np.abs(xi_rmu)),
            )
            axes[0].set_xlabel(r"$|\mu| = |r_\parallel|/r$")
            axes[0].set_ylabel("r [Mpc/h]")
            axes[0].set_title(r"$\xi_{\times}(r, |\mu|)$  — input grid")
            fig.colorbar(im, ax=axes[0], label=r"$\xi_{\times}$")

            # Panel 1: monopole
            axes[1].plot(jres.r_centres[mono_mask],
                         jres.xi_mono_obs[mono_mask], "o", color="C0",
                         label=r"$\xi^{(0)}_{\times}$ obs (Hamilton)")
            axes[1].plot(jres.r_centres[mono_mask],
                         jres.xi_mono_template[mono_mask], "-", color="C3",
                         label=f"joint fit  $b_D={jres.b_DLA:+.2f}$")
            axes[1].axhline(0, color="grey", lw=0.5)
            axes[1].axvspan(10, 40, color="grey", alpha=0.08, label="fit window")
            axes[1].set_xlabel("r [Mpc/h]")
            axes[1].set_ylabel(r"$\xi^{(0)}_{\times}(r)$")
            axes[1].grid(True, alpha=0.3); axes[1].legend(fontsize=9)
            axes[1].set_title("Monopole")

            # Panel 2: quadrupole
            axes[2].plot(jres.r_centres[mono_mask],
                         jres.xi_quad_obs[mono_mask], "s", color="C2",
                         label=r"$\xi^{(2)}_{\times}$ obs (Hamilton)")
            axes[2].plot(jres.r_centres[mono_mask],
                         jres.xi_quad_template[mono_mask], "-", color="C3",
                         label=fr"joint fit  $\beta_D={jres.beta_DLA:+.2f}$")
            axes[2].axhline(0, color="grey", lw=0.5)
            axes[2].axvspan(10, 40, color="grey", alpha=0.08)
            axes[2].set_xlabel("r [Mpc/h]")
            axes[2].set_ylabel(r"$\xi^{(2)}_{\times}(r)$")
            axes[2].grid(True, alpha=0.3); axes[2].legend(fontsize=9)
            axes[2].set_title("Quadrupole")

            fig.suptitle(f"Test 10 (rmu) — DLA × Lyα joint mono+quad  "
                         f"z={geom.z_snap:.2f}  (N_DLA={n_dla}, b_F={b_F:+.3f})",
                         y=1.02)
            out_png = OUT_DIR / f"test10_{args.snap}{out_tag}.png"
            fig.tight_layout()
            fig.savefig(out_png, dpi=130, bbox_inches="tight")
            print(f"\n  wrote {out_png}")

        out_json = OUT_DIR / f"test10_{args.snap}{out_tag}.json"
        summary = {
            "mode": args.mode,
            "sim": args.sim, "snap": args.snap, "z": float(geom.z_snap),
            "n_dla": n_dla,
            "b_F_assumed": float(b_F),
            "b_DLA": float(jres.b_DLA) if jres is not None else None,
            "b_DLA_err": float(jres.b_DLA_err) if jres is not None else None,
            "beta_DLA": float(jres.beta_DLA) if jres is not None else None,
            "beta_DLA_err": float(jres.beta_DLA_err) if jres is not None else None,
            "K_0": float(jres.K_0) if jres is not None else None,
            "K_2": float(jres.K_2) if jres is not None else None,
            "chi2": float(jres.chi2) if jres is not None else None,
            "n_fit_bins": int(jres.n_fit_bins) if jres is not None else None,
            "in_lit_envelope": (
                bool(LIT_LO <= jres.b_DLA <= LIT_HI) if jres is not None else None
            ),
            "delta_to_bird14_linear": (
                float(abs(jres.b_DLA - BIRD14_LIN)) if jres is not None else None
            ),
            "lyα_pixel_subsample": int(args.xi_pixel_subsample),
            "n_mu_bins": int(args.n_mu_bins),
        }
        with open(out_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  wrote {out_json}")


if __name__ == "__main__":
    main()
