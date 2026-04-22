"""
Rogers-style subset-based HCD test at snap_017 (z=3) of ns0.803.

Three P1Ds, all over the same full 691200 sightlines, using the same
global mean-F normalisation (= mean F over all sightlines, unmasked):

  P_clean  = P1D over sightlines with NO catalog entry
  P_dirty  = P1D over sightlines with ≥1 catalog entry (the HCD-contaminated
             subset); no mask applied
  P_maskA  = P1D over the dirty subset with τ-space damping-wing mask
  P_maskB  = P1D over the dirty subset with pixel-range (legacy) mask

Report:
  template(k) = P_dirty  / P_clean     # Rogers+2018 HCD template
  leftover_A  = P_maskA  / P_clean     # perfect τ-space mask → 1
  leftover_B  = P_maskB  / P_clean     # perfect pixrange mask → 1

Also plot Rogers' analytic template (with alpha=0.1 across classes) for
magnitude sanity check.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hcd_analysis.catalog import AbsorberCatalog
from hcd_analysis.io import read_header, pixel_dv_kms, iter_tau_batches
from hcd_analysis.masking import (
    apply_tauspace_mask_to_batch,
    apply_mask_to_batch,
)
from hcd_analysis.p1d import P1DAccumulator, compute_p1d_ratios

SIM = "ns0.803Ap2.2e-09herei4.05heref2.67alphaq2.21hub0.735omegamh20.141hireionz7.17bhfeedback0.056"
HDF5 = Path("/nfs/turbo/umor-yueyingn/mfho/emu_full") / SIM / "output" / "SPECTRA_017" / "lya_forest_spectra_grid_480.hdf5"
CAT_PATH = ROOT / "tests" / "out" / "snap_017_fastfix_catalog.npz"
DIAG = ROOT / "figures" / "diagnostics"


def rogers_dla_correction(kf, z, alpha):
    """Copy of the user-supplied Rogers+2018 template formula."""
    z_0 = 2.0
    a_0 = np.array([2.2001, 1.5083, 1.1415, 0.8633])
    a_1 = np.array([0.0134, 0.0994, 0.0937, 0.2943])
    b_0 = np.array([36.449, 81.388, 162.95, 429.58])
    b_1 = np.array([-0.0674, -0.2287, 0.0126, -0.4964])
    a_z = a_0 * ((1 + z) / (1 + z_0)) ** a_1
    b_z = b_0 * ((1 + z) / (1 + z_0)) ** b_1
    factor = np.ones_like(kf)
    contribs = {}
    for i, name in enumerate(["LLS", "Sub-DLA", "Small-DLA", "Large-DLA"]):
        term = alpha[i] * ((1 + z) / (1 + z_0)) ** -3.55 \
             * ((a_z[i] * np.exp(b_z[i] * kf) - 1) ** -2)
        factor += term
        contribs[name] = 1.0 + term
    return factor, contribs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-skewers", type=int, default=None,
                    help="Subsample for speed (None = all 691200)")
    ap.add_argument("--batch-size", type=int, default=4096)
    args = ap.parse_args()

    DIAG.mkdir(parents=True, exist_ok=True)
    header = read_header(HDF5)
    dv_kms = pixel_dv_kms(header)
    catalog = AbsorberCatalog.load_npz(CAT_PATH)
    print(f"snap_017 z={header.redshift:.2f}  catalog: {catalog.summary()}")

    # Build the boolean "dirty" membership array indexed by global sightline idx.
    n_total = header.n_skewers
    if args.n_skewers is not None:
        n_total = min(n_total, args.n_skewers)
    dirty_flag = np.zeros(n_total, dtype=bool)
    for ab in catalog.absorbers:
        if ab.skewer_idx < n_total:
            dirty_flag[ab.skewer_idx] = True
    n_dirty = int(dirty_flag.sum())
    n_clean = n_total - n_dirty
    print(f"  n_total={n_total}  dirty={n_dirty} ({100*n_dirty/n_total:.1f}%)  "
          f"clean={n_clean} ({100*n_clean/n_total:.1f}%)")

    # Pass 1: compute <F> separately for the four samples we will FFT.
    #   <F>_clean     — normalisation for P_clean
    #   <F>_dirty     — normalisation for P_dirty (template numerator)
    #   <F>_mask_A    — normalisation for P_maskA  (same mask as applied below)
    #   <F>_mask_B    — normalisation for P_maskB
    # Also keep <F>_all for the τ_eff used by the mask fill.
    t0 = time.time()
    F_sum_all = F_n_all = 0
    F_sum_clean = F_n_clean = 0
    F_sum_dirty = F_n_dirty = 0
    F_sum_A = F_n_A = 0
    F_sum_B = F_n_B = 0

    # We need τ_eff to compute the masked fills, which we need to know before
    # computing <F> of masked subsets.  So do this in two passes:
    #   Pass 1a: global <F> (gives τ_eff; also gives <F>_clean and <F>_dirty).
    #   Pass 1b: <F> of masked-dirty subsets (requires τ_eff).
    for row_start, row_end, tau_batch in iter_tau_batches(HDF5, batch_size=args.batch_size, n_skewers=n_total):
        tau_batch = tau_batch.astype(np.float64)
        F = np.exp(-tau_batch)
        is_dirty = dirty_flag[row_start:row_end]
        F_sum_all += F.sum(); F_n_all += F.size
        if (~is_dirty).any():
            F_sum_clean += F[~is_dirty].sum(); F_n_clean += F[~is_dirty].size
        if is_dirty.any():
            F_sum_dirty += F[is_dirty].sum(); F_n_dirty += F[is_dirty].size
    mean_F_all = F_sum_all / F_n_all
    mean_F_clean = F_sum_clean / F_n_clean
    mean_F_dirty = F_sum_dirty / F_n_dirty
    tau_eff = -np.log(max(mean_F_all, 1e-30))
    print(f"  pass 1a (unmasked <F>): {time.time()-t0:.1f}s  "
          f"<F>_all={mean_F_all:.4f} <F>_clean={mean_F_clean:.4f} "
          f"<F>_dirty={mean_F_dirty:.4f}  τ_eff={tau_eff:.4f}")

    t0 = time.time()
    for row_start, row_end, tau_batch in iter_tau_batches(HDF5, batch_size=args.batch_size, n_skewers=n_total):
        tau_batch = tau_batch.astype(np.float64)
        is_dirty = dirty_flag[row_start:row_end]
        if not is_dirty.any():
            continue
        tau_A = apply_tauspace_mask_to_batch(
            tau_batch, row_start, catalog,
            mask_classes=["LLS", "subDLA", "DLA"],
            tau_eff=tau_eff,
        )
        tau_B = apply_mask_to_batch(
            tau_batch, row_start, catalog,
            mask_classes=["LLS", "subDLA", "DLA"],
            strategy="mean_flux",
        )
        FA = np.exp(-tau_A[is_dirty])
        FB = np.exp(-tau_B[is_dirty])
        F_sum_A += FA.sum(); F_n_A += FA.size
        F_sum_B += FB.sum(); F_n_B += FB.size
    mean_F_mask_A = F_sum_A / F_n_A
    mean_F_mask_B = F_sum_B / F_n_B
    print(f"  pass 1b (masked <F>): {time.time()-t0:.1f}s  "
          f"<F>_maskA={mean_F_mask_A:.4f} <F>_maskB={mean_F_mask_B:.4f}")

    # Pass 2: four P1D accumulators, one pass over the file.
    t0 = time.time()
    acc_clean = P1DAccumulator(header.nbins, dv_kms)
    acc_dirty = P1DAccumulator(header.nbins, dv_kms)
    acc_maskA = P1DAccumulator(header.nbins, dv_kms)  # τ-space
    acc_maskB = P1DAccumulator(header.nbins, dv_kms)  # pixrange

    for row_start, row_end, tau_batch in iter_tau_batches(
        HDF5, batch_size=args.batch_size, n_skewers=n_total
    ):
        tau_batch = tau_batch.astype(np.float64)
        mask_dirty_batch = dirty_flag[row_start:row_end]
        mask_clean_batch = ~mask_dirty_batch

        if mask_clean_batch.any():
            acc_clean.add_batch(tau_batch[mask_clean_batch],
                                 mean_F_global=mean_F_clean)

        if mask_dirty_batch.any():
            dirty_rows = tau_batch[mask_dirty_batch]
            acc_dirty.add_batch(dirty_rows, mean_F_global=mean_F_dirty)

            # τ-space mask on the dirty subset
            tau_masked_A = apply_tauspace_mask_to_batch(
                tau_batch, row_start, catalog,
                mask_classes=["LLS", "subDLA", "DLA"],
                tau_eff=tau_eff,
            )
            acc_maskA.add_batch(tau_masked_A[mask_dirty_batch],
                                 mean_F_global=mean_F_mask_A)

            # pixel-range mask
            tau_masked_B = apply_mask_to_batch(
                tau_batch, row_start, catalog,
                mask_classes=["LLS", "subDLA", "DLA"],
                strategy="mean_flux",
            )
            acc_maskB.add_batch(tau_masked_B[mask_dirty_batch],
                                 mean_F_global=mean_F_mask_B)
    print(f"  pass 2 (4 P1Ds): {time.time()-t0:.1f}s")

    k_clean, p_clean = acc_clean.result(None)
    _, p_dirty = acc_dirty.result(None)
    _, p_maskA = acc_maskA.result(None)
    _, p_maskB = acc_maskB.result(None)

    template = p_dirty / p_clean
    leftover_A = p_maskA / p_clean
    leftover_B = p_maskB / p_clean

    # Rogers analytic prediction at z=3 with uniform alpha=0.1
    alpha_prior = np.array([0.1, 0.1, 0.1, 0.1])
    rogers_total, rogers_contribs = rogers_dla_correction(
        k_clean, header.redshift, alpha_prior,
    )

    # Numeric table
    print(f"\n{'k [s/km]':>10}  {'template':>10}  {'leftover_A':>12}  "
          f"{'leftover_B':>12}  {'Rogers α=0.1':>14}")
    for k_tgt in [0.001, 0.002, 0.003, 0.005, 0.010, 0.020, 0.030]:
        j = int(np.argmin(np.abs(k_clean - k_tgt)))
        print(f"  {k_clean[j]:>8.4f}  {template[j]:>10.3f}  "
              f"{leftover_A[j]:>12.3f}  {leftover_B[j]:>12.3f}  "
              f"{rogers_total[j]:>14.3f}")

    # -- plot --
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))

    ax[0].plot(k_clean, template, lw=2.0, color="k",
                label="P_dirty / P_clean  (measured template)")
    ax[0].plot(k_clean, rogers_total, "--", lw=1.5, color="C3",
                label="Rogers+2018 total (α=0.1 all classes)")
    for name, curve in rogers_contribs.items():
        ax[0].plot(k_clean, curve, ":", lw=1.0, alpha=0.7, label=f"Rogers {name}")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("k [s/km]"); ax[0].set_ylabel("P1D ratio")
    ax[0].set_title("HCD template: contaminated vs clean subsets")
    ax[0].grid(alpha=0.3, which="both")
    ax[0].legend(fontsize=8)
    ax[0].set_xlim(1e-3, 5e-2)

    ax[1].plot(k_clean, leftover_A, lw=2.0, color="C0",
                label="τ-space mask:   P_masked / P_clean")
    ax[1].plot(k_clean, leftover_B, lw=1.5, color="C1",
                label="pixrange mask:  P_masked / P_clean")
    ax[1].plot(k_clean, template, ":", color="k", lw=1.0,
                label="unmasked dirty / clean  (template)")
    ax[1].axhline(1.0, color="gray", ls="--", alpha=0.5)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("k [s/km]"); ax[1].set_ylabel("P1D ratio")
    ax[1].set_title("Mask quality — closer to 1 is better")
    ax[1].grid(alpha=0.3, which="both")
    ax[1].legend(fontsize=8)
    ax[1].set_xlim(1e-3, 5e-2)
    ax[1].set_ylim(0.5, 2.5)

    fig.suptitle(f"{SIM[:30]}...  snap_017  z={header.redshift:.2f}  "
                  f"(n_skewers={n_total}, dirty={n_dirty})")
    fig.tight_layout()
    out = DIAG / "rogers_subset_test.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\nFigure: {out}")


if __name__ == "__main__":
    main()
