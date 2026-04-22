"""
Compare four P1D variants on the FULL (all-sightlines) sample at snap 017:

  1. P_all_unmasked      — no mask (this should match PRIYA's stored P1D shape)
  2. P_all_priya         — exact PRIYA recipe from arXiv:2306.05471 §3.3:
                           only sightlines with max(tau) > 1e6 get a mask,
                           the mask is the ONE contiguous region around argmax
                           where tau > 0.25 + tau_eff, fill with tau_eff.
  3. P_all_tauspace      — my τ-space per-class mask (all HCD classes)
  4. P_all_pixrange      — legacy pix-range mask (all HCD classes)

Everything normalised by <F>_all (so the variants can be compared directly).
The expectation is that (1) and (2) should be monotonic-decreasing like
PRIYA's production data; (3) and (4) should show whatever artefacts my
more aggressive masks introduce.
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
    apply_tauspace_mask_to_batch, apply_mask_to_batch,
    iter_priya_masked_batches,
)
from hcd_analysis.p1d import P1DAccumulator

SIM = "ns0.803Ap2.2e-09herei4.05heref2.67alphaq2.21hub0.735omegamh20.141hireionz7.17bhfeedback0.056"
HDF5 = Path("/nfs/turbo/umor-yueyingn/mfho/emu_full") / SIM / "output" / "SPECTRA_017" / "lya_forest_spectra_grid_480.hdf5"
CAT = ROOT / "tests" / "out" / "snap_017_fastfix_catalog.npz"
DIAG = ROOT / "figures" / "diagnostics"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-skewers", type=int, default=100000)
    args = ap.parse_args()

    header = read_header(HDF5)
    dv_kms = pixel_dv_kms(header)
    cat = AbsorberCatalog.load_npz(CAT)
    print(f"snap_017 z={header.redshift:.2f}  dv={dv_kms:.2f} km/s  "
          f"catalog: {cat.summary()}")

    n_total = min(header.n_skewers, args.n_skewers)

    # Pass 1: <F>_all (drives τ_eff for PRIYA fill)
    F_sum = F_n = 0
    for _, _, tau in iter_tau_batches(HDF5, batch_size=4096, n_skewers=n_total):
        F = np.exp(-tau.astype(np.float64))
        F_sum += F.sum(); F_n += F.size
    mean_F_all = F_sum / F_n
    tau_eff = -np.log(max(mean_F_all, 1e-30))
    print(f"  <F>_all = {mean_F_all:.4f}  τ_eff = {tau_eff:.4f}")

    # Pass 2: four P1D accumulators
    acc = {name: P1DAccumulator(header.nbins, dv_kms)
           for name in ["unmasked", "priya", "tauspace", "pixrange"]}

    t0 = time.time()
    for row_start, row_end, tau_batch in iter_tau_batches(
        HDF5, batch_size=4096, n_skewers=n_total
    ):
        tau_batch = tau_batch.astype(np.float64)
        acc["unmasked"].add_batch(tau_batch, mean_F_global=mean_F_all)

        # τ-space and pixrange masks use the catalog
        tau_ts = apply_tauspace_mask_to_batch(
            tau_batch, row_start, cat,
            mask_classes=["LLS", "subDLA", "DLA"], tau_eff=tau_eff,
        )
        tau_px = apply_mask_to_batch(
            tau_batch, row_start, cat,
            mask_classes=["LLS", "subDLA", "DLA"], strategy="mean_flux",
        )
        acc["tauspace"].add_batch(tau_ts, mean_F_global=mean_F_all)
        acc["pixrange"].add_batch(tau_px, mean_F_global=mean_F_all)

    # PRIYA mask goes through its own iterator (no catalog dependence)
    for row_start, row_end, tau_priya in iter_priya_masked_batches(
        HDF5, tau_eff, batch_size=4096, n_skewers=n_total,
        tau_dla_detect=1e6, tau_mask_scale=0.25,
    ):
        acc["priya"].add_batch(tau_priya, mean_F_global=mean_F_all)
    print(f"  pass 2: {time.time()-t0:.1f}s")

    # Native k grid
    k, _ = acc["unmasked"].raw_power()
    P = {n: a.raw_power()[1] for n, a in acc.items()}

    # Table
    print(f"\n{'k [s/km]':>10}  {'P_unm':>10}  {'P_priya':>10}  "
          f"{'P_tauspace':>12}  {'P_pixrange':>12}  |  "
          f"{'priya/unm':>10}  {'ts/unm':>8}  {'px/unm':>8}")
    for k_tgt in [0.001, 0.002, 0.005, 0.010, 0.020, 0.030, 0.040, 0.050]:
        j = int(np.argmin(np.abs(k - k_tgt)))
        r_p = P["priya"][j] / P["unmasked"][j]
        r_ts = P["tauspace"][j] / P["unmasked"][j]
        r_px = P["pixrange"][j] / P["unmasked"][j]
        print(f"  {k[j]:>8.4f}  {P['unmasked'][j]:>10.4g}  {P['priya'][j]:>10.4g}  "
              f"{P['tauspace'][j]:>12.4g}  {P['pixrange'][j]:>12.4g}  |  "
              f"{r_p:>10.4f}  {r_ts:>8.4f}  {r_px:>8.4f}")

    # -- plot --
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))

    colors = {"unmasked": "k", "priya": "C3", "tauspace": "C0", "pixrange": "C1"}
    labels = {"unmasked": "no mask (should match PRIYA shape)",
              "priya": "PRIYA exact recipe",
              "tauspace": "my τ-space (all HCD classes)",
              "pixrange": "my pixrange (all HCD classes)"}
    for name in ["unmasked", "priya", "tauspace", "pixrange"]:
        ax[0].loglog(k[1:], P[name][1:], lw=1.5 if name!="priya" else 2.0,
                      color=colors[name], label=labels[name],
                      ls="-" if name in ("unmasked","priya") else "--")
    for kv in (0.02, 0.05): ax[0].axvline(kv, color="gray", ls=":", alpha=0.4)
    ax[0].set_xlabel("k [s/km]"); ax[0].set_ylabel("P1D(k)")
    ax[0].set_title("Absolute P1D on full sample (100k skewers)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, which="both")
    ax[0].set_xlim(5e-4, 5e-2)

    for name in ["priya", "tauspace", "pixrange"]:
        ax[1].semilogx(k[1:], P[name][1:] / P["unmasked"][1:], lw=1.8,
                        color=colors[name], label=labels[name])
    ax[1].axhline(1.0, color="gray", ls="--", alpha=0.5)
    for kv in (0.02, 0.05): ax[1].axvline(kv, color="gray", ls=":", alpha=0.4)
    ax[1].set_xlabel("k [s/km]"); ax[1].set_ylabel("P_masked / P_unmasked")
    ax[1].set_title("Mask effect on full sample")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, which="both")
    ax[1].set_xlim(5e-4, 5e-2); ax[1].set_ylim(0.8, 1.15)

    fig.suptitle(f"snap_017 z=3  (all {n_total} sightlines, normalised by <F>_all)")
    fig.tight_layout()
    outpath = DIAG / "priya_mask_comparison.png"
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    print(f"\nFigure: {outpath}")


if __name__ == "__main__":
    main()
