"""
End-to-end validation of Phase B (τ-space per-class damping-wing mask).

Uses:
  - fast-mode catalog from snap_017 (already built in tests/out/)
  - new compute_all_p1d_variants with mask_scheme='tauspace' vs 'pixrange'

Expected outcome: tauspace ratios drop well below 1 across the emulator k range
(Rogers+2018 template-shape), while pixrange ratios remain ≈1 (confirming the
pix-range mask is too narrow).
"""
from __future__ import annotations

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
from hcd_analysis.p1d import compute_all_p1d_variants, compute_p1d_ratios
from hcd_analysis.io import read_header, pixel_dv_kms

SIM = "ns0.803Ap2.2e-09herei4.05heref2.67alphaq2.21hub0.735omegamh20.141hireionz7.17bhfeedback0.056"
HDF5 = Path("/nfs/turbo/umor-yueyingn/mfho/emu_full") / SIM / "output" / "SPECTRA_017" / "lya_forest_spectra_grid_480.hdf5"
CAT_PATH = ROOT / "tests" / "out" / "snap_017_fastfix_catalog.npz"
DIAG = ROOT / "figures" / "diagnostics"


def main():
    DIAG.mkdir(parents=True, exist_ok=True)
    header = read_header(HDF5)
    dv_kms = pixel_dv_kms(header)
    catalog = AbsorberCatalog.load_npz(CAT_PATH)
    print(f"snap_017 z={header.redshift:.2f}  catalog: {catalog.summary()}")

    # Use a modest n_skewers subsample for speed — full file takes a while
    # per variant (two passes of 691200 skewers).  Ratios converge quickly.
    # 20k is enough for a visible signal; scale up to 200k for production
    # publication quality.
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-skewers", type=int, default=20000)
    args = ap.parse_args()
    n_sub = args.n_skewers
    print(f"(using n_skewers={n_sub})")

    print("\nCompute P1D variants with tauspace mask...")
    t0 = time.time()
    variants_ts = compute_all_p1d_variants(
        HDF5, header.nbins, dv_kms,
        catalog=catalog,
        variants=["all", "no_LLS", "no_subDLA", "no_DLA", "no_HCD"],
        batch_size=4096,
        n_skewers=n_sub,
        fill_strategy="mean_flux",
        mask_scheme="tauspace",
    )
    print(f"  took {time.time()-t0:.0f}s")
    ratios_ts = compute_p1d_ratios(variants_ts)

    print("\nCompute P1D variants with pixrange mask (legacy)...")
    t0 = time.time()
    variants_px = compute_all_p1d_variants(
        HDF5, header.nbins, dv_kms,
        catalog=catalog,
        variants=["all", "no_LLS", "no_subDLA", "no_DLA", "no_HCD"],
        batch_size=4096,
        n_skewers=n_sub,
        fill_strategy="mean_flux",
        mask_scheme="pixrange",
    )
    print(f"  took {time.time()-t0:.0f}s")
    ratios_px = compute_p1d_ratios(variants_px)

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    k = ratios_ts["k"]
    for name, key, color in [
        ("exclude DLA",   "ratio_noDLA_all",    "C3"),
        ("exclude subDLA","ratio_nosubDLA_all", "C1"),
        ("exclude LLS",   "ratio_noLLS_all",    "C2"),
        ("exclude all HCD","ratio_noHCD_all",   "C0"),
    ]:
        if key in ratios_ts:
            ax[0].plot(k, ratios_ts[key], lw=1.8, label=name, color=color)
        if key in ratios_px:
            ax[1].plot(k, ratios_px[key], lw=1.8, label=name, color=color)

    for a, title in zip(
        ax,
        ["τ-space damping-wing mask (Phase B)", "pixel-range core-only mask (legacy)"],
    ):
        a.set_xscale("log")
        a.axhline(1.0, color="k", ls="--", alpha=0.5)
        a.set_xlabel("k  [s/km]")
        a.set_title(title)
        a.legend()
        a.grid(alpha=0.3, which="both")
        a.set_ylim(0.6, 1.1)
    ax[0].set_ylabel("P1D(excl X) / P1D(all)")

    fig.suptitle(f"ns0.803 snap_017  z=3  (n_skewers={n_sub}, fast-mode catalog)")
    fig.tight_layout()
    outpath = DIAG / "p1d_mask_ab_test.png"
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    print(f"\nFigure: {outpath}")

    # Numeric summary at a few k
    print("\nRatios at key k (τ-space):")
    for kk in (0.002, 0.005, 0.01, 0.02):
        line = f"  k={kk:.3f}  "
        for name, key in [("no_DLA", "ratio_noDLA_all"), ("no_subDLA","ratio_nosubDLA_all"),
                           ("no_LLS","ratio_noLLS_all"), ("no_HCD","ratio_noHCD_all")]:
            if key in ratios_ts:
                val = float(np.interp(kk, k, ratios_ts[key]))
                line += f"{name}={val:.3f}  "
        print(line)

    print("\nRatios at key k (pixel-range legacy):")
    for kk in (0.002, 0.005, 0.01, 0.02):
        line = f"  k={kk:.3f}  "
        for name, key in [("no_DLA", "ratio_noDLA_all"), ("no_subDLA","ratio_nosubDLA_all"),
                           ("no_LLS","ratio_noLLS_all"), ("no_HCD","ratio_noHCD_all")]:
            if key in ratios_px:
                val = float(np.interp(kk, k, ratios_px[key]))
                line += f"{name}={val:.3f}  "
        print(line)


if __name__ == "__main__":
    main()
