"""
Same as validate_snap017_refit.py but with fast_mode=True (NHI from tau_peak
inversion on the detected system core only — no ±2000 km/s wing window).

Purpose: isolate whether the post-fix over-detection is from forest
contamination in the Voigt-fit wing window.
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

from hcd_analysis.catalog import build_catalog
from hcd_analysis.cddf import measure_cddf
from hcd_analysis.io import read_header, pixel_dv_kms

SIM_NAME = "ns0.803Ap2.2e-09herei4.05heref2.67alphaq2.21hub0.735omegamh20.141hireionz7.17bhfeedback0.056"
HDF5 = Path("/nfs/turbo/umor-yueyingn/mfho/emu_full") / SIM_NAME / "output" / "SPECTRA_017" / "lya_forest_spectra_grid_480.hdf5"
OUT_CAT = ROOT / "tests" / "out" / "snap_017_fast_catalog.npz"
DIAG_DIR = ROOT / "figures" / "diagnostics"


def prochaska2014_logf(logN):
    from scipy.interpolate import PchipInterpolator
    _logN = np.array([12.0, 15.0, 17.0, 18.0, 20.0, 21.0, 21.5, 22.0])
    _logf = np.array([-9.72, -14.41, -17.94, -19.39, -21.28, -22.82, -23.95, -25.50])
    return PchipInterpolator(_logN, _logf)(np.clip(np.asarray(logN, dtype=float),
                                                     _logN[0], _logN[-1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-workers", type=int, default=8)
    args = parser.parse_args()

    OUT_CAT.parent.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    header = read_header(HDF5)
    dv_kms = pixel_dv_kms(header)
    print(f"snap 017  z={header.redshift}  dv={dv_kms:.3f} km/s")

    print(f"Building catalog with fast_mode=True (tau_peak inversion, no wing window)...")
    t0 = time.time()
    cat = build_catalog(
        hdf5_path=HDF5, sim_name=SIM_NAME, snap=17, z=header.redshift, dv_kms=dv_kms,
        tau_threshold=100.0, merge_dv_kms=100.0, min_pixels=2,
        b_init=30.0, b_bounds=(1.0, 300.0),
        tau_fit_cap=1.0e6, voigt_max_iter=200,
        batch_size=4096, n_skewers=None,
        fast_mode=True, n_workers=args.n_workers, min_log_nhi=17.2,
    )
    cat.save_npz(OUT_CAT)
    print(f"  built in {time.time()-t0:.1f}s → {OUT_CAT}")
    print(f"  summary: {cat.summary()}")

    # Load other two catalogs
    new_d = np.load(ROOT / "tests/out/snap_017_fixed_catalog.npz", allow_pickle=True)
    old_d = np.load("/scratch/cavestru_root/cavestru0/mfho/hcd_outputs/" + SIM_NAME + "/snap_017/catalog.npz", allow_pickle=True)
    fast_logN = cat.log_nhi_array()
    new_logN = np.log10(np.maximum(new_d["NHI"].astype(np.float64), 1.0))
    old_logN = np.log10(np.maximum(old_d["NHI"].astype(np.float64), 1.0))

    # Histograms
    bins = np.linspace(17.0, 23.0, 61)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].hist(old_logN,  bins=bins, histtype="step", lw=1.6,
               label=f"pre-fix Voigt (broken)   N={len(old_logN)}", color="C3")
    ax[0].hist(new_logN,  bins=bins, histtype="step", lw=1.6,
               label=f"post-fix Voigt (+wing)   N={len(new_logN)}", color="C0")
    ax[0].hist(fast_logN, bins=bins, histtype="step", lw=1.8,
               label=f"fast mode (tau-peak only) N={len(fast_logN)}", color="C2")
    for thr in (17.2, 19.0, 20.3):
        ax[0].axvline(thr, color="gray", lw=0.7, ls=":")
    ax[0].set_yscale("log"); ax[0].set_xlabel("log10(N_HI)"); ax[0].set_ylabel("count")
    ax[0].set_title("Snap 017 NHI distribution")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    # CDDF comparison
    cddf_bins = np.linspace(17.0, 23.0, 31)

    class _Cat:
        def __init__(self, d, z):
            self.z = z
            nh = d["NHI"].astype(np.float64)
            ln = np.log10(np.maximum(nh, 1.0))
            class _Abs:
                def __init__(self, n, l):
                    self.NHI=n; self.log_NHI=l
                    self.absorber_class = ("DLA" if l>=20.3 else "subDLA" if l>=19
                                            else "LLS" if l>=17.2 else "forest")
            self.absorbers = [_Abs(n, l) for n, l in zip(nh, ln)]

    def cddf_of_raw(d, z):
        return measure_cddf(_Cat(d, z), header, log_nhi_bins=cddf_bins)

    cddf_old  = cddf_of_raw(old_d,  header.redshift)
    cddf_new  = cddf_of_raw(new_d,  header.redshift)
    cddf_fast = measure_cddf(cat, header, log_nhi_bins=cddf_bins)
    truth = 10.0**prochaska2014_logf(cddf_fast["log_nhi_centres"])

    ax[1].step(cddf_old["log_nhi_centres"],  cddf_old["f_nhi"],  where="mid",
               lw=1.6, label="pre-fix", color="C3")
    ax[1].step(cddf_new["log_nhi_centres"],  cddf_new["f_nhi"],  where="mid",
               lw=1.6, label="post-fix Voigt", color="C0")
    ax[1].step(cddf_fast["log_nhi_centres"], cddf_fast["f_nhi"], where="mid",
               lw=1.8, label="fast (tau-peak)", color="C2")
    ax[1].plot(cddf_fast["log_nhi_centres"], truth, "k--", lw=1.2,
               label="Prochaska+2014")
    for thr in (17.2, 19.0, 20.3):
        ax[1].axvline(thr, color="gray", lw=0.7, ls=":")
    ax[1].set_yscale("log"); ax[1].set_xlabel("log10(N_HI)"); ax[1].set_ylabel("f(N_HI, X)")
    ax[1].set_title("CDDF vs Prochaska+2014")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    ax[1].set_ylim(1e-28, 1e-15)

    fig.tight_layout()
    outpath = DIAG_DIR / "snap017_three_way.png"
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    print(f"\nThree-way comparison: {outpath}")


if __name__ == "__main__":
    main()
