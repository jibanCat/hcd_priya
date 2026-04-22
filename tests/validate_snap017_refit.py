"""
Validate the voigt_utils units fix by rebuilding the catalog for
snap_017 of ns0.803Ap2.2e-09... (z=3) and comparing to the pre-fix
catalog + Prochaska+2014 CDDF.

Writes the new catalog to a separate path (does NOT overwrite production),
and saves before/after figures to figures/diagnostics/.

Usage:
    python3 tests/validate_snap017_refit.py [--n-workers 8]
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
OLD_CAT = Path("/scratch/cavestru_root/cavestru0/mfho/hcd_outputs") / SIM_NAME / "snap_017" / "catalog.npz"
NEW_CAT = ROOT / "tests" / "out" / "snap_017_fixed_catalog.npz"
DIAG_DIR = ROOT / "figures" / "diagnostics"


def prochaska2014_logf(logN):
    """Prochaska+2014 CDDF PchipInterpolator (from scripts/plot_intermediate.py)."""
    from scipy.interpolate import PchipInterpolator
    _logN = np.array([12.0, 15.0, 17.0, 18.0, 20.0, 21.0, 21.5, 22.0])
    _logf = np.array([-9.72, -14.41, -17.94, -19.39, -21.28, -22.82, -23.95, -25.50])
    return PchipInterpolator(_logN, _logf)(np.clip(np.asarray(logN, dtype=float),
                                                     _logN[0], _logN[-1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--n-skewers", type=int, default=None,
                        help="Optional row cap for debug (None = all 691200)")
    args = parser.parse_args()

    NEW_CAT.parent.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    header = read_header(HDF5)
    dv_kms = pixel_dv_kms(header)
    print(f"snap 017  z={header.redshift}  nbins={header.nbins}  dv={dv_kms:.3f} km/s  n_skewers={header.n_skewers}")

    # -- Rebuild catalog with fixed fitter --------------------------------
    print(f"Building catalog with fixed fitter (n_workers={args.n_workers})...")
    t0 = time.time()
    cat_new = build_catalog(
        hdf5_path=HDF5,
        sim_name=SIM_NAME,
        snap=17,
        z=header.redshift,
        dv_kms=dv_kms,
        tau_threshold=100.0,
        merge_dv_kms=100.0,
        min_pixels=2,
        b_init=30.0,
        b_bounds=(1.0, 300.0),
        tau_fit_cap=1.0e6,
        voigt_max_iter=200,
        batch_size=4096,
        n_skewers=args.n_skewers,
        fast_mode=False,
        n_workers=args.n_workers,
        min_log_nhi=17.2,
    )
    cat_new.save_npz(NEW_CAT)
    print(f"  built in {time.time()-t0:.1f}s → {NEW_CAT}")
    print(f"  summary: {cat_new.summary()}")

    # -- Load old catalog for comparison ----------------------------------
    old_d = np.load(OLD_CAT, allow_pickle=True)
    old_logN = np.log10(np.maximum(old_d["NHI"].astype(np.float64), 1.0))
    new_logN = cat_new.log_nhi_array()

    # -- Compare NHI histograms -------------------------------------------
    bins = np.linspace(17.0, 23.0, 61)
    centres = 0.5 * (bins[:-1] + bins[1:])

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    ax[0].hist(old_logN, bins=bins, histtype="step", lw=1.8, label=f"pre-fix  N={len(old_logN)}", color="C3")
    ax[0].hist(new_logN, bins=bins, histtype="step", lw=1.8, label=f"post-fix N={len(new_logN)}", color="C0")
    for thr, name in [(17.2, "LLS"), (19.0, "subDLA"), (20.3, "DLA")]:
        ax[0].axvline(thr, color="gray", lw=0.7, ls=":")
        ax[0].text(thr + 0.02, 5e4, name, color="gray", fontsize=8)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("log10(N_HI)")
    ax[0].set_ylabel("count")
    ax[0].set_title(f"{SIM_NAME[:30]}... snap_017, z={header.redshift}")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    # -- Compare CDDF against Prochaska+2014 ------------------------------
    cddf_bins = np.linspace(17.0, 23.0, 31)
    cddf_old = measure_cddf(
        # make a minimal catalog object with the old entries
        _DummyCat(OLD_cat=old_d, z=header.redshift),
        header,
        log_nhi_bins=cddf_bins,
    )
    cddf_new = measure_cddf(cat_new, header, log_nhi_bins=cddf_bins)

    # Prochaska truth
    truth = 10.0**prochaska2014_logf(cddf_new["log_nhi_centres"])

    ax[1].step(cddf_old["log_nhi_centres"], cddf_old["f_nhi"], where="mid",
               lw=1.8, label="pre-fix", color="C3")
    ax[1].step(cddf_new["log_nhi_centres"], cddf_new["f_nhi"], where="mid",
               lw=1.8, label="post-fix", color="C0")
    ax[1].plot(cddf_new["log_nhi_centres"], truth, "k--", lw=1.4,
               label="Prochaska+2014")
    for thr in (17.2, 19.0, 20.3):
        ax[1].axvline(thr, color="gray", lw=0.7, ls=":")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("log10(N_HI)")
    ax[1].set_ylabel("f(N_HI, X)")
    ax[1].set_title("CDDF (one snap, z=3)")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    ax[1].set_ylim(1e-28, 1e-15)

    fig.tight_layout()
    outpath = DIAG_DIR / "snap017_fix_comparison.png"
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    print(f"\nComparison plot: {outpath}")

    # -- Summary ----------------------------------------------------------
    print("\nClass counts:")
    print(f"             LLS      subDLA    DLA")
    old_lls = int(((old_logN>=17.2)&(old_logN<19)).sum())
    old_sub = int(((old_logN>=19)&(old_logN<20.3)).sum())
    old_dla = int((old_logN>=20.3).sum())
    new_lls = int(((new_logN>=17.2)&(new_logN<19)).sum())
    new_sub = int(((new_logN>=19)&(new_logN<20.3)).sum())
    new_dla = int((new_logN>=20.3).sum())
    print(f"  pre-fix   {old_lls:>7d}  {old_sub:>7d}   {old_dla:>5d}")
    print(f"  post-fix  {new_lls:>7d}  {new_sub:>7d}   {new_dla:>5d}")
    print(f"  DLA gain: ×{new_dla/max(old_dla,1):.1f}")


class _DummyCat:
    """Adapter so measure_cddf can consume the raw pre-fix catalog.npz."""
    def __init__(self, OLD_cat, z):
        self.absorbers = _DummyAbs(OLD_cat)
        self.z = z


class _DummyAbs(list):
    def __init__(self, d):
        super().__init__()
        NHI = d["NHI"].astype(np.float64)
        logN = np.log10(np.maximum(NHI, 1.0))
        for i in range(len(NHI)):
            self.append(_DummyAbsorber(NHI[i], logN[i]))


class _DummyAbsorber:
    def __init__(self, NHI, log_NHI):
        self.NHI = NHI
        self.log_NHI = log_NHI
        if log_NHI >= 20.3: self.absorber_class = "DLA"
        elif log_NHI >= 19.0: self.absorber_class = "subDLA"
        elif log_NHI >= 17.2: self.absorber_class = "LLS"
        else: self.absorber_class = "forest"


if __name__ == "__main__":
    main()
