"""
Rebuild snap_017 (z=3) and snap_010 (z=4.2) catalogs in fast mode with
the final voigt_utils prefactor (sum-rule exact), then compare both
CDDFs to Prochaska+2014.
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

from hcd_analysis.catalog import build_catalog
from hcd_analysis.cddf import measure_cddf
from hcd_analysis.io import read_header, pixel_dv_kms

SIM = "ns0.803Ap2.2e-09herei4.05heref2.67alphaq2.21hub0.735omegamh20.141hireionz7.17bhfeedback0.056"
SNAP_PATHS = {
    17: Path("/nfs/turbo/umor-yueyingn/mfho/emu_full") / SIM / "output" / "SPECTRA_017" / "lya_forest_spectra_grid_480.hdf5",
    10: Path("/nfs/turbo/umor-yueyingn/mfho/emu_full") / SIM / "output" / "SPECTRA_010" / "lya_forest_spectra_grid_480.hdf5",
}


def prochaska2014(logN):
    from scipy.interpolate import PchipInterpolator
    _logN = np.array([12.0, 15.0, 17.0, 18.0, 20.0, 21.0, 21.5, 22.0])
    _logf = np.array([-9.72, -14.41, -17.94, -19.39, -21.28, -22.82, -23.95, -25.50])
    return PchipInterpolator(_logN, _logf)(np.clip(np.asarray(logN, dtype=float),
                                                     _logN[0], _logN[-1]))


def main():
    out = ROOT / "tests" / "out"
    out.mkdir(parents=True, exist_ok=True)
    diag = ROOT / "figures" / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)

    results = {}
    for snap, path in SNAP_PATHS.items():
        header = read_header(path)
        dv = pixel_dv_kms(header)
        print(f"\nsnap_{snap:03d}  z={header.redshift:.2f}  dv={dv:.2f} km/s  "
              f"nbins={header.nbins}  n_skewers={header.n_skewers}")

        t0 = time.time()
        cat = build_catalog(
            hdf5_path=path, sim_name=SIM, snap=snap, z=header.redshift, dv_kms=dv,
            tau_threshold=100.0, merge_dv_kms=100.0, min_pixels=2,
            b_init=30.0, b_bounds=(1.0, 300.0),
            tau_fit_cap=1.0e6, voigt_max_iter=200,
            batch_size=4096, n_skewers=None,
            fast_mode=True, n_workers=8, min_log_nhi=17.2,
        )
        cat.save_npz(out / f"snap_{snap:03d}_fastfix_catalog.npz")
        print(f"  built in {time.time()-t0:.1f}s  -> {cat.summary()}")
        cddf = measure_cddf(cat, header, log_nhi_bins=np.linspace(17.0, 23.0, 31))
        results[snap] = (header, cat, cddf)

    # Plot both CDDFs vs Prochaska
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for i, (snap, (header, cat, cddf)) in enumerate(sorted(results.items())):
        ax[i].step(cddf["log_nhi_centres"], cddf["f_nhi"], where="mid",
                   lw=1.8, label="fast mode (fixed)", color="C2")
        truth = 10.0**prochaska2014(cddf["log_nhi_centres"])
        ax[i].plot(cddf["log_nhi_centres"], truth, "k--", lw=1.3, label="Prochaska+2014")
        for thr in (17.2, 19.0, 20.3):
            ax[i].axvline(thr, color="gray", lw=0.6, ls=":")
        ax[i].set_yscale("log")
        ax[i].set_xlabel("log10(N_HI)")
        ax[i].set_ylabel("f(N_HI, X)")
        ax[i].set_title(f"snap_{snap:03d}   z = {header.redshift:.2f}")
        ax[i].grid(alpha=0.3)
        ax[i].legend()
        ax[i].set_ylim(1e-28, 1e-15)

        # Print class counts
        logN = cat.log_nhi_array()
        n_lls = int(((logN>=17.2)&(logN<19)).sum())
        n_sub = int(((logN>=19)&(logN<20.3)).sum())
        n_dla = int((logN>=20.3).sum())
        print(f"  snap_{snap:03d} classes: LLS={n_lls}  subDLA={n_sub}  DLA={n_dla}")

    fig.suptitle("Fast-mode CDDF vs Prochaska+2014 (after sqrt(pi) + 1e5 fixes)")
    fig.tight_layout()
    outpath = diag / "cddf_after_full_fix.png"
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    print(f"\nFigure: {outpath}")


if __name__ == "__main__":
    main()
