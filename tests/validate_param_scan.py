"""
CDDF scatter across 6 PRIYA parameter-extreme sims at snap 017 (z=3).

Purpose: test if the DLA-end CDDF excess we saw in ns0.803 is universal
across the PRIYA parameter space, or driven by the specific parameter point
(which happens to be the extreme low-ns corner — expected to over-produce DLAs).

Sims chosen (extremes of the 60-sim grid):
  ns0.803... ← min ns (our prior test case)
  ns1.04...  ← max ns
  ns0.901Ap1.22e-09... ← min Ap (lowest power amplitude)
  ns0.816Ap2.58e-09... ← max Ap (highest power amplitude)
  ns0.849...bhfeedback0.0307 ← min black-hole feedback
  ns0.966...bhfeedback0.0693 ← max black-hole feedback
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

DATA_ROOT = Path("/nfs/turbo/umor-yueyingn/mfho/emu_full")
SIMS = [
    ("min ns", "ns0.803Ap2.2e-09herei4.05heref2.67alphaq2.21hub0.735omegamh20.141hireionz7.17bhfeedback0.056"),
    ("max ns", "ns1.04Ap1.73e-09herei4.21heref2.49alphaq2.52hub0.671omegamh20.145hireionz7.81bhfeedback0.0683"),
    ("min Ap", "ns0.901Ap1.22e-09herei3.93heref2.87alphaq1.68hub0.712omegamh20.146hireionz6.97bhfeedback0.068"),
    ("max Ap", "ns0.816Ap2.58e-09herei3.73heref2.69alphaq2.16hub0.732omegamh20.143hireionz7.97bhfeedback0.0587"),
    ("min bhf", "ns0.849Ap1.64e-09herei3.77heref2.83alphaq2.4hub0.748omegamh20.143hireionz7.33bhfeedback0.0307"),
    ("max bhf", "ns0.966Ap1.46e-09herei3.55heref2.73alphaq1.65hub0.702omegamh20.142hireionz7.62bhfeedback0.0693"),
]


def prochaska2014(logN):
    from scipy.interpolate import PchipInterpolator
    _logN = np.array([12.0, 15.0, 17.0, 18.0, 20.0, 21.0, 21.5, 22.0])
    _logf = np.array([-9.72, -14.41, -17.94, -19.39, -21.28, -22.82, -23.95, -25.50])
    return PchipInterpolator(_logN, _logf)(np.clip(np.asarray(logN, dtype=float),
                                                     _logN[0], _logN[-1]))


def main():
    out = ROOT / "tests" / "out" / "param_scan"
    out.mkdir(parents=True, exist_ok=True)
    diag = ROOT / "figures" / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)

    results = []
    for label, sim in SIMS:
        path = DATA_ROOT / sim / "output" / "SPECTRA_017" / "lya_forest_spectra_grid_480.hdf5"
        if not path.exists():
            print(f"  {label}: NO SPECTRA_017  ({sim[:40]}...)")
            continue
        header = read_header(path)
        dv = pixel_dv_kms(header)
        t0 = time.time()
        cat = build_catalog(
            hdf5_path=path, sim_name=sim, snap=17, z=header.redshift, dv_kms=dv,
            tau_threshold=100.0, merge_dv_kms=100.0, min_pixels=2,
            b_init=30.0, b_bounds=(1.0, 300.0),
            tau_fit_cap=1.0e6, voigt_max_iter=200,
            batch_size=4096, n_skewers=None,
            fast_mode=True, n_workers=4, min_log_nhi=17.2,
        )
        cat.save_npz(out / f"cat_{label.replace(' ','_')}.npz")
        cddf = measure_cddf(cat, header, log_nhi_bins=np.linspace(17.0, 23.0, 31))
        logN = cat.log_nhi_array()
        n_lls = int(((logN>=17.2)&(logN<19)).sum())
        n_sub = int(((logN>=19)&(logN<20.3)).sum())
        n_dla = int((logN>=20.3).sum())
        print(f"  {label:>8s} ({sim[:16]}): built in {time.time()-t0:.1f}s  "
              f"LLS={n_lls:>7d} subDLA={n_sub:>6d} DLA={n_dla:>6d}")
        results.append((label, sim, header, cddf, (n_lls, n_sub, n_dla)))

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    for (label, sim, header, cddf, counts), c in zip(results, colors):
        ax[0].step(cddf["log_nhi_centres"], cddf["f_nhi"], where="mid",
                   lw=1.5, label=f"{label}  (DLA={counts[2]})", color=c)
    truth = 10.0**prochaska2014(cddf["log_nhi_centres"])
    ax[0].plot(cddf["log_nhi_centres"], truth, "k--", lw=1.6, label="Prochaska+2014")
    for thr in (17.2, 19.0, 20.3):
        ax[0].axvline(thr, color="gray", lw=0.6, ls=":")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("log10(N_HI)")
    ax[0].set_ylabel("f(N_HI, X)")
    ax[0].set_title(f"CDDF at snap_017 (z={results[0][2].redshift:.2f}) — 6 PRIYA parameter extremes")
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    ax[0].set_ylim(1e-28, 1e-15)

    # Panel 2: ratio to Prochaska
    for (label, sim, header, cddf, counts), c in zip(results, colors):
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = cddf["f_nhi"] / truth
        ok = np.isfinite(ratio) & (ratio > 0)
        ax[1].step(cddf["log_nhi_centres"][ok], ratio[ok], where="mid",
                   lw=1.5, label=label, color=c)
    ax[1].axhline(1.0, color="k", ls="--", alpha=0.6)
    ax[1].axhline(10.0, color="gray", ls=":", alpha=0.6)
    ax[1].axhline(0.1, color="gray", ls=":", alpha=0.6)
    for thr in (17.2, 19.0, 20.3):
        ax[1].axvline(thr, color="gray", lw=0.6, ls=":")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("log10(N_HI)")
    ax[1].set_ylabel("f_sim / f_Prochaska")
    ax[1].set_title("Ratio to Prochaska+2014")
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=8)
    ax[1].set_ylim(0.03, 100)

    fig.tight_layout()
    outpath = diag / "cddf_param_scan_z3.png"
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    print(f"\nFigure: {outpath}")

    # Text summary
    print("\nSummary (snap_017, z≈3):")
    print(f"{'label':>10s}  {'LLS':>8s}  {'subDLA':>8s}  {'DLA':>8s}")
    for label, sim, header, cddf, counts in results:
        print(f"  {label:>8s}  {counts[0]:>8d}  {counts[1]:>8d}  {counts[2]:>8d}")


if __name__ == "__main__":
    main()
