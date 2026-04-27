"""
Spot-check a single T2 case: plot tau and colden along the same
sightline, mark the truth-DLA peak and the recovered system span,
to see physically what is going on with the 154-pixel offset.
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hcd_analysis.catalog import find_systems_in_skewer
from hcd_analysis.voigt_utils import nhi_from_tau_fast
from hcd_analysis.dla_truth import (
    RecoveredDLA, find_truth_dlas_from_colden, match_dla_lists,
)

H5 = Path(
    "/nfs/turbo/lsa-cavestru/mfho/priya/emu_full_hires/"
    "ns0.914Ap1.32e-09herei3.85heref2.65alphaq1.57hub0.742"
    "omegamh20.141hireionz6.88bhfeedback0.04/output/SPECTRA_017/"
    "rand_spectra_DLA.hdf5"
)

with h5py.File(H5, "r") as f:
    box = float(f["Header"].attrs["box"])
    h = float(f["Header"].attrs["hubble"])
    z = float(f["Header"].attrs["redshift"])
    Hz = float(f["Header"].attrs["Hz"])
    n_pix = int(f["Header"].attrs["nbins"])
    dx_pix_mpc = (box / 1000.0) / n_pix / h * (1.0 / (1.0 + z))
    dv_kms = dx_pix_mpc * Hz
    tau = f["tau/H/1/1215"][:]
    colden = f["colden/H/1"][:]

print(f"dv_kms = {dv_kms:.2f}")
print(f"n_pix  = {n_pix}")
print(f"Box LOS extent: {n_pix * dv_kms:.0f} km/s = {n_pix * dx_pix_mpc * h:.1f} Mpc/h")

truth = find_truth_dlas_from_colden(colden, dla_threshold=2e20, pixel_floor=1e17,
                                    merge_gap_pixels=0, min_pixels=1)
recovered = []
merge_gap_pix = max(int(round(100.0 / dv_kms)), 1)
for sk in range(tau.shape[0]):
    sys_ = find_systems_in_skewer(tau[sk], tau_threshold=100.0,
                                  merge_gap_pixels=merge_gap_pix, min_pixels=2)
    for s, e in sys_:
        nhi = nhi_from_tau_fast(tau[sk, s:e+1], dv_kms)
        if not np.isfinite(nhi) or nhi < 10**17.2:
            continue
        log_nhi = float(np.log10(nhi))
        cls = ("DLA" if log_nhi >= 20.3 else
               "subDLA" if log_nhi >= 19.0 else "LLS")
        recovered.append(RecoveredDLA(sk, s, e, float(nhi), log_nhi, cls))

rec_loose = [r for r in recovered if r.absorber_class in ("LLS", "subDLA", "DLA")]
tol = max(int(round(100.0 / dv_kms)), 5)
res = match_dla_lists(truth, rec_loose, tol_pixels=tol)

# Pick a T2 case: unmatched truth, has nearest R but T.pix_peak outside (R ± tol)
by_row = {}
for r in rec_loose:
    by_row.setdefault(r.skewer_idx, []).append(r)

t2_cases = []
for t in res.unmatched_truth:
    cands = by_row.get(t.skewer_idx, [])
    if not cands:
        continue
    nearest = None
    nearest_d = 1e9
    for r in cands:
        if t.pix_peak < r.pix_start: d = r.pix_start - t.pix_peak
        elif t.pix_peak > r.pix_end: d = t.pix_peak - r.pix_end
        else: d = 0
        if d < nearest_d:
            nearest_d = d
            nearest = r
    if nearest_d > tol:           # genuine T2 (not collision)
        t2_cases.append((t, nearest, nearest_d))

# Sort by edge distance, take a representative spread
t2_cases.sort(key=lambda x: x[2])
samples = []
for target_d in (50, 100, 200, 500, 1000):
    closest = min(t2_cases, key=lambda x: abs(x[2] - target_d))
    samples.append(closest)

print(f"\nTotal T2 cases on this snap: {len(t2_cases)}")
print("Spot-check samples (target edge-distance, actual):")
for i, (t, r, d) in enumerate(samples):
    print(f"  sample {i}: target≈px, got d={d}px, sk={t.skewer_idx}, "
          f"T.peak={t.pix_peak} (NHI={t.NHI_truth:.2e}), "
          f"R.span=[{r.pix_start},{r.pix_end}] (NHI={r.NHI_recovered:.2e})")

fig, axes = plt.subplots(len(samples), 2, figsize=(13, 3 * len(samples)))
for i, (t, r, d) in enumerate(samples):
    sk = t.skewer_idx
    pix = np.arange(n_pix)
    # Window around the action: show ±400 pixels around midpoint of (T.peak, R.span)
    mid = (t.pix_peak + (r.pix_start + r.pix_end) // 2) // 2
    lo = max(0, mid - 600)
    hi = min(n_pix, mid + 600)
    axL, axR = axes[i]
    axL.plot(pix[lo:hi], np.log10(np.maximum(colden[sk, lo:hi], 1e10)), color="C0")
    axL.axvspan(t.pix_start, t.pix_end + 1, color="lime", alpha=0.4, label="truth span")
    axL.axvline(t.pix_peak, color="green", lw=1.5, label=f"truth peak (NHI {t.NHI_truth:.1e})")
    axL.axhline(np.log10(1e17), color="grey", ls="--", lw=0.5, label="pixel_floor=1e17")
    axL.set_ylabel("log10 colden / pixel")
    axL.set_title(f"skewer {sk}, edge-dist={d}px (~{d*dv_kms:.0f} km/s)")
    axL.legend(fontsize=8, loc="best")
    axL.grid(True, alpha=0.3)

    axR.semilogy(pix[lo:hi], np.maximum(tau[sk, lo:hi], 1e-3), color="C3")
    axR.axvspan(r.pix_start, r.pix_end + 1, color="orange", alpha=0.4,
                label=f"recovered span (NHI {r.NHI_recovered:.1e})")
    axR.axvline(t.pix_peak, color="green", lw=1.5, label="truth peak")
    axR.axhline(100, color="grey", ls="--", lw=0.5, label="τ_threshold=100")
    axR.set_ylabel("τ")
    axR.set_xlabel("pixel index")
    axR.legend(fontsize=8, loc="best")
    axR.grid(True, alpha=0.3, which="both")

OUT = ROOT / "figures" / "analysis" / "05_truth_validation" / "t2_spotcheck.png"
fig.suptitle(f"T2 spot-check: 5 unmatched truth-DLAs at varying T-peak ↔ R-edge distances",
             fontsize=11)
fig.tight_layout()
fig.savefig(OUT, dpi=130, bbox_inches="tight")
print(f"\nWrote: {OUT}")
