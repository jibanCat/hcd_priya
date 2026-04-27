"""
Sweep tolerance: how do completeness and purity scale with tol_pixels?

Hypothesis: most T2 cases are real DLAs at peculiar-velocity offsets
between colden (real-space-binned) and τ (redshift-space).  A larger
position tolerance (~500 km/s = ~500 pixels at HiRes 1 km/s/pix)
should resolve them.

Risk: increasing tol could also make the matcher pair UNRELATED
absorbers, lowering effective purity. Sweep and report.
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hcd_analysis.catalog import find_systems_in_skewer
from hcd_analysis.voigt_utils import nhi_from_tau_fast
from hcd_analysis.dla_truth import (
    RecoveredDLA, find_truth_dlas_from_colden, match_dla_lists,
)

EMU = Path("/nfs/turbo/lsa-cavestru/mfho/priya/emu_full_hires")
files = sorted(EMU.glob("*/output/SPECTRA_*/rand_spectra_DLA.hdf5"))

# Tolerance sweep in km/s (converted to pixels per file via dv_kms)
TOL_KMS_LIST = [10, 30, 50, 100, 200, 300, 500, 800, 1500]


def process(h5):
    with h5py.File(h5, "r") as f:
        box = float(f["Header"].attrs["box"])
        h = float(f["Header"].attrs["hubble"])
        z = float(f["Header"].attrs["redshift"])
        Hz = float(f["Header"].attrs["Hz"])
        n_pix = int(f["Header"].attrs["nbins"])
        dx_pix_mpc = (box / 1000.0) / n_pix / h * (1.0 / (1.0 + z))
        dv_kms = dx_pix_mpc * Hz
        tau = f["tau/H/1/1215"][:]
        colden = f["colden/H/1"][:]
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
    rec_dla = [r for r in recovered if r.absorber_class == "DLA"]
    rec_loose = [r for r in recovered if r.absorber_class in ("LLS", "subDLA", "DLA")]
    return truth, rec_dla, rec_loose, dv_kms


# Aggregate across all 10 (sim, snap)
agg = {tk: {"n_truth": 0, "n_rec_dla": 0, "n_matched_loose": 0,
            "n_matched_strict": 0, "matched_strict_dlog": []}
       for tk in TOL_KMS_LIST}

for h5 in files:
    truth, rec_dla, rec_loose, dv_kms = process(h5)
    for tk in TOL_KMS_LIST:
        tol_px = max(int(round(tk / dv_kms)), 1)
        m_loose = match_dla_lists(truth, rec_loose, tol_pixels=tol_px)
        m_strict = match_dla_lists(truth, rec_dla, tol_pixels=tol_px)
        agg[tk]["n_truth"] += len(truth)
        agg[tk]["n_rec_dla"] += len(rec_dla)
        agg[tk]["n_matched_loose"] += len(m_loose.matched)
        agg[tk]["n_matched_strict"] += len(m_strict.matched)
        for mp in m_strict.matched:
            agg[tk]["matched_strict_dlog"].append(mp.delta_log_nhi)

print(f"{'tol_kms':>8} {'compl(loose)':>14} {'compl(strict)':>14} "
      f"{'purity(strict)':>15} {'mean_dlog':>11} {'σ_dlog':>9}")
print("-" * 80)
for tk in TOL_KMS_LIST:
    a = agg[tk]
    cl = a["n_matched_loose"] / a["n_truth"]
    cs = a["n_matched_strict"] / a["n_truth"]
    ps = a["n_matched_strict"] / a["n_rec_dla"]
    dlog = np.array(a["matched_strict_dlog"])
    print(f"{tk:>8} {cl:>14.4f} {cs:>14.4f} {ps:>15.4f} "
          f"{dlog.mean():>+11.4f} {dlog.std():>9.4f}")
