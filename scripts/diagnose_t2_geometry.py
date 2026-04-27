"""
Inspect the T2 unmatched-truth subgroup: how far is the truth peak from
the nearest recovered span's edge, and is the truth-DLA span actually
overlapping the recovered span (just with a far-away peak)?

If most T2 cases have OVERLAPPING spans, an overlap-based matcher would
recover them; if they have gaps of hundreds of pixels, then we have a
genuine pipeline mismatch (multi-component truth-DLA being split, or
truth-finder merging two close τ-systems that the τ-finder splits).

Run::
    python3 scripts/diagnose_t2_geometry.py
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

# How far is the truth peak from R's span?  For each T2 case, also
# compute span overlap and integrated colden over the recovered span.
edge_distances = []
overlap_pixels = []          # ≥0 if T.span overlaps R.span; 0 if just-touching
truth_widths = []
recovered_widths = []
truth_NHI_T2 = []
recovered_NHI_T2 = []

# also track: for unmatched-recovered, did integrated colden exceed
# threshold but not have a truth peak in span — what was the colden's
# peak location vs R's centre?
r2_peak_distance = []

for h5 in files:
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

    truth = find_truth_dlas_from_colden(
        colden, dla_threshold=2.0e20, pixel_floor=1.0e17,
        merge_gap_pixels=0, min_pixels=1,
    )
    recovered = []
    merge_gap_pix = max(int(round(100.0 / dv_kms)), 1)
    for sk in range(tau.shape[0]):
        systems = find_systems_in_skewer(
            tau[sk], tau_threshold=100.0,
            merge_gap_pixels=merge_gap_pix, min_pixels=2,
        )
        for s, e in systems:
            nhi = nhi_from_tau_fast(tau[sk, s : e + 1], dv_kms)
            if not np.isfinite(nhi) or nhi < 10 ** 17.2:
                continue
            log_nhi = float(np.log10(nhi))
            if log_nhi < 17.2:
                continue
            cls = ("DLA" if log_nhi >= 20.3 else
                   "subDLA" if log_nhi >= 19.0 else "LLS")
            recovered.append(RecoveredDLA(sk, s, e, float(nhi), log_nhi, cls))

    rec_loose = [r for r in recovered if r.absorber_class in ("LLS", "subDLA", "DLA")]
    tol = max(int(round(100.0 / dv_kms)), 5)
    res = match_dla_lists(truth, rec_loose, tol_pixels=tol)

    # group recovered by skewer
    by_row = {}
    for r in rec_loose: by_row.setdefault(r.skewer_idx, []).append(r)

    for t in res.unmatched_truth:
        cands = by_row.get(t.skewer_idx, [])
        if not cands: continue
        # nearest R by edge distance to truth peak
        nearest = None
        nearest_d = 10**9
        for r in cands:
            if t.pix_peak < r.pix_start: d = r.pix_start - t.pix_peak
            elif t.pix_peak > r.pix_end: d = t.pix_peak - r.pix_end
            else: d = 0
            if d < nearest_d:
                nearest_d = d
                nearest = r
        if nearest is None: continue
        edge_distances.append(nearest_d)
        # Span overlap: positive if T.span ∩ R.span ≠ ∅
        overlap_lo = max(t.pix_start, nearest.pix_start)
        overlap_hi = min(t.pix_end, nearest.pix_end)
        overlap = max(0, overlap_hi - overlap_lo + 1)
        overlap_pixels.append(overlap)
        truth_widths.append(t.pix_end - t.pix_start + 1)
        recovered_widths.append(nearest.pix_end - nearest.pix_start + 1)
        truth_NHI_T2.append(t.NHI_truth)
        recovered_NHI_T2.append(nearest.NHI_recovered)

    # Unmatched recovered (DLA class only) — for R2 cases, distance
    # from R centre to colden peak in R.span
    rec_dla = [r for r in recovered if r.absorber_class == "DLA"]
    res_dla = match_dla_lists(truth, rec_dla, tol_pixels=tol)
    for r in res_dla.unmatched_recovered:
        row = colden[r.skewer_idx]
        nhi_in_span = float(row[r.pix_start : r.pix_end + 1].sum())
        if nhi_in_span >= 2.0e20:
            # locate the colden peak in this span
            span_argmax = r.pix_start + int(np.argmax(row[r.pix_start : r.pix_end + 1]))
            centre = (r.pix_start + r.pix_end) // 2
            r2_peak_distance.append(abs(span_argmax - centre))

edge_distances = np.array(edge_distances)
overlap_pixels = np.array(overlap_pixels)
truth_widths = np.array(truth_widths)
recovered_widths = np.array(recovered_widths)
truth_NHI_T2 = np.array(truth_NHI_T2)
recovered_NHI_T2 = np.array(recovered_NHI_T2)

print("=== T2 (unmatched truth, nearest R found) ===")
print(f"N = {len(edge_distances)}")
print()
print("Truth-peak to nearest-R-edge (pixels):")
print(f"  median={np.median(edge_distances):.1f}, p25={np.percentile(edge_distances,25):.1f}, "
      f"p75={np.percentile(edge_distances,75):.1f}, max={edge_distances.max()}")
print()
print("Span overlap (pixels), truth.span ∩ R.span:")
print(f"  median={np.median(overlap_pixels):.1f}, %=0(no-overlap)={(overlap_pixels==0).mean()*100:.1f}%, "
      f"max={overlap_pixels.max()}")
print(f"  fraction with ANY overlap: {(overlap_pixels>0).mean()*100:.1f}%")
print()
print(f"Truth widths:     median={np.median(truth_widths):.1f}, p95={np.percentile(truth_widths,95):.0f}")
print(f"Recovered widths: median={np.median(recovered_widths):.1f}, p95={np.percentile(recovered_widths,95):.0f}")
print()
print("Truth NHI: median=%.2e (log=%.2f)" % (np.median(truth_NHI_T2), np.log10(np.median(truth_NHI_T2))))
print("Recovered NHI of nearest R: median=%.2e (log=%.2f)" %
      (np.median(recovered_NHI_T2), np.log10(np.median(recovered_NHI_T2))))
print()
print("Cumulative hit-rate vs tol_pixels (i.e. how many T2 would resolve at higher tol):")
for tol_test in (10, 20, 30, 50, 80, 120, 200, 400):
    hits = (edge_distances <= tol_test).sum()
    print(f"  tol={tol_test:4d}px: {hits} ({hits / len(edge_distances) * 100:.1f}% of T2)")
print()
print("=== R2 (unmatched recovered with colden ≥ DLA threshold) ===")
print(f"N = {len(r2_peak_distance)}")
if r2_peak_distance:
    arr = np.array(r2_peak_distance)
    print(f"|colden-peak-in-span - R centre|: median={np.median(arr):.1f}, p95={np.percentile(arr,95):.0f}")
