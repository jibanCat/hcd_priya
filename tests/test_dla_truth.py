"""
Unit tests for hcd_analysis.dla_truth — the particle-based DLA truth finder
and the matching helper.

Covers:
  1. A single injected DLA is recovered with correct integrated NHI.
  2. Two contiguous DLAs in adjacent pixels are merged into one.
  3. A run that never reaches the integrated DLA threshold is dropped.
  4. Position-tolerant matching: matched / unmatched bookkeeping is correct.
  5. summary_stats computes completeness, purity, mean / σ Δlog NHI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hcd_analysis.dla_truth import (
    RecoveredDLA,
    TruthDLA,
    find_truth_dlas_from_colden,
    match_dla_lists,
    summary_stats,
)


# ---------------------------------------------------------------------------
# Helper: synthetic colden skewer with one or more injected absorbers
# ---------------------------------------------------------------------------

def _inject(nbins: int, peaks):
    """
    Build a 1-D colden array with Gaussian-shaped peaks.
    `peaks` is a list of (pix_centre, fwhm_pix, total_NHI) triples.
    """
    row = np.zeros(nbins, dtype=np.float64)
    pix = np.arange(nbins)
    for centre, fwhm, total in peaks:
        sigma = max(fwhm / 2.355, 0.5)
        g = np.exp(-0.5 * ((pix - centre) / sigma) ** 2)
        g = g / g.sum() * total
        row += g
    return row


# ---------------------------------------------------------------------------
# Test 1 — single injected DLA
# ---------------------------------------------------------------------------

def test_single_dla_recovered():
    nbins = 1000
    target = 5.0e20  # log NHI = 20.7
    row = _inject(nbins, [(500, 8, target)])
    truth = find_truth_dlas_from_colden(
        row, dla_threshold=2.0e20, pixel_floor=1.0e17,
    )
    assert len(truth) == 1, f"expected 1 DLA, got {len(truth)}"
    t = truth[0]
    # peak should be near pixel 500
    assert abs(t.pix_peak - 500) <= 1
    # integrated NHI should match within numerical roundoff (Gaussian tails
    # below pixel_floor are excluded — that's < 1e-6 of the total).
    assert abs(np.log10(t.NHI_truth) - np.log10(target)) < 0.01, (
        f"NHI_truth={t.NHI_truth:.3e}, expected {target:.3e}"
    )
    print(
        "  test_single_dla_recovered:",
        f"pix_peak={t.pix_peak} NHI_truth={t.NHI_truth:.3e} "
        f"(expected ≈ {target:.3e})",
    )


# ---------------------------------------------------------------------------
# Test 2 — adjacent DLAs merge into one
# ---------------------------------------------------------------------------

def test_adjacent_dlas_merge():
    """
    Two DLA-strength peaks at pixels 500 and 504 with overlapping wings —
    every pixel between them is above the per-pixel floor → they MUST be
    merged into a single integrated absorber.  This matches the production
    τ-finder convention (`merge_dv_kms`).

    A third "DLA at 700" is well separated → must remain a second record.
    """
    nbins = 1000
    row = _inject(nbins, [(500, 6, 4.0e20), (504, 6, 4.0e20), (700, 6, 4.0e20)])
    truth = find_truth_dlas_from_colden(
        row, dla_threshold=2.0e20, pixel_floor=1.0e17,
    )
    assert len(truth) == 2, f"expected 2 DLAs (one merged), got {len(truth)}"
    t_merged = truth[0]
    # The merged record's integrated NHI ≈ sum of both injected totals
    expected_merged = 8.0e20
    assert abs(np.log10(t_merged.NHI_truth) - np.log10(expected_merged)) < 0.01
    # Peak should be at one of the two injected centres
    assert t_merged.pix_peak in (500, 501, 502, 503, 504)
    # Span should span both injected runs
    assert t_merged.pix_start < 500 and t_merged.pix_end > 504
    print(
        "  test_adjacent_dlas_merge:",
        f"merged span={t_merged.pix_start}-{t_merged.pix_end}, "
        f"NHI={t_merged.NHI_truth:.3e}",
    )


# ---------------------------------------------------------------------------
# Test 3 — sub-threshold run is not flagged
# ---------------------------------------------------------------------------

def test_sub_dla_not_found():
    """
    A subDLA-strength absorber (integrated NHI = 5e19 cm⁻²) must not be
    flagged as a DLA when the threshold is set to the canonical 2e20.
    """
    nbins = 1000
    row = _inject(nbins, [(500, 6, 5.0e19)])
    truth = find_truth_dlas_from_colden(
        row, dla_threshold=2.0e20, pixel_floor=1.0e17,
    )
    assert len(truth) == 0, f"expected 0 DLAs, got {len(truth)}"
    print("  test_sub_dla_not_found: confirmed (NHI=5e19 < threshold 2e20)")


# ---------------------------------------------------------------------------
# Test 4 — position matching produces correct matched / unmatched lists
# ---------------------------------------------------------------------------

def test_position_matching():
    """
    Truth peaks vs recovered spans (overlap-with-margin matching, tol=10).

    Skewer 0: recovered span 99..101 contains truth peak 100 → match.
    Skewer 1: recovered span 64..68; truth peak 60 is 4 px outside the
              left edge (tol=10) → still inside, match.
    Skewer 2: recovered span 175..178 expanded by 10 = 165..188; truth
              peak 205 is outside → unmatched.
    Skewer 3: recovered system with no truth on the skewer → unmatched
              recovered.
    """
    truth = [
        TruthDLA(skewer_idx=0, pix_start=98, pix_end=104, pix_peak=100, NHI_truth=5.0e20),
        TruthDLA(skewer_idx=1, pix_start=58, pix_end=62, pix_peak=60, NHI_truth=3.0e20),
        TruthDLA(skewer_idx=2, pix_start=200, pix_end=210, pix_peak=205, NHI_truth=4.0e20),
    ]
    recovered = [
        # span contains truth[0].pix_peak
        RecoveredDLA(0, 99, 101, 5.0e20, 20.7, "DLA"),
        # span 64..68 + 10 margin reaches 54..78 → contains 60
        RecoveredDLA(1, 64, 68, 2.5e20, 20.4, "DLA"),
        # span 175..178 + 10 margin reaches 165..188 → does NOT contain 205
        RecoveredDLA(2, 175, 178, 3.0e20, 20.5, "DLA"),
        # spurious recovered on a skewer without truth
        RecoveredDLA(3, 50, 60, 1.0e21, 21.0, "DLA"),
    ]
    res = match_dla_lists(truth, recovered, tol_pixels=10)
    assert len(res.matched) == 2, (
        f"expected 2 matches, got {len(res.matched)}"
    )
    matched_skewers = {m.truth.skewer_idx for m in res.matched}
    assert matched_skewers == {0, 1}
    assert len(res.unmatched_truth) == 1
    assert res.unmatched_truth[0].skewer_idx == 2
    # spurious + the out-of-tol recovered → 2 unmatched recovered
    assert len(res.unmatched_recovered) == 2
    unmatched_rec_skewers = {r.skewer_idx for r in res.unmatched_recovered}
    assert unmatched_rec_skewers == {2, 3}
    print(
        "  test_position_matching:",
        f"matched={len(res.matched)} unmatched_truth={len(res.unmatched_truth)} "
        f"unmatched_recovered={len(res.unmatched_recovered)}",
    )


# ---------------------------------------------------------------------------
# Test 5 — summary_stats numerics
# ---------------------------------------------------------------------------

def test_summary_stats_numerics():
    """
    Construct a 4-truth, 5-recovered case where exactly 3 pair up with
    known Δlog NHI values [-0.10, +0.05, 0.00] and verify completeness,
    purity, mean and σ.
    """
    truth = [
        TruthDLA(0, 100, 110, 105, 1.0e21),  # log = 21.00
        TruthDLA(1, 200, 210, 205, 1.0e21),
        TruthDLA(2, 300, 310, 305, 1.0e21),
        TruthDLA(3, 400, 410, 405, 1.0e21),  # this one will be unmatched
    ]
    # Recovered spans contain (or with margin contain) the corresponding truth pix_peak.
    recovered = [
        RecoveredDLA(0, 103, 107, 10 ** 20.90, 20.90, "DLA"),  # span 103..107 contains 105 → Δlog = -0.10
        RecoveredDLA(1, 203, 207, 10 ** 21.05, 21.05, "DLA"),  # contains 205 → Δlog = +0.05
        RecoveredDLA(2, 303, 307, 10 ** 21.00, 21.00, "DLA"),  # contains 305 → Δlog = 0
        RecoveredDLA(99, 50, 60, 1.0e21, 21.0, "DLA"),         # spurious skewer
        RecoveredDLA(99, 70, 80, 1.0e21, 21.0, "DLA"),         # spurious skewer
    ]
    res = match_dla_lists(truth, recovered, tol_pixels=10)
    assert len(res.matched) == 3
    s = summary_stats(truth, recovered, res)
    assert s["N_truth"] == 4
    assert s["N_recovered"] == 5
    assert s["N_matched"] == 3
    assert abs(s["completeness"] - 0.75) < 1e-9
    assert abs(s["purity"] - 0.6) < 1e-9
    # Mean of [-0.1, 0.05, 0.0] = -0.0167
    assert abs(s["mean_dlog_nhi"] - (-0.05 / 3)) < 1e-9, (
        f"got mean={s['mean_dlog_nhi']}"
    )
    # σ of [-0.1, 0.05, 0.0] with ddof=1 = sqrt(0.00583)= 0.0764
    expected_std = float(np.std([-0.1, 0.05, 0.0], ddof=1))
    assert abs(s["std_dlog_nhi"] - expected_std) < 1e-9
    print(
        "  test_summary_stats_numerics:",
        f"completeness={s['completeness']:.3f} purity={s['purity']:.3f} "
        f"⟨Δlog⟩={s['mean_dlog_nhi']:+.4f} σ={s['std_dlog_nhi']:.4f}",
    )


# ---------------------------------------------------------------------------
# Test 6 — 2-D colden input, multi-row bookkeeping
# ---------------------------------------------------------------------------

def test_2d_colden_multirow():
    """
    A small (3, 1000) colden block: rows 0 and 2 carry one DLA each,
    row 1 has none.  Verify the skewer_idx field is set correctly.
    """
    nbins = 1000
    cd = np.zeros((3, nbins), dtype=np.float64)
    cd[0] = _inject(nbins, [(200, 6, 5.0e20)])
    cd[2] = _inject(nbins, [(700, 6, 3.0e20)])
    truth = find_truth_dlas_from_colden(cd, dla_threshold=2.0e20)
    assert len(truth) == 2
    skewers_found = sorted(t.skewer_idx for t in truth)
    assert skewers_found == [0, 2]
    print(
        "  test_2d_colden_multirow:",
        f"found DLAs on skewers {skewers_found}",
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    print("dla_truth unit tests")
    test_single_dla_recovered()
    test_adjacent_dlas_merge()
    test_sub_dla_not_found()
    test_position_matching()
    test_summary_stats_numerics()
    test_2d_colden_multirow()
    print("All tests passed.")


if __name__ == "__main__":
    main()
