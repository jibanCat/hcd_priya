"""
DLA truth from particle-based column-density skewers (rand_spectra_DLA.hdf5).

Purpose
-------
Validate the production τ-peak DLA finder (catalog.find_systems_in_skewer +
voigt_utils.nhi_from_tau_fast) against an independent particle-based ground
truth.  The HiRes rand_spectra_DLA.hdf5 files carry a per-pixel HI column
density dataset `colden/H/1` (cm⁻²) computed directly from the GADGET particle
snapshot, which the production grid spectra do NOT carry (their `colden`
group is empty).  We treat those colden skewers as truth, identify DLAs in
them by contiguous-pixel integration above the canonical 2 × 10²⁰ cm⁻²
threshold, then match them to whatever the τ-peak finder recovers from
`tau/H/1/1215`.

The matching algorithm and tolerance choices are documented in
`docs/dla_truth_validation.md`.

Public API
----------
- find_truth_dlas_from_colden(colden, dx_*, dla_threshold=2e20, ...) →
    list of TruthDLA records (skewer_idx, pix_start, pix_end, NHI_truth,
    pix_peak).
- match_dla_lists(truth, recovered, tol_pixels=...) → MatchResult with
    matched-pair table, unmatched truth, unmatched recovered.
- summary_stats(truth, recovered, matched) → dict of scalar diagnostics
    (N_truth, N_recovered, completeness, purity, mean Δlog NHI, σ Δlog NHI).

These are intentionally simple, dependency-light containers so they can be
unit-tested without HDF5 input.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Truth DLA record from a colden skewer
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TruthDLA:
    """A single DLA found by integrating colden along a sightline."""
    skewer_idx: int
    pix_start: int        # inclusive
    pix_end: int          # inclusive
    pix_peak: int         # pixel of maximum colden inside the run
    NHI_truth: float      # cm^-2 — Σ colden[skewer, pix_start:pix_end+1]


@dataclasses.dataclass
class RecoveredDLA:
    """A τ-peak-finder system mirrored to a small dataclass for matching."""
    skewer_idx: int
    pix_start: int        # inclusive
    pix_end: int          # inclusive
    NHI_recovered: float  # cm^-2
    log_NHI: float        # log10(NHI_recovered)
    absorber_class: str   # "LLS" | "subDLA" | "DLA"


@dataclasses.dataclass
class MatchedPair:
    truth: TruthDLA
    recovered: RecoveredDLA
    delta_log_nhi: float  # log10(NHI_rec) - log10(NHI_truth)
    delta_pix: int        # |centre_rec - pix_peak_truth|


@dataclasses.dataclass
class MatchResult:
    matched: List[MatchedPair]
    unmatched_truth: List[TruthDLA]
    unmatched_recovered: List[RecoveredDLA]


# ---------------------------------------------------------------------------
# Truth-DLA finder on a colden array
# ---------------------------------------------------------------------------

# Canonical DLA NHI threshold (Wolfe et al. 1986).  We use a per-pixel
# floor that is lower than this — the integrated-NHI test below is what
# decides whether a run is a DLA.  The pixel floor only controls how
# we stitch contiguous regions.
_PIXEL_FLOOR_CM2 = 1.0e17


def _runs_above(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return inclusive (start, end) runs of True pixels in a 1-D bool mask."""
    runs: List[Tuple[int, int]] = []
    in_run = False
    s = 0
    for i, v in enumerate(mask):
        if v and not in_run:
            s = i
            in_run = True
        elif not v and in_run:
            runs.append((s, i - 1))
            in_run = False
    if in_run:
        runs.append((s, len(mask) - 1))
    return runs


def _merge_close_runs(
    runs: List[Tuple[int, int]],
    merge_gap_pixels: int,
) -> List[Tuple[int, int]]:
    """Merge consecutive runs whose pixel gap ≤ merge_gap_pixels."""
    if not runs:
        return runs
    merged = [runs[0]]
    for s, e in runs[1:]:
        prev_s, prev_e = merged[-1]
        gap = s - prev_e - 1
        if gap <= merge_gap_pixels:
            merged[-1] = (prev_s, e)
        else:
            merged.append((s, e))
    return merged


def find_truth_dlas_from_colden(
    colden: np.ndarray,
    dla_threshold: float = 2.0e20,
    pixel_floor: float = _PIXEL_FLOOR_CM2,
    merge_gap_pixels: int = 0,
    min_pixels: int = 1,
) -> List[TruthDLA]:
    """
    Identify truth-DLAs in per-pixel colden array(s).

    A truth-DLA is a contiguous run of pixels whose integrated NHI
    (`Σ colden[pix_start:pix_end+1]` along the row) is ≥ `dla_threshold`.

    Parameters
    ----------
    colden : ndarray, shape (n_skewers, n_pixels) or (n_pixels,)
        Per-pixel HI column density in cm⁻². Standard fake_spectra
        rand_spectra_DLA convention: each pixel reports the column density
        deposited by SPH particles into that pixel — sums of contiguous
        pixels give the integrated NHI of the absorber.
    dla_threshold : float
        Integrated-NHI threshold for a run to count as a DLA. Default
        2 × 10²⁰ cm⁻² is the canonical Wolfe et al. value.
    pixel_floor : float
        Per-pixel threshold used to stitch contiguous runs. Pixels below
        this are treated as outside the absorber. Default 1 × 10¹⁷ cm⁻²
        is well below LLS strength and lets the merging step (next param)
        handle short forest dips inside a real DLA.
    merge_gap_pixels : int
        Two runs separated by ≤ this many sub-floor pixels are merged
        into one. Default 0 (strict contiguity). A small positive value
        approximates the merging convention in the production τ finder
        (`merge_dv_kms / dv_pix_kms`).
    min_pixels : int
        Minimum run length in pixels (after merging) before considering
        the threshold test. Default 1.

    Returns
    -------
    list[TruthDLA]
        One record per run whose integrated NHI exceeds `dla_threshold`.
        For a 2-D `colden`, records are emitted in (row, then run) order.
    """
    if colden.ndim == 1:
        colden = colden[None, :]
        single = True
    elif colden.ndim == 2:
        single = False
    else:
        raise ValueError(f"colden must be 1-D or 2-D, got ndim={colden.ndim}")

    if dla_threshold <= 0 or pixel_floor <= 0:
        raise ValueError("thresholds must be positive")
    if merge_gap_pixels < 0 or min_pixels < 1:
        raise ValueError("merge_gap_pixels must be ≥ 0 and min_pixels ≥ 1")

    out: List[TruthDLA] = []
    for irow in range(colden.shape[0]):
        row = np.asarray(colden[irow], dtype=np.float64)
        if not np.any(row > pixel_floor):
            continue
        runs = _runs_above(row > pixel_floor)
        runs = _merge_close_runs(runs, merge_gap_pixels)
        for s, e in runs:
            if (e - s + 1) < min_pixels:
                continue
            nhi = float(row[s : e + 1].sum())
            if nhi < dla_threshold:
                continue
            peak = int(s + np.argmax(row[s : e + 1]))
            out.append(
                TruthDLA(
                    skewer_idx=int(irow if not single else 0),
                    pix_start=int(s),
                    pix_end=int(e),
                    pix_peak=peak,
                    NHI_truth=nhi,
                )
            )
    return out


# ---------------------------------------------------------------------------
# Position-tolerant matching between truth and recovered DLA lists
# ---------------------------------------------------------------------------

def match_dla_lists(
    truth: Sequence[TruthDLA],
    recovered: Sequence[RecoveredDLA],
    tol_pixels: int = 10,
) -> MatchResult:
    """
    Greedy 1-to-1 matching between truth and recovered DLAs, restricted
    to the same skewer.

    Match criterion ("overlap-with-margin")
    ---------------------------------------
    A truth DLA T and a recovered system R on the same skewer match iff
    the truth's pixel of maximum colden falls within the recovered span
    expanded by a margin of `tol_pixels` on each side:

        R.pix_start - tol_pixels  ≤  T.pix_peak  ≤  R.pix_end + tol_pixels.

    Rationale: the τ-peak finder reports `(pix_start, pix_end)` for the
    contiguous τ > τ_threshold core only. For a saturated DLA the τ core
    is narrower than the underlying colden distribution, and the centre
    of the τ-core is NOT the colden peak — it can drift by several
    hundred pixels for very large absorbers (saturated cores are flat-
    bottomed; argmax-of-τ jitters across the core). Asking instead "is
    the truth peak inside the recovered span (with a small margin
    matching the merge_dv_kms convention)?" gives a near-1:1 matching
    that is insensitive to the core's flat-bottom argmax pathology.

    The margin tolerance is set by the caller. The natural choice is
    `tol_pixels = max(merge_dv_kms / dv_pix_kms, 5)`, matching the
    production τ-finder's merge gap.

    For 1-to-1 enforcement: when multiple truth DLAs land inside the
    same recovered span, we pair the truth whose peak is closest to the
    recovered span's centre and leave the others unmatched. When
    multiple recovered systems contain the same truth peak, we pick the
    one whose centre is closest to the peak.

    Parameters
    ----------
    truth : list[TruthDLA]
    recovered : list[RecoveredDLA]
    tol_pixels : int
        Margin added to each side of the recovered span before testing
        whether the truth peak is inside.

    Returns
    -------
    MatchResult with matched pairs, unmatched truth, unmatched recovered.
    """
    # Group recovered by skewer for fast lookup
    rec_by_row: Dict[int, List[int]] = {}
    for j, r in enumerate(recovered):
        rec_by_row.setdefault(r.skewer_idx, []).append(j)

    matched: List[MatchedPair] = []
    used_rec = set()
    unmatched_truth: List[TruthDLA] = []

    for t in truth:
        candidates = rec_by_row.get(t.skewer_idx, [])
        best_j = -1
        # Tracking absolute |centre_rec - pix_peak_truth| only as a tie-breaker.
        # Initial sentinel must exceed any plausible distance on the row.
        best_d = float("inf")
        for j in candidates:
            if j in used_rec:
                continue
            r = recovered[j]
            lo = r.pix_start - tol_pixels
            hi = r.pix_end + tol_pixels
            if lo <= t.pix_peak <= hi:
                centre = (r.pix_start + r.pix_end) // 2
                d = abs(centre - t.pix_peak)
                if d < best_d:
                    best_d = d
                    best_j = j
        if best_j >= 0:
            r = recovered[best_j]
            used_rec.add(best_j)
            dlog = float(np.log10(max(r.NHI_recovered, 1.0)) - np.log10(max(t.NHI_truth, 1.0)))
            matched.append(
                MatchedPair(
                    truth=t,
                    recovered=r,
                    delta_log_nhi=dlog,
                    delta_pix=int(best_d),
                )
            )
        else:
            unmatched_truth.append(t)

    unmatched_recovered = [r for j, r in enumerate(recovered) if j not in used_rec]
    return MatchResult(
        matched=matched,
        unmatched_truth=unmatched_truth,
        unmatched_recovered=unmatched_recovered,
    )


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summary_stats(
    truth: Sequence[TruthDLA],
    recovered: Sequence[RecoveredDLA],
    matched: MatchResult,
) -> Dict[str, float]:
    """
    Aggregate diagnostics for one (sim, snap) pair.

    Definitions
    -----------
    completeness = N_matched / N_truth
        Fraction of truth DLAs that were paired with a recovered system.
        A purely "did we find every real DLA" metric.
    purity = N_matched / N_recovered
        Fraction of recovered systems (DLA-class only — see note below)
        that match a truth DLA.
    mean Δlog NHI = mean(log10 NHI_recovered − log10 NHI_truth) over matched pairs.
    σ Δlog NHI = std(...) over matched pairs.

    Note: We deliberately use the full recovered list as denominator for
    purity. If the caller filters `recovered` to DLA-class only, purity
    is "DLA purity"; if they pass everything ≥ LLS, the reported number
    is the looser "any-class purity".

    Returns dict with keys:
        N_truth, N_recovered, N_matched,
        completeness, purity,
        mean_dlog_nhi, std_dlog_nhi, median_dlog_nhi,
        mean_abs_dpix, max_abs_dpix.
    """
    n_t = len(truth)
    n_r = len(recovered)
    n_m = len(matched.matched)

    if n_m == 0:
        return {
            "N_truth": float(n_t),
            "N_recovered": float(n_r),
            "N_matched": 0.0,
            "completeness": 0.0,
            "purity": 0.0,
            "mean_dlog_nhi": float("nan"),
            "std_dlog_nhi": float("nan"),
            "median_dlog_nhi": float("nan"),
            "mean_abs_dpix": float("nan"),
            "max_abs_dpix": float("nan"),
        }

    dlog = np.array([m.delta_log_nhi for m in matched.matched])
    dpix = np.array([m.delta_pix for m in matched.matched])
    return {
        "N_truth": float(n_t),
        "N_recovered": float(n_r),
        "N_matched": float(n_m),
        "completeness": float(n_m) / max(n_t, 1),
        "purity": float(n_m) / max(n_r, 1),
        "mean_dlog_nhi": float(np.mean(dlog)),
        "std_dlog_nhi": float(np.std(dlog, ddof=1)) if n_m > 1 else 0.0,
        "median_dlog_nhi": float(np.median(dlog)),
        "mean_abs_dpix": float(np.mean(np.abs(dpix))),
        "max_abs_dpix": float(np.max(np.abs(dpix))),
    }


# ---------------------------------------------------------------------------
# Convenience: build RecoveredDLA records from the production catalog API
# ---------------------------------------------------------------------------

def recovered_from_systems(
    systems_per_skewer: Iterable[Tuple[int, List[Tuple[int, int]]]],
    nhi_per_system: Iterable[Tuple[int, int, int, float]],
) -> List[RecoveredDLA]:
    """
    Helper to build a RecoveredDLA list from the production catalog return
    pattern, which yields per-skewer (pix_start, pix_end) runs and a separate
    NHI value per (skewer, pix_start, pix_end).

    `nhi_per_system` is a list of (skewer_idx, pix_start, pix_end, NHI) tuples.
    The first argument is currently unused (kept for symmetry with the
    typical pipeline call pattern); the function classifies based on the
    integrated NHI alone.
    """
    from .catalog import classify_system  # local import to avoid cycles

    out: List[RecoveredDLA] = []
    for irow, ps, pe, nhi in nhi_per_system:
        out.append(
            RecoveredDLA(
                skewer_idx=int(irow),
                pix_start=int(ps),
                pix_end=int(pe),
                NHI_recovered=float(nhi),
                log_NHI=float(np.log10(max(nhi, 1.0))),
                absorber_class=classify_system(nhi),
            )
        )
    return out


__all__ = [
    "TruthDLA",
    "RecoveredDLA",
    "MatchedPair",
    "MatchResult",
    "find_truth_dlas_from_colden",
    "match_dla_lists",
    "summary_stats",
    "recovered_from_systems",
]
