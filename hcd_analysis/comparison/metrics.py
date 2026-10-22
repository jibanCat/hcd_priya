"""Summaries of r = (P_test - P_ref) / P_ref and the max-|r| threshold verdict. No clipping, no cuts: every point the
caller passes is counted."""
from __future__ import annotations

import numpy as np


def ratio_residual(P_test, P_ref):
    a, b = np.asarray(P_test, float), np.asarray(P_ref, float)
    if a.shape != b.shape:
        raise ValueError(f"shape {a.shape} vs {b.shape}")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.all(b > 0)):
        raise ValueError("predictions must be finite and the reference positive")
    return (a - b) / b


def summarize(r, threshold):
    r = np.asarray(r, float).ravel()
    if r.size == 0:
        return {"n": 0}
    a = np.abs(r)
    return {"n": int(r.size), "median_abs": float(np.median(a)), "rms": float(np.sqrt(np.mean(r ** 2))),
            "p95_abs": float(np.percentile(a, 95)), "max_abs": float(a.max()), "mean": float(r.mean()),
            "exceed_frac": float(np.mean(a > threshold)), "threshold": float(threshold)}


def verdict(r, threshold):
    """PASS iff max |r| <= threshold over all points; an empty set is NOT RUN, never PASS."""
    r = np.asarray(r, float)
    if r.size == 0:
        return "NOT RUN"
    return "PASS" if float(np.max(np.abs(r))) <= threshold else "FAIL"


def band_index(k, edges):
    """Index of the band [edges[i], edges[i+1]) holding each k; the last band is open-ended."""
    return np.searchsorted(np.asarray(edges, float), np.asarray(k, float), side="right") - 1
