"""Paired read-out of the matched leave-one-simulation-out comparison of two emulators ("c1" and "gp") on the same
held-out simulations, as pre-registered for the release validation. NumPy and SciPy only.

Per method and fold, e = P_pred / T - 1 at every held-out (row, mode), both methods on the identical points. The
primary stratum is 1e-3 <= k <= 0.065 s/km at z 2.2-4.6, split into five equally weighted bands. Per band: the
median |e| and the RMS of e for each method, and their paired differences (C1 - GP). Uncertainty: a simulation-level
paired bootstrap (the folds, sorted by name, resampled with replacement; the same draw for both methods) giving
percentile 95% intervals, and fold-level Wilcoxon signed-rank tests (per band and band-averaged). Verdict on the
band-averaged differences (``verdict``): "C1 better" when both intervals lie entirely below 0, "GP better" when both
lie entirely above 0, otherwise "mixed / inconclusive". Every expected fold without a result (NOT_RUN, or no fold file
at all, for example a job stopped at its time limit) is left out for both methods and listed; a non-finite error on a
compared point is refused, never cut. All folds must come from one training arm."""

from __future__ import annotations

import numpy as np

BANDS = (
    (1e-3, 3e-3),
    (3e-3, 1e-2),
    (1e-2, 2e-2),
    (2e-2, 5e-2),
    (5e-2, 0.065),
)  # the last band includes 0.065
Z_RANGE = (2.2, 4.6)
N_BOOT, BOOT_SEED = 10000, 20261026
METHODS = ("c1", "gp")


def band_index(k, bands=BANDS):
    """Band of each k (0 .. len(bands) - 1), -1 outside; the last band is closed above."""
    k = np.asarray(k, dtype=np.float64)
    out = np.full(k.shape, -1, dtype=np.int64)
    for b, (lo, hi) in enumerate(bands):
        out[(k >= lo) & ((k < hi) if b < len(bands) - 1 else (k <= hi))] = b
    return out


def weighted_median(sorted_vals, w_sorted):
    """np.median of ``sorted_vals`` with each value repeated ``w_sorted`` times (integer weights), without repeating."""
    c = np.cumsum(w_sorted)
    total = int(c[-1])
    lo = sorted_vals[np.searchsorted(c, (total - 1) // 2, side="right")]
    hi = sorted_vals[np.searchsorted(c, total // 2, side="right")]
    return 0.5 * (lo + hi)


def _pool(folds, bands):
    """Per band and method: |e| sorted, its fold index in that order, and per-fold sums of e^2 and point counts."""
    ok = [f for f in folds if f["status"] == "OK"]
    out = []
    for b in range(len(bands)):
        entry = {}
        for m in METHODS:
            vals, fidx, sq, n = [], [], np.zeros(len(ok)), np.zeros(len(ok))
            for i, f in enumerate(ok):
                sel = f["_band"] == b
                e = f["e_" + m][sel]
                vals.append(np.abs(e))
                fidx.append(np.full(e.size, i))
                sq[i], n[i] = np.sum(e**2), e.size
            v, fi = np.concatenate(vals), np.concatenate(fidx)
            order = np.argsort(v, kind="stable")
            entry[m] = {"abs": v[order], "fold": fi[order], "sq": sq, "n": n}
        out.append(entry)
    return out


def _stats(pooled, w):
    """Per band and method (median |e|, RMS) for fold multiplicities ``w``."""
    res = np.empty((len(pooled), len(METHODS), 2))
    for b, entry in enumerate(pooled):
        for j, m in enumerate(METHODS):
            p = entry[m]
            res[b, j, 0] = weighted_median(p["abs"], w[p["fold"]])
            res[b, j, 1] = np.sqrt(np.dot(w, p["sq"]) / np.dot(w, p["n"]))
    return res


def _check(folds, bands):
    for f in folds:
        if f["status"] not in ("OK", "NOT_RUN"):
            raise ValueError(f"fold {f['fold']}: unknown status {f['status']!r}")
        if f["status"] != "OK":
            continue
        shapes = {np.shape(f[k]) for k in ("e_c1", "e_gp", "k_skm", "z")}
        if len(shapes) != 1:
            raise ValueError(
                f"fold {f['fold']}: the methods are not on identical points ({shapes})"
            )
        z = np.asarray(f["z"], dtype=np.float64)
        keep = (z >= Z_RANGE[0] - 1e-9) & (z <= Z_RANGE[1] + 1e-9)
        f["_band"] = np.where(keep, band_index(f["k_skm"], bands), -1)
        for m in METHODS:
            e = np.asarray(f["e_" + m], dtype=np.float64)
            if not np.all(np.isfinite(e[f["_band"] >= 0])):
                raise ValueError(f"fold {f['fold']}: non-finite {m} error on a compared point")
            f["e_" + m] = e
        empty = [b for b in range(len(bands)) if not np.any(f["_band"] == b)]
        if empty:
            raise ValueError(f"fold {f['fold']}: no point in band(s) {empty}")


def verdict(avg_ci):
    """The three-way verdict from the band-averaged 95% intervals ``avg_ci`` = [[lo_median, lo_rms], [hi_median,
    hi_rms]] of the differences C1 - GP."""
    lo, hi = np.asarray(avg_ci, dtype=np.float64)
    if np.all(hi < 0):
        return "C1 better"
    if np.all(lo > 0):
        return "GP better"
    return "mixed / inconclusive"


def readout(folds, *, expected=None, bands=BANDS, n_boot=N_BOOT, seed=BOOT_SEED):
    """The pre-registered statistics. ``folds``: one mapping per fold with ``fold`` (simulation), ``status`` ("OK" or
    "NOT_RUN"), optionally ``arm`` and ``gp_unconverged_z``, and for OK folds ``e_c1``, ``e_gp``, ``k_skm``, ``z``
    (same shape: the held-out points). ``expected``: every fold that should be present; a missing one is listed as
    NOT_RUN ("no fold file") and counts against completeness. Folds are sorted by name before the bootstrap.
    """
    from scipy.stats import wilcoxon

    folds = sorted((dict(f) for f in folds), key=lambda f: str(f["fold"]))
    names = [f["fold"] for f in folds]
    if len(set(names)) != len(names):
        raise ValueError("a fold appears twice")
    arms = {f.get("arm") for f in folds if f["status"] == "OK"}
    if len(arms) > 1:
        raise ValueError(f"folds from different training arms are mixed: {sorted(map(str, arms))}")
    missing = []
    if expected is not None:
        extra = sorted(set(names) - set(expected))
        if extra:
            raise ValueError(f"folds outside the expected list: {extra[:5]}")
        missing = sorted(set(expected) - set(names))
    _check(folds, bands)
    ok = [f for f in folds if f["status"] == "OK"]
    if len(ok) < 2:
        raise ValueError(f"{len(ok)} compared folds; a paired read-out needs at least 2")
    pooled = _pool(folds, bands)
    point = _stats(pooled, np.ones(len(ok)))
    delta = point[:, 0, :] - point[:, 1, :]  # (bands, [median, RMS]): C1 - GP
    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, len(bands), 2))
    for i in range(n_boot):
        w = np.bincount(rng.integers(0, len(ok), len(ok)), minlength=len(ok)).astype(np.float64)
        s = _stats(pooled, w)
        boot[i] = s[:, 0, :] - s[:, 1, :]
    ci = np.percentile(boot, [2.5, 97.5], axis=0)  # (2, bands, 2)
    avg = boot.mean(axis=1)  # (n_boot, 2) band-averaged, equal weights
    avg_ci = np.percentile(avg, [2.5, 97.5], axis=0)  # (2, 2)
    per_fold = np.empty((len(ok), len(bands), 2))  # each fold's own per-band delta
    for i in range(len(ok)):
        s = _stats(pooled, np.eye(len(ok))[i])
        per_fold[i] = s[:, 0, :] - s[:, 1, :]

    def wil(d):
        return float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0

    tests = {name: wil(per_fold[:, :, j].mean(axis=1)) for j, name in enumerate(("median", "rms"))}
    not_run = [
        {"fold": f["fold"], "reason": f.get("reason", "")}
        for f in folds
        if f["status"] == "NOT_RUN"
    ]
    not_run += [{"fold": m, "reason": "no fold file (time limit or not run)"} for m in missing]
    return {
        "bands": [list(b) for b in bands],
        "n_folds_compared": len(ok),
        "n_folds_expected": None if expected is None else len(expected),
        "not_run": not_run,
        "arm": next(iter(arms)) if arms else None,
        "gp_unconverged": {
            f["fold"]: f["gp_unconverged_z"] for f in ok if f.get("gp_unconverged_z")
        },
        "per_band": [
            {
                "band": list(bands[b]),
                "median_abs_e": {m: float(point[b, j, 0]) for j, m in enumerate(METHODS)},
                "rms_e": {m: float(point[b, j, 1]) for j, m in enumerate(METHODS)},
                "delta_median": float(delta[b, 0]),
                "delta_rms": float(delta[b, 1]),
                "delta_median_ci95": ci[:, b, 0].tolist(),
                "delta_rms_ci95": ci[:, b, 1].tolist(),
                "wilcoxon_p": {"median": wil(per_fold[:, b, 0]), "rms": wil(per_fold[:, b, 1])},
            }
            for b in range(len(bands))
        ],
        "band_averaged": {
            "delta_median": float(delta[:, 0].mean()),
            "delta_rms": float(delta[:, 1].mean()),
            "delta_median_ci95": avg_ci[:, 0].tolist(),
            "delta_rms_ci95": avg_ci[:, 1].tolist(),
        },
        "wilcoxon_p": tests,
        "verdict": verdict(avg_ci),
        "bootstrap": {"n": int(n_boot), "seed": int(seed)},
    }
