"""Gate C validation helpers (emulator-debug campaign 2026-10; spec GATE_C_SPEC.md in the notes repository).

Truth is the held-out simulation's own source product (cache rows on their own header-exact grid); predictions come
through the schema-2.0 physical-coordinate API. These helpers keep the comparison honest: coordinate agreement is
measured explicitly (C1), interpolation onto a fixed physical grid never extrapolates, and held-out entries are paired
with upstream's leave-one-out product by parameters, mean-flux rung and redshift (C4)."""
from __future__ import annotations

import numpy as np


def coordinate_agreement(k_pred, k_row):
    """max over the row's finite modes of |k_pred / k_row - 1| (C1)."""
    k_pred, k_row = np.asarray(k_pred, float), np.asarray(k_row, float)
    m = np.isfinite(k_row)
    return float(np.max(np.abs(k_pred[m] / k_row[m] - 1.0)))


def interp_to_grid(k_target, k, P):
    """Linear interpolation of P(k) onto k_target (the forward's convention); refuses any target outside [k_min, k_max]
    of the finite modes instead of clamping."""
    k, P, k_target = np.asarray(k, float), np.asarray(P, float), np.asarray(k_target, float)
    m = np.isfinite(k) & np.isfinite(P)
    if k_target.min() < k[m].min() or k_target.max() > k[m].max():
        raise ValueError(f"target k [{k_target.min():.4g}, {k_target.max():.4g}] outside the sampled modes "
                         f"[{k[m].min():.4g}, {k[m].max():.4g}]")
    return np.interp(k_target, k[m], P[m])


def class_rms(res_cls, keep):
    """Per-class RMS of fractional residuals res_cls (rows, 4, K) over the kept (rows, K) finite modes: the historical
    LOSO/ensemble metric (run_loso_sweep.py, validate_production_ensemble.py)."""
    r = np.where(np.asarray(keep, bool)[:, None, :] & np.isfinite(res_cls), res_cls, np.nan)
    return np.sqrt(np.nanmean(r ** 2, axis=(0, 2)))


def ensemble_residual(member_residuals):
    """Residual of the ensemble MEAN prediction from members' fractional residuals against the same truth
    (mean of P_pred / P_true - 1 = mean of the residuals, because the truth is common)."""
    return np.mean(np.stack([np.asarray(r, float) for r in member_residuals]), axis=0)


def match_upstream_loo(entries, up_params, up_zout, *, rtol=1e-6, rung_atol=2e-6, z_atol=1e-6):
    """Pair our held-out entries with upstream LOO rows. ``entries``: dicts with ``params`` (the simulation parameters,
    upstream's order), ``alpha`` (mean-flux rung), ``z`` and ``key``; ``up_params[:, 0]`` is upstream's rung and the
    rest its parameters. Returns {key: (upstream_row, upstream_z_index)} for entries with exactly one match."""
    up_params = np.asarray(up_params, float)
    out = {}
    for e in entries:
        rows = np.where(np.all(np.isclose(up_params[:, 1:], np.asarray(e["params"], float)[None, :], rtol=rtol, atol=0),
                               axis=1) & np.isclose(up_params[:, 0], e["alpha"], rtol=0, atol=rung_atol))[0]
        zi = np.where(np.isclose(np.asarray(up_zout, float), e["z"], rtol=0, atol=z_atol))[0]
        if rows.size == 1 and zi.size == 1:
            out[e["key"]] = (int(rows[0]), int(zi[0]))
    return out
