"""Building blocks of the gate E emulator-error products (GATE_E_AMENDMENT_A1.md in the notes repository, registered
before any final product is built): simulation-level cross-validation folds, per-cell second moments of the
leave-one-simulation-out ensemble residuals, PSD-preserving smoothing along ln(mode) and z, the Gaussian predictive
score used for model selection, and the modes that bracket a leg's data bins over the whole sampling box.
Products are calibrated uncertainty tables: nothing here is a function of the cosmological or nuisance parameters."""
from __future__ import annotations

import numpy as np

from .kcoord import kbounds_over_box


def cv_folds(sim_names, n_folds=10):
    """{simulation name: fold} with the simulations sorted by name and fold = index mod n_folds."""
    names = sorted(set(np.asarray(sim_names).astype(str)))
    return {n: i % n_folds for i, n in enumerate(names)}


def second_moment_cells(r, cell, n_cells):
    """Uncentered second moment per cell of the per-class residuals r (rows, C, K): rho (n_cells, C, C, K) and counts."""
    r = np.asarray(r, float)
    cell = np.asarray(cell, int)
    C, K = r.shape[1], r.shape[2]
    rho = np.zeros((n_cells, C, C, K))
    n = np.zeros(n_cells, int)
    for c in range(n_cells):
        sel = cell == c
        n[c] = int(sel.sum())
        if n[c]:
            rho[c] = np.einsum("rck,rdk->cdk", r[sel], r[sel]) / n[c]
    return rho, n


def _kernel(x, width):
    d = (np.asarray(x, float)[:, None] - np.asarray(x, float)[None, :]) / width
    w = np.exp(-0.5 * d ** 2)
    return w / w.sum(axis=1, keepdims=True)


def smooth_modes(rho, h):
    """Smooth the last (mode) axis with a Gaussian kernel of width h in ln(mode index); h = 0 returns rho unchanged.
    Row-normalised positive weights: a PSD matrix field stays PSD and a constant stays constant."""
    if h == 0:
        return rho
    W = _kernel(np.log(np.arange(1, rho.shape[-1] + 1)), h)
    return np.einsum("...k,jk->...j", rho, W)


def smooth_z(rho, z, s_z):
    """Smooth the first (z cell) axis with a Gaussian kernel of width s_z in z; s_z = 0 returns rho unchanged."""
    if s_z == 0:
        return rho
    W = _kernel(z, s_z)
    return np.einsum("z...,yz->y...", rho, W)


def gaussian_score(r, cov, jitter=1e-10):
    """Sum over rows and modes of the C-dimensional Gaussian log density of r (rows, C, K) under cov (rows, C, C, K),
    with the production jitter (jitter x the mean diagonal) added to each block."""
    r = np.moveaxis(np.asarray(r, float), 2, 1)                      # (rows, K, C)
    cov = np.moveaxis(np.asarray(cov, float), 3, 1)                  # (rows, K, C, C)
    C = r.shape[-1]
    diag_mean = np.einsum("...ii->...", cov) / C
    cov = cov + (jitter * diag_mean)[..., None, None] * np.eye(C)
    sign, logdet = np.linalg.slogdet(cov)
    if np.any(sign <= 0):
        raise ValueError("non-positive-definite covariance block in the Gaussian score")
    sol = np.linalg.solve(cov, r[..., None])[..., 0]
    quad = np.sum(r * sol, axis=-1)
    return float(np.sum(-0.5 * (quad + logdet + C * np.log(2 * np.pi))))


def bracket_modes(k_com_hmpc, z, k_lo, k_hi, lo_unit, hi_unit):
    """(first, last) 1-based mode indices that bracket every k in [k_lo, k_hi] at z for every theta in the sampling box
    [lo_unit, hi_unit]: k_skm,n = n k_skm,1 and k_skm,1 spans [kK_min / K, k1_max] over the box."""
    k1_max, kK_min = kbounds_over_box(k_com_hmpc, z, lo_unit, hi_unit)
    K = len(k_com_hmpc)
    k1_min = kK_min / K
    first = max(1, int(np.floor(k_lo / k1_max)))
    last = min(K, int(np.ceil(k_hi / k1_min)))
    return first, last
