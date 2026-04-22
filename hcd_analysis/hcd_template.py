"""
Rogers+2018 HCD contamination template for P1D (arXiv:1706.08532).

This module implements the four-parameter HCD correction

    P_total(k, z) / P_forest(k, z) = 1 + Σ_i  α_i · f_z(z) · g_i(k, z)

with

    f_z(z)   = ((1+z) / (1+z_0))^{-3.55},   z_0 = 2,
    g_i(k,z) = (a_i(z) · exp(b_i(z) · k) − 1)^{-2},
    a_i(z)   = a_i^0 · ((1+z)/(1+z_0))^{a_i^1},
    b_i(z)   = b_i^0 · ((1+z)/(1+z_0))^{b_i^1},

and i runs over (LLS, Sub-DLA, Small-DLA, Large-DLA). Coefficient tables
from Rogers, Bird, Peiris, Pontzen (2018), *MNRAS* 476, 3716 table 3.

k-CONVENTION
------------
The template uses k in the *angular*-frequency convention PRIYA and
Rogers both adopt:  k [rad·s/km] = 2π · k_cyclic [s/km].

Use `template_factor_from_cyclic_k(k_cyc, ...)` if your k is the cyclic
`rfftfreq`-style k that our `P1DAccumulator` produces.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

# Rogers+2018 table 3 coefficients.
# Parameter order: [LLS, Sub-DLA, Small-DLA, Large-DLA]
_A0 = np.array([2.2001, 1.5083, 1.1415, 0.8633])
_A1 = np.array([0.0134, 0.0994, 0.0937, 0.2943])
_B0 = np.array([36.449, 81.388, 162.95, 429.58])
_B1 = np.array([-0.0674, -0.2287, 0.0126, -0.4964])

_CLASS_LABELS = ("LLS", "Sub-DLA", "Small-DLA", "Large-DLA")

# Pivot redshift for the class-weight scaling (arXiv:1706.08532, eq. 4).
_Z_PIVOT = 2.0

# Fixed power-law index of the z-scaling of the per-class amplitude.
_Z_SCALING_INDEX = -3.55


def _az_bz(z: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return the z-evolved (a, b) coefficient vectors at redshift `z`.

    The arrays are indexed in the class order
    (LLS, Sub-DLA, Small-DLA, Large-DLA).
    """
    zfac = (1.0 + z) / (1.0 + _Z_PIVOT)
    a_z = _A0 * zfac ** _A1
    b_z = _B0 * zfac ** _B1
    return a_z, b_z


def template_contributions(
    k_angular: np.ndarray,
    z: float,
    alpha: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Per-class contribution to the HCD correction factor.

    Parameters
    ----------
    k_angular : (n_k,) k in angular units [rad·s/km] (PRIYA / Rogers convention).
    z         : redshift scalar.
    alpha     : length-4 amplitude vector [α_LLS, α_Sub, α_Small, α_Large].

    Returns
    -------
    dict with keys "LLS", "Sub-DLA", "Small-DLA", "Large-DLA"; values are
    arrays of shape `k_angular.shape` giving `(1 + α_i · f_z · g_i)` per class
    (so that 1.0 = no contribution, and the total factor equals
    1 + Σ_i [contribution_i - 1]).
    """
    alpha = np.asarray(alpha, dtype=np.float64).reshape(4)
    k = np.asarray(k_angular, dtype=np.float64)
    a_z, b_z = _az_bz(z)
    zfac = (1.0 + z) / (1.0 + _Z_PIVOT)
    z_weight = zfac ** _Z_SCALING_INDEX

    contribs = {}
    for i, name in enumerate(_CLASS_LABELS):
        g = (a_z[i] * np.exp(b_z[i] * k) - 1.0) ** -2
        contribs[name] = 1.0 + alpha[i] * z_weight * g
    return contribs


def template_factor(
    k_angular: np.ndarray,
    z: float,
    alpha: np.ndarray,
) -> np.ndarray:
    """
    Total Rogers+2018 HCD correction factor: `P_total / P_forest` at each k.

    Equivalent to `1 + Σ_i (template_contributions - 1)`.
    """
    c = template_contributions(k_angular, z, alpha)
    factor = np.ones_like(np.asarray(k_angular, dtype=np.float64))
    for name in _CLASS_LABELS:
        factor += (c[name] - 1.0)
    return factor


def template_factor_from_cyclic_k(
    k_cyclic: np.ndarray,
    z: float,
    alpha: np.ndarray,
) -> np.ndarray:
    """Convenience wrapper if your k is the cyclic rfftfreq convention (s/km)."""
    return template_factor(2.0 * np.pi * np.asarray(k_cyclic), z, alpha)


# ---------------------------------------------------------------------------
# α fitting
# ---------------------------------------------------------------------------

def fit_alpha(
    k_angular: np.ndarray,
    P_total: np.ndarray,
    P_forest: np.ndarray,
    z: float,
    alpha0: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    bounds: Tuple[np.ndarray, np.ndarray] | None = None,
    only_dlas: bool = False,
) -> Dict[str, object]:
    """
    Fit α = (α_LLS, α_Sub, α_Small, α_Large) so that

        P_total / P_forest  ≈  template_factor(k, z, α).

    Parameters
    ----------
    k_angular : (n_k,) k values [rad·s/km].
    P_total   : (n_k,) measured P1D on the full (HCD-containing) sample.
    P_forest  : (n_k,) reference P1D on the HCD-free sample (or post-PRIYA-mask).
    z         : redshift scalar.
    alpha0    : initial guess; default 0.1 for each component (Rogers prior).
    weights   : per-k weights; default uniform.
    bounds    : (lower, upper) length-4 arrays to restrict each α_i.
                Default: (0, +∞) — HCDs only add power.
    only_dlas : if True, fix α_LLS = α_Sub = 0 and fit just Small-DLA / Large-DLA.

    Returns
    -------
    dict with keys
      "alpha"      : best-fit vector (shape (4,))
      "alpha_err"  : 1-σ estimate from least-squares covariance (if available)
      "k"          : input k_angular
      "ratio_obs"  : P_total / P_forest
      "ratio_fit"  : template_factor(k, z, α_best)
      "residual"   : ratio_obs - ratio_fit
      "chi2"       : weighted sum of squared residuals
    """
    from scipy.optimize import least_squares

    k = np.asarray(k_angular, dtype=np.float64)
    r_obs = np.asarray(P_total, dtype=np.float64) / np.asarray(P_forest, dtype=np.float64)
    if weights is None:
        weights = np.ones_like(k)
    if alpha0 is None:
        alpha0 = np.full(4, 0.1, dtype=np.float64)
    else:
        alpha0 = np.asarray(alpha0, dtype=np.float64).copy()
    if bounds is None:
        lo = np.zeros(4); hi = np.full(4, 10.0)
    else:
        lo, hi = np.asarray(bounds[0]), np.asarray(bounds[1])

    if only_dlas:
        # Fit only indices 2, 3; freeze 0, 1 at 0.
        fixed_mask = np.array([True, True, False, False])
        free_i = np.where(~fixed_mask)[0]

        def residual(free_alpha):
            alpha = alpha0.copy()
            alpha[free_i] = free_alpha
            alpha[fixed_mask] = 0.0
            return weights * (template_factor(k, z, alpha) - r_obs)

        x0 = alpha0[free_i]
        lo_f, hi_f = lo[free_i], hi[free_i]
        res = least_squares(residual, x0, bounds=(lo_f, hi_f))
        alpha_best = alpha0.copy()
        alpha_best[free_i] = res.x
        alpha_best[fixed_mask] = 0.0
    else:
        def residual(alpha):
            return weights * (template_factor(k, z, alpha) - r_obs)
        res = least_squares(residual, alpha0, bounds=(lo, hi))
        alpha_best = res.x

    ratio_fit = template_factor(k, z, alpha_best)
    residual_arr = r_obs - ratio_fit

    # Covariance estimate from scipy residual Jacobian
    try:
        J = res.jac
        dof = max(1, len(r_obs) - len(res.x))
        sigma2 = (np.sum(residual_arr ** 2) / dof)
        cov = sigma2 * np.linalg.inv(J.T @ J)
        alpha_err = np.sqrt(np.abs(np.diag(cov)))
        # pad to length 4 if only_dlas
        if len(alpha_err) != 4:
            full = np.zeros(4)
            full[free_i] = alpha_err
            alpha_err = full
    except Exception:
        alpha_err = np.full(4, np.nan)

    return {
        "alpha": alpha_best,
        "alpha_err": alpha_err,
        "k": k,
        "ratio_obs": r_obs,
        "ratio_fit": ratio_fit,
        "residual": residual_arr,
        "chi2": float(np.sum((weights * residual_arr) ** 2)),
    }


# ---------------------------------------------------------------------------
# Apply the correction (template subtraction)
# ---------------------------------------------------------------------------

def correct_p1d(
    k_angular: np.ndarray,
    P_observed: np.ndarray,
    z: float,
    alpha: np.ndarray,
) -> np.ndarray:
    """
    Apply the Rogers template to divide out HCD contamination:

        P_forest_est(k) = P_observed(k) / template_factor(k, z, alpha)

    This assumes `P_observed` IS the HCD-contaminated P1D (e.g. the "all"
    variant without any spatial mask, or the "no_DLA_priya" variant which
    still contains LLS/Sub-DLA residuals).
    """
    return np.asarray(P_observed) / template_factor(k_angular, z, alpha)
