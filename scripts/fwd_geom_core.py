#!/usr/bin/env python
"""PI #18 forward-geometry scan -- ANALYTIC CORE (no forward evaluation).

Preregistered in `2026-08-08-FORWARD-GEOMETRY-SCAN-PREREG.md`. This module contains only
the parts that need NO likelihood evaluation: surface locations, whitening, signed
distances, crossing probabilities and occupancy. The expensive gradient-jump measurement
lives in `fwd_geom_scan.py`.

THE SURFACE IDENTITY (verified numerically to 1.1e-16, prereg v2.1):
    closure_legb.py:3140   alpha_z  = tau0_amp * ((1+z)/(1+z_p))**dtau0
    closure_legb.py:3141   tau0_vec = alpha_z * kim(z),  kim(z) = KIM_AMP*(1+z)**KIM_SLOPE
    likelihood.py:115      alpha    = tau0 / (KIM_AMP*(1+z)**KIM_SLOPE)   ==  alpha_z
so the jnp.interp abscissa IS alpha_z, and in coordinates
    (x, y) = (ln tau0_amp, dtau0)
the knot surface for redshift z_i and knot a_k is EXACTLY the straight line
    K(i,k):   x + c_i * y = ln a_k,      c_i = ln((1+z_i)/(1+z_p)).
"""
from __future__ import annotations

import numpy as np

# ---- frozen constants (prereg §1-§3) --------------------------------------- #
KIM_AMP, KIM_SLOPE = 2.3e-3, 3.65          # likelihood.py:7
TAU0_PIVOT_Z = 3.0                         # meanflux_prior.py:67
# checkpoints/error_vector.npz key `tau0_band_centres` (identical in _xclass)
ALPHA_CENTRES = np.array([0.65555638, 0.83336958, 1.15343333, 1.33124652])
Z_GRID = {                                 # frozen analysis.lock legs.*.z
    "DESI": np.arange(2.2, 4.21, 0.2),
    "eBOSS": np.arange(2.2, 4.61, 0.2),
    "KS": np.arange(2.4, 4.61, 0.2),       # NOT scanned (PI #18 §14); present for tests only
}
TAU0_AMP_RANGE = (0.75, 1.25)              # frozen analysis.lock legs.*.prior
DTAU0_RANGE = (-0.4, 0.25)
D_SAMPLED = {"DESI": 27, "eBOSS": 23}      # sampled-vector dimension (prereg §4)
DELTA_GRID = (1e-4, 1e-3, 1e-2)            # prereg §7; PRIMARY = 1e-3
DELTA_PRIMARY = 1e-3
EPS_SCALE_GRID = (0.5, 1.0, 2.0)           # prereg §8; PRIMARY multiplier = 1.0


def kim(z):
    return KIM_AMP * (1.0 + np.asarray(z, float)) ** KIM_SLOPE


def alpha_of(tau0_amp, dtau0, z, z_pivot=TAU0_PIVOT_Z):
    """alpha_z, the jnp.interp abscissa. closure_legb.py:3140."""
    tau0_amp = np.asarray(tau0_amp, float)[..., None]
    dtau0 = np.asarray(dtau0, float)[..., None]
    return tau0_amp * ((1.0 + np.asarray(z, float)) / (1.0 + z_pivot)) ** dtau0


def c_of_z(z, z_pivot=TAU0_PIVOT_Z):
    """The surface slope coefficient c_i = ln((1+z_i)/(1+z_p)). Zero at z = z_pivot."""
    return np.log((1.0 + np.asarray(z, float)) / (1.0 + z_pivot))


def eps_w(leg):
    """Preregistered parameter-free step-size proxy in the WHITENED coordinate: D^(-1/4).

    NO sampler metadata was persisted (no step size, no mass matrix) for EITHER leg, so the
    step size must be proxied. This is a proxy, not a measurement (prereg §8).
    """
    return float(D_SAMPLED[leg] ** -0.25)


def surfaces(leg):
    """All (i, k) knot surfaces for a leg. Returns (n_surf,) arrays (c, ln_a, z, knot)."""
    z = Z_GRID[leg]
    c = c_of_z(z)
    ln_a = np.log(ALPHA_CENTRES)
    C, A = np.meshgrid(c, ln_a, indexing="ij")
    Zg, K = np.meshgrid(z, np.arange(len(ALPHA_CENTRES)), indexing="ij")
    return C.ravel(), A.ravel(), Zg.ravel(), K.ravel()


# ---- whitening + signed distance ------------------------------------------- #
def plane_whitener(xy):
    """Cholesky whitener for the 2-D mean-flux plane from a mock's own draws.

    Returns (mu, L) with cov = L @ L.T, so z_w = solve(L, x - mu).
    """
    xy = np.asarray(xy, float)
    mu = xy.mean(0)
    C = np.cov(xy.T)
    C = C + 1e-12 * np.eye(2) * max(np.trace(C), 1e-300)
    return mu, np.linalg.cholesky(C)


def signed_distance(xy, c, ln_a, L=None):
    """Signed distance from points to the line  x + c*y = ln_a.

    Raw units if L is None; otherwise WHITENED units, where the correct transformation of a
    hyperplane normal n under x -> L^{-1}(x - mu) is n_w = L.T @ n. The whitened distance is
    (n.x - ln_a) / ||L.T n||, i.e. the raw residual divided by the normal's whitened length.
    Sign is preserved: positive means alpha > a_k.
    """
    xy = np.atleast_2d(np.asarray(xy, float))
    c = np.atleast_1d(np.asarray(c, float))
    ln_a = np.atleast_1d(np.asarray(ln_a, float))
    resid = xy[:, 0:1] + xy[:, 1:2] * c[None, :] - ln_a[None, :]      # (n_pts, n_surf)
    if L is None:
        nrm = np.sqrt(1.0 + c ** 2)
    else:
        n = np.stack([np.ones_like(c), c], axis=0)                    # (2, n_surf)
        nrm = np.linalg.norm(L.T @ n, axis=0)
    return resid / nrm[None, :]


def cross_prob(d_w, eps):
    """P(a single leapfrog step crosses the surface), prereg §9.

    Momentum is standard normal in the whitened metric, so the one-step normal displacement
    is N(0, eps^2) and P(cross) = P(|N(0,eps^2)| > |d_w|) = 2*Phi(-|d_w|/eps) = erfc(|d|/(eps*sqrt2)).
    """
    from scipy.special import erfc
    return erfc(np.abs(np.asarray(d_w, float)) / (np.asarray(eps, float) * np.sqrt(2.0)))


def crossings_between(d_w):
    """Sign changes between consecutive rows -> a surface was crossed.

    UPPER BOUND on per-leapfrog-step crossing: the stored chains are THINNED, so a
    consecutive pair spans many leapfrog steps (prereg §10.3).
    """
    d = np.asarray(d_w, float)
    return (np.sign(d[:-1]) != np.sign(d[1:]))


def xy_of(tau0_amp, dtau0):
    """(x, y) = (ln tau0_amp, dtau0), the coordinates in which the surfaces are straight."""
    return np.column_stack([np.log(np.asarray(tau0_amp, float)),
                            np.asarray(dtau0, float)])
