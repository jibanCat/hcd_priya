"""dN/dX -> w_c diagonal telescoping-Poisson map (M0) + JAX-pure delta_c(z).

See docs/superpowers/2026-05-29-wc-dndx-poisson-coupling.md. JAX-pure: no python
branching on traced values, so the whole dN/dX -> w_c -> P_obs path is one jit.
Class order throughout: (clean, LLS, subDLA, DLA); mu/dN-dX inputs are the 3 HCD
classes in order (LLS, subDLA, DLA).
"""
from __future__ import annotations
import jax.numpy as jnp


def w_c_from_mu(mu):
    """mu: (...,3) mean absorbers/sightline for (LLS, subDLA, DLA). Returns (...,4)."""
    mu_LLS, mu_sub, mu_DLA = mu[..., 0], mu[..., 1], mu[..., 2]
    w_DLA = 1.0 - jnp.exp(-mu_DLA)
    w_sub = (1.0 - jnp.exp(-mu_sub)) * jnp.exp(-mu_DLA)
    w_LLS = (1.0 - jnp.exp(-mu_LLS)) * jnp.exp(-(mu_sub + mu_DLA))
    w_clean = jnp.exp(-(mu_LLS + mu_sub + mu_DLA))
    return jnp.stack([w_clean, w_LLS, w_sub, w_DLA], axis=-1)


def mu_from_dndx(dndx, Xbar):
    """dndx: (...,3) per-class incidence; Xbar: (...) mean path per sightline."""
    return dndx * Xbar[..., None]


# Frozen delta_c(z) deg-2 coeffs from scripts/calibrate_delta_c.py (89 LF shards,
# 1060 snap-blocks, 60 sims). np.polyval order [a2,a1,a0]; class order
# (clean, LLS, subDLA, DLA). See docs/superpowers/2026-06-01-delta_c-coeffs.md.
_DELTA_C_COEFFS = jnp.array([
    [-0.0044601756018632,    0.02941370276821661,  -0.04147804040644924],   # clean
    [-7.391610723256435e-06, 0.008526578464137916, -0.03592195195833433],   # LLS
    [-0.0006304124952415818, 0.009001921148434061, -0.03672596701410354],   # subDLA
    [-0.0002204519439413697, 0.0033098470327118175,-0.0245078526302848],    # DLA
])


# Calibration z-range of the delta_c fit. The deg-2 polynomial extrapolates as an
# unbounded quadratic outside this; clamp z to the range so HMC / tau0-edge queries
# get the boundary value, not a divergent correction (CS finding I3).
Z_FIT_LO, Z_FIT_HI = 2.0, 5.4

# Per-class fit-residual std of delta_c (clean,LLS,subDLA,DLA), from
# docs/superpowers/2026-06-01-delta_c-coeffs.md. These are the per-class w_c/alpha_c
# PRIOR WIDTHS: the irreducible cosmology-dependent delta_c scatter at fixed z that the
# deg-2 z-trend fit cannot remove.
DELTA_C_RESID_STD = jnp.array([0.0093, 0.0059, 0.0032, 0.0016])


def delta_c(z):
    """Per-class diagonal-Poisson correction delta_c(z), JAX-pure deg-2 polynomial.
    z: scalar or (...,). Returns (...,4) in class order (clean,LLS,subDLA,DLA).
    z is clamped to [Z_FIT_LO, Z_FIT_HI] so the fit is never extrapolated."""
    z = jnp.clip(jnp.asarray(z), Z_FIT_LO, Z_FIT_HI)
    zz = jnp.stack([z**2, z, jnp.ones_like(z)], axis=-1)   # (...,3)
    return zz @ _DELTA_C_COEFFS.T                          # (...,4)


def w_c_corrected(dndx, Xbar, z):
    """Corrected per-sightline class weights: w_M0*(1+delta_c(z)), renormalised to 1.
    dndx: (...,3) HCD incidence (LLS,subDLA,DLA); Xbar: (...); z: (...). Returns (...,4)."""
    w0 = w_c_from_mu(mu_from_dndx(dndx, Xbar))   # (...,4)
    w = w0 * (1.0 + delta_c(z))
    return w / jnp.sum(w, axis=-1, keepdims=True)


# --- Task 3: alpha_c(z) incidence parametrization + analytic M0-inverse ----------
_LOG_FLOOR = 1e-12   # positive floor on (1 - w...) so log stays finite & differentiable


def dndx_powerlaw(z, A, gamma):
    """PW14 incidence law dN/dX_c(z) = A_c * (1+z)**gamma_c, per HCD class.
    z: scalar or (...); A, gamma: scalar or (3,) for (LLS,subDLA,DLA).
    Returns (...,3) dN/dX_c for (LLS, subDLA, DLA)."""
    z = jnp.asarray(z)
    A = jnp.asarray(A); gamma = jnp.asarray(gamma)
    return A * (1.0 + z[..., None]) ** gamma


def alpha_from_dndx_law(A, gamma, Xbar, z):
    """Forward alpha_c(z) parametrized by (A_c, gamma_c): the 3 HCD entries of
    w_c_corrected applied to the PW14 incidence law.
    A, gamma: scalar or (3,); Xbar: (...); z: scalar or (...). Returns (...,3)
    for (LLS, subDLA, DLA)."""
    dndx = dndx_powerlaw(z, A, gamma)                  # (...,3)
    return w_c_corrected(dndx, Xbar, z)[..., 1:]       # drop clean -> HCD (LLS,subDLA,DLA)


def alpha_to_dndx(alpha_hcd, Xbar, z, apply_delta=True):
    """Analytic M0-inverse (the readout): effective dN/dX_c(z) from per-class alpha_c.

    alpha_hcd: (...,3) per-class effective incidence weights for (LLS,subDLA,DLA).
    Xbar: (...); z: scalar or (...). Returns (...,3) dN/dX_c for (LLS,subDLA,DLA).

    Steps:
      1. If apply_delta, divide out the delta_c correction on the HCD slice:
             w = alpha_hcd / (1 + delta_c(z)[...,1:]).
         RENORM CAVEAT: w_c_corrected renormalises across ALL 4 classes (clean+3 HCD),
         so this division inverts the *correction factor* but not the global renorm.
         In the small-HCD-fraction regime the renorm factor ~= 1, so the inverse is
         exact up to that (typically sub-percent) factor. With apply_delta=False this
         step is skipped and the telescoping inverse below is the EXACT inverse of
         w_c_from_mu on the HCD classes.
      2. Invert the telescoping-Poisson (exact, top-down DLA->subDLA->LLS).
      3. dN/dX_c = mu_c / Xbar.
    """
    w = jnp.asarray(alpha_hcd)
    if apply_delta:
        w = w / (1.0 + delta_c(z)[..., 1:])
    w_LLS, w_sub, w_DLA = w[..., 0], w[..., 1], w[..., 2]

    # Exact top-down inverse of w_c_from_mu. Clip the (1 - ...) args to a tiny positive
    # floor so log stays finite for valid inputs (keeps it differentiable).
    mu_DLA = -jnp.log(jnp.clip(1.0 - w_DLA, _LOG_FLOOR, None))
    mu_sub = -jnp.log(jnp.clip(1.0 - w_sub / jnp.clip(1.0 - w_DLA, _LOG_FLOOR, None),
                               _LOG_FLOOR, None))
    mu_LLS = -jnp.log(jnp.clip(1.0 - w_LLS / jnp.clip((1.0 - w_DLA) - w_sub, _LOG_FLOOR, None),
                               _LOG_FLOOR, None))

    mu = jnp.stack([mu_LLS, mu_sub, mu_DLA], axis=-1)   # (...,3)
    return mu / jnp.asarray(Xbar)[..., None]
