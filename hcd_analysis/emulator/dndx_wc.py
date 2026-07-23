"""dN/dX -> w_c diagonal telescoping-Poisson map (M0) + JAX-pure delta_c(z).

See hcd_priya_notes/docs/superpowers/2026-05-29-wc-dndx-poisson-coupling.md. JAX-pure: no python
branching on traced values, so the whole dN/dX -> w_c -> P_obs path is one jit.
Class order throughout: (clean, LLS, subDLA, DLA); mu/dN-dX inputs are the 3 HCD
classes in order (LLS, subDLA, DLA).
"""
from __future__ import annotations
import numpy as np
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
# (clean, LLS, subDLA, DLA). See hcd_priya_notes/docs/superpowers/2026-06-01-delta_c-coeffs.md.
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
# hcd_priya_notes/docs/superpowers/2026-06-01-delta_c-coeffs.md. These are the per-class w_c/alpha_c
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
    """DEPRECATED approximate M0-inverse. RETAINED SOLELY TO REPRODUCE PRE-2026-07-22
    FROZEN ARTIFACTS (paper exports built with this approximate inverse). ALL NEW
    CODE MUST USE ``alpha_to_dndx_exact`` instead (readout defect B, 2026-07-22).

    Two measured defects (2026-07-22, on in-range deployed-prior draws):
      1. RENORM ERROR. With apply_delta=True this divides out the per-class
         (1+delta_c) factor but NOT the global 4-class renormalisation of
         ``w_c_corrected``, so it is NOT the inverse of the deployed forward. The
         median relative dN/dX error is ~2e-3 (the "typically sub-percent" claim
         formerly here was FALSE), with catastrophic tails near the simplex edge.
      2. SILENT SATURATION. Outside the valid domain (sum(alpha) >= 1, or infeasible
         telescoping args) the _LOG_FLOOR clip silently SATURATES the affected class
         at mu = -log(1e-12) = 27.631021, i.e. dN/dX = 27.631021/Xbar, instead of
         failing. 0.07-0.5% of in-range deployed-prior draws hit this
         (ensemble-dependent). This corrupted shipped paper artifacts.

    alpha_hcd: (...,3) per-class effective incidence weights for (LLS,subDLA,DLA).
    Xbar: (...); z: scalar or (...). Returns (...,3) dN/dX_c for (LLS,subDLA,DLA).

    Steps:
      1. If apply_delta, divide out the delta_c correction on the HCD slice:
             w = alpha_hcd / (1 + delta_c(z)[...,1:]).
         (Defect 1 above: the 4-class renorm of w_c_corrected is NOT undone here.)
         With apply_delta=False this step is skipped and the telescoping inverse
         below is the EXACT inverse of w_c_from_mu on the HCD classes -- except for
         the clip floor (defect 2).
      2. Invert the telescoping-Poisson (top-down DLA->subDLA->LLS), with the
         _LOG_FLOOR clip (defect 2).
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


def alpha_to_dndx_exact(alpha_hcd, Xbar, z, *, mode="raise"):
    """EXACT inverse of the ``w_c_corrected`` forward on the HCD classes (the readout).

    SUPERSEDES ``alpha_to_dndx`` for ALL new readouts/exports/plots (readout defect B,
    2026-07-22): that map ignores the 4-class renormalisation (median relative error
    ~2e-3 on in-range deployed-prior draws, catastrophic tails) and silently saturates
    at mu = -log(1e-12) = 27.631021 outside its domain. This function inverts the full
    forward exactly, with NO clip floor, and fails loud (or masks) outside the domain.
    Host-side numpy readout: NOT jit/vmap/grad-safe; do not call it from traced code.

    Derivation (one line): the forward is alpha_j = w0_j*(1+d_j)/S with
    S = sum_j w0_j*(1+d_j) and sum_j w0_j = 1 (j over clean + 3 HCD classes,
    d = delta_c(z)); imposing sum_j w0_j = 1 on w0_j = alpha_j*S/(1+d_j) gives
        Z = S = 1 / [ (1 - sum_c a_c)/(1 + d_clean) + sum_c a_c/(1 + d_c) ]
    (sums over the 3 HCD classes), then w0_c = a_c * Z / (1 + d_c), followed by the
    exact top-down telescoping inverse (log1p form, no clip):
        mu_DLA = -log1p(-w0_DLA)
        mu_sub = -log1p(-w0_sub / (1 - w0_DLA))
        mu_LLS = -log1p(-w0_LLS / ((1 - w0_DLA) - w0_sub))
    and dN/dX_c = mu_c / Xbar.

    Domain: all(alpha_hcd >= 0) AND (1 - sum(alpha_hcd)) > 0 strictly (the occupancy
    simplex with a strictly positive clean fraction). Inside it every log1p argument
    is strictly inside (-1, 0], so the result is finite with no floor needed.

    Conditioning: the round-trip error grows like 1/w0_clean near the simplex edge.
    Measured: at 1 - sum(alpha) ~ 1e-12 a 1-ulp input perturbation moves mu_LLS by
    ~3e-6 relative (~2.7e-4 at 1e-14; unbounded at 1e-16). Inputs with
    1 - sum(alpha) < ~1e-12 are therefore domain-VALID but numerically UNRELIABLE.

    alpha_hcd: (...,3) per-class effective incidence weights for (LLS,subDLA,DLA);
    Xbar: (...); z: scalar or (...). numpy or jax arrays accepted (materialised via
    np.asarray; the computation is plain numpy).

    mode="raise" (default): host-side domain check; raises ValueError naming the
      violation class(es) if any element is outside the domain. NOTE: draws from the
      pre-2026-07-22 UNBOUNDED alpha prior can legitimately violate the domain --
      use mode="mask" to read such historical draws.
    mode="mask": returns (dndx, valid) with valid = all(alpha >= 0, axis=-1)
      & (1 - sum(alpha) > 0); invalid entries are NaN in dndx, no exception.

    Returns: (...,3) dN/dX_c for (LLS,subDLA,DLA) [mode="raise"], or the tuple
    (dndx, valid) [mode="mask"].
    """
    if mode not in ("raise", "mask"):
        raise ValueError(f"alpha_to_dndx_exact: unknown mode {mode!r} (use 'raise' or 'mask')")
    a = np.asarray(alpha_hcd, dtype=np.float64)
    Xb = np.asarray(Xbar, dtype=np.float64)
    s = a.sum(axis=-1)
    a_clean = 1.0 - s
    nonneg = np.all(a >= 0.0, axis=-1)
    valid = nonneg & (a_clean > 0.0)

    if mode == "raise" and not np.all(valid):
        n_bad = int(np.size(valid) - np.count_nonzero(valid))
        worst_sum = float(np.max(np.atleast_1d(s)[~np.atleast_1d(valid)]))
        kinds = []
        n_neg = int(np.count_nonzero(~np.atleast_1d(nonneg)))
        if n_neg:
            kinds.append(f"negative alpha in {n_neg} row(s)")
        n_over = int(np.count_nonzero(np.atleast_1d(nonneg & ~(a_clean > 0.0))))
        if n_over:
            kinds.append(f"sum(alpha) >= 1 (outside the occupancy simplex) in {n_over} row(s)")
        raise ValueError(
            f"alpha_to_dndx_exact: {n_bad} input row(s) outside the valid domain "
            f"[all(alpha) >= 0 and 1 - sum(alpha) > 0]: " + "; ".join(kinds) +
            f". Worst sum(alpha) = {worst_sum!r}. Draws from the pre-2026-07-22 "
            f"unbounded alpha prior can legitimately violate this domain; use "
            f"mode='mask' to read such historical draws (invalid entries -> NaN).")

    # Mask invalid rows to NaN BEFORE the logs (NaN propagates silently -- no
    # log-of-negative warnings). In raise mode everything is valid at this point.
    a = np.where(valid[..., None], a, np.nan)
    a_clean = np.where(valid, a_clean, np.nan)

    d = np.asarray(delta_c(z), dtype=np.float64)                      # (...,4)
    Z = 1.0 / (a_clean / (1.0 + d[..., 0])
               + np.sum(a / (1.0 + d[..., 1:]), axis=-1))
    w0 = a * Z[..., None] / (1.0 + d[..., 1:])                        # (...,3)
    w_LLS, w_sub, w_DLA = w0[..., 0], w0[..., 1], w0[..., 2]

    mu_DLA = -np.log1p(-w_DLA)
    mu_sub = -np.log1p(-(w_sub / (1.0 - w_DLA)))
    mu_LLS = -np.log1p(-(w_LLS / ((1.0 - w_DLA) - w_sub)))
    dndx = np.stack([mu_LLS, mu_sub, mu_DLA], axis=-1) / Xb[..., None]
    if mode == "mask":
        return dndx, np.broadcast_to(valid, dndx.shape[:-1]).copy()
    return dndx
