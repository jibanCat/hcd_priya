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


def delta_c(z):
    """Per-class diagonal-Poisson correction delta_c(z), JAX-pure deg-2 polynomial.
    z: scalar or (...,). Returns (...,4) in class order (clean,LLS,subDLA,DLA)."""
    z = jnp.asarray(z)
    zz = jnp.stack([z**2, z, jnp.ones_like(z)], axis=-1)   # (...,3)
    return zz @ _DELTA_C_COEFFS.T                          # (...,4)


def w_c_corrected(dndx, Xbar, z):
    """Corrected per-sightline class weights: w_M0*(1+delta_c(z)), renormalised to 1.
    dndx: (...,3) HCD incidence (LLS,subDLA,DLA); Xbar: (...); z: (...). Returns (...,4)."""
    w0 = w_c_from_mu(mu_from_dndx(dndx, Xbar))   # (...,4)
    w = w0 * (1.0 + delta_c(z))
    return w / jnp.sum(w, axis=-1, keepdims=True)
