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
