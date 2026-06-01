"""Total-P1D likelihood contract (spec sec.6). Difference form is the DEFAULT."""
from __future__ import annotations
import jax.numpy as jnp


def total_p1d_difference(P_tier_p, w_hcd, A_hcd, delta_hcd):
    """P_obs = P_tier_p + Sum_{c in HCD} w_c*A_c*Delta_c. w_hcd,A_hcd:(3,); delta:(3,K)."""
    return P_tier_p + jnp.einsum("c,c,ck->k", w_hcd, A_hcd, delta_hcd)


def total_p1d_ratio(P_tier_p, w_hcd, A_hcd, ratio_hcd):
    """Alternative toggle: multiplicative per-class ratio form."""
    factor = 1.0 + jnp.einsum("c,c,ck->k", w_hcd, (A_hcd - 1.0), ratio_hcd)
    return P_tier_p * factor


def assemble_covariance(cosmic_var, sigma_emu, w_c, dla_shot_flag, shot_inflate=10.0):
    """Total cov = diag(cosmic variance) + per-class emulator error propagated through w_c
    (quadrature). DLA high-k shot-limited bins inflated so uncertainty isn't understated."""
    emu_var = jnp.einsum("c,ck->k", w_c**2, sigma_emu**2)
    emu_var = jnp.where(dla_shot_flag, emu_var * shot_inflate, emu_var)
    return jnp.diag(cosmic_var + emu_var)
