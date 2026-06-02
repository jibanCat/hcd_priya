"""Total-P1D likelihood contract (spec sec.6). Difference form is the DEFAULT."""
from __future__ import annotations
import jax.numpy as jnp


def total_p1d_difference(P_tier_p, alpha_hcd, delta_hcd):
    """P_obs = P_tier_p + Sum_{c in HCD} alpha_c * Delta_c. alpha_hcd:(...,3); delta:(...,3,K).

    alpha_c is the SINGLE free per-class effective residual (post-masking) incidence
    amplitude (spec sec.6; field-standard, cf. Rogers&Bird 2018 / DESI DR1 / PRIYA 2025).
    HCD classes are (LLS, subDLA, DLA). Difference form is the DEFAULT.
    """
    return P_tier_p + jnp.einsum("...c,...ck->...k", alpha_hcd, delta_hcd)


def total_p1d_ratio(P_tier_p, alpha_hcd, ratio_hcd):
    """Multiplicative toggle: P_obs = P_tier_p * (1 + Sum_c alpha_c * R_c).

    alpha_c is the same single per-class amplitude as in the difference form.
    """
    return P_tier_p * (1.0 + jnp.einsum("...c,...ck->...k", alpha_hcd, ratio_hcd))


def assemble_covariance(cosmic_var, sigma_emu, weights, dla_shot_flag, shot_inflate=10.0):
    """Total cov = diag(cosmic variance) + per-class emulator error propagated through w_c
    (quadrature). DLA high-k shot-limited bins inflated so uncertainty isn't understated."""
    emu_var = jnp.einsum("c,ck->k", weights**2, sigma_emu**2)
    emu_var = jnp.where(dla_shot_flag, emu_var * shot_inflate, emu_var)
    return jnp.diag(cosmic_var + emu_var)
