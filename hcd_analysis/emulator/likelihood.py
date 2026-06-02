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


def assemble_covariance(cosmic_cov, sigma_Pfilt, w_c, sigma_delta, alpha_hcd,
                        dla_shot_flag, shot_inflate=10.0):
    """Total covariance = cosmic covariance + diag(emulator-error variance).

    The two emulator-error channels propagate through DIFFERENT amplitudes, so
    they cannot share one weights array (the previous bug):
      - the structural baseline ``P_tier_p = Σ_c w_c·P_c^filt`` propagates the
        4-class filtered-P1D error ``sigma_Pfilt`` through ``w_c``;
      - the HCD add-back ``Σ_{c∈HCD} α_c·Δ_c`` propagates the 3-class delta error
        ``sigma_delta`` through ``alpha_hcd``.
    Both add in quadrature.

    Args:
      cosmic_cov:  (K,) variance vector (-> diag) OR full (K,K) covariance matrix
                   (off-diagonal cosmic variance). Detected by ndim.
      sigma_Pfilt: (4,K) per-class filtered-P1D emulator 1-sigma error.
      w_c:         (4,) structural class weights.
      sigma_delta: (3,K) per-class HCD-delta emulator 1-sigma error.
      alpha_hcd:   (3,) per-class HCD incidence amplitudes.
      dla_shot_flag: (K,) bool; high-k DLA shot-limited bins to inflate.
      shot_inflate:  multiplier applied to the emu variance on flagged bins.

    Returns (K,K) covariance. Differentiable / JAX-pure."""
    emu_var = (jnp.einsum("c,ck->k", w_c**2, sigma_Pfilt**2)            # P_filt channel
               + jnp.einsum("c,ck->k", alpha_hcd**2, sigma_delta**2))   # delta channel
    emu_var = jnp.where(dla_shot_flag, emu_var * shot_inflate, emu_var)
    cosmic_cov = jnp.asarray(cosmic_cov)
    # ndim is static at trace time, so a plain python branch keeps this jit-clean.
    cosmic_cov_full = jnp.diag(cosmic_cov) if cosmic_cov.ndim == 1 else cosmic_cov
    return cosmic_cov_full + jnp.diag(emu_var)
