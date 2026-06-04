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
                        P_filt, delta_scale, dla_shot_flag, shot_inflate=10.0):
    """Total covariance = cosmic covariance + diag(emulator-error variance).

    UNITS CONTRACT (the bug this signature fixes):
      ``sigma_Pfilt`` and ``sigma_delta`` are FRACTIONAL emulator errors
      (δP/P and δΔ/Δ, the RMS fractional residuals from
      ``run_loso_sweep.fold_resid_neff``). A fractional error is dimensionless;
      it CANNOT be added to ``cosmic_cov`` (an absolute P1D variance) as-is.
      The absolute scale is supplied SEPARATELY by ``P_filt`` (linear per-class
      P1D) and ``delta_scale`` (|Δ_c|, the linear per-class HCD add-back). The
      absolute 1-σ error on class c is therefore σ_frac,c·P_c (P_filt channel)
      and σ_frac,c·|Δ_c| (delta channel); their variances enter below.

    The two emulator-error channels propagate through DIFFERENT amplitudes, so
    they cannot share one weights array (the previous bug):
      - the structural baseline ``P_tier_p = Σ_c w_c·P_c^filt`` propagates the
        4-class filtered-P1D error through ``w_c``. The absolute variance from
        class c is ``(w_c · σ_frac,c · P_c)²`` -> ``Σ_c w_c²·σ_frac,c²·P_c²``;
      - the HCD add-back ``Σ_{c∈HCD} α_c·Δ_c`` propagates the 3-class delta error
        through ``alpha_hcd``. The absolute variance from class c is
        ``(α_c · σ_frac,c · |Δ_c|)²`` -> ``Σ_c α_c²·σ_frac,c²·Δ_c²``.
    Both add in quadrature.

    Args:
      cosmic_cov:  (K,) variance vector (-> diag) OR full (K,K) covariance matrix
                   (off-diagonal cosmic variance). Detected by ndim.
      sigma_Pfilt: (4,K) per-class filtered-P1D emulator FRACTIONAL error (δP/P).
      w_c:         (4,) structural class weights.
      sigma_delta: (3,K) per-class HCD-delta emulator FRACTIONAL error (δΔ/Δ).
      alpha_hcd:   (3,) per-class HCD incidence amplitudes.
      P_filt:      (4,K) predicted LINEAR per-class P1D — supplies the absolute
                   scale for the fractional ``sigma_Pfilt``.
      delta_scale: (3,K) predicted |Δ_c| (linear per-class HCD add-back) —
                   supplies the absolute scale for the fractional ``sigma_delta``.
      dla_shot_flag: (K,) bool; high-k DLA shot-limited bins to inflate.
      shot_inflate:  multiplier applied to the emu variance on flagged bins.

    DATA-RANGE CONTRACT: the emulator error vector (``sigma_Pfilt``/``sigma_delta``)
    is built DATA-RANGE-RESTRICTED upstream (``run_loso_sweep.fold_resid_neff``,
    ``data.datarange_mask``: z∈[2.2,4.6], k≥1e-3) so the emu-error budget only covers
    modes the DESI data constrain. Out-of-range (z,k) cells therefore arrive as NaN
    (no kept residuals). This function is NaN-SAFE: a NaN σ contributes ZERO emu
    variance for that (class,k) — the data does not constrain it, so it adds no
    emulator-error penalty. The cosmic covariance still governs those modes.

    Returns (K,K) covariance. Differentiable / JAX-pure."""
    # σ_* are FRACTIONAL; P_filt / delta_scale supply the absolute scale so the
    # emu variance is in the same (absolute P1D)² units as cosmic_cov. NaN-safe:
    # data-range-restricted out-of-range cells arrive NaN -> contribute 0 variance.
    sigma_Pfilt = jnp.nan_to_num(jnp.asarray(sigma_Pfilt), nan=0.0)
    sigma_delta = jnp.nan_to_num(jnp.asarray(sigma_delta), nan=0.0)
    emu_var = (jnp.einsum("c,ck,ck->k", w_c**2, sigma_Pfilt**2, P_filt**2)         # P_filt channel
               + jnp.einsum("c,ck,ck->k", alpha_hcd**2, sigma_delta**2,           # delta channel
                            delta_scale**2))
    emu_var = jnp.where(dla_shot_flag, emu_var * shot_inflate, emu_var)
    cosmic_cov = jnp.asarray(cosmic_cov)
    # ndim is static at trace time, so a plain python branch keeps this jit-clean.
    cosmic_cov_full = jnp.diag(cosmic_cov) if cosmic_cov.ndim == 1 else cosmic_cov
    return cosmic_cov_full + jnp.diag(emu_var)
