"""Total-P1D likelihood contract (spec sec.6). Difference form is the DEFAULT."""
from __future__ import annotations
import jax
import jax.numpy as jnp

# Kim2013 central mean-flux curve (== data.KIM_AMP/SLOPE) for the τ₀→α ladder mapping.
_KIM_AMP, _KIM_SLOPE = 2.3e-3, 3.65


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
                        P_filt, delta_scale, dla_shot_flag, shot_inflate=10.0,
                        cemu_inflate=1.0):
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
    # CONSERVATIVE INFLATION (design §1.3d/e): the rank-≤12 SVD basis makes the emulator
    # error strongly k-CORRELATED, so a purely DIAGONAL C_emu under-states the
    # k-integrated uncertainty and OVER-tightens the posterior (anti-conservative for SBC
    # coverage; cf. Rogers+2019). ``cemu_inflate`` is the documented scalar safeguard
    # (the closure/SBC, T4, brackets coverage between this diagonal and PRIYA's rank-1
    # fully-correlated outer(σ,σ); the k×k shrinkage upgrade is gated on that χ²).
    emu_var = emu_var * cemu_inflate
    cosmic_cov = jnp.asarray(cosmic_cov)
    # ndim is static at trace time, so a plain python branch keeps this jit-clean.
    cosmic_cov_full = jnp.diag(cosmic_cov) if cosmic_cov.ndim == 1 else cosmic_cov
    return cosmic_cov_full + jnp.diag(emu_var)


def sigma_at_tau0(sigma_zb, alpha_centres, z, tau0):
    """τ₀-interpolate the τ₀-BANDED fractional error to a sampled τ₀ (Phase-C §1.3b).

    The error vector is τ₀-aware: ``sigma`` is (C,K,Zb,Tb) over τ₀-LADDER bands whose
    centres are stored in the z-independent ladder coordinate α=τ₀/Kim(z)
    (``data.make_tau0_bands``). For a data bin at fixed ``z`` with SAMPLED mean-flux
    ``tau0``, select that z-band's slice ``sigma_zb`` (C,K,Tb), map τ₀→α, and linearly
    interpolate over the band centres -> (C,K). Differentiable in τ₀ (piecewise-linear;
    finite gradient a.e., fine for NUTS); ``jnp.interp`` clamps flat outside the centres
    so the ladder extremes are safe.

    Args:
      sigma_zb:      (C,K,Tb) the error vector at the data z-band.
      alpha_centres: (Tb,) ascending τ₀-band centres in α units.
      z:             scalar data redshift (fixed). tau0: scalar sampled mean-flux.
    Returns (C,K) fractional error at the sampled τ₀.
    """
    alpha = jnp.asarray(tau0) / (_KIM_AMP * (1.0 + jnp.asarray(z)) ** _KIM_SLOPE)
    # CRITICAL (jax-trap): sanitize the interp YDATA *before* jnp.interp, not the result
    # after. Interpolating between NaN y-points (an all-NaN-over-Tb (c,k) row — the high-k
    # Nyquist / out-of-range cells) gives value=NaN (sanitizable) but slope=NaN, which a
    # downstream nan_to_num does NOT fix -> jax.grad wrt τ₀ returns NaN and NUTS dies.
    sig = jnp.nan_to_num(jnp.asarray(sigma_zb), nan=0.0)
    interp1 = lambda s_tb: jnp.interp(alpha, jnp.asarray(alpha_centres), s_tb)
    return jax.vmap(jax.vmap(interp1))(sig)                        # (C,K)


def rho_at_tau0(rho_zb, alpha_centres, z, tau0):
    """τ₀-interpolate the τ₀-BANDED CROSS-CLASS block to a sampled τ₀ (the 4×4 analog of
    ``sigma_at_tau0``).

    The cross-class error block is τ₀-aware: ``rho_zb`` is (C,C,K,Tb) over τ₀-LADDER bands
    whose centres are stored in the z-independent ladder coordinate α=τ₀/Kim(z)
    (``data.make_tau0_bands``). For a data bin at fixed ``z`` with SAMPLED mean-flux
    ``tau0``, map τ₀→α and linearly interpolate the (C,C,K) block over the band centres ->
    (C,C,K). Differentiable in τ₀ (piecewise-linear; finite gradient a.e., fine for NUTS);
    ``jnp.interp`` clamps flat outside the centres so the ladder extremes are safe.

    Mirrors ``sigma_at_tau0`` EXACTLY (same NaN-guard, same interp), just on the 4×4×K block
    instead of the 4×K diagonal: the YDATA (over Tb) is sanitised BEFORE ``jnp.interp`` so an
    all-NaN-over-Tb (c,c',k) row (the high-k Nyquist / out-of-range cells) gives value=0 AND
    slope=0 — NOT a NaN slope that would poison ∂/∂τ₀ and kill NUTS.

    Args:
      rho_zb:        (C,C,K,Tb) the cross-class block at the data z-band.
      alpha_centres: (Tb,) ascending τ₀-band centres in α units.
      z:             scalar data redshift (fixed). tau0: scalar sampled mean-flux.
    Returns (C,C,K) cross-class block at the sampled τ₀.
    """
    alpha = jnp.asarray(tau0) / (_KIM_AMP * (1.0 + jnp.asarray(z)) ** _KIM_SLOPE)
    rho = jnp.nan_to_num(jnp.asarray(rho_zb), nan=0.0)            # (C,C,K,Tb)
    interp1 = lambda r_tb: jnp.interp(alpha, jnp.asarray(alpha_centres), r_tb)
    # vmap over the (C,C,K) leading axes; interp over the trailing Tb axis.
    return jax.vmap(jax.vmap(jax.vmap(interp1)))(rho)            # (C,C,K)


def gaussian_loglik(r, C, jitter=1e-10):
    """−½ rᵀC⁻¹r − ½ logdet C  (the logdet-bearing Gaussian, design §1.3c).

    C (K,K) SPD, r (K,). Cholesky for BOTH the quadratic form and the logdet so the
    term is exact and JAX differentiates through C's dependence on (θ,τ₀) — the logdet
    is NOT constant once C depends on sampled params and MUST be carried (omitting it
    biases τ₀ toward larger-σ regions). Returns a scalar.

    ``jitter`` adds a small SPD floor (relative to ⟨diag C⟩) BEFORE the factorization:
    ``jnp.linalg.cholesky`` RETURNS NaN (does not raise) on a non-SPD / zero-diagonal C,
    silently poisoning the value and the gradient. A zero diagonal can arise when a bin
    has zero cosmic variance AND zero emulator variance (a NaN-zeroed σ row); a
    finite-mock cosmic_cov can also be marginally indefinite. The jitter removes both.
    """
    r = jnp.asarray(r)
    C = jnp.asarray(C)
    K = C.shape[-1]
    C = C + (jitter * jnp.mean(jnp.diag(C))) * jnp.eye(K)
    L = jnp.linalg.cholesky(C)
    sol = jax.scipy.linalg.cho_solve((L, True), r)
    half_logdet = jnp.sum(jnp.log(jnp.diag(L)))     # ½ logdet C = Σ log diag(L)
    return -0.5 * (r @ sol) - half_logdet
