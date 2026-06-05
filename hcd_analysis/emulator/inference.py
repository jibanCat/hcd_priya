"""Phase-C differentiable likelihood driver — the object HMC (numpyro/blackjax) and the
Cobaya adapter both wrap (design §1.4).

Mirrors PRIYA's (Ho 2023/2024) parameter contract so the cosmology community drives it
as before: the 9 cosmo/IGM params (``data.PARAM_LIMITS`` order, identical to PRIYA's
``coarse_grid``), mean-flux as per-z ``tau0`` (PRIYA's ``mean_flux="per_z"``), and HCD
nuisance as per-class ``alpha_hcd`` (LLS, subDLA, DLA — the single-amplitude analog of
PRIYA's ``a_lls/a_dla``). Wires ``predict.predict_P_obs`` + the τ₀-aware per-class C_emu
(``likelihood.sigma_at_tau0`` interp, then ``emu_var = Σ_c coef_c²·σ_c²·P_c²`` assembled
INLINE in ``log_lik_single_z`` — NOT the legacy ``likelihood.assemble_covariance``, which is
superseded) + the logdet Gaussian (``likelihood.gaussian_loglik``) + a SMOOTH bounded prior
(replacing PRIYA's ``-inf`` wall so NUTS gets finite gradients).

Everything is JAX-pure and differentiable in (θ_unit, tau0, alpha_hcd).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .predict import predict_P_filt
from .likelihood import sigma_at_tau0, gaussian_loglik

# PRIYA / coarse_grid order — the names the Cobaya/numpyro adapters expose.
PARAM_NAMES = ("ns", "Ap", "herei", "heref", "alphaq", "hub", "omegamh2",
               "hireionz", "bhfeedback")


def unit_box_logprior(theta_unit, sharpness=1e3):
    """Smooth soft-wall log-prior on the unit cube [0,1]^9 (design: replaces PRIYA's
    ``return -inf`` hard wall, which has zero/inf gradient and breaks NUTS).

    0 inside the box; a smooth quadratic penalty outside so the gradient stays finite
    everywhere (HMC can be pushed back in). ``sharpness`` sets the wall steepness.
    """
    t = jnp.asarray(theta_unit)
    below = jnp.clip(-t, 0.0, None)            # >0 where t<0
    above = jnp.clip(t - 1.0, 0.0, None)       # >0 where t>1
    return -sharpness * jnp.sum(below ** 2 + above ** 2)


def meanflux_logprior(tau0, mu, sigma):
    """Per-z Gaussian prior on τ₀(z) at the measurement width (design §4); the
    τ_eff–cosmology degeneracy can bias cosmology, so anchor τ₀ on the observed ⟨F⟩(z).
    ``tau0``/``mu``/``sigma`` broadcast over z."""
    return -0.5 * jnp.sum(((jnp.asarray(tau0) - jnp.asarray(mu)) / jnp.asarray(sigma)) ** 2)


def gaussian_logprior(value, mu, sigma):
    """Generic additive Gaussian prior (mirror PRIYA's optional hub/omega/bhfeedback)."""
    return -0.5 * (((jnp.asarray(value) - mu) / sigma) ** 2)


# HCD incidence priors — literature-calibrated (2026-06-04 Lyα agent; sources: O'Meara+2013
# / Prochaska+2010 / Fumagalli+2013 (LLS), Zafar+2013 (subDLA), Prochaska&Wolfe2009 /
# Noterdaeme+2012 (DLA)). Fractional widths σ/μ per class; LLS TIGHT (cosmology-degenerate,
# DESI DR1). DLA centered on the residual post-masking fraction (masking ~70% complete).
HCD_PRIOR_FRAC_SIGMA = (0.15, 0.40, 0.10)   # σ/μ: LLS TIGHT (cosmology-degenerate); subDLA
#   BROAD (poor measurement — the Zafar-vs-O'Meara factor-2); DLA tight at z≤3.5, widened
#   above (see HCD_DLA_Z_RELIABLE).
HCD_DLA_RESIDUAL_FRAC = 0.30                 # DLA residual incidence ≈ 0.30 × data incidence
HCD_DLA_Z_RELIABLE = 3.5                     # DLA dN/dX unreliable beyond this z → widen σ_DLA
HCD_Z_PIVOT = 3.0
# PRIYA-sim-vs-observed dN/dX offset (literature / PRIYA-sim) per class, as a POWER-LAW in
# (1+z) — mirroring the τ₀ Kim-curve+slope model. PRIYA does NOT match the data: the
# observed dN/dX evolves FASTER with z than the sim (γ_lit > γ_sim), so a single
# z-independent ratio mis-centers the prior at the z edges. (lit/sim)@z_pivot + the slope
# d ln(lit/sim)/d ln(1+z), fit by scripts/plot_dndx_vs_literature.py (PRIYA vs
# Prochaska&Wolfe09 / Zafar13 / O'Meara13):
HCD_LIT_OVER_SIM = (1.06, 0.76, 1.34)        # (LLS, subDLA, DLA): data/sim at z_pivot=3.0
# DLA slope deliberately WEAK (0.4, the conservative Ω_DLA∝(1+z)^0.4): the raw fit (+1.08,
# or +1.90 on z≤3.5) is dominated by z>3.5 DLA dN/dX that the literature does not measure
# reliably — so do not impose a strong DLA z-evolution; let the data set it (σ widened above).
HCD_LIT_OVER_SIM_SLOPE = (0.95, 0.15, 0.40)  # d ln(lit/sim) / d ln(1+z)


def lit_over_sim_at_z(z, ratio_pivot=HCD_LIT_OVER_SIM, slope=HCD_LIT_OVER_SIM_SLOPE,
                      z_pivot=HCD_Z_PIVOT):
    """z-dependent (literature/sim) dN/dX ratio per class — a power-law in (1+z):
    r_c(z) = r_c(z_p)·((1+z)/(1+z_p))^s_c. The α-prior center tracks the OBSERVED dN/dX_c(z)
    evolution this way (the τ₀-analog: a fixed curve + slope, not a single number)."""
    r = jnp.asarray(ratio_pivot); s = jnp.asarray(slope)
    return r * ((1.0 + jnp.asarray(z)) / (1.0 + z_pivot)) ** s


def hcd_incidence_prior(w_c_fid, z=HCD_Z_PIVOT, lit_over_sim=None):
    """Per-class HCD incidence prior (μ, σ) on α_c (the effective per-class sightline weight
    in ``predict_P_obs``), centered on the OBSERVED incidence AT the data redshift ``z``
    (NOT the sim's), from the fiducial sim weights ``w_c_fid`` = (w_LLS, w_subDLA, w_DLA) ×
    the z-SLOPE literature/sim ratio (``lit_over_sim_at_z(z)``; override with
    ``lit_over_sim``). LLS/subDLA largely UNMASKED → center = (lit/sim)(z)·w_c; DLA
    ~70%-masked → center = HCD_DLA_RESIDUAL_FRAC·(lit/sim)(z)·w_DLA (the residual). α=w_c
    (the sim) is NOT the center: PRIYA mis-predicts subDLA/DLA dN/dX AND its z-slope (see
    scripts/plot_dndx_vs_literature.py). Widths = literature fractional σ/μ × center.
    Returns (alpha_mu (3,), alpha_sigma (3,)); DLA should additionally be one-sided
    (half-normal/softplus) in the sampler.
    """
    w = jnp.asarray(w_c_fid)                                    # (3,)
    r = lit_over_sim_at_z(z) if lit_over_sim is None else jnp.asarray(lit_over_sim)
    fl, fs, fd = HCD_PRIOR_FRAC_SIGMA
    mu = jnp.stack([r[0] * w[0], r[1] * w[1], HCD_DLA_RESIDUAL_FRAC * r[2] * w[2]])
    # DLA dN/dX is unreliable beyond z≈3.5 → widen σ_DLA above it (weak high-z prior) so the
    # data, not the prior, sets the high-z DLA incidence.
    dla_inflate = 1.0 + jnp.clip(jnp.asarray(z) - HCD_DLA_Z_RELIABLE, 0.0, None)
    sigma = jnp.stack([fl * mu[0], fs * mu[1], fd * mu[2] * dla_inflate])
    return mu, sigma


def predict_P_obs_and_cov_single_z(model, theta9, z_unit, z, tau0, alpha_hcd, *,
                                   pf_stats, sigma_zb, alpha_centres, cosmic_cov,
                                   dla_core, dla_shot_flag, shot_inflate=10.0,
                                   cemu_inflate=1.0):
    """Per-z (P_obs (K,), C (K,K)) — the EXACT forward model + covariance the
    likelihood uses, factored out so the closure mock can draw its noise from the
    IDENTICAL C (Leg A's contract: C_mock == C_like; closure_mocks calls this).

      P_obs = P_clean + Σ_{c∈HCD} α_c·(P_c − P_clean)
      C     = cosmic_cov + diag(emu_var),
      emu_var = Σ_c coef_c²·σ_c(k,z,τ₀)²·P_c²,  coef = [1−Σα, α_LLS, α_subDLA, α_DLA].

    No data, no residual, no jitter — JUST the model mean and covariance (the jitter is
    added inside ``gaussian_loglik`` / the mock's Cholesky). Differentiable in (θ9,τ₀,α).
    """
    P_filt = predict_P_filt(model, theta9, z_unit, tau0, pf_stats)          # (4,K) emulated
    P_clean = P_filt[0]
    P_dla_unf = P_filt[3] + jnp.asarray(dla_core)                           # unfiltered DLA
    P_cls = jnp.stack([P_clean, P_filt[1], P_filt[2], P_dla_unf])           # (4,K) class powers
    a = jnp.asarray(alpha_hcd)                                              # (3,)
    coef = jnp.concatenate([jnp.atleast_1d(1.0 - jnp.sum(a)), a])          # (4,) [clean,LLS,sub,DLA]
    P_obs = jnp.einsum("c,ck->k", coef, P_cls)                             # = P_clean + Σ α_c(P_c−P_clean)
    # C_emu: per-class fractional σ (τ₀-interp'd) weighted by each class's coeff in P_obs
    sigma_ck = sigma_at_tau0(sigma_zb, alpha_centres, z, tau0)              # (4,K) fractional
    emu_var = jnp.einsum("c,ck,ck->k", coef ** 2,
                         jnp.nan_to_num(sigma_ck) ** 2, P_cls ** 2)         # Σ coef²·σ²·P_c²
    emu_var = jnp.where(dla_shot_flag, emu_var * shot_inflate, emu_var) * cemu_inflate
    cosmic = jnp.asarray(cosmic_cov)
    C = (jnp.diag(cosmic) if cosmic.ndim == 1 else cosmic) + jnp.diag(emu_var)
    return P_obs, C


def log_lik_single_z(model, theta9, z_unit, z, tau0, alpha_hcd, *,
                     pf_stats, sigma_zb, alpha_centres, cosmic_cov, P_data, dla_core,
                     dla_shot_flag, shot_inflate=10.0, cemu_inflate=1.0,
                     valid_k=None, include_logdet=True):
    """Per-z Gaussian log-likelihood for the CORRECTED HCD forward model + the logdet term.

      P_obs = P_clean + Σ_{c∈HCD} α_c·(P_c − P_clean)          (clean-forest baseline)
      logL_z = −½ rᵀC⁻¹r − ½ logdet C,  r = P_data − P_obs,  C = cosmic_cov + C_emu(θ,τ₀).

    α_c = the effective post-masking per-class incidence (LLS, subDLA, DLA); α_c = w_c
    reproduces the sim's contaminated P_tier_p. ``P_c`` are LIVE-emulated per-class P_filt:
    filtered for LLS/subDLA, UNFILTERED for DLA via ``dla_core`` (K,) = the DLA-core add-back
    P_DLA^unf − P_DLA^filt. ∂P_obs/∂α_c = (P_c − P_clean) (≠0 for LLS — fixes the old
    Δ_LLS≡0 bug).

    C_emu is the per-class emulator error propagated through P_obs's class coefficients
    coef = [1−Σα, α_LLS, α_subDLA, α_DLA] over (clean, LLS, subDLA, DLA):
    emu_var = Σ_c coef_c²·σ_c(k,z,τ₀)²·P_c². The (4,K) FRACTIONAL error ``sigma_zb`` is
    τ₀-banded (Tb axis); ``cemu_inflate`` is the conservative inflation (review I1);
    ``valid_k`` (bool, K; FIXED, not traced) neutralises out-of-range/Nyquist bins (σ
    all-NaN, P_data NaN) so they carry no info and no NaN gradient.
    ``include_logdet=False`` = the negative control. Differentiable in (θ9, τ₀, α).

    The (P_obs, C) assembly is shared with ``predict_P_obs_and_cov_single_z`` so the
    closure mock draws noise from the IDENTICAL covariance.
    """
    P_obs, C = predict_P_obs_and_cov_single_z(
        model, theta9, z_unit, z, tau0, alpha_hcd, pf_stats=pf_stats, sigma_zb=sigma_zb,
        alpha_centres=alpha_centres, cosmic_cov=cosmic_cov, dla_core=dla_core,
        dla_shot_flag=dla_shot_flag, shot_inflate=shot_inflate, cemu_inflate=cemu_inflate)
    # NaN-safe residual: sanitise P_data (out-of-range bins NaN) BEFORE the subtract so
    # the where-branch can't poison the gradient; valid_k zeroes those bins' residual.
    r = jnp.nan_to_num(jnp.asarray(P_data), nan=0.0) - P_obs
    if valid_k is not None:
        r = jnp.where(jnp.asarray(valid_k), r, 0.0)
    if include_logdet:
        return gaussian_loglik(r, C)
    # negative control: chi2 only, no logdet (still SPD-jittered for a fair comparison)
    K = C.shape[-1]
    Cj = C + (1e-10 * jnp.mean(jnp.diag(C))) * jnp.eye(K)
    L = jnp.linalg.cholesky(Cj)
    sol = jax.scipy.linalg.cho_solve((L, True), r)
    return -0.5 * (r @ sol)


def log_lik_multiz(model, theta9, tau0_vec, alpha_hcd, *, pf_stats, z, z_unit, sigma_zb,
                   alpha_centres, cosmic_cov, P_data, dla_core, dla_shot_flag, valid_k,
                   shot_inflate=10.0, cemu_inflate=1.0, include_logdet=True):
    """LIKELIHOOD-ONLY multi-z log-likelihood = Σ_z log_lik_single_z (no priors).

    The data-bin sum is **vmap'd over the z-axis** (K is fixed across bins → no padding; CS
    review M2): θ9 + α are shared (closed over), ``tau0_vec`` is per-z, and the per-z arrays
    (``z, z_unit, sigma_zb, cosmic_cov, P_data, dla_core, dla_shot_flag, valid_k``) carry a
    leading z-axis. This is the LIKELIHOOD-ONLY payload for numpyro's ``factor`` / Cobaya's
    ``logp`` — priors are added separately (CS review M3: never double-count). Differentiable.
    """
    def one(tau0_z, z_z, zu_z, sz, cc, pd, dc, flag, vk):
        return log_lik_single_z(
            model, theta9, zu_z, z_z, tau0_z, alpha_hcd, pf_stats=pf_stats, sigma_zb=sz,
            alpha_centres=alpha_centres, cosmic_cov=cc, P_data=pd, dla_core=dc,
            dla_shot_flag=flag, valid_k=vk, shot_inflate=shot_inflate,
            cemu_inflate=cemu_inflate, include_logdet=include_logdet)
    per_z = jax.vmap(one)(jnp.asarray(tau0_vec), jnp.asarray(z), jnp.asarray(z_unit),
                          jnp.asarray(sigma_zb), jnp.asarray(cosmic_cov), jnp.asarray(P_data),
                          jnp.asarray(dla_core), jnp.asarray(dla_shot_flag),
                          jnp.asarray(valid_k))
    return jnp.sum(per_z)


def log_posterior_single_z(model, theta9, z_unit, z, tau0, alpha_hcd, *,
                           pf_stats, sigma_zb, alpha_centres, cosmic_cov, P_data, dla_core,
                           dla_shot_flag, tau0_mu, tau0_sigma, alpha_mu, alpha_sigma,
                           shot_inflate=10.0, cemu_inflate=1.0, valid_k=None,
                           include_logdet=True, box_sharpness=1e3):
    """Single-z log-POSTERIOR = log-likelihood + smooth unit-box prior + τ₀ mean-flux
    Gaussian + the per-class HCD **incidence prior** on α (TIGHT informative Gaussian
    centered on the structural w_c(dN/dX); LLS especially tight — it's cosmology-degenerate,
    DESI DR1; DLA centered on the residual post-masking fraction). The differentiable scalar
    a single-z-bin NUTS run targets (the multi-z posterior sums ``log_lik_single_z`` over
    data bins + ONE box prior + the per-z τ₀ Gaussian + the α prior — T4 driver).

    PRIOR NOTE (review M1): ``unit_box_logprior`` is a SMOOTH soft-wall, used only for the
    raw-``log_prob`` (blackjax) path; production numpyro uses Uniform(0,1)+auto-bijector
    (exactly-flat in-box, finite grads) — set box_sharpness=0 there."""
    ll = log_lik_single_z(
        model, theta9, z_unit, z, tau0, alpha_hcd, pf_stats=pf_stats, sigma_zb=sigma_zb,
        alpha_centres=alpha_centres, cosmic_cov=cosmic_cov, P_data=P_data, dla_core=dla_core,
        dla_shot_flag=dla_shot_flag, shot_inflate=shot_inflate, cemu_inflate=cemu_inflate,
        valid_k=valid_k, include_logdet=include_logdet)
    lp = unit_box_logprior(theta9, sharpness=box_sharpness)
    lp += meanflux_logprior(tau0, tau0_mu, tau0_sigma)
    lp += jnp.sum(gaussian_logprior(alpha_hcd, alpha_mu, alpha_sigma))     # HCD incidence prior
    return ll + lp
