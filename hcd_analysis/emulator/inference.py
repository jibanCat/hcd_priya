"""Phase-C differentiable likelihood driver — the object HMC (numpyro/blackjax) and the
Cobaya adapter both wrap (design §1.4).

Mirrors PRIYA's (Ho 2023/2024) parameter contract so the cosmology community drives it
as before: the 9 cosmo/IGM params (``data.PARAM_LIMITS`` order, identical to PRIYA's
``coarse_grid``), mean-flux as per-z ``tau0`` (PRIYA's ``mean_flux="per_z"``), and HCD
nuisance as per-class ``alpha_hcd`` (LLS, subDLA, DLA — the single-amplitude analog of
PRIYA's ``a_lls/a_dla``). Wires ``predict.predict_P_obs`` + the τ₀-aware C_emu
(``likelihood.sigma_at_tau0`` → ``assemble_covariance``) + the logdet Gaussian
(``likelihood.gaussian_loglik``) + a SMOOTH bounded prior (replacing PRIYA's ``-inf``
wall so NUTS gets finite gradients).

Everything is JAX-pure and differentiable in (θ_unit, tau0, alpha_hcd).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .predict import predict_P_obs, predict_P_filt
from .likelihood import sigma_at_tau0, assemble_covariance, gaussian_loglik

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


def log_lik_single_z(model, theta9, z_unit, z, tau0, alpha_hcd, *,
                     w_c, delta_hcd, pf_stats, sigma_zb, alpha_centres,
                     cosmic_cov, P_data, dla_shot_flag,
                     sigma_delta_zb=None, shot_inflate=10.0, include_logdet=True):
    """Per-z Gaussian log-likelihood with the τ₀-AWARE C_emu + the logdet term.

      logL_z = −½ rᵀC⁻¹r − ½ logdet C,   r = P_data − P_obs(θ,τ₀,α),
      C = cosmic_cov + C_emu(θ,τ₀)   (C_emu σ interpolated to the sampled τ₀).

    ``sigma_zb`` (4,K,Tb) is the τ₀-banded P_filt error at this data z-band;
    ``alpha_centres`` (Tb,) its band centres. ``sigma_delta_zb`` (3,K,Tb) is the
    optional HCD-Δ error (None → 0, the MVP; the Δ-channel error vector is a follow-up).
    ``include_logdet=False`` is the NEGATIVE CONTROL (drops the logdet — must change the
    τ₀ gradient). Differentiable in (θ9, τ₀, α).
    """
    P_obs = predict_P_obs(model, theta9, z_unit, tau0, alpha_hcd, w_c, pf_stats, delta_hcd)
    P_filt = predict_P_filt(model, theta9, z_unit, tau0, pf_stats)          # (4,K) abs scale
    sigma_ck = sigma_at_tau0(sigma_zb, alpha_centres, z, tau0)              # (4,K)
    n_k = P_filt.shape[1]
    if sigma_delta_zb is None:
        sigma_delta_ck = jnp.zeros((3, n_k))
    else:
        sigma_delta_ck = sigma_at_tau0(sigma_delta_zb, alpha_centres, z, tau0)
    delta_scale = jnp.abs(jnp.asarray(delta_hcd))                           # (3,K)
    C = assemble_covariance(cosmic_cov, sigma_ck, w_c, sigma_delta_ck, alpha_hcd,
                            P_filt, delta_scale, dla_shot_flag, shot_inflate)
    r = jnp.asarray(P_data) - P_obs
    if include_logdet:
        return gaussian_loglik(r, C)
    # negative control: chi2 only, no logdet
    L = jnp.linalg.cholesky(C)
    sol = jax.scipy.linalg.cho_solve((L, True), r)
    return -0.5 * (r @ sol)


def log_posterior_single_z(model, theta9, z_unit, z, tau0, alpha_hcd, *,
                           w_c, delta_hcd, pf_stats, sigma_zb, alpha_centres,
                           cosmic_cov, P_data, dla_shot_flag,
                           tau0_mu, tau0_sigma, sigma_delta_zb=None,
                           shot_inflate=10.0, include_logdet=True,
                           box_sharpness=1e3):
    """Single-z log-POSTERIOR = log-likelihood + smooth unit-box prior + τ₀ mean-flux
    Gaussian. The differentiable scalar a single-z-bin NUTS run targets (the multi-z
    posterior sums ``log_lik_single_z`` over data bins + ONE box prior + the per-z τ₀
    Gaussian — wired in the closure/data driver, T4)."""
    ll = log_lik_single_z(
        model, theta9, z_unit, z, tau0, alpha_hcd, w_c=w_c, delta_hcd=delta_hcd,
        pf_stats=pf_stats, sigma_zb=sigma_zb, alpha_centres=alpha_centres,
        cosmic_cov=cosmic_cov, P_data=P_data, dla_shot_flag=dla_shot_flag,
        sigma_delta_zb=sigma_delta_zb, shot_inflate=shot_inflate,
        include_logdet=include_logdet)
    lp = unit_box_logprior(theta9, sharpness=box_sharpness)
    lp += meanflux_logprior(tau0, tau0_mu, tau0_sigma)
    return ll + lp
