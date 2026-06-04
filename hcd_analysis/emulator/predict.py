"""Differentiable (JAX-pure) assembly of P_obs(θ, τ₀, α) from a trained Emulator.

``data.reconstruct_P_filt`` is numpy / eval-only; this module is its jnp mirror plus
the structural Tier-P + HCD add-back, i.e. the END-TO-END differentiable forward that
the Phase-C gradient gate (``scripts/diag_grad_fidelity.py``) and the likelihood driver
(Task 3) call. Every function here is JAX-pure and differentiable in (θ, τ₀, α).

Contract (matches ``likelihood.total_p1d_difference`` and ``model.structural_tier_p``):

    P_filt(θ,z,τ₀)  = exp( (m̂·sig_marg + mu_marg) + sig_cosmo·r̂ )      # (4,K) LINEAR
    P_tier_p        = Σ_c w_c · P_filt_c                                  # (K,)
    P_obs           = P_tier_p + Σ_{c∈HCD} α_c · Δ_c                      # (K,)

The θ-dependence enters ONLY through r̂ = HeadB's ``P_filt_resid`` (the baseline m̂ is
θ-blind), so ∂logP̂/∂θ = sig_cosmo·∂r̂/∂θ — verified by the gradient gate. Δ_c is passed
in as the per-class HCD template (field-standard fixed-template / free-amplitude form,
cf. Rogers & Bird 2018 / PRIYA 2025); ∂P_obs/∂α_c = Δ_c exactly. A future refinement
(Task 3) may re-evaluate an emulated Δ_c(θ,τ₀) each step — it would flow through the
same SVD-basis machinery this gate already exercises via ``P_filt_resid``.
"""
from __future__ import annotations

import jax.numpy as jnp

from .model import structural_tier_p


def reconstruct_P_filt_jax(P_filt_base, P_filt_resid, pf_stats):
    """jnp mirror of ``data.reconstruct_P_filt`` — LINEAR P_filt (...,4,K) = exp(logP̂).

    ``pf_stats`` is the structured P_filt norm dict (``mu_marg``, ``sig_marg``,
    ``sig_cosmo``, each (4,K)). Differentiable; the only θ dependence is through r̂.
    """
    sig_marg = jnp.asarray(pf_stats["sig_marg"])
    mu_marg = jnp.asarray(pf_stats["mu_marg"])
    sig_cosmo = jnp.asarray(pf_stats["sig_cosmo"])
    logP = (P_filt_base * sig_marg + mu_marg) + sig_cosmo * P_filt_resid
    return jnp.exp(logP)


def predict_P_filt(model, theta9, z_unit, tau0, pf_stats):
    """LINEAR per-class P_filt (4,K) at unit-cube (θ9, z_unit, τ₀).

    ``model`` is a trained ``Emulator``; ``x = [θ9(9), z_unit(1)]`` is the encoder input.
    Differentiable in θ9 and τ₀.
    """
    x = jnp.concatenate([jnp.asarray(theta9), jnp.asarray(z_unit)[None]])
    pred = model(x, tau0)
    return reconstruct_P_filt_jax(pred["P_filt_base"], pred["P_filt_resid"], pf_stats)


def predict_P_tier_p(model, theta9, z_unit, tau0, w_c, pf_stats):
    """Structural Tier-P total P1D (K,) = Σ_c w_c·P_filt_c. Differentiable in θ9, τ₀."""
    P_filt = predict_P_filt(model, theta9, z_unit, tau0, pf_stats)
    return structural_tier_p(jnp.asarray(w_c), P_filt)


def predict_P_obs(model, theta9, z_unit, tau0, alpha_hcd, w_c, pf_stats, delta_hcd):
    """Total observed P1D (K,): P_tier_p + Σ_{c∈HCD} α_c·Δ_c.

    Args:
      model:      trained ``Emulator``.
      theta9:     (9,) unit-cube cosmology/IGM params.
      z_unit:     scalar unit-cube redshift coordinate (x[9]).
      tau0:       scalar mean-flux coordinate (= −ln⟨F⟩).
      alpha_hcd:  (3,) per-class HCD effective-incidence amplitudes (LLS, subDLA, DLA).
      w_c:        (4,) structural class weights.
      pf_stats:   the structured P_filt norm dict (mu_marg/sig_marg/sig_cosmo).
      delta_hcd:  (3,K) per-class HCD Δ_c templates.

    Differentiable in (θ9, τ₀, α). ∂P_obs/∂α_c = Δ_c.
    """
    P_tier_p = predict_P_tier_p(model, theta9, z_unit, tau0, w_c, pf_stats)
    add_back = jnp.einsum("c,ck->k", jnp.asarray(alpha_hcd), jnp.asarray(delta_hcd))
    return P_tier_p + add_back
