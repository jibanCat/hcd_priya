"""Differentiable (JAX-pure) assembly of P_obs(θ, τ₀, α) from a trained Emulator.

``data.reconstruct_P_filt`` is numpy / eval-only; this module is its jnp mirror plus
the structural Tier-P + HCD add-back, i.e. the END-TO-END differentiable forward that
the Phase-C gradient gate (``scripts/diag_grad_fidelity.py``) and the likelihood driver
(Task 3) call. Every function here is JAX-pure and differentiable in (θ, τ₀, α).

Contract (CORRECTED HCD forward model — redesign 2026-06-04, supersedes the old
``P_tier_p + Σα·Δ_c`` with the ≡0-for-LLS filter-residual template):

    P_filt(θ,z,τ₀)  = exp( (m̂·sig_marg + mu_marg) + sig_cosmo·r̂ )      # (4,K) LINEAR
    R_c             = P_c − P_clean   (filtered LLS/subDLA; unfiltered DLA)# (3,K) excess
    P_obs           = P_clean + Σ_{c∈HCD} α_c · R_c                       # (K,)

The θ-dependence enters ONLY through r̂ = HeadB's ``P_filt_resid`` (the baseline m̂ is
θ-blind), so ∂logP̂/∂θ = sig_cosmo·∂r̂/∂θ — verified by the gradient gate. The excess
templates R_c are LIVE-emulated from P_filt each step (field-standard fixed-shape /
free-amplitude form, cf. Rogers & Bird 2018 / PRIYA 2025); ∂P_obs/∂α_c = R_c = (P_c −
P_clean), now ≠0 for LLS. ``predict_P_tier_p`` (the structural Σ_c w_c·P_filt) remains
for the clean-path diagnostics but is NOT the HCD forward model. See README.md §Forward-model.
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

    ``model`` may also be an ``ensemble.EnsembleEmulator`` (detected by duck-typing on its
    ``.members`` attribute — ``Emulator`` has no such field): then the result is the MEAN
    over members of the reconstructed (post-exp) per-class P_filt. The mean is taken AFTER
    reconstruction because P_filt is exp-nonlinear in (base, resid); this matches the
    production ensemble definition (``validate_production_ensemble.py``). Threading it here
    propagates the ensemble through predict_P_obs / predict_excess / the closure forward.
    """
    x = jnp.concatenate([jnp.asarray(theta9), jnp.asarray(z_unit)[None]])
    members = getattr(model, "members", None)
    if members is not None:
        per = jnp.stack([
            reconstruct_P_filt_jax(p["P_filt_base"], p["P_filt_resid"], pf_stats)
            for p in (m(x, tau0) for m in members)], axis=0)        # (M,4,K)
        return jnp.mean(per, axis=0)                                # (4,K)
    pred = model(x, tau0)
    return reconstruct_P_filt_jax(pred["P_filt_base"], pred["P_filt_resid"], pf_stats)


def predict_P_tier_p(model, theta9, z_unit, tau0, w_c, pf_stats):
    """Structural Tier-P total P1D (K,) = Σ_c w_c·P_filt_c. Differentiable in θ9, τ₀."""
    P_filt = predict_P_filt(model, theta9, z_unit, tau0, pf_stats)
    return structural_tier_p(jnp.asarray(w_c), P_filt)


def _excess_from_P_filt(P_filt, dla_core):
    """Per-class HCD excess-over-clean R_c = P_c − P_clean (3,K) from the emulated P_filt.

    The CORRECTED HCD contamination object (Rogers&Bird 2018 / DESI DR1 / PRIYA form):
    the EXCESS power of class-c sightlines over the clean forest — NOT the old filter
    residual P_c^unf − P_c^filt (which was ≡0 for LLS). FILTERED for LLS/subDLA (filt≈unfilt
    there); UNFILTERED for DLA (the data's residual unmasked DLAs are full systems, so
    P_DLA^unf = P_filt[DLA] + ``dla_core``, dla_core = the DLA-core add-back
    P_DLA^unf − P_DLA^filt). ``P_filt`` is (4,K): clean, LLS, subDLA, DLA (filtered).
    """
    P_clean = P_filt[0]
    P_dla_unf = P_filt[3] + jnp.asarray(dla_core)
    return jnp.stack([P_filt[1] - P_clean,        # LLS    filtered  − clean
                      P_filt[2] - P_clean,        # subDLA filtered  − clean
                      P_dla_unf - P_clean])        # DLA    unfiltered − clean


def predict_excess(model, theta9, z_unit, tau0, pf_stats, dla_core):
    """Live-emulated per-class HCD excess templates R_c = P_c − P_clean (3,K).
    Differentiable in (θ9, τ₀). See ``_excess_from_P_filt``."""
    return _excess_from_P_filt(
        predict_P_filt(model, theta9, z_unit, tau0, pf_stats), dla_core)


def predict_P_obs(model, theta9, z_unit, tau0, alpha_hcd, pf_stats, dla_core):
    """Total observed P1D (K,): **P_clean + Σ_{c∈HCD} α_c·(P_c − P_clean)** — the corrected
    HCD-marginalization forward model.

    Clean-forest baseline + the live-emulated excess templates (≡ the multiplicative
    P_clean·[1 + Σ_c α_c·(P_c/P_clean − 1)] form; the additive/multiplicative and
    Δ-vs-reweight forks are algebraically the same model — Lyα+CS agent verdict 2026-06-04).
    α_c = the effective post-masking per-class incidence (LLS, subDLA, DLA), prior-centered
    on the structural w_c(dN/dX); α_c = w_c reproduces the sim's contaminated P_tier_p.
    This fixes the Δ_LLS≡0 bug (the old P_c^unf−P_c^filt template).

    Args:
      alpha_hcd:  (3,) effective per-class incidence (LLS, subDLA, DLA).
      pf_stats:   structured P_filt norm dict; dla_core: (K,) DLA-core add-back template.

    Differentiable in (θ9, τ₀, α). ∂P_obs/∂α_c = (P_c − P_clean) (≠0 for LLS now).
    """
    P_filt = predict_P_filt(model, theta9, z_unit, tau0, pf_stats)   # (4,K) emulated
    P_clean = P_filt[0]
    excess = _excess_from_P_filt(P_filt, dla_core)                   # (3,K)
    return P_clean + jnp.einsum("c,ck->k", jnp.asarray(alpha_hcd), excess)
