"""Phase-C T4a — Leg-A mock generation (PURE: no NUTS, no I/O).

Leg A = TRUE-SBC: the mocks come from the sampler's OWN generative model, so any
rank non-uniformity is an attributable sampler/geometry/coding bug (plan §0). Two steps:

  ``draw_leg_a_truth(ctx, key)`` draws (θ̃, τ̃₀, α̃) from EXACTLY the ``numpyro_model``
  priors — θ~Uniform(0,1)^9, τ₀ via the ladder Normal × Kim(z), α_lls/α_subdla~Normal,
  α_dla~softplus(Normal latent). Same distributions, same parametrization → the truth is
  a genuine prior draw.

  ``make_leg_a_mock(ctx, truth, key)`` forward-models P_obs_emu(θ̃,τ̃₀,α̃) per z and adds
  ε ~ N(0, C_cosmic + C_emu) drawn via a jittered Cholesky of the SAME C the likelihood
  assembles (``inference.predict_P_obs_and_cov_single_z`` — so C_mock == C_like, Leg A's
  whole point). Returns a NEW ctx with ``P_data := mock`` + the packed truth + the key.

The DLA core is FIXED (the mock uses the IDENTICAL ``ctx.dla_core``; no τ₀-interp — MVP).
``valid_k`` is respected: invalid bins are NaN'd in the mock and carry no noise.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .closure_ctx import Ctx, ctx_kim, pack
from .sampler_numpyro import _dla_raw_mu
from .inference import predict_P_obs_and_cov_single_z

assert jax.config.read("jax_enable_x64"), \
    "x64 must be on (import hcd_analysis.emulator before jax)"

# Cholesky jitter — matches ``likelihood.gaussian_loglik`` (relative to ⟨diag C⟩) so the
# mock noise and the likelihood factorization see the SAME jittered C.
_JITTER = 1e-10


def draw_leg_a_truth(ctx: Ctx, key):
    """Draw a single truth (θ̃,τ̃₀,α̃) from EXACTLY the numpyro_model priors.

    Returns ``dict(theta9 (9,), tau0_vec (n_z,), alpha_hcd (3,), alpha_ladder (n_z,),
    alpha_dla_raw (scalar))`` — the latent (ladder/raw) sites are kept so a mock built
    from this truth lands at the same point the sampler would target. Pure (key → truth)."""
    k_th, k_lad, k_lls, k_sub, k_dla = jax.random.split(key, 5)
    kim = ctx_kim(ctx.z)                                          # (n_z,)

    # θ ~ Uniform(0,1)^9
    theta9 = jax.random.uniform(k_th, (9,))
    # τ₀ ladder coord α ~ Normal(μ_τ₀/Kim, σ_τ₀/Kim); τ₀ = α·Kim(z)
    alpha_mu = ctx.tau0_mu / kim
    alpha_sd = ctx.tau0_sigma / kim
    alpha_ladder = alpha_mu + alpha_sd * jax.random.normal(k_lad, (ctx.n_z,))
    tau0_vec = alpha_ladder * kim
    # α_lls/α_subdla ~ Normal; α_dla ~ softplus(Normal latent)
    a_lls = ctx.alpha_hcd_mu[0] + ctx.alpha_hcd_sigma[0] * jax.random.normal(k_lls)
    a_sub = ctx.alpha_hcd_mu[1] + ctx.alpha_hcd_sigma[1] * jax.random.normal(k_sub)
    a_dla_raw = _dla_raw_mu(ctx.alpha_hcd_mu[2]) + 1.0 * jax.random.normal(k_dla)
    a_dla = jax.nn.softplus(a_dla_raw)
    alpha_hcd = jnp.stack([a_lls, a_sub, a_dla])
    return dict(theta9=theta9, tau0_vec=tau0_vec, alpha_hcd=alpha_hcd,
                alpha_ladder=alpha_ladder, alpha_dla_raw=a_dla_raw)


def truth_vector(ctx: Ctx, truth):
    """Pack a truth dict into the canonical SBC vector [θ9, τ₀(n_z), α_lls,α_sub,α_dla]."""
    return pack(truth["theta9"], truth["tau0_vec"], truth["alpha_hcd"])


def _cov_at(ctx: Ctx, truth):
    """(P_obs, C) per z at the truth — the SAME assembly the likelihood uses.
    Returns (P_obs (n_z,K), C (n_z,K,K))."""
    def one(z, zu, t0, sz, cc, dc, flag):
        return predict_P_obs_and_cov_single_z(
            ctx.model, truth["theta9"], zu, z, t0, truth["alpha_hcd"],
            pf_stats=ctx.pf_stats, sigma_zb=sz, alpha_centres=ctx.alpha_centres,
            cosmic_cov=cc, dla_core=dc, dla_shot_flag=flag,
            shot_inflate=ctx.shot_inflate, cemu_inflate=ctx.cemu_inflate)
    P_obs, C = jax.vmap(one)(ctx.z, ctx.z_unit, truth["tau0_vec"], ctx.sigma_zb,
                             ctx.cosmic_cov, ctx.dla_core, ctx.dla_shot_flag)
    return P_obs, C


def _chol_jitter(C):
    """Lower Cholesky of the jittered C (matches gaussian_loglik's SPD floor)."""
    K = C.shape[-1]
    Cj = C + (_JITTER * jnp.mean(jnp.diag(C))) * jnp.eye(K)
    return jnp.linalg.cholesky(Cj)


def make_leg_a_mock(ctx: Ctx, truth, key):
    """Forward-model the truth and add ε ~ N(0, C_cosmic+C_emu) from the IDENTICAL C.

    Per z: draw ε = L·g, g~N(0,I), L = chol(C_jittered) — the SAME jittered C the
    likelihood factorizes (Leg A: C_mock == C_like). Invalid bins (``valid_k`` False)
    are NaN'd in the returned mock (no noise, no info), mirroring the real data range.

    Returns ``(ctx_mock, truth_vec, info)``:
      ctx_mock  — a NEW Ctx with ``P_data := mock`` (everything else copied);
      truth_vec — the packed truth [θ9, τ₀, α] for SBC ranking;
      info      — dict(key, L (n_z,K,K), P_obs (n_z,K)) for the C_mock==C_like assert.
    """
    P_obs, C = _cov_at(ctx, truth)                               # (n_z,K), (n_z,K,K)
    L = jax.vmap(_chol_jitter)(C)                                # (n_z,K,K) lower
    g = jax.random.normal(key, (ctx.n_z, ctx.K))                 # iid standard normal
    eps = jnp.einsum("zik,zk->zi", L, g)                         # ε = L g  ~ N(0,C)
    mock = P_obs + eps
    # respect valid_k: out-of-range bins carry no data (NaN), exactly like the likelihood
    # NaN-sanitises them (and the SBC then ranks only the informed bins via valid_k).
    mock = jnp.where(jnp.asarray(ctx.valid_k), mock, jnp.nan)

    ctx_mock = Ctx(
        model=ctx.model, pf_stats=ctx.pf_stats, z=ctx.z, z_unit=ctx.z_unit,
        sigma_zb=ctx.sigma_zb, alpha_centres=ctx.alpha_centres, cosmic_cov=ctx.cosmic_cov,
        P_data=mock, dla_core=ctx.dla_core, dla_shot_flag=ctx.dla_shot_flag,
        valid_k=ctx.valid_k, w_c_fid=ctx.w_c_fid, tau0_mu=ctx.tau0_mu,
        tau0_sigma=ctx.tau0_sigma, alpha_hcd_mu=ctx.alpha_hcd_mu,
        alpha_hcd_sigma=ctx.alpha_hcd_sigma, n_z=ctx.n_z, K=ctx.K, Tb=ctx.Tb,
        shot_inflate=ctx.shot_inflate, cemu_inflate=ctx.cemu_inflate,
        include_logdet=ctx.include_logdet)
    info = {"key": key, "L": L, "P_obs": P_obs}
    return ctx_mock, truth_vector(ctx, truth), info
