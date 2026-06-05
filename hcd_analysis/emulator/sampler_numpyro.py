"""Phase-C T4a — the numpyro sampler for the emulator-as-likelihood (Leg A).

``numpyro_model(ctx)`` places the priors as ``sample`` sites and the LIKELIHOOD-ONLY
``log_lik_from_ctx`` as a single ``factor`` (so numpyro's potential == Σ log_prior +
log_lik, never double-counted — see ``tests/test_closure_sbc.py::test_potential``). The
contract pins (plan §5):

  - θ ~ Uniform(0,1)^9 with numpyro's AUTO-BIJECTOR: NO soft wall. The Uniform log_prob
    is EXACTLY flat in-box, so the box never double-penalizes (the silent-bias trap of
    ``inference.unit_box_logprior``, which is only for the raw-log_prob blackjax path).
  - τ₀ in the α-LADDER coordinate α=τ₀/Kim(z): the Normal is placed on α (well-conditioned,
    z-isotropic), τ₀=α·Kim(z) is deterministic. Kim(z) is a CONSTANT wrt the sample, so the
    change of variable is exact+linear and Jacobian-free (it only shifts logp by a constant);
    placing the Normal directly on α with the transformed (μ,σ) is the clean encoding.
  - α_lls/α_subdla ~ Normal; α_dla ~ softplus(Normal latent) (one-sided, NOT a hard
    HalfNormal whose gradient is discontinuous at 0).
  - ``dense_mass=True`` over the unconstrained block (the [θ,τ₀] correlation is the physics),
    ``target_accept_prob=0.9``, ``init_to_median``.

x64 is hard-asserted on import; no ``donate``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC, init_to_median

from .closure_ctx import Ctx, ctx_kim, log_lik_from_ctx

assert jax.config.read("jax_enable_x64"), \
    "x64 must be on (import hcd_analysis.emulator before jax)"


def _dla_raw_mu(alpha_dla_mu):
    """The Normal-latent centre whose softplus is ≈ ``alpha_dla_mu``: softplus⁻¹(μ) =
    log(expm1(μ)). Clipped away from 0 so expm1 stays positive (softplus⁻¹ undefined ≤0)."""
    return jnp.log(jnp.expm1(jnp.clip(jnp.asarray(alpha_dla_mu), 1e-6)))


def numpyro_model(ctx: Ctx):
    """The Leg-A generative model: priors (sample sites) + the likelihood factor.

    Sites: ``theta_unit`` (Uniform^9), ``alpha_ladder`` (Normal, the τ₀ ladder coord),
    ``alpha_lls``/``alpha_subdla`` (Normal), ``alpha_dla_raw`` (Normal latent → softplus).
    Deterministics: ``tau0_vec`` (= α·Kim(z)), ``alpha_dla`` (= softplus(raw)). The single
    ``factor("loglik")`` is LIKELIHOOD-ONLY. ctx is closed over (jit traces over params)."""
    # (1) θ ~ Uniform(0,1)^9 — auto-bijector handles the box; NO soft wall (flat in-box).
    theta9 = numpyro.sample(
        "theta_unit", dist.Uniform(jnp.zeros(9), jnp.ones(9)).to_event(1))

    # (2) τ₀ in the α-ladder coordinate: Normal on α = τ₀/Kim(z), τ₀ = α·Kim(z) deterministic.
    kim = ctx_kim(ctx.z)                            # (n_z,) — constant wrt the sample
    alpha_mu = ctx.tau0_mu / kim                    # (n_z,)
    alpha_sd = ctx.tau0_sigma / kim                 # (n_z,)
    alpha_ladder = numpyro.sample(
        "alpha_ladder", dist.Normal(alpha_mu, alpha_sd).to_event(1))
    tau0_vec = numpyro.deterministic("tau0_vec", alpha_ladder * kim)

    # (3) HCD: LLS/subDLA ~ Normal; DLA one-sided via softplus on a Normal latent.
    a_lls = numpyro.sample(
        "alpha_lls", dist.Normal(ctx.alpha_hcd_mu[0], ctx.alpha_hcd_sigma[0]))
    a_sub = numpyro.sample(
        "alpha_subdla", dist.Normal(ctx.alpha_hcd_mu[1], ctx.alpha_hcd_sigma[1]))
    a_dla_raw = numpyro.sample(
        "alpha_dla_raw", dist.Normal(_dla_raw_mu(ctx.alpha_hcd_mu[2]), 1.0))
    a_dla = numpyro.deterministic("alpha_dla", jax.nn.softplus(a_dla_raw))
    alpha_hcd = jnp.stack([a_lls, a_sub, a_dla])

    # (likelihood factor — LIKELIHOOD-ONLY; priors are the sample sites above)
    numpyro.factor("loglik", log_lik_from_ctx(theta9, tau0_vec, alpha_hcd, ctx))


def make_nuts(ctx: Ctx, target_accept=0.9):
    """The NUTS kernel: dense mass over the unconstrained block (the [θ,τ₀] physics
    correlation), ``target_accept`` (0.9 default; raise toward 0.95–0.99 to clear
    divergences on a hard mock), init_to_median. ctx is bound via the model closure."""
    return NUTS(numpyro_model, dense_mass=True, target_accept_prob=float(target_accept),
                init_strategy=init_to_median)


def run_nuts(ctx: Ctx, *, n_warmup, n_samples, seed, num_chains=1,
             target_accept=0.9, progress_bar=False):
    """Run NUTS on ``ctx`` and return ``(samples_dict, n_divergences, extra)``.

    ``samples_dict`` holds every sample/deterministic site (incl. ``tau0_vec`` and
    ``alpha_dla``); ``n_divergences`` is the total divergence count (collected via
    ``extra_fields=("diverging",)``); ``extra`` carries the raw diverging mask and the
    MCMC object for downstream thinning/diagnostics.

    Trace/jit happens ONLY over the sampled params (ctx closed over the model closure),
    so re-running on a NEW mock with the SAME shapes does not recompile."""
    kernel = make_nuts(ctx, target_accept=target_accept)
    mcmc = MCMC(kernel, num_warmup=int(n_warmup), num_samples=int(n_samples),
                num_chains=int(num_chains), progress_bar=progress_bar)
    mcmc.run(jax.random.PRNGKey(int(seed)), ctx, extra_fields=("diverging",))
    samples = mcmc.get_samples()
    extra_fields = mcmc.get_extra_fields()
    diverging = np.asarray(extra_fields.get("diverging", np.zeros(0, bool)))
    n_div = int(diverging.sum())
    extra = {"mcmc": mcmc, "diverging": diverging}
    return samples, n_div, extra
