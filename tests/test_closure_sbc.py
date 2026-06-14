"""Phase-C T4a — Leg-A closure / SBC foundation tests.

Ports + extends the validated /tmp/t4a_proto.py checks:
  (a) pack/unpack round-trip + names align with PARAM_NAMES;
  (b) numpyro log_density == log_lik + Σ log_prior  (the no-double-count proof);
  (c) no-soft-box: Uniform(0,1)^9 is EXACTLY flat in-box;
  (d) vmap-sum == loop-sum for log_lik_multiz;
  (e) tiny-NUTS smoke (8 warmup / 8 samples, n_z=1) returns finite samples + a div count;
  (f) Leg-A mock: the Cholesky factor used for the noise draw == the likelihood's
      assembled C at the truth (C_mock == C_like).

Uses the REAL final_fold0 checkpoint + error_vector.npz; env-gated/skipped if absent
(like the existing tests). Run:
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu \
    /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_closure_sbc.py -q
"""
import os

import hcd_analysis.emulator  # noqa: F401  enables x64
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import numpyro.distributions as dist
from numpyro.infer.util import log_density

from hcd_analysis.emulator.inference import (
    PARAM_NAMES, predict_P_obs_and_cov_single_z,
)
from hcd_analysis.emulator import closure_ctx as C
from hcd_analysis.emulator.closure_ctx import ctx_kim, log_lik_from_ctx
from hcd_analysis.emulator import sampler_numpyro as S
from hcd_analysis.emulator.sampler_numpyro import numpyro_model, _dla_raw_mu, run_nuts
from hcd_analysis.emulator import closure_mocks as M

REPO = "/home/mfho/hcd_priya"
CKPT = f"{REPO}/checkpoints/final_fold0"
ERROR_VECTOR = f"{REPO}/checkpoints/error_vector.npz"

_have_ckpt = os.path.exists(CKPT + ".eqx") and os.path.exists(ERROR_VECTOR)
pytestmark = pytest.mark.skipif(
    not _have_ckpt, reason="final_fold0 checkpoint or error_vector.npz absent")


@pytest.fixture(scope="module")
def ctx3():
    from hcd_analysis.emulator.closure_sbc import build_ctx
    return build_ctx(n_z=3, seed=0)


@pytest.fixture(scope="module")
def ctx1():
    from hcd_analysis.emulator.closure_sbc import build_ctx
    return build_ctx(n_z=1, seed=0)


# ---------------------------------------------------------------------------
# (a) pack / unpack round-trip + names
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_z", [1, 3])
def test_pack_unpack_roundtrip_and_names(n_z):
    th = jnp.arange(9.0) / 10
    t0 = jnp.arange(n_z) + 1.0
    al = jnp.asarray([0.1, 0.2, 0.3])
    vec = C.pack(th, t0, al)
    th2, t02, al2 = C.unpack(vec, n_z)
    assert jnp.allclose(th, th2) and jnp.allclose(t0, t02) and jnp.allclose(al, al2)
    d = C.to_dict(vec, n_z)
    vec2 = C.from_dict(d, n_z)
    assert jnp.allclose(vec, vec2)
    names = C.param_names(n_z)
    assert names[:9] == list(PARAM_NAMES)
    assert list(d.keys())[:9] == list(PARAM_NAMES)
    assert names[9:9 + n_z] == [f"tau0_z{i}" for i in range(n_z)]
    assert names[-3:] == ["alpha_lls", "alpha_subdla", "alpha_dla"]


# ---------------------------------------------------------------------------
# (b) numpyro potential == log_lik + Σ log_prior   (no double-count)
# ---------------------------------------------------------------------------
def _manual_log_prior(params, ctx):
    """Independent Σ log_prior in numpyro's CONSTRAINED sample space."""
    kim = ctx_kim(ctx.z)
    lp = dist.Uniform(jnp.zeros(9), jnp.ones(9)).to_event(1).log_prob(params["theta_unit"])
    lp += dist.Normal(ctx.tau0_mu / kim, ctx.tau0_sigma / kim).to_event(1).log_prob(
        params["alpha_ladder"])
    lp += dist.Normal(ctx.alpha_hcd_mu[0], ctx.alpha_hcd_sigma[0]).log_prob(params["alpha_lls"])
    lp += dist.Normal(ctx.alpha_hcd_mu[1], ctx.alpha_hcd_sigma[1]).log_prob(params["alpha_subdla"])
    lp += dist.Normal(_dla_raw_mu(ctx.alpha_hcd_mu[2]), 1.0).log_prob(params["alpha_dla_raw"])
    return lp


def test_potential_equals_loglik_plus_logprior(ctx3):
    ctx = ctx3
    rng = np.random.default_rng(1)
    kim = np.asarray(ctx_kim(ctx.z))
    params = {
        "theta_unit": jnp.asarray(rng.uniform(0.3, 0.7, 9)),
        "alpha_ladder": jnp.asarray(np.asarray(ctx.tau0_mu) / kim),
        "alpha_lls": jnp.asarray(float(ctx.alpha_hcd_mu[0])),
        "alpha_subdla": jnp.asarray(float(ctx.alpha_hcd_mu[1])),
        "alpha_dla_raw": jnp.asarray(0.3),
    }
    ld, _ = log_density(numpyro_model, (ctx,), {}, params)

    th = params["theta_unit"]
    tau0_vec = params["alpha_ladder"] * jnp.asarray(kim)
    a_dla = jax.nn.softplus(params["alpha_dla_raw"])
    alpha_hcd = jnp.stack([params["alpha_lls"], params["alpha_subdla"], a_dla])
    ll = log_lik_from_ctx(th, tau0_vec, alpha_hcd, ctx)
    lp = _manual_log_prior(params, ctx)
    recon = ll + lp
    assert jnp.allclose(ld, recon, atol=1e-6, rtol=0), (float(ld), float(recon))


# ---------------------------------------------------------------------------
# (c) no-soft-box: Uniform(0,1)^9 is exactly flat in-box
# ---------------------------------------------------------------------------
def test_no_soft_box_uniform_flat_inbox():
    rng = np.random.default_rng(2)
    u = dist.Uniform(jnp.zeros(9), jnp.ones(9)).to_event(1)
    a = jnp.asarray(rng.uniform(0.1, 0.4, 9))
    b = jnp.asarray(rng.uniform(0.6, 0.9, 9))
    assert jnp.allclose(u.log_prob(a), u.log_prob(b)), "Uniform must be exactly flat in-box"
    # and == 0 (unit cube): log density of Uniform(0,1) is 0 everywhere in-box
    assert jnp.allclose(u.log_prob(a), 0.0)


# ---------------------------------------------------------------------------
# (d) vmap-sum == loop-sum for log_lik_multiz
# ---------------------------------------------------------------------------
def test_vmap_sum_equals_loop_sum(ctx3):
    from hcd_analysis.emulator.inference import log_lik_multiz, log_lik_single_z
    ctx = ctx3
    rng = np.random.default_rng(3)
    theta9 = jnp.asarray(rng.uniform(0.3, 0.7, 9))
    tau0_vec = jnp.asarray(np.asarray(ctx.tau0_mu))
    alpha_hcd = jnp.asarray(np.maximum(np.asarray(ctx.alpha_hcd_mu), 1e-4))
    total = float(log_lik_multiz(
        ctx.model, theta9, tau0_vec, alpha_hcd, pf_stats=ctx.pf_stats, z=ctx.z,
        z_unit=ctx.z_unit, sigma_zb=ctx.sigma_zb, alpha_centres=ctx.alpha_centres,
        cosmic_cov=ctx.cosmic_cov, P_data=ctx.P_data, dla_core=ctx.dla_core,
        dla_shot_flag=ctx.dla_shot_flag, valid_k=ctx.valid_k,
        shot_inflate=ctx.shot_inflate, cemu_inflate=ctx.cemu_inflate,
        include_logdet=ctx.include_logdet))
    loop = sum(float(log_lik_single_z(
        ctx.model, theta9, float(ctx.z_unit[i]), float(ctx.z[i]), float(tau0_vec[i]),
        alpha_hcd, pf_stats=ctx.pf_stats, sigma_zb=ctx.sigma_zb[i],
        alpha_centres=ctx.alpha_centres, cosmic_cov=ctx.cosmic_cov[i], P_data=ctx.P_data[i],
        dla_core=ctx.dla_core[i], dla_shot_flag=ctx.dla_shot_flag[i], valid_k=ctx.valid_k[i],
        shot_inflate=ctx.shot_inflate, cemu_inflate=ctx.cemu_inflate,
        include_logdet=ctx.include_logdet)) for i in range(ctx.n_z))
    assert np.isclose(total, loop, rtol=1e-10), f"vmap {total} vs loop {loop}"


# ---------------------------------------------------------------------------
# (e) tiny-NUTS smoke: finite samples + a divergence count
# ---------------------------------------------------------------------------
def test_tiny_nuts_runs_finite(ctx1):
    # build a Leg-A mock so the data is self-consistent, then a tiny NUTS run.
    truth = M.draw_leg_a_truth(ctx1, jax.random.PRNGKey(0))
    ctx_mock, _vec, _info = M.make_leg_a_mock(ctx1, truth, jax.random.PRNGKey(1))
    samples, n_div, extra = run_nuts(ctx_mock, n_warmup=8, n_samples=8, seed=0)
    for site in ("theta_unit", "alpha_ladder", "tau0_vec", "alpha_lls",
                 "alpha_subdla", "alpha_dla"):
        assert site in samples, f"missing site {site}"
        assert np.isfinite(np.asarray(samples[site])).all(), f"{site} non-finite"
    assert np.asarray(samples["theta_unit"]).shape == (8, 9)
    assert np.asarray(samples["tau0_vec"]).shape == (8, ctx1.n_z)
    assert isinstance(n_div, int) and n_div >= 0
    # softplus α_dla is strictly positive (one-sided)
    assert np.all(np.asarray(samples["alpha_dla"]) > 0.0)


# ---------------------------------------------------------------------------
# (f) Leg-A mock: C_mock (the noise Cholesky) == the likelihood's C at the truth
# ---------------------------------------------------------------------------
def test_leg_a_mock_C_matches_likelihood(ctx3):
    ctx = ctx3
    truth = M.draw_leg_a_truth(ctx, jax.random.PRNGKey(7))
    ctx_mock, truth_vec, info = M.make_leg_a_mock(ctx, truth, jax.random.PRNGKey(8))
    L_mock = np.asarray(info["L"])                     # (n_z,K,K) — used for the noise draw

    # independently assemble the likelihood's C at the truth, jitter + Cholesky it the
    # SAME way the mock does, and require an exact match per z.
    for i in range(ctx.n_z):
        _P, Ci = predict_P_obs_and_cov_single_z(
            ctx.model, truth["theta9"], ctx.z_unit[i], ctx.z[i], truth["tau0_vec"][i],
            truth["alpha_hcd"], pf_stats=ctx.pf_stats, sigma_zb=ctx.sigma_zb[i],
            alpha_centres=ctx.alpha_centres, cosmic_cov=ctx.cosmic_cov[i],
            dla_core=ctx.dla_core[i], dla_shot_flag=ctx.dla_shot_flag[i],
            shot_inflate=ctx.shot_inflate, cemu_inflate=ctx.cemu_inflate)
        K = Ci.shape[-1]
        Cj = Ci + (1e-10 * jnp.mean(jnp.diag(Ci))) * jnp.eye(K)
        L_ref = np.asarray(jnp.linalg.cholesky(Cj))
        assert np.allclose(L_mock[i], L_ref, atol=1e-12, rtol=0), \
            f"C_mock != C_like at z-index {i}"

    # and the packed truth round-trips through unpack
    th, t0, al = C.unpack(jnp.asarray(truth_vec), ctx.n_z)
    assert np.allclose(np.asarray(th), np.asarray(truth["theta9"]))
    assert np.allclose(np.asarray(t0), np.asarray(truth["tau0_vec"]))
    assert np.allclose(np.asarray(al), np.asarray(truth["alpha_hcd"]))
