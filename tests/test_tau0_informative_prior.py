"""Informative-τ₀ prior + subDLA truth-displacement — TDD + golden-guarded tests.

Two env-gated likelihood options on ``closure_legb``:
  CHANGE 1 — _sample_tau0_sites: Uniform by default (byte-identical) OR a Kim-centered
             TruncatedNormal (truncated to the SAME physical range) when ctx.{tau0_amp_gauss,
             dtau0_gauss} is set. IDENTICAL sample-site names/order in both model twins.
  CHANGE 2 — apply_subdla_truth_boost: the subDLA (index 1) sibling of apply_lls_truth_boost
             (index 0). Scales ONLY the subDLA column; input unmutated; boost=1 is identity.

The cheap tests trace ``_sample_tau0_sites`` on a MINIMAL stand-in ctx (no Cholesky / no real
cache) — fast. The site-order test traces the real ``_legb_priors_only`` (priors-only → cheap,
no 681×681 Cholesky) and is skipped when the real cache/ckpt/DESI are absent.

Env (MANDATORY):
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
    /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_tau0_informative_prior.py -q
"""
import os
from collections import namedtuple

import numpy as np
import pytest

import hcd_analysis.emulator  # noqa: F401  x64 BEFORE jax
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions.truncated import TwoSidedTruncatedDistribution
from numpyro import handlers


def _is_truncnorm(d):
    """``dist.TruncatedNormal`` with both low+high is a TwoSidedTruncatedDistribution (a factory,
    not a class) whose base_dist is Normal — assert that shape."""
    return isinstance(d, TwoSidedTruncatedDistribution) and isinstance(d.base_dist, dist.Normal)

from hcd_analysis.emulator import closure_legb as C

_CACHE = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/observables_tau0_lf.h5"
_CKPT0 = "/home/mfho/hcd_priya/checkpoints/final_fold0.eqx"
_DESI_NPZ = "/home/mfho/data/desi_dr1_p1d/desi_dr1_p1d.npz"
_have = all(os.path.exists(p) for p in (_CACHE, _CKPT0, _DESI_NPZ))

# The physical ranges _sample_tau0_sites truncates to (PRIYA Bird+2023 §2.7.1).
_AMP_RANGE = C.TAU0_AMP_RANGE     # (0.75, 1.25)
_DTAU0_RANGE = C.DTAU0_RANGE      # (-0.40, 0.25)

# A MINIMAL stand-in ctx exposing only what _sample_tau0_sites reads (keeps the test fast).
_MiniCtx = namedtuple("_MiniCtx", ["tau0_amp_range", "dtau0_range",
                                   "tau0_amp_gauss", "dtau0_gauss"])


def _mini(tau0_amp_gauss=None, dtau0_gauss=None):
    return _MiniCtx(_AMP_RANGE, _DTAU0_RANGE, tau0_amp_gauss, dtau0_gauss)


def _trace_tau0(ctx, seed=0):
    """Trace the two τ₀ sites of _sample_tau0_sites on a mini ctx (cheap — no factor/Cholesky)."""
    tr = handlers.trace(handlers.seed(
        lambda: C._sample_tau0_sites(ctx), jax.random.PRNGKey(seed))).get_trace()
    return tr


def _sites(model, *a, seed=1):
    tr = handlers.trace(handlers.seed(model, jax.random.PRNGKey(seed))).get_trace(*a)
    return [k for k, v in tr.items() if v["type"] == "sample" and not v.get("is_observed")]


# --------------------------------------------------------------------------- #
#  1. Default (None) branch → Uniform with the right bounds; site names/order.
# --------------------------------------------------------------------------- #
def test_default_branch_is_uniform_right_bounds():
    """tau0_amp_gauss=None → both τ₀ sites are Uniform on the physical range (byte-identical)."""
    tr = _trace_tau0(_mini())
    # site NAMES + ORDER (tau0_amp before dtau0).
    sample_sites = [k for k, v in tr.items() if v["type"] == "sample"]
    assert sample_sites == ["tau0_amp", "dtau0"], sample_sites
    da, dd = tr["tau0_amp"]["fn"], tr["dtau0"]["fn"]
    assert isinstance(da, dist.Uniform) and isinstance(dd, dist.Uniform)
    assert float(da.low) == _AMP_RANGE[0] and float(da.high) == _AMP_RANGE[1]
    assert float(dd.low) == _DTAU0_RANGE[0] and float(dd.high) == _DTAU0_RANGE[1]


@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI not present")
def test_priors_only_site_order_unchanged_default():
    """The full _legb_priors_only sample-site list (default ctx) is the recorded reference:
    theta_unit, tau0_amp, dtau0, then the HCD/zslope/res_corr block. The τ₀ refactor must NOT
    move/rename a site (constrain_fn depends on the order)."""
    ctx, d = C.build_legb_ctx(desi_kwargs=dict(z_lo=0.0, z_hi=2.6), use_xclass=True)
    ctx = ctx._replace(legs=[l for l in ctx.legs if l.name == "DESI"])
    s = _sites(C._legb_priors_only, ctx)
    # default config: marginalize_zslope True, metals off, alpha_res sampled.
    assert s == ["theta_unit", "tau0_amp", "dtau0",
                 "alpha_lls", "alpha_subdla", "alpha_dla_raw",
                 "s_lls", "s_subdla", "s_dla",
                 "alpha_res", "alpha_res_slope"], s


# --------------------------------------------------------------------------- #
#  2. Gauss branch → TruncatedNormal(loc, scale, low, high) on both sites.
# --------------------------------------------------------------------------- #
def test_gauss_branch_is_truncated_normal():
    """tau0_amp_gauss=(1.0,0.05)/dtau0_gauss=(0.0,0.05) → TruncatedNormal with loc/scale/low/high
    == expected; the density at the center > near the truncation edge; support stays in range."""
    ctx = _mini(tau0_amp_gauss=(1.0, 0.05), dtau0_gauss=(0.0, 0.05))
    tr = _trace_tau0(ctx)
    da, dd = tr["tau0_amp"]["fn"], tr["dtau0"]["fn"]
    assert _is_truncnorm(da) and _is_truncnorm(dd)
    # loc/scale (on base_dist) / low/high (the truncation bounds).
    assert float(da.base_dist.loc) == 1.0 and float(da.base_dist.scale) == 0.05
    assert float(da.low) == _AMP_RANGE[0] and float(da.high) == _AMP_RANGE[1]
    assert float(dd.base_dist.loc) == 0.0 and float(dd.base_dist.scale) == 0.05
    assert float(dd.low) == _DTAU0_RANGE[0] and float(dd.high) == _DTAU0_RANGE[1]
    # log_prob at the center is HIGHER than near the truncation edge (it IS centered).
    assert float(da.log_prob(jnp.asarray(1.0))) > float(da.log_prob(jnp.asarray(_AMP_RANGE[1] - 1e-3)))
    assert float(dd.log_prob(jnp.asarray(0.0))) > float(dd.log_prob(jnp.asarray(_DTAU0_RANGE[1] - 1e-3)))
    # the support stays strictly within the physical range over many draws.
    for sd in range(40):
        tr_s = _trace_tau0(ctx, seed=sd)
        amp = float(tr_s["tau0_amp"]["value"]); dt = float(tr_s["dtau0"]["value"])
        assert _AMP_RANGE[0] <= amp <= _AMP_RANGE[1], amp
        assert _DTAU0_RANGE[0] <= dt <= _DTAU0_RANGE[1], dt


def test_gauss_branch_one_sided_amp_only():
    """Only one of the two sites informative (amp gauss, dtau0 None) → amp TruncatedNormal,
    dtau0 still Uniform. Independently switchable; site names/order unchanged."""
    ctx = _mini(tau0_amp_gauss=(1.0, 0.05), dtau0_gauss=None)
    tr = _trace_tau0(ctx)
    assert [k for k, v in tr.items() if v["type"] == "sample"] == ["tau0_amp", "dtau0"]
    assert _is_truncnorm(tr["tau0_amp"]["fn"])
    assert isinstance(tr["dtau0"]["fn"], dist.Uniform)


# --------------------------------------------------------------------------- #
#  3. Site-order IDENTICAL across modes (uniform vs gauss) on the real model.
# --------------------------------------------------------------------------- #
def test_site_order_identical_across_modes_mini():
    """The mini-ctx site list is the same (names + order) in uniform and gauss modes."""
    s_uni = [k for k, v in _trace_tau0(_mini()).items() if v["type"] == "sample"]
    s_g = [k for k, v in _trace_tau0(_mini((1.0, 0.05), (0.0, 0.05))).items()
           if v["type"] == "sample"]
    assert s_uni == s_g == ["tau0_amp", "dtau0"]


@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI not present")
def test_priors_only_site_order_identical_uniform_vs_gauss():
    """_legb_priors_only's full sample-site list is IDENTICAL whether the τ₀ sites are uniform or
    the informative TruncatedNormal — constrain_fn relies on the order being mode-invariant."""
    ctx, d = C.build_legb_ctx(desi_kwargs=dict(z_lo=0.0, z_hi=2.6), use_xclass=True)
    ctx = ctx._replace(legs=[l for l in ctx.legs if l.name == "DESI"])
    s_uni = _sites(C._legb_priors_only, ctx)
    ctx_g = ctx._replace(tau0_amp_gauss=(1.0, 0.05), dtau0_gauss=(0.0, 0.05))
    s_g = _sites(C._legb_priors_only, ctx_g)
    assert s_uni == s_g, f"{s_uni} != {s_g}"
    # and the model twin agrees with priors-only on those sites too (the constrain_fn invariant).
    sims, _ = C.held_out_sims(d, fold=0)
    truth = C.make_truth_from_sim(d, sims[0], fold=0)
    mock_legs, _, _ = C.make_legb_mock(ctx_g, truth, jax.random.PRNGKey(0))
    core = C._mock_core_per_leg(ctx_g, truth)
    s_model = _sites(C._legb_model, ctx_g, mock_legs, core)
    assert s_model == s_g, f"model {s_model} != priors {s_g}"


# --------------------------------------------------------------------------- #
#  4. subDLA truth boost — scales ONLY index 1; input unmutated; boost=1 identity.
# --------------------------------------------------------------------------- #
def _truth_pack():
    rng = np.random.default_rng(0)
    return dict(
        theta9=rng.normal(size=9),
        tau0_global=rng.uniform(0.5, 1.5, size=5),
        alpha_hcd=np.array([0.27, 0.09, 0.0012]),     # (3,) [LLS, subDLA, DLA]
        alpha_hcd_z=rng.uniform(0.05, 0.5, size=(5, 3)),
        a_siiii=0.03,
    )


def test_subdla_boost_scales_only_index_1():
    """apply_subdla_truth_boost scales ONLY alpha_hcd[1] and alpha_hcd_z[:,1]; LLS(0)/DLA(2),
    theta, tau0, a_siiii untouched; the input dict is NOT mutated."""
    tp = _truth_pack()
    tp_ref = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in tp.items()}
    boost = 1.7
    out = C.apply_subdla_truth_boost(tp, boost)
    # subDLA column scaled.
    assert np.isclose(out["alpha_hcd"][1], tp_ref["alpha_hcd"][1] * boost)
    np.testing.assert_allclose(out["alpha_hcd_z"][:, 1], tp_ref["alpha_hcd_z"][:, 1] * boost)
    # LLS(0)/DLA(2) untouched.
    assert out["alpha_hcd"][0] == tp_ref["alpha_hcd"][0]
    assert out["alpha_hcd"][2] == tp_ref["alpha_hcd"][2]
    np.testing.assert_array_equal(out["alpha_hcd_z"][:, 0], tp_ref["alpha_hcd_z"][:, 0])
    np.testing.assert_array_equal(out["alpha_hcd_z"][:, 2], tp_ref["alpha_hcd_z"][:, 2])
    # theta / tau0 / a_siiii untouched.
    np.testing.assert_array_equal(out["theta9"], tp_ref["theta9"])
    np.testing.assert_array_equal(out["tau0_global"], tp_ref["tau0_global"])
    assert out["a_siiii"] == tp_ref["a_siiii"]
    # INPUT NOT mutated.
    np.testing.assert_array_equal(tp["alpha_hcd"], tp_ref["alpha_hcd"])
    np.testing.assert_array_equal(tp["alpha_hcd_z"], tp_ref["alpha_hcd_z"])


def test_subdla_boost_one_is_identity():
    """boost=1 reproduces the input arrays exactly (new arrays, same values)."""
    tp = _truth_pack()
    out = C.apply_subdla_truth_boost(tp, 1.0)
    np.testing.assert_array_equal(out["alpha_hcd"], tp["alpha_hcd"])
    np.testing.assert_array_equal(out["alpha_hcd_z"], tp["alpha_hcd_z"])


def test_lls_boost_still_only_index_0():
    """Confirm apply_lls_truth_boost still touches ONLY index 0 (the sibling invariant)."""
    tp = _truth_pack()
    tp_ref = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in tp.items()}
    boost = 1.5
    out = C.apply_lls_truth_boost(tp, boost)
    assert np.isclose(out["alpha_hcd"][0], tp_ref["alpha_hcd"][0] * boost)
    np.testing.assert_allclose(out["alpha_hcd_z"][:, 0], tp_ref["alpha_hcd_z"][:, 0] * boost)
    # subDLA(1)/DLA(2) untouched.
    assert out["alpha_hcd"][1] == tp_ref["alpha_hcd"][1]
    assert out["alpha_hcd"][2] == tp_ref["alpha_hcd"][2]
    np.testing.assert_array_equal(out["alpha_hcd_z"][:, 1], tp_ref["alpha_hcd_z"][:, 1])
    np.testing.assert_array_equal(out["alpha_hcd_z"][:, 2], tp_ref["alpha_hcd_z"][:, 2])
    # input unmutated.
    np.testing.assert_array_equal(tp["alpha_hcd"], tp_ref["alpha_hcd"])


def test_subdla_boost_handles_missing_alpha_hcd_z():
    """When alpha_hcd_z is absent (None / missing), only the pivot is scaled (no crash)."""
    tp = dict(alpha_hcd=np.array([0.27, 0.09, 0.0012]), alpha_hcd_z=None)
    out = C.apply_subdla_truth_boost(tp, 2.0)
    assert np.isclose(out["alpha_hcd"][1], 0.18)
    assert out["alpha_hcd_z"] is None
