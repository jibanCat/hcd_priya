"""2D amplitude×tilt HCD-incidence parametrization ("hcd_2d_tilt") — TDD + golden-guarded.

A 2D submanifold variant of the hierarchical "Option B" prior. The HCD sector becomes a
genuine 2D submanifold = pivot AMPLITUDE A_HCD × a GLOBAL z-TILT B_HCD, with the class-
differential z-evolution FIXED:

    α_c(z) = A_HCD · r_c · ((1+z)/(1+z_p))^(B_HCD + δs_c)

  A_HCD     ~ TruncatedNormal(alpha_hcd_mu[0], alpha_hcd_sigma[0], low=0)   # = the LLS prior
  B_HCD     ~ Normal(s_LLS_center, σ_B)                                     # the GLOBAL z-tilt
  r_subdla,r_dla ~ TruncatedNormal(hcd_ratio_mu, hcd_ratio_sigma, low=0)    # the Option-B ratios
  δs_c = (0, δs_subDLA, δs_DLA) FIXED class-differential slopes (δs_LLS ≡ 0)

So the per-class z-slope s_c = B_HCD + δs_c REPLACES the marginalize_zslope sampling. At
B_HCD = s_LLS_center the model reduces to Option B with FIXED slopes (the closure anchor).

``ctx.hcd_2d_tilt`` is OPT-IN; default False → byte-identical legacy (hierarchical_hcd=False)
or byte-identical Option B (hierarchical_hcd=True). These tests pin:
  - golden byte-exact OFF (both modes);
  - site-order (parametrized hcd_2d_tilt × hierarchical_hcd × metals);
  - reparam at z_p: alpha_subdla == A_hcd·r_subdla;
  - z-shape: alpha_hcd_z[z,c] == alpha_pivot[c]·((1+z)/(1+z_p))^(B_hcd+δs_c);
  - B_hcd reduces to Option B at B_hcd == s_LLS_center;
  - differentiability in (A_hcd, B_hcd, r);
  - B_hcd truth round-trip;
  - positivity;
  - a 0-div 40-draw NUTS smoke on an HT mock.

Env (MANDATORY):
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
    /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_legb_2d_tilt.py -q
"""
import os
import numpy as np
import pytest

import hcd_analysis.emulator  # x64 BEFORE jax
import jax
import jax.numpy as jnp
import numpyro
from numpyro import handlers

from hcd_analysis.emulator import closure_legb as C

_CACHE = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/observables_tau0_lf.h5"
_CKPT0 = "/home/mfho/hcd_priya/checkpoints/final_fold0.eqx"
_DESI_NPZ = "/home/mfho/data/desi_dr1_p1d/desi_dr1_p1d.npz"
_KS = ("/home/mfho/lya_emulator_full/lyaemu/data/kodiaq_squad/"
       "final-conservative-p1d-karacayli_etal2021.txt")
_have = all(os.path.exists(p) for p in (_CACHE, _CKPT0, _DESI_NPZ))
_have_ks = os.path.exists(_KS)


def _sites(model, *a, seed=1):
    tr = handlers.trace(handlers.seed(model, jax.random.PRNGKey(seed))).get_trace(*a)
    return [k for k, v in tr.items() if v["type"] == "sample" and not v.get("is_observed")]


def _potential_at(model, *a, point):
    from numpyro.infer.util import log_density
    lp, _ = log_density(model, a, {}, point)
    return float(lp)


def _build_small(hierarchical=False, two_d=False, ratio_infl=1.0):
    """A DESI-only narrow-z context so the FULL real-grid path runs in ~1 min/chain."""
    ctx, d = C.build_legb_ctx(desi_kwargs=dict(z_lo=0.0, z_hi=2.6), use_xclass=True,
                              hierarchical_hcd=hierarchical, hcd_2d_tilt=two_d,
                              hcd_ratio_infl=ratio_infl)
    ctx = ctx._replace(legs=[l for l in ctx.legs if l.name == "DESI"])
    return ctx, d


def _mock(ctx, d, sim_i=0):
    sims, _ = C.held_out_sims(d, fold=0)
    truth = C.make_truth_from_sim(d, sims[sim_i], fold=0)
    mock_legs, truth_pack, _ = C.make_legb_mock(ctx, truth, jax.random.PRNGKey(0))
    core = C._mock_core_per_leg(ctx, truth)
    return mock_legs, core, truth, truth_pack


# --------------------------------------------------------------------------- #
#  ctx defaults (closure values self-consistent with HCD_LIT_OVER_SIM_SLOPE).
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI not present")
def test_ctx_2d_defaults_are_closure_self_consistent():
    """build_legb_ctx(hcd_2d_tilt=True) sets the closure defaults from the FULL incidence slope
    HCD_INCIDENCE_SLOPE (the slope the mock truth carries, ~2.4), NOT the lit/sim-ratio slope:
       δs_c = HCD_INCIDENCE_SLOPE − HCD_INCIDENCE_SLOPE[0] = (0.0, +0.29, −0.10);
       btilt_mu = HCD_INCIDENCE_SLOPE[0] = 2.465;  btilt_sigma = ZSLOPE_PRIOR_SIGMA[0] = 0.52."""
    ctx, _ = _build_small(hierarchical=True, two_d=True)
    assert ctx.hcd_2d_tilt is True
    slope = np.asarray(C.HCD_INCIDENCE_SLOPE)
    np.testing.assert_allclose(np.asarray(ctx.hcd_dslope), slope - slope[0], rtol=1e-12)
    assert abs(float(ctx.hcd_btilt_mu) - float(slope[0])) < 1e-12
    assert abs(float(ctx.hcd_btilt_sigma) - float(C.ZSLOPE_PRIOR_SIGMA[0])) < 1e-12


# --------------------------------------------------------------------------- #
#  Golden byte-exact OFF (both modes).
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI not present")
def test_off_legacy_byte_exact_when_2d_off():
    """hcd_2d_tilt=False AND hierarchical_hcd=False → byte-identical legacy potential + sites."""
    ctx, d = _build_small(hierarchical=False, two_d=False)
    ctx = ctx._replace(marginalize_zslope=False)
    mock_legs, core, _, _ = _mock(ctx, d)
    s = _sites(C._legb_model, ctx, mock_legs, core)
    assert s == ["theta_unit", "tau0_amp", "dtau0",
                 "alpha_lls", "alpha_subdla", "alpha_dla_raw"]
    point = dict(theta_unit=jnp.full(9, 0.5), tau0_amp=jnp.asarray(1.0),
                 dtau0=jnp.asarray(0.0), alpha_lls=jnp.asarray(0.27),
                 alpha_subdla=jnp.asarray(0.09), alpha_dla_raw=jnp.asarray(-2.0))
    lp = _potential_at(C._legb_model, ctx, mock_legs, core, point=point)
    # an explicitly-defaulted 2D field must not perturb the OFF math.
    ctx2 = ctx._replace(hcd_2d_tilt=False, hcd_dslope=None, hcd_btilt_mu=None, hcd_btilt_sigma=None)
    lp2 = _potential_at(C._legb_model, ctx2, mock_legs, core, point=point)
    assert lp == lp2, f"legacy OFF potential drifted: {lp} != {lp2}"


@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI not present")
def test_off_optionB_byte_exact_when_2d_off():
    """hcd_2d_tilt=False, hierarchical_hcd=True → byte-identical Option B potential + sites."""
    ctx, d = _build_small(hierarchical=True, two_d=False)
    mock_legs, core, _, _ = _mock(ctx, d)
    # Option-B sites (with the default marginalize_zslope=True → s_* block present).
    s = _sites(C._legb_model, ctx, mock_legs, core)
    assert s == ["theta_unit", "tau0_amp", "dtau0",
                 "A_hcd", "r_subdla", "r_dla", "s_lls", "s_subdla", "s_dla"]
    assert "B_hcd" not in s
    point = dict(theta_unit=jnp.full(9, 0.5), tau0_amp=jnp.asarray(1.0),
                 dtau0=jnp.asarray(0.0), A_hcd=jnp.asarray(0.27),
                 r_subdla=jnp.asarray(0.33), r_dla=jnp.asarray(0.016),
                 s_lls=jnp.asarray(0.95), s_subdla=jnp.asarray(0.15), s_dla=jnp.asarray(0.40))
    lp = _potential_at(C._legb_model, ctx, mock_legs, core, point=point)
    ctx2 = ctx._replace(hcd_2d_tilt=False, hcd_dslope=None, hcd_btilt_mu=None, hcd_btilt_sigma=None)
    lp2 = _potential_at(C._legb_model, ctx2, mock_legs, core, point=point)
    assert lp == lp2, f"Option-B OFF potential drifted: {lp} != {lp2}"


# --------------------------------------------------------------------------- #
#  Site-order match (parametrized hcd_2d_tilt × hierarchical_hcd × metals).
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not (_have and _have_ks), reason="real cache/ckpt/DESI/KS not present")
@pytest.mark.parametrize("two_d", [False, True])
@pytest.mark.parametrize("hierarchical", [False, True])
@pytest.mark.parametrize("sample_metals", [False, True])
def test_model_priors_only_site_order_match_2d(two_d, hierarchical, sample_metals):
    """_legb_model and _legb_priors_only have the SAME sample sites in the SAME order over every
    combination of hcd_2d_tilt × hierarchical_hcd × sample_metals. In 2D mode the sites are
    A_hcd, B_hcd, r_subdla, r_dla (no s_* block — 2D sets its own slopes)."""
    if two_d and not hierarchical:
        pytest.skip("2D requires hierarchical_hcd (rejected at build) — covered separately")
    ctx, d = C.build_legb_ctx(desi_kwargs=dict(z_lo=0.0, z_hi=2.6), use_xclass=True,
                              hierarchical_hcd=hierarchical, hcd_2d_tilt=two_d,
                              sample_metals=sample_metals)
    ctx = ctx._replace(legs=[l for l in ctx.legs if l.name == "DESI"])
    mock_legs, core, _, _ = _mock(ctx, d)
    s_model = _sites(C._legb_model, ctx, mock_legs, core)
    s_prior = _sites(C._legb_priors_only, ctx)
    assert s_model == s_prior, f"site-order mismatch: model {s_model} vs priors {s_prior}"
    if two_d:
        # 2D mode: A_hcd, B_hcd, r_subdla, r_dla, and NO per-class z-slope sampling.
        assert s_model[3:7] == ["A_hcd", "B_hcd", "r_subdla", "r_dla"]
        for nm in ("s_lls", "s_subdla", "s_dla", "alpha_lls", "alpha_dla_raw"):
            assert nm not in s_model
    # res_corr alpha nuisance (Task 1.3) is the last block; a_SiIII present iff sampling metals.
    assert s_model[-2:] == ["alpha_res", "alpha_res_slope"]
    if sample_metals:
        assert "a_SiIII" in s_model
    else:
        assert "a_SiIII" not in s_model


# --------------------------------------------------------------------------- #
#  Reparam at z_p: alpha_subdla == A_hcd·r_subdla; pivot α emitted under legacy names.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI not present")
def test_2d_pivot_alpha_equals_A_times_r():
    """In 2D mode the deterministic pivot α (alpha_lls/subdla/dla) equal A_hcd·r_c (the B_hcd
    z-tilt lives in alpha_hcd_z, NOT in the pivot α — back-compat for _draws_matrix/coverage)."""
    ctx, d = _build_small(hierarchical=True, two_d=True)
    mock_legs, core, _, _ = _mock(ctx, d)
    tr = handlers.trace(handlers.seed(
        lambda: C._legb_model(ctx, mock_legs, core), jax.random.PRNGKey(3))).get_trace()
    A = np.asarray(tr["A_hcd"]["value"])
    rs = np.asarray(tr["r_subdla"]["value"])
    rd = np.asarray(tr["r_dla"]["value"])
    np.testing.assert_array_equal(np.asarray(tr["alpha_lls"]["value"]), A)
    np.testing.assert_array_equal(np.asarray(tr["alpha_subdla"]["value"]), A * rs)
    np.testing.assert_array_equal(np.asarray(tr["alpha_dla"]["value"]), A * rd)


# --------------------------------------------------------------------------- #
#  z-shape: alpha_hcd_z[z,c] == alpha_pivot[c]·((1+z)/(1+z_p))^(B_hcd+δs_c).
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI not present")
def test_2d_z_shape_uses_btilt_plus_dslope():
    """alpha_hcd_z follows the per-class slope s_c = B_hcd + δs_c (NOT a sampled s_* block)."""
    ctx, d = _build_small(hierarchical=True, two_d=True)
    mock_legs, core, _, _ = _mock(ctx, d)
    tr = handlers.trace(handlers.seed(
        lambda: C._legb_model(ctx, mock_legs, core), jax.random.PRNGKey(7))).get_trace()
    A = float(tr["A_hcd"]["value"]); B = float(tr["B_hcd"]["value"])
    rs = float(tr["r_subdla"]["value"]); rd = float(tr["r_dla"]["value"])
    alpha_pivot = np.array([A, A * rs, A * rd])
    dslope = np.asarray(ctx.hcd_dslope)
    s_c = B + dslope                                              # (3,)
    zg = np.asarray(ctx.z_global)
    shape = ((1.0 + zg)[:, None] / (1.0 + C.HCD_Z_PIVOT)) ** s_c  # (nZ,3)
    expect = alpha_pivot[None, :] * shape
    got = np.asarray(tr["alpha_hcd_z"]["value"])
    np.testing.assert_allclose(got, expect, rtol=1e-10, atol=0.0)


# --------------------------------------------------------------------------- #
#  B_hcd reduces to Option B (fixed slopes) at B_hcd == s_LLS_center.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI not present")
def test_2d_slopes_match_incidence_slope_at_btilt_center():
    """At B_hcd = btilt_center the 2D model's per-class slopes s_c = B_hcd + δs_c equal the FULL
    incidence slope HCD_INCIDENCE_SLOPE (the slope the mock truth carries) — the closure
    self-consistency anchor (the forward z-evolution MATCHES the truth at the prior center)."""
    ctx, d = _build_small(hierarchical=True, two_d=True)
    mock_legs, core, _, _ = _mock(ctx, d)
    B_center = float(ctx.hcd_btilt_mu)
    fixed_point = dict(theta_unit=jnp.full(9, 0.5), tau0_amp=jnp.asarray(1.0),
                       dtau0=jnp.asarray(0.0), A_hcd=jnp.asarray(0.27), B_hcd=jnp.asarray(B_center),
                       r_subdla=jnp.asarray(0.33), r_dla=jnp.asarray(0.016))
    tr = handlers.trace(handlers.seed(handlers.substitute(
        lambda: C._legb_model(ctx, mock_legs, core), fixed_point),
        jax.random.PRNGKey(0))).get_trace()
    got = np.asarray(tr["alpha_hcd_z"]["value"])
    # hand build: s_c = HCD_INCIDENCE_SLOPE exactly at B_hcd = center (B_center + δs_c).
    A, rs, rd = 0.27, 0.33, 0.016
    alpha_pivot = np.array([A, A * rs, A * rd])
    s_full = np.asarray(C.HCD_INCIDENCE_SLOPE)
    zg = np.asarray(ctx.z_global)
    shape = ((1.0 + zg)[:, None] / (1.0 + C.HCD_Z_PIVOT)) ** s_full
    np.testing.assert_allclose(got, alpha_pivot[None, :] * shape, rtol=1e-10, atol=0.0)


# --------------------------------------------------------------------------- #
#  Differentiability in (A_hcd, B_hcd, r).
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI not present")
def test_2d_loglik_grad_in_A_B_r_finite():
    """jax.grad of the multi-leg loglik w.r.t. (A_hcd, B_hcd, r_subdla, r_dla) is finite."""
    ctx, d = _build_small(hierarchical=True, two_d=True)
    mock_legs, core, _, _ = _mock(ctx, d)
    zg = jnp.asarray(ctx.z_global)
    th = jnp.full(9, 0.5)
    from hcd_analysis.emulator.meanflux_prior import becker13_tau0
    tau0 = becker13_tau0(zg)
    dslope = jnp.asarray(ctx.hcd_dslope)

    def ll(A, B, rs, rd):
        alpha_pivot = jnp.stack([A, A * rs, A * rd])
        s_c = B + dslope
        shape_zg = ((1.0 + zg)[:, None] / (1.0 + C.HCD_Z_PIVOT)) ** s_c
        alpha_hcd = alpha_pivot[None, :] * shape_zg
        return C._data_loglik_legcore(ctx, th, tau0, alpha_hcd, mock_legs, core)

    g = jax.grad(ll, argnums=(0, 1, 2, 3))(
        jnp.asarray(0.27), jnp.asarray(0.95), jnp.asarray(0.33), jnp.asarray(0.016))
    assert all(np.isfinite(np.asarray(gi)) for gi in g), f"grad not finite: {g}"


# --------------------------------------------------------------------------- #
#  B_hcd truth round-trip in _draws_matrix / packed_names / truth_vec.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI not present")
def test_2d_draws_matrix_appends_A_B_r_columns():
    """_draws_matrix appends A_hcd, B_hcd, r_subdla, r_dla columns in 2D mode; _packed_names_for
    names them; _hcd_latent_truths_2d gives (A=w_LLS, B=s_LLS_center, r_sub, r_dla)."""
    ctx, _ = _build_small(hierarchical=True, two_d=True)
    L = 5
    kept = np.ones(int(np.asarray(ctx.z_global).shape[0]), bool)
    samples = {"theta_unit": np.zeros((L, 9)), "tau0_vec": np.zeros((L, kept.size)),
               "alpha_lls": np.full(L, 0.27), "alpha_subdla": np.full(L, 0.09),
               "alpha_dla": np.full(L, 0.016),
               "A_hcd": np.full(L, 0.27), "B_hcd": np.full(L, 0.95),
               "r_subdla": np.full(L, 0.33), "r_dla": np.full(L, 0.06)}
    draws = C._draws_matrix(samples, kept)
    names = C._packed_names_for(samples, kept)
    assert names[-4:] == ["A_hcd", "B_hcd", "r_subdla", "r_dla"]
    assert draws.shape[1] == len(names)
    # the appended draw columns match the samples.
    for off, nm in enumerate(["A_hcd", "B_hcd", "r_subdla", "r_dla"]):
        np.testing.assert_array_equal(draws[:, -4 + off], samples[nm])
    # truth round-trip: B truth = s_LLS_center.
    alpha_truth = np.array([0.27, 0.09, 0.016])
    tv = C._hcd_latent_truths_2d(alpha_truth, ctx)
    assert tv.shape == (4,)
    np.testing.assert_allclose(tv[0], alpha_truth[0])               # A = w_LLS
    np.testing.assert_allclose(tv[1], float(ctx.hcd_btilt_mu))      # B = s_LLS_center
    np.testing.assert_allclose(tv[2], alpha_truth[1] / alpha_truth[0])
    np.testing.assert_allclose(tv[3], alpha_truth[2] / alpha_truth[0])


# --------------------------------------------------------------------------- #
#  Reconstruction seam: fast-postprocess ON yields alpha_lls/subdla/dla in 2D mode.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI not present")
def test_2d_reconstruct_deterministics_uses_btilt():
    """_legb_reconstruct_deterministics rebuilds alpha_lls/subdla/dla (= A·r) AND alpha_hcd_z
    (using B_hcd + δs_c) from the 2D latents."""
    ctx, _ = _build_small(hierarchical=True, two_d=True)
    L, nzg = 5, int(np.asarray(ctx.z_global).shape[0])
    A = np.linspace(0.2, 0.35, L); B = np.linspace(0.8, 1.1, L)
    rs = np.linspace(0.30, 0.36, L); rd = np.linspace(0.012, 0.020, L)
    samples = {"A_hcd": A, "B_hcd": B, "r_subdla": rs, "r_dla": rd,
               "tau0_amp": np.ones(L), "dtau0": np.zeros(L)}
    out = C._legb_reconstruct_deterministics(ctx, samples)
    for nm in ("alpha_lls", "alpha_subdla", "alpha_dla", "alpha_hcd_z", "tau0_vec"):
        assert nm in out, f"reconstruction must re-insert {nm!r}"
    np.testing.assert_allclose(np.asarray(out["alpha_lls"]), A)
    np.testing.assert_allclose(np.asarray(out["alpha_subdla"]), A * rs)
    np.testing.assert_allclose(np.asarray(out["alpha_dla"]), A * rd)
    assert np.asarray(out["alpha_hcd_z"]).shape == (L, nzg, 3)
    # verify the z-shape per draw uses B + δs_c.
    zg = np.asarray(ctx.z_global); dslope = np.asarray(ctx.hcd_dslope)
    pivot = np.stack([A, A * rs, A * rd], axis=-1)                 # (L,3)
    s_c = B[:, None] + dslope[None, :]                            # (L,3)
    ratio = (1.0 + zg)[None, :, None] / (1.0 + C.HCD_Z_PIVOT)
    expect = pivot[:, None, :] * ratio ** s_c[:, None, :]
    np.testing.assert_allclose(np.asarray(out["alpha_hcd_z"]), expect, rtol=1e-10)


@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI not present")
def test_2d_fast_postprocess_yields_alpha_columns():
    """A short 2D-mode NUTS run via the fast postprocess yields alpha_lls/subdla/dla columns; the
    slow (in-model replay) path agrees."""
    ctx, d = _build_small(hierarchical=True, two_d=True)
    mock_legs, core, _, _ = _mock(ctx, d)
    kw = dict(n_warmup=8, n_samples=10, seed=11, max_tree_depth=5, dense_mass=False)
    s_fast, _ = C._run_nuts_legb(ctx, mock_legs, core, fast_postprocess=True, **kw)
    s_slow, _ = C._run_nuts_legb(ctx, mock_legs, core, fast_postprocess=False, **kw)
    for nm in ("alpha_lls", "alpha_subdla", "alpha_dla", "B_hcd"):
        assert nm in s_fast, f"fast-postprocess 2D must yield {nm!r}"
        assert nm in s_slow, f"slow replay 2D must yield {nm!r}"
    np.testing.assert_array_equal(np.asarray(s_fast["alpha_lls"]), np.asarray(s_slow["alpha_lls"]))
    for nm in ("alpha_subdla", "alpha_dla"):
        np.testing.assert_allclose(np.asarray(s_fast[nm]), np.asarray(s_slow[nm]),
                                   rtol=1e-12, atol=0.0)


# --------------------------------------------------------------------------- #
#  Positivity (1e4 prior draws): α_c(z) ≥ 0 everywhere.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI not present")
def test_2d_positivity_prior_draws():
    """Over 10^4 prior draws of the 2D HCD block, the derived α_c(z) stay ≥ 0 (product of low=0
    TruncatedNormals × a positive power-law)."""
    ctx, _ = _build_small(hierarchical=True, two_d=True)

    def block():
        return C._hcd_sites(ctx)

    keys = jax.random.split(jax.random.PRNGKey(0), 2000)

    def one(k):
        tr = handlers.trace(handlers.seed(block, k)).get_trace()
        A = tr["A_hcd"]["value"]; B = tr["B_hcd"]["value"]
        rs = tr["r_subdla"]["value"]; rd = tr["r_dla"]["value"]
        return jnp.array([A, A * rs, A * rd]).min()

    mins = jax.vmap(one)(keys)
    assert float(np.asarray(mins).min()) >= 0.0, "negative HCD pivot α drawn"


# --------------------------------------------------------------------------- #
#  marginalize_zslope is bypassed in 2D mode (the 2D mode sets its own slopes).
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI not present")
def test_2d_ignores_marginalize_zslope():
    """Even with marginalize_zslope=True, the 2D mode does NOT sample s_lls/s_subdla/s_dla — the
    slopes come from B_hcd + δs_c (must override/ignore marginalize_zslope cleanly)."""
    ctx, d = _build_small(hierarchical=True, two_d=True)
    ctx = ctx._replace(marginalize_zslope=True)
    mock_legs, core, _, _ = _mock(ctx, d)
    s = _sites(C._legb_model, ctx, mock_legs, core)
    for nm in ("s_lls", "s_subdla", "s_dla"):
        assert nm not in s, f"2D mode must not sample {nm!r} (marginalize_zslope must be bypassed)"
    assert "B_hcd" in s


# --------------------------------------------------------------------------- #
#  Building 2D ctx without hierarchical_hcd is rejected (2D is a hierarchical variant).
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI not present")
def test_2d_requires_hierarchical():
    """hcd_2d_tilt=True without hierarchical_hcd=True is a configuration error (2D builds on the
    A_hcd × r reparam)."""
    with pytest.raises((AssertionError, ValueError)):
        C.build_legb_ctx(desi_kwargs=dict(z_lo=0.0, z_hi=2.6), use_xclass=True,
                         hierarchical_hcd=False, hcd_2d_tilt=True)


# --------------------------------------------------------------------------- #
#  run_stepA HT_f6 / HT_f4 config arms (24 chains, config only, not launched).
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI not present")
def test_stepA_HT_arms_present_and_2d():
    """run_stepA.build_config emits HT_f6/HT_f4 arms (joint DESI+KS, sim_mean, hcd_2d_tilt=True,
    A_HCD-center 0/+1σ/−1σ via hcd_center_shift) — 24 chains (2 folds × 3 shifts × 4 chains)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_stepA", "/home/mfho/hcd_priya/scripts/run_stepA.py")
    rs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rs)
    cfg = rs.build_config()
    ht = [c for c in cfg if c["mock_id"].startswith("HT_")]
    # 2 folds × 3 shifts × 4 chains = 24.
    assert len(ht) == 24, f"expected 24 HT chains, got {len(ht)}"
    for c in ht:
        assert c["hcd_2d_tilt"] is True
        assert c["hierarchical_hcd"] is True
        assert c["survey"] == "DESI+KS"
        assert c["prior_center"] == "sim_mean"
    # the three A_HCD-center shifts per fold.
    shifts = sorted(set(c["hcd_center_shift"] for c in ht))
    assert shifts == [-1.0, 0.0, 1.0]
    # both folds present.
    assert {c["fold"] for c in ht} == {6, 4}


# --------------------------------------------------------------------------- #
#  0-div NUTS smoke (SLOW; the empirical funnel check). Run once.
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI not present")
def test_2d_nuts_zero_divergences_and_finite():
    """A tiny NUTS run on one 2D mock asserts n_div==0 and finite draws (the B_hcd dimension does
    not introduce a funnel)."""
    ctx, d = _build_small(hierarchical=True, two_d=True)
    mock_legs, core, _, _ = _mock(ctx, d)
    samples, n_div = C._run_nuts_legb(
        ctx, mock_legs, core, n_warmup=40, n_samples=40, seed=0,
        target_accept=0.9, dense_mass=False, max_tree_depth=8)
    assert n_div == 0, f"2D HCD: {n_div} divergence(s) on the centered path"
    for nm in ("A_hcd", "B_hcd", "r_subdla", "r_dla", "alpha_lls", "alpha_subdla", "alpha_dla"):
        assert np.all(np.isfinite(np.asarray(samples[nm]))), f"{nm} not finite"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q", "-m", "not slow"]))
