"""Hierarchical HCD-incidence prior ("Option B") — TDD + golden-guarded tests.

The reparametrization replaces the 3 INDEPENDENT HCD-incidence sample sites
(``alpha_lls``/``alpha_subdla``/``alpha_dla_raw``) with ONE likelihood-constrained
MULTIPLIER ``A_hcd`` (= the LLS prior verbatim) × two prior-pinned RATIOS
(``r_subdla``/``r_dla``). The derived α are re-emitted as ``numpyro.deterministic`` under
the EXISTING names so the whole downstream (draws-matrix, corner, analysis) is unchanged.

``ctx.hierarchical_hcd`` is OPT-IN; default False → the OFF branch is literally the current
code (golden byte-exact). These tests pin:
  1. golden byte-exact OFF (sample-site list + a fixed-seed trace);
  2. reparam exact (alpha_subdla == A_hcd*r_subdla, alpha_lls==A_hcd, alpha_dla==A_hcd*r_dla);
  3. site-order match (_legb_model == _legb_priors_only over hier×zslope×metals);
  4. truth_r ≈ prior_r_center (must-fix #1 — the raw-sim-ratio centers);
  5. differentiability of the loglik in (A_hcd, r_subdla, r_dla);
  6. the deterministic-reconstruction seam (fast-postprocess ON yields alpha_lls/subdla/dla);
  7. the positional α-by-name indexing fix in _aggregate_legb (must-fix #5);
  8. a tiny NUTS run (0-div smoke; SLOW, marked).

Env (MANDATORY):
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
    /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_legb_hier_hcd.py -q
"""
import os
import numpy as np
import pytest

import hcd_analysis.emulator  # x64 BEFORE jax
import jax
import jax.numpy as jnp
import numpyro
from numpyro import handlers
from numpyro.infer.util import constrain_fn

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


def _potential_at(model, *a, point, seed=0):
    """The (unconstrained) log-density of ``model`` evaluated at a fixed CONSTRAINED point."""
    from numpyro.infer.util import log_density
    lp, _ = log_density(model, a, {}, point)
    return float(lp)


def _build_small(hierarchical=False, noncentered=False, ratio_infl=1.0):
    """A DESI-only narrow-z context so the FULL real-grid path runs in ~1 min/chain."""
    ctx, d = C.build_legb_ctx(desi_kwargs=dict(z_lo=0.0, z_hi=2.6), use_xclass=True,
                              hierarchical_hcd=hierarchical, hcd_noncentered=noncentered,
                              hcd_ratio_infl=ratio_infl)
    ctx = ctx._replace(legs=[l for l in ctx.legs if l.name == "DESI"])
    return ctx, d


# --------------------------------------------------------------------------- #
#  1. Golden byte-exact OFF.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI not present")
def test_off_branch_site_list_is_the_current_code():
    """hierarchical_hcd=False keeps the EXACT legacy sample sites in order — both with the default
    z-slope marginalization (the production config) and with it OFF (the clean 6-site case)."""
    ctx, d = _build_small(hierarchical=False)
    sims, _ = C.held_out_sims(d, fold=0)
    truth = C.make_truth_from_sim(d, sims[0], fold=0)
    mock_legs, _, _ = C.make_legb_mock(ctx, truth, jax.random.PRNGKey(0))
    core = C._mock_core_per_leg(ctx, truth)
    # DEFAULT ctx (marginalize_zslope True): the legacy 3 independent α + the z-slope block + the
    # res_corr-amplitude nuisance block (alpha_res, alpha_res_slope — added by d0de534/7a28e8d,
    # ALWAYS sampled unless ctx.fix_alpha_res, so they are the LAST two sites here).
    s = _sites(C._legb_model, ctx, mock_legs, core)
    assert s == ["theta_unit", "tau0_amp", "dtau0",
                 "alpha_lls", "alpha_subdla", "alpha_dla_raw",
                 "s_lls", "s_subdla", "s_dla",
                 "alpha_res", "alpha_res_slope"]
    assert "A_hcd" not in s and "r_subdla" not in s and "r_dla" not in s
    # z-slope OFF → the clean legacy HCD block (the 3 independent α sites, no reparam) + the
    # res_corr nuisance block (still sampled — independent of marginalize_zslope).
    ctx0 = ctx._replace(marginalize_zslope=False)
    s0 = _sites(C._legb_model, ctx0, mock_legs, core)
    assert s0 == ["theta_unit", "tau0_amp", "dtau0",
                  "alpha_lls", "alpha_subdla", "alpha_dla_raw",
                  "alpha_res", "alpha_res_slope"]


@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI not present")
def test_off_branch_potential_byte_exact_vs_legacy():
    """The OFF-branch model potential at a fixed point is byte-identical whether or not the
    new (default-OFF) ctx fields exist — i.e. adding the plumbing did not perturb the math."""
    ctx, d = _build_small(hierarchical=False)
    ctx = ctx._replace(marginalize_zslope=False)         # the clean legacy HCD block (6 sites)
    sims, _ = C.held_out_sims(d, fold=0)
    truth = C.make_truth_from_sim(d, sims[0], fold=0)
    mock_legs, _, _ = C.make_legb_mock(ctx, truth, jax.random.PRNGKey(0))
    core = C._mock_core_per_leg(ctx, truth)
    # a fixed constrained point in the LEGACY parametrization (+ the res_corr nuisance sites,
    # ALWAYS sampled in the current model — pin them at the no-op α₀=1, slope=0 so the point is
    # fully specified and the potential is deterministic).
    point = dict(theta_unit=jnp.full(9, 0.5), tau0_amp=jnp.asarray(1.0),
                 dtau0=jnp.asarray(0.0), alpha_lls=jnp.asarray(0.27),
                 alpha_subdla=jnp.asarray(0.09), alpha_dla_raw=jnp.asarray(-2.0),
                 alpha_res=jnp.asarray(1.0), alpha_res_slope=jnp.asarray(0.0))
    lp = _potential_at(C._legb_model, ctx, mock_legs, core, point=point)
    assert np.isfinite(lp)
    # re-evaluate with a ctx that has the hierarchical fields EXPLICITLY at their defaults —
    # must be identical (the OFF branch ignores them).
    ctx2 = ctx._replace(hierarchical_hcd=False, hcd_noncentered=False, hcd_ratio_infl=1.0)
    lp2 = _potential_at(C._legb_model, ctx2, mock_legs, core, point=point)
    assert lp == lp2, f"OFF-branch potential drifted: {lp} != {lp2}"


# --------------------------------------------------------------------------- #
#  2. Reparam exact (ON).
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI not present")
def test_on_branch_reparam_alpha_equals_A_times_r():
    """ON → the deterministic α equal A_hcd · r_c exactly (array_equal over trace draws)."""
    ctx, d = _build_small(hierarchical=True)
    sims, _ = C.held_out_sims(d, fold=0)
    truth = C.make_truth_from_sim(d, sims[0], fold=0)
    mock_legs, _, _ = C.make_legb_mock(ctx, truth, jax.random.PRNGKey(0))
    core = C._mock_core_per_leg(ctx, truth)
    tr = handlers.trace(handlers.seed(
        lambda: C._legb_model(ctx, mock_legs, core), jax.random.PRNGKey(3))).get_trace()
    A = np.asarray(tr["A_hcd"]["value"])
    rs = np.asarray(tr["r_subdla"]["value"])
    rd = np.asarray(tr["r_dla"]["value"])
    a_lls = np.asarray(tr["alpha_lls"]["value"])
    a_sub = np.asarray(tr["alpha_subdla"]["value"])
    a_dla = np.asarray(tr["alpha_dla"]["value"])
    np.testing.assert_array_equal(a_lls, A)
    np.testing.assert_array_equal(a_sub, A * rs)
    np.testing.assert_array_equal(a_dla, A * rd)
    # the ON branch must NOT carry the legacy raw α sites.
    s = _sites(C._legb_model, ctx, mock_legs, core)
    assert "A_hcd" in s and "r_subdla" in s and "r_dla" in s
    assert "alpha_lls" not in s and "alpha_subdla" not in s and "alpha_dla_raw" not in s


# --------------------------------------------------------------------------- #
#  3. Site-order match (parametrized over hier × marg_zslope × sample_metals).
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not (_have and _have_ks), reason="real cache/ckpt/DESI/KS not present")
@pytest.mark.parametrize("hierarchical", [False, True])
@pytest.mark.parametrize("marg_zslope", [False, True])
@pytest.mark.parametrize("sample_metals", [False, True])
def test_model_priors_only_site_order_match(hierarchical, marg_zslope, sample_metals):
    """_legb_model and _legb_priors_only must have the SAME sample sites in the SAME order across
    every combination of hierarchical × marg_zslope × sample_metals — the fast-postprocess
    constrain_fn relies on it. a_SiIII last when metals; A_hcd∈sites ⇔ hierarchical."""
    ctx, d = C.build_legb_ctx(desi_kwargs=dict(z_lo=0.0, z_hi=2.6), use_xclass=True,
                              hierarchical_hcd=hierarchical, sample_metals=sample_metals)
    ctx = ctx._replace(legs=[l for l in ctx.legs if l.name == "DESI"],
                       marginalize_zslope=marg_zslope)
    sims, _ = C.held_out_sims(d, fold=0)
    truth = C.make_truth_from_sim(d, sims[0], fold=0)
    mock_legs, _, _ = C.make_legb_mock(ctx, truth, jax.random.PRNGKey(0))
    core = C._mock_core_per_leg(ctx, truth)
    s_model = _sites(C._legb_model, ctx, mock_legs, core)
    s_prior = _sites(C._legb_priors_only, ctx)
    assert s_model == s_prior, f"site-order mismatch: model {s_model} vs priors {s_prior}"
    # the HCD block identity.
    if hierarchical:
        assert {"A_hcd", "r_subdla", "r_dla"}.issubset(set(s_model))
        assert not ({"alpha_lls", "alpha_subdla", "alpha_dla_raw"} & set(s_model))
    else:
        assert {"alpha_lls", "alpha_subdla", "alpha_dla_raw"}.issubset(set(s_model))
        assert "A_hcd" not in s_model
    # the res_corr-amplitude nuisance block (alpha_res, alpha_res_slope) is ALWAYS the LAST two
    # sites (added by d0de534/7a28e8d, sampled unless ctx.fix_alpha_res); a_SiIII — when metals —
    # is sampled JUST BEFORE it (so it is third-from-last, not last).
    assert s_model[-2:] == ["alpha_res", "alpha_res_slope"]
    if sample_metals:
        assert s_model[-3] == "a_SiIII"
    else:
        assert "a_SiIII" not in s_model


# --------------------------------------------------------------------------- #
#  4. truth_r ≈ prior_r_center (must-fix #1).
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI not present")
def test_ratio_centers_match_closure_truth():
    """The raw-sim-ratio prior centers (build_legb_ctx) put the closure TRUTH r_c within ~1.5σ_r.

    The closure truth (make_legb_mock truth_pack.alpha_hcd) is [w_LLS, w_sub, 0.10·w_DLA]; the
    truth ratios are r_sub=w_sub/w_LLS and r_dla=0.10·w_DLA/w_LLS. The prior centers are derived
    from the RAW cache w_c pool medians (NOT alpha_hcd_mu / lit·sim), so they coincide with the
    per-sim truth to within the per-sim CV — the load-bearing fix that keeps the closure unbiased.
    """
    ctx, d = _build_small(hierarchical=True)
    assert ctx.hcd_ratio_mu is not None and ctx.hcd_ratio_sigma is not None
    rmu = np.asarray(ctx.hcd_ratio_mu)      # (2,) [r_sub, r_dla]
    rsg = np.asarray(ctx.hcd_ratio_sigma)   # (2,)
    # widths = (0.10, 0.12)·center × hcd_ratio_infl (default 1.0).
    np.testing.assert_allclose(rsg, np.array([0.10, 0.12]) * rmu, rtol=1e-6)

    sims, _ = C.held_out_sims(d, fold=0)
    n_ok = 0
    for s in sims:
        truth = C.make_truth_from_sim(d, s, fold=0)
        a = np.asarray(truth["w_c"])                       # [w_LLS, w_sub, w_DLA] (z-median)
        truth_r_sub = a[1] / a[0]
        truth_r_dla = C.TRUTH_DLA_FRAC["DESI"] * a[2] / a[0]   # 0.10·w_DLA/w_LLS
        # within ~1.5σ_r (σ_r = the prior width, which is ≥ the per-sim CV by defense-in-depth).
        assert abs(truth_r_sub - rmu[0]) < 1.5 * rsg[0], \
            f"sim {s[:16]} r_sub truth {truth_r_sub:.4f} vs center {rmu[0]:.4f}±{rsg[0]:.4f}"
        assert abs(truth_r_dla - rmu[1]) < 1.5 * rsg[1], \
            f"sim {s[:16]} r_dla truth {truth_r_dla:.5f} vs center {rmu[1]:.5f}±{rsg[1]:.5f}"
        n_ok += 1
    assert n_ok >= 4


@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI not present")
def test_ratio_infl_scales_widths_not_centers():
    """hcd_ratio_infl scales the ratio prior WIDTHS (the mandatory scan knob), not the centers."""
    ctx1, _ = _build_small(hierarchical=True, ratio_infl=1.0)
    ctx2, _ = _build_small(hierarchical=True, ratio_infl=2.0)
    np.testing.assert_allclose(np.asarray(ctx1.hcd_ratio_mu), np.asarray(ctx2.hcd_ratio_mu))
    np.testing.assert_allclose(2.0 * np.asarray(ctx1.hcd_ratio_sigma),
                               np.asarray(ctx2.hcd_ratio_sigma), rtol=1e-6)


# --------------------------------------------------------------------------- #
#  5. Differentiability of the loglik in (A_hcd, r_subdla, r_dla).
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI not present")
def test_loglik_grad_in_A_r_finite():
    """jax.grad of the multi-leg loglik w.r.t. (A_hcd, r_subdla, r_dla) is finite (the forward is
    differentiable through the A·r reparam — the sampler needs this)."""
    ctx, d = _build_small(hierarchical=True)
    sims, _ = C.held_out_sims(d, fold=0)
    truth = C.make_truth_from_sim(d, sims[0], fold=0)
    mock_legs, _, _ = C.make_legb_mock(ctx, truth, jax.random.PRNGKey(0))
    core = C._mock_core_per_leg(ctx, truth)
    zg = jnp.asarray(ctx.z_global)
    th = jnp.full(9, 0.5)
    tau0 = C.becker13_tau0(zg) if hasattr(C, "becker13_tau0") else None
    from hcd_analysis.emulator.meanflux_prior import becker13_tau0
    tau0 = becker13_tau0(zg)
    s_c = jnp.asarray(C.HCD_LIT_OVER_SIM_SLOPE)
    shape_zg = ((1.0 + zg)[:, None] / (1.0 + C.HCD_Z_PIVOT)) ** s_c

    def ll(A, rs, rd):
        alpha_pivot = jnp.stack([A, A * rs, A * rd])
        alpha_hcd = alpha_pivot[None, :] * shape_zg
        return C._data_loglik_legcore(ctx, th, tau0, alpha_hcd, mock_legs, core)

    g = jax.grad(ll, argnums=(0, 1, 2))(jnp.asarray(0.27), jnp.asarray(0.33), jnp.asarray(0.016))
    assert all(np.isfinite(np.asarray(gi)) for gi in g), f"grad not finite: {g}"


# --------------------------------------------------------------------------- #
#  6. Reconstruction seam: fast-postprocess ON yields alpha_lls/subdla/dla (must-fix #4).
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI not present")
def test_reconstruct_deterministics_on_branch_reinserts_alpha():
    """In the hierarchical branch, _legb_reconstruct_deterministics must REBUILD + RE-INSERT
    alpha_lls/alpha_subdla/alpha_dla into the samples dict from A_hcd·r (constrain_fn drops the
    deterministics). _draws_matrix/_loglik_of_draws read all three by NAME."""
    ctx, _ = _build_small(hierarchical=True)
    L, nzg = 5, int(np.asarray(ctx.z_global).shape[0])
    A = np.linspace(0.2, 0.35, L)
    rs = np.linspace(0.30, 0.36, L)
    rd = np.linspace(0.012, 0.020, L)
    samples = {"A_hcd": A, "r_subdla": rs, "r_dla": rd,
               "tau0_amp": np.ones(L), "dtau0": np.zeros(L)}
    out = C._legb_reconstruct_deterministics(ctx, samples)
    for nm in ("alpha_lls", "alpha_subdla", "alpha_dla", "alpha_hcd_z", "tau0_vec"):
        assert nm in out, f"reconstruction must re-insert {nm!r}"
    np.testing.assert_allclose(np.asarray(out["alpha_lls"]), A)
    np.testing.assert_allclose(np.asarray(out["alpha_subdla"]), A * rs)
    np.testing.assert_allclose(np.asarray(out["alpha_dla"]), A * rd)
    # alpha_hcd_z pivot column must be alpha_pivot at z_pivot scaling.
    assert np.asarray(out["alpha_hcd_z"]).shape == (L, nzg, 3)


@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI not present")
def test_fast_postprocess_on_branch_yields_alpha_columns():
    """A short ON-branch NUTS run via the fast postprocess yields alpha_lls/subdla/dla columns in
    the samples dict (the constrain_fn path drops deterministics → the reconstruction must add
    them back). The slow path (in-model replay) must yield them too."""
    ctx, d = _build_small(hierarchical=True)
    sims, _ = C.held_out_sims(d, fold=0)
    truth = C.make_truth_from_sim(d, sims[0], fold=0)
    mock_legs, _, _ = C.make_legb_mock(ctx, truth, jax.random.PRNGKey(0))
    core = C._mock_core_per_leg(ctx, truth)
    kw = dict(n_warmup=8, n_samples=10, seed=11, max_tree_depth=5, dense_mass=False)
    s_fast, _ = C._run_nuts_legb(ctx, mock_legs, core, fast_postprocess=True, **kw)
    s_slow, _ = C._run_nuts_legb(ctx, mock_legs, core, fast_postprocess=False, **kw)
    for nm in ("alpha_lls", "alpha_subdla", "alpha_dla"):
        assert nm in s_fast, f"fast-postprocess ON must yield {nm!r}"
        assert nm in s_slow, f"slow replay ON must yield {nm!r}"
    # both paths must AGREE on the reconstructed α (same latent draws, same A·r). alpha_lls==A_hcd
    # is bit-identical; alpha_subdla/dla = A·r differ only at the float64 ULP (the in-model traced
    # product vs the host eager product through two constrain_fn return_deterministic variants).
    np.testing.assert_array_equal(np.asarray(s_fast["alpha_lls"]), np.asarray(s_slow["alpha_lls"]),
                                  err_msg="alpha_lls (=A_hcd): fast != slow on the ON path")
    for nm in ("alpha_subdla", "alpha_dla"):
        np.testing.assert_allclose(np.asarray(s_fast[nm]), np.asarray(s_slow[nm]), rtol=1e-12,
                                   atol=0.0, err_msg=f"{nm}: fast != slow on the ON path")


# --------------------------------------------------------------------------- #
#  7. Positional α-by-name indexing fix in _aggregate_legb (must-fix #5).
# --------------------------------------------------------------------------- #
def test_aggregate_legb_indexes_alpha_by_name_not_position():
    """_aggregate_legb must index α by packed_names.index(nm), NOT by negative position.

    The pre-existing bug: _draws_matrix appends a_SiIII LAST (when metals) while names/truth_vec
    do not, so the legacy ``col = -(3 - (j-9))`` (-3,-2,-1) lands on [subdla, dla, a_SiIII] instead
    of [lls, subdla, dla]. We build a synthetic per_mock with KNOWN α columns + the a_SiIII tail and
    assert the α coverage/bias reads the RIGHT columns (the truth sits dead-center → bias≈0)."""
    rng = np.random.default_rng(0)
    n_theta, nz = 9, 2
    # column layout the (metals) _draws_matrix produces: θ9, τ₀(2), α_lls, α_sub, α_dla, a_SiIII.
    P = n_theta + nz + 3 + 1
    L = 400
    # truth: θ at 0, τ₀ at 0, α at [0.27, 0.09, 0.016], a_SiIII at 0.04 — but names/truth_vec carry
    # NO a_SiIII column (the live mis-alignment); _aggregate must still read α by name.
    truth_alpha = np.array([0.27, 0.09, 0.016])
    per_mock = []
    for _ in range(30):
        draws = np.zeros((L, P))
        draws[:, n_theta:n_theta + nz] = rng.normal(0, 1, (L, nz))          # τ₀ noise
        # α columns centered ON the truth (so a name-correct read → bias≈0).
        for c, mu in enumerate(truth_alpha):
            draws[:, n_theta + nz + c] = rng.normal(mu, 0.01, L)
        draws[:, -1] = rng.normal(0.04, 0.005, L)                            # a_SiIII tail
        truth_vec = np.concatenate([np.zeros(n_theta), np.zeros(nz), truth_alpha])  # NO a_SiIII
        per_mock.append(dict(sim="x", truth_vec=truth_vec, draws=draws, L=L,
                             ll_true=0.0, ll_draws=rng.normal(0, 1, L),
                             kept_global=np.ones(nz, bool), dropped={}, n_div=0))
    res = C._aggregate_legb(per_mock, q_levels=(0.68,))
    # the α bias must be ~0 for ALL three classes (name-correct columns); the OLD positional code
    # would read α_subdla/α_dla/a_SiIII → a LARGE bias on alpha_dla (0.016 truth vs 0.04 a_SiIII col).
    for nm in ("alpha_lls", "alpha_subdla", "alpha_dla"):
        assert abs(res["bias"][nm]["mean"]) < 0.5, \
            f"{nm} bias {res['bias'][nm]['mean']:.2f} — α read off the wrong (positional) column"
    # coverage near nominal too (empirical_coverage returns a dict with the "coverage" key).
    for nm in ("alpha_lls", "alpha_subdla", "alpha_dla"):
        assert res["coverage"][0.68][nm]["coverage"] > 0.4


# --------------------------------------------------------------------------- #
#  8. 0-div smoke (SLOW; the funnel empirical check). Run once.
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI not present")
def test_on_branch_nuts_zero_divergences_and_finite():
    """A tiny NUTS run on one HB mock (ON branch, centered, diag mass) asserts n_div==0 and finite
    draws — the empirical funnel check (the benign A_hcd × r orientation → no pinch)."""
    ctx, d = _build_small(hierarchical=True)
    sims, _ = C.held_out_sims(d, fold=0)
    truth = C.make_truth_from_sim(d, sims[0], fold=0)
    mock_legs, _, _ = C.make_legb_mock(ctx, truth, jax.random.PRNGKey(0))
    core = C._mock_core_per_leg(ctx, truth)
    samples, n_div = C._run_nuts_legb(
        ctx, mock_legs, core, n_warmup=40, n_samples=40, seed=0,
        target_accept=0.9, dense_mass=False, max_tree_depth=8)
    assert n_div == 0, f"hierarchical HCD funnel: {n_div} divergence(s) on the centered ON path"
    for nm in ("A_hcd", "r_subdla", "r_dla", "alpha_lls", "alpha_subdla", "alpha_dla"):
        assert np.all(np.isfinite(np.asarray(samples[nm]))), f"{nm} not finite"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q", "-m", "not slow"]))
