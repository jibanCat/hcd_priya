"""Option-b spectral-resolution nuisance f_res: a forward-only 2-param b_res(z) = f_res_amp *
((1+z)/(1+z_p))^f_res_slope threaded through _resolution_factor = exp(2 b_res k^2 R_z^2). Cloned from
the alpha_res forward-only pattern. Default OFF (ctx.sample_res=False) -> byte-identical golden.

4-lens-verified design (2026-07-02): keep EXP; TIGHT prior on physics grounds (amp ~ Normal(0,0.02));
per-z-block rank-1 cov surgery for option-b; forward-only (not in truth -> no closure cancellation).

Env: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_legb_resolution.py -q
"""
import os
import numpy as np
import pytest
import hcd_analysis.emulator  # x64 before jax
import jax
import jax.numpy as jnp
from numpyro import handlers
from numpyro.infer.util import constrain_fn
from hcd_analysis.emulator import closure_legb as C
from hcd_analysis.emulator import data_likelihood as DL

_CACHE = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/observables_tau0_lf.h5"
_CKPT0 = "/home/mfho/hcd_priya/checkpoints/final_fold0.eqx"
_DESI_NPZ = "/home/mfho/data/desi_dr1_p1d/desi_dr1_p1d.npz"
_EBOSS_NPZ = "/home/mfho/data/eboss_dr14_p1d/eboss_dr14_p1d.npz"
_have = all(os.path.exists(p) for p in (_CACHE, _CKPT0, _DESI_NPZ))


def _priors_only_sites(ctx):
    tr = handlers.trace(handlers.seed(lambda: C._legb_priors_only(ctx), jax.random.PRNGKey(0))).get_trace()
    return [n for n, s in tr.items() if s["type"] == "sample"]


def _model_sites(model, *a, seed=1):
    """Sample-site names (in trace order) of a numpyro model, tracing the FULL model (needs its args).
    Mirrors test_res_corr_alpha._sites so f_res twin-parity is checked against _legb_model, not only the
    priors-only twin (code-lens step-review #3)."""
    tr = handlers.trace(handlers.seed(model, jax.random.PRNGKey(seed))).get_trace(*a)
    return [k for k, v in tr.items() if v["type"] == "sample" and not v.get("is_observed")]


def test_bres_of_z_formula():
    """b_res(z) = f_res_amp * ((1+z)/(1+z_pivot))^f_res_slope (mirrors tau0_alpha_priya). amp=0 is the
    no-op (b_res=0 for ANY slope) -> _resolution_factor = 1 -> golden identity."""
    from hcd_analysis.emulator.closure_legb import _bres_of_z, F_RES_PIVOT_Z
    assert float(F_RES_PIVOT_Z) == 3.0
    z = np.array([2.2, 3.0, 4.2])
    amp, slope = 0.02, 0.5
    expect = amp * ((1.0 + z) / (1.0 + 3.0)) ** slope
    np.testing.assert_allclose(np.asarray(_bres_of_z(z, amp, slope)), expect, rtol=1e-12)
    # at the pivot z=3, b_res == amp regardless of slope
    assert np.isclose(float(_bres_of_z(3.0, amp, 1.7)), amp, rtol=1e-12)
    # amp=0 -> 0 for any slope (the golden no-op)
    assert np.allclose(np.asarray(_bres_of_z(z, 0.0, 3.0)), 0.0)


def test_fres_prior_constants_tight():
    """The f_res amplitude prior is TIGHT (physics: DESI deconvolved to ~few%), NOT cup1d's wide
    [-0.5,0.5] code default. 4-lens verdict C1."""
    from hcd_analysis.emulator.closure_legb import F_RES_AMP_SIGMA, F_RES_SLOPE_SIGMA
    assert 0.01 <= float(F_RES_AMP_SIGMA) <= 0.05, "f_res amp prior must be tight (~0.02), not wide"
    assert 0.0 < float(F_RES_SLOPE_SIGMA) <= 1.0


@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI not present")
def test_fres_sites_present_ordered_and_golden_off():
    """sample_res=True adds f_res_amp/f_res_slope to _legb_priors_only IN ORDER AFTER alpha_res (the
    constrain_fn mirror invariant _legb_model must match); sample_res=False (golden) adds NO f_res sites."""
    ctx, _d = C.build_legb_ctx(use_xclass=True, desi_kwargs=dict(z_lo=0.0, z_hi=2.6))
    on = _priors_only_sites(ctx._replace(sample_res=True))
    off = _priors_only_sites(ctx._replace(sample_res=False))
    assert "f_res_amp" in on and "f_res_slope" in on, "sample_res=True must sample both f_res sites"
    assert on.index("f_res_slope") == on.index("f_res_amp") + 1, "amp THEN slope (fixed order)"
    if "alpha_res" in on:
        assert on.index("f_res_amp") > on.index("alpha_res"), "f_res must come AFTER alpha_res"
    # GOLDEN-safe: default off -> the site set is unchanged (no f_res_*)
    assert "f_res_amp" not in off and "f_res_slope" not in off
    assert set(off) == set(on) - {"f_res_amp", "f_res_slope"}, "f_res is the ONLY site-set change"


@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI not present")
def test_cov_b_resolution_float_spd_and_golden():
    """option-b: load_desi_leg(resolution_float=True) removes the per-z-block rank-1 resolution mode
    (cov_b = cov - sum_z outer(e_res|z)) -> SPD + diagonal reduced by the resolution variance, and flips
    resolution_on True. Default (resolution_float=False) is byte-identical to option-a (golden)."""
    from hcd_analysis.emulator.data_likelihood import load_desi_leg
    leg_a = load_desi_leg(z_lo=0.0, z_hi=2.6)                          # option-a: resolution IN cov
    leg_b = load_desi_leg(z_lo=0.0, z_hi=2.6, resolution_float=True)   # option-b: cov_b + float f_res
    leg_off = load_desi_leg(z_lo=0.0, z_hi=2.6, resolution_float=False)
    assert leg_a.resolution_on is False and leg_b.resolution_on is True
    Ca, Cb = np.asarray(leg_a.C_data, float), np.asarray(leg_b.C_data, float)
    np.linalg.cholesky(Cb)                                            # cov_b is SPD (the correct removal)
    dA, dB = np.diag(Ca), np.diag(Cb)
    assert np.all(dB <= dA + 1e-9) and np.any(dB < dA - 1e-12), "resolution variance removed from diagonal"
    # golden: resolution_float=False is byte-identical to option-a
    np.testing.assert_array_equal(np.asarray(leg_off.C_data, float), Ca)


@pytest.mark.skipif(not os.path.exists(_EBOSS_NPZ), reason="eBOSS npz not present")
def test_cov_b_eboss_rescale_spd_and_golden():
    """eBOSS option-b uses the MULTIPLICATIVE sigma-rescale (cov=corr(x)sigma-sigma^T, resolution baked
    into sigma): sigma'^2 = sigma^2 - res^2 -> cov' rescaled. Reference-verified SPD; the per-z rank-1
    that works for DESI is NOT SPD here. Default (off) is byte-identical (golden)."""
    from hcd_analysis.emulator.data_likelihood import load_eboss_leg
    leg_a = load_eboss_leg()
    leg_b = load_eboss_leg(resolution_float=True)            # rescale mode
    leg_off = load_eboss_leg(resolution_float=False)
    assert leg_a.resolution_on is False and leg_b.resolution_on is True
    Ca, Cb = np.asarray(leg_a.C_data, float), np.asarray(leg_b.C_data, float)
    np.linalg.cholesky(Cb)                                   # sigma-rescale is SPD
    dA, dB = np.diag(Ca), np.diag(Cb)
    assert np.all(dB <= dA + 1e-9) and np.any(dB < dA - 1e-12), "resolution variance removed from diagonal"
    np.testing.assert_array_equal(np.asarray(leg_off.C_data, float), Ca)


_GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden")
_MF_GOLDEN = os.path.join(_GOLDEN_DIR, "legb_mf_golden.npz")


@pytest.mark.skipif(not (_have and os.path.exists(_MF_GOLDEN)),
                    reason="real cache/ckpt/DESI or MF golden not present")
def test_bres_vec_forward_response_per_z_indexed():
    """THE load-bearing change: predict_P_obs_on_leg threads a per-z sampled b_res_vec into the forward.
    The clone of alpha_res dropped alpha_res's forward-response test (code-lens step-review #2), so a
    future misindex of iz (wrong b_res_vec[iz] or R_z[iz]) would pass CI silently. This restores it.

    The resolution factor is applied at leg-k directly (P interpolated to leg-k FIRST, then multiplied),
    so P_obs(b_res_vec)/P_obs(0) == exp(2*b_res_vec[iz]*k^2*R_z[iz]^2) per row EXACTLY (no interp
    nonlinearity, unlike alpha_res). DISTINCT per-z b_res makes any iz misindex an O(1) failure."""
    from hcd_analysis.emulator import meanflux_prior as MF
    g = np.load(_MF_GOLDEN, allow_pickle=True)
    ctx, _d = C.build_legb_ctx(with_mf=True, mf_with_floor=True, sample_res=True)
    leg = [l for l in ctx.legs if l.name == "DESI"][0]
    assert leg.resolution_on is True, "sample_res=True must flip DESI resolution_on"
    theta9 = jnp.asarray(g["theta9"]); alpha3 = jnp.asarray(g["alpha3"])
    tau0_vec = MF.becker13_tau0(jnp.asarray(leg.z))
    szb = ctx.sigma_zb_per_leg.get(leg.name)
    rzb = ctx.rho_zb_per_leg.get(leg.name) if ctx.rho_zb_per_leg is not None else None
    core = jnp.mean(jnp.asarray(ctx.dla_core_leg[leg.name]), axis=0)

    def _fwd(**kw):
        P, _ = DL.predict_P_obs_on_leg(
            ctx.model, theta9, tau0_vec, alpha3, pf_stats=ctx.pf_stats, dla_core=core,
            cache_k=ctx.cache_k, leg=leg, sigma_zb=szb, alpha_centres=ctx.alpha_centres,
            cemu_inflate=ctx.cemu_inflate, rho_zb=rzb, mf=ctx.mf, **kw)
        return np.asarray(P)

    n_z = int(leg.n_z)
    P0 = _fwd(b_res_vec=jnp.zeros(n_z))
    # (a) all-zero b_res_vec == scalar b_res=0 == golden no-op (the override is a true no-op at 0).
    np.testing.assert_array_equal(P0, _fwd(b_res=0.0, b_res_vec=None))
    # (b) DISTINCT per-z b_res -> ratio == analytic exp(2*b[iz]*k^2*R_z[iz]^2) at the CORRECT z-index.
    rng = np.random.default_rng(0)
    b_vec = 0.01 + 0.03 * rng.random(n_z)
    P1 = _fwd(b_res_vec=jnp.asarray(b_vec))
    z_idx = np.asarray(leg.z_idx); k = np.asarray(leg.k); R_z = np.asarray(leg.R_z)
    expect = np.exp(2.0 * b_vec[z_idx] * k ** 2 * R_z[z_idx] ** 2)
    np.testing.assert_allclose(P1 / P0, expect, rtol=1e-9,
        err_msg="forward b_res_vec threading misindexes iz (b_res_vec[iz] or R_z[iz])")


@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI not present")
def test_fres_sites_in_both_models_same_order():
    """f_res_amp/f_res_slope appear in BOTH _legb_model AND _legb_priors_only at the SAME relative
    position (the constrain_fn parity invariant). The existing golden-off test traces only the
    priors-only twin; a reorder of the f_res block in _legb_model alone would corrupt constrain_fn with
    no test failure (code-lens step-review #3; mirrors test_res_corr_alpha's both-twins alpha test)."""
    ctx, d = C.build_legb_ctx(use_xclass=True, sample_res=True, desi_kwargs=dict(z_lo=0.0, z_hi=2.6))
    ctx = ctx._replace(legs=[l for l in ctx.legs if l.name == "DESI"])
    sims, _ = C.held_out_sims(d, fold=0)
    truth = C.make_truth_from_sim(d, sims[0], fold=0)
    mock_legs, _, _ = C.make_legb_mock(ctx, truth, jax.random.PRNGKey(0))
    core = C._mock_core_per_leg(ctx, truth)
    s_model = _model_sites(C._legb_model, ctx, mock_legs, core)
    s_prior = _model_sites(C._legb_priors_only, ctx)
    for nm in ("f_res_amp", "f_res_slope"):
        assert nm in s_model, f"{nm!r} missing from _legb_model: {s_model}"
        assert nm in s_prior, f"{nm!r} missing from _legb_priors_only: {s_prior}"
    assert s_model.index("f_res_amp") < s_model.index("f_res_slope")   # amp precedes slope in BOTH
    assert s_prior.index("f_res_amp") < s_prior.index("f_res_slope")
    assert s_model == s_prior, f"site-order mismatch model {s_model} vs priors {s_prior} -> constrain_fn"
    assert s_model.index("f_res_amp") == s_prior.index("f_res_amp")


@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI not present")
def test_cov_b_desi_deployed_cut_spd():
    """The SPD guarantee must hold on the DEPLOYED z-cut (z_lo=2.2, z_hi=4.2 -> 11 z-bins), NOT only the
    reduced 3-bin z<=2.6 slice the other SPD test uses (code-lens step-review #5). The high-z He-II bins
    carry the LARGEST e_res, so they are exactly where a mis-specified removal would break SPD."""
    from hcd_analysis.emulator.data_likelihood import load_desi_leg
    leg_b = load_desi_leg(z_lo=2.2, z_hi=4.2, resolution_float=True)     # the config the fit runs
    assert leg_b.n_z == 11, f"deployed DESI cut should be 11 z-bins, got {leg_b.n_z}"
    Cb = np.asarray(leg_b.C_data, float)
    np.linalg.cholesky(Cb)                                    # SPD on the deployed grid
    assert float(np.linalg.eigvalsh(Cb).min()) > 0.0


def test_resolution_sites_extra_truth_and_presence():
    """_resolution_sites_extra packs the option-b f_res_amp/f_res_slope draws + the injected-arm TRUTH
    into the per-mock sites_extra (they are sampled by _legb_model but NOT packed into _draws_matrix), so
    the rail/coverage check -- is f_res_amp railing the tight N(0,0.02) prior vs the data speaking? -- is
    possible from the shard pkls (code-lens step-review #4 / domain #4). A constant-b injection equals
    (amp=b*, slope=0) EXACTLY in the 2-param span, so the truth is (b*, 0.0); NaN on the clean arm; EMPTY
    when f_res is not sampled (additive-only => golden-safe)."""
    samples = {"f_res_amp": np.arange(20.0), "f_res_slope": np.arange(20.0) + 100.0, "ns": np.zeros(20)}
    # (a) injected resolution arm -> truth (b*, 0.0), draws thinned by step then capped at L
    se = C._resolution_sites_extra(samples, step=2, L=5,
                                   inject_spec={"resolution": {"b_res": 0.02}}, leg_a=True)
    assert set(se) == {"f_res_amp", "f_res_slope"}
    assert se["f_res_amp"]["truth"] == 0.02 and se["f_res_slope"]["truth"] == 0.0
    np.testing.assert_array_equal(se["f_res_amp"]["draws"], np.arange(20.0)[::2][:5])
    # (b) clean arm (no resolution inject) -> NaN truth, draws still stored
    se2 = C._resolution_sites_extra(samples, step=2, L=5, inject_spec=None, leg_a=True)
    assert np.isnan(se2["f_res_amp"]["truth"]) and np.isnan(se2["f_res_slope"]["truth"])
    # (c) held-out (leg_a=False) -> injection is not honoured there -> NaN truth
    se3 = C._resolution_sites_extra(samples, step=2, L=5,
                                    inject_spec={"resolution": {"b_res": 0.02}}, leg_a=False)
    assert np.isnan(se3["f_res_amp"]["truth"])
    # (d) f_res not sampled -> EMPTY (byte-identical golden when sample_res is off)
    se4 = C._resolution_sites_extra({"ns": np.zeros(20)}, step=2, L=5,
                                    inject_spec={"resolution": {"b_res": 0.02}}, leg_a=True)
    assert se4 == {}


def test_check_resolution_injectable_guards_unready_leg():
    """Defense-in-depth (4-referee panel, CS + domain #10): the res_b injection must RAISE, not silently
    skip, when a leg is not resolution_ready. KS reuses the DESI proxy R_z (~7-15x too large) so a b_res
    injection there is a ~70% distortion the forward cannot fit (the -21sigma ESS collapse). Gate on the
    R_z-valid flag, NOT resolution_on (option-a injects with resolution_on=False). res_b=None is a no-op."""
    from types import SimpleNamespace
    ready = SimpleNamespace(name="DESI", resolution_ready=True)
    unready = SimpleNamespace(name="KS", resolution_ready=False)
    C._check_resolution_injectable([ready], 0.02)                 # all-ready + res_b -> ok
    C._check_resolution_injectable([ready, unready], None)        # res_b None -> no-op even with KS
    with pytest.raises((ValueError, RuntimeError), match="[Kk][Ss]|resolution_ready"):
        C._check_resolution_injectable([ready, unready], 0.02)    # stray KS + res_b -> RAISE


@pytest.mark.skipif(not os.path.exists(_EBOSS_NPZ), reason="eBOSS npz not present")
def test_cov_arm_d_eboss_coherent_mode():
    """ARM-D (first-class bracket arm, esp. eBOSS): remove resolution the deployed way (sigma-rescale) then
    RE-ADD it as ONE coherent cross-z mode s^2*outer(e,e) -- NO forward float. Invariants: SPD; PRESERVES the
    total variance (diag == option-a, unlike option-b which reduces it); arm-D == option-b cov_b + a single
    rank-1 PSD coherent mode; FILLS IN the cross-z correlations option-a (block-diagonal) lacks; resolution_on
    stays False (no f_res in the forward), resolution_coherent_on True. Default off = byte-identical golden."""
    from hcd_analysis.emulator.data_likelihood import load_eboss_leg
    leg_a = load_eboss_leg()                                   # option-a: resolution IN cov (block-diagonal)
    leg_b = load_eboss_leg(resolution_float=True)              # option-b: cov_b (sigma-rescale)
    leg_d = load_eboss_leg(resolution_coherent=True)           # arm-D: cov_b + coherent mode
    leg_off = load_eboss_leg(resolution_coherent=False)
    assert leg_a.resolution_on is False and leg_a.resolution_coherent_on is False
    assert leg_d.resolution_on is False, "arm-D must NOT float f_res in the forward"
    assert leg_d.resolution_coherent_on is True
    Ca, Cb, Cd = (np.asarray(x.C_data, float) for x in (leg_a, leg_b, leg_d))
    np.linalg.cholesky(Cd)                                     # SPD
    np.testing.assert_allclose(np.diag(Cd), np.diag(Ca), rtol=1e-9)   # variance PRESERVED (re-correlated)
    # arm-D = cov_b + exactly one rank-1 PSD coherent mode
    D = Cd - Cb
    ev = np.linalg.eigvalsh(D)
    assert ev.min() > -1e-9 * abs(ev).max(), "coherent add must be PSD"
    assert int(np.sum(ev > 1e-9 * ev.max())) == 1, "coherent add must be rank-1 (one common Delta R/R)"
    zr = np.asarray(leg_d.z_row); crossz = zr[:, None] != zr[None, :]
    assert np.abs(Ca[crossz]).max() == 0.0, "eBOSS option-a is block-diagonal in z"
    assert np.abs(Cd[crossz]).max() > 0.0, "arm-D fills in cross-z coherent correlations"
    np.testing.assert_array_equal(np.asarray(leg_off.C_data, float), Ca)   # golden


@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI not present")
def test_cov_arm_d_desi_coherent_and_amplitude():
    """Arm-D on DESI (rank-1 removal + coherent add) + the amplitude knob: cov_D(s) - cov_b scales as s^2."""
    from hcd_analysis.emulator.data_likelihood import load_desi_leg
    kw = dict(z_lo=2.2, z_hi=4.2)
    leg_a = load_desi_leg(**kw)
    leg_b = load_desi_leg(resolution_float=True, **kw)
    leg_d = load_desi_leg(resolution_coherent=True, **kw)
    leg_d2 = load_desi_leg(resolution_coherent=True, resolution_coh_amp=2.0, **kw)
    assert leg_d.resolution_on is False and leg_d.resolution_coherent_on is True
    Ca, Cb, Cd, Cd2 = (np.asarray(x.C_data, float) for x in (leg_a, leg_b, leg_d, leg_d2))
    np.linalg.cholesky(Cd); np.linalg.cholesky(Cd2)           # SPD at both amplitudes
    np.testing.assert_allclose(np.diag(Cd), np.diag(Ca), rtol=1e-9)   # variance preserved
    # coherent mode is rank-1 PSD and scales as amp^2: (Cd2 - Cb) == 4*(Cd - Cb)
    np.testing.assert_allclose(Cd2 - Cb, 4.0 * (Cd - Cb), rtol=1e-9, atol=1e-12)


def test_check_single_instrument_for_res_guards_multileg():
    """sample_res floats ONE global f_res site, but resolution is a per-INSTRUMENT systematic (DESI and
    eBOSS spectrographs are independent, unlike the globally-shared metals). Until per-instrument f_res
    sites are built (4-referee panel #8), a MULTI-instrument ctx + sample_res must RAISE -- the per-leg
    cert is single-instrument. No-op when sample_res is off (golden-safe)."""
    from types import SimpleNamespace
    desi = SimpleNamespace(name="DESI"); eboss = SimpleNamespace(name="eBOSS")
    C._check_single_instrument_for_res([desi], True)              # single instrument -> ok
    C._check_single_instrument_for_res([desi, eboss], False)      # sample_res off -> no-op
    with pytest.raises((ValueError, RuntimeError), match="instrument|per-instrument|f_res"):
        C._check_single_instrument_for_res([desi, eboss], True)   # multi-instrument + sample_res -> RAISE


@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI not present")
def test_resolution_ready_flag_per_leg():
    """The per-leg resolution_ready flag the injection guard reads (panel domain #10): DESI/eBOSS R_z is
    trustworthy (True); KS reuses the DESI proxy (8-15x too large) -> False. Default DataLeg is True."""
    from hcd_analysis.emulator.data_likelihood import load_desi_leg, load_eboss_leg, load_ks_leg
    assert load_desi_leg(z_lo=2.2, z_hi=4.2).resolution_ready is True
    if os.path.exists(_EBOSS_NPZ):
        assert load_eboss_leg().resolution_ready is True
    _KS = ("/home/mfho/lya_emulator_full/lyaemu/data/kodiaq_squad/"
           "final-conservative-p1d-karacayli_etal2021.txt")
    if os.path.exists(_KS):
        assert load_ks_leg().resolution_ready is False, "KS proxy R_z is untrustworthy -> not resolution_ready"
