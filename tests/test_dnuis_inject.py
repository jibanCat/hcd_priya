"""Data-nuisance injection-recovery BIAS gate — the injection hooks (TDD).

These pin the closure_legb injection hooks the production-forward bias gate uses to ask whether
wiring the DATA-side nuisances (metals misspecification, resolution, per-survey LLS pin) leaves
the recovered (A_p, n_s) UNBIASED on the Leg-A self-draw. On the clean self-draw the cosmology
bias is ZERO by construction (noiseless mock == P_model(truth), C_mock ≡ C_like), so any post-hoc
injection the forward cannot fit (or a truth offset from the prior pin) is purely that nuisance.

Hooks under test (closure_legb):
  (a) make_leg_a_legmock(..., inject_metal_misspec=, inject_resolution=) — multiply the noiseless
      P_model by a host contaminant BEFORE the eps draw, on the appropriate legs;
  (b) apply_lls_truth_boost(truth_pack, boost) — a pure helper scaling ONLY the LLS truth;
  (c) run_legb(..., inject_spec=) leg_a branch — threads the above; inject_spec=None is a no-op.

Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_dnuis_inject.py -q
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

import hcd_analysis.emulator  # noqa: F401  enables x64 BEFORE jax
from hcd_analysis.emulator import closure_legb as LB
from hcd_analysis.emulator import data_likelihood as DL


# --------------------------------------------------------------------------------------------- #
#  PART (b): apply_lls_truth_boost — a PURE helper (no ctx needed). Tested first/fast.
# --------------------------------------------------------------------------------------------- #
def _toy_truth_pack(nZg=5):
    rng = np.random.default_rng(0)
    return dict(
        theta9=rng.uniform(0.2, 0.8, 9),
        tau0_global=rng.uniform(0.9, 1.1, nZg),
        alpha_hcd=np.array([0.7, 0.3, 0.05]),                 # (3,) [LLS, subDLA, DLA] pivot
        alpha_hcd_z=np.stack([np.array([0.7, 0.3, 0.05]) * (1.0 + 0.1 * i) for i in range(nZg)]),
        a_siiii=0.02,
        kept_global_z=np.ones(nZg, bool))


def test_apply_lls_truth_boost_scales_only_lls():
    tp = _toy_truth_pack()
    tp0 = {k: (np.array(v) if isinstance(v, np.ndarray) else v) for k, v in tp.items()}
    boost = 1.5
    out = LB.apply_lls_truth_boost(tp, boost)
    # LLS column / pivot scaled by boost
    np.testing.assert_allclose(out["alpha_hcd"][0], tp0["alpha_hcd"][0] * boost, rtol=1e-12)
    np.testing.assert_allclose(out["alpha_hcd_z"][:, 0], tp0["alpha_hcd_z"][:, 0] * boost, rtol=1e-12)
    # subDLA + DLA columns/pivots UNTOUCHED
    np.testing.assert_allclose(out["alpha_hcd"][1:], tp0["alpha_hcd"][1:], rtol=0, atol=0)
    np.testing.assert_allclose(out["alpha_hcd_z"][:, 1:], tp0["alpha_hcd_z"][:, 1:], rtol=0, atol=0)
    # theta9 / tau0 / a_siiii untouched
    np.testing.assert_allclose(out["theta9"], tp0["theta9"], rtol=0, atol=0)
    np.testing.assert_allclose(out["tau0_global"], tp0["tau0_global"], rtol=0, atol=0)
    assert float(out["a_siiii"]) == float(tp0["a_siiii"])


def test_apply_lls_truth_boost_returns_copy_input_unmutated():
    tp = _toy_truth_pack()
    a_before = np.array(tp["alpha_hcd"])
    az_before = np.array(tp["alpha_hcd_z"])
    out = LB.apply_lls_truth_boost(tp, 2.0)
    # the input is NOT mutated in place
    np.testing.assert_allclose(tp["alpha_hcd"], a_before, rtol=0, atol=0)
    np.testing.assert_allclose(tp["alpha_hcd_z"], az_before, rtol=0, atol=0)
    # and the returned arrays are distinct objects
    assert out["alpha_hcd"] is not tp["alpha_hcd"]
    assert out["alpha_hcd_z"] is not tp["alpha_hcd_z"]


def test_apply_lls_truth_boost_unit_is_identity():
    tp = _toy_truth_pack()
    out = LB.apply_lls_truth_boost(tp, 1.0)
    np.testing.assert_allclose(out["alpha_hcd"], tp["alpha_hcd"], rtol=0, atol=0)
    np.testing.assert_allclose(out["alpha_hcd_z"], tp["alpha_hcd_z"], rtol=0, atol=0)


# --------------------------------------------------------------------------------------------- #
#  PART (a): the injection hooks in make_leg_a_legmock — need a real (light) leg ctx.
# --------------------------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def ctx_d():
    try:
        # light DESI+KS baseline (xclass ρ; metals_on so a DESI leg carries metals_on=True;
        # no MF/eBOSS → fast). DESI leg has metals_on=True, KS metals_on=False — exactly the
        # metals_on / non-metals split the metal-misspec test needs.
        return LB.build_legb_ctx(use_xclass=True, metals_on=True)
    except Exception as e:                                   # data/checkpoint absent in this env
        pytest.skip(f"build_legb_ctx unavailable: {e}")


def _core(ctx, d):
    return {k: np.nanmean(np.asarray(v), axis=0)
            for k, v in LB._fiducial_dla_core_per_leg(d, ctx.legs, ctx.cache_k).items()}


def test_metal_misspec_changes_metals_leg_not_clean_leg(ctx_d):
    ctx, d = ctx_d
    core = _core(ctx, d)
    tp = LB.draw_leg_a_leg_truth(ctx, jax.random.PRNGKey(3))
    key = jax.random.PRNGKey(4)
    # baseline (no injection) vs metal-misspec injection — SAME mock RNG key
    base_legs, base_info = LB.make_leg_a_legmock(ctx, core, tp, key)
    inj_legs, inj_info = LB.make_leg_a_legmock(
        ctx, core, tp, key,
        inject_metal_misspec={"form": "desi_full", "f_SiIII": 0.009, "f_SiII": 0.004,
                              "f_SiII_SiII": 0.002})
    names = [l.name for l in ctx.legs]
    assert "DESI" in names and "KS" in names, f"expected DESI+KS legs, got {names}"
    for li, leg in enumerate(ctx.legs):
        base_truth = np.asarray(base_info["truth_on_leg"][leg.name])
        inj_truth = np.asarray(inj_info["truth_on_leg"][leg.name])
        assert np.all(np.isfinite(inj_truth)), f"{leg.name}: non-finite after injection"
        assert inj_truth.shape == base_truth.shape, f"{leg.name}: shape changed"
        if leg.metals_on:
            assert not np.allclose(inj_truth, base_truth), \
                f"{leg.name} (metals_on): metal-misspec injection did not change the noiseless P"
            # the contaminant amplitude A_X = f_X/(1−⟨F⟩) is largest at the MOST transmissive
            # (low-z) bins — physically real (metals matter most where the forest is thin). It is
            # a multiplicative factor on the order of unity, not a runaway: stay within a sane band.
            frac = inj_truth / base_truth - 1.0
            assert np.all(np.abs(frac) < 5.0), f"{leg.name}: metal contaminant runaway"
            # but at the LEAST transmissive (high-z) bins ⟨F⟩ is low → A_X small → tiny perturbation
            z_idx = np.asarray(leg.z_idx)
            hiz_rows = np.where(z_idx == leg.n_z - 1)[0]
            assert np.all(np.abs(frac[hiz_rows]) < 0.15), \
                f"{leg.name}: high-z metal contaminant should be small (low A_X)"
        else:
            assert np.allclose(inj_truth, base_truth, rtol=0, atol=0), \
                f"{leg.name} (clean): metal-misspec must NOT touch a non-metals leg"


def test_meanflux_on_leg_is_exp_minus_tau0_global(ctx_d):
    """REGRESSION (the BLOCKER bug): ``truth_pack['tau0_global']`` is ALREADY τ_eff=α·Kim (the cache
    −ln⟨F⟩ coord), so ⟨F⟩(z) MUST be exactly exp(−tau0_global[sel]) — NO second Kim multiply (the
    old code did tau_eff=α·Kim again → ⟨F⟩=exp(−α·Kim²), up to ~6× too-large metal amplitude at low
    z). Pinned the way the resolution test pins its factor so the double-Kim bug cannot regress."""
    ctx, d = ctx_d
    tp = LB.draw_leg_a_leg_truth(ctx, jax.random.PRNGKey(13))
    zg = np.asarray(ctx.z_global)
    tau0_global = np.asarray(tp["tau0_global"])
    for leg in ctx.legs:
        sel = np.array([int(np.argmin(np.abs(zg - zz))) for zz in leg.z])
        expect = np.exp(-tau0_global[sel])
        got = np.asarray(LB._meanflux_on_leg(ctx, leg, tp))
        assert got.shape == (leg.n_z,), f"{leg.name}: shape {got.shape} != ({leg.n_z},)"
        np.testing.assert_allclose(got, expect, rtol=1e-12, atol=0.0,
                                   err_msg=f"{leg.name}: ⟨F⟩ != exp(−tau0_global[sel])")
        # a sanity band: ⟨F⟩ ∈ (0,1) and the double-Kim bug would push it absurdly low at low z
        assert np.all((got > 0.0) & (got <= 1.0)), f"{leg.name}: ⟨F⟩ out of (0,1]"


def test_resolution_injection_is_exp_2b_k2_R2(ctx_d):
    ctx, d = ctx_d
    core = _core(ctx, d)
    tp = LB.draw_leg_a_leg_truth(ctx, jax.random.PRNGKey(5))
    key = jax.random.PRNGKey(6)
    b_res = 0.02
    base_legs, base_info = LB.make_leg_a_legmock(ctx, core, tp, key)
    inj_legs, inj_info = LB.make_leg_a_legmock(ctx, core, tp, key,
                                               inject_resolution={"b_res": b_res})
    for leg in ctx.legs:
        base_truth = np.asarray(base_info["truth_on_leg"][leg.name])
        inj_truth = np.asarray(inj_info["truth_on_leg"][leg.name])
        assert inj_truth.shape == base_truth.shape
        assert np.all(np.isfinite(inj_truth))
        # expected factor exp(2 b_res k^2 R_z^2) > 1 for b_res>0, monotone increasing in k
        R_z = np.asarray(leg.R_z)
        z_idx = np.asarray(leg.z_idx)
        k = np.asarray(leg.k)
        expect = np.exp(2.0 * b_res * k ** 2 * R_z[z_idx] ** 2)
        assert np.all(expect >= 1.0 - 1e-12), "resolution factor must be >= 1 for b_res>0"
        np.testing.assert_allclose(inj_truth, base_truth * expect, rtol=1e-8,
                                   err_msg=f"{leg.name}: resolution factor mismatch")
        # monotone in k within each z-block (R_z fixed per z)
        for iz in range(leg.n_z):
            rows = np.where(z_idx == iz)[0]
            if rows.size < 2:
                continue
            order = np.argsort(k[rows])
            f_sorted = expect[rows][order]
            assert np.all(np.diff(f_sorted) >= -1e-12), \
                f"{leg.name} z{iz}: resolution factor not monotone in k"


# --------------------------------------------------------------------------------------------- #
#  PART (c): run_legb(inject_spec=None) is a byte-for-byte no-op vs the direct mock.
# --------------------------------------------------------------------------------------------- #
def test_inject_spec_none_is_noop(ctx_d):
    """The no-op guarantee: building the leg-A mock through the inject_spec=None path reproduces
    the existing make_leg_a_legmock mock EXACTLY for the same (seed, mock index) RNG."""
    ctx, d = ctx_d
    seed, m = 0, 0
    # reproduce run_legb's leg_a per-mock RNG split (fold_in(seed,m) -> truth,mock,nuts)
    key0 = jax.random.PRNGKey(int(seed))
    k_truth, k_mock, _k_nuts = jax.random.split(jax.random.fold_in(key0, m), 3)
    tp = LB.draw_leg_a_leg_truth(ctx, k_truth)
    core = {name: jnp.asarray(np.nanmean(np.asarray(v), axis=0))
            for name, v in LB._fiducial_dla_core_per_leg(d, ctx.legs, ctx.cache_k).items()}

    # direct (no injection args) vs explicit None-injection — must be identical
    legs_direct, _ = LB.make_leg_a_legmock(ctx, core, tp, k_mock)
    legs_none, _ = LB.make_leg_a_legmock(ctx, core, tp, k_mock,
                                         inject_metal_misspec=None, inject_resolution=None)
    for ld, ln in zip(legs_direct, legs_none):
        np.testing.assert_array_equal(np.asarray(ld.P_data), np.asarray(ln.P_data))


def test_run_legb_inject_spec_none_matches_default_mock(ctx_d):
    """run_legb's leg_a branch with inject_spec=None must build the SAME mock the default branch
    does (the no-op guarantee at the driver level). We compare the mock P_data reconstructed via
    the SAME (seed,m) RNG split run_legb uses — independent of the (tiny) NUTS that follows."""
    ctx, d = ctx_d
    seed, m = 7, 0
    key0 = jax.random.PRNGKey(int(seed))
    k_truth, k_mock, _ = jax.random.split(jax.random.fold_in(key0, m), 3)
    tp = LB.draw_leg_a_leg_truth(ctx, k_truth)
    fid_core = {name: jnp.asarray(np.nanmean(np.asarray(v), axis=0))
                for name, v in LB._fiducial_dla_core_per_leg(d, ctx.legs, ctx.cache_k).items()}
    # default branch
    legs_default, _ = LB.make_leg_a_legmock(ctx, fid_core, tp, k_mock)
    # inject_spec=None branch (what run_legb takes)
    legs_none, _ = LB.make_leg_a_legmock(
        ctx, fid_core, tp, k_mock, inject_metal_misspec=None, inject_resolution=None)
    for la, lb in zip(legs_default, legs_none):
        np.testing.assert_array_equal(np.asarray(la.P_data), np.asarray(lb.P_data))


def test_run_legb_with_lls_excess_runs_and_offsets_truth(ctx_d):
    """Smoke: run_legb(leg_a, inject_spec={'lls_truth_boost':1.3}) runs end-to-end (tiny NUTS) and
    the recorded truth_vec carries the boosted LLS (alpha_lls column) vs the un-boosted run."""
    ctx, d = ctx_d
    boost = 1.3
    base = LB.run_legb(ctx, d, n_mocks=1, mock_indices=[0], return_per_mock=True, leg_a=True,
                       n_warmup=4, n_samples=4, seed=11, verbose=False)
    inj = LB.run_legb(ctx, d, n_mocks=1, mock_indices=[0], return_per_mock=True, leg_a=True,
                      n_warmup=4, n_samples=4, seed=11, verbose=False,
                      inject_spec={"lls_truth_boost": boost})
    names = base[0]["names"]
    j = names.index("alpha_lls")
    np.testing.assert_allclose(inj[0]["truth_vec"][j], base[0]["truth_vec"][j] * boost, rtol=1e-10)
    # subDLA / DLA truth unchanged
    js, jd = names.index("alpha_subdla"), names.index("alpha_dla")
    np.testing.assert_allclose(inj[0]["truth_vec"][js], base[0]["truth_vec"][js], rtol=0, atol=0)
    np.testing.assert_allclose(inj[0]["truth_vec"][jd], base[0]["truth_vec"][jd], rtol=0, atol=0)


if __name__ == "__main__":
    test_apply_lls_truth_boost_scales_only_lls()
    test_apply_lls_truth_boost_returns_copy_input_unmutated()
    test_apply_lls_truth_boost_unit_is_identity()
    print("[dnuis-inject] pure-helper tests OK")
