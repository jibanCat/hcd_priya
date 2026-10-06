"""Gate E task 4d (GATE_E_SPEC v1 section 2): the production context and log-likelihood on the new forward.
LegBCtx holds the box modes (no velocity grid); the log-likelihood core is predict_leg + the Gaussian on the kept
rows, and refuses (-inf, finite gradients) any kept bin outside the simulated modes; the build checks the whole
sampling box; the single-grid MF rebuild and the pre-2026-10 mock builders are gone from the production module."""
import hcd_analysis.emulator  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hcd_analysis.emulator import closure_legb as CL
from hcd_analysis.emulator import forward as FW
from hcd_analysis.emulator.likelihood import gaussian_loglik
from tests.test_forward_v2 import KCOM, _emu, _leg, _t1


def _ctx(model, pf, legs, z_global, rho_per_leg=None, ac=None):
    req = {f: None for f in CL.LegBCtx._fields if f not in CL.LegBCtx._field_defaults}
    req.update(model=model, pf_stats=pf, legs=legs, k_com_hmpc=jnp.asarray(KCOM), z_global=np.asarray(z_global),
               rho_zb_per_leg=rho_per_leg, alpha_centres=ac, cemu_inflate=1.0,
               alpha_hcd_mu=jnp.array([0.17, 0.062, 0.0044]), alpha_hcd_sigma=jnp.array([0.05, 0.025, 0.002]))
    return CL.LegBCtx(**req)


def test_ctx_holds_the_box_modes_not_a_velocity_grid():
    f = CL.LegBCtx._fields
    assert "k_com_hmpc" in f and "cache_k" not in f
    for gone in ("sigma_zb_per_leg", "mf_shape_per_leg", "mf_shape_infl", "mf_emucoh_infl", "mf_emucoh_offdiag_only"):
        assert gone not in f
    assert "t2_per_leg" in f and "t3_per_leg" in f


def test_loglik_core_is_predict_leg_plus_the_gaussian_on_kept_rows():
    model, pf, core = _emu(20)
    a = _leg([2.4, 3.2], name="DESI"); b = _leg([3.2, 4.0], name="KS", dff=0.0)
    Pd = np.asarray(a.P_data).copy(); Pd[3] = np.nan
    a = a._replace(P_data=Pd)
    zg = np.array([2.4, 3.2, 4.0])
    rho_a, ac = _t1(2, seed=1); rho_b, _ = _t1(2, seed=2)
    ctx = _ctx(model, pf, [a, b], zg, {"DESI": rho_a, "KS": rho_b}, ac)
    th = jnp.asarray(np.full(9, 0.45)); tau0 = jnp.asarray([0.25, 0.4, 0.6])
    alpha = jnp.asarray([[0.05, 0.02, 0.004], [0.06, 0.025, 0.005], [0.08, 0.03, 0.006]])
    cores = {"DESI": core, "KS": core}
    got = CL._data_loglik_legcore(ctx, th, tau0, alpha, [a, b], cores)
    expect = 0.0
    for leg, rho, sel in ((a, rho_a, [0, 1]), (b, rho_b, [1, 2])):
        out = FW.predict_leg(model, th, tau0[jnp.asarray(sel)], alpha[jnp.asarray(sel)], leg=leg, k_com=KCOM,
                             pf_stats=pf, dla_core=core, t1=(rho, ac))
        keep = np.where(np.isfinite(np.asarray(leg.P_data)))[0]
        expect += gaussian_loglik(jnp.asarray(np.asarray(leg.P_data)[keep]) - out.P_model[keep],
                                  out.C_total[jnp.ix_(keep, keep)])
    np.testing.assert_allclose(float(got), float(expect), rtol=1e-12)


def test_kept_bins_outside_the_modes_give_minus_inf_with_finite_gradients():
    model, pf, core = _emu(21)
    leg = _leg([3.0], k_lo=1.2e-3, k_hi=0.12, n_per_z=10)
    ctx = _ctx(model, pf, [leg], [3.0])
    th = jnp.asarray(np.full(9, 0.5))
    args = (jnp.asarray([0.3]), jnp.asarray([[0.05, 0.02, 0.004]]), [leg], {"DESI": core})
    ll, aux = CL._data_loglik_legcore(ctx, th, *args, return_aux=True)
    assert float(ll) == -np.inf and int(aux["n_out"]) > 0
    g = jax.grad(lambda t: CL._data_loglik_legcore(ctx, t, *args))(th)
    assert np.all(np.isfinite(np.asarray(g)))


def test_out_of_range_bins_that_are_not_kept_are_not_refused():
    model, pf, core = _emu(22)
    leg = _leg([3.0], k_lo=1.2e-3, k_hi=0.12, n_per_z=10)
    kg_top = 2 * np.pi * 172 / 120.0 * 4.0 / (100 * 4.5)     # below any k_skm,172 at z = 3 in the box
    Pd = np.where(np.asarray(leg.k) > kg_top, np.nan, np.asarray(leg.P_data))
    leg = leg._replace(P_data=Pd)
    ctx = _ctx(model, pf, [leg], [3.0])
    ll, aux = CL._data_loglik_legcore(ctx, jnp.asarray(np.full(9, 0.5)), jnp.asarray([0.3]),
                                      jnp.asarray([[0.05, 0.02, 0.004]]), [leg], {"DESI": core}, return_aux=True)
    assert np.isfinite(float(ll)) and int(aux["n_out"]) == 0


def test_build_time_box_check():
    ok = _leg([4.6], k_lo=1.2e-3, k_hi=0.06)
    CL._assert_bins_inside_modes([ok], KCOM, np.zeros(9), np.ones(9))
    bad = _leg([4.6], k_lo=1.2e-3, k_hi=0.07)          # k_skm,172 at z = 4.6 reaches down to ~0.0644 in the box
    with pytest.raises(ValueError, match="outside the simulated modes"):
        CL._assert_bins_inside_modes([bad], KCOM, np.zeros(9), np.ones(9))
    Pd = np.where(np.asarray(bad.k) > 0.06, np.nan, np.asarray(bad.P_data))
    CL._assert_bins_inside_modes([bad._replace(P_data=Pd)], KCOM, np.zeros(9), np.ones(9))   # only kept bins


def test_fiducial_dla_core_needs_no_velocity_grid():
    d = {"z_grid": np.array([3.0, 3.0, 3.2]), "delta": np.random.default_rng(0).uniform(0, 1, (3, 3, 172))}
    out = CL._fiducial_dla_core_per_leg(d, [_leg([3.0])])
    np.testing.assert_allclose(np.asarray(out["DESI"])[0], d["delta"][:2, 2].mean(0))


def test_retired_paths_are_gone_and_mock_builders_wait_for_gate_f():
    assert not hasattr(CL, "build_mf_correction")
    for name in ("make_legb_mock", "make_leg_a_legmock", "make_hr_truth_from_cache", "make_truth_from_sim"):
        with pytest.raises(NotImplementedError, match="gate F"):
            getattr(CL, name)()


def test_chain_rescore_refuses_draws_with_bins_outside_the_modes():
    import importlib
    import sys
    from hcd_analysis.paths import REPO_ROOT
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    RF = importlib.import_module("run_real_fit")
    model, pf, core = _emu(24)
    samples = {"theta_unit": np.full((2, 9), 0.5), "tau0_vec": np.full((2, 1), 0.3),
               "alpha_hcd_z": np.tile(np.array([0.05, 0.02, 0.004]), (2, 1, 1))}
    ok = _leg([3.0], k_lo=1.2e-3, k_hi=0.05)
    ctx = _ctx(model, pf, [ok], [3.0])._replace(fix_alpha_res=True, metal_prior="uniform")
    ll = RF._loglik_chain(ctx, {"DESI": core}, samples, None)
    assert np.asarray(ll).shape == (2,) and np.all(np.isfinite(np.asarray(ll)))
    bad = _leg([3.0], k_lo=1.2e-3, k_hi=0.12)
    ctx_bad = _ctx(model, pf, [bad], [3.0])._replace(fix_alpha_res=True, metal_prior="uniform")
    with pytest.raises(ValueError, match="outside the simulated modes"):
        RF._loglik_chain(ctx_bad, {"DESI": core}, samples, None)


def test_forward_signature_and_stamp_carry_the_coordinate_and_the_product_digests():
    import hashlib
    import json
    from hcd_analysis.emulator import data_likelihood as DL
    old = hashlib.sha256(json.dumps({"PROD_FORWARD_BY_LEG": CL.PROD_FORWARD_BY_LEG, "PROD_RES_CORR_ON": False,
                                     "DESI_DLA_COV_REDUCE": bool(DL.DESI_DLA_COV_REDUCE)},
                                    sort_keys=True).encode()).hexdigest()
    assert CL.forward_signature() != old                      # the pre-2026-10 forward's signature cannot recur
    model, pf, core = _emu(23)
    leg = _leg([3.0])
    digests = {"lf_ensemble_eqx_sha256": ["aa" * 32], "mf_product_sha256": "bb" * 32, "cache_sha256": "cc" * 32}
    ctx = _ctx(model, pf, [leg], [3.0])._replace(product_digests=digests, res_corr_on=False, fix_alpha_res=True,
                                                   sample_res=False, metal_prior="uniform", metal_node_z=(2.2, 4.2))
    st = CL.forward_stamp(ctx, leg)
    assert st["products"] == digests and st["coordinate"].startswith("k_skm")
    assert st["hcd_prior_centres"] == [0.17, 0.062, 0.0044]
