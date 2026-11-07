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
from tests.test_forward_v2 import KCOM, _core_data, _emu, _leg, _t1


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
    cores = {"DESI": _core_data(a, th, core), "KS": _core_data(b, th, core)}
    got = CL._data_loglik_legcore(ctx, th, tau0, alpha, [a, b], cores)
    expect = 0.0
    for leg, rho, sel in ((a, rho_a, [0, 1]), (b, rho_b, [1, 2])):
        out = FW.predict_leg(model, th, tau0[jnp.asarray(sel)], alpha[jnp.asarray(sel)], leg=leg, k_com=KCOM,
                             pf_stats=pf, dla_core=cores[leg.name], t1=(rho, ac))
        keep = np.where(np.isfinite(np.asarray(leg.P_data)))[0]
        expect += gaussian_loglik(jnp.asarray(np.asarray(leg.P_data)[keep]) - out.P_model[keep],
                                  out.C_total[jnp.ix_(keep, keep)])
    np.testing.assert_allclose(float(got), float(expect), rtol=1e-12)


def test_kept_bins_outside_the_modes_give_minus_inf_with_finite_gradients():
    model, pf, core = _emu(21)
    leg = _leg([3.0], k_lo=1.2e-3, k_hi=0.12, n_per_z=10)
    ctx = _ctx(model, pf, [leg], [3.0])
    th = jnp.asarray(np.full(9, 0.5))
    args = (jnp.asarray([0.3]), jnp.asarray([[0.05, 0.02, 0.004]]), [leg], {"DESI": jnp.zeros(leg.k.size)})
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
                                      jnp.asarray([[0.05, 0.02, 0.004]]), [leg], {"DESI": jnp.zeros(leg.k.size)},
                                      return_aux=True)
    assert np.isfinite(float(ll)) and int(aux["n_out"]) == 0


def test_build_time_box_check():
    ok = _leg([4.6], k_lo=1.2e-3, k_hi=0.06)
    CL._assert_bins_inside_modes([ok], KCOM, np.zeros(9), np.ones(9))
    bad = _leg([4.6], k_lo=1.2e-3, k_hi=0.07)          # k_skm,172 at z = 4.6 reaches down to ~0.0644 in the box
    with pytest.raises(ValueError, match="outside the simulated modes"):
        CL._assert_bins_inside_modes([bad], KCOM, np.zeros(9), np.ones(9))
    Pd = np.where(np.asarray(bad.k) > 0.06, np.nan, np.asarray(bad.P_data))
    CL._assert_bins_inside_modes([bad._replace(P_data=Pd)], KCOM, np.zeros(9), np.ones(9))   # only kept bins


def _cache_rows(zs=(3.0, 3.0, 3.2), seed=0):
    rng = np.random.default_rng(seed)
    kf = np.stack([np.linspace(4.0e-4, 0.095, 172) * s for s in (1.0, 1.05, 0.97)])
    return {"z_grid": np.asarray(zs), "delta": rng.uniform(0, 1, (3, 3, 172)), "kfkms": kf}


def test_dla_core_product_is_required_when_a_leg_forwards_the_dla_class():
    with pytest.raises(ValueError, match="DLA-core product"):
        CL._dla_core_data_per_leg([_leg([3.0])], None, "cache")                    # DESI-like: dla_forward_frac 1
    out, digest = CL._dla_core_data_per_leg([_leg([3.0], name="KS", dff=0.0)], None, "cache")
    assert np.all(np.asarray(out["KS"]) == 0.0) and out["KS"].shape == (9,) and digest.startswith("not used")


def test_dla_core_product_is_provenance_checked_and_matches_the_legs_z(tmp_path):
    from hcd_analysis.emulator import cemu_build as CB
    from hcd_analysis.emulator import products as PRD
    from hcd_analysis.emulator.products import save_product
    d = _cache_rows()
    leg = _leg([3.0])
    c = CB.dla_core_leg(d["kfkms"], d["delta"][:, 2], d["z_grid"], [3.0], CB.DLA_CORE_GRID)
    prov = dict(code_commit="x", cache_sha256="cache", inputs={}, row_rule="A1 s4", leg_z={"DESI": [3.0]})
    path = str(tmp_path / "core.npz")
    save_product(path, "dla_core", k_com_hmpc=KCOM, provenance=prov, k_grid=CB.DLA_CORE_GRID, core_DESI=c)
    out, digest = CL._dla_core_data_per_leg([leg], path, "cache")
    np.testing.assert_allclose(np.asarray(out["DESI"]), PRD.dla_core_at(leg.k, CB.DLA_CORE_GRID, c), rtol=1e-13)
    assert len(digest) == 64
    with pytest.raises(ValueError, match="z"):
        CL._dla_core_data_per_leg([_leg([3.0, 3.2])], path, "cache")      # the product's z-mean is another leg's
    with pytest.raises(Exception):
        CL._dla_core_data_per_leg([leg], path, "another-cache")


def test_t1_product_maps_leg_z_to_its_cells(tmp_path):
    from hcd_analysis.emulator.products import save_product
    rng = np.random.default_rng(3)
    zc = np.round(np.arange(2.2, 4.61, 0.2), 1)
    rho = rng.uniform(0, 1e-4, (zc.size, 1, 4, 4, 172))
    prov = dict(code_commit="x", cache_sha256="cache", inputs={}, row_rule="A1 s1")
    path = str(tmp_path / "t1.npz")
    save_product(path, "cemu_t1", k_com_hmpc=KCOM, provenance=prov, rho=rho, z_cells=zc, alpha_centres=np.array([1.0]))
    per_leg, ac, digest = CL._t1_per_leg(path, [_leg([2.4, 3.8])], KCOM, "cache")
    assert per_leg["DESI"].shape == (2, 4, 4, 172, 1) and np.asarray(ac).tolist() == [1.0]
    np.testing.assert_array_equal(np.asarray(per_leg["DESI"])[1, ..., 0], rho[8, 0])
    with pytest.raises(ValueError, match="cell"):
        CL._t1_per_leg(path, [_leg([5.0])], KCOM, "cache")


def test_t3_product_is_bound_to_the_legs_exact_bins(tmp_path):
    from hcd_analysis.emulator.products import save_product
    leg = _leg([2.6, 3.4])
    rng = np.random.default_rng(4)
    U, w = rng.normal(0, 0.01, (leg.k.size, 3)), rng.uniform(0.5, 1, 3)
    prov = dict(code_commit="x", cache_sha256="cache", inputs={}, row_rule="A1 s3", representation="P")
    path = str(tmp_path / "t3.npz")
    save_product(path, "cemu_t3", k_com_hmpc=KCOM, provenance=prov, U_DESI=U, w_DESI=w, k_DESI=leg.k, z_DESI=leg.z_row)
    per_leg, digest = CL._t3_per_leg(path, [leg, _leg([3.0], name="eBOSS")], "cache")
    np.testing.assert_array_equal(np.asarray(per_leg["DESI"][0]), U)
    assert "eBOSS" not in per_leg                                       # production: no T3 on eBOSS
    moved = leg._replace(k=leg.k * 1.001)
    with pytest.raises(ValueError, match="bins"):
        CL._t3_per_leg(path, [moved], "cache")


def test_t2_product_is_held_for_the_pi_ruling():
    with pytest.raises(NotImplementedError, match="S14"):
        CL.build_legb_ctx(ensemble_ckpts=["x"], t2_product="t2.npz", ks_kwargs={"k_max": 0.065})


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
    ll = RF._loglik_chain(ctx, {"DESI": jnp.zeros(ok.k.size)}, samples, None)
    assert np.asarray(ll).shape == (2,) and np.all(np.isfinite(np.asarray(ll)))
    bad = _leg([3.0], k_lo=1.2e-3, k_hi=0.12)
    ctx_bad = _ctx(model, pf, [bad], [3.0])._replace(fix_alpha_res=True, metal_prior="uniform")
    with pytest.raises(ValueError, match="outside the simulated modes"):
        RF._loglik_chain(ctx_bad, {"DESI": jnp.zeros(bad.k.size)}, samples, None)


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
