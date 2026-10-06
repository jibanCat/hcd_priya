"""Gate E task 4 (GATE_E_SPEC v1 section 2): the leg forward on the canonical coordinate (``forward.py``).

Reference: on a single-z leg, the pre-2026-10 forward fed ``cache_k`` = that z's own physical grid for the query theta
is the correct forward (the incident was feeding ONE grid to every z and theta). The new forward must equal it there,
and must place every z on its own grid on a multi-z leg. Cache-free (random-init Emulator, synthetic legs)."""
import hcd_analysis.emulator  # noqa: F401  x64 before jax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hcd_analysis.emulator import data_likelihood as DL
from tests.regression import legacy_forward_pre2026_10 as LEG  # the pre-2026-10 forward (historical fixture, gate E)
from hcd_analysis.emulator import forward as FW
from hcd_analysis.emulator import kcoord as KC
from hcd_analysis.emulator.model import Emulator
from hcd_analysis.emulator.schema import L_BOX_HMPC

K = 172
KCOM = 2 * np.pi * np.arange(1, K + 1) / L_BOX_HMPC


def _emu(seed=0):
    rng = np.random.default_rng(seed)
    model = Emulator(in_dim=10, n_k=K, n_basis=12, key=jax.random.PRNGKey(seed))
    pf = {"mu_marg": jnp.asarray(rng.normal(-2, 0.5, (4, K))),
          "sig_marg": jnp.asarray(rng.uniform(0.5, 1.5, (4, K))),
          "sig_cosmo": jnp.asarray(rng.uniform(0.02, 0.1, (4, K)))}
    return model, pf, jnp.asarray(rng.uniform(0.0, 0.5, K))


def _leg(zs, n_per_z=9, k_lo=1.2e-3, k_hi=0.05, metals=False, resolution=False, name="DESI", dff=1.0):
    zs = np.asarray(zs, float)
    k_one = np.geomspace(k_lo, k_hi, n_per_z)
    k = np.concatenate([k_one] * zs.size)
    z_idx = np.repeat(np.arange(zs.size), n_per_z)
    return DL.DataLeg(name=name, z=zs, z_unit=(zs - 2.0) / 3.4, k=k, z_row=zs[z_idx], z_idx=z_idx,
                      P_data=np.full(k.size, 0.1), C_data=np.eye(k.size) * 1e-4, R_z=np.full(zs.size, 12.0),
                      n_z=zs.size, n_per_z=np.full(zs.size, n_per_z), metals_on=metals, resolution_on=resolution,
                      dla_forward_frac=dff)


NUIS = dict(f_SiIII_nodes=jnp.asarray([0.006, 0.012]), f_SiII_nodes=jnp.asarray([0.002, 0.003]),
            k_SiIII_nodes=jnp.asarray([0.05, 0.04]), k_SiII_nodes=jnp.asarray([0.05, 0.05]), b_res=0.05)


@pytest.mark.parametrize("z", [2.2, 3.0, 4.2])
@pytest.mark.parametrize("on", [False, True], ids=["bare", "metals_resolution"])
def test_single_z_leg_equals_the_old_forward_on_that_z_grid(z, on):
    model, pf, core = _emu()
    th = jnp.asarray(np.random.default_rng(int(10 * z)).uniform(0.05, 0.95, 9))
    leg = _leg([z], metals=on, resolution=on)
    tau0 = jnp.asarray([0.25])
    alpha = jnp.asarray([[0.05, 0.02, 0.004]])
    nuis = NUIS if on else {}
    grid = np.asarray(KC.kgrid(KCOM, z, th).k_skm)
    P_old, _ = LEG.predict_P_obs_on_leg(model, th, tau0, alpha, pf_stats=pf, dla_core=core, cache_k=grid, leg=leg,
                                       require_zresolved=True, **nuis)
    out = FW.predict_leg(model, th, tau0, alpha, leg=leg, k_com=KCOM, pf_stats=pf, dla_core=core, nuis=nuis)
    assert int(out.n_out) == 0
    np.testing.assert_allclose(np.asarray(out.P_model), np.asarray(P_old), rtol=1e-12)


def test_every_z_of_a_leg_is_placed_on_its_own_grid():
    model, pf, core = _emu(1)
    th = jnp.asarray(np.full(9, 0.3))
    zs = [2.4, 3.2, 4.0]
    leg = _leg(zs)
    tau0 = jnp.asarray([0.2, 0.35, 0.6]); alpha = jnp.asarray([[0.05, 0.02, 0.004]] * 3)
    out = FW.predict_leg(model, th, tau0, alpha, leg=leg, k_com=KCOM, pf_stats=pf, dla_core=core)
    for iz, z in enumerate(zs):
        sub = _leg([z])
        P1, _ = LEG.predict_P_obs_on_leg(model, th, tau0[iz:iz + 1], alpha[iz:iz + 1], pf_stats=pf, dla_core=core,
                                        cache_k=np.asarray(KC.kgrid(KCOM, z, th).k_skm), leg=sub)
        np.testing.assert_allclose(np.asarray(out.P_model)[leg.z_idx == iz], np.asarray(P1), rtol=1e-12)
    # the incident: one fixed grid for every z differs
    P_row0, _ = LEG.predict_P_obs_on_leg(model, th, tau0, alpha, pf_stats=pf, dla_core=core,
                                        cache_k=np.asarray(KC.kgrid(KCOM, zs[0], th).k_skm), leg=leg)
    assert np.max(np.abs(np.asarray(P_row0) / np.asarray(out.P_model) - 1)) > 1e-3


def test_bins_outside_the_modes_are_counted():
    model, pf, core = _emu(2)
    th = jnp.asarray(np.full(9, 0.5))
    leg = _leg([3.0], k_lo=2e-4, k_hi=0.12, n_per_z=12)
    out = FW.predict_leg(model, th, jnp.asarray([0.3]), jnp.asarray([[0.05, 0.02, 0.004]]), leg=leg, k_com=KCOM,
                         pf_stats=pf, dla_core=core)
    kg = KC.kgrid(KCOM, 3.0, th)
    expect = int(np.sum((leg.k < float(kg.k_skm[0])) | (leg.k > float(kg.k_skm[-1]))))
    assert expect > 0 and int(out.n_out) == expect


def test_mode_axis_mf_multiplies_each_class_before_binding():
    from hcd_analysis.emulator.mf_modes import ModeMF
    from tests.test_mode_mf_module import _synthetic_tables
    model, pf, core = _emu(3)
    t = _synthetic_tables(K=K)
    mf = ModeMF.from_tables(t)
    th = jnp.asarray(np.full(9, 0.4)); z = 3.0
    leg = _leg([z]); tau0 = jnp.asarray([0.5]); alpha = jnp.asarray([[0.05, 0.02, 0.004]])
    out = FW.predict_leg(model, th, tau0, alpha, leg=leg, k_com=KCOM, pf_stats=pf, dla_core=core, mf=mf)
    from hcd_analysis.emulator.predict import predict_P_filt
    x = jnp.concatenate([th, jnp.asarray([(z - 2.0) / 3.4])])
    Pf = predict_P_filt(model, th, (z - 2.0) / 3.4, tau0[0], pf) * jnp.exp(mf(x, tau0[0]))
    b = KC.bind(KC.kgrid(KCOM, z, th), jnp.asarray(leg.k))
    Pc = KC.at_data(b, Pf); cr = KC.at_data(b, core)
    a = alpha[0]
    expect = Pc[0] + a[0] * (Pc[1] - Pc[0]) + a[1] * (Pc[2] - Pc[0]) + a[2] * (Pc[3] + cr - Pc[0])
    np.testing.assert_allclose(np.asarray(out.P_model), np.asarray(expect), rtol=1e-12)


def test_z_flat_alpha_is_refused_by_default():
    model, pf, core = _emu(4)
    leg = _leg([2.4, 3.6])
    with pytest.raises(ValueError, match="z-RESOLVED"):
        FW.predict_leg(model, jnp.full(9, 0.5), jnp.asarray([0.2, 0.4]), jnp.asarray([0.05, 0.02, 0.004]), leg=leg,
                       k_com=KCOM, pf_stats=pf, dla_core=core)


def test_metal_factor_at_z_model_cplus_equals_the_formula():
    k = jnp.geomspace(1e-3, 0.05, 30); z = 3.1; tau0 = 0.4
    f3, f2, k3, k2 = (0.006, 0.012), (0.002, 0.003), (0.05, 0.04), (0.05, 0.05)
    lz, xp = np.log10(1 + z), np.log10(1 + np.array([2.2, 4.2]))
    omF = 1 - np.exp(-tau0)
    li = lambda v: 10 ** np.interp(lz, xp, np.log10(v))
    expect = DL._metal_factor(k, a_SiIII=li(f3) / omF, a_SiII=li(f2) / omF, k_SiIII=li(k3), k_SiII=li(k2), cross=True)
    got = DL.metal_factor_at_z(k, z, tau0, f_SiIII_nodes=jnp.asarray(f3), f_SiII_nodes=jnp.asarray(f2),
                               k_SiIII_nodes=jnp.asarray(k3), k_SiII_nodes=jnp.asarray(k2))
    np.testing.assert_allclose(np.asarray(got), np.asarray(expect), rtol=1e-13)


def _t1(n_z, seed=7, Tb=4):
    """A synthetic cross-class block per leg z: (n_z, 4, 4, K, Tb), SPD per (k, band), NaN cells at mode 1."""
    rng = np.random.default_rng(seed)
    A = rng.normal(0, 0.01, (n_z, 4, 4, K, Tb))
    rho = np.einsum("zcekt,zdekt->zcdkt", A, A)
    rho[:, :, :, 0, :] = np.nan
    return jnp.asarray(rho), jnp.asarray([0.656, 0.833, 1.153, 1.331])


@pytest.mark.parametrize("on", [False, True], ids=["bare", "metals_resolution"])
def test_t1_variance_equals_the_old_path_on_each_z_grid(on):
    model, pf, core = _emu(5)
    th = jnp.asarray(np.random.default_rng(9).uniform(0.05, 0.95, 9))
    zs = [2.6, 3.4]
    leg = _leg(zs, metals=on, resolution=on)
    tau0 = jnp.asarray([0.3, 0.5]); alpha = jnp.asarray([[0.05, 0.02, 0.004], [0.07, 0.03, 0.006]])
    rho, ac = _t1(len(zs))
    nuis = NUIS if on else {}
    out = FW.predict_leg(model, th, tau0, alpha, leg=leg, k_com=KCOM, pf_stats=pf, dla_core=core, nuis=nuis,
                         t1=(rho, ac))
    for iz, z in enumerate(zs):
        sub = _leg([z], metals=on, resolution=on)
        sub = sub._replace(R_z=np.asarray(leg.R_z)[iz:iz + 1])
        _, C_old = LEG.predict_P_obs_on_leg(model, th, tau0[iz:iz + 1], alpha[iz:iz + 1], pf_stats=pf, dla_core=core,
                                           cache_k=np.asarray(KC.kgrid(KCOM, z, th).k_skm), leg=sub,
                                           rho_zb=rho[iz:iz + 1], alpha_centres=ac, **nuis)
        rows = np.where(leg.z_idx == iz)[0]
        np.testing.assert_allclose(np.diag(np.asarray(out.C_total))[rows], np.diag(np.asarray(C_old)), rtol=1e-11)


def test_without_t1_the_covariance_is_the_data_covariance():
    model, pf, core = _emu(6)
    leg = _leg([3.0])
    out = FW.predict_leg(model, jnp.full(9, 0.5), jnp.asarray([0.3]), jnp.asarray([[0.05, 0.02, 0.004]]), leg=leg,
                         k_com=KCOM, pf_stats=pf, dla_core=core)
    assert np.array_equal(np.asarray(out.C_total), np.asarray(leg.C_data))


def test_t2_floor_variance_equals_the_old_single_band_floor():
    rng = np.random.default_rng(11)
    zg = np.array([2.2, 3.0, 3.8, 4.6]); sig = rng.uniform(0.005, 0.02, 4); slp = rng.uniform(0, 0.2, 4)
    old = LEG.MFFloor(z_grid=zg, sigma_floor=np.stack([sig, sig], 1), slope=np.stack([slp, slp], 1),
                     k_band_split=0.07, ns_box=np.array([0.86, 0.98]), floor_min=0.0, edge_slope_mult=2.0)
    k = jnp.geomspace(1e-3, 0.065, 15); P = jnp.asarray(rng.uniform(0.05, 0.2, 15))
    for z in (2.6, 3.0, 4.4):
        for ns in (0.9, 1.01, 0.83):
            s_z, l_z = np.interp(z, zg, sig), np.interp(z, zg, slp)
            np.testing.assert_allclose(np.asarray(FW.t2_var(s_z, l_z, P, ns)),
                                       np.asarray(LEG._mf_floor_var_on_k(old, z, k, P, ns)), rtol=1e-13)


def test_covariance_assembly_equals_the_production_algebra():
    rng = np.random.default_rng(12)
    N, m = 20, 4
    C_data = np.eye(N) * 1e-4; t1 = rng.uniform(1e-6, 1e-5, N); t2 = rng.uniform(1e-7, 1e-6, N)
    U = rng.normal(0, 0.01, (N, m)); w = rng.uniform(0.5, 2, m); P_fid = rng.uniform(0.05, 0.2, N)
    got = np.asarray(FW.assemble_cov(jnp.asarray(C_data), jnp.asarray(t1), jnp.asarray(t2),
                                     t3=(jnp.asarray(U), jnp.asarray(w)), P_fid=jnp.asarray(P_fid)))
    term = (U * w) @ U.T * np.outer(P_fid, P_fid)                 # the dense production term (infl 1)
    td = np.diag(term).copy()
    emu = np.maximum(t1, td); Cs = term - np.diag(td)
    expect = C_data + np.diag(emu + np.maximum(0.0, t2 - np.diag(Cs))) + Cs
    np.testing.assert_allclose(got, expect, rtol=1e-13, atol=1e-20)
    assert np.all(np.linalg.eigvalsh(got) > 0)
    np.testing.assert_allclose(np.asarray(FW.assemble_cov(jnp.asarray(C_data), jnp.asarray(t1), jnp.asarray(t2))),
                               C_data + np.diag(t1 + t2), rtol=1e-15)


def test_t3_offdiagonal_term_equals_the_old_dense_path_on_a_z_grid():
    model, pf, core = _emu(8)
    th = jnp.asarray(np.full(9, 0.45)); z = 3.2
    leg = _leg([z], n_per_z=12)
    tau0 = jnp.asarray([0.4]); alpha = jnp.asarray([[0.05, 0.02, 0.004]])
    rho, ac = _t1(1)
    rng = np.random.default_rng(13)
    U = rng.normal(0, 0.01, (12, 3)); w = rng.uniform(0.5, 2, 3)
    F = (U * w) @ U.T
    _, C_old = LEG.predict_P_obs_on_leg(model, th, tau0, alpha, pf_stats=pf, dla_core=core,
                                       cache_k=np.asarray(KC.kgrid(KCOM, z, th).k_skm), leg=leg, rho_zb=rho,
                                       alpha_centres=ac, mf_emucoh_cov=F, mf_emucoh_offdiag_only=True)
    out = FW.predict_leg(model, th, tau0, alpha, leg=leg, k_com=KCOM, pf_stats=pf, dla_core=core, t1=(rho, ac),
                         t3=(jnp.asarray(U), jnp.asarray(w)))
    np.testing.assert_allclose(np.asarray(out.C_total), np.asarray(C_old), rtol=1e-11, atol=1e-22)
