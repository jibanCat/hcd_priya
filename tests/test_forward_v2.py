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
    P_old, _ = DL.predict_P_obs_on_leg(model, th, tau0, alpha, pf_stats=pf, dla_core=core, cache_k=grid, leg=leg,
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
        P1, _ = DL.predict_P_obs_on_leg(model, th, tau0[iz:iz + 1], alpha[iz:iz + 1], pf_stats=pf, dla_core=core,
                                        cache_k=np.asarray(KC.kgrid(KCOM, z, th).k_skm), leg=sub)
        np.testing.assert_allclose(np.asarray(out.P_model)[leg.z_idx == iz], np.asarray(P1), rtol=1e-12)
    # the incident: one fixed grid for every z differs
    P_row0, _ = DL.predict_P_obs_on_leg(model, th, tau0, alpha, pf_stats=pf, dla_core=core,
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
