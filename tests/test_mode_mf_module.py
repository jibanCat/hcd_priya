"""Gate E task 2 (GATE_E_SPEC v1 section 2): the JAX-pure MF correction on the mode axis. One implementation
(``mf_modes.ModeMF``); ``apply_mode_mf`` delegates to it. The correction reads only (z, tau0): its derivative with
respect to every cosmology parameter is exactly zero (E4a, E6a)."""
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hcd_analysis.emulator import mf_modes as MM

PRODUCT = "/nfs/turbo/umor-yueyingn/mfho/hcd/emulator_v2/mf/gateD/mf_modes_all6.npz"


def _synthetic_tables(K=5, C=4, nz=3, nr=4, seed=0):
    rng = np.random.default_rng(seed)
    z_tab = np.array([2.6, 3.0, 3.4])[:nz]
    tau_by_z = np.sort(rng.uniform(0.2, 0.9, (nz, nr)), axis=1)
    return dict(log_rho=rng.normal(0, 0.02, K), gbar_z_tab=rng.normal(0, 0.02, (nz, C, K)),
                gtau_tab=rng.normal(0, 0.01, (nr, C, K)), a_k=rng.normal(0, 0.01, (C, K)),
                u_z=rng.normal(0, 1, nz), u_tau=rng.normal(0, 1, nr), z_tab=z_tab,
                tau_tab=np.arange(nr, dtype=float), tau_by_z=tau_by_z)


def _x(z, seed=1):
    th = np.random.default_rng(seed).uniform(0.1, 0.9, 9)
    return np.concatenate([th, [(z - 2.0) / 3.4]])


def test_module_equals_the_table_application_bitwise():
    t = _synthetic_tables()
    mf = MM.ModeMF.from_tables(t)
    for z, tau0 in ((2.7, 0.4), (3.0, 0.55), (3.3, 0.8)):
        assert np.array_equal(np.asarray(mf(jnp.asarray(_x(z)), tau0)), MM.apply_mode_mf(t, _x(z), tau0))


def test_module_is_jittable_and_exactly_blind_to_cosmology():
    mf = MM.ModeMF.from_tables(_synthetic_tables())
    x = jnp.asarray(_x(3.1)); tau0 = 0.5
    f = jax.jit(lambda xx, tt: mf(xx, tt))
    np.testing.assert_allclose(np.asarray(f(x, tau0)), np.asarray(mf(x, tau0)), rtol=1e-14, atol=0)   # XLA reorders: 1 ulp
    J = jax.jacfwd(lambda xx: mf(xx, tau0))(x)            # (C, K, 10)
    assert np.all(np.asarray(J[..., :9]) == 0.0)
    assert np.any(np.asarray(J[..., 9]) != 0.0)             # it does depend on z


def test_rank1_off_removes_exactly_the_interaction_term():
    t = _synthetic_tables()
    mf, off = MM.ModeMF.from_tables(t), MM.ModeMF.from_tables(t).rank1_off()
    t0 = dict(t, a_k=np.zeros_like(t["a_k"]))
    x = _x(2.9)
    assert np.array_equal(np.asarray(off(jnp.asarray(x), 0.45)), MM.apply_mode_mf(t0, x, 0.45))
    assert not np.array_equal(np.asarray(mf(jnp.asarray(x), 0.45)), np.asarray(off(jnp.asarray(x), 0.45)))


@pytest.mark.parametrize("key", ["log_rho", "gbar_z_tab", "gtau_tab", "a_k", "tau_by_z"])
def test_non_finite_tables_are_refused(key):
    t = _synthetic_tables()
    t[key] = np.array(t[key], float)
    t[key].flat[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        MM.ModeMF.from_tables(t)


def test_production_product_module_equals_table_application():
    if not os.path.exists(PRODUCT):
        if os.environ.get("HCD_GATE_RUN") == "1":
            pytest.fail(f"gate D product absent: {PRODUCT}")
        pytest.skip(f"gate D product absent: {PRODUCT}")
    tables, k_com, prov = MM.load_mode_mf(PRODUCT)
    mf = MM.ModeMF.from_tables(tables)
    for z, tau0 in ((2.2, 0.15), (3.0, 0.35), (4.6, 1.0)):
        assert np.array_equal(np.asarray(mf(jnp.asarray(_x(z)), tau0)), MM.apply_mode_mf(tables, _x(z), tau0))
