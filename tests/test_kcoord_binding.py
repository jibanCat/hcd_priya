"""Gate E task 1 (GATE_E_SPEC v1 section 2): the one per-(leg, z) binding that places mode-axis quantities at data k.
u = k_data / k_skm,1(z, theta); because k_skm,n = n k_skm,1 exactly, linear interpolation on the fixed mode axis at the
fractional index u IS the production linear-in-k interpolation, without a theta-dependent abscissa."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hcd_analysis.emulator import kcoord as KC, schema as S
from hcd_analysis.emulator.data import PARAM_LIMITS

K = 172
KCOM = 2 * np.pi * np.arange(1, K + 1) / S.L_BOX_HMPC


def _theta(seed):
    return jnp.asarray(np.random.default_rng(seed).uniform(0.05, 0.95, 9))


@pytest.mark.parametrize("z", [2.2, 3.0, 4.6])
def test_binding_equals_linear_interpolation_in_k(z):
    rng = np.random.default_rng(1)
    th = _theta(int(10 * z))
    kg = KC.kgrid(KCOM, z, th)
    k1, kK = float(kg.k_skm[0]), float(kg.k_skm[-1])
    kd = jnp.asarray(np.sort(rng.uniform(k1, kK, 200)))
    P = jnp.asarray(np.exp(rng.normal(size=K)))
    b = KC.bind(kg, kd)
    assert int(b.n_out) == 0
    # identical up to round-off: jnp.interp divides by k_{j+1} - k_j (relative cancellation ~ n eps), amplified by
    # this test spectrum's large random mode-to-mode jumps (measured <= 2.5e-13)
    np.testing.assert_allclose(np.asarray(KC.at_data(b, P)), np.asarray(jnp.interp(kd, kg.k_skm, P)), rtol=1e-12)


def test_binding_gradient_in_theta_equals_interp_with_moving_abscissa():
    rng = np.random.default_rng(2)
    kd = jnp.asarray(np.sort(rng.uniform(1.2e-3, 0.05, 100)))
    P = jnp.asarray(np.exp(rng.normal(size=K)))

    def via_binding(th):
        return jnp.sum(KC.at_data(KC.bind(KC.kgrid(KCOM, 3.0, th), kd), P) ** 2)

    def via_interp(th):
        return jnp.sum(jnp.interp(kd, KC.kgrid(KCOM, 3.0, th).k_skm, P) ** 2)

    th = _theta(3)
    np.testing.assert_allclose(np.asarray(jax.grad(via_binding)(th)), np.asarray(jax.grad(via_interp)(th)),
                               rtol=1e-10, atol=1e-14)


def test_binding_counts_bins_outside_the_modes_and_clamps_like_interp():
    kg = KC.kgrid(KCOM, 3.0, _theta(4))
    k1, kK = float(kg.k_skm[0]), float(kg.k_skm[-1])
    kd = jnp.asarray([0.5 * k1, k1, 0.5 * (k1 + kK), kK, 1.5 * kK])
    P = jnp.arange(1.0, K + 1.0)
    b = KC.bind(kg, kd)
    assert int(b.n_out) == 2
    np.testing.assert_allclose(np.asarray(KC.at_data(b, P)), np.asarray(jnp.interp(kd, kg.k_skm, P)), rtol=1e-13)


def test_binding_log_weights():
    kg = KC.kgrid(KCOM, 2.6, _theta(5))
    kd = jnp.asarray([2.5 * float(kg.k_skm[0]), 37.25 * float(kg.k_skm[0])])
    b = KC.bind(kg, kd)
    u = np.array([2.5, 37.25]); j = np.floor(u)
    np.testing.assert_allclose(np.asarray(b.j), j)
    np.testing.assert_allclose(np.asarray(b.t_lin), u - j, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(b.t_log), (np.log(u) - np.log(j)) / (np.log(j + 1) - np.log(j)), rtol=1e-12)


@pytest.mark.parametrize("z", [2.2, 3.4, 4.6])
def test_box_bounds_equal_a_brute_force_scan(z):
    lo = np.zeros(9); hi = np.ones(9)
    k1max, kKmin = KC.kbounds_over_box(KCOM, z, lo, hi)
    lim = np.asarray(PARAM_LIMITS, float)
    hubs = np.linspace(lim[5, 0], lim[5, 1], 41); oms = np.linspace(lim[6, 0], lim[6, 1], 41)
    k1 = [float(KC.k_skm_from_kcom(KCOM[:1], z, h, o)[0]) for h in hubs for o in oms]
    kK = [float(KC.k_skm_from_kcom(KCOM[-1:], z, h, o)[0]) for h in hubs for o in oms]
    assert abs(k1max / max(k1) - 1) < 1e-14 and abs(kKmin / min(kK) - 1) < 1e-14


def test_box_bounds_follow_a_narrowed_sampling_box():
    lo = np.zeros(9); hi = np.ones(9)
    lo[5], hi[5] = 0.4, 0.6
    a = KC.kbounds_over_box(KCOM, 3.0, lo, hi)
    b = KC.kbounds_over_box(KCOM, 3.0, np.zeros(9), np.ones(9))
    assert a[0] < b[0] and a[1] > b[1]


def test_kgrid_and_binding_pass_through_jit():
    f = jax.jit(lambda th: KC.bind(KC.kgrid(KCOM, 3.0, th), jnp.asarray([0.01, 0.02])))
    b = f(_theta(6))
    assert b.kg.z == 3.0 and b.kg.schema_version == S.CHECKPOINT_SCHEMA_VERSION
    assert int(b.n_out) == 0
