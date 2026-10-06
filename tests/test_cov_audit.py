"""Gate E amendment A1 rev 1 sections 1 and 5: the covariance-force audit kit (cov_audit). The expected-likelihood
gradient equals its trace formula, crossings are exact, the covariance-information ratio is its definition, and the
nonlinear (Newton) shift reduces to the linear one for a quadratic. Cache-free."""
import hcd_analysis.emulator  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np

from hcd_analysis.emulator import cov_audit as CA
from hcd_analysis.emulator import kcoord as KC
from hcd_analysis.emulator.schema import L_BOX_HMPC

KCOM = 2 * np.pi * np.arange(1, 173) / L_BOX_HMPC


def _C(p, A, Bs):
    return A + sum(pi * B for pi, B in zip(p, Bs))


def test_expected_loglik_gradient_is_the_trace_formula():
    rng = np.random.default_rng(0)
    N = 6
    X = rng.normal(0, 1, (N, N)); A = X @ X.T + N * np.eye(N)
    Bs = [(lambda Y: Y @ Y.T)(rng.normal(0, 0.3, (N, N))) for _ in range(3)]
    S = (lambda Y: Y @ Y.T)(rng.normal(0, 1, (N, N)))
    p = jnp.asarray([0.1, -0.2, 0.3])
    g = np.asarray(jax.grad(CA.expected_loglik(lambda q: _C(q, A, Bs)))(p, jnp.asarray(S)))
    C = np.asarray(_C(p, A, Bs)); Ci = np.linalg.inv(C)
    expect = [-0.5 * np.trace(Ci @ B) + 0.5 * np.trace(Ci @ B @ Ci @ S) for B in Bs]
    np.testing.assert_allclose(g, expect, rtol=1e-10)


def test_crossings_are_where_a_bin_sits_exactly_on_a_mode():
    centre = np.full(9, 0.5)
    k_bins, iz = np.array([0.012, 0.03, 0.055]), np.array([0, 0, 1])
    zs = np.array([2.6, 3.8])
    xs = CA.crossings(KCOM, zs, iz, k_bins, centre, 5, 0.0, 1.0)
    assert len(xs) > 0
    for x, b in xs:
        th = centre.copy(); th[5] = x
        u = k_bins[b] / float(KC.k_skm_from_theta9(KCOM, float(zs[iz[b]]), jnp.asarray(th))[0])
        assert abs(u - round(u)) < 1e-9
    for b in range(3):                           # the count per bin is the number of integers between the end u's
        u = [k_bins[b] / float(KC.k_skm_from_theta9(KCOM, float(zs[iz[b]]), jnp.asarray(np.r_[centre[:5], e, centre[6:]]))[0])
             for e in (0.0, 1.0)]
        n_int = int(np.floor(max(u))) - int(np.ceil(min(u))) + 1
        assert sum(1 for _, bb in xs if bb == b) == n_int


def test_information_ratio_is_its_definition():
    rng = np.random.default_rng(1)
    N, n = 7, 3
    X = rng.normal(0, 1, (N, N)); C = X @ X.T + N * np.eye(N)
    dC = [(lambda Y: Y + Y.T)(rng.normal(0, 0.2, (N, N))) for _ in range(n)]
    J = rng.normal(0, 1, (N, n))
    Ci = np.linalg.inv(C)
    expect = [0.5 * np.trace(Ci @ dC[i] @ Ci @ dC[i]) / (J.T @ Ci @ J)[i, i] for i in range(n)]
    np.testing.assert_allclose(CA.info_ratio(C, dC, J), expect, rtol=1e-12)


def test_newton_shift_equals_the_linear_shift_for_a_quadratic():
    rng = np.random.default_rng(2)
    n = 4
    H = (lambda Y: Y @ Y.T + n * np.eye(n))(rng.normal(0, 1, (n, n)))
    F = (lambda Y: Y @ Y.T + np.eye(n))(rng.normal(0, 1, (n, n)))
    g = rng.normal(0, 0.1, n)
    LC = lambda d: jnp.dot(jnp.asarray(g), d) - 0.5 * d @ jnp.asarray(H) @ d      # quadratic covariance part
    d = CA.newton_shift(LC, F, np.zeros(n), sigma=np.full(n, 10.0))
    np.testing.assert_allclose(d, np.linalg.solve(F + H, g), rtol=1e-8)


def test_fisher_scoring_converges_to_the_same_maximizer_for_a_quadratic():
    rng = np.random.default_rng(3)
    n = 4
    H = (lambda Y: Y @ Y.T + n * np.eye(n))(rng.normal(0, 1, (n, n)))
    F = (lambda Y: Y @ Y.T + np.eye(n))(rng.normal(0, 1, (n, n)))
    g = rng.normal(0, 0.1, n)
    LC = lambda d: jnp.dot(jnp.asarray(g), d) - 0.5 * d @ jnp.asarray(H) @ d
    d = CA.newton_shift(LC, F, np.zeros(n), sigma=np.full(n, 10.0), metric=H)
    np.testing.assert_allclose(d, np.linalg.solve(F + H, g), rtol=1e-8)
