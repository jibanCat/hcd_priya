"""Gate E amendment A1 rev 1 section 3: the T3 representation comparison (mode-aligned M vs fixed physical k P). The
bin placement uses the same log-k weights as the production binding; truncation; the Gaussian score; and the comparison
favours the representation the synthetic truth was built in. Cache-free."""
import hcd_analysis.emulator  # noqa: F401
import jax.numpy as jnp
import numpy as np
import pytest

from hcd_analysis.emulator import cemu_t3 as T3
from hcd_analysis.emulator import kcoord as KC
from hcd_analysis.emulator.schema import L_BOX_HMPC

K = 172
KCOM = 2 * np.pi * np.arange(1, K + 1) / L_BOX_HMPC


def test_bin_weights_are_the_production_log_k_binding():
    th = np.random.default_rng(0).uniform(0.1, 0.9, 9)
    zs = np.array([2.4, 3.0]); kb = np.array([0.012, 0.02, 0.031, 0.015, 0.045])
    iz = np.array([0, 0, 0, 1, 1])
    W = T3.weights(KCOM, th, zs, iz, kb)
    f = np.random.default_rng(1).normal(0, 1, (2, K))
    for i in range(kb.size):
        b = KC.bind(KC.kgrid(KCOM, float(zs[iz[i]]), jnp.asarray(th)), jnp.asarray([kb[i]]))
        j, t = int(b.j[0]), float(b.t_log[0])
        np.testing.assert_allclose(W[i] @ f.ravel(), (1 - t) * f[iz[i], j - 1] + t * f[iz[i], j], rtol=1e-13)


def test_truncation_keeps_the_top_rank_and_the_discarded_diagonal_is_the_topup():
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, (30, 12))
    S = X.T @ X / 30
    F = T3.top_r(S, 4)
    w = np.linalg.eigvalsh(S)[::-1]
    np.testing.assert_allclose(np.linalg.eigvalsh(F)[::-1][:4], w[:4], rtol=1e-10)
    assert np.linalg.matrix_rank(F, tol=1e-10) == 4
    Sig = T3.with_topup(F, S)
    np.testing.assert_allclose(np.diag(Sig), np.diag(S), rtol=1e-12)


def test_gaussian_score_matches_scipy():
    from scipy.stats import multivariate_normal
    rng = np.random.default_rng(3)
    A = rng.normal(0, 1, (8, 8)); S = A @ A.T + np.eye(8)
    v = rng.normal(0, 1, 8)
    np.testing.assert_allclose(T3.logpdf(v, S, jitter=0.0), multivariate_normal(np.zeros(8), S).logpdf(v), rtol=1e-12)


def _synthetic(kind, n_sim=40, seed=4):
    """coh per simulation built as a fixed function of physical k ('P') or of the mode index ('M'), random amplitude
    of 3 smooth shapes; simulations spread over (hub, omegamh2)."""
    rng = np.random.default_rng(seed)
    zs = np.array([2.6, 3.4])
    th = rng.uniform(0.0, 1.0, (n_sim, 9))
    shapes_k = lambda k: np.stack([np.ones_like(k), np.log(k / 0.02), np.sin(8 * np.log(k / 0.01))])
    coh = np.zeros((n_sim, zs.size, K))
    for s in range(n_sim):
        a = rng.normal(0, 0.01, 3)
        for iz, z in enumerate(zs):
            if kind == "P":
                kk = np.asarray(KC.k_skm_from_theta9(KCOM, float(z), jnp.asarray(th[s])))
            else:
                kk = np.asarray(KC.k_skm_from_theta9(KCOM, float(z), jnp.asarray(np.full(9, 0.5))))
            coh[s, iz] = a @ shapes_k(kk) + rng.normal(0, 1e-4, K)
    kb = np.geomspace(0.011, 0.05, 15)
    bins = dict(z=zs, iz=np.repeat([0, 1], kb.size), k=np.concatenate([kb, kb]))
    return coh, th, bins


@pytest.mark.parametrize("kind", ["P", "M"])
def test_the_comparison_favours_the_representation_of_the_truth(kind):
    coh, th, bins = _synthetic(kind)
    res = T3.compare(coh, th, bins, KCOM, ranks=(3,), n_folds=5, lo=np.zeros(9), hi=np.ones(9))
    gain = res["score"]["M"][3].mean() - res["score"]["P"][3].mean()      # per held-out simulation
    assert (gain > 0) == (kind == "M"), gain


def test_hook_sees_every_held_out_simulation_with_its_covariance():
    coh, th, bins = _synthetic("P", n_sim=10)
    seen = []
    res = T3.compare(coh, th, bins, KCOM, ranks=(2,), n_folds=5, lo=np.zeros(9), hi=np.ones(9),
                     hook=lambda rep, r, s, Sig, v: (seen.append((rep, s)), Sig.shape)[1])
    assert sorted(seen) == sorted([(rep, s) for rep in ("M", "P") for s in range(10)])
    assert all(res["hook"][rep][2][s] == (30, 30) for rep in ("M", "P") for s in range(10))
