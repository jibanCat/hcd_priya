import numpy as np, jax, jax.numpy as jnp
from hcd_analysis.emulator.likelihood import total_p1d_difference, total_p1d_ratio, assemble_covariance


def test_cross_class_additivity():
    K = 8; w = jnp.array([0.7, 0.18, 0.07, 0.05])
    P_filt = jnp.abs(jnp.ones((4, K)))
    delta = jnp.array(np.random.default_rng(0).normal(0, 0.01, (3, K)))
    A = jnp.ones(3)
    P_tier_p = jnp.einsum("c,ck->k", w, P_filt)
    P_total = total_p1d_difference(P_tier_p, w[1:], A, delta)
    assert jnp.allclose(P_total - P_tier_p, jnp.einsum("c,c,ck->k", w[1:], A, delta), atol=1e-12)


def test_difference_form_differentiable_through_wc():
    K = 8
    P_filt = jnp.ones((4, K)); delta = jnp.zeros((3, K)); A = jnp.ones(3)
    w = jnp.array([0.7, 0.18, 0.07, 0.05])

    def f(d):
        P_tier_p = jnp.einsum("c,ck->k", w, P_filt)
        return total_p1d_difference(P_tier_p, w[1:], A, d).sum()

    g = jax.grad(f)(delta)
    assert jnp.all(jnp.isfinite(g))


def test_covariance_inflates_dla_high_k():
    K = 8
    sigma = jnp.ones((4, K)) * 0.01
    w = jnp.array([0.7, 0.18, 0.07, 0.05])
    cov_noflag = assemble_covariance(jnp.ones(K) * 1e-3, sigma, w, dla_shot_flag=jnp.zeros(K, bool))
    flag = jnp.arange(K) >= 6
    cov_flag = assemble_covariance(jnp.ones(K) * 1e-3, sigma, w, dla_shot_flag=flag)
    assert jnp.all(jnp.diag(cov_flag)[6:] > jnp.diag(cov_noflag)[6:])
