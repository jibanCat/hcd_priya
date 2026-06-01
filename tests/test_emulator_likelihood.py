import numpy as np, jax, jax.numpy as jnp
from hcd_analysis.emulator.likelihood import total_p1d_difference, total_p1d_ratio, assemble_covariance
from hcd_analysis.emulator.model import structural_tier_p


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


def test_difference_grad_through_w_and_A():
    """grad must flow through ALL likelihood inputs, not just delta: w_hcd, A_hcd, delta."""
    K = 8
    rng = np.random.default_rng(2)
    P_filt = jnp.array(rng.uniform(0.5, 1.5, (4, K)))
    delta = jnp.array(rng.normal(0, 0.05, (3, K)))
    ratio = jnp.array(rng.normal(0, 0.05, (3, K)))
    w_c = jnp.array([0.7, 0.18, 0.07, 0.05])
    A = jnp.array([1.1, 0.9, 1.2])

    def diff_loss(w_hcd, A_hcd, d):
        P_tier_p = jnp.einsum("c,ck->k", w_c, P_filt)  # constant in these args
        return total_p1d_difference(P_tier_p, w_hcd, A_hcd, d).sum()

    gw, gA, gd = jax.grad(diff_loss, argnums=(0, 1, 2))(w_c[1:], A, delta)
    for g in (gw, gA, gd):
        assert jnp.all(jnp.isfinite(g)) and g.dtype == jnp.float64
    # difference form: d/dA_c = w_c * sum_k delta_c (nonzero -> grad actually flows through A)
    assert jnp.allclose(gA, w_c[1:] * delta.sum(axis=1), atol=1e-12)

    def ratio_loss(w_hcd, A_hcd, r):
        P_tier_p = jnp.einsum("c,ck->k", w_c, P_filt)
        return total_p1d_ratio(P_tier_p, w_hcd, A_hcd, r).sum()

    gw2, gA2, gr2 = jax.grad(ratio_loss, argnums=(0, 1, 2))(w_c[1:], A, ratio)
    for g in (gw2, gA2, gr2):
        assert jnp.all(jnp.isfinite(g)) and g.dtype == jnp.float64


def test_structural_coupling_grad_through_wc():
    """Head-A<->Head-B coupling (spec sec.6): the SAME w_c feeds both the structural
    P_tier_p (Head A) and the HCD add-back. grad d(total)/d(w_c) must flow through BOTH,
    so each HCD class's grad = structural term + add-back term, while the clean class
    (index 0, not in the add-back) gets the structural term only."""
    K = 8
    rng = np.random.default_rng(3)
    P_filt = jnp.array(rng.uniform(0.5, 1.5, (4, K)))
    delta = jnp.array(rng.normal(0, 0.05, (3, K)))
    A = jnp.array([1.1, 0.9, 1.2])
    w_c = jnp.array([0.7, 0.18, 0.07, 0.05])

    def total_loss(w_full):
        P_tier_p = structural_tier_p(w_full, P_filt)  # depends on ALL 4 weights
        return total_p1d_difference(P_tier_p, w_full[1:], A, delta).sum()  # same w -> coupling

    g = jax.grad(total_loss)(w_c)
    assert jnp.all(jnp.isfinite(g)) and g.dtype == jnp.float64

    struct = P_filt.sum(axis=1)  # (4,)  d(P_tier_p.sum)/d(w_c)
    addback = jnp.concatenate([jnp.zeros(1), (A[:, None] * delta).sum(axis=1)])  # (4,), clean=0
    assert jnp.allclose(g, struct + addback, atol=1e-12)
    # clean class: structural only; HCD classes: strictly both terms (coupling is live)
    assert jnp.allclose(g[0], struct[0], atol=1e-12)
    assert not jnp.allclose(g[1:], struct[1:], atol=1e-9)


def test_covariance_inflates_dla_high_k():
    K = 8
    sigma = jnp.ones((4, K)) * 0.01
    w = jnp.array([0.7, 0.18, 0.07, 0.05])
    cov_noflag = assemble_covariance(jnp.ones(K) * 1e-3, sigma, w, dla_shot_flag=jnp.zeros(K, bool))
    flag = jnp.arange(K) >= 6
    cov_flag = assemble_covariance(jnp.ones(K) * 1e-3, sigma, w, dla_shot_flag=flag)
    assert jnp.all(jnp.diag(cov_flag)[6:] > jnp.diag(cov_noflag)[6:])
