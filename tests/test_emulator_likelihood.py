import numpy as np, jax, jax.numpy as jnp
from hcd_analysis.emulator.likelihood import total_p1d_difference, total_p1d_ratio, assemble_covariance
from hcd_analysis.emulator.model import structural_tier_p


def test_cross_class_additivity():
    K = 8; w = jnp.array([0.7, 0.18, 0.07, 0.05])
    P_filt = jnp.abs(jnp.ones((4, K)))
    delta = jnp.array(np.random.default_rng(0).normal(0, 0.01, (3, K)))
    alpha = jnp.array(np.random.default_rng(1).uniform(0.0, 1.0, 3))
    P_tier_p = jnp.einsum("c,ck->k", w, P_filt)
    P_total = total_p1d_difference(P_tier_p, alpha, delta)
    assert jnp.allclose(P_total - P_tier_p, jnp.einsum("c,ck->k", alpha, delta), atol=1e-12)


def test_difference_form_differentiable_through_wc():
    K = 8
    P_filt = jnp.ones((4, K)); delta = jnp.zeros((3, K)); alpha = jnp.ones(3)
    w = jnp.array([0.7, 0.18, 0.07, 0.05])

    def f(d):
        P_tier_p = jnp.einsum("c,ck->k", w, P_filt)
        return total_p1d_difference(P_tier_p, alpha, d).sum()

    g = jax.grad(f)(delta)
    assert jnp.all(jnp.isfinite(g))


def test_difference_grad_through_alpha():
    """grad must flow through ALL free likelihood inputs: alpha_hcd and delta.

    alpha_c is the single per-class amplitude; the alpha-path must be live so the
    sampler can constrain residual HCD incidence. Difference form: d(P_obs)/d(alpha_c)
    summed over k = sum_k delta_c.
    """
    K = 8
    rng = np.random.default_rng(2)
    P_filt = jnp.array(rng.uniform(0.5, 1.5, (4, K)))
    delta = jnp.array(rng.normal(0, 0.05, (3, K)))
    ratio = jnp.array(rng.normal(0, 0.05, (3, K)))
    w_c = jnp.array([0.7, 0.18, 0.07, 0.05])
    alpha = jnp.array([1.1, 0.9, 1.2])

    def diff_loss(alpha_hcd, d):
        P_tier_p = jnp.einsum("c,ck->k", w_c, P_filt)  # constant in these args
        return total_p1d_difference(P_tier_p, alpha_hcd, d).sum()

    ga, gd = jax.grad(diff_loss, argnums=(0, 1))(alpha, delta)
    for g in (ga, gd):
        assert jnp.all(jnp.isfinite(g)) and g.dtype == jnp.float64
    # difference form: d/d(alpha_c) of the summed loss = sum_k delta_c (alpha-path is live)
    assert jnp.allclose(ga, delta.sum(axis=1), atol=1e-12)

    def ratio_loss(alpha_hcd, r):
        P_tier_p = jnp.einsum("c,ck->k", w_c, P_filt)
        return total_p1d_ratio(P_tier_p, alpha_hcd, r).sum()

    ga2, gr2 = jax.grad(ratio_loss, argnums=(0, 1))(alpha, ratio)
    for g in (ga2, gr2):
        assert jnp.all(jnp.isfinite(g)) and g.dtype == jnp.float64


def test_structural_coupling_grad_through_wc():
    """Structural baseline (spec sec.6): P_tier_p = structural_tier_p(w_c, P_filt) still
    uses the SIM 4-class w_c (the forest baseline; NOT reparametrized). The HCD add-back
    now uses the single standalone amplitude alpha (not w[1:]*A). grad d(total)/d(w_c)
    must stay finite and flow through P_tier_p for ALL 4 classes; since the add-back is
    independent of w_c, the w_c-grad is purely the structural term."""
    K = 8
    rng = np.random.default_rng(3)
    P_filt = jnp.array(rng.uniform(0.5, 1.5, (4, K)))
    delta = jnp.array(rng.normal(0, 0.05, (3, K)))
    alpha = jnp.array([1.1, 0.9, 1.2])
    w_c = jnp.array([0.7, 0.18, 0.07, 0.05])

    def total_loss(w_full):
        P_tier_p = structural_tier_p(w_full, P_filt)  # depends on ALL 4 weights
        return total_p1d_difference(P_tier_p, alpha, delta).sum()  # add-back independent of w

    g = jax.grad(total_loss)(w_c)
    assert jnp.all(jnp.isfinite(g)) and g.dtype == jnp.float64

    struct = P_filt.sum(axis=1)  # (4,)  d(P_tier_p.sum)/d(w_c)
    # add-back is independent of w_c -> w_c-grad is purely structural for every class
    assert jnp.allclose(g, struct, atol=1e-12)


def test_batched_alpha_matches_vmap_and_loop():
    """The ellipsis einsum must batch over leading dims: (B,3)/(B,3,K)->(B,K),
    consistent with a per-row vmap. Guards the `...c,...ck->...k` contract that the
    single-alpha refactor introduced."""
    K, B = 8, 5
    rng = np.random.default_rng(10)
    P_tier_p = jnp.array(rng.uniform(0.5, 1.5, (B, K)))
    alpha = jnp.array(rng.uniform(0.0, 1.0, (B, 3)))
    delta = jnp.array(rng.normal(0, 0.05, (B, 3, K)))
    ratio = jnp.array(rng.normal(0, 0.05, (B, 3, K)))

    bd = total_p1d_difference(P_tier_p, alpha, delta)
    br = total_p1d_ratio(P_tier_p, alpha, ratio)
    assert bd.shape == (B, K) and br.shape == (B, K)
    vd = jax.vmap(total_p1d_difference)(P_tier_p, alpha, delta)
    vr = jax.vmap(total_p1d_ratio)(P_tier_p, alpha, ratio)
    assert jnp.allclose(bd, vd, atol=1e-14)
    assert jnp.allclose(br, vr, atol=1e-14)


def test_ratio_form_alpha_zero_identity():
    """Ratio form must collapse to P_obs == P_tier_p EXACTLY at alpha=0 (spec sec.6);
    difference form likewise. Exactness, not just allclose."""
    K = 8
    rng = np.random.default_rng(11)
    P_tier_p = jnp.array(rng.uniform(0.5, 1.5, K))
    ratio = jnp.array(rng.normal(0, 0.05, (3, K)))
    delta = jnp.array(rng.normal(0, 0.05, (3, K)))
    assert jnp.array_equal(total_p1d_ratio(P_tier_p, jnp.zeros(3), ratio), P_tier_p)
    assert jnp.array_equal(total_p1d_difference(P_tier_p, jnp.zeros(3), delta), P_tier_p)


def test_jit_matches_eager_both_forms():
    """jax.jit must reproduce eager for both forms, unbatched and batched."""
    K, B = 8, 4
    rng = np.random.default_rng(12)
    P_tier_p = jnp.array(rng.uniform(0.5, 1.5, K))
    alpha = jnp.array(rng.uniform(0.0, 1.0, 3))
    delta = jnp.array(rng.normal(0, 0.05, (3, K)))
    ratio = jnp.array(rng.normal(0, 0.05, (3, K)))
    assert jnp.allclose(jax.jit(total_p1d_difference)(P_tier_p, alpha, delta),
                        total_p1d_difference(P_tier_p, alpha, delta), atol=1e-14)
    assert jnp.allclose(jax.jit(total_p1d_ratio)(P_tier_p, alpha, ratio),
                        total_p1d_ratio(P_tier_p, alpha, ratio), atol=1e-14)
    P_b = jnp.array(rng.uniform(0.5, 1.5, (B, K)))
    a_b = jnp.array(rng.uniform(0.0, 1.0, (B, 3)))
    d_b = jnp.array(rng.normal(0, 0.05, (B, 3, K)))
    assert jnp.allclose(jax.jit(total_p1d_difference)(P_b, a_b, d_b),
                        total_p1d_difference(P_b, a_b, d_b), atol=1e-14)


def test_covariance_inflates_dla_high_k():
    K = 8
    sigma = jnp.ones((4, K)) * 0.01
    w = jnp.array([0.7, 0.18, 0.07, 0.05])
    cov_noflag = assemble_covariance(jnp.ones(K) * 1e-3, sigma, w, dla_shot_flag=jnp.zeros(K, bool))
    flag = jnp.arange(K) >= 6
    cov_flag = assemble_covariance(jnp.ones(K) * 1e-3, sigma, w, dla_shot_flag=flag)
    assert jnp.all(jnp.diag(cov_flag)[6:] > jnp.diag(cov_noflag)[6:])
