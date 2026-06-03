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


def _cov_args(K=8, seed=0):
    rng = np.random.default_rng(seed)
    return dict(
        sigma_Pfilt=jnp.array(rng.uniform(0.005, 0.02, (4, K))),   # FRACTIONAL (δP/P)
        w_c=jnp.array([0.7, 0.18, 0.07, 0.05]),
        sigma_delta=jnp.array(rng.uniform(0.005, 0.02, (3, K))),   # FRACTIONAL (δΔ/Δ)
        alpha_hcd=jnp.array([1.1, 0.9, 1.2]),
        P_filt=jnp.array(rng.uniform(0.5, 1.5, (4, K))),           # absolute scale (4,K)
        delta_scale=jnp.array(rng.uniform(0.05, 0.5, (3, K))),     # |Δ_c| absolute scale (3,K)
    )


def test_covariance_inflates_dla_high_k():
    K = 8
    a = _cov_args(K)
    cosmic = jnp.ones(K) * 1e-3
    cov_noflag = assemble_covariance(cosmic, a["sigma_Pfilt"], a["w_c"],
                                     a["sigma_delta"], a["alpha_hcd"],
                                     a["P_filt"], a["delta_scale"],
                                     dla_shot_flag=jnp.zeros(K, bool))
    flag = jnp.arange(K) >= 6
    cov_flag = assemble_covariance(cosmic, a["sigma_Pfilt"], a["w_c"],
                                   a["sigma_delta"], a["alpha_hcd"],
                                   a["P_filt"], a["delta_scale"],
                                   dla_shot_flag=flag)
    assert jnp.all(jnp.diag(cov_flag)[6:] > jnp.diag(cov_noflag)[6:])
    # off-flag bins unchanged
    assert jnp.allclose(jnp.diag(cov_flag)[:6], jnp.diag(cov_noflag)[:6])


def test_covariance_full_offdiagonal_preserved():
    """A full (K,K) cosmic_cov is preserved (off-diagonals nonzero in output) and
    only the diagonal gets the emu variance added; a (K,) vector -> diag form."""
    K = 8
    a = _cov_args(K, seed=1)
    rng = np.random.default_rng(7)
    A = rng.standard_normal((K, K))
    cosmic_full = jnp.array(A @ A.T) + 1e-3 * jnp.eye(K)   # SPD with off-diagonals
    flag = jnp.zeros(K, bool)
    cov = assemble_covariance(cosmic_full, a["sigma_Pfilt"], a["w_c"],
                              a["sigma_delta"], a["alpha_hcd"],
                              a["P_filt"], a["delta_scale"], dla_shot_flag=flag)
    assert cov.shape == (K, K)
    # off-diagonals preserved exactly (emu var is diagonal-only)
    off = ~jnp.eye(K, dtype=bool)
    assert jnp.allclose(cov[off], cosmic_full[off])
    assert jnp.any(cov[off] != 0.0)
    # diagonal = cosmic diag + emu var (σ_* FRACTIONAL × absolute scale²)
    emu_var = (jnp.einsum("c,ck,ck->k", a["w_c"]**2, a["sigma_Pfilt"]**2, a["P_filt"]**2)
               + jnp.einsum("c,ck,ck->k", a["alpha_hcd"]**2, a["sigma_delta"]**2,
                            a["delta_scale"]**2))
    assert jnp.allclose(jnp.diag(cov), jnp.diag(cosmic_full) + emu_var)

    # (K,) vector input -> pure diagonal cov
    cov_vec = assemble_covariance(jnp.diag(cosmic_full), a["sigma_Pfilt"], a["w_c"],
                                  a["sigma_delta"], a["alpha_hcd"],
                                  a["P_filt"], a["delta_scale"], dla_shot_flag=flag)
    assert jnp.allclose(cov_vec, jnp.diag(jnp.diag(cov)))


def test_covariance_both_error_channels_contribute():
    """Both emu-error channels must contribute: zeroing sigma_Pfilt OR sigma_delta
    each strictly reduces the emu variance (the previous single-weights bug merged
    them under one amplitude)."""
    K = 8
    a = _cov_args(K, seed=2)
    cosmic = jnp.zeros(K)
    flag = jnp.zeros(K, bool)
    full = jnp.diag(assemble_covariance(cosmic, a["sigma_Pfilt"], a["w_c"],
                                        a["sigma_delta"], a["alpha_hcd"],
                                        a["P_filt"], a["delta_scale"], dla_shot_flag=flag))
    no_pf = jnp.diag(assemble_covariance(cosmic, jnp.zeros((4, K)), a["w_c"],
                                         a["sigma_delta"], a["alpha_hcd"],
                                         a["P_filt"], a["delta_scale"], dla_shot_flag=flag))
    no_dl = jnp.diag(assemble_covariance(cosmic, a["sigma_Pfilt"], a["w_c"],
                                         jnp.zeros((3, K)), a["alpha_hcd"],
                                         a["P_filt"], a["delta_scale"], dla_shot_flag=flag))
    assert jnp.all(no_pf < full) and jnp.all(no_dl < full)
    # the two channels sum to the full emu variance (quadrature/additive in var)
    assert jnp.allclose(no_pf + no_dl, full)


def test_covariance_jit_and_differentiable():
    """assemble_covariance stays jit-clean and differentiable through both
    error amplitudes (w_c and alpha_hcd)."""
    K = 8
    a = _cov_args(K, seed=3)
    cosmic = jnp.ones(K) * 1e-3
    flag = jnp.arange(K) >= 6
    eager = assemble_covariance(cosmic, a["sigma_Pfilt"], a["w_c"],
                                a["sigma_delta"], a["alpha_hcd"],
                                a["P_filt"], a["delta_scale"], dla_shot_flag=flag)
    jitted = jax.jit(assemble_covariance, static_argnums=())(
        cosmic, a["sigma_Pfilt"], a["w_c"], a["sigma_delta"], a["alpha_hcd"],
        a["P_filt"], a["delta_scale"], flag)
    assert jnp.allclose(eager, jitted)

    def trace_cov(w_c, alpha):
        return jnp.trace(assemble_covariance(cosmic, a["sigma_Pfilt"], w_c,
                                             a["sigma_delta"], alpha,
                                             a["P_filt"], a["delta_scale"], dla_shot_flag=flag))
    gw, ga = jax.grad(trace_cov, argnums=(0, 1))(a["w_c"], a["alpha_hcd"])
    assert jnp.all(jnp.isfinite(gw)) and jnp.all(jnp.isfinite(ga))
    assert jnp.any(gw != 0.0) and jnp.any(ga != 0.0)


def test_covariance_fractional_units_dimensional():
    """UNITS CONTRACT: σ_Pfilt / σ_delta are FRACTIONAL; P_filt / delta_scale
    supply the absolute scale. Dimensional consequences:

      (a) scaling P_filt by 10× scales the P_filt-channel emu variance by 100×
          (variance ∝ P_filt²), with the delta channel held fixed;
      (b) scaling delta_scale by 10× scales the delta-channel emu variance by 100×;
      (c) a fractional σ of 0.0 gives ZERO emu contribution from that channel
          regardless of the absolute scale (the pre-fix absolute-add bug would
          have leaked a nonzero, scale-independent term);
      (d) the assembled covariance is PSD.
    """
    K = 8
    a = _cov_args(K, seed=5)
    cosmic = jnp.zeros(K)               # isolate the emu variance
    flag = jnp.zeros(K, bool)

    def emu(P_filt=None, delta_scale=None, s_pf=None, s_dl=None):
        return jnp.diag(assemble_covariance(
            cosmic, a["sigma_Pfilt"] if s_pf is None else s_pf, a["w_c"],
            a["sigma_delta"] if s_dl is None else s_dl, a["alpha_hcd"],
            a["P_filt"] if P_filt is None else P_filt,
            a["delta_scale"] if delta_scale is None else delta_scale,
            dla_shot_flag=flag))

    # (a) P_filt channel scales as P_filt²; isolate it by zeroing the delta σ.
    base_pf = emu(s_dl=jnp.zeros((3, K)))
    scal_pf = emu(P_filt=a["P_filt"] * 10.0, s_dl=jnp.zeros((3, K)))
    assert jnp.allclose(scal_pf, base_pf * 100.0)

    # (b) delta channel scales as delta_scale²; isolate it by zeroing the P_filt σ.
    base_dl = emu(s_pf=jnp.zeros((4, K)))
    scal_dl = emu(delta_scale=a["delta_scale"] * 10.0, s_pf=jnp.zeros((4, K)))
    assert jnp.allclose(scal_dl, base_dl * 100.0)

    # (c) zero FRACTIONAL σ -> exactly zero emu contribution for that channel,
    #     independent of the (large) absolute scale.
    assert jnp.array_equal(emu(s_pf=jnp.zeros((4, K))), base_dl)   # only delta survives
    assert jnp.array_equal(emu(s_dl=jnp.zeros((3, K))), base_pf)   # only P_filt survives
    assert jnp.all(emu(s_pf=jnp.zeros((4, K)), s_dl=jnp.zeros((3, K))) == 0.0)

    # (d) PSD: add a cosmic SPD floor and check eigenvalues >= 0.
    rng = np.random.default_rng(21)
    A = rng.standard_normal((K, K))
    cosmic_spd = jnp.array(A @ A.T) + 1e-3 * jnp.eye(K)
    cov = assemble_covariance(cosmic_spd, a["sigma_Pfilt"], a["w_c"],
                              a["sigma_delta"], a["alpha_hcd"],
                              a["P_filt"], a["delta_scale"], dla_shot_flag=flag)
    evals = jnp.linalg.eigvalsh(0.5 * (cov + cov.T))
    assert jnp.all(evals >= -1e-9)


def test_covariance_nan_safe_datarange_restriction():
    """DATA-RANGE CONTRACT: the emulator error vector is built data-range-restricted
    (z∈[2.2,4.6], k≥1e-3) upstream, so out-of-range (class,k) cells arrive as NaN.
    assemble_covariance must be NaN-SAFE: a NaN σ contributes EXACTLY ZERO emu
    variance for that cell (the data doesn't constrain it) — identical to passing a
    0 σ there — while the cosmic covariance for those modes is untouched."""
    K = 8
    a = _cov_args(K, seed=11)
    cosmic = jnp.ones(K) * 1e-3
    flag = jnp.zeros(K, bool)

    # mark the lowest 2 k-bins (k<1e-3 proxy) out-of-range -> NaN in BOTH channels
    sig_pf_nan = a["sigma_Pfilt"].at[:, :2].set(jnp.nan)
    sig_dl_nan = a["sigma_delta"].at[:, :2].set(jnp.nan)
    # the equivalent "zeroed" error vector (the data-range cells contribute nothing)
    sig_pf_zero = a["sigma_Pfilt"].at[:, :2].set(0.0)
    sig_dl_zero = a["sigma_delta"].at[:, :2].set(0.0)

    cov_nan = assemble_covariance(cosmic, sig_pf_nan, a["w_c"], sig_dl_nan,
                                  a["alpha_hcd"], a["P_filt"], a["delta_scale"],
                                  dla_shot_flag=flag)
    cov_zero = assemble_covariance(cosmic, sig_pf_zero, a["w_c"], sig_dl_zero,
                                   a["alpha_hcd"], a["P_filt"], a["delta_scale"],
                                   dla_shot_flag=flag)
    assert jnp.all(jnp.isfinite(cov_nan))                       # no NaN leak
    assert jnp.allclose(cov_nan, cov_zero, atol=1e-14)          # NaN == zeroed cell
    # the out-of-range diagonal carries ONLY the cosmic variance (no emu penalty)
    dnan = jnp.diag(cov_nan)
    assert jnp.allclose(dnan[:2], cosmic[:2], atol=1e-14)
    # in-range bins still carry the emu variance (strictly above the cosmic floor)
    assert jnp.all(dnan[2:] > cosmic[2:])

    # differentiable through the NaN-safe path (grad wrt w_c finite)
    def trace_of(w_c):
        return jnp.trace(assemble_covariance(
            cosmic, sig_pf_nan, w_c, sig_dl_nan, a["alpha_hcd"],
            a["P_filt"], a["delta_scale"], dla_shot_flag=flag))
    g = jax.grad(trace_of)(a["w_c"])
    assert jnp.all(jnp.isfinite(g))
