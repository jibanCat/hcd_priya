import jax, jax.numpy as jnp, equinox as eqx
from hcd_analysis.emulator.model import (
    Encoder, HeadA, HeadB, BaselineHead, Emulator, structural_tier_p)
from hcd_analysis.emulator.model import masked_mse
from hcd_analysis.emulator.model import joint_loss, p_resid_loss
from hcd_analysis.emulator.model import svd_basis_init


def test_encoder_headA_shapes_and_tau0_independence():
    key = jax.random.PRNGKey(0)
    enc = Encoder(in_dim=10, key=key)
    ha = HeadA(latent=64, n_k_cddf=30, key=key)
    x = jnp.zeros(10)
    lat = enc(x)
    assert lat.shape == (64,)
    out = ha(lat)
    assert out["f_nhi"].shape == (30,)
    assert out["dndx"].shape == (3,)
    # x64 active via package import: latent must be float64
    assert lat.dtype == jnp.float64


def test_encoder_layer_keys_distinct():
    # PRNG key-reuse trap: each Linear must get its own split subkey, else
    # layers share correlated init weights (silent bug). Rebuilds must be
    # reproducible and key-sensitive.
    key = jax.random.PRNGKey(0)
    enc_a = Encoder(in_dim=10, key=key)
    enc_b = Encoder(in_dim=10, key=key)
    enc_c = Encoder(in_dim=10, key=jax.random.PRNGKey(1))
    # same key -> identical weights (deterministic, no hidden global RNG)
    for la, lb in zip(enc_a.layers, enc_b.layers):
        assert jnp.array_equal(la.weight, lb.weight)
    # different top-level key -> different weights (key actually consumed)
    assert not jnp.array_equal(enc_a.layers[0].weight, enc_c.layers[0].weight)


def test_jit_vmap_and_finite_grads():
    key = jax.random.PRNGKey(0)
    enc = Encoder(in_dim=10, key=key)
    ha = HeadA(latent=64, n_k_cddf=30, key=key)
    x = jnp.ones(10)

    # filter_jit must run and agree with eager
    lat = enc(x)
    assert jnp.allclose(eqx.filter_jit(enc)(x), lat)

    # vmap over a leading batch dim, both modules
    batch = jnp.ones((5, 10))
    lat_b = jax.vmap(enc)(batch)
    assert lat_b.shape == (5, 64)
    out_b = jax.vmap(ha)(lat_b)
    assert out_b["f_nhi"].shape == (5, 30)
    assert out_b["dndx"].shape == (5, 3)

    # gradients must be finite (no NaN through gelu / Linear stack)
    g_enc = jax.grad(lambda v: enc(v).sum())(x)
    assert jnp.all(jnp.isfinite(g_enc))
    g_ha = jax.grad(lambda l: ha(l)["f_nhi"].sum() + ha(l)["dndx"].sum())(lat)
    assert jnp.all(jnp.isfinite(g_ha))


def test_headB_outputs_and_structural_tier_p():
    key = jax.random.PRNGKey(1)
    hb = HeadB(latent=64, n_k=8, key=key)
    lat = jnp.zeros(64); tau0 = jnp.array(0.3)
    out = hb(lat, tau0)
    # REDESIGN: HeadB's P_filt output is now the residual r̂ (key P_filt_resid).
    assert out["P_filt_resid"].shape == (4, 8)
    assert out["delta"].shape == (3, 8)
    w = jnp.array([0.7, 0.18, 0.07, 0.05]); P_filt_lin = jnp.ones((4, 8))
    tp = structural_tier_p(w, P_filt_lin)
    assert tp.shape == (8,)
    assert jnp.allclose(tp, w.sum())


def test_emulator_endtoend_runs():
    key = jax.random.PRNGKey(2)
    m = Emulator(in_dim=10, n_k=8, key=key)
    pred = m(jnp.zeros(10), tau0=jnp.array(0.3))
    # REDESIGN: P_filt is split into P_filt_base (θ-blind) + P_filt_resid (residual).
    assert set(pred) >= {"f_nhi", "dndx", "P_filt_base", "P_filt_resid", "delta"}


def test_emulator_vmap_over_tau0_and_structural_grad():
    # JAX foot-gun guards for Head B / structural total (all model tests above
    # are UNBATCHED). Two real risks the unbatched tests miss:
    #   (a) jnp.atleast_1d(tau0)+concatenate must batch a per-row scalar tau0
    #       under jax.vmap. A scalar reshaped inside a vmapped fn (e.g. a naive
    #       tau0.reshape(1)) silently breaks here; atleast_1d is the safe form.
    #   (b) grad must flow finitely through structural_tier_p w.r.t. BOTH inputs.
    key = jax.random.PRNGKey(5)
    m = Emulator(in_dim=10, n_k=6, key=key)

    # (a) vmap over BOTH x and a batched (per-row scalar) tau0
    X = jnp.ones((5, 10))
    T = jnp.linspace(0.1, 0.5, 5)  # shape (5,), one scalar tau0 per row
    pred = jax.vmap(lambda x, t: m(x, t))(X, T)
    assert pred["P_filt_resid"].shape == (5, 4, 6)
    assert pred["P_filt_base"].shape == (5, 4, 6)
    assert pred["delta"].shape == (5, 3, 6)
    assert pred["f_nhi"].shape == (5, 30)
    assert pred["P_filt_resid"].dtype == jnp.float64
    # batched matches per-row eager (no silent shape/broadcast corruption)
    eager0 = m(X[0], T[0])
    assert jnp.allclose(pred["P_filt_resid"][0], eager0["P_filt_resid"])

    # (b) structural_tier_p: batched einsum == manual loop; grad finite on both args
    w = jax.random.uniform(jax.random.PRNGKey(6), (5, 4))
    P = jax.random.uniform(jax.random.PRNGKey(7), (5, 4, 6)) + 0.1
    tp = structural_tier_p(w, P)
    assert tp.shape == (5, 6)
    ref = jnp.stack([jnp.einsum("c,ck->k", w[i], P[i]) for i in range(5)])
    assert jnp.allclose(tp, ref)
    gw = jax.grad(lambda v: structural_tier_p(v, P[0]).sum())(w[0])
    gp = jax.grad(lambda v: structural_tier_p(w[0], v).sum())(P[0])
    assert jnp.all(jnp.isfinite(gw)) and jnp.all(jnp.isfinite(gp))
    # all-ones P_filt -> Tier-P total == sum(w) (the structural identity)
    assert jnp.allclose(structural_tier_p(w[0], jnp.ones((4, 6))), w[0].sum())


def test_masked_mse_nan_safe_gradients():
    pred = jnp.array([1.0, 2.0, 3.0, 4.0])
    targ = jnp.array([1.0, jnp.nan, 3.0, jnp.nan])
    mask = jnp.isfinite(targ)
    val, grad = jax.value_and_grad(lambda p: masked_mse(p, targ, mask))(pred)
    assert jnp.isfinite(val)
    assert jnp.all(jnp.isfinite(grad))
    assert grad[1] == 0.0 and grad[3] == 0.0


def test_masked_mse_inf_target_at_masked_position():
    # Adversarial: a masked-out target may be +/-inf (not just NaN). The masked
    # branch must zero it BEFORE it can poison the where-gradient (0*inf=NaN trap).
    pred = jnp.array([1.0, 2.0, 3.0, 4.0])
    targ = jnp.array([1.0, jnp.inf, 3.0, -jnp.inf])
    mask = jnp.array([True, False, True, False])
    val, grad = jax.value_and_grad(lambda p: masked_mse(p, targ, mask))(pred)
    assert jnp.isfinite(val)
    assert jnp.all(jnp.isfinite(grad))
    assert grad[1] == 0.0 and grad[3] == 0.0  # masked inf positions: zero gradient


def test_masked_mse_fully_masked_no_div_zero():
    # Adversarial: mask all-False -> denom floored to 1 (no 0/0). Forward 0, grad all-0.
    pred = jnp.array([1.0, 2.0, 3.0, 4.0])
    targ = jnp.full(4, jnp.nan)
    mask = jnp.zeros(4, bool)
    val, grad = jax.value_and_grad(lambda p: masked_mse(p, targ, mask))(pred)
    assert val == 0.0
    assert jnp.all(jnp.isfinite(grad)) and jnp.all(grad == 0.0)


def test_masked_mse_zero_weight_empty_class():
    # Adversarial: a per-class 1/n_c weight can have a 0 entry (empty class). A zero
    # weight must contribute zero gradient, not NaN, even at a masked NaN target.
    pred = jnp.array([1.0, 2.0, 3.0, 4.0])
    targ = jnp.array([0.5, jnp.nan, 1.0, 2.0])
    mask = jnp.isfinite(targ)
    weight = jnp.array([0.5, 0.0, 1.0, 2.0])  # idx1: empty class (zero weight) + NaN target
    val, grad = jax.value_and_grad(lambda p: masked_mse(p, targ, mask, weight))(pred)
    assert jnp.isfinite(val)
    assert jnp.all(jnp.isfinite(grad))
    assert grad[1] == 0.0  # zero weight -> zero gradient


def test_masked_mse_batched_broadcast_jit_and_double_grad():
    # Realistic joint-loss shape: pred/target (B,4,K), mask broadcast to full shape
    # (as the joint loss does via `m3 & ones_like(target, bool)`), NaN above Nyquist.
    import numpy as np
    B, C, K = 3, 4, 6
    rng = np.random.default_rng(0)
    pred = jnp.array(rng.standard_normal((B, C, K)))
    targ = rng.standard_normal((B, C, K))
    m2d = np.ones((B, K), bool)
    m2d[0, 4:] = False
    m2d[2, 5:] = False
    full = np.broadcast_to(m2d[:, None, :], (B, C, K))
    targ[~full] = np.nan
    targ = jnp.array(targ)
    mask = jnp.array(full)  # materialised full mask, as the joint loss passes
    eager = masked_mse(pred, targ, mask)
    jitted = eqx.filter_jit(masked_mse)(pred, targ, mask)
    assert jnp.isfinite(eager) and jnp.allclose(eager, jitted)
    grad = jax.grad(lambda p: masked_mse(p, targ, mask))(pred)
    assert jnp.all(jnp.isfinite(grad))
    assert jnp.all(grad[jnp.array(~full)] == 0.0)  # zero grad at masked positions
    # denom counts exactly the unmasked elements (mean over intended elements)
    assert float(jnp.sum(mask.astype(jnp.float64))) == int(full.sum())
    # float64 under x64
    assert eager.dtype == jnp.float64 and grad.dtype == jnp.float64
    # second-order: hessian-diag finite on the NaN batch
    hess = jax.jacfwd(jax.grad(lambda p: masked_mse(p, targ, mask)))(pred)
    assert jnp.all(jnp.isfinite(hess))


def test_joint_loss_scalar_finite_and_grads_finite():
    key = jax.random.PRNGKey(3)
    m = Emulator(in_dim=10, n_k=8, key=key)
    _tpf = jnp.where(jnp.arange(8) < 6, 1.0, jnp.nan)[None,None,:]*jnp.ones((2,4,8))
    batch = {
        "x": jnp.zeros((2,10)), "tau0": jnp.array([0.3, 0.5]),
        "t_f_nhi": jnp.zeros((2,30)), "t_dndx": jnp.zeros((2,3)),
        "t_p_base": _tpf,
        "t_p_resid": _tpf,
        "t_delta": jnp.zeros((2,3,8)),
        "t_f_nhi_mask": jnp.ones((2,30), bool),
        "t_dndx_mask": jnp.ones((2,3), bool),
        "mask": (jnp.arange(8) < 6)[None,:]*jnp.ones((2,8), bool),
        "inv_nc": jnp.array([[1.,1/10,1/7,0.],[1.,1/10,1/7,1/3]]),
        "inv_nalpha": jnp.array([0.25, 0.25]),
        "mean_F_clean": jnp.array([0.7, 0.6]),
    }
    val, grad = jax.value_and_grad(lambda mm: joint_loss(mm, batch))(m)
    assert jnp.isfinite(val)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grad, eqx.is_array))
    assert all(jnp.all(jnp.isfinite(g)) for g in leaves)


def test_headB_n_k_static_field_serialise_roundtrip():
    # HeadB.n_k is eqx.field(static=True): (a) it must NOT be a differentiable leaf
    # so PLAIN jax.value_and_grad(joint_loss)(model,...) works (a dynamic int leaf
    # raises "grad requires real- or complex-valued inputs, got int64"); (b) it must
    # survive tree_serialise/deserialise, which reconstructs static fields from the
    # SKELETON (the bytes hold only array leaves) — the n_k must round-trip and the
    # restored model must reproduce predictions bit-for-bit.
    import io
    key = jax.random.PRNGKey(31)
    m = Emulator(in_dim=10, n_k=8, key=key)

    # (a) n_k is a STATIC field, not an array leaf of the pytree
    arr_leaves = jax.tree_util.tree_leaves(eqx.filter(m, eqx.is_array))
    assert all(jnp.issubdtype(l.dtype, jnp.floating) for l in arr_leaves)
    assert m.head_b.n_k == 8

    # (b) serialise -> deserialise into a fresh skeleton; n_k restored, preds identical
    buf = io.BytesIO()
    eqx.tree_serialise_leaves(buf, m)
    buf.seek(0)
    skeleton = Emulator(in_dim=10, n_k=8, key=jax.random.PRNGKey(999))
    m2 = eqx.tree_deserialise_leaves(buf, skeleton)
    assert m2.head_b.n_k == 8
    x = jnp.ones((3, 10)); t = jnp.linspace(0.1, 0.5, 3)
    p1 = jax.vmap(m)(x, t); p2 = jax.vmap(m2)(x, t)
    for kk in ("f_nhi", "dndx", "P_filt_base", "P_filt_resid", "delta"):
        assert jnp.array_equal(p1[kk], p2[kk])


def test_emulator_vmap_equals_python_loop_all_heads():
    # vmap(model) over a batch must equal a python loop row-by-row for EVERY head
    # (no leading-axis bug in Head B's reshape(7, n_k) / atleast_1d(tau0) path,
    # nor in the BaselineHead's z=x[...,9] indexing).
    key = jax.random.PRNGKey(17)
    m = Emulator(in_dim=10, n_k=8, key=key)
    x = jax.random.normal(jax.random.PRNGKey(18), (4, 10))
    t = jnp.linspace(0.05, 0.6, 4)
    pv = jax.vmap(m)(x, t)
    for kk in ("f_nhi", "dndx", "P_filt_base", "P_filt_resid", "delta"):
        loop = jnp.stack([m(x[i], t[i])[kk] for i in range(4)])
        assert jnp.allclose(pv[kk], loop, atol=0, rtol=0) or jnp.allclose(pv[kk], loop)


def test_baseline_head_is_theta_blind():
    # REDESIGN: the BaselineHead's P_filt_base output must be EXACTLY blind to the 9
    # cosmology/IGM params (it takes only z, τ₀ by construction). The Jacobian of
    # P_filt_base wrt the 9 params is exactly zero. (z = x[9], so we differentiate
    # only the first 9 entries of x.)
    key = jax.random.PRNGKey(8)
    m = Emulator(in_dim=10, n_k=8, key=key)
    z_unit = jnp.array(0.42)
    tau0 = jnp.array(0.4)

    def base_of_theta(theta9):
        x = jnp.concatenate([theta9, jnp.atleast_1d(z_unit)])
        return m(x, tau0)["P_filt_base"]

    theta0 = jnp.linspace(0.1, 0.9, 9)
    J = jax.jacfwd(base_of_theta)(theta0)            # (4, n_k, 9)
    assert jnp.allclose(J, 0.0, atol=0.0)            # exactly zero: θ never enters the baseline
    # but the residual head IS θ-dependent (sanity: not also accidentally blind)
    def resid_of_theta(theta9):
        x = jnp.concatenate([theta9, jnp.atleast_1d(z_unit)])
        return m(x, tau0)["P_filt_resid"]
    Jr = jax.jacfwd(resid_of_theta)(theta0)
    assert jnp.any(jnp.abs(Jr) > 0.0)                # θ flows into the residual
    # and the baseline DOES depend on z (it is the (z,τ₀)-conditional mean)
    Jz = jax.jacfwd(lambda zz: m(
        jnp.concatenate([theta0, jnp.atleast_1d(zz)]), tau0)["P_filt_base"])(z_unit)
    assert jnp.any(jnp.abs(Jz) > 0.0)


def test_baseline_head_standalone_is_theta_blind_and_shaped():
    # BaselineHead in isolation: input is ONLY (z, τ₀); output (4, n_k); low-rank ok.
    key = jax.random.PRNGKey(80)
    for n_basis in (None, 5):
        bh = BaselineHead(n_k=8, n_basis=n_basis, key=key)
        out = bh(jnp.array(0.3), jnp.array(0.5))
        assert out.shape == (4, 8)
        # vmap over a batch of (z, τ₀)
        zb = jnp.linspace(0.1, 0.9, 6); tb = jnp.linspace(0.2, 0.6, 6)
        ob = jax.vmap(bh)(zb, tb)
        assert ob.shape == (6, 4, 8)
        assert ob.dtype == jnp.float64


def test_p_resid_loss_is_residual_term_only():
    # p_resid_loss == the σ_cosmo residual MSE term inside joint_loss (the early-stop
    # metric). Build a batch where t_p_base != t_p_resid; p_resid_loss must depend ONLY
    # on the residual head + t_p_resid.
    key = jax.random.PRNGKey(91)
    n_k = 8
    m = Emulator(in_dim=10, n_k=n_k, key=key)
    batch = _joint_batch(n_k)
    base_loss = float(p_resid_loss(m, batch))
    # corrupt t_p_base only -> p_resid_loss unchanged
    b2 = dict(batch); b2["t_p_base"] = batch["t_p_base"] + 7.0
    assert float(p_resid_loss(m, b2)) == base_loss
    # corrupt t_p_resid -> p_resid_loss DOES change
    b3 = dict(batch); b3["t_p_resid"] = jnp.zeros_like(batch["t_p_resid"])
    assert float(p_resid_loss(m, b3)) != base_loss


def test_headA_outputs_are_tau0_invariant_by_gradient():
    # Task 12: Head A (f_nhi, dN/dX) must be EXACTLY tau0-invariant. It is structural
    # (Head A consumes only the latent, never tau0), so the Jacobian wrt tau0 is exactly
    # zero -- a gradient test, not a value test (spec sec.8).
    key = jax.random.PRNGKey(4)
    m = Emulator(in_dim=10, n_k=8, key=key)
    x = jnp.ones(10)
    def fa(tau0):
        p = m(x, tau0)
        return jnp.concatenate([p["f_nhi"], p["dndx"]])
    J = jax.jacfwd(fa)(jnp.array(0.4))
    assert jnp.allclose(J, 0.0, atol=0.0)  # exactly zero: Head A does not consume tau0


# ---------------------------------------------------------------------------
# Task 2 (Phase-2b): learned low-rank P_filt output basis (toggle n_basis).
# Delta_c stays DENSE; SVD warm-start; dense default unchanged.
# ---------------------------------------------------------------------------

def _lowrank_batch(n_k):
    """joint_loss batch for n_basis tests (same shapes as the dense joint-loss test)."""
    _tpf = (jnp.where(jnp.arange(n_k) < n_k - 2, 1.0, jnp.nan)[None, None, :]
            * jnp.ones((2, 4, n_k)))
    return {
        "x": jnp.zeros((2, 10)), "tau0": jnp.array([0.3, 0.5]),
        "t_f_nhi": jnp.zeros((2, 30)), "t_dndx": jnp.zeros((2, 3)),
        "t_p_base": _tpf,
        "t_p_resid": _tpf,
        "t_delta": jnp.zeros((2, 3, n_k)),
        "t_f_nhi_mask": jnp.ones((2, 30), bool),
        "t_dndx_mask": jnp.ones((2, 3), bool),
        "mask": (jnp.arange(n_k) < n_k - 2)[None, :] * jnp.ones((2, n_k), bool),
        "inv_nc": jnp.array([[1., 1 / 10, 1 / 7, 0.], [1., 1 / 10, 1 / 7, 1 / 3]]),
        "inv_nalpha": jnp.array([0.25, 0.25]),
        "mean_F_clean": jnp.array([0.7, 0.6]),
    }


def test_headB_dense_unchanged():
    # n_basis=None must be byte-for-byte the pre-change dense head: P_filt (4,n_k),
    # delta (3,n_k), and identical outputs for a fixed key (dense path untouched).
    key = jax.random.PRNGKey(1)
    n_k = 8
    hb = HeadB(latent=64, n_k=n_k, key=key)           # default n_basis=None
    hb_explicit = HeadB(latent=64, n_k=n_k, n_basis=None, key=key)
    assert hb.n_basis is None and hb.p_filt_basis is None
    lat = jnp.zeros(64); tau0 = jnp.array(0.3)
    out = hb(lat, tau0)
    # REDESIGN: P_filt output key is now P_filt_resid (the residual), same shape.
    assert out["P_filt_resid"].shape == (4, n_k)
    assert out["delta"].shape == (3, n_k)
    # explicit None == default None (same key -> identical weights and outputs)
    out2 = hb_explicit(lat, tau0)
    assert jnp.array_equal(out["P_filt_resid"], out2["P_filt_resid"])
    assert jnp.array_equal(out["delta"], out2["delta"])
    # regression vs the literal pre-change computation: single dense Linear,
    # reshape(7,n_k), split [:4]/[4:] (the arithmetic is byte-for-byte unchanged).
    h = jax.nn.gelu(hb.trunk(jnp.concatenate([lat, jnp.atleast_1d(tau0)])))
    y = hb.out(h).reshape(7, n_k)
    assert jnp.array_equal(out["P_filt_resid"], y[:4])
    assert jnp.array_equal(out["delta"], y[4:])


def test_headB_lowrank_shapes():
    # n_basis=int -> P_filt (4,n_k) decoded through the trainable basis; delta (3,n_k)
    # dense; the output-layer param count is MUCH smaller than the dense head.
    key = jax.random.PRNGKey(1)
    n_k, n_basis = 172, 12
    hb_dense = HeadB(latent=64, n_k=n_k, key=key)
    hb_lr = HeadB(latent=64, n_k=n_k, n_basis=n_basis, key=key)
    lat = jnp.zeros(64); tau0 = jnp.array(0.3)
    out = hb_lr(lat, tau0)
    assert out["P_filt_resid"].shape == (4, n_k)
    assert out["delta"].shape == (3, n_k)
    assert hb_lr.p_filt_basis.shape == (n_basis, n_k)
    # output-layer weight+bias param counts
    def out_params(hb):
        return hb.out.weight.size + hb.out.bias.size
    dense_n = out_params(hb_dense)        # 256*(7*n_k) + 7*n_k
    lr_out_n = out_params(hb_lr)          # 256*(4*nb+3*n_k) + (4*nb+3*n_k)
    lr_total = lr_out_n + hb_lr.p_filt_basis.size
    assert lr_out_n < dense_n
    # even counting the trainable basis, the low-rank head is much smaller
    assert lr_total < 0.7 * dense_n


def test_svd_warmstart_beats_random():
    import numpy as np
    rng = np.random.default_rng(0)
    n_rows, n_k, r = 200, 30, 5
    # synthetic exactly rank-r matrix: (n_rows, r) @ (r, n_k)
    A = rng.standard_normal((n_rows, r))
    B = rng.standard_normal((r, n_k))
    M = jnp.asarray(A @ B)
    n_basis = 8  # >= r
    basis = svd_basis_init(M, n_basis)
    assert basis.shape == (n_basis, n_k)
    # reconstruction: project rows onto the basis and back. With n_basis >= rank,
    # error ~ 0 (the top-r right sing. vectors span the row space exactly).
    coeffs = M @ basis.T                 # (n_rows, n_basis)
    recon = coeffs @ basis               # (n_rows, n_k)
    err = jnp.linalg.norm(M - recon) / jnp.linalg.norm(M)
    assert err < 1e-6
    # rank-collapse guard: every singular value of the returned basis is > 0
    sv = jnp.linalg.svd(basis, compute_uv=False)
    assert jnp.all(sv > 1e-8)
    # warm-start strictly beats a random basis at the same n_basis
    rand = jax.nn.initializers.orthogonal()(jax.random.PRNGKey(0), (n_basis, n_k))
    recon_rand = (M @ rand.T) @ rand
    err_rand = jnp.linalg.norm(M - recon_rand) / jnp.linalg.norm(M)
    assert err < err_rand


def test_lowrank_grads_finite_and_jit():
    key = jax.random.PRNGKey(3)
    n_k = 8
    m = Emulator(in_dim=10, n_k=n_k, n_basis=12, key=key)
    # forward + vmap over a batch
    x = jnp.ones((2, 10)); t = jnp.array([0.3, 0.5])
    pv = jax.vmap(m)(x, t)
    assert pv["P_filt_resid"].shape == (2, 4, n_k)
    assert pv["P_filt_base"].shape == (2, 4, n_k)
    assert pv["delta"].shape == (2, 3, n_k)
    assert pv["P_filt_resid"].dtype == jnp.float64  # x64 active
    # eqx.filter_value_and_grad(joint_loss) finite (incl. the trainable basis leaf)
    batch = _lowrank_batch(n_k)
    val, grad = eqx.filter_value_and_grad(lambda mm: joint_loss(mm, batch))(m)
    assert jnp.isfinite(val)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grad, eqx.is_array))
    assert all(jnp.all(jnp.isfinite(g)) for g in leaves)
    # the basis got a finite (and non-trivial) gradient
    assert grad.head_b.p_filt_basis is not None
    assert jnp.all(jnp.isfinite(grad.head_b.p_filt_basis))
    # eqx.filter_jit matches eager
    eager = joint_loss(m, batch)
    jitted = eqx.filter_jit(joint_loss)(m, batch)
    assert jnp.allclose(eager, jitted)
    # vmap matches a python loop row-by-row (no leading-axis bug in the basis matmul)
    for kk in ("P_filt_resid", "P_filt_base", "delta"):
        loop = jnp.stack([m(x[i], t[i])[kk] for i in range(2)])
        assert jnp.allclose(pv[kk], loop)


def test_delta_still_dense_with_basis():
    # With n_basis set, Delta_c must remain the dense (3,n_k) head -- NOT routed
    # through the P_filt basis. Verify the delta output is independent of the basis:
    # perturbing p_filt_basis changes P_filt but leaves delta bit-identical.
    key = jax.random.PRNGKey(7)
    n_k, n_basis = 8, 4
    m = Emulator(in_dim=10, n_k=n_k, n_basis=n_basis, key=key)
    x = jnp.ones(10); t = jnp.array(0.4)
    out0 = m(x, t)
    assert out0["delta"].shape == (3, n_k)
    # mutate HeadB's basis (P_filt residual path only) and re-run
    new_basis = m.head_b.p_filt_basis + 5.0
    m2 = eqx.tree_at(lambda mm: mm.head_b.p_filt_basis, m, new_basis)
    out1 = m2(x, t)
    assert not jnp.allclose(out0["P_filt_resid"], out1["P_filt_resid"])  # residual DID change
    assert jnp.array_equal(out0["delta"], out1["delta"])     # delta unchanged (dense)
    # and the delta gradient does not flow into the basis
    g = jax.grad(lambda b: m2.__class__.__call__(
        eqx.tree_at(lambda mm: mm.head_b.p_filt_basis, m, b), x, t)["delta"].sum()
    )(m.head_b.p_filt_basis)
    assert jnp.all(g == 0.0)


def test_lowrank_serialise_roundtrip_and_structural_composition():
    # JAX-traps-log #8 generalised to the low-rank head: (a) tree_serialise/deserialise
    # must round-trip an n_basis=int model bit-for-bit -- the trainable p_filt_basis is
    # an array leaf carried in the bytes, while n_basis (static) is rebuilt from the
    # SKELETON; deserialising into a wrong-n_basis skeleton must RAISE (shape mismatch),
    # not silently mis-load. (b) the bottleneck's downstream contract: linear-space
    # P_filt produced by the low-rank head still satisfies structural_tier_p's einsum
    # (4,n_k) and the all-ones w_c -> sum-over-classes identity.
    import io
    key = jax.random.PRNGKey(123)
    n_k, n_basis = 8, 6
    m = Emulator(in_dim=10, n_k=n_k, n_basis=n_basis, key=key)

    # (a) every differentiable leaf (incl. the basis) is float; basis is a real leaf
    arr_leaves = jax.tree_util.tree_leaves(eqx.filter(m, eqx.is_array))
    assert all(jnp.issubdtype(l.dtype, jnp.floating) for l in arr_leaves)
    assert m.head_b.p_filt_basis.shape == (n_basis, n_k)

    buf = io.BytesIO()
    eqx.tree_serialise_leaves(buf, m)
    buf.seek(0)
    skeleton = Emulator(in_dim=10, n_k=n_k, n_basis=n_basis, key=jax.random.PRNGKey(999))
    m2 = eqx.tree_deserialise_leaves(buf, skeleton)
    assert m2.head_b.n_basis == n_basis
    assert jnp.array_equal(m.head_b.p_filt_basis, m2.head_b.p_filt_basis)
    x = jnp.ones((3, 10)); t = jnp.linspace(0.1, 0.5, 3)
    p1 = jax.vmap(m)(x, t); p2 = jax.vmap(m2)(x, t)
    for kk in ("f_nhi", "dndx", "P_filt_base", "P_filt_resid", "delta"):
        assert jnp.array_equal(p1[kk], p2[kk])

    # deserialising into a wrong-n_basis (and dense) skeleton must raise, never silently load
    for bad_skel in (Emulator(in_dim=10, n_k=n_k, n_basis=n_basis - 1, key=key),
                     Emulator(in_dim=10, n_k=n_k, n_basis=None, key=key)):
        buf.seek(0)
        try:
            eqx.tree_deserialise_leaves(buf, bad_skel)
            raised = False
        except Exception:
            raised = True
        assert raised

    # (b) structural_tier_p contract holds on any (3,4,n_k) LINEAR-shaped P_filt; the
    # low-rank residual head feeds the reconstruction downstream, so we pin the einsum
    # identity on exp(P_filt_resid) (a stand-in linear (4,n_k) array of the right shape).
    pf_lin = jnp.exp(p1["P_filt_resid"])                # (3,4,n_k) shape contract
    w = jnp.ones((3, 4))
    tp = structural_tier_p(w, pf_lin)
    assert tp.shape == (3, n_k)
    assert jnp.allclose(tp, pf_lin.sum(axis=1))         # all-ones w_c -> sum over classes
    # grad flows finitely back through HeadB's basis via the structural sum
    def struct_loss(b):
        mm = eqx.tree_at(lambda z: z.head_b.p_filt_basis, m, b)
        pf = jnp.exp(jax.vmap(mm)(x, t)["P_filt_resid"])
        return structural_tier_p(w, pf).sum()
    gb = jax.grad(struct_loss)(m.head_b.p_filt_basis)
    assert jnp.all(jnp.isfinite(gb)) and jnp.any(gb != 0.0)


# ---------------------------------------------------------------------------
# Phase-2b final review batch: A2 (structural-identity roundtrip), A3 (dead
# meanF removed), A4 (uniform/comparable term_w).
# ---------------------------------------------------------------------------

def test_structural_identity_through_transform_roundtrip(tmp_path):
    """A2 (REDESIGN): the baseline+residual transform/inverse COMPOSITION preserves
    the structural identity the spec leans on. Take a cache row's LINEAR P_filt (4,K)
    and its w_c; build the baseline/residual targets via make_batch, reconstruct
    LINEAR P_filt (reconstruct_P_filt, as untransform_prediction does), and confirm
    structural_tier_p(w_c, recovered linear P_filt) reproduces the cache's
    einsum(w_c, linear P_filt) to ~1e-10. (NOT a trained-model claim — this pins the
    transform composition on cache arrays.)"""
    import numpy as np
    from tests.emulator._fixture import write_synthetic_cache
    from hcd_analysis.emulator.data import (
        load_cache, fit_target_norm, make_batch, reconstruct_P_filt,
    )
    path = tmp_path / "obs.h5"
    write_synthetic_cache(path, n_sims=3, snaps_per_sim=2, n_alpha=4, n_k=8)
    d = load_cache(path)
    R = d["x"].shape[0]
    norm = fit_target_norm(d, np.arange(R))

    # pick a fully-finite (below-Nyquist) row so the whole (4,K) is comparable
    finite_rows = np.where(np.isfinite(d["P_filt"]).all(axis=(1, 2)))[0]
    assert finite_rows.size > 0
    row = int(finite_rows[0])
    P_lin = d["P_filt"][row]                          # (4,K) LINEAR
    w_c = d["w_c_cache"][row]                          # (4,)

    # forward to the baseline/residual target space, then reconstruct (untransform style)
    b = make_batch(d, np.array([row]), norm)
    P_lin_rt = reconstruct_P_filt(b["t_p_base"][0], b["t_p_resid"][0], norm["P_filt"])

    cache_tier_p = np.einsum("c,ck->k", w_c, P_lin)    # cache structural total
    rt_tier_p = np.asarray(structural_tier_p(jnp.asarray(w_c), jnp.asarray(P_lin_rt)))
    assert np.allclose(rt_tier_p, cache_tier_p, atol=1e-10, rtol=1e-10)
    # also matches the cache's stored P_tier_p on the finite bins
    assert np.allclose(rt_tier_p, d["P_tier_p"][row], atol=1e-10, rtol=1e-10)


def _joint_batch(n_k=8):
    """A standardized synthetic joint-loss batch (Head-A masks present)."""
    _tpf = (jnp.where(jnp.arange(n_k) < n_k - 2, 1.0, jnp.nan)[None, None, :]
            * jnp.ones((2, 4, n_k)))
    return {
        "x": jnp.zeros((2, 10)), "tau0": jnp.array([0.3, 0.5]),
        "t_f_nhi": jnp.zeros((2, 30)), "t_dndx": jnp.zeros((2, 3)),
        "t_p_base": _tpf,
        "t_p_resid": _tpf,
        "t_delta": jnp.zeros((2, 3, n_k)),
        "t_f_nhi_mask": jnp.ones((2, 30), bool),
        "t_dndx_mask": jnp.ones((2, 3), bool),
        "mask": (jnp.arange(n_k) < n_k - 2)[None, :] * jnp.ones((2, n_k), bool),
        "inv_nc": jnp.array([[1., 1 / 10, 1 / 7, 0.], [1., 1 / 10, 1 / 7, 1 / 3]]),
        "inv_nalpha": jnp.array([0.25, 0.25]),
        "mean_F_clean": jnp.array([0.7, 0.6]),
    }


def test_joint_loss_independent_of_meanF_and_grads_finite():
    """A3: the dead meanF term is gone. joint_loss must (a) not reference
    mean_F_clean (perturbing it leaves the loss & grad bit-identical), and
    (b) still produce a finite scalar with finite model gradients."""
    key = jax.random.PRNGKey(33)
    m = Emulator(in_dim=10, n_k=8, key=key)
    batch = _joint_batch(8)
    val, grad = jax.value_and_grad(lambda mm: joint_loss(mm, batch))(m)
    assert jnp.isfinite(val)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grad, eqx.is_array))
    assert all(jnp.all(jnp.isfinite(g)) for g in leaves)

    # mean_F_clean no longer enters the loss: corrupt it -> identical value & grad
    batch2 = dict(batch)
    batch2["mean_F_clean"] = jnp.array([99.0, -99.0])
    val2, grad2 = jax.value_and_grad(lambda mm: joint_loss(mm, batch2))(m)
    assert jnp.array_equal(val, val2)
    g1 = jax.tree_util.tree_leaves(eqx.filter(grad, eqx.is_array))
    g2 = jax.tree_util.tree_leaves(eqx.filter(grad2, eqx.is_array))
    for a, b in zip(g1, g2):
        assert jnp.array_equal(a, b)

    # joint_loss also works if mean_F_clean is absent entirely (truly unused)
    batch3 = {k: v for k, v in batch.items() if k != "mean_F_clean"}
    val3 = joint_loss(m, batch3)
    assert jnp.array_equal(val, val3)


def test_uniform_term_w_terms_comparable():
    """A4: with A1-standardized targets, the four per-channel loss terms are
    within ~1 order of magnitude of each other (no channel dominates), so the
    uniform default term_w is justified — no hand-sweep needed."""
    import numpy as np
    from hcd_analysis.emulator.model import masked_mse
    key = jax.random.PRNGKey(41)
    m = Emulator(in_dim=10, n_k=12, key=key)
    rng = np.random.default_rng(0)
    n_k = 12
    # standardized synthetic targets: ~unit-variance (the A1 transformed space)
    batch = {
        "x": jnp.array(rng.standard_normal((4, 10))),
        "tau0": jnp.array(rng.uniform(0.1, 0.6, 4)),
        "t_f_nhi": jnp.array(rng.standard_normal((4, 30))),
        "t_dndx": jnp.array(rng.standard_normal((4, 3))),
        "t_p_base": jnp.array(rng.standard_normal((4, 4, n_k))),
        "t_p_resid": jnp.array(rng.standard_normal((4, 4, n_k))),
        "t_delta": jnp.array(rng.standard_normal((4, 3, n_k))),
        "t_f_nhi_mask": jnp.ones((4, 30), bool),
        "t_dndx_mask": jnp.ones((4, 3), bool),
        "mask": jnp.ones((4, n_k), bool),
        "inv_nc": jnp.ones((4, 4)),
        "inv_nalpha": jnp.ones(4),
        "mean_F_clean": jnp.array(rng.uniform(0.4, 0.8, 4)),
    }
    preds = jax.vmap(m)(batch["x"], batch["tau0"])
    m3 = batch["mask"][:, None, :]
    mask4 = m3 & jnp.ones_like(batch["t_p_resid"], bool)
    terms = {
        "f_nhi": masked_mse(preds["f_nhi"], batch["t_f_nhi"], batch["t_f_nhi_mask"],
                            weight=batch["inv_nalpha"][:, None]),
        "dndx": masked_mse(preds["dndx"], batch["t_dndx"], batch["t_dndx_mask"],
                           weight=batch["inv_nalpha"][:, None]),
        "p_base": masked_mse(preds["P_filt_base"], batch["t_p_base"], mask4,
                             weight=batch["inv_nc"][:, :, None]),
        "p_resid": masked_mse(preds["P_filt_resid"], batch["t_p_resid"], mask4,
                              weight=batch["inv_nc"][:, :, None]),
        "delta": masked_mse(preds["delta"], batch["t_delta"],
                            m3 & jnp.ones_like(batch["t_delta"], bool),
                            weight=batch["inv_nc"][:, 1:, None]),
    }
    vals = jnp.array(list(terms.values()))
    assert jnp.all(jnp.isfinite(vals)) and jnp.all(vals > 0)
    # within ~1 order of magnitude: max/min ratio < 10
    assert float(jnp.max(vals) / jnp.min(vals)) < 10.0

    # default term_w is uniform and meanF-free: passing only the five uniform
    # weights must reproduce the default-arg behaviour exactly (no missing key).
    default = joint_loss(m, batch)
    explicit = joint_loss(m, batch, term_w={"f_nhi": 1.0, "dndx": 1.0,
                                            "p_base": 1.0, "p_resid": 1.0, "delta": 1.0})
    assert jnp.array_equal(default, explicit)
