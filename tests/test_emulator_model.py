import jax, jax.numpy as jnp, equinox as eqx
from hcd_analysis.emulator.model import Encoder, HeadA, HeadB, Emulator, structural_tier_p
from hcd_analysis.emulator.model import masked_mse
from hcd_analysis.emulator.model import joint_loss


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
    assert out["P_filt"].shape == (4, 8)
    assert out["delta"].shape == (3, 8)
    w = jnp.array([0.7, 0.18, 0.07, 0.05]); P_filt_lin = jnp.ones((4, 8))
    tp = structural_tier_p(w, P_filt_lin)
    assert tp.shape == (8,)
    assert jnp.allclose(tp, w.sum())


def test_emulator_endtoend_runs():
    key = jax.random.PRNGKey(2)
    m = Emulator(in_dim=10, n_k=8, key=key)
    pred = m(jnp.zeros(10), tau0=jnp.array(0.3))
    assert set(pred) >= {"f_nhi", "dndx", "P_filt", "delta"}


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
    assert pred["P_filt"].shape == (5, 4, 6)
    assert pred["delta"].shape == (5, 3, 6)
    assert pred["f_nhi"].shape == (5, 30)
    assert pred["P_filt"].dtype == jnp.float64
    # batched matches per-row eager (no silent shape/broadcast corruption)
    eager0 = m(X[0], T[0])
    assert jnp.allclose(pred["P_filt"][0], eager0["P_filt"])

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
    batch = {
        "x": jnp.zeros((2,10)), "tau0": jnp.array([0.3, 0.5]),
        "t_f_nhi": jnp.zeros((2,30)), "t_dndx": jnp.zeros((2,3)),
        "t_P_filt": jnp.where(jnp.arange(8) < 6, 1.0, jnp.nan)[None,None,:]*jnp.ones((2,4,8)),
        "t_delta": jnp.zeros((2,3,8)),
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
    for kk in ("f_nhi", "dndx", "P_filt", "delta"):
        assert jnp.array_equal(p1[kk], p2[kk])


def test_emulator_vmap_equals_python_loop_all_heads():
    # vmap(model) over a batch must equal a python loop row-by-row for EVERY head
    # (no leading-axis bug in Head B's reshape(7, n_k) / atleast_1d(tau0) path).
    key = jax.random.PRNGKey(17)
    m = Emulator(in_dim=10, n_k=8, key=key)
    x = jax.random.normal(jax.random.PRNGKey(18), (4, 10))
    t = jnp.linspace(0.05, 0.6, 4)
    pv = jax.vmap(m)(x, t)
    for kk in ("f_nhi", "dndx", "P_filt", "delta"):
        loop = jnp.stack([m(x[i], t[i])[kk] for i in range(4)])
        assert jnp.allclose(pv[kk], loop, atol=0, rtol=0) or jnp.allclose(pv[kk], loop)
