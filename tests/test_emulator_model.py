import jax, jax.numpy as jnp, equinox as eqx
from hcd_analysis.emulator.model import Encoder, HeadA, HeadB, Emulator, structural_tier_p


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
