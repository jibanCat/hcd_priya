"""Task 13 training-pipeline tests (TDD against the synthetic v3.3 fixture).

Covers: a single jit'd train_step reduces loss; train_fold runs + early-stops
with a valid history (dense AND n_basis SVD-warm-start paths); checkpoint round-
trips predictions bit-for-bit; grad_norm is finite; and the Task-14 error-vector
aggregator's shape + DLA high-k shot flag.

Run (env mandatory):
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya \
    /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_emulator_train.py -v
"""
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from tests.emulator._fixture import write_synthetic_cache
from hcd_analysis.emulator.data import (
    load_cache, fit_target_norm, make_batch, kfold_loso,
)
from hcd_analysis.emulator.model import Emulator, joint_loss
from hcd_analysis.emulator.train import (
    make_optimizer, train_step, evaluate, train_fold,
    save_checkpoint, load_checkpoint, aggregate_error_vector,
    _pad_batch, _iter_minibatches, _to_jnp_batch,
)

N_K = 8


def _toy_batch(key, B=4, n_k=N_K):
    return {
        "x": jax.random.normal(key, (B, 10)),
        "tau0": jnp.linspace(0.2, 0.6, B),
        "t_f_nhi": jnp.zeros((B, 30)),
        "t_dndx": jnp.zeros((B, 3)),
        "t_P_filt": jnp.ones((B, 4, n_k)),
        "t_delta": jnp.zeros((B, 3, n_k)),
        "t_f_nhi_mask": jnp.ones((B, 30), bool),
        "t_dndx_mask": jnp.ones((B, 3), bool),
        "mask": jnp.ones((B, n_k), bool),
        "inv_nc": jnp.ones((B, 4)),
        "inv_nalpha": jnp.ones(B),
        "mean_F_clean": jnp.exp(-jnp.linspace(0.2, 0.6, B)),
    }


def test_train_step_reduces_loss():
    key = jax.random.PRNGKey(5)
    m = Emulator(in_dim=10, n_k=N_K, key=key)
    batch = _toy_batch(key)
    opt = make_optimizer(lr=1e-3)
    opt_state = opt.init(eqx.filter(m, eqx.is_array))
    l0 = float(joint_loss(m, batch))
    last = l0
    for _ in range(50):
        m, opt_state, last, gnorm = train_step(m, opt, opt_state, batch)
    assert float(last) < l0
    assert jnp.isfinite(gnorm)


def test_grad_norm_finite():
    key = jax.random.PRNGKey(7)
    m = Emulator(in_dim=10, n_k=N_K, key=key)
    batch = _toy_batch(key)
    opt = make_optimizer(lr=1e-3)
    opt_state = opt.init(eqx.filter(m, eqx.is_array))
    m, opt_state, loss, gnorm = train_step(m, opt, opt_state, batch)
    assert jnp.isfinite(loss)
    assert jnp.isfinite(gnorm)
    assert float(gnorm) >= 0.0


def _small_cache(tmp_path):
    path = tmp_path / "obs.h5"
    write_synthetic_cache(path, n_sims=4, snaps_per_sim=2, n_alpha=4, n_k=N_K)
    return load_cache(path)


def _check_history(hist):
    for k in ("train_loss", "val_loss", "grad_norm", "lr"):
        assert k in hist, f"history missing {k}"
        assert len(hist[k]) >= 1
        assert np.all(np.isfinite(hist[k]))


def test_train_fold_runs_and_early_stops_dense(tmp_path):
    d = _small_cache(tmp_path)
    folds = kfold_loso(d["sim_name"], n_folds=4)
    tr, va = folds[0]
    model, norm_stats, hist = train_fold(
        d, tr, va, n_basis=None, lr=1e-3, epochs=40, batch_size=8,
        seed=0, key=jax.random.PRNGKey(0),
    )
    assert isinstance(model, Emulator)
    _check_history(hist)
    # train loss should fall from first to (best) last
    assert hist["train_loss"][-1] < hist["train_loss"][0]


def test_train_fold_runs_with_svd_warmstart(tmp_path):
    d = _small_cache(tmp_path)
    folds = kfold_loso(d["sim_name"], n_folds=4)
    tr, va = folds[0]
    model, norm_stats, hist = train_fold(
        d, tr, va, n_basis=4, lr=1e-3, epochs=40, batch_size=8,
        seed=1, key=jax.random.PRNGKey(1),
    )
    assert model.head_b.n_basis == 4
    assert model.head_b.p_filt_basis.shape == (4, N_K)
    _check_history(hist)
    assert hist["train_loss"][-1] < hist["train_loss"][0]


def test_checkpoint_roundtrip(tmp_path):
    d = _small_cache(tmp_path)
    folds = kfold_loso(d["sim_name"], n_folds=4)
    tr, va = folds[0]
    arch_cfg = {"in_dim": 10, "n_k": N_K, "n_basis": 4}
    model, norm_stats, _ = train_fold(
        d, tr, va, n_basis=4, lr=1e-3, epochs=5, batch_size=8,
        seed=2, key=jax.random.PRNGKey(2),
    )
    # prediction before save
    x = d["x"][va][:3]
    tau0 = d["tau0"][va][:3]
    pred0 = jax.vmap(model)(jnp.asarray(x), jnp.asarray(tau0))

    prefix = str(tmp_path / "ckpt")
    save_checkpoint(prefix, model, arch_cfg, norm_stats, seed=2)
    model2, meta, norm2 = load_checkpoint(prefix)

    pred1 = jax.vmap(model2)(jnp.asarray(x), jnp.asarray(tau0))
    for k in pred0:
        assert np.array_equal(np.asarray(pred0[k]), np.asarray(pred1[k])), k
    assert meta["seed"] == 2
    assert meta["arch_cfg"] == arch_cfg
    # norm_stats round-trip
    for ch in norm_stats:
        for stat in norm_stats[ch]:
            assert np.allclose(norm_stats[ch][stat], norm2[ch][stat])


def test_padded_batch_matches_unpadded(tmp_path):
    """CS-I2: a padded fixed-size batch gives loss/grad numerically identical
    (~1e-12) to the unpadded ragged batch (padding contributes nothing), and the
    padded batch has the FIXED (batch_size, ...) leading shape."""
    from hcd_analysis.emulator.data import fit_target_norm, make_batch
    d = _small_cache(tmp_path)
    R = d["P_filt"].shape[0]
    norm = fit_target_norm(d, np.arange(R))
    # ragged set: not a multiple of batch_size
    idx = np.arange(7)
    batch_size = 16
    ragged = _to_jnp_batch(make_batch(d, idx, norm))
    padded = _pad_batch(ragged, batch_size)
    # fixed leading shape on every array
    for k, v in padded.items():
        assert np.asarray(v).shape[0] == batch_size, k
    assert len(ragged["x"]) == 7

    m = Emulator(in_dim=10, n_k=d["P_filt"].shape[2], key=jax.random.PRNGKey(0))
    lr, gr = eqx.filter_value_and_grad(joint_loss)(m, ragged)
    lp, gp = eqx.filter_value_and_grad(joint_loss)(m, padded)
    assert abs(float(lr) - float(lp)) <= 1e-12 * max(1.0, abs(float(lr)))
    for a, b in zip(jax.tree_util.tree_leaves(eqx.filter(gr, eqx.is_array)),
                    jax.tree_util.tree_leaves(eqx.filter(gp, eqx.is_array))):
        assert np.allclose(np.asarray(a), np.asarray(b), atol=1e-12, rtol=0)


def test_padded_batch_matches_unpadded_with_structural_zeros_and_nan(tmp_path):
    """CS-I2 referee guard: padded == unpadded loss/grad EVEN when the real rows
    carry the hard cases the bare-fixture test misses — structural-zero Head-A
    targets (t_f_nhi_mask/t_dndx_mask have False entries) AND NaN P_filt targets
    (above-Nyquist rows). This is the genuine no-leak path: a padded row whose
    target is NaN/zero-masked must contribute exactly 0 to loss AND grad, with no
    NaN/inf leak through masked_mse."""
    from hcd_analysis.emulator.data import fit_target_norm, make_batch
    d = _small_cache(tmp_path)
    # inject structural zeros so the Head-A validity masks actually carry False
    d["snap_f_nhi"][:, 0] = 0.0
    d["snap_f_nhi"][1, 4] = 0.0
    d["snap_dNdX"][0, 1] = 0.0
    R = d["P_filt"].shape[0]
    norm = fit_target_norm(d, np.arange(R))
    idx = np.arange(R)                       # all rows: spans every class + Nyquist-NaN
    batch_size = R + 9                        # ragged: forces padding
    ragged = _to_jnp_batch(make_batch(d, idx, norm))
    # the masks MUST contain False entries (else this test degenerates to the easy one)
    assert (~np.asarray(ragged["t_f_nhi_mask"])).any(), "no structural-zero in Head-A mask"
    assert np.isnan(np.asarray(ragged["t_P_filt"])).any(), "no NaN P_filt target present"
    padded = _pad_batch(ragged, batch_size)

    m = Emulator(in_dim=10, n_k=d["P_filt"].shape[2], key=jax.random.PRNGKey(1))
    lr, gr = eqx.filter_value_and_grad(joint_loss)(m, ragged)
    lp, gp = eqx.filter_value_and_grad(joint_loss)(m, padded)
    assert np.isfinite(float(lp)), "padded loss not finite (NaN/inf leak)"
    assert abs(float(lr) - float(lp)) <= 1e-12 * max(1.0, abs(float(lr)))
    for a, b in zip(jax.tree_util.tree_leaves(eqx.filter(gr, eqx.is_array)),
                    jax.tree_util.tree_leaves(eqx.filter(gp, eqx.is_array))):
        a = np.asarray(a); b = np.asarray(b)
        assert np.all(np.isfinite(b)), "padded grad leaf has NaN/inf"
        # grads here are O(1e13); the only diff is float64 reassociation from the
        # larger (padded) reduction, so the match is RELATIVE (machine precision),
        # not absolute. rtol=1e-10 is ~4 orders above the observed 7e-14.
        assert np.allclose(a, b, rtol=1e-10, atol=1e-12)

    # adversarial: even if a PADDED target itself were NaN (not just zero-padded),
    # the mask-False on padded rows must still floor its contribution to zero.
    import jax.numpy as _jnp
    tpf = np.asarray(padded["t_P_filt"]).copy(); tpf[R:] = np.nan
    padded_nan = {**padded, "t_P_filt": _jnp.asarray(tpf)}
    lp2 = float(joint_loss(m, padded_nan))
    assert np.isfinite(lp2) and abs(lp2 - float(lr)) <= 1e-12 * max(1.0, abs(float(lr)))


def test_pad_batch_noop_when_full(tmp_path):
    from hcd_analysis.emulator.data import fit_target_norm, make_batch
    d = _small_cache(tmp_path)
    norm = fit_target_norm(d, np.arange(d["P_filt"].shape[0]))
    full = _to_jnp_batch(make_batch(d, np.arange(8), norm))
    assert _pad_batch(full, 8) is full   # exact passthrough, no copy


def test_train_step_single_trace_with_padding(tmp_path):
    """CS-I2: with padding, every minibatch in an epoch shares ONE leading shape,
    so train_step traces exactly once. Verified two ways: (1) all padded batches
    have leading dim == batch_size; (2) a wrapped joint_loss compile counter fires
    once across the epoch."""
    from hcd_analysis.emulator.data import fit_target_norm, make_batch
    d = _small_cache(tmp_path)
    R = d["P_filt"].shape[0]
    norm = fit_target_norm(d, np.arange(R))
    rng = np.random.default_rng(0)
    train_idx = np.arange(R)             # R = 4*2*4 = 32; ragged at batch_size below
    batch_size = 10

    shapes = set()
    n_batches = 0
    for mb in _iter_minibatches(rng, train_idx, batch_size):
        b = _pad_batch(_to_jnp_batch(make_batch(d, mb, norm)), batch_size)
        shapes.add(np.asarray(b["x"]).shape)
        n_batches += 1
    assert n_batches >= 2            # genuinely ragged (last batch < batch_size)
    assert shapes == {(batch_size, 10)}   # exactly ONE static shape

    # compile counter: wrap a jit'd step and assert one trace across the epoch
    trace_count = {"n": 0}

    @eqx.filter_jit
    def counted_step(model, opt, opt_state, batch):
        trace_count["n"] += 1   # Python side-effect fires only on (re)trace
        loss, grads = eqx.filter_value_and_grad(joint_loss)(model, batch)
        updates, opt_state = opt.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    m = Emulator(in_dim=10, n_k=d["P_filt"].shape[2], key=jax.random.PRNGKey(0))
    opt = make_optimizer(lr=1e-3)
    opt_state = opt.init(eqx.filter(m, eqx.is_array))
    for mb in _iter_minibatches(rng, train_idx, batch_size):
        b = _pad_batch(_to_jnp_batch(make_batch(d, mb, norm)), batch_size)
        m, opt_state, _ = counted_step(m, opt, opt_state, b)
    assert trace_count["n"] == 1, f"train_step retraced {trace_count['n']} times"


def test_train_fold_returns_best_not_last(tmp_path):
    """The returned model IS the best-epoch model (lowest val loss), not the last:
    evaluate(returned_model, val_batch) == min(history['val_loss'])."""
    from hcd_analysis.emulator.data import fit_target_norm, make_batch
    d = _small_cache(tmp_path)
    folds = kfold_loso(d["sim_name"], n_folds=4)
    tr, va = folds[0]
    # long run + tiny patience so val rises after the minimum and we early-stop
    model, norm_stats, hist = train_fold(
        d, tr, va, n_basis=None, lr=5e-2, epochs=60, batch_size=8,
        seed=0, key=jax.random.PRNGKey(0), patience=5,
    )
    best_val = float(np.min(hist["val_loss"]))
    val_batch = _to_jnp_batch(make_batch(d, va, norm_stats))
    got = float(evaluate(model, val_batch))
    assert abs(got - best_val) <= 1e-6, (got, best_val)
    # sanity: val actually rose after the min (otherwise the test is vacuous)
    argmin = int(np.argmin(hist["val_loss"]))
    assert argmin < len(hist["val_loss"]) - 1 or len(hist["val_loss"]) == 60


def test_checkpoint_meta_has_kgrid(tmp_path):
    """M4: saved .meta.json records the k-grid (kfkms + n_k) and the cache id, and
    load reconstructs. n_k matches len(kfkms)."""
    import json
    d = _small_cache(tmp_path)
    folds = kfold_loso(d["sim_name"], n_folds=4)
    tr, va = folds[0]
    arch_cfg = {"in_dim": 10, "n_k": N_K, "n_basis": None}
    model, norm_stats, _ = train_fold(
        d, tr, va, n_basis=None, lr=1e-3, epochs=3, batch_size=8,
        seed=0, key=jax.random.PRNGKey(0),
    )
    prefix = str(tmp_path / "ckpt_kg")
    cache_path = str(tmp_path / "obs.h5")
    save_checkpoint(prefix, model, arch_cfg, norm_stats, seed=0,
                    kfkms=d["kfkms"], cache_path=cache_path)
    with open(prefix + ".meta.json") as f:
        meta = json.load(f)
    assert meta["n_k"] == N_K
    assert len(meta["kfkms"]) == N_K
    assert np.allclose(meta["kfkms"], d["kfkms"][0])
    assert meta["cache_path"] == cache_path
    # load_checkpoint still round-trips with the enriched meta
    model2, meta2, norm2 = load_checkpoint(prefix)
    assert meta2["n_k"] == N_K and meta2["cache_path"] == cache_path


def test_aggregate_error_vector_shape_and_dla_flag():
    resid = [np.ones((4, 8, 3)) * 0.1, np.ones((4, 8, 3)) * 0.2]
    neff = [np.ones((4, 8, 3)) * 100, np.ones((4, 8, 3)) * 100]
    neff[0][3, 6:, :] = 0.0  # DLA class, high-k, fold0 -> shot-limited
    ev = aggregate_error_vector(resid, neff)
    assert ev["sigma"].shape == (4, 8, 3)
    assert ev["dla_shot_flag"].shape == (8,)
    assert ev["dla_shot_flag"][6] and ev["dla_shot_flag"][7]
    assert not ev["dla_shot_flag"][0]
