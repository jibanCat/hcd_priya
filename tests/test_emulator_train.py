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


def test_aggregate_error_vector_shape_and_dla_flag():
    resid = [np.ones((4, 8, 3)) * 0.1, np.ones((4, 8, 3)) * 0.2]
    neff = [np.ones((4, 8, 3)) * 100, np.ones((4, 8, 3)) * 100]
    neff[0][3, 6:, :] = 0.0  # DLA class, high-k, fold0 -> shot-limited
    ev = aggregate_error_vector(resid, neff)
    assert ev["sigma"].shape == (4, 8, 3)
    assert ev["dla_shot_flag"].shape == (8,)
    assert ev["dla_shot_flag"][6] and ev["dla_shot_flag"][7]
    assert not ev["dla_shot_flag"][0]
