"""Phase-2b training pipeline (plan Task 13).

optax AdamW loop (cosine-decay LR if total steps known), NaN-safe ``joint_loss``
via ``eqx.filter_value_and_grad``, early stopping on the validation fold, optional
SVD warm-start of HeadB's low-rank P_filt basis, and a round-trippable checkpoint
bundle (eqx leaves + JSON arch/seed meta + pickled train-split norm stats).

Also hosts ``aggregate_error_vector`` (plan Task-14 stub): RMS-over-folds error
vector + DLA high-k shot-noise flag, feeding the likelihood covariance.

Env (MANDATORY): PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya, emu-jax python.
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from hcd_analysis.emulator.model import Emulator, joint_loss
from hcd_analysis.emulator.data import (
    fit_target_norm, make_batch, safe_log, apply_norm,
)


def make_optimizer(lr=1e-3, steps=None, weight_decay=1e-4):
    """AdamW with a cosine-decay LR schedule over ``steps`` (else constant LR)."""
    sched = optax.cosine_decay_schedule(lr, steps) if steps else optax.constant_schedule(lr)
    return optax.adamw(sched, weight_decay=weight_decay)


def _global_grad_norm(grads):
    """Global L2 norm of the array leaves of a grad pytree (diagnostics)."""
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    sq = sum(jnp.sum(jnp.square(g)) for g in leaves)
    return jnp.sqrt(sq)


@eqx.filter_jit
def train_step(model, opt, opt_state, batch):
    """One AdamW step. Returns (model, opt_state, loss, grad_norm)."""
    loss, grads = eqx.filter_value_and_grad(joint_loss)(model, batch)
    grad_norm = _global_grad_norm(grads)
    updates, opt_state = opt.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss, grad_norm


@eqx.filter_jit
def evaluate(model, batch):
    """Scalar ``joint_loss`` with no gradient (validation)."""
    return joint_loss(model, batch)


def _iter_minibatches(rng, idx, batch_size):
    """Yield shuffled minibatches of row indices for one epoch."""
    idx = np.asarray(idx)
    perm = rng.permutation(len(idx))
    for start in range(0, len(idx), batch_size):
        yield idx[perm[start:start + batch_size]]


def _to_jnp_batch(b):
    return {k: jnp.asarray(v) for k, v in b.items()}


def train_fold(d, train_idx, val_idx, *, n_basis=None, lr=1e-3, epochs=100,
               batch_size=512, seed=0, key=None, patience=10, n_k=None):
    """Train one LOSO fold on ``train_idx``, validating on ``val_idx`` each epoch.

    - Fits the train-split target normalisation (``fit_target_norm``) on ``train_idx``.
    - SVD warm-starts HeadB's P_filt basis from the train split's transformed,
      standardized P_filt when ``n_basis`` is set.
    - Minibatched AdamW with cosine-decay LR (over the total #steps), early stop on
      the validation loss with ``patience`` epochs.

    Returns ``(best_model, norm_stats, history)`` where history holds per-epoch
    ``train_loss, val_loss, grad_norm, lr`` lists.
    """
    if key is None:
        key = jax.random.PRNGKey(seed)
    train_idx = np.asarray(train_idx)
    val_idx = np.asarray(val_idx)
    if n_k is None:
        n_k = d["P_tier_p"].shape[1]

    rng = np.random.default_rng(seed)
    norm_stats = fit_target_norm(d, train_idx)

    # SVD warm-start: build basis in the SAME standardized-log space HeadB emits.
    p_filt_basis_init = None
    if n_basis is not None:
        from hcd_analysis.emulator.model import svd_basis_init
        P = d["P_filt"][train_idx]                              # (n,4,K)
        P_std = apply_norm(safe_log(P), norm_stats["P_filt"])   # standardized-log
        P_std = P_std.reshape(-1, n_k)
        P_std = P_std[np.all(np.isfinite(P_std), axis=1)]       # drop NaN (Nyquist) rows
        if P_std.shape[0] >= n_basis:
            p_filt_basis_init = svd_basis_init(jnp.asarray(P_std), n_basis)

    model = Emulator(in_dim=10, n_k=n_k, n_basis=n_basis,
                     p_filt_basis_init=p_filt_basis_init, key=key)

    steps_per_epoch = max(1, int(np.ceil(len(train_idx) / batch_size)))
    total_steps = steps_per_epoch * epochs
    opt = make_optimizer(lr=lr, steps=total_steps)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))
    lr_sched = optax.cosine_decay_schedule(lr, total_steps)

    val_batch = _to_jnp_batch(make_batch(d, val_idx, norm_stats))

    history = {"train_loss": [], "val_loss": [], "grad_norm": [], "lr": []}
    best_val = np.inf
    best_model = model
    stall = 0
    step = 0
    for ep in range(epochs):
        ep_losses = []
        ep_gnorms = []
        ep_lr = float(lr_sched(step))
        for mb in _iter_minibatches(rng, train_idx, batch_size):
            batch = _to_jnp_batch(make_batch(d, mb, norm_stats))
            model, opt_state, loss, gnorm = train_step(model, opt, opt_state, batch)
            ep_losses.append(float(loss))
            ep_gnorms.append(float(gnorm))
            step += 1
        val_loss = float(evaluate(model, val_batch))
        history["train_loss"].append(float(np.mean(ep_losses)))
        history["val_loss"].append(val_loss)
        history["grad_norm"].append(float(np.mean(ep_gnorms)))
        history["lr"].append(ep_lr)

        if val_loss < best_val - 1e-9:
            best_val = val_loss
            best_model = model
            stall = 0
        else:
            stall += 1
            if stall >= patience:
                break

    history = {k: np.asarray(v) for k, v in history.items()}
    return best_model, norm_stats, history


def save_checkpoint(path, model, arch_cfg, norm_stats, seed):
    """Serialise eqx leaves + JSON meta {arch_cfg, seed} + pickled norm_stats."""
    eqx.tree_serialise_leaves(str(path) + ".eqx", model)
    with open(str(path) + ".meta.json", "w") as f:
        json.dump({"arch_cfg": arch_cfg, "seed": int(seed)}, f)
    with open(str(path) + ".norm.pkl", "wb") as f:
        pickle.dump(norm_stats, f)


def load_checkpoint(path):
    """Inverse of ``save_checkpoint``. Returns (model, meta, norm_stats)."""
    with open(str(path) + ".meta.json") as f:
        meta = json.load(f)
    skeleton = Emulator(**meta["arch_cfg"], key=jax.random.PRNGKey(meta["seed"]))
    model = eqx.tree_deserialise_leaves(str(path) + ".eqx", skeleton)
    with open(str(path) + ".norm.pkl", "rb") as f:
        norm = pickle.load(f)
    return model, meta, norm


# --- Task 14 stub: k-fold LOSO error vector + DLA high-k shot flags -----------

def aggregate_error_vector(resid_folds, neff_folds, shot_thresh=1.0):
    """RMS-over-folds error vector + a DLA high-k shot-noise flag.

    ``resid_folds``, ``neff_folds``: lists of (4, K, Zb) arrays (per fold). Returns
    ``sigma`` = RMS of residuals over folds and ``dla_shot_flag`` (K,) True where
    the worst-case DLA effective count over folds/z-bands is below ``shot_thresh``.
    """
    R = np.stack(resid_folds, 0)                 # (F,4,K,Zb)
    N = np.stack(neff_folds, 0)
    sigma = np.sqrt(np.nanmean(R ** 2, axis=0))  # (4,K,Zb)
    dla_neff = N[:, 3, :, :].min(axis=(0, 2))    # worst-case DLA neff per k
    return {"sigma": sigma, "dla_shot_flag": dla_neff < shot_thresh}
