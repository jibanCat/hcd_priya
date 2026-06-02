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


def _pad_batch(batch, batch_size):
    """Pad a (possibly ragged) batch up to a FIXED ``batch_size`` leading dim with
    ZERO-CONTRIBUTION rows, so ``train_step`` traces ONCE (CS-I2).

    The final minibatch has size ``len(train_idx) % batch_size`` — a 2nd leading
    shape that forces ``train_step`` to recompile. We pad every array up to
    ``batch_size`` and neutralise the padded rows so ``joint_loss`` gives them
    exactly zero contribution AND zero gradient:
      - ``mask`` -> False (all k): masked_mse zeros the diff there and the False
        rows add nothing to its ``sum(mask)`` denominator, so the loss is
        numerically identical to the unpadded ragged batch;
      - ``t_f_nhi_mask`` / ``t_dndx_mask`` -> False (Head-A masks);
      - ``inv_nc`` -> 0 (all classes), ``inv_nalpha`` -> 0: belt-and-suspenders
        zero weight so even an unmasked element would carry no gradient.
    Other arrays are zero-padded (their values are irrelevant once masked/zero-
    weighted). Returns the batch unchanged when it is already ``batch_size``."""
    n = len(batch["x"])
    if n >= batch_size:
        return batch
    pad = batch_size - n
    out = {}
    for k, v in batch.items():
        v = jnp.asarray(v)
        pad_width = [(0, pad)] + [(0, 0)] * (v.ndim - 1)
        if k in ("mask", "t_f_nhi_mask", "t_dndx_mask"):
            out[k] = jnp.pad(v, pad_width, constant_values=False)  # padded -> masked out
        elif k in ("inv_nc", "inv_nalpha"):
            out[k] = jnp.pad(v, pad_width, constant_values=0)      # zero weight (belt+suspenders)
        else:
            out[k] = jnp.pad(v, pad_width, constant_values=0)
    return out


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
        for mb in _iter_minibatches(rng, train_idx, batch_size):
            # CS-I2: pad the (ragged) final minibatch to a FIXED batch_size with
            # zero-contribution rows, so train_step keeps ONE static shape (traces once).
            batch = _pad_batch(_to_jnp_batch(make_batch(d, mb, norm_stats)), batch_size)
            model, opt_state, loss, gnorm = train_step(model, opt, opt_state, batch)
            ep_losses.append(float(loss))
            ep_gnorms.append(float(gnorm))
            step += 1
        # CS-I1: the optimizer (cosine_decay_schedule) owns the schedule and advances
        # per minibatch step — it is the source of truth. Log the LR actually in effect
        # AFTER the epoch's last update (step == #updates so far), not the epoch's first step.
        ep_lr = float(lr_sched(step))
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


def save_checkpoint(path, model, arch_cfg, norm_stats, seed,
                    kfkms=None, cache_path=None):
    """Serialise eqx leaves + JSON meta + pickled norm_stats.

    M4: the meta now records the k-grid identity so a checkpoint is self-
    describing — ``kfkms`` (the cache's k-grid, list of floats; ``n_k`` derived
    from it) and ``cache_path`` (the cache the model was trained on). Either may
    be omitted (back-compat), in which case the corresponding key is null."""
    eqx.tree_serialise_leaves(str(path) + ".eqx", model)
    meta = {"arch_cfg": arch_cfg, "seed": int(seed),
            "cache_path": (str(cache_path) if cache_path is not None else None)}
    if kfkms is not None:
        # cache kfkms is (R, n_k) (per-row, shared grid); store the single k-grid.
        kf = np.asarray(kfkms)
        kgrid = kf[0] if kf.ndim == 2 else np.atleast_1d(kf).ravel()
        meta["kfkms"] = kgrid.astype(float).tolist()
        meta["n_k"] = int(kgrid.shape[0])
    else:
        meta["kfkms"] = None
        meta["n_k"] = arch_cfg.get("n_k")
    with open(str(path) + ".meta.json", "w") as f:
        json.dump(meta, f)
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
