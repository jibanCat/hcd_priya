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

from hcd_analysis.emulator.model import Emulator, joint_loss, p_resid_loss
from hcd_analysis.emulator.data import (
    fit_target_norm, fit_baseline_residual_norm, make_batch, safe_log, apply_norm,
)


def _fit_norm(d, train_idx):
    """Merged target normalization: the per-channel transforms (fit_target_norm)
    PLUS the P_filt baseline/residual (z,tau0)-conditional stats
    (fit_baseline_residual_norm). make_batch needs BOTH to emit t_p_base/t_p_resid
    (the normalization REDESIGN); with only fit_target_norm it silently falls back
    to the old global-sigma P_filt path."""
    return {**fit_target_norm(d, train_idx),
            **fit_baseline_residual_norm(d, train_idx)}


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


def _loss_with_term_w(term_w):
    """Build a joint_loss closure with a fixed term_w (for the staged schedule).

    Zeroing a term's weight removes both its forward contribution AND its gradient,
    so a stage trains only the heads that feed its non-zero terms (in conjunction
    with parameter freezing for clean isolation)."""
    def _loss(model, batch):
        return joint_loss(model, batch, term_w=term_w)
    return _loss


@eqx.filter_jit
def train_step_partitioned(diff_model, static_model, opt, opt_state, batch, loss_fn):
    """One AdamW step over a FROZEN/TRAINABLE partition (staged training).

    ``diff_model``/``static_model`` come from ``eqx.partition`` on a trainable-leaf
    filter spec; grads are taken only wrt ``diff_model`` so the static (frozen)
    leaves never move. ``loss_fn(model, batch)`` is a closure (e.g. a term_w-fixed
    joint_loss). Returns (diff_model, opt_state, loss, grad_norm)."""
    def _wrapped(dm):
        model = eqx.combine(dm, static_model)
        return loss_fn(model, batch)
    loss, grads = eqx.filter_value_and_grad(_wrapped)(diff_model)
    grad_norm = _global_grad_norm(grads)
    updates, opt_state = opt.update(grads, opt_state, eqx.filter(diff_model, eqx.is_array))
    diff_model = eqx.apply_updates(diff_model, updates)
    return diff_model, opt_state, loss, grad_norm


@eqx.filter_jit
def evaluate(model, batch):
    """Scalar ``joint_loss`` with no gradient (validation)."""
    return joint_loss(model, batch)


def _warmstart_bases(d, train_idx, norm_stats, n_basis, n_k):
    """SVD warm-starts for (residual_basis, baseline_basis) in the REDESIGN.

    Builds the per-head target arrays from a make_batch on train rows (which already
    produces t_p_base / t_p_resid in each head's standardized space), stacks over
    (rows*classes, n_k), drops NaN (above-Nyquist) rows, and SVD-warm-starts each
    basis. Returns (resid_basis_init, baseline_basis_init), each None when n_basis
    is None or there aren't enough finite rows."""
    if n_basis is None:
        return None, None
    from hcd_analysis.emulator.model import svd_basis_init
    b = make_batch(d, train_idx, norm_stats)
    out = []
    for key in ("t_p_resid", "t_p_base"):
        M = np.asarray(b[key]).reshape(-1, n_k)
        M = M[np.all(np.isfinite(M), axis=1)]
        out.append(svd_basis_init(jnp.asarray(M), n_basis)
                   if M.shape[0] >= n_basis else None)
    return out[0], out[1]   # (residual, baseline)


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
               batch_size=512, seed=0, key=None, patience=10, n_k=None,
               staged=False):
    """Train one LOSO fold on ``train_idx``, validating on ``val_idx`` each epoch.

    - Fits the train-split target normalisation (``fit_target_norm``) on ``train_idx``.
    - SVD warm-starts both P_filt bases (REDESIGN: baseline + residual) when
      ``n_basis`` is set.
    - Minibatched AdamW with cosine-decay LR (over the total #steps), early stop on
      the validation loss with ``patience`` epochs.

    ``staged=True`` dispatches to the 3-stage schedule (``train_fold_staged``):
    (1) baseline head only, (2) freeze baseline + train residual/delta/Head-A,
    (3) joint fine-tune. Early-stop on the val RESIDUAL (cosmology) loss.

    Returns ``(best_model, norm_stats, history)`` where history holds per-epoch
    ``train_loss, val_loss, grad_norm, lr`` lists.
    """
    if staged:
        return train_fold_staged(
            d, train_idx, val_idx, n_basis=n_basis, lr=lr, epochs=epochs,
            batch_size=batch_size, seed=seed, key=key, patience=patience, n_k=n_k)
    if key is None:
        key = jax.random.PRNGKey(seed)
    train_idx = np.asarray(train_idx)
    val_idx = np.asarray(val_idx)
    if n_k is None:
        n_k = d["P_tier_p"].shape[1]

    rng = np.random.default_rng(seed)
    norm_stats = _fit_norm(d, train_idx)

    # SVD warm-start the two P_filt bases (REDESIGN): the BASELINE basis from the
    # σ_marg-standardized cell-mean targets (t_p_base), the RESIDUAL basis from the
    # σ_cosmo-whitened residual targets (t_p_resid) -- each in its head's own space.
    p_filt_basis_init, baseline_basis_init = _warmstart_bases(
        d, train_idx, norm_stats, n_basis, n_k)

    model = Emulator(in_dim=10, n_k=n_k, n_basis=n_basis,
                     p_filt_basis_init=p_filt_basis_init,
                     baseline_basis_init=baseline_basis_init, key=key)

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


def _trainable_mask(model, train_baseline, train_rest):
    """Boolean pytree marking which array leaves are TRAINABLE in a stage.

    ``train_baseline`` toggles the BaselineHead (head_base) leaves; ``train_rest``
    toggles every OTHER array leaf (encoder, Head A, Head B). Used with
    ``eqx.partition`` to freeze a head: a False leaf is moved to ``static`` and its
    gradient is never taken, so it cannot move."""
    is_arr = eqx.filter(model, eqx.is_array)
    # all array leaves -> train_rest, then override the head_base subtree -> train_baseline.
    full = jax.tree_util.tree_map(lambda _: train_rest, is_arr)
    full = eqx.tree_at(
        lambda m: m.head_base, full,
        replace=jax.tree_util.tree_map(
            lambda _: train_baseline, eqx.filter(model.head_base, eqx.is_array)))
    return full


def _run_stage(model, norm_stats, d, train_idx, val_batch, *, n_epochs,
               batch_size, rng, term_w, train_baseline, train_rest, lr,
               history, patience_resid):
    """Run one training stage with frozen/trainable partition + term_w, early-
    stopping on the val residual loss. Returns the best-by-resid model in this
    stage and appends per-epoch diagnostics to ``history``."""
    mask = _trainable_mask(model, train_baseline, train_rest)
    diff_model, static_model = eqx.partition(model, mask)
    steps_per_epoch = max(1, int(np.ceil(len(train_idx) / batch_size)))
    opt = make_optimizer(lr=lr, steps=steps_per_epoch * max(n_epochs, 1))
    opt_state = opt.init(eqx.filter(diff_model, eqx.is_array))
    loss_fn = _loss_with_term_w(term_w)

    best_resid = np.inf
    best_model = eqx.combine(diff_model, static_model)
    stall = 0
    for _ in range(n_epochs):
        ep_losses, ep_g = [], []
        for mb in _iter_minibatches(rng, train_idx, batch_size):
            batch = _pad_batch(_to_jnp_batch(make_batch(d, mb, norm_stats)), batch_size)
            diff_model, opt_state, loss, gnorm = train_step_partitioned(
                diff_model, static_model, opt, opt_state, batch, loss_fn)
            ep_losses.append(float(loss)); ep_g.append(float(gnorm))
        model_now = eqx.combine(diff_model, static_model)
        vresid = float(p_resid_loss(model_now, val_batch))
        history["train_loss"].append(float(np.mean(ep_losses)))
        history["val_loss"].append(float(evaluate(model_now, val_batch)))
        history["val_resid_loss"].append(vresid)
        history["grad_norm"].append(float(np.mean(ep_g)))
        history["lr"].append(lr)
        if vresid < best_resid - 1e-9:
            best_resid = vresid
            best_model = model_now
            stall = 0
        else:
            stall += 1
            if stall >= patience_resid:
                break
    return best_model


def train_fold_staged(d, train_idx, val_idx, *, n_basis=None, lr=1e-3, epochs=100,
                      batch_size=512, seed=0, key=None, patience=10, n_k=None):
    """3-stage training (normalization REDESIGN), early-stop on val RESIDUAL loss.

    Stage 1 (~1/3 epochs): train BaselineHead (+ its basis) on the BASELINE term
      only (term_w p_base=1, all others 0; only head_base trainable).
    Stage 2 (~1/2 epochs): FREEZE BaselineHead; train Head B (residual + delta) and
      Head A on their terms (p_base=0; p_resid/f_nhi/dndx/delta=1).
    Stage 3 (rest): joint fine-tune ALL heads at a small LR (all terms on).

    The early-stop metric in every stage is the val RESIDUAL (cosmology) loss
    (``p_resid_loss``) — the quantity inference needs, NOT the marginal MSE. Returns
    ``(best_model, norm_stats, history)`` with a ``val_resid_loss`` history key.
    """
    if key is None:
        key = jax.random.PRNGKey(seed)
    train_idx = np.asarray(train_idx)
    val_idx = np.asarray(val_idx)
    if n_k is None:
        n_k = d["P_tier_p"].shape[1]
    rng = np.random.default_rng(seed)
    norm_stats = _fit_norm(d, train_idx)

    p_filt_basis_init, baseline_basis_init = _warmstart_bases(
        d, train_idx, norm_stats, n_basis, n_k)
    model = Emulator(in_dim=10, n_k=n_k, n_basis=n_basis,
                     p_filt_basis_init=p_filt_basis_init,
                     baseline_basis_init=baseline_basis_init, key=key)

    val_batch = _to_jnp_batch(make_batch(d, val_idx, norm_stats))
    history = {"train_loss": [], "val_loss": [], "val_resid_loss": [],
               "grad_norm": [], "lr": [], "stage": []}

    e1 = max(1, epochs // 3)
    e2 = max(1, epochs // 2)
    e3 = max(1, epochs - e1 - e2)
    patience_resid = patience

    # Stage 1: baseline head only (θ-blind cell-mean fit).
    n0 = len(history["val_resid_loss"])
    model = _run_stage(
        model, norm_stats, d, train_idx, val_batch, n_epochs=e1, batch_size=batch_size,
        rng=rng, term_w={"f_nhi": 0., "dndx": 0., "p_base": 1., "p_resid": 0., "delta": 0.},
        train_baseline=True, train_rest=False, lr=lr, history=history,
        patience_resid=patience_resid)
    history["stage"] += [1] * (len(history["val_resid_loss"]) - n0)

    # Stage 2: freeze baseline; train residual + delta + Head A.
    n0 = len(history["val_resid_loss"])
    model = _run_stage(
        model, norm_stats, d, train_idx, val_batch, n_epochs=e2, batch_size=batch_size,
        rng=rng, term_w={"f_nhi": 1., "dndx": 1., "p_base": 0., "p_resid": 1., "delta": 1.},
        train_baseline=False, train_rest=True, lr=lr, history=history,
        patience_resid=patience_resid)
    history["stage"] += [2] * (len(history["val_resid_loss"]) - n0)

    # Stage 3: joint fine-tune ALL heads at a small LR.
    n0 = len(history["val_resid_loss"])
    model = _run_stage(
        model, norm_stats, d, train_idx, val_batch, n_epochs=e3, batch_size=batch_size,
        rng=rng, term_w={"f_nhi": 1., "dndx": 1., "p_base": 1., "p_resid": 1., "delta": 1.},
        train_baseline=True, train_rest=True, lr=lr * 0.1, history=history,
        patience_resid=patience_resid)
    history["stage"] += [3] * (len(history["val_resid_loss"]) - n0)

    history = {k: np.asarray(v) for k, v in history.items()}
    return model, norm_stats, history


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
