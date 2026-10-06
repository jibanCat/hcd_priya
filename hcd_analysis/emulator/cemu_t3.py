"""Gate E amendment A1 rev 1 section 3: the T3 (k-coherence) representation comparison. Builds the coherent residual's
second moment in two representations from simulations' per-z coherent vectors coh_s(z, mode):

- M (mode-aligned): the matrix lives on (z, mode), restricted to the modes bracketing the leg's bins anywhere in the
  sampling box, truncated, and bound to the data bins at the query theta;
- P (fixed physical k): each simulation's coh is placed on the data bins at its OWN coordinate, v_s = W_s coh_s, and
  the matrix is mean_s v_s v_s^T, truncated (a constant matrix).

Both use the production log-k binding weights. Held-out simulations are scored under N(0, F_r + diag(topup)), topup
= the diagonal discarded by the truncation (the same definition for both; for M at the held-out theta). Builds no
product; the registered decision rule is applied by the caller."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from . import kcoord as KC
from .cemu_build import bracket_modes, cv_folds, paired_se


def weights(k_com, theta9_unit, z_cells, iz, k_bins):
    """(N, n_z K) linear-in-ln-k weights placing a (z, mode) vector at the bins (each bin's z cell ``iz``) for one
    theta: the production binding's (j, t_log)."""
    K = len(k_com)
    W = np.zeros((len(k_bins), len(z_cells) * K))
    th = jnp.asarray(theta9_unit)
    for c in np.unique(iz):
        sel = np.where(iz == c)[0]
        b = KC.bind(KC.kgrid(k_com, float(z_cells[c]), th), jnp.asarray(np.asarray(k_bins)[sel]))
        j, t = np.asarray(b.j, int), np.asarray(b.t_log, float)
        W[sel, c * K + j - 1] = 1.0 - t
        W[sel, c * K + j] = t
    return W


def mode_set(k_com, z_cells, iz, k_bins, lo, hi):
    """Flattened (z, mode) indices bracketing the bins of each z cell anywhere in the sampling box [lo, hi]."""
    K = len(k_com)
    idx = []
    for c in np.unique(iz):
        k = np.asarray(k_bins)[iz == c]
        first, last = bracket_modes(k_com, float(z_cells[c]), float(k.min()), float(k.max()), lo, hi)
        idx.extend(c * K + n - 1 for n in range(first, last + 1))
    return np.asarray(idx, int)


def top_r(S, r):
    """The rank-r truncation of a symmetric PSD matrix (its top-r eigenpairs)."""
    S = 0.5 * (np.asarray(S, float) + np.asarray(S, float).T)
    w, U = np.linalg.eigh(S)
    w, U = w[::-1][:r], U[:, ::-1][:, :r]
    return (U * np.maximum(w, 0.0)) @ U.T


def with_topup(F, S):
    """F + diag(diag(S) - diag(F)): the truncation's discarded diagonal restored."""
    return F + np.diag(np.maximum(np.diag(S) - np.diag(F), 0.0))


def logpdf(v, S, jitter=1e-10):
    """Gaussian log density of v under N(0, S) with the production jitter (jitter x the mean diagonal)."""
    S = np.asarray(S, float)
    S = S + jitter * np.mean(np.diag(S)) * np.eye(S.shape[0])
    L = np.linalg.cholesky(S)
    x = np.linalg.solve(L, np.asarray(v, float))
    return float(-0.5 * (x @ x) - np.sum(np.log(np.diag(L))) - 0.5 * len(v) * np.log(2 * np.pi))


def compare(coh, thetas, bins, k_com, *, ranks=(5, 10, 15, 25), n_folds=10, lo, hi, names=None, C_frac=None, hook=None):
    """CV comparison of M and P on one leg. ``coh`` (n_sim, n_z, K), ``thetas`` (n_sim, 9) unit cube, ``bins`` dict
    (z: the z cells, iz: each bin's cell, k: the bins' k). Returns per held-out simulation and rank the log scores of
    both representations, and, with ``C_frac`` (the data covariance over P_data P_data^T on the bins), the calibration
    terms of kappa: v^T C^-1 v and tr(C^-1 Sigma) with C = C_frac + Sigma. ``hook(rep, r, s, Sigma, v)`` is called for
    every held-out simulation (its returns are collected in ``out['hook'][rep][r][s]``)."""
    n_sim = coh.shape[0]
    names = [f"s{i:03d}" for i in range(n_sim)] if names is None else list(names)
    folds = cv_folds(names, n_folds)
    fold_of = np.array([folds[n] for n in names])
    z, iz, kb = np.asarray(bins["z"]), np.asarray(bins["iz"]), np.asarray(bins["k"])
    flat = coh.reshape(n_sim, -1)
    Ws = [weights(k_com, thetas[s], z, iz, kb) for s in range(n_sim)]
    V = np.stack([Ws[s] @ flat[s] for s in range(n_sim)])                       # P: each simulation at its own k
    Mi = mode_set(k_com, z, iz, kb, lo, hi)
    Cm = flat[:, Mi]
    out = {"score": {"M": {r: np.zeros(n_sim) for r in ranks}, "P": {r: np.zeros(n_sim) for r in ranks}},
           "quad": {"M": {r: np.zeros(n_sim) for r in ranks}, "P": {r: np.zeros(n_sim) for r in ranks}},
           "trace": {"M": {r: np.zeros(n_sim) for r in ranks}, "P": {r: np.zeros(n_sim) for r in ranks}},
           "hook": {"M": {r: {} for r in ranks}, "P": {r: {} for r in ranks}}}
    for f in range(n_folds):
        tr, te = np.where(fold_of != f)[0], np.where(fold_of == f)[0]
        if te.size == 0:
            continue
        S_P = V[tr].T @ V[tr] / tr.size
        S_M = Cm[tr].T @ Cm[tr] / tr.size
        for r in ranks:
            F_P = top_r(S_P, r)
            Sig_P = with_topup(F_P, S_P)
            F_M = top_r(S_M, r)
            for s in te:
                B = Ws[s][:, Mi]
                if np.max(np.abs(Ws[s] @ flat[s] - B @ Cm[s])) > 1e-12 * max(1.0, np.max(np.abs(V[s]))):
                    raise ValueError("a bin's bracketing modes fall outside the mode set")
                Sig_M = B @ F_M @ B.T
                Sig_M = Sig_M + np.diag(np.maximum(np.einsum("ij,jk,ik->i", B, S_M, B) - np.diag(Sig_M), 0.0))
                for rep, Sig in (("P", Sig_P), ("M", Sig_M)):
                    out["score"][rep][r][s] = logpdf(V[s], Sig)
                    if C_frac is not None:
                        C = np.asarray(C_frac) + Sig
                        out["quad"][rep][r][s] = float(V[s] @ np.linalg.solve(C, V[s]))
                        out["trace"][rep][r][s] = float(np.trace(np.linalg.solve(C, Sig)))
                    if hook is not None:
                        out["hook"][rep][r][s] = hook(rep, r, s, Sig, V[s])
    out["names"], out["V"], out["mode_set"] = names, V, Mi
    return out


def decide(res, ranks, *, n_boot=1000, seed=0, min_nats=1.0):
    """The registered rule's statistical part: each representation at its CV-best rank (ties to the lower rank); the
    paired-SE gain of M over P. The calibration conjunct and the stop are applied by the caller."""
    best = {}
    for rep in ("M", "P"):
        means = [res["score"][rep][r].mean() for r in ranks]
        best[rep] = ranks[int(np.argmax(np.round(means, 9)))]
    sM, sP = res["score"]["M"][best["M"]], res["score"]["P"][best["P"]]
    gain, se = float(np.mean(sM - sP)), paired_se(sM, sP, n_boot, seed)
    return dict(best_rank=best, gain=gain, gain_se=se, m_wins_score=bool(gain > 2 * se and gain > min_nats))
