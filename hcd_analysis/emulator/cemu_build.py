"""Building blocks of the gate E emulator-error products (GATE_E_AMENDMENT_A1.md in the notes repository, registered
before any final product is built): simulation-level cross-validation folds, per-cell second moments of the
leave-one-simulation-out ensemble residuals, PSD-preserving smoothing along ln(mode) and z, the Gaussian predictive
score used for model selection, and the modes that bracket a leg's data bins over the whole sampling box.
Products are calibrated uncertainty tables: nothing here is a function of the cosmological or nuisance parameters."""
from __future__ import annotations

import numpy as np

from .kcoord import kbounds_over_box


def load_loo_ensemble_residuals(eval_dir, d, sims=None, n_members=5):
    """The 60-fold leave-one-simulation-out ENSEMBLE residuals (gate C): for each held-out simulation index, the mean over
    its n_members evaluations (seed 0 ``eval_loo60_s{NN}.npz``, seeds 1.. ``loo60_ensemble/eval_loo60_s{NN}_seed{S}.npz``)
    of the per-class fractional residuals at every mode of every one of its cache rows (common truth, so the member
    mean is the residual of the mean prediction). Returns rows (cache indices), sim, z, alpha_idx, tau0 and r (rows, 4, K)."""
    sims = range(60) if sims is None else sims
    out = {k: [] for k in ("rows", "r")}
    for n in sims:
        files = [f"{eval_dir}/eval_loo60_s{n:02d}.npz"] + [f"{eval_dir}/loo60_ensemble/eval_loo60_s{n:02d}_seed{s}.npz"
                                                            for s in range(1, n_members)]
        ev = [np.load(p, allow_pickle=True) for p in files]
        rows = np.asarray(ev[0]["rows"])
        for e in ev[1:]:
            if not np.array_equal(np.asarray(e["rows"]), rows):
                raise ValueError(f"members of held-out simulation {n} disagree on rows")
        out["rows"].append(rows)
        out["r"].append(np.mean([np.asarray(e["res_cls"], float) for e in ev], axis=0))
    rows = np.concatenate(out["rows"])
    return dict(rows=rows, r=np.concatenate(out["r"]), sim=np.asarray(d["sim_name"]).astype(str)[rows],
                z=np.asarray(d["z_grid"], float)[rows], alpha_idx=np.asarray(d["alpha_idx"])[rows],
                tau0=np.asarray(d["tau0"], float)[rows])


def cv_folds(sim_names, n_folds=10):
    """{simulation name: fold} with the simulations sorted by name and fold = index mod n_folds."""
    names = sorted(set(np.asarray(sim_names).astype(str)))
    return {n: i % n_folds for i, n in enumerate(names)}


def second_moment_cells(r, cell, n_cells):
    """Uncentered second moment per cell of the per-class residuals r (rows, C, K): rho (n_cells, C, C, K) and counts."""
    r = np.asarray(r, float)
    cell = np.asarray(cell, int)
    C, K = r.shape[1], r.shape[2]
    rho = np.zeros((n_cells, C, C, K))
    n = np.zeros(n_cells, int)
    for c in range(n_cells):
        sel = cell == c
        n[c] = int(sel.sum())
        if n[c]:
            rho[c] = np.einsum("rck,rdk->cdk", r[sel], r[sel]) / n[c]
    return rho, n


def _kernel(x, width):
    d = (np.asarray(x, float)[:, None] - np.asarray(x, float)[None, :]) / width
    w = np.exp(-0.5 * d ** 2)
    return w / w.sum(axis=1, keepdims=True)


def smooth_modes(rho, h):
    """Smooth the last (mode) axis with a Gaussian kernel of width h in ln(mode index); h = 0 returns rho unchanged.
    Row-normalised positive weights: a PSD matrix field stays PSD and a constant stays constant."""
    if h == 0:
        return rho
    W = _kernel(np.log(np.arange(1, rho.shape[-1] + 1)), h)
    return np.einsum("...k,jk->...j", rho, W)


def smooth_z(rho, z, s_z):
    """Smooth the first (z cell) axis with a Gaussian kernel of width s_z in z; s_z = 0 returns rho unchanged."""
    if s_z == 0:
        return rho
    W = _kernel(z, s_z)
    return np.einsum("z...,yz->y...", rho, W)


def gaussian_score(r, cov, jitter=1e-10):
    """Sum over rows and modes of the C-dimensional Gaussian log density of r (rows, C, K) under cov (rows, C, C, K),
    with the production jitter (jitter x the mean diagonal) added to each block."""
    r = np.moveaxis(np.asarray(r, float), 2, 1)                      # (rows, K, C)
    cov = np.moveaxis(np.asarray(cov, float), 3, 1)                  # (rows, K, C, C)
    C = r.shape[-1]
    diag_mean = np.einsum("...ii->...", cov) / C
    cov = cov + (jitter * diag_mean)[..., None, None] * np.eye(C)
    sign, logdet = np.linalg.slogdet(cov)
    if np.any(sign <= 0):
        raise ValueError("non-positive-definite covariance block in the Gaussian score")
    sol = np.linalg.solve(cov, r[..., None])[..., 0]
    quad = np.sum(r * sol, axis=-1)
    return float(np.sum(-0.5 * (quad + logdet + C * np.log(2 * np.pi))))


def rho_interp_alpha(rho_zb, alpha_centres, alpha):
    """Rows-batched numpy mirror of the deployed ``likelihood.rho_at_tau0``: rho_zb (C, C, K, B) at the tau0-ladder factors
    ``alpha`` (R,) -> (R, C, C, K), linear in alpha between the band centres, clamped flat outside them, NaN cells 0.
    B = 1 (pooled over tau0) returns the single block for every row."""
    rho = np.nan_to_num(np.asarray(rho_zb, float))
    c = np.asarray(alpha_centres, float)
    a = np.atleast_1d(np.asarray(alpha, float))
    if c.size == 1:
        return np.broadcast_to(rho[..., 0], (a.size,) + rho.shape[:-1]).copy()
    a = np.clip(a, c[0], c[-1])
    j = np.clip(np.searchsorted(c, a, side="right") - 1, 0, c.size - 2)
    t = (a - c[j]) / (c[j + 1] - c[j])
    lo = np.moveaxis(rho[..., j], -1, 0)
    hi = np.moveaxis(rho[..., j + 1], -1, 0)
    return lo * (1.0 - t)[:, None, None, None] + hi * t[:, None, None, None]


def class_coef(w_c, masked):
    """Production class coefficients (1 - sum a, a) from per-row class weights w_c (R, 4) (a = w_c[:, 1:]); the DLA mask
    (the main arm, ``dla_forward_frac = 0``) sets a_DLA = 0 as ``forward.predict_leg`` does."""
    a = np.array(np.asarray(w_c, float)[:, 1:], copy=True)
    if masked:
        a[:, 2] = 0.0
    return np.concatenate([1.0 - a.sum(axis=1, keepdims=True), a], axis=1)


def combined_variance(P, coef, rho):
    """Fractional variance of the production-combined spectrum P_obs = sum_c coef_c P_c under the cross-class second
    moment rho: (sum_cc' coef_c coef_c' rho_cc' P_c P_c') / P_obs^2 (the ``emu_var_modes`` algebra), and P_obs.
    Shapes: P (..., C, K), coef (..., C), rho (..., C, C, K)."""
    P_obs = np.einsum("...c,...ck->...k", coef, P)
    var = np.einsum("...c,...d,...cdk,...ck,...dk->...k", coef, coef, rho, P, P)
    return var / P_obs ** 2, P_obs


def combined_logpdf(r, P, coef, rho, mask):
    """Gaussian log density of the production-combined fractional residual e = sum_c coef_c P_c r_c / P_obs under its
    predicted fractional variance, summed over the modes in ``mask`` (K,) or (..., K). r, P (..., C, K)."""
    var, P_obs = combined_variance(P, coef, rho)
    e = np.einsum("...c,...ck,...ck->...k", coef, P, r) / P_obs
    lp = -0.5 * (e ** 2 / var + np.log(2.0 * np.pi * var))
    return np.sum(np.where(mask, lp, 0.0), axis=-1)


def paired_se(a, b, n_boot=1000, seed=0):
    """Simulation-bootstrap SE of mean(a - b) over paired per-simulation values (resampling simulations)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    idx = np.random.default_rng(seed).integers(0, d.size, (n_boot, d.size))
    return float(np.std(d[idx].mean(axis=1), ddof=1))


def select_argmax(mean_scores, smooth_rank, tol=1e-6):
    """Index of the largest mean CV score; candidates within ``tol`` (relative) of the best go to the most smoothing
    (largest ``smooth_rank``)."""
    m = np.asarray(mean_scores, float)
    best = m.max()
    cand = np.where(m >= best - tol * abs(best))[0]
    return int(cand[np.argmax(np.asarray(smooth_rank)[cand])])


def choose_tau0(banded, pooled, n_boot=1000, seed=0):
    """'banded' if the banded T1 beats the pooled one by more than 2 paired simulation-bootstrap SE in mean CV score,
    else 'pooled' (amendment A1 rev 1 section 1: a banded win STOPS for the PI; the caller acts on it)."""
    gain = float(np.mean(np.asarray(banded) - np.asarray(pooled)))
    return "banded" if gain > 2.0 * paired_se(banded, pooled, n_boot, seed) else "pooled"


def guard(scores, selected, n_boot=1000, seed=0, k_trigger=3.0, k_ok=1.0):
    """Hidden-feature guard. ``scores`` (n_sim, n_h, n_z, n_band): per-simulation CV scores per smoothing (index 0 = raw,
    increasing smoothing with index) per z cell and k band. A (z, band) cell triggers if raw beats ``selected`` by more
    than ``k_trigger`` paired SE; a triggered z cell (ALL its modes) takes the largest smoothing whose score is within
    ``k_ok`` paired SE of raw in every triggering band (raw if none). Returns the smoothing index per z cell."""
    s = np.asarray(scores, float)
    n_h, n_z, n_band = s.shape[1:]
    out = np.full(n_z, int(selected))
    for z in range(n_z):
        trig = [b for b in range(n_band)
                if np.mean(s[:, 0, z, b] - s[:, selected, z, b])
                > k_trigger * paired_se(s[:, 0, z, b], s[:, selected, z, b], n_boot, seed)]
        if not trig:
            continue
        ok = [h for h in range(n_h)
              if all(np.mean(s[:, 0, z, b] - s[:, h, z, b]) <= k_ok * paired_se(s[:, 0, z, b], s[:, h, z, b], n_boot, seed)
                     for b in trig)]
        out[z] = max(ok) if ok else 0
    return out


def bracket_modes(k_com_hmpc, z, k_lo, k_hi, lo_unit, hi_unit):
    """(first, last) 1-based mode indices that bracket every k in [k_lo, k_hi] at z for every theta in the sampling box
    [lo_unit, hi_unit]: k_skm,n = n k_skm,1 and k_skm,1 spans [kK_min / K, k1_max] over the box."""
    k1_max, kK_min = kbounds_over_box(k_com_hmpc, z, lo_unit, hi_unit)
    K = len(k_com_hmpc)
    k1_min = kK_min / K
    first = max(1, int(np.floor(k_lo / k1_max)))
    last = min(K, int(np.ceil(k_hi / k1_min)))
    return first, last
