"""Building blocks of the gate E emulator-error products (GATE_E_AMENDMENT_A1.md in the notes repository, registered
before any final product is built): simulation-level cross-validation folds, per-cell second moments of the
leave-one-simulation-out ensemble residuals, PSD-preserving smoothing along ln(mode) and z, the Gaussian predictive
score used for model selection, and the modes that bracket a leg's data bins over the whole sampling box.
Products are calibrated uncertainty tables: nothing here is a function of the cosmological or nuisance parameters."""
from __future__ import annotations

from typing import NamedTuple

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


def combined_logpdf_modes(r, P, coef, rho, mask):
    """Per-mode Gaussian log density of the production-combined fractional residual e = sum_c coef_c P_c r_c / P_obs
    under its predicted fractional variance; 0 outside ``mask`` (K,) or (..., K). r, P (..., C, K)."""
    var, P_obs = combined_variance(P, coef, rho)
    e = np.einsum("...c,...ck,...ck->...k", coef, P, r) / P_obs
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = -0.5 * (e ** 2 / var + np.log(2.0 * np.pi * var))
    return np.where(mask, lp, 0.0)


def combined_logpdf(r, P, coef, rho, mask):
    """``combined_logpdf_modes`` summed over the modes."""
    return np.sum(combined_logpdf_modes(r, P, coef, rho, mask), axis=-1)


def class_logpdf_modes(r, cov, mask, jitter=1e-10):
    """Per-mode 4 x 4 (cross-class) Gaussian log density of r (..., C, K) under cov (..., C, C, K) with the production
    jitter; 0 outside ``mask``. A non-positive-definite block inside the mask raises."""
    rr = np.moveaxis(np.asarray(r, float), -1, -2)                       # (..., K, C)
    cc = np.moveaxis(np.asarray(cov, float), -1, -3)                     # (..., K, C, C)
    C = rr.shape[-1]
    cc = cc + (jitter * np.einsum("...ii->...", cc) / C)[..., None, None] * np.eye(C)
    sign, logdet = np.linalg.slogdet(cc)
    m = np.broadcast_to(mask, sign.shape)
    if np.any((sign <= 0) & m):
        raise ValueError("non-positive-definite cross-class block inside the scored modes")
    sol = np.linalg.solve(np.where(m[..., None, None], cc, np.eye(C)), rr[..., None])[..., 0]
    lp = -0.5 * (np.sum(rr * sol, axis=-1) + logdet + C * np.log(2.0 * np.pi))
    return np.where(m, lp, 0.0)


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


class T1Data(NamedTuple):
    """Rows (held-out ensemble residuals) entering the T1 selection: r, P (R, 4, K) residuals and truth per-class P_filt;
    coef (R, 4) production class coefficients; alpha (R,) tau0-ladder factor, band (R,) tau0 band, centres (B,) band
    centres; zc (R,) z cell index, sim (R,) simulation name; z_cells (n_z,); mask (n_z, K) scored modes; kband (n_z, K)
    k band of each mode (box-centre physical k)."""
    r: np.ndarray
    P: np.ndarray
    coef: np.ndarray
    alpha: np.ndarray
    band: np.ndarray
    centres: np.ndarray
    zc: np.ndarray
    sim: np.ndarray
    z_cells: np.ndarray
    mask: np.ndarray
    kband: np.ndarray


def t1_raw_cells(T, rows, pooled):
    """Raw (unsmoothed) second moments from ``rows``: (n_z, B, 4, 4, K), B = 1 when pooled over tau0."""
    rows = np.asarray(rows)
    B = 1 if pooled else T.centres.size
    band = np.zeros(rows.size, int) if pooled else np.asarray(T.band)[rows]
    rho, n = second_moment_cells(T.r[rows], np.asarray(T.zc)[rows] * B + band, T.z_cells.size * B)
    if np.any(n == 0):
        raise ValueError("empty T1 cell")
    return rho.reshape((T.z_cells.size, B) + rho.shape[1:])


def t1_smooth(raw, h_of_z, s_z, z_cells):
    """Mode smoothing with a per-z-cell width, then smoothing along z (global width)."""
    out = np.stack([smooth_modes(raw[iz], h) for iz, h in enumerate(h_of_z)])
    return smooth_z(out, z_cells, s_z)


def t1_rho_rows(T, rows, rho, pooled):
    """The deployed T1 block (R, 4, 4, K) of each row: its z cell, tau0-interpolated at its alpha (``rho_interp_alpha``)."""
    rows = np.asarray(rows)
    out = np.empty((rows.size,) + rho.shape[2:])
    centres = np.array([1.0]) if pooled else T.centres
    zc = np.asarray(T.zc)[rows]
    for iz in np.unique(zc):
        sel = zc == iz
        out[sel] = rho_interp_alpha(np.moveaxis(rho[iz], 0, -1), centres, np.asarray(T.alpha)[rows][sel])
    return out


def _smooth_rank(cands, idx):
    order = sorted(idx, key=lambda i: (cands[i][0], cands[i][1]))
    return np.array([order.index(i) for i in idx])


def t1_cv(T, cands, n_folds=10):
    """Simulation-level CV of the T1 candidates ``cands`` [(h, s_z, pooled), ...]: per held-out simulation (sorted
    names), the summed production-combined score (``main``), its split per (z cell, k band) (``cell``), and the 4 x 4
    per-class score (``per_class``)."""
    folds = cv_folds(T.sim, n_folds)
    sims = sorted(folds)
    sim_idx = {s: i for i, s in enumerate(sims)}
    n_z, n_kb = T.z_cells.size, int(np.max(T.kband)) + 1
    main = np.zeros((len(sims), len(cands)))
    cell = np.zeros((len(sims), len(cands), n_z, n_kb))
    per_class = np.zeros((len(sims), len(cands)))
    fold_of_row = np.array([folds[s] for s in np.asarray(T.sim)])
    for f in range(n_folds):
        tr, te = np.where(fold_of_row != f)[0], np.where(fold_of_row == f)[0]
        if te.size == 0:
            continue
        si = np.array([sim_idx[s] for s in np.asarray(T.sim)[te]], dtype=int)
        raws = {p: t1_raw_cells(T, tr, p) for p in {c[2] for c in cands}}
        mask = T.mask[np.asarray(T.zc)[te]]
        kb = T.kband[np.asarray(T.zc)[te]]
        for ci, (h, s_z, pooled) in enumerate(cands):
            rho = t1_smooth(raws[pooled], [h] * n_z, s_z, T.z_cells)
            rr = t1_rho_rows(T, te, rho, pooled)
            lp = combined_logpdf_modes(T.r[te], T.P[te], T.coef[te], rr, mask)
            np.add.at(main[:, ci], si, lp.sum(axis=1))
            zc = np.asarray(T.zc)[te]
            for b in range(n_kb):
                np.add.at(cell[:, ci, :, b], (si, zc), np.sum(np.where(kb == b, lp, 0.0), axis=1))
            np.add.at(per_class[:, ci], si, class_logpdf_modes(T.r[te], rr, mask).sum(axis=1))
    return dict(main=main, cell=cell, per_class=per_class, sims=sims)


def t1_select(cv, cands, n_boot=1000, seed=0):
    """Amendment A1 rev 1 section 1 rules on the CV scores: per tau0 option the (h, s_z) argmax (ties to more
    smoothing); pooled unless banded wins by > 2 paired SE (then ``tau0`` = 'banded' and the caller STOPS; the pooled
    choice is the working one); the whole-z-cell guard against raw (h = 0, s_z = 0) at 3 paired SE."""
    main = cv["main"]
    means = main.mean(axis=0)
    best = {}
    for pooled in sorted({c[2] for c in cands}):
        idx = [i for i, c in enumerate(cands) if c[2] == pooled]
        best[pooled] = idx[select_argmax(means[idx], _smooth_rank(cands, idx))]
    tau0 = choose_tau0(main[:, best[False]], main[:, best[True]], n_boot, seed) if len(best) == 2 else \
        ("pooled" if True in best else "banded")
    chosen = best[True] if True in best else best[False]
    h_sel, s_sel, pooled = cands[chosen]
    raw_i = cands.index((0.0, 0.0, pooled))
    ladder = [raw_i] + sorted((i for i, c in enumerate(cands) if c[2] == pooled and c[1] == s_sel and i != raw_i),
                              key=lambda i: cands[i][0])
    g = guard(cv["cell"][:, ladder], ladder.index(chosen), n_boot, seed)
    h_of_z = [cands[ladder[i]][0] for i in g]
    return dict(chosen=chosen, tau0=tau0, best_pooled=best.get(True), best_banded=best.get(False),
                tau0_gain=(float(np.mean(main[:, best[False]] - main[:, best[True]])) if len(best) == 2 else None),
                tau0_gain_se=(paired_se(main[:, best[False]], main[:, best[True]], n_boot, seed) if len(best) == 2 else None),
                means=means.tolist(), h_of_z=h_of_z, guard_triggered=[int(iz) for iz in np.where(np.array(h_of_z) != h_sel)[0]])


def t1_build(T, rows, cands, sel):
    """The T1 product from ``rows`` with the selection ``sel``: (n_z, B, 4, 4, K)."""
    h, s_z, pooled = cands[sel["chosen"]]
    return t1_smooth(t1_raw_cells(T, rows, pooled), sel["h_of_z"], s_z, T.z_cells)


DLA_CORE_GRID = np.geomspace(4e-4, 0.1, 400)                  # s/km, amendment A1 rev 1 section 4


def dla_core_z(kfkms, core, z_rows, z, k_grid=DLA_CORE_GRID):
    """Mean over the cache rows with |z_row - z| < 0.05 of each row's DLA core interpolated linearly in k at the fixed
    physical ``k_grid`` from the row's OWN stored grid (finite modes only). A row not covering a grid point is excluded
    there; a point covered by no row is NaN."""
    sel = np.where(np.abs(np.asarray(z_rows, float) - z) < 0.05)[0]
    if sel.size == 0:
        raise ValueError(f"no cache row at z {z}")
    g = np.asarray(k_grid, float)
    vals = np.full((sel.size, g.size), np.nan)
    for i, r in enumerate(sel):
        kr, cr = np.asarray(kfkms[r], float), np.asarray(core[r], float)
        fin = np.isfinite(kr) & np.isfinite(cr)
        kr, cr = kr[fin], cr[fin]
        inside = (g >= kr[0]) & (g <= kr[-1])
        vals[i, inside] = np.interp(g[inside], kr, cr)
    with np.errstate(invalid="ignore"):
        cnt = np.sum(np.isfinite(vals), axis=0)
        out = np.nansum(vals, axis=0) / np.where(cnt > 0, cnt, 1)
    return np.where(cnt > 0, out, np.nan)


def dla_core_leg(kfkms, core, z_rows, leg_z, k_grid=DLA_CORE_GRID):
    """One core per leg: the mean over the leg's z of ``dla_core_z`` (NaN where any z is uncovered)."""
    return np.mean([dla_core_z(kfkms, core, z_rows, float(z), k_grid) for z in leg_z], axis=0)


def dla_core_at(k_data, k_grid, core_grid):
    """The core at data k: linear interpolation in ln k on the fixed grid; refuses k outside the finite part."""
    g, c = np.asarray(k_grid, float), np.asarray(core_grid, float)
    fin = np.isfinite(c)
    kd = np.asarray(k_data, float)
    if np.any(kd < g[fin][0]) or np.any(kd > g[fin][-1]) or not np.all(fin[(g >= kd.min()) & (g <= kd.max())]):
        raise ValueError("data k outside the finite DLA-core grid")
    return np.interp(np.log(kd), np.log(g[fin]), c[fin])


def bracket_modes(k_com_hmpc, z, k_lo, k_hi, lo_unit, hi_unit):
    """(first, last) 1-based mode indices that bracket every k in [k_lo, k_hi] at z for every theta in the sampling box
    [lo_unit, hi_unit]: k_skm,n = n k_skm,1 and k_skm,1 spans [kK_min / K, k1_max] over the box."""
    k1_max, kK_min = kbounds_over_box(k_com_hmpc, z, lo_unit, hi_unit)
    K = len(k_com_hmpc)
    k1_min = kK_min / K
    first = max(1, int(np.floor(k_lo / k1_max)))
    last = min(K, int(np.ceil(k_hi / k1_min)))
    return first, last
