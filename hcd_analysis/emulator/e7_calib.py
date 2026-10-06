"""Gate E amendment A1 rev 1 section 6 (E7): the emulator-error calibration statistics, the registered two-sided FAIL
rule with v1's multiplicity rule, and the parametric null and power.

Per held-out simulation (its leg vectors averaged first): Q = r^T C_emu^+ r / rank(C_emu) (residual space; pseudo-
inverse on eigenvalues >= 1e-8 lambda_max) and T = delta^T V^-1 delta / 3 with delta = [F^-1 J^T C_total^-1 r] over (n_s,
A_p, tau0_amp) and V the same block of F^-1 J^T C_total^-1 C_emu C_total^-1 J F^-1 (uncentered, standardized). A
statistic FAILS only if its mean over simulations is outside the materiality band AND its simulation-bootstrap
percentile interval at the family-wise level (two-sided 0.05 / n_tests) excludes 1. Null: r_s ~ a_s N(0, C_emu,s), ln a_s
~ N(0, sigma_ln^2) normalized to mean square one, residual draws Gaussian or Student-t 5 dof (unit variance);
power at true-to-modelled
variance scales."""
from __future__ import annotations

import numpy as np


def _pinv_parts(C, rel=1e-8):
    w, U = np.linalg.eigh(0.5 * (np.asarray(C, float) + np.asarray(C, float).T))
    keep = w >= rel * w.max()
    return U[:, keep], w[keep]


def Q_one(r, C_emu):
    U, w = _pinv_parts(C_emu)
    x = U.T @ np.asarray(r, float)
    return float(x ** 2 @ (1.0 / w) / w.size)


def T_one(r, J, C_total, C_emu, P, idx):
    """(T, q per parameter) for one residual vector."""
    J = np.asarray(J, float)
    CiJ = np.linalg.solve(np.asarray(C_total, float), J)
    F = J.T @ CiJ + np.asarray(P, float)
    Fi = np.linalg.inv(F)
    L = Fi @ CiJ.T                                              # parameters per unit residual
    d = (L @ np.asarray(r, float))[idx]
    V = (L @ np.asarray(C_emu, float) @ L.T)[np.ix_(idx, idx)]
    return float(d @ np.linalg.solve(V, d) / len(idx)), d ** 2 / np.diag(V)


def bootstrap_interval(values, alpha, n_boot=10000, seed=0):
    v = np.asarray(values, float)
    idx = np.random.default_rng(seed).integers(0, v.size, (n_boot, v.size))
    means = v[idx].mean(axis=1)
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def fail(values, band, *, n_tests, n_boot=10000, seed=0, alpha=0.05):
    """The registered rule: {'mean', 'interval', 'fail': None | 'low' | 'high'}."""
    m = float(np.mean(values))
    lo, hi = bootstrap_interval(values, alpha / n_tests, n_boot, seed)
    out = None
    if m < band[0] and hi < 1.0:
        out = "low"
    elif m > band[1] and lo > 1.0:
        out = "high"
    return dict(mean=m, interval=(lo, hi), fail=out)


def amplitudes(rng, n, sigma_ln):
    """Per-simulation amplitude factors a with E[a^2] = 1: ln a ~ N(0, sigma_ln^2), divided by the population
    normalization sqrt(exp(2 sigma_ln^2)) (not the sample's)."""
    return np.exp(sigma_ln * rng.standard_normal(n)) / np.sqrt(np.exp(2 * sigma_ln ** 2))


def unit_draws(rng, shape, tails=None):
    """Unit-variance draws: standard normal, or Student-t with ``tails`` dof scaled to unit variance (the registered
    heavy-tail variant of the residual draws)."""
    if tails is None:
        return rng.standard_normal(shape)
    return rng.standard_t(tails, shape) / np.sqrt(tails / (tails - 2.0))


def null_and_power(sims, *, P, idx, sigma_ln, scales=(0.5, 0.67, 1.0, 1.5, 2.0), n_rep=1000, n_tests=8, seed=0,
                   n_boot=2000, bands=((0.8, 1.25), (0.67, 1.5)), tails=None):
    """For each true-to-modelled variance scale: the distribution of the arm's (Q, T) under draws r ~ sqrt(scale) a_s
    N(0, C_emu,s) (``n_vec`` leg vectors per simulation, averaged), and P(FAIL) per statistic and for either (the arm's
    family-wise false-fail rate at scale 1, the power elsewhere). ``sims``: dicts J, C_emu ('Cemu'), C_total ('Ctot')
    and optionally n_vec."""
    rng = np.random.default_rng(seed)
    pre = []
    for s in sims:
        U, w = _pinv_parts(s["Cemu"])
        root = U * np.sqrt(w)
        J = np.asarray(s["J"], float)
        CiJ = np.linalg.solve(np.asarray(s["Ctot"], float), J)
        Fi = np.linalg.inv(J.T @ CiJ + np.asarray(P, float))
        L = (Fi @ CiJ.T)[idx]
        V = L @ np.asarray(s["Cemu"], float) @ L.T
        pre.append((root, U, w, L, np.linalg.inv(V), int(s.get("n_vec", 1))))
    out = {}
    for sc in scales:
        fails = {"Q": 0, "T": 0, "any": 0}
        Qm, Tm = [], []
        for rep in range(n_rep):
            a = amplitudes(rng, len(pre), sigma_ln)
            Qs, Ts = np.zeros(len(pre)), np.zeros(len(pre))
            for i, (root, U, w, L, Vi, nv) in enumerate(pre):
                x = unit_draws(rng, (nv, w.size), tails)
                r = np.sqrt(sc) * a[i] * x @ root.T                         # (nv, N)
                Qs[i] = np.mean(((r @ U) ** 2 @ (1.0 / w)) / w.size)
                d = r @ L.T
                Ts[i] = np.mean(np.einsum("vi,ij,vj->v", d, Vi, d) / len(idx))
            fq = fail(Qs, bands[0], n_tests=n_tests, n_boot=n_boot, seed=rep)["fail"]
            ft = fail(Ts, bands[1], n_tests=n_tests, n_boot=n_boot, seed=rep)["fail"]
            fails["Q"] += fq is not None
            fails["T"] += ft is not None
            fails["any"] += (fq is not None) or (ft is not None)
            Qm.append(Qs.mean()); Tm.append(Ts.mean())
        out[sc] = dict(p_fail_Q=fails["Q"] / n_rep, p_fail_T=fails["T"] / n_rep, p_fail_any=fails["any"] / n_rep,
                       Q_quantiles=np.percentile(Qm, [0.3125, 2.5, 50, 97.5, 99.6875]).tolist(),
                       T_quantiles=np.percentile(Tm, [0.3125, 2.5, 50, 97.5, 99.6875]).tolist())
    return out
