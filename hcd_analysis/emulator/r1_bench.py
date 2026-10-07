"""Research round R1 (PI ruling 2026-10-07, PU-0077), T3 benchmark kit. A RESEARCH module: nothing here is used by the
production forward. Provides the factorial likelihood (quadratic term from one covariance, log-determinant from another,
so the mode-aligned T3's quadratic gain and its log-det force can be separated), the mode-aligned T3 factor bound to a
leg's bins at the query theta (jax), a bounded MAP with Laplace widths, and the benchmark summaries (pulls, coverage,
paired simulation-level bootstrap)."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from . import kcoord as KC


def factorial_loglik(r, C_quad, C_logdet):
    """-1/2 r^T C_quad^-1 r - 1/2 log|C_logdet| - N/2 log 2 pi (equal matrices: the Gaussian log density)."""
    Lq = jnp.linalg.cholesky(C_quad)
    x = jax.scipy.linalg.solve_triangular(Lq, r, lower=True)
    Ld = jnp.linalg.cholesky(C_logdet)
    return -0.5 * x @ x - jnp.sum(jnp.log(jnp.diag(Ld))) - 0.5 * r.shape[0] * jnp.log(2.0 * jnp.pi)


def mode_aligned_factor(theta9, k_com, z_cells, iz, k_bins, Mi, U_modes, rows, n_leg):
    """The M representation's factor on the leg's bins at the query theta: rows ``rows`` of an (n_leg, r) matrix =
    W(theta)[:, Mi] U_modes, W the production log-k binding of the (z, mode) vector at the bins (z cell ``iz``); other
    rows 0. Differentiable in theta (through k_skm,1)."""
    K = len(k_com)
    U_modes = jnp.asarray(U_modes)
    r = U_modes.shape[1]
    full = jnp.zeros((len(z_cells) * K, r)).at[jnp.asarray(Mi)].set(U_modes)       # (n_z K, r), zero off the mode set
    out = jnp.zeros((n_leg, r))
    for c in np.unique(iz):
        sel = np.where(iz == c)[0]
        b = KC.bind(KC.kgrid(k_com, float(z_cells[c]), theta9), jnp.asarray(np.asarray(k_bins)[sel]))
        j = b.j.astype(jnp.int32)
        blk = full[c * K:(c + 1) * K]
        val = (1.0 - b.t_log)[:, None] * blk[j - 1] + b.t_log[:, None] * blk[j]
        out = out.at[jnp.asarray(np.asarray(rows)[sel])].set(val)
    return out


def truth_rows(d, sim, alpha_rung, leg_z, tol=1e-3):
    """Cache rows of simulation ``sim`` at mean-flux ladder factor ``alpha_rung`` for each z in ``leg_z`` (one each)."""
    from .data import tau0_ladder_factor
    names = np.asarray(d["sim_name"]).astype(str)
    zg = np.asarray(d["z_grid"], float)
    a = tau0_ladder_factor(np.asarray(d["tau0"], float), zg)
    rows = []
    for z in np.asarray(leg_z, float):
        idx = np.where((names == sim) & (np.abs(zg - z) < 0.05) & (np.abs(a - alpha_rung) < tol))[0]
        if idx.size != 1:
            raise ValueError(f"{sim}: {idx.size} rows at z {z}, rung {alpha_rung}")
        rows.append(int(idx[0]))
    return np.asarray(rows)


def truth_on_leg(d, rows, leg, mf, alpha_leg, core_data):
    """The truth model P1D on every bin of ``leg`` (R1 T3 benchmark): per z, the truth row's per-class P_filt on its own
    stored grid times the MF factor at the truth's (theta, z, tau0), interpolated linearly in k onto the data k (truth
    side, independent of kcoord), combined with the production HCD algebra at ``alpha_leg`` (n_z, 3) and the DLA forward
    fraction, the physical-k DLA core ``core_data`` (N,) in the DLA amplitude. ``rows``: one cache row per leg z."""
    from .data import Z_LIMITS
    k = np.asarray(leg.k, float)
    z_idx = np.asarray(leg.z_idx)
    out = np.zeros(k.size)
    dff = float(getattr(leg, "dla_forward_frac", 1.0))
    scale = np.array([1.0, 1.0, dff])
    for iz, z in enumerate(np.asarray(leg.z, float)):
        sel = np.where(z_idx == iz)[0]
        if sel.size == 0:
            continue
        row = rows[iz]
        P = np.asarray(d["P_filt"], float)[row]
        if mf is not None:
            th = np.asarray(d["params_unit"], float)[row]
            z_unit = (z - Z_LIMITS[0]) / (Z_LIMITS[1] - Z_LIMITS[0])
            x = jnp.concatenate([jnp.asarray(th), jnp.asarray([z_unit])])
            P = P * np.exp(np.asarray(mf(x, float(np.asarray(d["tau0"], float)[row]))))
        kr = np.asarray(d["kfkms"], float)[row]
        Pc = np.stack([np.interp(k[sel], kr, P[c]) for c in range(4)])
        a = np.asarray(alpha_leg, float)[iz] * scale
        out[sel] = (Pc[0] + a[0] * (Pc[1] - Pc[0]) + a[1] * (Pc[2] - Pc[0])
                    + a[2] * (Pc[3] + np.asarray(core_data, float)[sel] - Pc[0]))
    return out


def map_laplace(neg_log_post, p0, bounds, starts=(), maxiter=2000, ftol=1e-12, gtol=1e-8):
    """Bounded MAP (L-BFGS-B, jax value and gradient) from ``p0`` and any extra ``starts`` (the best optimum kept), the
    Laplace widths sqrt(diag H^-1) from the jax Hessian at the MAP, and flags: converged, at_bound per parameter, and the
    spread of the optima over the starts."""
    vg = jax.jit(jax.value_and_grad(lambda p: neg_log_post(p)))

    def fg(p):
        v, g = vg(jnp.asarray(p, float))
        return float(v), np.asarray(g, float)
    best, optima = None, []
    for s in (p0,) + tuple(starts):
        res = minimize(fg, np.asarray(s, float), jac=True, method="L-BFGS-B", bounds=bounds,
                       options=dict(maxiter=maxiter, ftol=ftol, gtol=gtol))
        optima.append(res.x)
        if best is None or res.fun < best.fun:
            best = res
    H = np.asarray(jax.hessian(lambda p: neg_log_post(p))(jnp.asarray(best.x, float)))
    Hi = np.linalg.inv(0.5 * (H + H.T))
    lo = np.array([b[0] if b[0] is not None else -np.inf for b in bounds])
    hi = np.array([b[1] if b[1] is not None else np.inf for b in bounds])
    span = np.where(np.isfinite(hi - lo), hi - lo, 1.0)
    at_bound = (np.abs(best.x - lo) < 1e-6 * span) | (np.abs(best.x - hi) < 1e-6 * span)
    return dict(map=best.x, sigma=np.sqrt(np.clip(np.diag(Hi), 0, None)), cov=Hi, H=H, fun=float(best.fun),
                converged=bool(best.success), at_bound=at_bound,
                start_spread=(np.max(np.abs(np.array(optima) - best.x), axis=0) if len(optima) > 1 else None))


def pull_summary(pulls, groups, n_boot=2000, seed=0):
    """Mean and RMS pull, 68% and 95% coverage (|pull| <= 1, 1.96) per parameter, with simulation-level bootstrap
    intervals (``groups``: the truth of each mock; all mocks of a truth move together)."""
    pulls = np.asarray(pulls, float)
    g = np.asarray(groups)
    ug = np.unique(g)
    idx = [np.where(g == u)[0] for u in ug]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = np.concatenate([idx[i] for i in rng.integers(0, ug.size, ug.size)])
        boots.append(pulls[pick].mean(axis=0))
    boots = np.array(boots)
    return dict(mean=pulls.mean(axis=0), rms=np.sqrt((pulls ** 2).mean(axis=0)),
                cov68=(np.abs(pulls) <= 1.0).mean(axis=0), cov95=(np.abs(pulls) <= 1.96).mean(axis=0),
                mean_ci=np.percentile(boots, [2.5, 97.5], axis=0))


def paired_ci(a, b, groups, n_boot=2000, seed=0, stat=np.mean):
    """95% simulation-level bootstrap interval of stat(b - a) over paired per-mock values (whole truths resampled)."""
    d = np.asarray(b, float) - np.asarray(a, float)
    g = np.asarray(groups)
    ug = np.unique(g)
    idx = [np.where(g == u)[0] for u in ug]
    rng = np.random.default_rng(seed)
    vals = [stat(d[np.concatenate([idx[i] for i in rng.integers(0, ug.size, ug.size)])]) for _ in range(n_boot)]
    return tuple(np.percentile(vals, [2.5, 97.5]))
