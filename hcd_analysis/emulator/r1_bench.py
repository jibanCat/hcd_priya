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


FACTORIAL = dict(P=("P", "P"), P15=("P15", "P15"), M=("M", "M"), M10=("M10", "M10"), MF=("MF", "MF"), T1=("T1", "T1"),
                 MQ=("M", "MF"), MD=("MF", "M"))      # R1 T3 prereg rev 1 section 4: (quadratic term, log-det term)


def leg_parts(ctx, leg, spec):
    """(p, args, which) -> (model P on the leg's kept bins, C_total on them) for T3 representation ``which`` in
    {'P', 'P15', 'M', 'M10', 'MF', 'T1'} with the production algebra (T3 off-diagonal, diagonal max with T1, amplitude
    args['Pamp']). ``args``: model, pf (normalizer stats), rho (T1 per leg z, pooled), UP/wP (P factor on the leg's bins),
    UP15/wP15, UM/wM (M factor on the mode set, rank 15), UM10/wM10, UMF (M rank 15 bound at theta_ref, on the leg's
    bins: only the T3 binding frozen), Pamp (T3 amplitude vector), d (the mock). ``spec``: the M binding (z_cells, iz,
    k_bins, Mi, rows, n_leg) on the leg's T3 bins."""
    from . import fisher_kit as FK
    from . import forward as FW
    zg = np.asarray(ctx.z_global)
    sel = jnp.asarray(np.array([int(np.argmin(np.abs(zg - z))) for z in np.asarray(leg.z)]))
    kr = jnp.asarray(FK.kept(leg))
    Cd = jnp.asarray(leg.C_data)
    core = ctx.dla_core_leg[leg.name]
    centres = ctx.alpha_centres

    def parts(p, a, which):
        th, tau0, alpha = FK.forward_inputs(ctx, p)
        out = FW.predict_leg(a["model"], th, tau0[sel], alpha[sel], leg=leg, k_com=ctx.k_com_hmpc, pf_stats=a["pf"],
                             dla_core=core, mf=ctx.mf, t1=(a["rho"], centres), cemu_inflate=ctx.cemu_inflate)
        t1v = jnp.diag(out.C_total) - jnp.diag(Cd)
        if which == "T1":
            C = Cd + jnp.diag(t1v)
        else:
            if which in ("M", "M10"):
                um, wm = (a["UM"], a["wM"]) if which == "M" else (a["UM10"], a["wM10"])
                U, w = mode_aligned_factor(th, ctx.k_com_hmpc, spec["z_cells"], spec["iz"], spec["k_bins"], spec["Mi"],
                                           um, spec["rows"], spec["n_leg"]), wm
            elif which == "MF":
                U, w = a["UMF"], a["wM"]
            elif which == "P15":
                U, w = a["UP15"], a["wP15"]
            else:
                U, w = a["UP"], a["wP"]
            C = FW.assemble_cov(Cd, t1v, jnp.zeros_like(t1v), t3=(U, w), P_fid=jnp.nan_to_num(a["Pamp"]))
        return out.P_model[kr], C[jnp.ix_(kr, kr)]
    return parts


def log_prior(ctx):
    """The production prior's log density up to a constant on the HCD sites (theta9, tau0_amp, dtau0 are uniform:
    handled by the MAP bounds): Gaussian at the production centres and widths (TruncatedNormal on alpha_lls/subdla,
    whose truncation at 0 is a bound)."""
    from . import fisher_kit as FK
    P = FK.prior_precision(ctx)
    p0 = FK.p_centre(ctx, np.full(9, 0.5))
    m = np.zeros(len(p0), bool); m[11:] = True
    Pm, c = jnp.asarray(np.diag(P)[m]), jnp.asarray(p0[m])
    idx = jnp.asarray(np.where(m)[0])
    return lambda p: -0.5 * jnp.sum(Pm * (p[idx] - c) ** 2)


def make_nlp(ctx, leg, spec, variant):
    """(p, args) -> the negative log posterior of the mock args['d'] under ``variant`` (FACTORIAL: quadratic term from
    the first representation, log-determinant from the second)."""
    from . import fisher_kit as FK
    parts = leg_parts(ctx, leg, spec)
    lp = log_prior(ctx)
    kr = jnp.asarray(FK.kept(leg))
    cq, cd = FACTORIAL[variant]

    def nlp(p, a):
        mu, Cq = parts(p, a, cq)
        Cl = Cq if cd == cq else parts(p, a, cd)[1]
        return -factorial_loglik(a["d"][kr] - mu, Cq, Cl) - lp(p)
    return nlp


def force_terms(ctx, leg, spec, variant, p, args):
    """The log-likelihood gradient at ``p`` split into the mean term J^T C^-1 r, the covariance-quadratic term
    1/2 r^T C^-1 dC C^-1 r and the log-determinant term -1/2 tr(C^-1 dC) (variant's factorial covariances)."""
    from . import fisher_kit as FK
    parts = leg_parts(ctx, leg, spec)
    kr = jnp.asarray(FK.kept(leg))
    cq, cd = FACTORIAL[variant]
    d = args["d"][kr]

    def quad(pm, pc):
        mu = parts(pm, args, cq)[0]
        C = parts(pc, args, cq)[1]
        r = d - mu
        return -0.5 * r @ jnp.linalg.solve(C, r)

    def logdet(pc):
        return -0.5 * jnp.linalg.slogdet(parts(pc, args, cd)[1])[1]
    p = jnp.asarray(p)
    return dict(mean=jax.grad(quad, argnums=0)(p, p), cov_quad=jax.grad(quad, argnums=1)(p, p),
                logdet=jax.grad(logdet)(p))


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



def trunc_pit(truth, mean, sigma, lo, hi):
    """(u, pull): the CDF at the truth of a Gaussian (mean, sigma) truncated to [lo, hi], and pull = Phi^-1(u) (defined at
    prior bounds; the R1 coverage rule uses u)."""
    from scipy.stats import norm
    a, b = norm.cdf((lo - mean) / sigma), norm.cdf((hi - mean) / sigma)
    u = (norm.cdf((truth - mean) / sigma) - a) / max(b - a, 1e-300)
    u = float(np.clip(u, 1e-12, 1 - 1e-12))
    return u, float(norm.ppf(u))


def covered(u):
    """(68%, 95%) central-interval coverage indicators from PIT values."""
    u = np.asarray(u, float)
    return np.abs(u - 0.5) <= 0.341345, np.abs(u - 0.5) <= 0.475


def _gpd_fit(x):
    """Zhang & Stephens (2009) estimate of the generalized Pareto shape k and scale for exceedances x > 0 (as in
    Vehtari et al. PSIS), with the weakly informative prior on k."""
    x = np.sort(np.asarray(x, float))
    n = x.size
    m = 30 + int(np.sqrt(n))
    b = 1.0 - np.sqrt(m / (np.arange(1, m + 1) - 0.5))
    b = b / (3.0 * x[int(n / 4 + 0.5) - 1]) + 1.0 / x[-1]
    k = np.mean(np.log1p(-b[:, None] * x[None, :]), axis=1)
    L = n * (np.log(-b / k) - k - 1.0)
    w = 1.0 / np.sum(np.exp(L[None, :] - L[:, None]), axis=1)
    bh = np.sum(b * w)
    kh = np.mean(np.log1p(-bh * x))
    sig = -kh / bh
    kh = (n * kh + 10 * 0.5) / (n + 10)                       # the PSIS weakly informative prior toward 0.5
    return kh, sig


def psis(logw):
    """Pareto-smoothed importance weights (normalized) and the Pareto k-hat diagnostic (Vehtari et al. 2015/2022)."""
    lw = np.asarray(logw, float) - np.max(logw)
    S = lw.size
    M = int(min(0.2 * S, 3 * np.sqrt(S)))
    order = np.argsort(lw)
    cut = lw[order[S - M - 1]]
    tail = order[S - M:]
    x = np.exp(lw[tail]) - np.exp(cut)
    k, sig = _gpd_fit(x)
    if np.isfinite(k) and sig > 0:
        q = (np.arange(1, M + 1) - 0.5) / M
        sm = np.exp(cut) + (sig / k) * ((1 - q) ** (-k) - 1) if abs(k) > 1e-12 else np.exp(cut) - sig * np.log(1 - q)
        lw = lw.copy()
        lw[tail] = np.log(np.minimum(sm, 1.0))                 # sorted tail replaced by GPD quantiles, truncated at max
    w = np.exp(lw - np.max(lw))
    return w / w.sum(), float(k)


def phase_stat(u, weights):
    """Weighted mean of cos(2 pi u) over bins with fractional mode index u: 1 when every bin sits on a mode (where the
    mode-locked bindings kink), -1 halfway between modes."""
    w = np.asarray(weights, float)
    return float(np.sum(w * np.cos(2.0 * np.pi * np.asarray(u, float))) / np.sum(w))
