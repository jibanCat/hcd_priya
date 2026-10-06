"""Gate E amendment A1 rev 1 sections 1 (A1-A4) and 5: the covariance-force audit kit. For a covariance C(p) of a
leg's kept bins (the production algebra, every term present), the audited term is isolated in the DERIVATIVE: the
log-determinant pull g_ld = -1/2 d log|C|, the expected-likelihood gradient at a fixed second moment S (the net pull
-1/2 tr(C^-1 dC) + 1/2 tr(C^-1 dC C^-1 S)), the exact mode crossings along a parameter path, the covariance-information
ratio, and the nonlinear (Newton) maximizer of L_C(p0 + d) - 1/2 d^T F d."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq

from . import forward as FW
from . import fisher_kit as FK
from . import kcoord as KC


def cov_fn(ctx, leg, nuis=None, t2=None):
    """p -> C_total on the leg's kept bins with ctx's T1 and T3 (and ``t2`` when given), production algebra."""
    zg = np.asarray(ctx.z_global)
    sel = jnp.asarray(np.array([int(np.argmin(np.abs(zg - z))) for z in np.asarray(leg.z)]))
    kr = jnp.asarray(FK.kept(leg))
    nu = FK.nuis_centre(ctx, leg) if nuis is None else nuis
    t1 = ((ctx.rho_zb_per_leg[leg.name], ctx.alpha_centres)
          if ctx.rho_zb_per_leg and leg.name in ctx.rho_zb_per_leg else None)
    t3 = ctx.t3_per_leg.get(leg.name) if ctx.t3_per_leg else None

    def C(p):
        th, tau0, alpha = FK.forward_inputs(ctx, p)
        out = FW.predict_leg(ctx.model, th, tau0[sel], alpha[sel], leg=leg, k_com=ctx.k_com_hmpc,
                             pf_stats=ctx.pf_stats, dla_core=ctx.dla_core_leg[leg.name], mf=ctx.mf, nuis=nu,
                             t1=t1, t2=t2, t3=t3, cemu_inflate=ctx.cemu_inflate)
        return out.C_total[jnp.ix_(kr, kr)]
    return C


def logdet_half(C_of):
    """p -> -1/2 log|C(p)| (its gradient is the unbalanced log-determinant pull g_ld)."""
    return lambda p: -0.5 * jnp.linalg.slogdet(C_of(p))[1]


def expected_loglik(C_of):
    """(p, S) -> L_C = -1/2 (log|C(p)| + tr(C(p)^-1 S)); its gradient at fixed S is the net covariance pull."""
    def L(p, S):
        C = C_of(p)
        return -0.5 * (jnp.linalg.slogdet(C)[1] + jnp.trace(jnp.linalg.solve(C, S)))
    return L


def crossings(k_com, z_cells, iz, k_bins, centre, axis, lo, hi, xtol=1e-14):
    """[(x, bin)]: the values x of parameter ``axis`` in (lo, hi), others at ``centre``, where bin b's fractional mode
    index u = k_b / k_skm,1(z_b, theta) is an integer (a mode crossing of the binding), found exactly."""
    def k1(x, c):
        th = np.array(centre, float); th[axis] = x
        return float(KC.k_skm_from_theta9(k_com, float(z_cells[c]), jnp.asarray(th))[0])
    out = []
    for b, (kb, c) in enumerate(zip(k_bins, iz)):
        ua, ub = kb / k1(lo, c), kb / k1(hi, c)
        for m in range(int(np.ceil(min(ua, ub))), int(np.floor(max(ua, ub))) + 1):
            f = lambda x: kb / k1(x, c) - m
            if f(lo) == 0.0 or f(hi) == 0.0:
                continue
            try:
                out.append((brentq(f, lo, hi, xtol=xtol), b))
            except ValueError:
                continue
    return out


def info_ratio(C, dC, J):
    """1/2 tr(C^-1 C_,i C^-1 C_,i) / (J^T C^-1 J)_ii per parameter i (``dC`` the list of dC/dp_i)."""
    Ci = np.linalg.inv(np.asarray(C, float))
    JCJ = np.asarray(J).T @ Ci @ np.asarray(J)
    return np.array([0.5 * np.trace(Ci @ D @ Ci @ D) / JCJ[i, i] for i, D in enumerate(dC)])


def newton_shift(LC_of_d, F, d0, sigma, n_iter=20, trust=2.0, metric=None):
    """The maximizer over d of LC_of_d(d) - 1/2 d^T F d (Newton from d0, steps clipped to the trust region of +-trust
    marginal sigma). ``metric``: Fisher scoring with the curvature F + metric (the covariance information matrix)
    instead of the exact Hessian."""
    F = np.asarray(F, float)
    obj = lambda d: LC_of_d(d) - 0.5 * d @ jnp.asarray(F) @ d
    g_fn = jax.grad(obj)
    h_fn = None if metric is not None else jax.hessian(obj)
    d = np.asarray(d0, float)
    lim = trust * np.asarray(sigma, float)
    for _ in range(n_iter):
        g = np.asarray(g_fn(jnp.asarray(d)))
        H = -(F + np.asarray(metric, float)) if metric is not None else np.asarray(h_fn(jnp.asarray(d)))
        step = -np.linalg.solve(H, g)
        d_new = np.clip(d + step, -lim, lim)
        if np.max(np.abs(d_new - d)) < 1e-12 * max(1.0, np.max(np.abs(d))):
            d = d_new
            break
        d = d_new
    return d
