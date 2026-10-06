#!/usr/bin/env python3
"""Gate E amendment A1 rev 1 sections 1 (A1-A4) and 5, step 5: the covariance-force audit of one leg in the full
production covariance (T1 + T3 with the production algebra; T2 off until amendment A2; re-run with T2). The audited
term is isolated in the derivative: with C_data and the fixed-physical-k T3 constant (and T2 off), every derivative of C
is T1's (within the max algebra).

A1 scans: the (hub, omegamh2) grid (n x n, other parameters at the sampling-box centre, tau0 and HCD at their prior
centres) and 201-point paths along every parameter (theta9 over the box, tau0_amp and dtau0 over their prior ranges,
HCD sites over +-3 prior sigma): f = 1/2 log|C|, the unbalanced log-det pull F^-1 (-1/2 grad log|C|) in marginal sigma
of n_s, A_p, tau0_amp (A3 a), with J and F at each point. A2: every mode crossing of every kept bin along the hub and
omegamh2 paths through the centre, the jumps of the log-det and net (S = mean_s S_s) gradients from one-sided
directional derivatives at x +- 1e-6 of the box width (the jump vector has hub and omegamh2 components only, in the
ratio of d ln k_skm,1 / d theta at the bin's z), projected with the box-centre Fisher. A3 b: the net pull at each
simulation's theta and each of its 20 rung leg vectors, S_s = r_s r_s^T + C_data (T2 off), per-simulation means first;
A3 c: the Fisher-scoring maximizer of L_C(theta_s + d) - 1/2 d^T F d at the central rung; A3 d: the covariance-
information ratio. A4: T1 at the data bins, selected over raw, along the hub and omegamh2 paths. Derivatives by jax
(T1 and T3 carry no stop-gradient; finite-difference agreement checked on a subset and reported). Write-once outputs.

ENV: PYTHONNOUSERSITE=1 PYTHONPATH=<repo> JAX_PLATFORMS=cpu /home/mfho/.conda/envs/emu-jax/bin/python3 \
     scripts/run_gate_e_cov_audit.py --leg DESI --t1 <..> --t1-raw <..> [--t3 <..>] --dla-core <..> --mf <..> \
     --eval-dir <..> --out-json <..> --out-npz <..>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import jax
import jax.numpy as jnp
import numpy as np

from hcd_analysis.emulator import cemu_build as CB
from hcd_analysis.emulator import closure_legb as CL
from hcd_analysis.emulator import cov_audit as CA
from hcd_analysis.emulator import data_likelihood as DL
from hcd_analysis.emulator import fisher_kit as FK
from hcd_analysis.emulator import kcoord as KC
from hcd_analysis.emulator.data import sampling_unit_bounds, tau0_ladder_factor
from hcd_analysis.emulator.prod_ensemble import production_member_paths

SURVEYS = {"DESI": dict(metals=True, with_eboss=False), "KS": dict(metals=False, with_eboss=False),
           "eBOSS": dict(metals=True, with_eboss=True)}
IDX = (0, 1, 9)                                   # n_s, A_p, tau0_amp
I_HUB, I_OMH2 = 5, 6
THRESH = dict(kink=0.05, net_mean=0.1, net_rms=0.2, net_max=0.5, logdet=0.3)     # A2/A3 (registered, analyst choice)


def _stage(msg):
    """Time-stamped progress on stderr (diagnostic only; the outputs do not depend on it)."""
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}", file=sys.stderr, flush=True)


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build(name, members, mf, dla, t1, t3):
    kw = SURVEYS[name]
    ctx, d = CL.build_legb_ctx(ensemble_ckpts=members, mf_product=mf, dla_core_product=dla, t1_product=t1,
                               t3_product=t3, metals_on=kw["metals"], sample_metals=kw["metals"],
                               metal_prior=("flatlog2node" if kw["metals"] else "uniform"), survey=name,
                               with_eboss=kw["with_eboss"], ks_kwargs=dict(k_max=0.065))
    leg = next(l for l in ctx.legs if l.name == name)
    return ctx._replace(legs=[leg]), d, leg


def leg_residuals(R, d, leg, ctx):
    """Per simulation and rung: the realized LF ensemble residual on the leg's kept bins in power units, truth side
    (each row's own stored grid, linear in k), classes combined with the leg's coefficients at the row's w_c; and the
    rung's ladder factor (tau0_amp of the leg vector)."""
    kept = FK.kept(leg)
    k, zr = np.asarray(leg.k)[kept], np.asarray(leg.z_row, float)[kept]
    dff = float(getattr(leg, "dla_forward_frac", 1.0))
    rows_all = np.asarray(R["rows"])
    sims = sorted(set(np.asarray(R["sim"]).astype(str)))
    zg, ai = np.asarray(d["z_grid"], float), np.asarray(d["alpha_idx"])
    kf, Pf, wc = np.asarray(d["kfkms"], float), np.asarray(d["P_filt"], float), np.asarray(d["w_c_cache"], float)
    nu = FK.nuis_centre(ctx, leg)
    metal_kw = {kk: v for kk, v in nu.items() if kk != "b_res"}
    out = {}
    for s in sims:
        sel = np.where(np.asarray(R["sim"]).astype(str) == s)[0]
        vecs = {}
        for rung in np.unique(ai[rows_all[sel]]):
            r = np.full(kept.size, np.nan)
            ok = True
            for zz in np.unique(zr):
                rr = sel[(np.abs(zg[rows_all[sel]] - zz) < 0.05) & (ai[rows_all[sel]] == rung)]
                if rr.size != 1:
                    ok = False
                    break
                row = rows_all[rr[0]]
                a = wc[row, 1:].copy(); a[2] *= dff
                coef = np.r_[1.0 - a.sum(), a]
                A = Pf[row].copy()
                m = np.abs(zr - zz) < 0.05
                resid_modes = np.einsum("c,cn,cn->n", coef, A, np.asarray(R["r"])[rr[0]])
                fac = (np.asarray(DL.metal_factor_at_z(jnp.asarray(k[m]), float(zz), float(np.asarray(d["tau0"])[row]),
                                                       **metal_kw)) if leg.metals_on else 1.0)
                r[m] = np.interp(k[m], kf[row], resid_modes) * fac
            if ok:
                alpha_rung = float(tau0_ladder_factor(np.asarray(d["tau0"])[rows_all[sel][ai[rows_all[sel]] == rung][0]],
                                                      zg[rows_all[sel][ai[rows_all[sel]] == rung][0]]))
                vecs[int(rung)] = (alpha_rung, r)
        out[s] = vecs
    return out, sims


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--leg", required=True, choices=sorted(SURVEYS))
    ap.add_argument("--t1", required=True)
    ap.add_argument("--t1-raw", required=True)
    ap.add_argument("--t3", default=None)
    ap.add_argument("--dla-core", required=True)
    ap.add_argument("--mf", required=True)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-npz", required=True)
    ap.add_argument("--grid", type=int, default=41)
    ap.add_argument("--path-points", type=int, default=201)
    ap.add_argument("--sims", type=int, default=None, help="first N simulations only (smoke)")
    a = ap.parse_args(argv)
    for p in (a.out_json, a.out_npz):
        if os.path.exists(p):
            raise SystemExit(f"refusing to overwrite {p}")
    _stage(f"start leg {a.leg}")
    members = production_member_paths()
    ctx, d, leg = build(a.leg, members, a.mf, a.dla_core, a.t1, a.t3)
    ctx_raw, _, _ = build(a.leg, members, a.mf, a.dla_core, a.t1_raw, a.t3)
    lo, hi = (np.asarray(x, float) for x in sampling_unit_bounds())
    centre_th = 0.5 * (lo + hi)
    p_c = FK.p_centre(ctx, centre_th)
    Pprior = FK.prior_precision(ctx)
    sig_prior = np.where(np.diag(Pprior) > 0, 1.0 / np.sqrt(np.where(np.diag(Pprior) > 0, np.diag(Pprior), 1.0)), np.nan)
    C_of = CA.cov_fn(ctx, leg)
    C_raw = CA.cov_fn(ctx_raw, leg)
    g_ld = jax.jit(jax.grad(CA.logdet_half(C_of)))
    f_ld = jax.jit(lambda p: 0.5 * jnp.linalg.slogdet(C_of(p))[1])
    LC = CA.expected_loglik(C_of)
    g_LC = jax.jit(jax.grad(LC))
    jac = jax.jit(jax.jacfwd(FK.mean_fn(ctx, leg)))

    def fisher_at(p):
        return FK.fisher(np.asarray(jac(jnp.asarray(p))), np.asarray(C_of(jnp.asarray(p))), Pprior)

    def proj(F, g):
        return FK.projected(F, np.asarray(g))[list(IDX)]
    out = dict(leg=a.leg, n_kept=int(FK.kept(leg).size), param_names=list(FK.param_names(ctx)))
    _stage("contexts built; A1 grid")
    # A1 grid and A3 a
    n = a.grid
    hubs = lo[I_HUB] + (hi[I_HUB] - lo[I_HUB]) * np.linspace(0, 1, n)
    oms = lo[I_OMH2] + (hi[I_OMH2] - lo[I_OMH2]) * np.linspace(0, 1, n)
    f_grid, pull_grid = np.zeros((n, n)), np.zeros((n, n, 3))
    for i, h in enumerate(hubs):
        for j, o in enumerate(oms):
            p = p_c.copy(); p[I_HUB], p[I_OMH2] = h, o
            f_grid[i, j] = float(f_ld(jnp.asarray(p)))
            pull_grid[i, j] = proj(fisher_at(p), g_ld(jnp.asarray(p)))
    _stage("A1 paths")
    # A1 paths along every parameter
    ranges = {}
    for ax in range(len(p_c)):
        if ax < 9:
            ranges[ax] = (lo[ax], hi[ax])
        elif ax == 9:
            ranges[ax] = tuple(ctx.tau0_amp_range)
        elif ax == 10:
            ranges[ax] = tuple(ctx.dtau0_range)
        else:
            ranges[ax] = (p_c[ax] - 3 * sig_prior[ax], p_c[ax] + 3 * sig_prior[ax])
    path_pull = {}
    for ax, (x0, x1) in ranges.items():
        vals = []
        for x in np.linspace(x0, x1, a.path_points):
            p = p_c.copy(); p[ax] = x
            vals.append(proj(fisher_at(p), g_ld(jnp.asarray(p))))
        path_pull[out["param_names"][ax]] = np.abs(np.array(vals)).max(axis=0).tolist()
    _stage("leg residuals")
    # simulations' leg vectors (A2 S_bar, A3 b, c)
    R = CB.load_loo_ensemble_residuals(a.eval_dir, d)
    vecs, sims = leg_residuals(R, d, leg, ctx)
    if a.sims:
        sims = sims[: a.sims]
    pu = np.asarray(d["params_unit"], float)
    theta_of = {s: pu[np.asarray(R["rows"])[np.asarray(R["sim"]).astype(str) == s][0]] for s in sims}
    Ck = np.asarray(leg.C_data, float)[np.ix_(FK.kept(leg), FK.kept(leg))]
    S_bar = Ck + np.mean([np.outer(v[1], v[1]) for s in sims for v in vecs[s].values()], axis=0)
    _stage("A2 kinks")
    # A2 kinks along hub and omegamh2 through the centre
    kept = FK.kept(leg)
    kb, zb = np.asarray(leg.k)[kept], np.asarray(leg.z_row, float)[kept]
    zc = np.unique(zb)
    iz = np.searchsorted(zc, zb)
    dlnk1 = {}
    for c, zz in enumerate(zc):
        gk = np.asarray(jax.grad(lambda th: jnp.log(KC.k_skm_from_theta9(ctx.k_com_hmpc, float(zz), th)[0]))(jnp.asarray(centre_th)))
        dlnk1[c] = gk
    Fc = fisher_at(p_c)
    dir_ld = jax.jit(lambda p, e: jax.jvp(CA.logdet_half(C_of), (p,), (e,))[1])
    dir_net = jax.jit(lambda p, e: jax.jvp(lambda q: LC(q, jnp.asarray(S_bar)), (p,), (e,))[1])
    kinks = []
    for ax in (I_HUB, I_OMH2):
        eps = 1e-6 * (hi[ax] - lo[ax])
        e = np.zeros(len(p_c)); e[ax] = 1.0
        for x, b in CA.crossings(np.asarray(ctx.k_com_hmpc), zc, iz, kb, centre_th, ax, lo[ax], hi[ax]):
            if x - eps <= lo[ax] or x + eps >= hi[ax]:
                continue
            pm, pp = p_c.copy(), p_c.copy(); pm[ax], pp[ax] = x - eps, x + eps
            jl = float(dir_ld(jnp.asarray(pp), jnp.asarray(e))) - float(dir_ld(jnp.asarray(pm), jnp.asarray(e)))
            jn = float(dir_net(jnp.asarray(pp), jnp.asarray(e))) - float(dir_net(jnp.asarray(pm), jnp.asarray(e)))
            w = dlnk1[iz[b]]
            vec = np.zeros(len(p_c))
            vec[I_HUB], vec[I_OMH2] = w[I_HUB] / w[ax], w[I_OMH2] / w[ax]
            kinks.append(np.r_[np.abs(proj(Fc, jl * vec)), np.abs(proj(Fc, jn * vec))])
    kinks = np.array(kinks) if kinks else np.zeros((0, 6))
    _stage(f"A3 b-d ({len(kinks)} crossings done)")
    # A3 b (all rungs), c and d (central rung)
    net_sim, newton, info = [], [], []
    for s in sims:
        per = []
        for rung, (alpha_r, r) in sorted(vecs[s].items()):
            p = FK.p_centre(ctx, theta_of[s]); p[9], p[10] = alpha_r, 0.0
            S = np.outer(r, r) + Ck
            per.append(proj(fisher_at(p), np.asarray(g_LC(jnp.asarray(p), jnp.asarray(S)))))
        net_sim.append(np.mean(per, axis=0))
        rungs = sorted(vecs[s])
        alpha_c, r_c = vecs[s][rungs[len(rungs) // 2]]
        p = FK.p_centre(ctx, theta_of[s]); p[9], p[10] = alpha_c, 0.0
        S = jnp.asarray(np.outer(r_c, r_c) + Ck)
        J = np.asarray(jac(jnp.asarray(p)))
        Cs = np.asarray(C_of(jnp.asarray(p)))
        F = FK.fisher(J, Cs, Pprior)
        dC = np.moveaxis(np.asarray(jax.jacfwd(C_of)(jnp.asarray(p))), -1, 0)
        Ci = np.linalg.inv(Cs)
        A = np.array([Ci @ dC[i] for i in range(len(p))])
        I_C = 0.5 * np.einsum("iab,jba->ij", A, A)                     # 1/2 tr(C^-1 C_,i C^-1 C_,j)
        info.append((np.diag(I_C) / np.diag(J.T @ Ci @ J))[list(IDX)])   # cov_audit.info_ratio, from I_C
        d_lin = np.linalg.solve(F, np.asarray(g_LC(jnp.asarray(p), S)))
        sig = FK.marginal_sigma(F)
        d_nl = CA.newton_shift(lambda dd: LC(jnp.asarray(p) + dd, S), F, d_lin, sig, metric=I_C)
        newton.append(np.r_[(d_lin / sig)[list(IDX)], (d_nl / sig)[list(IDX)]])
    net_sim, newton, info = np.array(net_sim), np.array(newton), np.array(info)
    _stage("A4 texture")
    # A4 texture: T1 selected over raw at the data bins along the hub and omegamh2 paths (diagonal of C - C_data)
    tex = {}
    for ax in (I_HUB, I_OMH2):
        ratios = []
        for x in np.linspace(lo[ax], hi[ax], 21):
            p = p_c.copy(); p[ax] = x
            t1s = np.diag(np.asarray(C_of(jnp.asarray(p)))) - np.diag(Ck)
            t1r = np.diag(np.asarray(C_raw(jnp.asarray(p)))) - np.diag(Ck)
            ratios.append(t1s / t1r)
        ratios = np.array(ratios)
        tex[out["param_names"][ax]] = dict(median=float(np.median(ratios)), p05=float(np.percentile(ratios, 5)),
                                           p95=float(np.percentile(ratios, 95)))
    _stage("FD agreement")
    # finite-difference agreement on a subset (the registered primary derivative; jax used for T1/T3)
    rng = np.random.default_rng(0)
    fd_rel = []
    for _ in range(3):
        p = p_c.copy(); p[I_HUB] = rng.uniform(lo[I_HUB], hi[I_HUB]); p[I_OMH2] = rng.uniform(lo[I_OMH2], hi[I_OMH2])
        g = np.asarray(g_ld(jnp.asarray(p)))
        for ax in (0, 1, I_HUB, I_OMH2, 9):
            h = 1e-4 * (ranges[ax][1] - ranges[ax][0])
            pp, pm = p.copy(), p.copy(); pp[ax] += h; pm[ax] -= h
            fd = -(float(f_ld(jnp.asarray(pp))) - float(f_ld(jnp.asarray(pm)))) / (2 * h)
            fd_rel.append(abs(fd - g[ax]) / max(abs(g[ax]), 1e-12))
    out.update(
        A1=dict(logdet_half_range=float(f_grid.max() - f_grid.min())),
        A3a=dict(grid_max=np.abs(pull_grid).max(axis=(0, 1)).tolist(), path_max=path_pull),
        A2=dict(n_crossings=int(len(kinks)), logdet_jump_max=(kinks[:, :3].max(axis=0).tolist() if len(kinks) else [0, 0, 0]),
                net_jump_max=(kinks[:, 3:].max(axis=0).tolist() if len(kinks) else [0, 0, 0])),
        A3b=dict(n_sims=len(sims), mean=np.abs(net_sim.mean(axis=0)).tolist(),
                 rms=np.sqrt((net_sim ** 2).mean(axis=0)).tolist(), max=np.abs(net_sim).max(axis=0).tolist()),
        A3c=dict(linear_mean=newton[:, :3].mean(axis=0).tolist(), nonlinear_mean=newton[:, 3:].mean(axis=0).tolist(),
                 max_abs_difference=np.abs(newton[:, 3:] - newton[:, :3]).max(axis=0).tolist()),
        A3d=dict(info_ratio_mean=info.mean(axis=0).tolist(), info_ratio_max=info.max(axis=0).tolist()),
        A4=tex, fd_agreement_max_rel=float(max(fd_rel)), thresholds=THRESH, params_projected=["n_s", "A_p", "tau0_amp"])
    m = out
    material = (max(m["A2"]["logdet_jump_max"] + m["A2"]["net_jump_max"]) > THRESH["kink"]
                or max(m["A3b"]["mean"]) > THRESH["net_mean"] or max(m["A3b"]["rms"]) > THRESH["net_rms"]
                or max(m["A3b"]["max"]) > THRESH["net_max"]
                or max(m["A3a"]["grid_max"] + [v for vv in m["A3a"]["path_max"].values() for v in vv]) > THRESH["logdet"]
                or max(abs(v) for v in m["A3c"]["nonlinear_mean"]) > THRESH["net_mean"])
    out["material"] = bool(material)
    commit = subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    out["meta"] = dict(created_utc=datetime.now(timezone.utc).isoformat(), code_commit=commit, members=members,
                       t1=a.t1, t1_sha256=_sha(a.t1), t1_raw=a.t1_raw, t3=a.t3,
                       t3_sha256=(_sha(a.t3) if a.t3 else None), dla_core_sha256=_sha(a.dla_core), mf_sha256=_sha(a.mf),
                       grid=n, path_points=a.path_points, t2="off (amendment A2 pending)")
    os.makedirs(os.path.dirname(os.path.abspath(a.out_npz)), exist_ok=True)
    np.savez(a.out_npz, f_grid=f_grid, pull_grid=pull_grid, kinks=kinks, net_sim=net_sim, newton=newton, info=info,
             hubs=hubs, oms=oms)
    out["meta"]["npz_sha256"] = _sha(a.out_npz)
    with open(a.out_json, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ("leg", "material", "A2", "A3b")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
