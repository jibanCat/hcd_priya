#!/usr/bin/env python3
"""Gate E amendment A1 rev 1 section 3, step 3: the T3 representation comparison, mode-aligned (M) vs fixed physical k
(P), per leg (DESI, KS; eBOSS reported), by 10-fold simulation CV on the 60 LOO ensemble residuals. Builds no product.

Coherent residual coh_s(z, mode): the clean-class ensemble residual averaged over simulation s's rows at z, z grid
2.0-4.4. Leg bins: kept, k >= 0.01 s/km, z on that grid. Quantities per representation and rank {5, 10, 15, 25}:
(a) CV log score per held-out simulation (and the paired SE of M - P), (b) kappa = mean v^T C^-1 v / mean tr(C^-1
Sigma), C = C_data / P_data P_data^T + Sigma, (c) stationarity (mean correlation at |Delta ln k| in [0.15, 0.25] within
z, low k < 0.031 vs high), (h) the projected statistic T = mean_s delta^T V^-1 delta / 3 over (n_s, A_p, tau0_amp) with
the production Fisher setup (fisher_kit) at each simulation's theta. For M only (P is a constant matrix: its log det
is theta-independent, so (d)-(g) are identically zero): (d) the range of log|C| over the (hub, omegamh2) box (41 x 41,
other parameters at the box centre), (e) its gradient, (f) gradient jumps at every enumerated mode crossing along the
hub and omegamh2 paths through the box centre, (g) the unbalanced log-det pull F^-1 (-1/2 grad log|C|) over the grid
and the net pull at each simulation's theta with S_s = r_s r_s^T + C_data; all projected in marginal sigma of n_s,
A_p, tau0_amp. Decision (registered): P unless M wins by > 2 paired SE and > 1 nat per held-out simulation AND is
meaningfully better calibrated; a qualifying M with a material (f) or (g) STOPS (stop 2). Write-once outputs.

ENV: PYTHONNOUSERSITE=1 PYTHONPATH=<repo> JAX_PLATFORMS=cpu /home/mfho/.conda/envs/emu-jax/bin/python3 \
     scripts/run_gate_e_t3_compare.py --eval-dir <gateC_eval> --mf <mf product> --dla-core <dla_core product> \
     --out-json <..> --out-npz <..>
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
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import brentq

from hcd_analysis.emulator import cemu_build as CB
from hcd_analysis.emulator import cemu_t3 as T3
from hcd_analysis.emulator import closure_legb as CL
from hcd_analysis.emulator import fisher_kit as FK
from hcd_analysis.emulator import kcoord as KC
from hcd_analysis.emulator.data import PARAM_LIMITS, load_cache, sampling_unit_bounds
from hcd_analysis.emulator.prod_ensemble import production_member_paths

T3_Z = np.round(np.arange(2.0, 4.41, 0.2), 1)
T3_KMIN = 0.01
RANKS = (5, 10, 15, 25)
SURVEYS = {"DESI": dict(metals=True, with_eboss=False), "KS": dict(metals=False, with_eboss=False),
           "eBOSS": dict(metals=True, with_eboss=True)}
AUDIT_LEGS = ("DESI", "KS")                                    # production applies T3 on DESI and KS only
I_NS, I_AP, I_TAU = 0, 1, 9
I_HUB, I_OMH2 = 5, 6


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build_leg_ctx(name, members, mf, dla):
    kw = SURVEYS[name]
    ctx, d = CL.build_legb_ctx(ensemble_ckpts=members, mf_product=mf, dla_core_product=dla, metals_on=kw["metals"],
                               sample_metals=kw["metals"], metal_prior=("flatlog2node" if kw["metals"] else "uniform"),
                               survey=name, with_eboss=kw["with_eboss"], ks_kwargs=dict(k_max=0.065))
    leg = next(l for l in ctx.legs if l.name == name)
    return ctx._replace(legs=[leg]), d, leg


def coh_per_sim(R, d):
    sims = sorted(set(np.asarray(R["sim"]).astype(str)))
    z = np.asarray(R["z"], float)
    coh = np.zeros((len(sims), T3_Z.size, R["r"].shape[-1]))
    theta = np.zeros((len(sims), 9))
    pu = np.asarray(d["params_unit"], float)
    for i, s in enumerate(sims):
        sel = np.asarray(R["sim"]).astype(str) == s
        theta[i] = pu[np.asarray(R["rows"])[sel][0]]
        for j, zz in enumerate(T3_Z):
            rows = sel & (np.abs(z - zz) < 0.05)
            coh[i, j] = np.asarray(R["r"])[rows, 0, :].mean(axis=0) if rows.any() else np.nan   # 7 sims lack z 2.0
    return sims, coh, theta


def leg_bins(leg):
    keep = np.isfinite(np.asarray(leg.P_data))
    k, zr = np.asarray(leg.k), np.asarray(leg.z_row, float)
    iz = np.array([int(np.argmin(np.abs(T3_Z - z))) for z in zr])
    onz = np.abs(T3_Z[iz] - zr) < 0.05
    B = np.where(keep & (k >= T3_KMIN) & onz)[0]
    kept = np.where(keep)[0]
    pos = np.searchsorted(kept, B)                             # positions of B inside the kept-bin vector
    return dict(z=T3_Z, iz=iz[B], k=k[B]), B, kept, pos


def stationarity(Sig, bins):
    sd = np.sqrt(np.diag(Sig))
    rho = Sig / np.outer(sd, sd)
    lk, iz = np.log(bins["k"]), bins["iz"]
    dl = np.abs(lk[:, None] - lk[None, :])
    same = (iz[:, None] == iz[None, :]) & (dl >= 0.15) & (dl <= 0.25)
    low = (bins["k"][:, None] < 0.031) & (bins["k"][None, :] < 0.031)
    high = (bins["k"][:, None] >= 0.031) & (bins["k"][None, :] >= 0.031)
    f = lambda m: float(np.mean(rho[m])) if m.any() else float("nan")
    return dict(low=f(same & low), high=f(same & high))


def jax_weights(k_com, theta, bins, Mi):
    K = len(k_com)
    cols = []
    rows_W = jnp.zeros((len(bins["k"]), T3_Z.size * K))
    for c in np.unique(bins["iz"]):
        sel = np.where(bins["iz"] == c)[0]
        b = KC.bind(KC.kgrid(k_com, float(T3_Z[c]), theta), jnp.asarray(bins["k"][sel]))
        j = b.j.astype(jnp.int32)
        rows_W = rows_W.at[sel, c * K + j - 1].set(1.0 - b.t_log)
        rows_W = rows_W.at[sel, c * K + j].set(b.t_log)
    del cols
    return rows_W[:, Mi]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--mf", required=True)
    ap.add_argument("--dla-core", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-npz", required=True)
    ap.add_argument("--legs", default="DESI,KS,eBOSS")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--grid", type=int, default=41)
    a = ap.parse_args(argv)
    for p in (a.out_json, a.out_npz):
        if os.path.exists(p):
            raise SystemExit(f"refusing to overwrite {p}")
    members = production_member_paths()
    lo, hi = (np.asarray(x, float) for x in sampling_unit_bounds())
    centre = 0.5 * (lo + hi)
    summary, arrays = {}, {}
    R = None
    for name in a.legs.split(","):
        ctx, d, leg = build_leg_ctx(name, members, a.mf, a.dla_core)
        if R is None:
            R = CB.load_loo_ensemble_residuals(a.eval_dir, d)
            sims, coh, theta = coh_per_sim(R, d)
        k_com = np.asarray(ctx.k_com_hmpc)
        bins, B, kept, pos = leg_bins(leg)
        used = np.unique(bins["iz"])
        if not np.all(np.isfinite(coh[:, used])):
            raise SystemExit(f"{name}: a z cell used by the leg's bins lacks rows in some simulation")
        coh_leg = np.nan_to_num(coh)                              # unused z cells carry zero weight
        P = np.asarray(leg.P_data, float)
        Ck = np.asarray(leg.C_data, float)[np.ix_(kept, kept)]
        PB = P[B]
        C_frac = np.asarray(leg.C_data, float)[np.ix_(B, B)] / np.outer(PB, PB)
        Pprior = FK.prior_precision(ctx)
        J = {}
        for i in range(len(sims)):
            J[i] = np.asarray(FK.jacobian(ctx, leg, FK.p_centre(ctx, theta[i])))

        def hook(rep, r, s, Sig, v):
            C = Ck.copy()
            C[np.ix_(pos, pos)] += Sig * np.outer(PB, PB)
            res = np.zeros(kept.size)
            res[pos] = v * PB
            cf = cho_factor(C)
            CiJ = cho_solve(cf, J[s])
            F = J[s].T @ CiJ + Pprior
            Finv = np.linalg.inv(F)
            delta = Finv @ (J[s].T @ cho_solve(cf, res))
            Spow = np.zeros_like(C)
            Spow[np.ix_(pos, pos)] = Sig * np.outer(PB, PB)
            V = Finv @ (CiJ.T @ Spow @ CiJ) @ Finv
            idx = [I_NS, I_AP, I_TAU]
            db, Vb = delta[idx], V[np.ix_(idx, idx)]
            return dict(T=float(db @ np.linalg.solve(Vb, db) / 3.0), q=(db ** 2 / np.diag(Vb)).tolist())

        res = T3.compare(coh_leg, theta, bins, k_com, ranks=RANKS, n_folds=10, lo=lo, hi=hi, names=sims, C_frac=C_frac,
                         hook=hook)
        dec = T3.decide(res, RANKS, n_boot=a.n_boot, seed=0)
        out = dict(n_bins=int(B.size), n_sims=len(sims), decision_stats=dec, per_rank={})
        for rep in ("M", "P"):
            for r in RANKS:
                Ts = np.array([res["hook"][rep][r][s]["T"] for s in range(len(sims))])
                qs = np.array([res["hook"][rep][r][s]["q"] for s in range(len(sims))])
                out["per_rank"][f"{rep}{r}"] = dict(
                    score_mean=float(res["score"][rep][r].mean()),
                    kappa=float(res["quad"][rep][r].mean() / res["trace"][rep][r].mean()),
                    T=float(Ts.mean()), q_ns=float(qs[:, 0].mean()), q_Ap=float(qs[:, 1].mean()),
                    q_tau0=float(qs[:, 2].mean()))
                arrays[f"{name}_score_{rep}{r}"] = res["score"][rep][r]
        kM = out["per_rank"][f"M{dec['best_rank']['M']}"]["kappa"]
        kP = out["per_rank"][f"P{dec['best_rank']['P']}"]["kappa"]
        TM = out["per_rank"][f"M{dec['best_rank']['M']}"]["T"]
        TP = out["per_rank"][f"P{dec['best_rank']['P']}"]["T"]
        band = lambda t: 0.67 <= t <= 1.5
        calib_gain = (abs(np.log(kM)) < abs(np.log(kP)) - 0.1) or ((not band(TP)) and band(TM))
        out["m_qualifies"] = bool(dec["m_wins_score"] and calib_gain)
        out["calibration_conjunct"] = bool(calib_gain)
        # (c) stationarity on all 60 simulations at the best ranks (M bound at the box centre)
        V = res["V"]
        Mi = res["mode_set"]
        S_P = V.T @ V / V.shape[0]
        Sig_P = T3.with_topup(T3.top_r(S_P, dec["best_rank"]["P"]), S_P)
        Cm = coh_leg.reshape(coh_leg.shape[0], -1)[:, Mi]
        S_M = Cm.T @ Cm / Cm.shape[0]
        F_M = T3.top_r(S_M, dec["best_rank"]["M"])
        Bc = T3.weights(k_com, centre, bins["z"], bins["iz"], bins["k"])[:, Mi]
        Sig_Mc = Bc @ F_M @ Bc.T
        Sig_Mc += np.diag(np.maximum(np.einsum("ij,jk,ik->i", Bc, S_M, Bc) - np.diag(Sig_Mc), 0.0))
        out["stationarity"] = dict(P=stationarity(Sig_P, bins), M=stationarity(Sig_Mc, bins))
        # (d)-(g) for M (P is theta-independent: identically zero)
        if name in AUDIT_LEGS:
            out["M_theta_dependence"] = m_theta_audit(ctx, leg, k_com, bins, Mi, F_M, S_M, Ck, pos, PB, kept, Pprior,
                                                      centre, lo, hi, theta, J, coh_leg, a.grid)
        out["choice"] = "M" if out["m_qualifies"] else "P"
        out["stop_2"] = bool(out["m_qualifies"] and out.get("M_theta_dependence", {}).get("material", False))
        summary[name] = out
        print(name, json.dumps({k: out[k] for k in ("choice", "m_qualifies", "stop_2")}), dec)
    commit = subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    meta = dict(created_utc=datetime.now(timezone.utc).isoformat(), code_commit=commit, eval_dir=a.eval_dir,
                mf=a.mf, mf_sha256=_sha(a.mf), dla_core=a.dla_core, dla_core_sha256=_sha(a.dla_core),
                members=members, ranks=list(RANKS), t3_z=T3_Z.tolist(), t3_kmin=T3_KMIN)
    os.makedirs(os.path.dirname(os.path.abspath(a.out_npz)), exist_ok=True)
    np.savez(a.out_npz, sims=np.array(sims), **arrays)
    meta["npz"], meta["npz_sha256"] = a.out_npz, _sha(a.out_npz)
    with open(a.out_json, "w") as f:
        json.dump(dict(meta=meta, legs=summary), f, indent=1)
    return 0


def m_theta_audit(ctx, leg, k_com, bins, Mi, F_M, S_M, Ck, pos, PB, kept, Pprior, centre, lo, hi, theta, J, coh, n):
    """(d)-(g) for the M representation (A1 rev 1 section 3), log|C| with C = C_data + T3_M(theta) on the kept bins."""
    F_Mj, S_Mj, Ckj = jnp.asarray(F_M), jnp.asarray(S_M), jnp.asarray(Ck)
    PP = jnp.asarray(np.outer(PB, PB))
    posj = jnp.asarray(pos)

    def C_of(th):
        Bm = jax_weights(k_com, th, bins, Mi)
        Sig = Bm @ F_Mj @ Bm.T
        Sig = Sig + jnp.diag(jnp.maximum(jnp.einsum("ij,jk,ik->i", Bm, S_Mj, Bm) - jnp.diag(Sig), 0.0))
        return Ckj.at[jnp.ix_(posj, posj)].add(Sig * PP)

    ld = jax.jit(lambda th: jnp.linalg.slogdet(C_of(th))[1])
    gld = jax.jit(jax.grad(lambda th: -0.5 * jnp.linalg.slogdet(C_of(th))[1]))

    def LC(th, S):
        C = C_of(th)
        return -0.5 * (jnp.linalg.slogdet(C)[1] + jnp.trace(jnp.linalg.solve(C, S)))
    gLC = jax.jit(jax.grad(LC))
    # Fisher at the box centre (for the grid and path projections)
    Jc = np.asarray(FK.jacobian(ctx, leg, FK.p_centre(ctx, centre)))
    Cc = np.asarray(C_of(jnp.asarray(centre)))
    Fc = FK.fisher(Jc, Cc, Pprior)
    idx = [I_NS, I_AP, I_TAU]

    def proj(F, g9):
        g = np.zeros(F.shape[0])
        g[:9] = g9
        return np.abs(FK.projected(F, g)[idx])
    hubs = lo[I_HUB] + (hi[I_HUB] - lo[I_HUB]) * np.linspace(0, 1, n)
    oms = lo[I_OMH2] + (hi[I_OMH2] - lo[I_OMH2]) * np.linspace(0, 1, n)
    lds, pulls = np.zeros((n, n)), np.zeros((n, n, 3))
    for i, h in enumerate(hubs):
        for j, o in enumerate(oms):
            th = centre.copy(); th[I_HUB], th[I_OMH2] = h, o
            lds[i, j] = float(ld(jnp.asarray(th)))
            pulls[i, j] = proj(Fc, np.asarray(gld(jnp.asarray(th))))
    # (f) mode crossings along the hub and omegamh2 paths through the centre
    jumps = []
    for ax in (I_HUB, I_OMH2):
        width = hi[ax] - lo[ax]
        eps = 1e-6 * width

        def k1(x):
            th = centre.copy(); th[ax] = x
            return {c: float(KC.k_skm_from_theta9(k_com, float(T3_Z[c]), jnp.asarray(th))[0]) for c in np.unique(bins["iz"])}
        k1_lo, k1_hi = k1(lo[ax]), k1(hi[ax])
        for kb, c in zip(bins["k"], bins["iz"]):
            ua, ub = kb / k1_lo[c], kb / k1_hi[c]
            for m in range(int(np.ceil(min(ua, ub))), int(np.floor(max(ua, ub))) + 1):
                try:
                    x = brentq(lambda xx: kb / k1(xx)[c] - m, lo[ax], hi[ax], xtol=1e-14)
                except ValueError:
                    continue
                if x - eps <= lo[ax] or x + eps >= hi[ax]:
                    continue
                tm, tp = centre.copy(), centre.copy()
                tm[ax], tp[ax] = x - eps, x + eps
                g = np.asarray(gld(jnp.asarray(tp))) - np.asarray(gld(jnp.asarray(tm)))
                jumps.append(proj(Fc, g))
    jumps = np.array(jumps) if jumps else np.zeros((0, 3))
    # net pull at each simulation's theta, S_s = r_s r_s^T + C_data
    net = []
    for s in range(theta.shape[0]):
        Bs = T3.weights(k_com, theta[s], bins["z"], bins["iz"], bins["k"])
        v = Bs @ coh[s].ravel()
        r = np.zeros(kept.size); r[pos] = v * PB
        S = np.outer(r, r) + Ck
        th = jnp.asarray(theta[s])
        Cs = np.asarray(C_of(th))
        Fs = FK.fisher(J[s], Cs, Pprior)
        net.append(FK.projected(Fs, np.r_[np.asarray(gLC(th, jnp.asarray(S))), np.zeros(Fs.shape[0] - 9)])[idx])
    net = np.array(net)
    out = dict(logdet_range=float(lds.max() - lds.min()), logdet_pull_max=pulls.max(axis=(0, 1)).tolist(),
               kink_max=(jumps.max(axis=0).tolist() if jumps.size else [0.0, 0.0, 0.0]), n_crossings=int(len(jumps)),
               net_pull_mean=np.abs(net.mean(axis=0)).tolist(), net_pull_rms=np.sqrt((net ** 2).mean(axis=0)).tolist(),
               net_pull_max=np.abs(net).max(axis=0).tolist())
    out["material"] = bool(max(out["kink_max"]) > 0.05 or max(out["net_pull_mean"]) > 0.1 or max(out["net_pull_rms"]) > 0.2
                           or max(out["net_pull_max"]) > 0.5 or max(out["logdet_pull_max"]) > 0.3)   # A2/A3 thresholds
    return out


if __name__ == "__main__":
    sys.exit(main())
