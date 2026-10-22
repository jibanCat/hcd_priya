#!/usr/bin/env python3
"""Gate D (spec GATE_D_SPEC.md in the notes repository; criteria D1-D3 registered in PU-0053 before measurement).

1. matches HR and LF rows exactly on (theta, z_grid, rung); D1: coordinate mapping error on modes 1..172 against the
   stored LF grids and the canonical coordinate k_skm(z, theta);
2. LF backbone = the 5-member production ensemble (mean P_filt) on the matched LF rows; targets g per mode;
3. the all-HR-simulation correction tables -> the gate D product (k_com labelled, provenance);
4. leave-one-HR-simulation-out (6 folds): held-out residuals of MF and of LF alone vs HR truth -> D2, D3, R-D2a;
5. FULLY held-out arm (R-D2b): the held-out simulation's LF from its own 5-seed leave-one-simulation-out ensemble.

Usage (from the checkout root):
  PYTHONNOUSERSITE=1 PYTHONPATH=<repo> JAX_PLATFORMS=cpu python3 scripts/run_gate_d.py --ckpt-dir DIR --out-dir DIR
"""
from __future__ import annotations

import os as _os_rr
_REPO_ROOT = _os_rr.path.dirname(_os_rr.path.dirname(_os_rr.path.abspath(__file__)))
import argparse
import json
import subprocess
import sys

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, _REPO_ROOT)
import hcd_analysis.emulator  # noqa: F401,E402
from hcd_analysis.emulator import data as D  # noqa: E402
from hcd_analysis.emulator import gate_d as GD  # noqa: E402
from hcd_analysis.emulator import mf_modes as MM  # noqa: E402
from hcd_analysis.emulator import train as T  # noqa: E402
from hcd_analysis.emulator.predict import predict_P_filt  # noqa: E402

LF = f"{_REPO_ROOT}/hcd_analysis/_emulator_data/observables_tau0_lf.h5"
HR = f"{_REPO_ROOT}/hcd_analysis/_emulator_data/observables_tau0_hr.h5"
NM = 172
CLS = ("clean", "LLS", "subDLA", "DLA")


def ensemble_P(ckpts, x, tau0):
    """Mean over members of the predicted P_filt (rows, 4, K) at encoder inputs x (rows, 10) and tau0 (rows,)."""
    out = []
    for c in ckpts:
        model, meta, norm = T.load_checkpoint(c)
        pf = {k: jnp.asarray(norm["P_filt"][k]) for k in ("mu_marg", "sig_marg", "sig_cosmo")}
        f = jax.jit(jax.vmap(lambda xi, ti: predict_P_filt(model, xi[:9], xi[9], ti, pf)))
        out.append(np.asarray(f(jnp.asarray(x), jnp.asarray(tau0))))
    return np.mean(np.stack(out), axis=0)


def residuals(P_pred, P_hr):
    ok = np.isfinite(P_pred) & np.isfinite(P_hr) & (P_hr > 0) & (P_pred > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ok, P_pred / P_hr - 1.0, np.nan), np.where(ok, np.log(np.where(ok, P_pred, 1) / np.where(ok, P_hr, 1)), np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", required=True, help="gate C checkpoint directory (prod_repaired_seed*, loo60_s*)")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os = _os_rr
    os.makedirs(a.out_dir, exist_ok=True)
    lf, hr = D.load_cache(LF), D.load_cache(HR)
    pairs = MM.match_pairs(lf, hr)
    h_rows = np.array([p[0] for p in pairs]); l_rows = np.array([p[1] for p in pairs])
    R = {"n_pairs": len(pairs), "n_hr_rows": int(hr["z_grid"].size)}

    # D1
    R["D1"] = dict(stored_grids=MM.mode_mapping_error(lf, hr, pairs, NM),
                   canonical=MM.canonical_mapping_error(hr, pairs, lf["k_com_hmpc"]))
    R["D1"]["pass_"] = bool(R["D1"]["stored_grids"] <= 1e-12 and R["D1"]["canonical"] <= 1e-12)

    # LF ensemble on matched rows, targets
    prod = [f"{a.ckpt_dir}/prod_repaired_seed{s}" for s in range(5)]
    P_lf = ensemble_P(prod, lf["x"][l_rows], lf["tau0"][l_rows])
    t = MM.measure_mode_targets(lf, hr, pairs, P_lf, x=lf["x"], n_modes=NM)
    P_hr = np.asarray(hr["P_filt"])[h_rows][:, :, :NM]
    k_hr = np.asarray(hr["kfkms"])[h_rows][:, :NM]
    z = np.asarray(hr["z_grid"])[h_rows]
    in_range = (z[:, None] >= 2.2 - 1e-9) & (z[:, None] <= 4.6 + 1e-9) & (k_hr >= 1e-3)
    sims = sorted(set(t["sim"]))
    R["hr_sims_by_lf_name"] = sims

    # all-6 product
    tables = MM.fit_mode_mf(t)
    commit = subprocess.run(["git", "-C", _REPO_ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    prov = dict(lf_backbone=prod, lf_cache_sha256=T._sha256_or_none(LF), hr_cache_sha256=T._sha256_or_none(HR),
                lf_ckpt_sha256=[T._sha256_or_none(p + ".eqx") for p in prod], code_commit=commit or "unknown",
                spec="GATE_D_SPEC.md (PU-0053)", n_modes=NM, res_corr="off (NORC)")
    MM.save_mode_mf(f"{a.out_dir}/mf_modes_all6.npz", tables, k_com_hmpc=lf["k_com_hmpc"][:NM], provenance=prov)
    g_all = np.stack([MM.apply_mode_mf(tables, t["x"][i], t["tau0"][i]) for i in range(len(pairs))])

    # leave-one-HR-simulation-out
    r_mf, r_lf, l_mf, l_lf = (np.full(P_hr.shape, np.nan) for _ in range(4))
    for s in sims:
        held = np.where(t["sim"] == s)[0]
        tab = MM.fit_mode_mf(t, train_rows=np.where(t["sim"] != s)[0])
        g = np.stack([MM.apply_mode_mf(tab, t["x"][i], t["tau0"][i]) for i in held])
        r_mf[held], l_mf[held] = residuals(P_lf[held] * np.exp(g), P_hr[held])
        r_lf[held], l_lf[held] = residuals(P_lf[held], P_hr[held])
    keep = in_range[:, None, :] & np.isfinite(r_mf) & np.isfinite(r_lf)
    rms = lambda r: [float(np.sqrt(np.nanmean(np.where(keep[:, c], r[:, c], np.nan) ** 2))) for c in range(4)]
    R["D2"] = dict(rms_mf=dict(zip(CLS, rms(r_mf))), rms_lf=dict(zip(CLS, rms(r_lf))))
    R["D2"]["pass_"] = bool(all(R["D2"]["rms_mf"][c] < R["D2"]["rms_lf"][c] for c in CLS))
    band = keep & ((z >= 2.8 - 1e-9) & (z <= 3.4 + 1e-9))[:, None, None] & (k_hr >= 0.0442)[:, None, :]
    mean_band = lambda lg: [float(np.nanmean(np.where(band[:, c], lg[:, c], np.nan))) for c in range(4)]
    R["D3"] = dict(mean_log_mf=dict(zip(CLS, mean_band(l_mf))), mean_log_lf=dict(zip(CLS, mean_band(l_lf))))
    R["D3"]["pass_"] = bool(all(abs(R["D3"]["mean_log_mf"][c]) < abs(R["D3"]["mean_log_lf"][c]) for c in CLS))
    # gate D review (PU-0063): the pooled D3 of a leave-one-out mean correction is fixed by construction; read the
    # high-k band per held-out simulation and on a common mode-index band (gate_d.d3_per_simulation)
    R["D3_per_simulation"] = GD.d3_per_simulation(l_mf, l_lf, t["sim"], z, k_hr, keep)

    # R-D2a stratification of the held-out MF residual
    def strat(r, sel_fn):
        return {k: [float(np.nanmedian(np.abs(np.where(keep[:, c] & m, r[:, c], np.nan)))) for c in range(4)]
                for k, m in sel_fn()}
    R["R_D2a"] = dict(
        by_z=strat(r_mf, lambda: [(str(zz), np.broadcast_to((np.round(z, 1) == zz)[:, None], keep[:, 0].shape))
                                  for zz in np.unique(np.round(z, 1))]),
        by_k=strat(r_mf, lambda: [(f"[{lo},{hi})", (k_hr >= lo) & (k_hr < hi)) for lo, hi in
                                  ((1e-3, 0.005), (0.005, 0.01), (0.01, 0.02), (0.02, 0.04), (0.04, 0.07), (0.07, 1.0))]),
        by_sim=strat(r_mf, lambda: [(s[:24], np.broadcast_to((t["sim"] == s)[:, None], keep[:, 0].shape)) for s in sims]),
        lf_alone_by_k=strat(r_lf, lambda: [(f"[{lo},{hi})", (k_hr >= lo) & (k_hr < hi)) for lo, hi in
                                           ((1e-3, 0.005), (0.005, 0.01), (0.01, 0.02), (0.02, 0.04), (0.04, 0.07), (0.07, 1.0))]))

    # R-D2b fully held-out (LF from the held-out simulation's own 5-seed leave-one-simulation-out ensemble)
    lf_names = sorted(set(np.asarray(lf["sim_name"]).astype(str)))
    r_full = np.full(P_hr.shape, np.nan)
    for s in sims:
        held = np.where(t["sim"] == s)[0]
        n = lf_names.index(s)
        loo = [f"{a.ckpt_dir}/loo60_s{n:02d}_seed{k}" for k in range(5)]
        P_loo = ensemble_P(loo, lf["x"][l_rows[held]], lf["tau0"][l_rows[held]])
        tab = MM.fit_mode_mf(t, train_rows=np.where(t["sim"] != s)[0])
        g = np.stack([MM.apply_mode_mf(tab, t["x"][i], t["tau0"][i]) for i in held])
        r_full[held], _ = residuals(P_loo * np.exp(g), P_hr[held])
    R["R_D2b_fully_held_out"] = dict(rms=dict(zip(CLS, rms(r_full))),
                                     median_abs=dict(zip(CLS, [float(np.nanmedian(np.abs(np.where(keep[:, c], r_full[:, c], np.nan))))
                                                               for c in range(4)])))
    # R-D1 the genuine correction (all-6), median over rows of g in the data band, per class and z
    R["R_D1"] = {str(zz): [float(np.nanmedian(np.where(in_range[np.round(z, 1) == zz], g_all[np.round(z, 1) == zz][:, c], np.nan)))
                           for c in range(4)] for zz in np.unique(np.round(z, 1))}
    np.savez_compressed(f"{a.out_dir}/gate_d_arrays.npz", g_all=g_all, g_measured=t["g"], r_mf=r_mf, r_lf=r_lf, r_full=r_full,
                        z=z, k_hr=k_hr, sim=t["sim"], alpha_idx=t["alpha_idx"], in_range=in_range)
    json.dump(R, open(f"{a.out_dir}/gate_d_results.json", "w"), indent=1)
    print(json.dumps({k: R[k] for k in ("n_pairs", "D1", "D2", "D3", "R_D2b_fully_held_out")}, indent=1))


if __name__ == "__main__":
    main()
