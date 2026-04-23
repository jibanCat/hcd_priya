"""
Held-out test of the global MF correction using the 4th HR sim (ns0.914…)
which has no direct LF counterpart.  Strategy:

  1. Fit the global G1/G2 MF correction on the 3 matched HR–LF training
     pairs (`mf_global_fit`).
  2. For the 4th HR sim at parameter point θ_4 = (A_p_4, n_s_4, …) that is
     NOT in the training set: predict Q_LF at θ_4 from the 60 LF sims via
     nearest-parameter-neighbour and A_p-only linear regression.
  3. Apply the MF correction: Q_HR_pred(θ_4, z) = Q_LF_pred(θ_4, z) ·
     R_MF(A_p_4, z).
  4. Compare Q_HR_pred to the measured Q_HR(θ_4, z) from the HR summary.

Residuals < few % → MF correction generalises out-of-sample.
Large residuals → correction is sample-specific (n=3 overfitting).

Outputs
-------
  figures/analysis/mf_fourth_sim_test.png
  figures/analysis/mf_fourth_sim_test.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from collections import defaultdict

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
import sys as _sys_common
_sys_common.path.insert(0, str(Path(__file__).resolve().parent))
from common import data_dir
DATA = data_dir()
OUT = ROOT / "figures" / "analysis" / "04_hcd_mf"
OUT.mkdir(parents=True, exist_ok=True)
LF = DATA / "hcd_summary_lf.h5"
HR = DATA / "hcd_summary_hr.h5"

sys.path.insert(0, str(ROOT / "scripts"))
from mf_global_fit import (
    _load, _pair, build_samples, fit_global, predict,
    Z_FLAT_SAFE_MIN, Z_FLAT_SAFE_MAX, Z_REF, CLASSES,
)

QUANTITIES = [
    ("dndx_LLS", "dN/dX (LLS)"),
    ("dndx_subDLA", "dN/dX (subDLA)"),
    ("dndx_DLA", "dN/dX (DLA)"),
    ("omega_LLS", "Ω_HI (LLS)"),
    ("omega_subDLA", "Ω_HI (subDLA)"),
    ("omega_DLA", "Ω_HI (DLA)"),
]

# ns0.914… is the 4th HR sim with no LF counterpart
HELD_OUT_HR_PREFIX = "ns0.914Ap"


def lf_summary_arrays():
    """Return a dict of per-sim per-z records keyed for easy lookup."""
    with h5py.File(LF, "r") as f:
        sims = np.array([s.decode() for s in f["sim"][:]])
        zs = f["z"][:]
        Ap = f["params/Ap"][:]
        ns = f["params/ns"][:]
        data = {}
        for qkey, _ in QUANTITIES:
            prefix, cls = qkey.split("_", 1)
            ds = f[f"dndx/{cls}"] if prefix == "dndx" else f[f"Omega_HI/{cls}"]
            data[qkey] = ds[:]
    return sims, zs, Ap, ns, data


def held_out_hr(hr_all):
    """Find the HR sim whose name starts with HELD_OUT_HR_PREFIX."""
    held = {}
    for (sim, zr), r in hr_all.items():
        if sim.startswith(HELD_OUT_HR_PREFIX):
            held[(sim, zr)] = r
    return held


def predict_lf_at(theta_ap, theta_ns, z_target,
                  lf_sims, lf_zs, lf_Ap, lf_ns, lf_Q,
                  method="nearest"):
    """Predict Q_LF at parameters (A_p, n_s) and redshift z using the
    60-LF-sim sample.  Two methods:

      'nearest' : pick the single LF (sim, z) record closest in
                  (log A_p, n_s, z) Euclidean distance.  Simple baseline.
      'ap_regression' : linear regression  log10(Q_LF) = α + β · log10(A_p)
                        across sims at the nearest z, then evaluate at
                        log10(A_p_target).  Isolates the A_p dependence.
    """
    # restrict to z-matched LF records within tolerance
    dz = np.abs(lf_zs - z_target)
    sel = dz < 0.05
    if not sel.any():
        return np.nan
    Ap_sel = lf_Ap[sel]; ns_sel = lf_ns[sel]
    Q_sel = lf_Q[sel]
    # Remove sims with non-positive Q
    pos = Q_sel > 0
    Ap_sel = Ap_sel[pos]; ns_sel = ns_sel[pos]; Q_sel = Q_sel[pos]
    if len(Q_sel) == 0:
        return np.nan

    if method == "nearest":
        # Normalised distance over log-A_p and n_s.
        # Guard against zero range (all sims identical in one dimension).
        logAp_t = np.log10(theta_ap)
        logAp_s = np.log10(Ap_sel)
        rng_ap = np.ptp(logAp_s)
        rng_ns = np.ptp(ns_sel)
        if rng_ap == 0.0 and rng_ns == 0.0:
            # All candidates identical in both parameters — just return the first.
            return float(Q_sel[0])
        dAp = (logAp_s - logAp_t) / rng_ap if rng_ap > 0 else np.zeros_like(logAp_s)
        dns = (ns_sel - theta_ns) / rng_ns if rng_ns > 0 else np.zeros_like(ns_sel)
        d = np.sqrt(dAp ** 2 + dns ** 2)
        return float(Q_sel[int(np.argmin(d))])

    if method == "ap_regression":
        logQ = np.log10(Q_sel)
        logAp = np.log10(Ap_sel)
        A = np.column_stack([np.ones_like(logAp), logAp])
        coef, *_ = np.linalg.lstsq(A, logQ, rcond=None)
        logQ_pred = coef[0] + coef[1] * np.log10(theta_ap)
        return float(10 ** logQ_pred)

    raise ValueError(method)


def main():
    hr_all = _load(HR); lf_all = _load(LF)
    pairs = _pair(hr_all, lf_all)

    held_out = held_out_hr(hr_all)
    if not held_out:
        print("Error: no held-out HR sim found.")
        return
    held_sims = sorted({s for (s, _) in held_out.keys()})
    print(f"Held-out HR sim(s): {held_sims}")

    theta_ap = None; theta_ns = None
    for (_sim, _z), rec in held_out.items():
        theta_ap = rec["Ap"]; theta_ns = rec["ns"]; break
    print(f"  held-out θ: A_p = {theta_ap:.3e},  n_s = {theta_ns:.3f}")

    # LF suite arrays for prediction of Q_LF(θ_4, z)
    lf_sims_arr, lf_zs_arr, lf_Ap_arr, lf_ns_arr, lf_Q_map = lf_summary_arrays()

    # Fit global MF on the 3 training pairs for every quantity
    fits_G1 = {}
    fits_G2 = {}
    for qkey, _ in QUANTITIES:
        samples = build_samples(pairs, qkey)
        fits_G1[qkey] = fit_global(samples, use_z_drift=False)
        fits_G2[qkey] = fit_global(samples, use_z_drift=True)

    rows = []
    for (sim, zr), rec in sorted(held_out.items()):
        z = rec["z"]
        in_win = Z_FLAT_SAFE_MIN <= z <= Z_FLAT_SAFE_MAX
        for qkey, qlabel in QUANTITIES:
            hr_actual = rec[qkey]
            # predict LF at held-out θ via both methods
            lf_near = predict_lf_at(theta_ap, theta_ns, z,
                                    lf_sims_arr, lf_zs_arr, lf_Ap_arr, lf_ns_arr,
                                    lf_Q_map[qkey], method="nearest")
            lf_reg = predict_lf_at(theta_ap, theta_ns, z,
                                   lf_sims_arr, lf_zs_arr, lf_Ap_arr, lf_ns_arr,
                                   lf_Q_map[qkey], method="ap_regression")

            # MF predictions
            if in_win:
                R_G1 = predict(fits_G1[qkey], theta_ap, z)
                R_G2 = predict(fits_G2[qkey], theta_ap, z)
            else:
                R_G1 = R_G2 = np.nan

            hr_pred_near_G1 = lf_near * R_G1 if np.isfinite(R_G1) else np.nan
            hr_pred_near_G2 = lf_near * R_G2 if np.isfinite(R_G2) else np.nan
            hr_pred_reg_G1 = lf_reg * R_G1 if np.isfinite(R_G1) else np.nan
            hr_pred_reg_G2 = lf_reg * R_G2 if np.isfinite(R_G2) else np.nan

            # Residuals (frac)
            def rel(p, a): return (p - a) / a if (np.isfinite(p) and a) else np.nan
            rows.append({
                "quantity": qkey, "z": round(z, 3), "in_window": int(in_win),
                "hr_actual": hr_actual,
                "lf_near": lf_near, "lf_reg": lf_reg,
                "hr_pred_near_G1": hr_pred_near_G1,
                "hr_pred_near_G2": hr_pred_near_G2,
                "hr_pred_reg_G1": hr_pred_reg_G1,
                "hr_pred_reg_G2": hr_pred_reg_G2,
                "frac_err_near_G1": rel(hr_pred_near_G1, hr_actual),
                "frac_err_near_G2": rel(hr_pred_near_G2, hr_actual),
                "frac_err_reg_G1": rel(hr_pred_reg_G1, hr_actual),
                "frac_err_reg_G2": rel(hr_pred_reg_G2, hr_actual),
            })

    # Print compact summary
    print()
    print(f"{'quantity':<14s} {'z':>5s} {'win':>4s} "
          f"{'near+G1':>9s} {'near+G2':>9s} {'reg+G1':>9s} {'reg+G2':>9s}")
    for r in rows:
        mark = "in" if r["in_window"] else "OUT"
        def fmt(x): return f"{x*100:+7.2f}%" if np.isfinite(x) else "   —   "
        print(f"  {r['quantity']:<14s} {r['z']:>5.2f} {mark:>4s} "
              f"{fmt(r['frac_err_near_G1']):>9s} "
              f"{fmt(r['frac_err_near_G2']):>9s} "
              f"{fmt(r['frac_err_reg_G1']):>9s} "
              f"{fmt(r['frac_err_reg_G2']):>9s}")

    # Per-quantity summary inside window
    print()
    print("In-window summary  —  median |frac err| (prediction − actual) / actual:")
    print(f"  {'quantity':<14s} {'near+G1':>9s} {'near+G2':>9s} {'reg+G1':>9s} {'reg+G2':>9s}")
    for qkey, qlabel in QUANTITIES:
        errs = {"near+G1": [], "near+G2": [], "reg+G1": [], "reg+G2": []}
        for r in rows:
            if r["quantity"] != qkey or not r["in_window"]: continue
            for k in errs:
                v = r[f"frac_err_{k.replace('+', '_')}"]
                if np.isfinite(v):
                    errs[k].append(abs(v))
        def med(xs): return f"{np.median(xs)*100:>8.2f}%" if xs else "   —   "
        print(f"  {qkey:<14s} {med(errs['near+G1']):>9s} "
              f"{med(errs['near+G2']):>9s} "
              f"{med(errs['reg+G1']):>9s} "
              f"{med(errs['reg+G2']):>9s}")

    # CSV
    csv_p = DATA / "mf_fourth_sim_test.csv"
    with open(csv_p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\n  wrote {csv_p}")

    # Plot: predicted vs actual HR per quantity
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (qkey, qlabel) in zip(axes.flat, QUANTITIES):
        sub = [r for r in rows if r["quantity"] == qkey]
        sub.sort(key=lambda r: r["z"])
        z = np.array([r["z"] for r in sub])
        actual = np.array([r["hr_actual"] for r in sub])
        pred_G1 = np.array([r["hr_pred_near_G1"] for r in sub])
        pred_G2 = np.array([r["hr_pred_near_G2"] for r in sub])
        pred_reg_G1 = np.array([r["hr_pred_reg_G1"] for r in sub])
        in_win = np.array([r["in_window"] for r in sub]).astype(bool)

        ax.plot(z, actual, "ko-", lw=1.5, ms=6, label="HR measured (ns0.914)")
        ax.plot(z[in_win], pred_G1[in_win], "r^", ms=8,
                label="MF pred G1, nearest-LF")
        ax.plot(z[in_win], pred_G2[in_win], "C2s", ms=6, alpha=0.85,
                label="MF pred G2, nearest-LF")
        ax.plot(z[in_win], pred_reg_G1[in_win], "bv", ms=5, alpha=0.7,
                label="MF pred G1, A_p-regression")
        ax.axvspan(Z_FLAT_SAFE_MIN, Z_FLAT_SAFE_MAX,
                   color="gray", alpha=0.08, label="fit window")
        ax.set_yscale("log")
        ax.set_xlabel("z"); ax.set_ylabel(qlabel)
        ax.set_title(qlabel, fontsize=10)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)

    fig.suptitle(
        "Held-out 4th-HR-sim test (ns0.914, no matched LF pair):\n"
        "does the global MF correction predict HR from a LF predictor + R_MF?",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_png = OUT / "mf_fourth_sim_test.png"
    fig.savefig(out_png, dpi=120); plt.close(fig)
    print(f"  wrote {out_png}")


if __name__ == "__main__":
    main()
