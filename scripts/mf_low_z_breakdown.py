"""
Step-by-step visualisation of how the MF correction evolves as we move
from z in the flat-safe window to the z = 2.0–2.4 regime where the
flat-MF assumption breaks.  Four strategies are compared:

  S1. Flat              R̄_z = mean of the 3 sims at this z.
  S2. Linear-in-A_p     R(A_p) = a_z + b_z · (A_p - ⟨A_p⟩), fit per z.
  S3. Extrapolated-slope
                        R(A_p) = a_z + b̄_win · (A_p - ⟨A_p⟩),  where
                        b̄_win is the *median* linear slope from the
                        in-window z-bins (z ∈ [2.6, 4.6]).  Uses the
                        stable slope estimate instead of re-fitting at a
                        z-extreme with small counts.
  S4. No correction     R = 1.

For each strategy we report the residual RMS at z ∈ {2.0, 2.2, 2.4} and
compare to a reference mid-z (z = 3.4) where the flat-MF is already
near-safe.  Outputs one multi-panel figure plus a CSV table.

Note: only runs on Ω_HI(DLA) as the test case — it is the quantity with
the largest in-window flat-MF deviation AND the clearest z-extreme
breakdown.  Same pipeline works for any other quantity by changing QKEY.
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
OUT = ROOT / "figures" / "analysis" / "04_hcd_mf"
DATA = ROOT / "figures" / "analysis" / "data"
OUT.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)
LF = DATA / "hcd_summary_lf.h5"
HR = DATA / "hcd_summary_hr.h5"

Z_TOL = 0.05
Z_FLAT_SAFE_MIN = 2.6
Z_FLAT_SAFE_MAX = 4.6
QKEY = "omega_DLA"
Q_LABEL = "Ω_HI (DLA)"
Z_LOW = [2.0, 2.2, 2.4]    # low-z extreme
Z_REF = 3.4                # mid-z reference
Z_HIGH = [4.8, 5.0, 5.2, 5.4]


def _load(path):
    out = {}
    with h5py.File(path, "r") as f:
        sims = [s.decode() for s in f["sim"][:]]
        zs = f["z"][:]
        for i in range(len(sims)):
            r = {"z": float(zs[i]),
                 "Ap": float(f["params/Ap"][i]),
                 "ns": float(f["params/ns"][i])}
            for cls in ["LLS", "subDLA", "DLA"]:
                r[f"dndx_{cls}"] = float(f[f"dndx/{cls}"][i])
                r[f"omega_{cls}"] = float(f[f"Omega_HI/{cls}"][i])
                r[f"counts_{cls}"] = int(f[f"counts/{cls}"][i])
            out[(sims[i], round(float(zs[i]), 3))] = r
    return out


def _pair(hr, lf):
    lf_by_sim = defaultdict(list)
    for (sim, _), r in lf.items():
        lf_by_sim[sim].append(r)
    pairs = []
    for (sim, _), h in hr.items():
        if sim not in lf_by_sim:
            continue
        best = min(lf_by_sim[sim], key=lambda r: abs(r["z"] - h["z"]))
        if abs(best["z"] - h["z"]) <= Z_TOL:
            pairs.append((sim, h["z"], best, h))
    return pairs


def build_Rz(pairs, qkey):
    out = defaultdict(list)
    for sim, z_hr, l, h in pairs:
        zkey = round(z_hr, 2)
        lv = l[qkey]; hv = h[qkey]
        if lv > 0 and np.isfinite(hv) and np.isfinite(lv):
            out[zkey].append((sim, l["Ap"], hv / lv))
    return out


def fit_linear(Ap, R):
    A = np.vstack([np.ones_like(Ap), Ap - Ap.mean()]).T
    coef, *_ = np.linalg.lstsq(A, R, rcond=None)
    return float(coef[0]), float(coef[1]), float(Ap.mean())


def window_median_slope(R_by_z):
    slopes = []
    for z, triples in R_by_z.items():
        if not (Z_FLAT_SAFE_MIN <= z <= Z_FLAT_SAFE_MAX) or len(triples) < 2:
            continue
        Ap = np.array([t[1] for t in triples]); R = np.array([t[2] for t in triples])
        _, b, _ = fit_linear(Ap, R)
        slopes.append(b)
    return float(np.median(slopes)) if slopes else 0.0


def evaluate_strategies(Ap, R, b_win):
    """Return {strategy: (prediction_array, residual_rms, strategy_label_with_fit)}."""
    Ap_ref = Ap.mean()
    flat = float(R.mean())
    a, b, _ = fit_linear(Ap, R)
    # S3: use intercept from data, slope from window median
    # Intercept for extrapolated model = R̄ (same as flat at A_p_ref)
    # (since ΣR_i = ΣR_pred implies intercept = R̄)
    a_ext = flat
    strategies = {
        "S1_flat":    (np.full_like(Ap, flat),
                       f"Flat   R̄={flat:.4f}"),
        "S2_linear":  (a + b * (Ap - Ap_ref),
                       f"Linear (fit)  a={a:.4f}  b={b:+.2e}"),
        "S3_extrap":  (a_ext + b_win * (Ap - Ap_ref),
                       f"Extrap.-slope  a={a_ext:.4f}  b̄_win={b_win:+.2e}"),
        "S4_none":    (np.ones_like(Ap),
                       "No correction  R≡1"),
    }
    rms = {}
    for k, (pred, _label) in strategies.items():
        rms[k] = float(np.sqrt(np.mean((R - pred) ** 2)))
    return strategies, rms


def main():
    hr = _load(HR); lf = _load(LF)
    pairs = _pair(hr, lf)
    R_by_z = build_Rz(pairs, QKEY)

    b_win = window_median_slope(R_by_z)
    print(f"Window median linear slope for {QKEY}: {b_win:+.3e} per A_p")

    # ------- Figure: z_low (2.0, 2.2, 2.4), z_ref, z_high (4.8, 5.0, 5.2, 5.4) -------
    z_list = Z_LOW + [Z_REF] + Z_HIGH
    n_z = len(z_list)
    fig, axes = plt.subplots(2, n_z, figsize=(3.2 * n_z, 7),
                             gridspec_kw=dict(height_ratios=[2.0, 1.0]))

    csv_rows = []
    for col, z in enumerate(z_list):
        triples = R_by_z.get(round(z, 2), [])
        if not triples:
            axes[0, col].set_title(f"z={z:.1f}  (no data)")
            continue
        triples = sorted(triples, key=lambda t: t[1])
        Ap = np.array([t[1] for t in triples]); R = np.array([t[2] for t in triples])

        strategies, rms = evaluate_strategies(Ap, R, b_win)

        # Top panel: scatter + all strategies
        ax = axes[0, col]
        ax.scatter(Ap, R, s=90, color="k", zorder=6,
                   label=f"3 matched sims")
        x_fit = np.linspace(Ap.min() * 0.9, Ap.max() * 1.1, 100)

        styles = {
            "S1_flat":   ("k", "--", "flat"),
            "S2_linear": ("r", "-",  "linear (fit)"),
            "S3_extrap": ("C2", "-.", "extrap slope"),
            "S4_none":   ("gray", ":", "no corr"),
        }
        for k, (color, ls, lbl) in styles.items():
            pred, label = strategies[k]
            if k == "S1_flat":
                y = np.full_like(x_fit, pred[0])
            elif k == "S2_linear":
                _, _, Ap_mean = fit_linear(Ap, R)
                a, b, _ = fit_linear(Ap, R)
                y = a + b * (x_fit - Ap_mean)
            elif k == "S3_extrap":
                flat = float(R.mean())
                Ap_mean = Ap.mean()
                y = flat + b_win * (x_fit - Ap_mean)
            else:
                y = np.ones_like(x_fit)
            ax.plot(x_fit, y, color=color, ls=ls, lw=1.5,
                    label=f"{lbl}  σ={rms[k]*100/abs(R.mean()):.2f}%")

        in_win = Z_FLAT_SAFE_MIN <= z <= Z_FLAT_SAFE_MAX
        title_tag = "[in window]" if in_win else "[OUT-of-window]"
        ax.set_title(f"{Q_LABEL} at z = {z:.1f}\n{title_tag}", fontsize=9)
        ax.set_xscale("log")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=6.5, loc="best")
        if col == 0:
            ax.set_ylabel("R = Q_HR / Q_LF")

        # Bottom panel: residuals per strategy
        axr = axes[1, col]
        markers = {"S1_flat": "s", "S2_linear": "o", "S3_extrap": "^", "S4_none": "x"}
        colors = {"S1_flat": "k", "S2_linear": "r", "S3_extrap": "C2", "S4_none": "gray"}
        for k in styles:
            pred, _ = strategies[k]
            axr.scatter(Ap, R - pred,
                        marker=markers[k], s=50, color=colors[k], zorder=5)
        axr.axhline(0, color="gray", lw=0.7, ls=":")
        axr.set_xscale("log")
        axr.set_xlabel("A_p")
        if col == 0:
            axr.set_ylabel("residual")
        axr.grid(alpha=0.3)

        # CSV rows
        for k, rms_val in rms.items():
            csv_rows.append({
                "z": z, "strategy": k,
                "rms": rms_val,
                "frac_rms_of_R": rms_val / abs(R.mean()) if R.mean() else np.nan,
                "R_mean": float(R.mean()),
                "n_sims": len(R),
                "in_window": int(Z_FLAT_SAFE_MIN <= z <= Z_FLAT_SAFE_MAX),
            })

    fig.suptitle(
        f"{Q_LABEL}:  how MF strategies compare across z\n"
        f"S1=flat   S2=linear-in-A_p fit per z   "
        f"S3=intercept from data, slope from window median   S4=no correction",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "mf_low_z_breakdown.png", dpi=120); plt.close(fig)
    print(f"  wrote {OUT/'mf_low_z_breakdown.png'}")

    # CSV
    csv_p = DATA / "mf_low_z_breakdown.csv"
    with open(csv_p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        for r in csv_rows: w.writerow(r)
    print(f"  wrote {csv_p}")

    # Brief verdict table
    print()
    print(f"Summary — fractional residual RMS (σ_R / R̄) per strategy, {Q_LABEL}")
    print(f"  {'z':>4s}  {'S1_flat':>10s} {'S2_linear':>10s} {'S3_extrap':>10s} {'S4_none':>10s}")
    by_z_rms = defaultdict(dict)
    for r in csv_rows:
        by_z_rms[r["z"]][r["strategy"]] = r["frac_rms_of_R"]
    for z in z_list:
        rms = by_z_rms.get(z, {})
        if not rms:
            continue
        print(f"  {z:>4.1f}  {rms['S1_flat']*100:>9.2f}% "
              f"{rms['S2_linear']*100:>9.2f}% "
              f"{rms['S3_extrap']*100:>9.2f}% "
              f"{rms['S4_none']*100:>9.2f}%")


if __name__ == "__main__":
    main()
