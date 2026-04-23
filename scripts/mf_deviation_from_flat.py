"""
Tabulate the deviation from the "flat resolution-correction" MF assumption
that R(z, θ) = Q_HR(z, θ) / Q_LF(z, θ) is only weakly θ-dependent.

For each HCD quantity and each matched z, we compute the mean R across
the 3 HR–LF matched sims and the relative spread σ(R)/|R̄|.  If
σ(R)/|R̄| is small at every z, a flat MF (single multiplicative curve
per z, shared across all parameter points) is safe.

Outputs
-------
  figures/analysis/mf_deviation_from_flat.csv   (machine-readable)
  stdout prints a compact human-readable table
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from collections import defaultdict

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures" / "analysis" / "04_hcd_mf"
DATA = ROOT / "figures" / "analysis" / "data"
OUT.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)
LF = DATA / "hcd_summary_lf.h5"
HR = DATA / "hcd_summary_hr.h5"

Z_TOL = 0.05
CLASSES = ["LLS", "subDLA", "DLA"]
QUANTITIES = [("dndx", "dN/dX"), ("omega", "Ω_HI")]

# z-range where the flat-MF assumption is proposed to be valid.  Outside
# this window the HR/LF ratio picks up strong scatter from small absorber
# counts — we flag those rather than pretend a flat correction works.
Z_FLAT_SAFE_MIN = 2.6
Z_FLAT_SAFE_MAX = 4.6


def _load(path):
    out = {}
    with h5py.File(path, "r") as f:
        sims = [s.decode() for s in f["sim"][:]]
        zs = f["z"][:]
        for i in range(len(sims)):
            r = {"z": float(zs[i])}
            for cls in CLASSES:
                r[f"dndx_{cls}"] = float(f[f"dndx/{cls}"][i])
                r[f"omega_{cls}"] = float(f[f"Omega_HI/{cls}"][i])
            out[(sims[i], round(float(zs[i]), 3))] = r
    return out


def _pair(hr, lf):
    lf_by_sim = defaultdict(list)
    for (sim, _), rec in lf.items():
        lf_by_sim[sim].append(rec)
    pairs = []
    for (sim, _), h in hr.items():
        if sim not in lf_by_sim:
            continue
        best = min(lf_by_sim[sim], key=lambda r: abs(r["z"] - h["z"]))
        if abs(best["z"] - h["z"]) <= Z_TOL:
            pairs.append((sim, h["z"], best, h))
    return pairs


def _fit_linear_in_ap(Ap_vals, R_vals):
    """Return (intercept, slope, residual_rms) for R = a + b · (A_p - <A_p>)."""
    Ap = np.asarray(Ap_vals, dtype=float)
    R = np.asarray(R_vals, dtype=float)
    if len(R) < 2:
        return (float(R.mean()), 0.0, 0.0)
    A = np.vstack([np.ones_like(Ap), Ap - Ap.mean()]).T
    coef, *_ = np.linalg.lstsq(A, R, rcond=None)
    resid = R - A @ coef
    rms = float(np.sqrt(np.mean(resid ** 2)))
    return (float(coef[0]), float(coef[1]), rms)


def main():
    hr = _load(HR)
    lf = _load(LF)
    pairs = _pair(hr, lf)

    # Build R[quantity][z] = list of (sim, R) tuples
    from collections import defaultdict
    R = defaultdict(lambda: defaultdict(list))
    for sim, z, l, h in pairs:
        zkey = round(z, 2)
        for q_prefix, _q_label in QUANTITIES:
            for cls in CLASSES:
                k = f"{q_prefix}_{cls}"
                if l[k] > 0 and np.isfinite(h[k]) and np.isfinite(l[k]):
                    R[k][zkey].append((sim, h[k] / l[k]))

    # Need the LF Ap per sim for the linear fit; grab from the lf dict directly.
    # Any z-row has the same Ap per sim, so pull the first one.
    lf_ap = {}
    for (sim, _), rec in lf.items():
        if sim not in lf_ap:
            lf_ap[sim] = rec.get("Ap", np.nan) if False else None
    # Actually the LF summary stored under the 'params/Ap' dataset; re-read just that.
    with h5py.File(LF, "r") as f:
        sims_all = [s.decode() for s in f["sim"][:]]
        Ap_all = f["params/Ap"][:]
        for s, a in zip(sims_all, Ap_all):
            lf_ap[s] = float(a)

    # Tabulate per (quantity, z)
    rows = []
    for q_prefix, q_label in QUANTITIES:
        for cls in CLASSES:
            key = f"{q_prefix}_{cls}"
            for zkey in sorted(R[key].keys()):
                triples = R[key][zkey]
                if len(triples) < 2:
                    continue
                sims_here = [s for s, _ in triples]
                vals = np.array([v for _, v in triples])
                R_mean = float(vals.mean())
                R_std = float(vals.std(ddof=1))
                frac = R_std / abs(R_mean) if R_mean != 0 else np.nan

                Ap_vec = np.array([lf_ap[s] for s in sims_here])
                a, b, lin_rms = _fit_linear_in_ap(Ap_vec, vals)
                lin_frac = lin_rms / abs(R_mean) if R_mean != 0 else np.nan
                improvement = (1.0 - lin_frac / frac) if frac > 0 else 0.0
                in_window = (zkey >= Z_FLAT_SAFE_MIN) and (zkey <= Z_FLAT_SAFE_MAX)

                rows.append({
                    "quantity": key,
                    "z": zkey,
                    "n_sims": int(len(vals)),
                    "in_flat_window": int(in_window),
                    "R_mean": R_mean,
                    "R_std_across_sims": R_std,
                    "frac_spread": frac,
                    "lin_intercept": a,
                    "lin_slope_Ap": b,
                    "lin_resid_rms": lin_rms,
                    "lin_frac_spread": lin_frac,
                    "fractional_improvement": improvement,
                })

    # --- Human-readable print ---
    print(f"Flat-MF window: z ∈ [{Z_FLAT_SAFE_MIN}, {Z_FLAT_SAFE_MAX}]")
    print(f"{'quantity':<14s} {'z':>5s} {'in_win':>6s} {'R̄':>8s} "
          f"{'σ_flat/R̄':>9s} {'σ_lin/R̄':>9s} {'lin improve':>11s}")
    for r in rows:
        mark = "in" if r["in_flat_window"] else "OUT"
        print(f"{r['quantity']:<14s} {r['z']:>5.2f} {mark:>6s} "
              f"{r['R_mean']:>8.4f} "
              f"{r['frac_spread']*100:>8.2f}% "
              f"{r['lin_frac_spread']*100:>8.2f}% "
              f"{r['fractional_improvement']*100:>10.1f}%")

    # --- Summary per quantity, separating in-window vs full range ---
    print()
    print("Summary per quantity  —  flat-MF deviation inside z ∈ [2.6, 4.6]")
    print(f"  {'quantity':<14s} {'flat σ/R̄ med':>14s} {'flat σ/R̄ max':>14s} "
          f"{'lin σ/R̄ med':>13s} {'lin σ/R̄ max':>13s} {'verdict':>22s}")
    grouped = defaultdict(list)
    for r in rows:
        if r["in_flat_window"]:
            grouped[r["quantity"]].append((r["frac_spread"], r["lin_frac_spread"]))
    for q, pairs_ in grouped.items():
        flat = np.array([p[0] for p in pairs_])
        lin = np.array([p[1] for p in pairs_])
        verdict = ("flat safe, < 1 %" if flat.max() < 0.01
                   else "marginal, 1-3 %" if flat.max() < 0.03
                   else "linear-in-A_p needed" if lin.max() < 0.03
                   else "GP / more HR sims")
        print(f"  {q:<14s} {np.median(flat)*100:>13.2f}% {flat.max()*100:>13.2f}% "
              f"{np.median(lin)*100:>12.2f}% {lin.max()*100:>12.2f}%  {verdict:>22s}")

    print()
    print("Summary per quantity  —  same over full z range")
    print(f"  {'quantity':<14s} {'flat σ/R̄ med':>14s} {'flat σ/R̄ max':>14s} "
          f"{'flat max @ z':>12s}")
    grp_all = defaultdict(list)
    for r in rows:
        grp_all[r["quantity"]].append((r["z"], r["frac_spread"]))
    for q, pairs_ in grp_all.items():
        zs = np.array([p[0] for p in pairs_])
        fl = np.array([p[1] for p in pairs_])
        i = int(np.argmax(fl))
        print(f"  {q:<14s} {np.median(fl)*100:>13.2f}% {fl.max()*100:>13.2f}% "
              f"{zs[i]:>12.2f}")

    # CSV
    csv_p = DATA / "mf_deviation_from_flat.csv"
    with open(csv_p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\n  wrote {csv_p}")


if __name__ == "__main__":
    main()
