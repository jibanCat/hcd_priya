"""
Global (across-z) MF correction for HCD quantities.

Replaces the earlier per-z linear-in-A_p fit, which over-saturates with
n=3 sims per bin.  Here we fit *all* (sim, z) points inside the flat-safe
window z ∈ [2.6, 4.6] simultaneously, with two nested models:

    Model G1:  R(A_p, z) = a(z) + b₀ · (A_p − ⟨A_p⟩)
               → 1 slope parameter, shared across z.
    Model G2:  R(A_p, z) = a(z) + [b₀ + b₁ · (z − z_ref)] · (A_p − ⟨A_p⟩)
               → 2 slope parameters (slope + mild z-drift).

a(z) is the *flat mean* of R at each z (3-sim average per z), so the
slope terms only have to capture A_p-dependence, not the overall
z-evolution of R̄.  Fitting therefore becomes a simple least-squares on
the residuals `Y = R - a(z)`.

A third ingredient — the "bad-fit flag":

    flag_z = 1   if the global model's residual |R − R_pred| at this z
                 exceeds FLAG_THRESHOLD (default 2%) relative to R̄_z.

Points with flag_z = 1 should fall back to flat or be skipped by the
downstream emulator.

Inputs  : figures/analysis/hcd_summary_{lf,hr}.h5
Outputs : figures/analysis/mf_global_fit_coefficients.csv
          figures/analysis/mf_global_fit_visual.png

Run:
    python3 scripts/mf_global_fit.py
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
Z_REF = 3.6                       # centre of the fit window
FLAG_THRESHOLD = 0.02             # "bad fit" if fractional resid > 2 %
CLASSES = ["LLS", "subDLA", "DLA"]
QUANTITIES = [
    ("dndx_LLS", "dN/dX (LLS)"),
    ("dndx_subDLA", "dN/dX (subDLA)"),
    ("dndx_DLA", "dN/dX (DLA)"),
    ("omega_LLS", "Ω_HI (LLS)"),
    ("omega_subDLA", "Ω_HI (subDLA)"),
    ("omega_DLA", "Ω_HI (DLA)"),
]


def _load(path):
    out = {}
    with h5py.File(path, "r") as f:
        sims = [s.decode() for s in f["sim"][:]]
        zs = f["z"][:]
        for i in range(len(sims)):
            r = {"z": float(zs[i]),
                 "Ap": float(f["params/Ap"][i]),
                 "ns": float(f["params/ns"][i])}
            for cls in CLASSES:
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


def build_samples(pairs, qkey):
    """Return (sim, z, Ap, R) tuples sorted by z, sim — all of them."""
    out = []
    for sim, z_hr, l, h in pairs:
        lv = l[qkey]; hv = h[qkey]
        if lv > 0 and np.isfinite(hv) and np.isfinite(lv):
            out.append((sim, z_hr, l["Ap"], hv / lv))
    return out


def fit_global(samples, use_z_drift=False, z_ref=Z_REF,
               z_min=Z_FLAT_SAFE_MIN, z_max=Z_FLAT_SAFE_MAX):
    """
    Fit R − a(z) = b0 · (A_p − ⟨A_p⟩)                          [use_z_drift=False]
                 = (b0 + b1·(z − z_ref)) · (A_p − ⟨A_p⟩)        [use_z_drift=True]

    where a(z) is the per-z mean of R across the 3 sims.
    Returns:
      Ap_ref        : pivot A_p (mean across samples in-window)
      az_map        : {z_rounded: a(z)}
      b0, b1        : fit coefficients (b1 = 0 if use_z_drift=False)
      fit_resid_rms : overall fractional RMS across samples
    """
    in_window = [(s, z, Ap, R) for (s, z, Ap, R) in samples
                 if z_min <= z <= z_max]
    if not in_window:
        return None

    Ap_arr = np.array([t[2] for t in in_window])
    z_arr = np.array([t[1] for t in in_window])
    R_arr = np.array([t[3] for t in in_window])
    Ap_ref = float(Ap_arr.mean())

    # Per-z mean — use rounded z for exact grouping
    by_z = defaultdict(list)
    for s, z, Ap, R in in_window:
        by_z[round(z, 2)].append(R)
    az_map = {z: float(np.mean(r)) for z, r in by_z.items()}
    az_arr = np.array([az_map[round(z, 2)] for z in z_arr])
    Y = R_arr - az_arr

    dAp = Ap_arr - Ap_ref
    if use_z_drift:
        M = np.column_stack([dAp, dAp * (z_arr - z_ref)])
        coef, *_ = np.linalg.lstsq(M, Y, rcond=None)
        b0, b1 = float(coef[0]), float(coef[1])
        pred = az_arr + b0 * dAp + b1 * dAp * (z_arr - z_ref)
    else:
        M = dAp.reshape(-1, 1)
        coef, *_ = np.linalg.lstsq(M, Y, rcond=None)
        b0 = float(coef[0])
        b1 = 0.0
        pred = az_arr + b0 * dAp

    resid = R_arr - pred
    resid_rms = float(np.sqrt(np.mean(resid ** 2)))

    return {
        "Ap_ref": Ap_ref,
        "z_ref": z_ref,
        "az_map": az_map,
        "b0": b0,
        "b1": b1,
        "resid_rms": resid_rms,
        "in_sample_n": len(in_window),
        "frac_rms_of_meanR": resid_rms / float(R_arr.mean()),
    }


def predict(fit, Ap, z):
    """Evaluate R at (A_p, z) using the global fit.

    If z is outside the fit window, a(z) is not available — we
    extrapolate using the edge a(z_min) or a(z_max) as a practical fallback
    (same-as-flat extrapolation).  Caller should also consult flag_z.
    """
    az_map = fit["az_map"]
    z_key = round(z, 2)
    if z_key in az_map:
        a_z = az_map[z_key]
    else:
        zs_fit = sorted(az_map)
        if z_key < zs_fit[0]:
            a_z = az_map[zs_fit[0]]
        elif z_key > zs_fit[-1]:
            a_z = az_map[zs_fit[-1]]
        else:
            # interpolate
            a_z = float(np.interp(z_key, zs_fit, [az_map[k] for k in zs_fit]))
    return (a_z
            + fit["b0"] * (Ap - fit["Ap_ref"])
            + fit["b1"] * (Ap - fit["Ap_ref"]) * (z - fit["z_ref"]))


def evaluate_flags(fit, samples, threshold=FLAG_THRESHOLD):
    """For each (sim, z) sample, compute per-z flag = 1 if |R - R_pred|/R̄_z > threshold."""
    flags = defaultdict(list)
    for sim, z, Ap, R in samples:
        R_pred = predict(fit, Ap, z)
        z_key = round(z, 2)
        if z_key in fit["az_map"]:
            R_bar = fit["az_map"][z_key]
        else:
            R_bar = fit["az_map"][min(fit["az_map"], key=lambda k: abs(k - z_key))]
        frac_err = abs(R - R_pred) / abs(R_bar) if R_bar else np.nan
        flags[z_key].append((sim, frac_err))
    per_z_flag = {}
    for z_key, items in flags.items():
        max_frac = max(frac for _, frac in items)
        per_z_flag[z_key] = {
            "max_frac_resid": float(max_frac),
            "flag": int(max_frac > threshold),
        }
    return per_z_flag


def main():
    hr = _load(HR); lf = _load(LF)
    pairs = _pair(hr, lf)

    all_rows = []
    slope_rows = []
    print(f"Global fit across z ∈ [{Z_FLAT_SAFE_MIN}, {Z_FLAT_SAFE_MAX}], "
          f"flag threshold {FLAG_THRESHOLD*100:.1f} %")
    print()
    print(f"  {'quantity':<14s} {'model':<8s} {'b0':>12s} {'b1':>12s} "
          f"{'resid':>8s} {'flagged_z':>12s}")
    for qkey, qlabel in QUANTITIES:
        samples = build_samples(pairs, qkey)
        fit1 = fit_global(samples, use_z_drift=False)
        fit2 = fit_global(samples, use_z_drift=True)

        flags1 = evaluate_flags(fit1, samples)
        flags2 = evaluate_flags(fit2, samples)
        flagged_z_1 = sorted(z for z, v in flags1.items() if v["flag"])
        flagged_z_2 = sorted(z for z, v in flags2.items() if v["flag"])

        print(f"  {qlabel:<14s} {'G1':<8s} "
              f"{fit1['b0']:>+12.3e} {'—':>12s} "
              f"{fit1['frac_rms_of_meanR']*100:>7.2f}% "
              f"{str(flagged_z_1):>12s}")
        print(f"  {qlabel:<14s} {'G2':<8s} "
              f"{fit2['b0']:>+12.3e} {fit2['b1']:>+12.3e} "
              f"{fit2['frac_rms_of_meanR']*100:>7.2f}% "
              f"{str(flagged_z_2):>12s}")

        slope_rows.append({
            "quantity": qkey, "model": "G1_single_slope",
            "b0": fit1["b0"], "b1": 0.0,
            "resid_rms": fit1["resid_rms"],
            "frac_rms_of_meanR": fit1["frac_rms_of_meanR"],
            "n_in_sample": fit1["in_sample_n"],
            "flagged_z_count": len(flagged_z_1),
            "flagged_z": str(flagged_z_1),
        })
        slope_rows.append({
            "quantity": qkey, "model": "G2_with_z_drift",
            "b0": fit2["b0"], "b1": fit2["b1"],
            "resid_rms": fit2["resid_rms"],
            "frac_rms_of_meanR": fit2["frac_rms_of_meanR"],
            "n_in_sample": fit2["in_sample_n"],
            "flagged_z_count": len(flagged_z_2),
            "flagged_z": str(flagged_z_2),
        })

        # Per-z table: a(z), predicted R, actual R for each sim, flag
        for z_key in sorted(fit1["az_map"]):
            samples_z = [(s, z, Ap, R) for (s, z, Ap, R) in samples
                         if round(z, 2) == z_key]
            for sim, z, Ap, R in samples_z:
                all_rows.append({
                    "quantity": qkey, "z": z_key, "sim": sim, "Ap": Ap,
                    "R_observed": R,
                    "a_z": fit1["az_map"][z_key],
                    "R_pred_G1": predict(fit1, Ap, z),
                    "R_pred_G2": predict(fit2, Ap, z),
                    "G1_flag": flags1[z_key]["flag"],
                    "G1_max_frac_resid": flags1[z_key]["max_frac_resid"],
                    "G2_flag": flags2[z_key]["flag"],
                    "G2_max_frac_resid": flags2[z_key]["max_frac_resid"],
                })

    # --- CSV output ---
    coeff_csv = DATA / "mf_global_fit_coefficients.csv"
    with open(coeff_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(slope_rows[0].keys()))
        w.writeheader()
        for r in slope_rows: w.writerow(r)
    print(f"\n  wrote {coeff_csv}")

    pred_csv = DATA / "mf_global_fit_predictions.csv"
    with open(pred_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        for r in all_rows: w.writerow(r)
    print(f"  wrote {pred_csv}")

    # --- Visualise: one row per quantity, panels = R vs A_p at 3 representative z ---
    fig, axes = plt.subplots(6, 3, figsize=(14, 18), sharex=False)
    z_picks = [2.8, 3.4, 4.2]
    for row, (qkey, qlabel) in enumerate(QUANTITIES):
        samples = build_samples(pairs, qkey)
        fit1 = fit_global(samples, use_z_drift=False)
        fit2 = fit_global(samples, use_z_drift=True)
        for col, z_p in enumerate(z_picks):
            ax = axes[row, col]
            triples_z = [(Ap, R) for (_s, z, Ap, R) in samples if abs(z - z_p) < 0.05]
            if not triples_z:
                ax.set_title(f"{qlabel}  z={z_p}: no data"); continue
            triples_z.sort()
            Ap_v = np.array([t[0] for t in triples_z])
            R_v = np.array([t[1] for t in triples_z])
            ax.scatter(Ap_v, R_v, s=80, color="k", zorder=5, label="data")
            x_fit = np.linspace(Ap_v.min() * 0.9, Ap_v.max() * 1.1, 100)
            ax.plot(x_fit, [predict(fit1, a, z_p) for a in x_fit],
                    "r-",  lw=1.5, label=f"G1 single-slope")
            ax.plot(x_fit, [predict(fit2, a, z_p) for a in x_fit],
                    "C2--", lw=1.5, label=f"G2 with z-drift")
            ax.axhline(fit1["az_map"].get(round(z_p, 2),
                                          np.mean(R_v)),
                       color="gray", lw=0.8, ls=":", label="flat a(z)")
            ax.set_title(f"{qlabel}  at z = {z_p:.1f}", fontsize=9)
            ax.set_xscale("log")
            if col == 0: ax.set_ylabel("R = Q_HR / Q_LF", fontsize=8)
            ax.grid(alpha=0.3, which="both")
            ax.legend(fontsize=6)

    fig.suptitle(
        "Global linear-in-A_p fit: single-slope (G1) vs slope-with-z-drift (G2)\n"
        "(fit sample = all points inside window, shown here at 3 representative z)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_png = OUT / "mf_global_fit_visual.png"
    fig.savefig(out_png, dpi=120); plt.close(fig)
    print(f"  wrote {out_png}")


if __name__ == "__main__":
    main()
