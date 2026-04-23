"""
Flat vs linear-in-A_p MF fit visualisation, with residuals, for all six
HCD scalar quantities.  Inside the flat-safe window z ∈ [2.6, 4.6] we
fit

    R(A_p) = a + b · (A_p - ⟨A_p⟩)                  (linear in A_p)
    R      = ⟨R⟩                                    (flat baseline)

to the 3 matched HR–LF sims at each z-bin.  This script generates three
figures and a CSV:

1. `mf_fit_vs_residuals_z3.png` — 2-row × 6-col layout at z = 3.0.
   Top: R vs A_p per sim with both fits overlaid.  Bottom: residuals.
2. `mf_slope_stability.png` — per-z linear-in-A_p slope for every
   quantity, with leave-one-out error bars (drop each sim, refit).
   Visualises the stability argument for requesting a 4th HR sim.
3. `mf_linear_coefficients_all.csv` — per (quantity, z) intercept,
   slope, flat-σ, linear-resid-σ, flat-σ / linear-σ ratio.
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

Z_TOL = 0.05
Z_FLAT_SAFE_MIN = 2.6
Z_FLAT_SAFE_MAX = 4.6
CLASSES = ["LLS", "subDLA", "DLA"]
QUANTITY_ORDER = [
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


def fit_flat_and_linear(Ap, R):
    """Return (flat_mean, intercept, slope, flat_rms, lin_rms)."""
    Ap = np.asarray(Ap, dtype=float); R = np.asarray(R, dtype=float)
    flat = float(R.mean())
    flat_rms = float(np.sqrt(np.mean((R - flat) ** 2)))
    if len(R) >= 2:
        A = np.vstack([np.ones_like(Ap), Ap - Ap.mean()]).T
        coef, *_ = np.linalg.lstsq(A, R, rcond=None)
        pred = A @ coef
        lin_rms = float(np.sqrt(np.mean((R - pred) ** 2)))
    else:
        coef = np.array([flat, 0.0]); lin_rms = flat_rms
    return flat, float(coef[0]), float(coef[1]), flat_rms, lin_rms


def build_R_table(pairs):
    """Return {quantity: {zkey: [(sim, Ap, R), ...]}}."""
    out = defaultdict(lambda: defaultdict(list))
    for sim, z_hr, l, h in pairs:
        zkey = round(z_hr, 2)
        for q_prefix, _ in QUANTITY_ORDER:
            lv = l[q_prefix.replace("dndx_", "dndx_").replace("omega_", "omega_")]
            hv = h[q_prefix]
            if lv > 0 and np.isfinite(hv) and np.isfinite(lv):
                R = hv / lv
                out[q_prefix][zkey].append((sim, l["Ap"], R))
    return out


def leave_one_out_slope(Ap, R):
    """Return list of slopes with each sim left out."""
    slopes = []
    if len(R) < 3:
        return slopes
    for i in range(len(R)):
        keep = np.ones(len(R), dtype=bool); keep[i] = False
        Ap_s = np.array(Ap)[keep]; R_s = np.array(R)[keep]
        if len(R_s) < 2:
            continue
        A = np.vstack([np.ones_like(Ap_s), Ap_s - Ap_s.mean()]).T
        coef, *_ = np.linalg.lstsq(A, R_s, rcond=None)
        slopes.append(float(coef[1]))
    return slopes


def plot_fits_and_residuals(R_table, z_target, outpath):
    """2-row × 6-col at fixed z."""
    fig, axes = plt.subplots(2, 6, figsize=(22, 7),
                             gridspec_kw=dict(height_ratios=[2, 1]))

    for col, (qkey, qlabel) in enumerate(QUANTITY_ORDER):
        triples = R_table.get(qkey, {}).get(round(z_target, 2), [])
        if not triples:
            axes[0, col].set_title(f"{qlabel}\n(no data at z={z_target})")
            continue
        triples = sorted(triples, key=lambda t: t[1])
        sims = [t[0] for t in triples]
        Ap = np.array([t[1] for t in triples])
        R = np.array([t[2] for t in triples])

        flat, a, b, flat_rms, lin_rms = fit_flat_and_linear(Ap, R)

        # Top: scatter + fits
        ax = axes[0, col]
        ax.scatter(Ap, R, s=80, color="k", zorder=4,
                   label=f"matched sims ({len(R)})")
        # Flat fit
        x_fit = np.linspace(Ap.min() * 0.95, Ap.max() * 1.05, 50)
        ax.plot(x_fit, np.full_like(x_fit, flat),
                "k--", lw=1.2, alpha=0.7,
                label=f"flat  R̄={flat:.3f}, σ={flat_rms*100/abs(flat):.2f}%")
        # Linear fit
        ax.plot(x_fit, a + b * (x_fit - Ap.mean()),
                "r-", lw=1.5,
                label=f"linear  a={a:.3f}, b={b:+.1e}, σ={lin_rms*100/abs(flat):.2f}%")
        ax.set_title(f"{qlabel}", fontsize=10)
        ax.set_ylabel("R = Q_HR / Q_LF" if col == 0 else "")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6.5, loc="best")
        ax.set_xscale("log")

        # Bottom: residuals
        axr = axes[1, col]
        flat_resid = R - flat
        lin_pred = a + b * (Ap - Ap.mean())
        lin_resid = R - lin_pred
        axr.scatter(Ap, flat_resid, s=60, marker="s", color="k",
                    label="flat", zorder=4)
        axr.scatter(Ap, lin_resid, s=60, marker="o", color="r",
                    label="linear", zorder=5)
        axr.axhline(0, color="gray", lw=0.8, ls=":")
        axr.set_xscale("log")
        axr.set_xlabel("A_p")
        axr.set_ylabel("residual" if col == 0 else "")
        axr.grid(alpha=0.3)
        axr.legend(fontsize=7, loc="best")

    fig.suptitle(
        f"Flat vs linear-in-A_p MF fit at z = {z_target:.1f}  "
        f"(3 matched HR–LF sims)\n"
        f"top row: R(A_p) with both fits  —  bottom row: residuals"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(outpath, dpi=120); plt.close(fig)


def plot_slope_stability(R_table, outpath):
    """Per-z linear slope with leave-one-out range per quantity.

    The leave-one-out spread of the slope is the analytical stand-in for
    the missing 2nd residual DOF with only 3 sims.  Large LOO spread →
    slope is essentially unconstrained.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    axes = axes.flatten()
    for ax, (qkey, qlabel) in zip(axes, QUANTITY_ORDER):
        zs_in = []; slopes = []; loo_min = []; loo_max = []
        for zkey in sorted(R_table[qkey].keys()):
            if not (Z_FLAT_SAFE_MIN <= zkey <= Z_FLAT_SAFE_MAX):
                continue
            triples = R_table[qkey][zkey]
            if len(triples) < 3:
                continue
            Ap = np.array([t[1] for t in triples])
            R = np.array([t[2] for t in triples])
            _, _, b, _, _ = fit_flat_and_linear(Ap, R)
            loo = leave_one_out_slope(Ap, R)
            zs_in.append(zkey)
            slopes.append(b)
            loo_min.append(min(loo))
            loo_max.append(max(loo))
        if not zs_in:
            continue
        zs_in = np.array(zs_in)
        ax.plot(zs_in, slopes, "o-", color="C3", zorder=5, label="3-sim fit slope")
        ax.fill_between(zs_in, loo_min, loo_max, color="C3", alpha=0.25,
                        label="leave-one-out range")
        ax.axhline(0, color="k", lw=0.6, ls="--")
        ax.axvspan(Z_FLAT_SAFE_MIN, Z_FLAT_SAFE_MAX, color="gray",
                   alpha=0.08, label="flat-safe window")
        ax.set_title(qlabel, fontsize=11)
        ax.set_xlabel("z"); ax.set_ylabel("slope (1/A_p)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle(
        "Per-z linear-in-A_p slope stability — 3-sim fit, leave-one-out range\n"
        "Large LOO envelopes → slope is essentially unconstrained; a 4th HR sim "
        "would add the residual DOF needed for proper error bars.",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(outpath, dpi=120); plt.close(fig)


def main():
    hr = _load(HR); lf = _load(LF)
    pairs = _pair(hr, lf)
    R_table = build_R_table(pairs)

    # Figure 1: fits + residuals at z=3
    plot_fits_and_residuals(R_table, z_target=3.0,
                            outpath=OUT / "mf_fit_vs_residuals_z3.png")
    print(f"  wrote {OUT/'mf_fit_vs_residuals_z3.png'}")

    # Figure 2: slope stability (4th-sim argument)
    plot_slope_stability(R_table, outpath=OUT / "mf_slope_stability.png")
    print(f"  wrote {OUT/'mf_slope_stability.png'}")

    # CSV of all per-(quantity, z) linear coefficients including LLS
    csv_p = DATA / "mf_linear_coefficients_all.csv"
    with open(csv_p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "quantity", "z", "n_sims", "in_flat_window",
            "Ap_ref", "flat_mean", "lin_intercept", "lin_slope",
            "flat_rms", "lin_resid_rms",
            "flat_frac_spread", "lin_frac_spread",
            "loo_slope_min", "loo_slope_max",
        ])
        for qkey, _ in QUANTITY_ORDER:
            for zkey in sorted(R_table[qkey].keys()):
                triples = R_table[qkey][zkey]
                if len(triples) < 2:
                    continue
                Ap = np.array([t[1] for t in triples])
                R = np.array([t[2] for t in triples])
                flat, a, b, flat_rms, lin_rms = fit_flat_and_linear(Ap, R)
                loo = leave_one_out_slope(Ap, R)
                in_win = Z_FLAT_SAFE_MIN <= zkey <= Z_FLAT_SAFE_MAX
                w.writerow([
                    qkey, round(zkey, 3), len(triples),
                    int(in_win), float(Ap.mean()),
                    flat, a, b, flat_rms, lin_rms,
                    flat_rms / abs(flat) if flat else np.nan,
                    lin_rms / abs(flat) if flat else np.nan,
                    min(loo) if loo else "", max(loo) if loo else "",
                ])
    print(f"  wrote {csv_p}")


if __name__ == "__main__":
    main()
