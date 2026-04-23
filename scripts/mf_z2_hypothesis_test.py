"""
Hypothesis test for why the flat-MF assumption breaks at z = 2–2.4
and z = 4.8–5.4, plus per-z linear-in-A_p fit coefficients for Ω_HI(DLA)
(the only quantity that "needs" linear-MF inside the flat-safe window).

Hypotheses for the z-extreme break
-----------------------------------
H0  Absorber counts are smaller at z < 2.4 and z > 4.8, so the sample-
    size floor on σ(R) = σ(Q_HR/Q_LF) is larger (Poisson-dominated).
H1  Physics differs at z-extremes (e.g. different UV-background era,
    different halo-population dominance) so the HR/LF ratio genuinely
    picks up parameter dependence.

Test
----
For every (quantity, z) we compare

    σ_observed_across_sims    — spread of R across the 3 matched pairs
    σ_Poisson_expected        — quadrature σ that Poisson alone predicts

If σ_observed ≈ σ_Poisson at z-extremes (within ~1.3×), H0 is favoured.
If σ_observed >> σ_Poisson at z-extremes, H1 is favoured.

σ_Poisson for R = Q_HR/Q_LF is derived from per-sim absorber counts:
  σ(R)/R ≈ √(1/n_HR + 1/n_LF) .

Outputs
-------
  figures/analysis/mf_z2_hypothesis_test.png
  figures/analysis/mf_z2_hypothesis_test.csv
  figures/analysis/mf_linear_coefficients_omega_dla.csv

Run:
    python3 scripts/mf_z2_hypothesis_test.py
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


def main():
    hr = _load(HR); lf = _load(LF)
    pairs = _pair(hr, lf)

    # R[quantity][zkey] = list of dicts with R, sigma_poisson
    by_qz = defaultdict(lambda: defaultdict(list))
    for sim, z_hr, l, h in pairs:
        zkey = round(z_hr, 2)
        for q_prefix in ["dndx", "omega"]:
            for cls in CLASSES:
                key = f"{q_prefix}_{cls}"
                lv = l[key]; hv = h[key]
                if lv > 0 and np.isfinite(hv) and np.isfinite(lv):
                    R = hv / lv
                    n_hr = h[f"counts_{cls}"]
                    n_lf = l[f"counts_{cls}"]
                    if n_hr < 2 or n_lf < 2:
                        continue
                    sigma_rel_poisson = float(np.sqrt(1.0 / n_hr + 1.0 / n_lf))
                    by_qz[key][zkey].append({
                        "sim": sim,
                        "R": R,
                        "Ap": l["Ap"],
                        "n_hr": n_hr,
                        "n_lf": n_lf,
                        "sigma_R_poisson": R * sigma_rel_poisson,
                    })

    # Per (quantity, z): σ_observed across sims, σ_Poisson (average),
    # ratio σ_obs / σ_Poi.
    rows = []
    for qkey, by_z in by_qz.items():
        for zkey, samples in sorted(by_z.items()):
            if len(samples) < 2:
                continue
            Rs = np.array([s["R"] for s in samples])
            sig_poi = np.array([s["sigma_R_poisson"] for s in samples])
            obs = float(Rs.std(ddof=1))
            poi = float(np.sqrt(np.mean(sig_poi ** 2)))
            frac_obs = obs / abs(Rs.mean())
            frac_poi = poi / abs(Rs.mean())
            ratio = obs / poi if poi > 0 else np.nan
            in_win = Z_FLAT_SAFE_MIN <= zkey <= Z_FLAT_SAFE_MAX
            rows.append({
                "quantity": qkey, "z": zkey, "n_sims": len(samples),
                "R_mean": float(Rs.mean()),
                "sigma_observed": obs, "frac_observed": frac_obs,
                "sigma_poisson_mean": poi, "frac_poisson": frac_poi,
                "ratio_obs_to_poisson": ratio,
                "in_flat_window": int(in_win),
            })

    print("Per-quantity table — σ_observed vs σ_Poisson across 3 matched sims")
    print()
    print(f"{'quantity':<13s} {'z':>5s} {'n':>2s} {'R̄':>7s} "
          f"{'σ_obs%':>7s} {'σ_Poi%':>7s} {'ratio':>7s} {'in_win':>6s}")
    for r in rows:
        print(f"{r['quantity']:<13s} {r['z']:>5.2f} {r['n_sims']:>2d} "
              f"{r['R_mean']:>7.3f} {r['frac_observed']*100:>6.2f}% "
              f"{r['frac_poisson']*100:>6.2f}% "
              f"{r['ratio_obs_to_poisson']:>6.2f}× "
              f"{'in' if r['in_flat_window'] else 'OUT':>6s}")

    # Summary: median ratio inside vs outside window per quantity
    print()
    print("Summary — median σ_obs / σ_Poi ratio (>1 means param-dependent signal beyond Poisson)")
    print(f"  {'quantity':<13s} {'inside window':>18s} {'outside window':>18s}")
    grp = defaultdict(lambda: {"in": [], "out": []})
    for r in rows:
        grp[r["quantity"]]["in" if r["in_flat_window"] else "out"].append(r["ratio_obs_to_poisson"])
    for q, d in grp.items():
        mi = np.median(d["in"]) if d["in"] else np.nan
        mo = np.median(d["out"]) if d["out"] else np.nan
        print(f"  {q:<13s} {mi:>17.2f}× {mo:>17.2f}×")

    # CSV
    csv_p = DATA / "mf_z2_hypothesis_test.csv"
    with open(csv_p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\n  wrote {csv_p}")

    # ---- Plot: σ_obs vs σ_Poi per quantity, colour by in/out window ----
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    quantities = [("dndx_LLS", "dN/dX (LLS)"), ("dndx_subDLA", "dN/dX (subDLA)"),
                  ("dndx_DLA", "dN/dX (DLA)"),
                  ("omega_LLS", "Ω_HI (LLS)"), ("omega_subDLA", "Ω_HI (subDLA)"),
                  ("omega_DLA", "Ω_HI (DLA)")]
    for (q, title), ax in zip(quantities, axes.flat):
        sub = [r for r in rows if r["quantity"] == q]
        sub.sort(key=lambda r: r["z"])
        z = np.array([r["z"] for r in sub])
        obs_pct = np.array([r["frac_observed"] for r in sub]) * 100
        poi_pct = np.array([r["frac_poisson"] for r in sub]) * 100
        ax.plot(z, obs_pct, "o-", color="C3",
                label="σ_observed across sims")
        ax.plot(z, poi_pct, "s--", color="C0",
                label="σ_Poisson expected")
        # Shade flat-MF-safe window
        ax.axvspan(Z_FLAT_SAFE_MIN, Z_FLAT_SAFE_MAX, color="gray",
                   alpha=0.12, label="flat-MF-safe window")
        ax.set_xlabel("z"); ax.set_ylabel("σ(R) / R̄  [%]")
        ax.set_title(title, fontsize=10)
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, loc="best")
    fig.suptitle(
        "H0 test: is the z-extreme spread explained by Poisson counting noise?\n"
        "(red = observed across-sim spread, blue = Poisson-only prediction)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    outp = OUT / "mf_z2_hypothesis_test.png"
    fig.savefig(outp, dpi=120); plt.close(fig)
    print(f"  wrote {outp}")

    # ---- Linear-in-A_p coefficients for Ω_HI(DLA), the only flagged quantity ----
    print()
    print("Linear-in-A_p MF coefficients for Ω_HI(DLA) (inside flat-safe window):")
    print(f"  {'z':>5s} {'n':>2s} {'intercept':>10s} {'slope (1/A_p)':>15s} "
          f"{'lin σ/R̄':>10s} {'flat σ/R̄':>10s}")
    coef_rows = []
    for zkey in sorted(by_qz["omega_DLA"].keys()):
        if not (Z_FLAT_SAFE_MIN <= zkey <= Z_FLAT_SAFE_MAX):
            continue
        samples = by_qz["omega_DLA"][zkey]
        if len(samples) < 2:
            continue
        Ap = np.array([s["Ap"] for s in samples])
        R = np.array([s["R"] for s in samples])
        A = np.vstack([np.ones_like(Ap), Ap - Ap.mean()]).T
        coef, *_ = np.linalg.lstsq(A, R, rcond=None)
        pred = A @ coef
        lin_rms = float(np.sqrt(np.mean((R - pred) ** 2)))
        flat_rms = float(R.std(ddof=0))
        R_bar = float(R.mean())
        coef_rows.append({
            "z": zkey,
            "n_sims": len(samples),
            "Ap_ref": float(Ap.mean()),
            "intercept": float(coef[0]),
            "slope_per_Ap": float(coef[1]),
            "lin_resid_rms": lin_rms,
            "flat_rms": flat_rms,
        })
        print(f"  {zkey:>5.2f} {len(samples):>2d} "
              f"{coef[0]:>10.4f} {coef[1]:>15.2e} "
              f"{lin_rms/R_bar*100:>9.2f}% {flat_rms/R_bar*100:>9.2f}%")
    csv_p = DATA / "mf_linear_coefficients_omega_dla.csv"
    with open(csv_p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(coef_rows[0].keys()))
        w.writeheader()
        for r in coef_rows: w.writerow(r)
    print(f"\n  wrote {csv_p}")


if __name__ == "__main__":
    main()
