"""
HF/LF convergence of the HCD scalar quantities (dN/dX and Ω_HI per class).

For each of the 3 common (HR, LF) sim pairs, at each z matched within
|Δz| ≤ 0.05, compute the ratio R(sim, z) = Q_HR / Q_LF where
Q ∈ {Ω_HI(LLS), Ω_HI(subDLA), Ω_HI(DLA), dN/dX(LLS), dN/dX(subDLA),
dN/dX(DLA)}.

Outputs
-------
  figures/analysis/hf_lf_ratio_scalars.png   — 2×3 grid of R(z) per class
  figures/analysis/mf_necessity_scalars.csv  — per-quantity MF verdict

MF necessity test
-----------------
We have only 3 matched sims, so a full θ-dependent fit is underpowered.
What we CAN do is compare two hypotheses per quantity:

  H_flat:  R(z, θ) = μ(z)          (same ratio for every sim;
                                     fixed scalar correction works)
  H_var:   R(z, θ) has sim-dependent  component beyond Poisson noise

Decision rule: if the *within-z, across-sim* standard deviation σ_sim(z)
is larger than the Poisson shot-noise expected from the per-sim counts,
reject H_flat and recommend MF treatment.  Otherwise a single mean
correction suffices.

Usage:
    python3 scripts/hf_lf_scalar_convergence.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

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
CLASSES = ["LLS", "subDLA", "DLA"]
CLASS_COLORS = {"LLS": "C2", "subDLA": "C1", "DLA": "C3"}
PARAM_KEYS = ["ns", "Ap", "herei", "heref", "alphaq",
              "hub", "omegamh2", "hireionz", "bhfeedback"]


def load_summary(path):
    """Return dict of arrays keyed by (sim, z_rounded) → record."""
    out = {}
    with h5py.File(path, "r") as f:
        sims = [s.decode() for s in f["sim"][:]]
        zs = f["z"][:]
        n = len(sims)
        for i in range(n):
            key = (sims[i], round(float(zs[i]), 3))
            r = {
                "z": float(zs[i]),
                "hubble": float(f["hubble"][i]),
                "dX_total": float(f["dX_total"][i]),
                "n_skewers": int(f["n_skewers"][i]),
            }
            for cls in ["LLS", "subDLA", "DLA", "total"]:
                r[f"counts_{cls}"] = int(f[f"counts/{cls}"][i])
                r[f"dndx_{cls}"] = float(f[f"dndx/{cls}"][i])
                r[f"omega_{cls}"] = float(f[f"Omega_HI/{cls}"][i])
            for pk in PARAM_KEYS:
                r[pk] = float(f[f"params/{pk}"][i])
            out[key] = r
    return out


def build_pairs(hr_summary, lf_summary, z_tol=0.05):
    """
    For each HR (sim, z) record, find the LF record on the same sim
    with closest z.  Return list of tuples: (sim, z_hr, z_lf, lf_rec, hr_rec).
    """
    # Group LF records by sim
    lf_by_sim = {}
    for (sim, zr), rec in lf_summary.items():
        lf_by_sim.setdefault(sim, []).append(rec)

    out = []
    for (sim, zr_hr), hr_rec in hr_summary.items():
        if sim not in lf_by_sim:
            continue
        z_hr = hr_rec["z"]
        # pick closest LF
        best = min(lf_by_sim[sim], key=lambda r: abs(r["z"] - z_hr))
        if abs(best["z"] - z_hr) > z_tol:
            continue
        out.append((sim, z_hr, best["z"], best, hr_rec))
    # sort by z_hr
    out.sort(key=lambda t: (t[0], t[1]))
    return out


def compute_ratios(pairs):
    """
    Return a nested dict: ratios[quantity][sim] = list of (z, R, R_err).
    R_err is the Poisson-propagated error for count-based quantities.
    """
    from collections import defaultdict
    ratios = defaultdict(lambda: defaultdict(list))
    for sim, z_hr, z_lf, lf, hr in pairs:
        for cls in CLASSES:
            # dN/dX
            n_hr, n_lf = hr[f"counts_{cls}"], lf[f"counts_{cls}"]
            dndx_hr = hr[f"dndx_{cls}"]
            dndx_lf = lf[f"dndx_{cls}"]
            if dndx_lf > 0 and n_hr >= 1 and n_lf >= 1:
                R = dndx_hr / dndx_lf
                # Poisson error on counts → error on R
                rel = np.sqrt(1.0 / n_hr + 1.0 / n_lf)
                ratios[f"dndx_{cls}"][sim].append((z_hr, R, R * rel))
            # Ω_HI — approximate error using same Poisson rel factor
            o_hr, o_lf = hr[f"omega_{cls}"], lf[f"omega_{cls}"]
            if o_lf > 0 and n_hr >= 1 and n_lf >= 1 and np.isfinite(o_hr) and np.isfinite(o_lf):
                R = o_hr / o_lf
                rel = np.sqrt(1.0 / n_hr + 1.0 / n_lf)
                ratios[f"omega_{cls}"][sim].append((z_hr, R, R * rel))
    return ratios


def plot_ratios(ratios, outpath):
    """
    2 rows × 3 cols.  Top row: dN/dX ratios (LLS, subDLA, DLA).
    Bottom row: Ω_HI ratios.  One line per sim, shaded Poisson band.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    sim_list = sorted({sim for q in ratios.values() for sim in q.keys()})
    sim_colors = {s: f"C{i}" for i, s in enumerate(sim_list)}
    row_labels = [("dndx", "dN/dX"), ("omega", "Ω_HI")]
    for r_i, (qprefix, qlabel) in enumerate(row_labels):
        for c_i, cls in enumerate(CLASSES):
            ax = axes[r_i, c_i]
            key = f"{qprefix}_{cls}"
            if key not in ratios:
                continue
            for sim in sim_list:
                if sim not in ratios[key]:
                    continue
                data = sorted(ratios[key][sim], key=lambda t: t[0])
                z = np.array([t[0] for t in data])
                R = np.array([t[1] for t in data])
                E = np.array([t[2] for t in data])
                ax.errorbar(
                    z, R, yerr=E, fmt="o-", ms=4, lw=1.2, capsize=2,
                    color=sim_colors[sim], alpha=0.85,
                    label=(sim[:14]+"…") if r_i == 0 and c_i == 0 else None,
                )
            ax.axhline(1.0, color="k", lw=0.6, ls="--")
            ax.set_title(f"{qlabel}({cls})", fontsize=10)
            ax.grid(alpha=0.3)
            if r_i == 1:
                ax.set_xlabel("z")
            if c_i == 0:
                ax.set_ylabel(f"R = {qlabel}_HR / {qlabel}_LF")
    fig.suptitle(
        "HR/LF convergence of HCD scalars  (3 matched sims, z-matched |Δz|≤0.05)"
    )
    axes[0, 0].legend(fontsize=7, loc="lower left", ncol=1)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outpath, dpi=120)
    plt.close(fig)


def mf_necessity_verdict(ratios):
    """
    For each quantity, per z-bin: is σ_across-sims > σ_Poisson?

    We compute σ_across-sims using the observed scatter across the 3 sims at
    each z (after matching).  σ_Poisson is the quadrature combination of
    individual sims' Poisson errors (treating them as independent noise).

    Rule: if z-averaged σ_across / σ_Poisson > 2, reject H_flat and flag
    "MF recommended".  Between 1 and 2 → borderline.  ≤ 1 → flat suffices.
    """
    from collections import defaultdict
    verdicts = []
    for qkey, by_sim in ratios.items():
        # Build table of R[sim][z]
        z_set = sorted({z for vals in by_sim.values() for z, _, _ in vals})
        n_sim_per_z = []
        sigma_acr = []
        sigma_poi = []
        for z_target in z_set:
            Rs, Es = [], []
            for sim, vals in by_sim.items():
                for (z, R, E) in vals:
                    if abs(z - z_target) < 0.02:
                        Rs.append(R); Es.append(E)
                        break
            if len(Rs) < 2:
                continue
            Rs = np.array(Rs); Es = np.array(Es)
            n_sim_per_z.append(len(Rs))
            sigma_acr.append(float(np.std(Rs, ddof=1)))
            # Effective Poisson σ on the *mean* of R across sims:
            # If all independent, σ_mean = sqrt(mean(E^2)/n)
            # but for comparing to sample σ across sims, use typical Poisson
            # scale per sim: rms(E) / sqrt(1) — i.e. expected per-sim scatter.
            sigma_poi.append(float(np.sqrt(np.mean(Es ** 2))))
        if not sigma_acr:
            continue
        # z-averaged ratio
        ratio = np.array(sigma_acr) / np.array(sigma_poi)
        typ = float(np.median(ratio))
        if typ > 2.0:
            verdict = "MF recommended"
        elif typ > 1.0:
            verdict = "borderline"
        else:
            verdict = "flat scalar suffices"
        verdicts.append({
            "quantity": qkey,
            "n_z_bins": len(sigma_acr),
            "median_sigma_across_to_poisson": typ,
            "median_mean_R": float(np.median([
                np.mean([t[1] for t in by_sim[sim]]) for sim in by_sim
            ])),
            "verdict": verdict,
        })
    return verdicts


def fit_linear_R_vs_params(ratios, lf_summary):
    """
    For each quantity, fit R = a + Σ_i b_i · (θ_i - θ̄_i) at the sim level,
    z-averaged.  Reports the largest |b_i / σ_R| for each quantity.

    With only 3 sims this is under-determined (9 params, 3 data points),
    so we instead fit a 1-parameter model for the top-ranked param (A_p)
    and report the t-stat.
    """
    out = []
    # Extract LF per-sim param values
    sim_params = {}
    for (sim, z), rec in lf_summary.items():
        if sim not in sim_params:
            sim_params[sim] = {pk: rec[pk] for pk in PARAM_KEYS}
    for qkey, by_sim in ratios.items():
        if len(by_sim) < 3:
            continue
        # z-average R per sim
        sim_means = {}
        for sim, vals in by_sim.items():
            sim_means[sim] = np.mean([t[1] for t in vals])
        sims = list(sim_means.keys())
        Ap = np.array([sim_params[s]["Ap"] for s in sims])
        R = np.array([sim_means[s] for s in sims])
        ns = np.array([sim_params[s]["ns"] for s in sims])
        # Linear fit R = a + b * Ap  (1 DOF, 3 points — overdetermined-but-limited)
        A = np.vstack([np.ones_like(Ap), Ap - Ap.mean()]).T
        coeffs, res, rk, sv = np.linalg.lstsq(A, R, rcond=None)
        residuals = R - A @ coeffs
        # With only 3 points and 2 unknowns, residual variance has 1 DOF.
        dof = len(R) - 2
        if dof > 0:
            rss = float(np.sum(residuals ** 2))
            sigma = np.sqrt(rss / dof)
        else:
            sigma = 0.0
        out.append({
            "quantity": qkey,
            "sims": sims,
            "R_per_sim": R.tolist(),
            "Ap": Ap.tolist(),
            "slope_vs_Ap": float(coeffs[1]),
            "intercept": float(coeffs[0]),
            "residual_sigma": sigma,
            "R_range": (float(R.min()), float(R.max())),
            "R_range_frac": float(np.ptp(R) / R.mean()) if R.mean() > 0 else float("nan"),
        })
    return out


def main():
    print("Loading summaries…")
    lf = load_summary(LF)
    hr = load_summary(HR)
    print(f"  LF: {len(lf)} (sim,z) records")
    print(f"  HR: {len(hr)} (sim,z) records")

    pairs = build_pairs(hr, lf, z_tol=0.05)
    n_sims = len({p[0] for p in pairs})
    print(f"  Paired: {len(pairs)} pairs across {n_sims} common sims")

    ratios = compute_ratios(pairs)

    outp = OUT / "hf_lf_ratio_scalars.png"
    plot_ratios(ratios, outp)
    print(f"  wrote {outp}")

    # MF verdict
    verdicts = mf_necessity_verdict(ratios)
    print("\nMF necessity per quantity:")
    print(f"  {'quantity':<16s} {'nz':>3s} {'σ_sim/σ_poi':>12s} {'<R>':>6s}  verdict")
    for v in verdicts:
        print(f"  {v['quantity']:<16s} {v['n_z_bins']:>3d} "
              f"{v['median_sigma_across_to_poisson']:>12.2f} "
              f"{v['median_mean_R']:>6.3f}  {v['verdict']}")

    # Linear fits to (Ap) — are these R's A_p-dependent?
    lin = fit_linear_R_vs_params(ratios, lf)
    print("\nLinear fit R = a + b·(A_p - <A_p>) at sim level (3 sims, z-avg):")
    print(f"  {'quantity':<16s} {'intercept':>10s} {'slope/Ap':>12s} "
          f"{'resid σ':>10s} {'R-range/<R>':>12s}")
    for f in lin:
        print(f"  {f['quantity']:<16s} {f['intercept']:>10.3f} "
              f"{f['slope_vs_Ap']:>12.2e} {f['residual_sigma']:>10.4f} "
              f"{f['R_range_frac']:>12.3f}")

    # Dump everything to CSV
    csv_path = DATA / "mf_necessity_scalars.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "n_z_bins", "median_sigma_across_to_poisson",
                    "median_mean_R", "verdict", "slope_vs_Ap",
                    "R_range_frac"])
        lin_by_q = {f["quantity"]: f for f in lin}
        for v in verdicts:
            lrec = lin_by_q.get(v["quantity"], {})
            w.writerow([v["quantity"], v["n_z_bins"],
                        v["median_sigma_across_to_poisson"],
                        v["median_mean_R"], v["verdict"],
                        lrec.get("slope_vs_Ap", ""),
                        lrec.get("R_range_frac", "")])
    print(f"  wrote {csv_path}")


if __name__ == "__main__":
    main()
