"""
First-pass analytical multi-fidelity (MF) correction for HCD quantities.

For each quantity Q(θ) ∈ {Ω_HI(class), dN/dX(class), T_class(k,z)} we
model the HR/LF ratio R(θ) at fixed z (and k) as

    R_MF(θ) = a + b · (A_p - <A_p>)                               (linear-in-A_p)
    R_MF(θ) = a + b · (A_p - <A_p>) + c · (n_s - <n_s>)           (+ ns)

Since only 3 HR sims are available, the 2-parameter model saturates
the data at each (k, z); a 3-parameter fit has zero residual DOF.
We nonetheless fit the 1-parameter-in-A_p model as a first-pass
analytical correction, and quote:

  * intercept a (the "flat" mean correction — what you'd use if A_p
    were ignored)
  * slope b (the analytical MF strength)
  * residual σ after subtracting the linear fit (proxy for higher-
    order dependence or cosmic variance)
  * fractional RMS error of (a) flat vs (b) linear model on the 3 points
    — ratio tells you how much a linear-in-A_p correction buys you

This is the analytical precursor to a GP / NN MF emulator.  If the
"flat" RMS is already < 1 %, MF gives < 1 % improvement and isn't
needed.  If "flat" RMS is several % and "linear" residual is < 1 %,
the linear model is a cheap, effective MF correction.

Outputs:
  figures/analysis/mf_fit_scalars.csv
  figures/analysis/mf_fit_templates.csv
  figures/analysis/mf_fit_templates_per_k.png
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

CLASSES = ["LLS", "subDLA", "DLA"]
K_ANG_MIN = 0.0009
K_ANG_MAX = 0.20


# -----------------------------------------------------------------------------
# Helpers shared with hf_lf_template_convergence (avoid import to keep this
# script self-contained)
# -----------------------------------------------------------------------------

def _load_summary(path):
    out = {}
    with h5py.File(path, "r") as f:
        sims = [s.decode() for s in f["sim"][:]]
        zs = f["z"][:]
        for i in range(len(sims)):
            r = {
                "z": float(zs[i]),
                "hubble": float(f["hubble"][i]),
            }
            for cls in ["LLS", "subDLA", "DLA", "total"]:
                r[f"dndx_{cls}"] = float(f[f"dndx/{cls}"][i])
                r[f"omega_{cls}"] = float(f[f"Omega_HI/{cls}"][i])
                r[f"counts_{cls}"] = int(f[f"counts/{cls}"][i])
            for pk in ["ns", "Ap", "herei", "heref", "alphaq",
                       "hub", "omegamh2", "hireionz", "bhfeedback"]:
                r[pk] = float(f[f"params/{pk}"][i])
            out[(sims[i], round(float(zs[i]), 3))] = r
    return out


def _pair_records(hr, lf, z_tol=0.05):
    lf_by_sim = defaultdict(list)
    for (sim, _), r in lf.items():
        lf_by_sim[sim].append(r)
    pairs = []
    for (sim, _), h in hr.items():
        if sim not in lf_by_sim:
            continue
        best = min(lf_by_sim[sim], key=lambda l: abs(l["z"] - h["z"]))
        if abs(best["z"] - h["z"]) <= z_tol:
            pairs.append((sim, h["z"], best, h))
    return pairs


# -----------------------------------------------------------------------------
# Analytical fit: flat vs linear-in-A_p
# -----------------------------------------------------------------------------

def fit_flat_and_linear(Ap, R):
    """
    Return dict with:
      flat_mean, flat_rms, linear_intercept, linear_slope_Ap, linear_resid_rms
    """
    Ap = np.asarray(Ap, dtype=float); R = np.asarray(R, dtype=float)
    n = len(R)
    flat_mean = float(np.mean(R))
    flat_rms  = float(np.sqrt(np.mean((R - flat_mean) ** 2)))
    if n >= 2:
        A = np.vstack([np.ones_like(Ap), Ap - Ap.mean()]).T
        coeff, *_ = np.linalg.lstsq(A, R, rcond=None)
        pred = A @ coeff
        lin_rms = float(np.sqrt(np.mean((R - pred) ** 2)))
        a, b = float(coeff[0]), float(coeff[1])
    else:
        a, b, lin_rms = flat_mean, 0.0, flat_rms
    # Fractional improvement
    if flat_rms > 0:
        improvement = 1.0 - lin_rms / flat_rms
    else:
        improvement = 0.0
    return {
        "n": n,
        "flat_mean": flat_mean,
        "flat_rms": flat_rms,
        "linear_intercept": a,
        "linear_slope_Ap": b,
        "linear_resid_rms": lin_rms,
        "fractional_rms_improvement": improvement,
    }


def analyze_scalars():
    hr = _load_summary(HR)
    lf = _load_summary(LF)
    pairs = _pair_records(hr, lf)

    # per-sim z-averaged R for each quantity
    by_sim_q = defaultdict(lambda: defaultdict(list))
    sim_params = {}
    for sim, z_hr, l, h in pairs:
        sim_params[sim] = {"Ap": h["Ap"], "ns": h["ns"]}
        for cls in CLASSES + ["total"]:
            if l[f"dndx_{cls}"] > 0:
                by_sim_q[f"dndx_{cls}"][sim].append(
                    h[f"dndx_{cls}"] / l[f"dndx_{cls}"])
            if l[f"omega_{cls}"] > 0 and np.isfinite(h[f"omega_{cls}"]):
                by_sim_q[f"omega_{cls}"][sim].append(
                    h[f"omega_{cls}"] / l[f"omega_{cls}"])

    rows = []
    for q, per_sim in sorted(by_sim_q.items()):
        sims = sorted(per_sim.keys())
        R = np.array([np.mean(per_sim[s]) for s in sims])
        Ap = np.array([sim_params[s]["Ap"] for s in sims])
        if len(R) < 2:
            continue
        fit = fit_flat_and_linear(Ap, R)
        rows.append({"quantity": q, **fit})
    return rows


def analyze_templates():
    """
    For per-class template T_class = P_class_only/P_clean, build R(sim, z, k)
    and fit linear-in-A_p at each (z, k).  Summarize by class, aggregating
    across z and k in the emulator range.
    """
    SCRATCH = Path("/scratch/cavestru_root/cavestru0/mfho/hcd_outputs")
    LFR = SCRATCH
    HRR = SCRATCH / "hires"
    lf_sims = {p.name for p in LFR.iterdir() if p.is_dir() and p.name.startswith("ns")}
    hr_sims = {p.name for p in HRR.iterdir() if p.is_dir() and p.name.startswith("ns")}
    common = sorted(lf_sims & hr_sims)

    import json
    def enumerate_snaps(sim_dir: Path):
        out = {}
        for p in sorted(sim_dir.iterdir()):
            if not p.is_dir() or not p.name.startswith("snap_"):
                continue
            pc = p / "p1d_per_class.h5"
            meta = p / "meta.json"
            if pc.exists() and meta.exists() and (p / "done").exists():
                try:
                    z = float(json.load(open(meta))["z"])
                except Exception:
                    continue
                out[p.name] = (p, z)
        return out

    # Parse Ap from sim name
    import re
    def ap_of(sim_name):
        m = re.search(r"Ap([0-9.e+-]+)", sim_name)
        return float(m.group(1)) if m else np.nan

    # Gather R[cls][z_key][sim] = (k, R_k)
    R_data = defaultdict(lambda: defaultdict(dict))
    for sim in common:
        lf_snaps = enumerate_snaps(LFR / sim)
        hr_snaps = enumerate_snaps(HRR / sim)
        for hname, (hr_dir, z_hr) in hr_snaps.items():
            best = None
            for lname, (lf_dir, z_lf) in lf_snaps.items():
                if best is None or abs(z_lf - z_hr) < abs(best[1] - z_hr):
                    best = (lf_dir, z_lf)
            if best is None or abs(best[1] - z_hr) > 0.05:
                continue
            lf_dir, _ = best
            try:
                with h5py.File(hr_dir / "p1d_per_class.h5", "r") as fh, \
                     h5py.File(lf_dir / "p1d_per_class.h5", "r") as fl:
                    nk = min(len(fh["k"]), len(fl["k"]))
                    k = fl["k"][:nk]
                    k_ang = 2 * np.pi * k
                    sel = (k_ang >= K_ANG_MIN) & (k_ang <= K_ANG_MAX) & (k > 0)
                    for cls in CLASSES:
                        Pc_lf = fl["P_clean"][:nk][sel]
                        Pc_hr = fh["P_clean"][:nk][sel]
                        Pcl_lf = fl[f"P_{cls}_only"][:nk][sel]
                        Pcl_hr = fh[f"P_{cls}_only"][:nk][sel]
                        mask = (Pc_lf > 0) & (Pc_hr > 0) & (Pcl_lf > 0) & (Pcl_hr > 0)
                        if not mask.any():
                            continue
                        T_lf = Pcl_lf[mask] / Pc_lf[mask]
                        T_hr = Pcl_hr[mask] / Pc_hr[mask]
                        R = T_hr / T_lf
                        R_data[cls][round(z_hr, 3)][sim] = (k[sel][mask], R)
            except Exception:
                continue

    # At each (cls, z, k), fit flat vs linear across sims
    per_class = []
    curves = []  # for plot
    for cls in CLASSES:
        flat_rms_accum = []
        lin_rms_accum = []
        improvement_accum = []
        slope_accum = []
        # Pick representative z bins for plotting
        for zv in sorted(R_data[cls].keys()):
            sims_here = sorted(R_data[cls][zv].keys())
            if len(sims_here) < 3:
                continue
            # Common k among the three — just intersect by index
            ks = [R_data[cls][zv][s][0] for s in sims_here]
            Rs = [R_data[cls][zv][s][1] for s in sims_here]
            nk = min(len(k) for k in ks)
            ks = [k[:nk] for k in ks]
            Rs = [r[:nk] for r in Rs]
            k_ref = ks[0]
            R_mat = np.array(Rs)   # shape (n_sim, nk)
            Ap = np.array([ap_of(s) for s in sims_here])
            for j in range(nk):
                if not np.all(np.isfinite(R_mat[:, j])):
                    continue
                fit = fit_flat_and_linear(Ap, R_mat[:, j])
                flat_rms_accum.append(fit["flat_rms"])
                lin_rms_accum.append(fit["linear_resid_rms"])
                improvement_accum.append(fit["fractional_rms_improvement"])
                slope_accum.append(fit["linear_slope_Ap"])
            # Save a representative curve for plot
            if abs(zv - 3.0) < 0.1:
                curves.append({
                    "class": cls,
                    "z": zv,
                    "k": k_ref,
                    "R_mat": R_mat,
                    "Ap": Ap,
                    "sims": sims_here,
                })
        per_class.append({
            "class": cls,
            "median_flat_rms": float(np.median(flat_rms_accum)) if flat_rms_accum else np.nan,
            "median_linear_resid_rms": float(np.median(lin_rms_accum)) if lin_rms_accum else np.nan,
            "median_fractional_improvement": (
                float(np.median(improvement_accum)) if improvement_accum else np.nan),
            "median_abs_slope_Ap": float(np.median(np.abs(slope_accum))) if slope_accum else np.nan,
        })
    return per_class, curves


def plot_template_fits_at_z3(curves, outpath):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for col, cls in enumerate(CLASSES):
        entries = [c for c in curves if c["class"] == cls]
        if not entries:
            continue
        c = entries[0]
        # top row: R vs k per sim + flat and linear predictions at A_p = <A_p>
        ax = axes[0, col]
        sims_colors = plt.cm.tab10(np.linspace(0, 0.9, len(c["sims"])))
        for i, (sim, col_) in enumerate(zip(c["sims"], sims_colors)):
            ax.plot(2 * np.pi * c["k"], c["R_mat"][i, :], lw=1.0, alpha=0.85,
                    color=col_, label=f"A_p={c['Ap'][i]:.1e}")
        ax.plot(2 * np.pi * c["k"], c["R_mat"].mean(axis=0),
                "k--", lw=1.5, label="flat (mean across sims)")
        ax.axhline(1.0, color="k", lw=0.5, ls=":")
        ax.set_xscale("log")
        ax.set_xlim(K_ANG_MIN, K_ANG_MAX)
        ax.set_title(f"{cls}: T_HR/T_LF at z≈3, 3 A_p values")
        ax.set_xlabel("k_ang"); ax.grid(alpha=0.3, which="both")
        if col == 0: ax.set_ylabel("R(k)")
        ax.legend(fontsize=7, loc="upper right")

        # bottom row: linear fit residual vs k compared to flat RMS
        ax = axes[1, col]
        flat_res = np.std(c["R_mat"], axis=0, ddof=0)
        lin_res = np.zeros_like(flat_res)
        for j in range(c["R_mat"].shape[1]):
            Rs = c["R_mat"][:, j]
            Ap = c["Ap"]
            A = np.vstack([np.ones_like(Ap), Ap - Ap.mean()]).T
            coeff, *_ = np.linalg.lstsq(A, Rs, rcond=None)
            lin_res[j] = np.sqrt(np.mean((Rs - A @ coeff) ** 2))
        ax.plot(2 * np.pi * c["k"], flat_res,
                color="C0", lw=1.2, label="flat (RMS across sims)")
        ax.plot(2 * np.pi * c["k"], lin_res,
                color="C3", lw=1.2, label="linear-in-A_p residual")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(K_ANG_MIN, K_ANG_MAX)
        ax.set_title(f"{cls}: flat RMS vs linear residual")
        ax.set_xlabel("k_ang"); ax.grid(alpha=0.3, which="both")
        if col == 0: ax.set_ylabel("RMS of R(k) across 3 sims")
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "Analytical MF (linear in A_p): does it beat a flat correction?\n"
        "top row: R(k); bottom row: RMS of flat vs linear-fit residual"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outpath, dpi=120); plt.close(fig)


def main():
    print("=" * 72)
    print("Analytical MF fit: flat vs linear-in-A_p")
    print("=" * 72)

    print("\nScalar quantities (z-averaged, per sim):")
    scalar_rows = analyze_scalars()
    print(f"  {'quantity':<14s} {'flat <R>':>10s} {'flat rms':>10s} "
          f"{'lin resid':>10s} {'improve':>9s} {'slope':>11s}")
    for r in scalar_rows:
        print(f"  {r['quantity']:<14s} "
              f"{r['flat_mean']:>10.3f} {r['flat_rms']:>10.4f} "
              f"{r['linear_resid_rms']:>10.4f} "
              f"{r['fractional_rms_improvement']*100:>8.1f}% "
              f"{r['linear_slope_Ap']:>11.2e}")

    csv_path = DATA / "mf_fit_scalars.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=list(scalar_rows[0].keys()))
        w.writeheader()
        for r in scalar_rows: w.writerow(r)
    print(f"  wrote {csv_path}")

    print("\nPer-class templates (z-averaged, per (z, k)):")
    tmpl_rows, curves = analyze_templates()
    print(f"  {'class':<8s} {'flat RMS':>10s} {'lin resid':>10s} "
          f"{'improve':>9s} {'|slope|':>11s}")
    for r in tmpl_rows:
        print(f"  {r['class']:<8s} "
              f"{r['median_flat_rms']:>10.4f} "
              f"{r['median_linear_resid_rms']:>10.4f} "
              f"{r['median_fractional_improvement']*100:>8.1f}% "
              f"{r['median_abs_slope_Ap']:>11.2e}")

    plot_template_fits_at_z3(curves, OUT / "mf_fit_templates_per_k.png")
    print(f"  wrote {OUT/'mf_fit_templates_per_k.png'}")

    csv_path = DATA / "mf_fit_templates.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(tmpl_rows[0].keys()))
        w.writeheader()
        for r in tmpl_rows: w.writerow(r)
    print(f"  wrote {csv_path}")


if __name__ == "__main__":
    main()
