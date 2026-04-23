"""
Two hypothesis tests on claims in docs/analysis.md:

  (H1) "PRIYA DLA dN/dX sits slightly below PW09/N12/Ho21 by factor
       1.5-2" — but 60 sims averaged to a point estimate.  Test:
       does the under-prediction survive a 1000× bootstrap over
       the 60-sim ensemble?

  (H2) "A_p dominates HCD-count variation, ρ ≈ +0.84" — but ns
       and A_p covary in the PRIYA sampling.  Test: partial Spearman
       controlling for ns, plus leave-one-out stability.

Inputs: figures/analysis/hcd_summary_lf.h5 (60 LF sims, all z).
Outputs: figures/analysis/hypothesis_dndx_bootstrap.png
         figures/analysis/hypothesis_partial_corr.png
         stdout report of numerical findings.

Run:
    python3 scripts/hypothesis_dndx_and_ap.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, rankdata

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from common import data_dir
DATA = data_dir()
SUMMARY = DATA / "hcd_summary_lf.h5"
# This script produces two figures — the bootstrap belongs with the MF work
# (04_hcd_mf) and the partial-correlation is a sensitivity diagnostic (02).
OUT_MF = ROOT / "figures" / "analysis" / "04_hcd_mf"
OUT_SENS = ROOT / "figures" / "analysis" / "02_param_sensitivity"
OUT_MF.mkdir(parents=True, exist_ok=True)
OUT_SENS.mkdir(parents=True, exist_ok=True)

# Observational DLA dN/dX values (sbird/dla_data verbatim) — for z≈3 band.
# PW09 at z∈[2.7,3.0] : 0.067 ± 0.006
# PW09 at z∈[3.0,3.5] : 0.084 ± 0.006
# N12 at z=3.05       : 0.29·(1620/5834) ≈ 0.081
# Ho21 z=3.08         : 0.0706 (68% 0.0685-0.0729)
#                z=3.25: 0.0748 (68% 0.0722-0.0777)
OBS_Z3 = {
    "PW09 z∈[2.7,3.0]":  (0.067, 0.006),
    "PW09 z∈[3.0,3.5]":  (0.084, 0.006),
    "N12 z=3.05":        (0.0805, None),
    "Ho21 z=3.08":       (0.0706, 0.0022),   # 68% half-width ≈ (0.0729-0.0685)/2
    "Ho21 z=3.25":       (0.0748, 0.0028),
}


def load_z3_per_sim():
    """Per-sim arrays at z≈3 only, for bootstrap."""
    with h5py.File(SUMMARY, "r") as f:
        z_all = f["z"][:]
        sim_all = [s.decode() for s in f["sim"][:]]
        sel = np.abs(z_all - 3.0) < 0.05
        out = {
            "sim":    np.array(sim_all)[sel],
            "z":      z_all[sel],
            "dX":     f["dX_total"][:][sel],
            "n_LLS":  f["counts/LLS"][:][sel],
            "n_sub":  f["counts/subDLA"][:][sel],
            "n_DLA":  f["counts/DLA"][:][sel],
            "omega_DLA": f["Omega_HI/DLA"][:][sel],
        }
        for pk in ["ns", "Ap", "herei", "heref", "alphaq",
                   "hub", "omegamh2", "hireionz", "bhfeedback"]:
            out[pk] = f[f"params/{pk}"][:][sel]
    return out


# -----------------------------------------------------------------------------
# (H1) Bootstrap dN/dX(DLA) at z≈3 across the 60-sim ensemble
# -----------------------------------------------------------------------------

def h1_bootstrap(d, n_boot=5000, rng=None):
    """
    For each bootstrap draw, resample sims with replacement, compute the
    ensemble-averaged dN/dX(DLA) = Σ n_DLA / Σ dX across the resample.
    Returns array of shape (n_boot,).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    nsim = len(d["sim"])
    n_DLA = d["n_DLA"].astype(float)
    dX = d["dX"].astype(float)

    # Also stratify by sim to keep the per-sim count/path coupling.
    # Bootstrap index-level over sims, then sum both numerator and denominator.
    idx_draws = rng.integers(0, nsim, size=(n_boot, nsim))
    num = n_DLA[idx_draws].sum(axis=1)
    den = dX[idx_draws].sum(axis=1)
    return num / den


def report_h1(d):
    dndx_per_sim = d["n_DLA"] / d["dX"]
    point_est = d["n_DLA"].sum() / d["dX"].sum()    # ensemble-weighted
    mean_per_sim = dndx_per_sim.mean()               # per-sim average

    boots = h1_bootstrap(d, n_boot=5000)
    q = np.percentile(boots, [2.5, 16, 50, 84, 97.5])

    print("\n" + "="*70)
    print("H1 — Bootstrap test on dN/dX(DLA) under-prediction at z≈3")
    print("="*70)
    print(f"  60-sim ensemble-weighted dN/dX(DLA) = {point_est:.4f}")
    print(f"  mean across sims                   = {mean_per_sim:.4f}")
    print(f"  Bootstrap (5 000 resamples, draw-60-sims-with-replacement):")
    print(f"    2.5%  16%  50%  84%  97.5%")
    print(f"    {q[0]:.4f} {q[1]:.4f} {q[2]:.4f} {q[3]:.4f} {q[4]:.4f}")
    print()
    print("  Significance vs observations:")
    for name, (val, err) in OBS_Z3.items():
        # two-sided: what fraction of bootstrap samples land >= obs?
        frac_above = (boots >= val).mean()
        err_str = f"±{err:.4f}" if err else "     "
        print(f"    {name:<22s} obs={val:.3f}{err_str} | "
              f"bootstrap frac≥obs = {frac_above*100:6.2f}%")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(boots, bins=60, color="C0", alpha=0.7, edgecolor="k", lw=0.3,
            label=f"PRIYA bootstrap ({len(boots)} draws)")
    ax.axvline(point_est, color="C0", lw=2, ls="-",
               label=f"60-sim point estimate = {point_est:.3f}")
    colors_obs = plt.cm.tab10(np.linspace(0, 0.9, len(OBS_Z3)))
    for i, (name, (val, err)) in enumerate(OBS_Z3.items()):
        ax.axvline(val, color=colors_obs[i], lw=1.5, ls="--", label=f"{name}: {val:.3f}")
        if err:
            ax.axvspan(val - err, val + err, color=colors_obs[i], alpha=0.12)
    ax.set_xlabel("dN/dX(DLA)  at z ≈ 3")
    ax.set_ylabel("bootstrap count")
    ax.set_title("H1: Is the PRIYA DLA dN/dX significantly below observations?\n"
                 "60-sim bootstrap vs PW09 / N12 / Ho21")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    outp = OUT_MF / "hypothesis_dndx_bootstrap.png"
    fig.savefig(outp, dpi=120); plt.close(fig)
    print(f"  wrote {outp}")
    return q, boots, point_est


# -----------------------------------------------------------------------------
# (H2) Partial Spearman on A_p → HCD counts, controlling for ns
# -----------------------------------------------------------------------------

def partial_spearman(x, y, z):
    """
    Partial Spearman rank correlation of (x, y) controlling for z.
    Computed as Spearman of the residuals after linearly regressing
    rank(x) on rank(z) and rank(y) on rank(z).
    """
    rx = rankdata(x); ry = rankdata(y); rz = rankdata(z)
    # regress rx on rz
    bx = np.polyfit(rz, rx, 1)
    ex = rx - np.polyval(bx, rz)
    by = np.polyfit(rz, ry, 1)
    ey = ry - np.polyval(by, rz)
    rho, p = spearmanr(ex, ey)
    return rho, p


def h2_partial_correlation(d):
    print("\n" + "="*70)
    print("H2 — Partial Spearman: A_p drives HCD counts controlling for ns")
    print("="*70)
    targets = [("LLS", d["n_LLS"]), ("subDLA", d["n_sub"]),
               ("DLA", d["n_DLA"]), ("Ω_HI(DLA)", d["omega_DLA"])]
    Ap, ns = d["Ap"], d["ns"]

    print(f"  {'target':<12s} {'ρ(target,Ap)':>14s} {'ρ(target,Ap|ns)':>18s} "
          f"{'p':>10s} {'ρ_LOO_top':>12s}")
    results = []
    for name, y in targets:
        rho_raw, _  = spearmanr(Ap, y)
        rho_par, p_par = partial_spearman(Ap, y, ns)
        # Leave-one-out stability on the highest-A_p sim:
        top = np.argmax(Ap)
        idx = np.ones(len(Ap), dtype=bool); idx[top] = False
        rho_loo, _  = spearmanr(Ap[idx], y[idx])
        print(f"  {name:<12s} {rho_raw:>14.3f} {rho_par:>18.3f} "
              f"{p_par:>10.2g} {rho_loo:>12.3f}")
        results.append((name, rho_raw, rho_par, p_par, rho_loo))

    # Visualization — bar chart of raw vs partial ρ
    fig, ax = plt.subplots(figsize=(8, 5))
    names = [r[0] for r in results]
    raw   = [r[1] for r in results]
    par   = [r[2] for r in results]
    loo   = [r[4] for r in results]
    x = np.arange(len(names)); w = 0.28
    ax.bar(x - w, raw, w, color="C0", label="Spearman(A_p, target)")
    ax.bar(x,     par, w, color="C1", label="Partial Spearman | ns")
    ax.bar(x + w, loo, w, color="C3", label="LOO: drop highest-A_p sim")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel(r"Spearman $\rho$")
    ax.set_title(
        "H2: Does A_p dominance survive partial-correlation and leave-one-out?\n"
        "(blue = raw, orange = partial | ns, red = drop highest-A_p sim)"
    )
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(-0.1, 1.0)
    fig.tight_layout()
    outp = OUT_SENS / "hypothesis_partial_corr.png"
    fig.savefig(outp, dpi=120); plt.close(fig)
    print(f"  wrote {outp}")
    return results


def main():
    d = load_z3_per_sim()
    print(f"Loaded z≈3 record for {len(d['sim'])} LF sims.")
    report_h1(d)
    h2_partial_correlation(d)


if __name__ == "__main__":
    main()
