"""PRIYA sim per-class dN/dX vs the OBSERVED literature incidence (Head-A overlay).

The HCD incidence prior α_c centers on PRIYA's SIM weight w_c — so we must check whether
PRIYA's sim dN/dX actually matches the observed literature dN/dX. If PRIYA is offset, then
"α=1 (sim)" ≠ "the data's true incidence", and the prior center must be shifted by the
offset (the data would otherwise pull α to absorb the sim-vs-data discrepancy). This plots
PRIYA's sim dN/dX(z) per class (mean±std over sims) against the literature, per class, +
the PRIYA/literature ratio.

Literature (2026-06-04 Lyα agent): DLA = Prochaska&Wolfe2009 (Table 1); subDLA = Zafar+2013
(Table 3); LLS = O'Meara+2013 / Fumagalli+2013 / Prochaska+2010 (τ_LL≥2). CAVEAT: PRIYA's
LLS class floor is log N_HI≥17.2 (τ≥1) while the literature LLS is τ≥2 (N≥17.5), so PRIYA's
LLS is EXPECTED ~25–40% higher — a known definitional offset to calibrate, not a bug.

Env:
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu \
      /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/plot_dndx_vs_literature.py
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hcd_analysis.emulator.data import load_cache

ROOT = Path(__file__).resolve().parents[1]
CACHE = str(ROOT / "hcd_analysis/_emulator_data/observables_tau0_lf.h5")
OUT = ROOT / "figures/analysis/06_performance_walkthrough/A6_dndx_vs_literature.png"

# Observed dN/dX (z, value, ±err) per class
LIT = {
    "LLS": ([2.4, 2.8, 3.35, 3.47, 3.58, 3.74, 3.97, 4.23],
            [0.29, 0.33, 0.35, 0.57, 0.41, 0.52, 0.72, 0.78],
            [0.05, 0.08, 0.14, 0.12, 0.07, 0.08, 0.15, 0.19],
            "O'Meara13 / Fumagalli13 / Prochaska10 (τ≥2)"),
    "subDLA": ([2.27, 2.73, 3.25, 3.77, 4.20],
               [0.07, 0.06, 0.08, 0.10, 0.10],
               [0.01, 0.01, 0.02, 0.02, 0.03],
               "Zafar+2013 (Table 3)"),
    "DLA": ([2.31, 2.57, 2.86, 3.22, 3.70, 4.39],
            [0.048, 0.055, 0.067, 0.084, 0.075, 0.106],
            [0.006, 0.005, 0.006, 0.006, 0.009, 0.018],
            "Prochaska & Wolfe 2009 (Table 1)"),
}
CLS = ["LLS", "subDLA", "DLA"]


def main():
    d = load_cache(CACHE)
    gid = np.asarray(d["snap_group_idx"]); zrow = np.asarray(d["z_grid"])
    dndx = np.asarray(d["snap_dNdX"])                       # (Ngroup, 3) LLS, subDLA, DLA
    ng = dndx.shape[0]
    zg = np.array([zrow[gid == g][0] for g in range(ng)])   # per-group z

    # bin PRIYA by the z grid (Δz=0.2): mean ± std over sims
    zbins = np.unique(np.round(zg, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 3, figsize=(16, 8), height_ratios=[2.2, 1])
    for j, c in enumerate(CLS):
        zc, mc, sc = [], [], []
        for zb in zbins:
            sel = np.isclose(zg, zb, atol=0.05)
            if sel.sum() >= 2:
                zc.append(zb); mc.append(np.nanmean(dndx[sel, j])); sc.append(np.nanstd(dndx[sel, j]))
        zc, mc, sc = map(np.array, (zc, mc, sc))
        a = ax[0, j]
        a.fill_between(zc, mc - sc, mc + sc, color="tab:blue", alpha=0.25)
        a.plot(zc, mc, "-", color="tab:blue", lw=2, label="PRIYA sim (mean±std)")
        zl, vl, el, src = LIT[c]
        a.errorbar(zl, vl, yerr=el, fmt="o", color="tab:red", ms=5, capsize=3, label="observed")
        a.set_title(f"{c}   (lit: {src})", fontsize=9)
        a.set_ylabel("dN/dX"); a.set_yscale("log"); a.legend(fontsize=8); a.grid(alpha=0.3)
        # ratio panel: PRIYA / literature (interp lit onto PRIYA z where they overlap)
        lit_at = np.interp(zc, zl, vl, left=np.nan, right=np.nan)
        ratio = mc / lit_at
        b = ax[1, j]
        b.plot(zc, ratio, "s-", color="tab:purple")
        b.axhline(1.0, color="k", lw=0.8, ls="--")
        b.set_xlabel("z"); b.set_ylabel("PRIYA / obs"); b.grid(alpha=0.3)
        b.set_ylim(0, max(2.0, np.nanmax(ratio) * 1.1) if np.isfinite(np.nanmax(ratio)) else 2.0)
        med = np.nanmedian(ratio)
        b.set_title(f"median ratio {med:.2f}", fontsize=8)
        print(f"{c}: PRIYA/obs median ratio = {med:.2f}  "
              f"(z-range {zc.min():.1f}-{zc.max():.1f})")
    fig.suptitle("PRIYA sim per-class dN/dX vs observed literature "
                 "(the α-prior center is the SIM weight — check it matches the data)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150); plt.close(fig)
    print("wrote", OUT)

    # --- power-law fits for the z-SLOPE incidence prior (inference.HCD_LIT_OVER_SIM* ) ---
    # The lit/sim ratio is a power-law in (1+z): r_c(z) = r_c(z_p)·((1+z)/(1+z_p))^s_c,
    # s_c = γ_lit − γ_sim. Mirrors the τ₀ Kim-curve+slope model.
    z_p = 3.0
    print(f"\nz-slope incidence-prior fits (pivot z={z_p}):")
    for j, c in enumerate(CLS):
        inr = (zg >= 2.2) & (zg <= 4.6) & (dndx[:, j] > 0)
        gs = np.polyfit(np.log(1 + zg[inr]), np.log(dndx[inr, j]), 1)   # sim slope, intercept
        zl, vl, _, _ = LIT[c]
        gl = np.polyfit(np.log(1 + np.array(zl)), np.log(np.array(vl)), 1)
        sim_zp = np.exp(gs[1]) * (1 + z_p) ** gs[0]
        lit_zp = np.exp(gl[1]) * (1 + z_p) ** gl[0]
        print(f"  {c:7} gamma_sim={gs[0]:+.2f} gamma_lit={gl[0]:+.2f}  "
              f"(lit/sim)@z{z_p:.0f}={lit_zp/sim_zp:.2f}  ratio_slope={gl[0]-gs[0]:+.2f}")


if __name__ == "__main__":
    main()
