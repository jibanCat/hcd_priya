"""PRIYA sim per-class dN/dX vs the OBSERVED literature incidence (Head-A overlay).

The HCD incidence prior α_c centers on PRIYA's SIM weight w_c — so we must check whether
PRIYA's sim dN/dX actually matches the observed literature dN/dX. If PRIYA is offset, then
"α=1 (sim)" ≠ "the data's true incidence", and the prior center must be shifted by the
offset (the data would otherwise pull α to absorb the sim-vs-data discrepancy). This plots
PRIYA's sim dN/dX(z) per class (mean±std over sims) against the literature, per class, +
the PRIYA/literature ratio.

Literature (CORRECTED estimands, re-derivation 2026-07-18): DLA = Prochaska&Wolfe2009
(Table 1); subDLA = Zafar+2013 (Table 3) COUNTS n/dX, binned [19.0,20.3) (the old array
here was the mislabeled Peroux DLA column); LLS = the kernel-corrected BINNED [17.2,19.0)
points (POW10/O'Meara13/Fumagalli13 cumulative τ≥2 compilation × the adopted K1a PRIYA-CDDF
kernel — definition-MATCHED to PRIYA's LLS class, so the old "PRIYA expected ~25–40% higher"
definitional caveat DISSOLVES; served from the committed derivation JSON via
lit_dndx.lit_points_for_display). Old wrong-object arrays: tombstoned in
hcd_analysis/emulator/lit_dndx.py, never plotted as truth again.

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
from hcd_analysis.emulator.lit_dndx import lit_points_for_display
from hcd_analysis.emulator.inference import HCD_LIT_DNDX_ESTIMAND, assert_dndx_law_estimand

ROOT = Path(__file__).resolve().parents[1]
CACHE = str(ROOT / "hcd_analysis/_emulator_data/observables_tau0_lf.h5")
OUT = ROOT / "figures/analysis/06_performance_walkthrough/A6_dndx_vs_literature.png"

# Observed dN/dX (z, value, ±err, source) per class — the CORRECTED estimands, single
# source lit_dndx.lit_points_for_display (estimand-asserted below).
LIT = lit_points_for_display()
# estimand assert (independent literals, NOT the dict's own values — a wrong-object
# reinstatement in HCD_LIT_DNDX_ESTIMAND trips here):
assert_dndx_law_estimand("LLS", "binned_17.2_19.0", "plot_dndx_vs_literature")
assert_dndx_law_estimand("subDLA", "binned_19.0_20.3", "plot_dndx_vs_literature")
assert_dndx_law_estimand("DLA", "binned_ge20.3", "plot_dndx_vs_literature")
assert set(HCD_LIT_DNDX_ESTIMAND) == {"LLS", "subDLA", "DLA"}
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
