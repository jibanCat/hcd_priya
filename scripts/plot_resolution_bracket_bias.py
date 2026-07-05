#!/usr/bin/env python3
"""Gate-B RESOLUTION bracket result figure: per-treatment |Delta|+2SE (n_s, A_p) per leg.

Numbers are the paired clean-vs-injected bias from analyze_dnuis_bias.py on the completed
brackets (verified 2026-07-04):
  eBOSS  eboss_bres_0044  (job 52814342, N=6): a/b/c/d
  DESI   desi_bres_002    (jobs 52791523 + 52714399): a/b/d  (no wide-c arm on DESI)
Gate = |mean Delta_bias_z| + 2 SE < 0.30 sigma_post on BOTH n_s and A_p.
The panel winner is eBOSS treatment c (option-b WIDE, N(0,0.05)), the only gate-passer.

Env: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu ... python3 this.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({"font.size": 12, "figure.facecolor": "white", "savefig.facecolor": "white",
                     "axes.grid": True, "grid.alpha": 0.25})

OUT = "/home/mfho/hcd_priya_notes/figures/analysis/08_resolution/resolution_bracket_bias.png"

# treatment -> label; |Delta|+2SE per leg per param (None = arm not run on that leg)
TREAT = ["a", "b", "c", "d"]
LABEL = {"a": "option-a\n(res in cov)", "b": "option-b\ntight N(0,.02)",
         "c": "option-b WIDE\nN(0,.05)", "d": "arm-D\n(coh cov)"}
# |Delta|+2SE   eBOSS,      DESI
NS = {"a": (0.867, 2.227), "b": (0.410, 0.314), "c": (0.221, None), "d": (0.915, 2.214)}
AP = {"a": (4.214, 1.442), "b": (0.750, 0.214), "c": (0.102, None), "d": (0.847, 1.707)}
GATE = 0.30

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
x = np.arange(len(TREAT)); w = 0.38
c_eboss, c_desi, c_win = "#4C78A8", "#F58518", "#54A24B"

for ax, DATA, title in ((axes[0], NS, r"$n_s$  bias"), (axes[1], AP, r"$A_p$  bias")):
    eb = [DATA[t][0] for t in TREAT]
    de = [DATA[t][1] if DATA[t][1] is not None else 0.0 for t in TREAT]
    de_missing = [DATA[t][1] is None for t in TREAT]
    b1 = ax.bar(x - w/2, eb, w, label="eBOSS", color=c_eboss, edgecolor="k", linewidth=0.4)
    b2 = ax.bar(x + w/2, de, w, label="DESI", color=c_desi, edgecolor="k", linewidth=0.4)
    # highlight the gate-passing winner (eBOSS c)
    b1[TREAT.index("c")].set_color(c_win); b1[TREAT.index("c")].set_edgecolor("k")
    ax.axhline(GATE, color="crimson", ls="--", lw=1.6)
    ax.text(3.35, GATE + 0.05, "0.30 gate", color="crimson", ha="right", fontsize=10, va="bottom")
    # cap tall bars for readability, annotate the true value
    ymax = 1.6
    for bars, vals in ((b1, eb), (b2, de)):
        for rect, v in zip(bars, vals):
            if v > ymax:
                rect.set_height(ymax)
                ax.text(rect.get_x()+rect.get_width()/2, ymax+0.02, f"{v:.2f}↑",
                        ha="center", va="bottom", fontsize=8.5, color="dimgray")
    for i, miss in enumerate(de_missing):
        if miss:
            ax.text(x[i]+w/2, 0.02, "n/a", ha="center", va="bottom", fontsize=8.5, color="gray")
    # PASS/FAIL tag under each eBOSS bar
    for i, t in enumerate(TREAT):
        passed = DATA[t][0] < GATE
        ax.text(x[i]-w/2, -0.10, "PASS" if passed else "FAIL", ha="center", va="top",
                fontsize=8.5, color=(c_win if passed else "crimson"), fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([LABEL[t] for t in TREAT], fontsize=9.5)
    ax.set_ylim(-0.25, ymax + 0.15); ax.set_ylabel(r"$|\Delta_{\rm bias}|+2\,$SE  [$\sigma_{\rm post}$]")
    ax.set_title(title)

axes[0].legend(handles=[Patch(fc=c_eboss, ec="k", label="eBOSS"),
                        Patch(fc=c_desi, ec="k", label="DESI"),
                        Patch(fc=c_win, ec="k", label="eBOSS gate-passer (winner)")],
               loc="upper right", fontsize=9)
fig.suptitle("Gate-B resolution bracket: option-b WIDE (c) is the only gate-passer "
             "(eBOSS N=6; DESI a/b/d)", fontsize=12.5, y=1.005)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=135, bbox_inches="tight"); plt.close(fig)
print(f"[resolution-bracket] wrote {OUT}")
