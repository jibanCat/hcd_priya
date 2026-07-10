#!/usr/bin/env python3
"""Task-2 Phase-2 OOS-instrument-resolution NUTS confirmation: gate figure.

Numbers verified 2026-07-10 by re-running scripts/analyze_dnuis_bias.py on each of the 6
`bstar` +/-1sigma cells in /scratch/cavestru_root/cavestru0/mfho/res_oos/ (see
docs/superpowers/2026-07-10-oos-instrument-resolution-phase2-results.md in the notes repo for
the full per-cell table, mean-flux co-report, and ops notes). Blind-safe: every quantity plotted
is a paired |Delta_bias_z|+2SE or Delta_bias_z in sigma_ref units, never an absolute A_p or n_s.

Pure numpy/matplotlib, no NUTS/GPU, run inline:
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya /home/mfho/.conda/envs/emu-jax/bin/python3 \
      scripts/plot_res_oos_phase2_gate.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({"font.size": 12, "figure.facecolor": "white", "savefig.facecolor": "white",
                      "axes.grid": True, "grid.alpha": 0.25})

OUT = "/home/mfho/hcd_priya_notes/figures/analysis/08_resolution/res_instr_oos_phase2_gate.png"

# cell label -> (n_s |Delta|+2SE, n_s verdict, A_p |Delta|+2SE, A_p verdict, n_s Delta_mean, N, ndiv)
# read directly from `analyze_dnuis_bias.py --shard-dir <cell>`, 2026-07-10.
CELLS = [
    ("DESI\n-1sigma",  0.360, "FLAG", 0.094, "PASS", -0.023, 7, 0),
    ("DESI\n+1sigma",  0.092, "PASS", 0.215, "PASS", +0.052, 8, 0),
    ("eBOSS\n-1sigma", 0.392, "FLAG", 0.270, "PASS", +0.153, 8, 0),
    ("eBOSS\n+1sigma", 0.367, "FLAG", 0.242, "PASS", -0.126, 8, 0),
    ("KS\n-1sigma",    0.312, "FLAG", 0.191, "PASS", -0.064, 8, 0),
    ("KS\n+1sigma",    0.168, "PASS", 0.237, "PASS", -0.026, 8, 0),
]

GOOD, WARN = "#0ca30c", "#fab219"
INK, MUTED = "#1a1a19", "#8a8a88"
VCOLOR = {"PASS": GOOD, "FLAG": WARN}

labels = [c[0] for c in CELLS]
ns_ub = [c[1] for c in CELLS]
ns_v = [c[2] for c in CELLS]
ap_ub = [c[3] for c in CELLS]
ap_v = [c[4] for c in CELLS]
ns_mean = [c[5] for c in CELLS]

x = np.arange(len(CELLS))
w = 0.36

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.5, 8.4), height_ratios=[2.1, 1.0])

# --- Panel A: grouped |Delta|+2SE bars, n_s and A_p, colored by verdict ---
b_ns = ax1.bar(x - w/2, ns_ub, w, color=[VCOLOR[v] for v in ns_v], edgecolor="k", linewidth=0.5)
b_ap = ax1.bar(x + w/2, ap_ub, w, color=[VCOLOR[v] for v in ap_v], edgecolor="k", linewidth=0.5,
                hatch="///")
for bar, val in zip(b_ns, ns_ub):
    ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.012, f"{val:.3f}", ha="center",
              va="bottom", fontsize=9.5, color=INK)
for bar, val in zip(b_ap, ap_ub):
    ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.012, f"{val:.3f}", ha="center",
              va="bottom", fontsize=9.5, color=INK)

ax1.axhline(0.30, color="crimson", ls="--", lw=1.6, zorder=0)
ax1.axhline(0.50, color="#7a1010", ls=":", lw=1.6, zorder=0)
ax1.text(len(CELLS) - 0.55, 0.30 + 0.012, "0.30 PASS gate", color="crimson", ha="right",
          fontsize=10, va="bottom")
ax1.text(len(CELLS) - 0.55, 0.50 + 0.012, "0.50 Tier-A adversarial FLAG budget", color="#7a1010",
          ha="right", fontsize=10, va="bottom")

ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=10.5)
ax1.set_ylabel(r"paired $|\Delta_{\rm mean}| + 2\,{\rm SE}$   ($\sigma_{\rm ref}$)")
ax1.set_ylim(0, 0.62)
ax1.set_title("Task-2 Phase-2: OOS-instrument-resolution ($b^*$) NUTS confirmation, "
              "$n_s$ vs $A_p$ gate\n(0 divergences on all 6 cells; no FAIL)", fontsize=13)

legend_handles = [
    Patch(facecolor=GOOD, edgecolor="k", label="PASS ($<0.30\\sigma$)"),
    Patch(facecolor=WARN, edgecolor="k", label="FLAG ($0.30$-$0.50\\sigma$, adversarial budget)"),
    Patch(facecolor="white", edgecolor="k", label="$n_s$ (solid)"),
    Patch(facecolor="white", edgecolor="k", hatch="///", label="$A_p$ (hatched)"),
]
ax1.legend(handles=legend_handles, loc="upper left", fontsize=9.5, ncol=2, framealpha=0.9)

# --- Panel B: n_s Delta_mean (near 0) vs n_s |Delta|+2SE, "FLAG is SE-driven" ---
b_mean = ax2.bar(x - w/2, ns_mean, w, color=MUTED, edgecolor="k", linewidth=0.5,
                  label=r"$n_s$ $\Delta_{\rm mean}$ (signed)")
b_ub2 = ax2.bar(x + w/2, ns_ub, w, color=[VCOLOR[v] for v in ns_v], edgecolor="k", linewidth=0.5,
                 alpha=0.9, label=r"$n_s$ $|\Delta_{\rm mean}|+2{\rm SE}$")
for bar, val in zip(b_mean, ns_mean):
    ax2.text(bar.get_x() + bar.get_width() / 2, val + (0.012 if val >= 0 else -0.018),
              f"{val:+.3f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=9,
              color=INK)
ax2.axhline(0.0, color=INK, lw=0.8)
ax2.axhline(0.30, color="crimson", ls="--", lw=1.2, zorder=0)
ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=10.5)
ax2.set_ylabel(r"$n_s$ $\sigma_{\rm ref}$")
ax2.set_ylim(-0.22, 0.45)
ax2.set_title(r"$n_s$ $|\Delta_{\rm mean}|$ small ($\lesssim 0.15\sigma$), but not uniformly SE-driven:"
              "\n"
              r"DESI $-1\sigma$/KS $-1\sigma$ FLAGs are single-outlier SE artifacts; eBOSS shows a "
              r"real, sign-coherent $\sim\!0.14\sigma$ response", fontsize=10.5)
ax2.legend(loc="upper left", fontsize=9.5, framealpha=0.9)

fig.tight_layout()
fig.savefig(OUT, dpi=150)
print(f"wrote {OUT}")
