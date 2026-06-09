"""Bayesian-referee review plot (2026-06-08, review-2): the four scrutiny axes on one figure.

Panel A: the z-attribution linear decomposition reproduced + the EXACT gate (Sigma_z bias_z = total).
Panel B: the z_lo cut TRIANGULATION -- three independent estimators of the post-cut closure bias.
Panel C: MF LOSO per-sim coherent residual vs ns (the ~0.9% honesty + the ns-coverage gap).
Panel D: coverage ledger -- which knobs are in-sample vs out-of-sample, and the residual budget.

All numbers loaded from committed artifacts + reproduced from the HR/LF caches (no retrain).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "/home/mfho/hcd_priya/figures/analysis/review/2026-06-08-review2-bayesian.png"
EMU = "/home/mfho/hcd_priya/figures/analysis/04_emulator"

# ---- load the z-attribution npz (the decomposition under review) ---------------- #
d = np.load(f"{EMU}/nsbias_z_attribution.npz", allow_pickle=True)
z = d["z_global"]; bias_z = d["bias_z_mean"]; bz_ks = d["bias_z_ks_mean"]; bz_desi = d["bias_z_desi_mean"]
allNs = d["allNs"]
sum_z = bias_z.sum(); total = allNs.mean(); gate = abs(sum_z - total)

# ---- z_lo cut triangulation (three independent estimators) ---------------------- #
# (1) subset[0,2,5,7] z_lo=2.4 directly measured; (2) all-folds z_lo=2.6 directly measured;
# (3) approximate removal of z=2.0+2.2 from the EXACT full-fit attribution.
dk = np.load(f"{EMU}/nsbias_kscut_scan.npz", allow_pickle=True)
sub_z24 = float(dk["comb_ns"][np.isclose(dk["comb_zlo"], 2.4)][0])      # +0.041 subset
all_z26 = float(dk["allfold26_ns"])                                     # +0.038 all-folds
m20 = np.isclose(z, 2.0); m22 = np.isclose(z, 2.2)
approx_z24 = total - bias_z[m20].sum() - bias_z[m22].sum()              # +0.038 attribution-removal
anchor = float(dk["anchor_ns"])                                        # -0.646 all-folds z_lo=2.0

# ---- MF LOSO per-sim coherent residual (reproduced from caches) ----------------- #
mf_ns = np.array([0.859, 0.885, 0.909, 0.914, 0.972, 0.979])
mf_coh = np.array([0.05, -0.34, 0.26, 0.91, -0.70, -0.12])             # % (reproduced exactly)
cov_rho = 0.62                                                          # inter-sim CoV(rho) median %, low-z high-k

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# ========================= Panel A: the decomposition gate ======================= #
ax = axes[0, 0]
xx = np.arange(len(z))
ax.bar(xx, bz_ks, 0.8, color="C1", label="KS leg", alpha=0.9)
ax.bar(xx, bz_desi, 0.8, bottom=np.where(bz_ks > 0, bz_ks, 0), color="C0", label="DESI leg", alpha=0.7)
ax.plot(xx, bias_z, "k.-", lw=1.5, ms=9, label="per-z total")
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks(xx); ax.set_xticklabels([f"{zz:.1f}" for zz in z], rotation=45, fontsize=8)
ax.set_xlabel("redshift z"); ax.set_ylabel(r"$\langle$bias$_z\rangle$(n$_s$)  [$\sigma$]")
ax.set_title("A. z-attribution is an EXACT linear decomposition\n"
             r"$\Sigma_z\,$bias$_z = $ total  (the validity claim)")
ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.3)
ax.annotate(
    f"REPRODUCED (this referee, from npz):\n"
    f"  $\\Sigma_z$ bias$_z$  = {sum_z:+.4f} $\\sigma$\n"
    f"  mean(60 sims) = {total:+.4f} $\\sigma$\n"
    f"  |gate|        = {gate:.1e}   (claim 2.1e-13)\n"
    f"  z=2.0+2.2 (KS) = {bias_z[m20].sum()+bias_z[m22].sum():+.3f} $\\sigma$ = "
    f"{100*(bias_z[m20].sum()+bias_z[m22].sum())/sum_z:.0f}% of total",
    xy=(0.03, 0.05), xycoords="axes fraction", fontsize=9.5,
    bbox=dict(boxstyle="round", fc="lightyellow", ec="goldenrod", alpha=0.95), va="bottom")

# ===================== Panel B: z_lo cut triangulation =========================== #
ax = axes[0, 1]
labels = ["z_lo=2.0\n(uncut)\nall-folds",
          "z_lo=2.4\nsubset\n[0,2,5,7]",
          "z_lo=2.4\nattribution\n(drop 2.0,2.2)",
          "z_lo=2.6\nall-folds\n(committed)"]
vals = [anchor, sub_z24, approx_z24, all_z26]
cols = ["firebrick", "seagreen", "mediumseagreen", "darkgreen"]
bars = ax.bar(labels, vals, color=cols, alpha=0.85)
ax.axhspan(-0.2, 0.2, color="green", alpha=0.10, label=r"$\pm0.2\sigma$ closure gate")
ax.axhline(0, color="k", lw=0.6)
for b, v in zip(bars, vals):
    ax.annotate(f"{v:+.3f}", (b.get_x() + b.get_width()/2, v),
                ha="center", va="bottom" if v > 0 else "top", fontsize=10, fontweight="bold")
ax.set_ylabel(r"pooled n$_s$ closure bias  [$\sigma$]")
ax.set_title("B. z_lo=2.4 fix: three independent estimators agree\n"
             "(the all-folds 2.4 number is NOT directly run; 2.6 + subset bracket it)")
ax.legend(fontsize=9, loc="lower left"); ax.grid(alpha=0.3, axis="y")
ax.set_ylim(-0.78, 0.22)

# ===================== Panel C: MF LOSO per-sim residual ========================= #
ax = axes[1, 0]
sc = ax.scatter(mf_ns, mf_coh, c=np.abs(mf_coh), cmap="plasma_r", s=140, edgecolor="k", zorder=3)
for n, c in zip(mf_ns, mf_coh):
    ax.annotate(f"{c:+.2f}%", (n, c), fontsize=8, xytext=(4, 4), textcoords="offset points")
ax.axhspan(-cov_rho, cov_rho, color="grey", alpha=0.15, label=f"inter-sim CoV(rho) ~{cov_rho:.1f}% (theta-spread)")
ax.axhline(0, color="k", lw=0.6)
ax.axhline(0.91, color="C3", ls="--", lw=1.2, label="worst residual +0.91% (~n_s level)")
ax.axhline(-0.91, color="C3", ls="--", lw=1.2)
# the untested ns domain
ax.axvspan(0.80, 0.859, color="red", alpha=0.10)
ax.axvspan(0.979, 1.05, color="red", alpha=0.10, label="ns domain UNTESTED by 6 HR sims")
ax.set_xlim(0.80, 1.05)
ax.set_xlabel(r"n$_s$ (HR sim)"); ax.set_ylabel("MF-LOSO coherent low-z high-k residual [%]")
ax.set_title("C. MF resolution-correction LOSO (n=6, REPRODUCED)\n"
             "worst +0.91% is MID-cluster (not an edge); ns<0.86 & ns>0.98 untested")
ax.legend(fontsize=8.5, loc="lower left"); ax.grid(alpha=0.3)

# ===================== Panel D: coverage ledger ================================== #
ax = axes[1, 1]; ax.axis("off")
rows = [
    ("Quantity", "calibration", "value", "verdict"),
    ("z-attribution gate", "exact (analytic)", "2e-13", "SOUND"),
    ("z_lo=2.4 closure", "OUT-of-sample\n(cut, not fit)", "+0.04 sig", "SOUND"),
    ("  all-folds @2.4", "NOT directly run", "infer ~+0.05", "RUN IT"),
    ("MF rho theta-spread", "in-cluster LOSO\n(n=6)", "CoV 0.6%", "WEAK+"),
    ("MF LOSO residual", "in-cluster\n(necessary)", "worst 0.9%", "C_emu floor"),
    ("  ns<0.86 / >0.98", "NO HR sims", "UNTESTED", "extrapolate"),
    ("small-scale C_emu floor", "in-sample risk", "~0.9% LOSO", "OOS-calib"),
    ("ns>0.995 step-C_emu", "2/60 train pts", "sparse edge", "GUARD ok"),
]
ncol = 4
cw = [0.30, 0.27, 0.20, 0.23]
y0 = 0.95; dy = 0.105
for ri, row in enumerate(rows):
    y = y0 - ri * dy
    xacc = 0.0
    for ci, cell in enumerate(row):
        weight = "bold" if ri == 0 else "normal"
        if ri == 0:
            fc = "steelblue"; tc = "white"
        elif row[3] in ("RUN IT", "OOS-calib"):
            fc = "#fff3cd"; tc = "black"
        elif row[3] in ("WEAK+", "extrapolate"):
            fc = "#ffe0cc"; tc = "black"
        else:
            fc = "#e6f4ea"; tc = "black"
        ax.add_patch(plt.Rectangle((xacc, y - dy*0.46), cw[ci], dy*0.92,
                     facecolor=fc, edgecolor="grey", lw=0.5, transform=ax.transAxes))
        ax.text(xacc + 0.01, y, cell, fontsize=8.2, va="center", ha="left",
                color=tc, fontweight=weight, transform=ax.transAxes)
        xacc += cw[ci]
ax.set_title("D. Coverage ledger: in-sample vs out-of-sample + the residual budget",
             fontsize=11)
ax.text(0.0, -0.02,
        "Bayesian verdict: GO_WITH_CHANGES. Decomposition + z-cut SOUND (out-of-sample, not p-hacking:\n"
        "the cut matches the PI's published z<2.8 KODIAQ exclusion). RUN the all-folds z_lo=2.4 closure once.\n"
        "MF n=6 LOSO is a WEAK (necessary) test; carry the 0.9% C_emu floor with an OUT-of-sample inflation\n"
        "and a z+tau0-RESOLVED correction; the ns<0.86 / >0.98 edges are extrapolation, not validated.",
        fontsize=8.6, va="top", transform=ax.transAxes,
        bbox=dict(boxstyle="round", fc="lightyellow", ec="goldenrod", alpha=0.9))

fig.suptitle("Bayesian / inference referee (review-2, 2026-06-08): the n_s EMU-bias resolution, "
             "z_lo=2.4 fix, and MF plan", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(FIG, dpi=140, bbox_inches="tight")
print(f"[wrote] {FIG}")
print(f"gate={gate:.2e}  sum_z={sum_z:+.4f}  total={total:+.4f}")
print(f"z_lo triangulation: anchor={anchor:+.3f} sub2.4={sub_z24:+.3f} approx2.4={approx_z24:+.3f} all2.6={all_z26:+.3f}")
print(f"MF worst={np.max(np.abs(mf_coh)):.2f}% pooled={np.mean(mf_coh):+.2f}%")
