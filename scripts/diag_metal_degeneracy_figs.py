"""Explainer figures for the metal->A_p/n_s degeneracy (READ-ONLY).
Fig 1: the metal contamination Delta P/P(k) decomposed into the k-independent broadband DC (a^2)
       and the k-structured oscillation, overlaid on the DESI vs eBOSS k-reach (the high-k lever).
Fig 2: the A_p<->tau0 amplitude degeneracy in the posterior, DESI vs eBOSS (from the shard pkls)."""
import glob, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/mfho/hcd_priya_notes/figures/analysis/07_gate_b_metal"
SHARD = "/scratch/cavestru_root/cavestru1/mfho/metal_zevo"
kd = np.unique(np.load("/home/mfho/data/desi_dr1_p1d/desi_dr1_p1d.npz")["k"])
ke = np.unique(np.load("/home/mfho/data/eboss_dr14_p1d/eboss_dr14_p1d.npz")["k"])
kd = kd[kd > 0]; ke = ke[ke > 0]
KD_MAX, KE_MAX = kd.max(), ke.max()

# ---- metal model pieces (z~3 representative): a=f/(1-<F>), f_SiIII=0.009 -> a~0.03 ----
C_KMS, LAM_LYA, LAM_SiIII = 299792.458, 1215.67, 1206.50
a = 0.030                                   # a_SiIII at z~3 (f=0.009, 1-<F>~0.30); a(z) 0.015..0.06
dv = C_KMS * np.log(LAM_LYA / LAM_SiIII)     # 2269.5 km/s
kSiIII = 0.05                                # arm1 decorrelation scale
kk = np.linspace(1e-4, 0.058, 3000)
D = 2.0 - 2.0 / (1.0 + np.exp(-kk / kSiIII))
dc = a ** 2 * np.ones_like(kk)               # BROADBAND (k-independent) -> amplitude-degenerate
osc = 2.0 * a * np.cos(kk * dv) * D          # k-structured metal fingerprint
tot = dc + osc

# =========================================================================
# FIG 1: decomposition + survey k-reach
# =========================================================================
fig, ax = plt.subplots(figsize=(11, 6))
ax.axhspan(0, 0, color="none")
ax.axvspan(ke.min(), KE_MAX, color="#d62728", alpha=0.06, zorder=0)
ax.axvspan(KE_MAX, KD_MAX, color="#1f77b4", alpha=0.09, zorder=0,
           label="DESI-only high-k lever (eBOSS blind here)")
ax.plot(kk, 100 * tot, color="0.55", lw=1.3, alpha=0.9, label=r"total metal $\Delta P/P$", zorder=2)
ax.plot(kk, 100 * osc, color="#2077b4", lw=1.6, label=r"oscillation $2a\cos(k\Delta v)D(k)$ (k-structured $\to n_s$)", zorder=3)
ax.axhline(100 * a ** 2, color="#c0392b", lw=2.2, ls="--",
           label=r"broadband DC $a^2$ (k-INDEPENDENT $\to A_p/\tau_0$)", zorder=4)
# survey k-points as rug markers
ax.plot(kd, np.full_like(kd, -6.4), "|", color="#1f77b4", ms=9, mew=1.2, label=f"DESI k-grid (85 pts, max {KD_MAX:.3f})")
ax.plot(ke, np.full_like(ke, -6.9), "|", color="#d62728", ms=9, mew=1.2, label=f"eBOSS k-grid (35 pts, max {KE_MAX:.3f})")
ax.axvline(KE_MAX, color="#d62728", lw=1.3, ls=":")
ax.axvline(KD_MAX, color="#1f77b4", lw=1.3, ls=":")
ax.annotate("eBOSS stops here\n(no high-k modes to\nseparate metal from cosmology)",
            (KE_MAX, 4.2), (KE_MAX - 0.017, 4.4), fontsize=9, color="#a01919",
            arrowprops=dict(arrowstyle="->", color="#a01919"))
ax.annotate("DC is a flat offset -> looks like an\namplitude change -> A_p & tau0 absorb it",
            (0.045, 100 * a ** 2), (0.028, 2.3), fontsize=9, color="#7a1f1f",
            arrowprops=dict(arrowstyle="->", color="#7a1f1f"))
ax.set_xlim(0, 0.058); ax.set_ylim(-7.3, 6.8)
ax.set_xlabel(r"$k$  [s/km]"); ax.set_ylabel(r"metal contamination  $\Delta P/P$  [%]")
ax.set_title("Why the metal leaks into cosmology: a flat broadband term (amplitude-degenerate) "
             "+ a k-structured wiggle\nDESI's high-k reach measures the wiggle and subtracts it; eBOSS cannot",
             fontsize=11)
ax.legend(loc="upper right", fontsize=8.4, ncol=1, framealpha=0.95)
ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(f"{OUT}/degen_kspace_decomp.png", dpi=135); plt.close(fig)

# =========================================================================
# FIG 2: the A_p <-> tau0 degeneracy in the posterior, DESI vs eBOSS
# =========================================================================
def bz(rec, nm):
    j = list(rec["names"]).index(nm); dc = np.asarray(rec["draws"])[:, j]
    return (dc.mean() - rec["truth_vec"][j]) / dc.std()
def bze(rec, nm):
    se = rec.get("sites_extra") or {}; d = np.asarray(se[nm]["draws"])
    return (d.mean() - se[nm]["truth"]) / d.std()
def load(arm, sv):
    cl, inj = [], []
    for p in sorted(glob.glob(f"{SHARD}/metal_zevo_{arm}_{sv}_shard_*.pkl")):
        d = pickle.load(open(p, "rb")); cl += d["clean_per_mock"]; inj += d["inj_per_mock"]
    return cl, inj
INCLASS = ["arm1_decreasing", "arm2_increasing"]
pts = {"desi": {"ap": [], "t0": []}, "eboss": {"ap": [], "t0": []}}
for sv in ("desi", "eboss"):
    for arm in INCLASS:
        cl, inj = load(arm, sv); n = min(len(cl), len(inj))
        for m in range(n):
            pts[sv]["ap"].append(bz(inj[m], "Ap") - bz(cl[m], "Ap"))
            pts[sv]["t0"].append(bze(inj[m], "tau0_amp") - bze(cl[m], "tau0_amp"))
fig, ax = plt.subplots(figsize=(7.4, 6.6))
for sv, c, lab in (("desi", "#1f77b4", "DESI (high-k lever)"), ("eboss", "#d62728", "eBOSS (no high-k lever)")):
    x = np.array(pts[sv]["t0"]); y = np.array(pts[sv]["ap"])
    ax.scatter(x, y, c=c, s=46, alpha=0.8, edgecolor="k", linewidth=0.3,
               label=f"{lab}: mean A_p {y.mean():+.2f}")
    mx, my = x.mean(), y.mean()
    ax.plot([mx], [my], marker="*", ms=20, color=c, mec="white", mew=1.2, zorder=6)
allx = np.array(pts["desi"]["t0"] + pts["eboss"]["t0"]); ally = np.array(pts["desi"]["ap"] + pts["eboss"]["ap"])
b, a0 = np.polyfit(allx, ally, 1); xs = np.linspace(allx.min(), allx.max(), 40)
ax.plot(xs, a0 + b * xs, "k-", lw=1.6, alpha=0.7, label=f"degeneracy axis (slope {b:.2f})")
ax.axhline(0, color="0.5", lw=0.7); ax.axvline(0, color="0.5", lw=0.7)
ax.plot(0, 0, marker="+", ms=14, mew=2, color="k")
ax.set_xlabel(r"$\Delta$bias$_z$( mean-flux amplitude $\tau_0$ )")
ax.set_ylabel(r"$\Delta$bias$_z$( $A_p$ )")
ax.set_title("The A_p <-> tau0 amplitude degeneracy (in-class arms)\n"
             "both surveys ride the same axis; eBOSS (no high-k lever) rides it FARTHER from truth",
             fontsize=10.5)
ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(f"{OUT}/degen_ap_tau0_desi_vs_eboss.png", dpi=135); plt.close(fig)
print("wrote degen_kspace_decomp.png + degen_ap_tau0_desi_vs_eboss.png")
print(f"  DESI in-class: mean dAp {np.mean(pts['desi']['ap']):+.3f}  dtau0 {np.mean(pts['desi']['t0']):+.3f}  (n={len(pts['desi']['ap'])})")
print(f"  eBOSS in-class: mean dAp {np.mean(pts['eboss']['ap']):+.3f}  dtau0 {np.mean(pts['eboss']['t0']):+.3f}  (n={len(pts['eboss']['ap'])})")
print(f"  DESI k_max {KD_MAX:.4f}  eBOSS k_max {KE_MAX:.4f}  ratio {KD_MAX/KE_MAX:.2f}x")
