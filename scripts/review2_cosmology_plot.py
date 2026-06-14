"""Cosmology-lens review-2 plot (2026-06-08): the four physics checks in one figure.

Panel A: z-attribution of the -0.65sigma n_s bias (KS vs DESI), with the z_lo cuts marked
         -> shows the bias lives ENTIRELY in KS z=2.0+2.2; z=2.4/2.6 are clean (the
            2.4-vs-2.8 cut is closure-indistinguishable).
Panel B: independent recompute of the LF-HR clean-P1D deficit z-trend (filtered clean
         class, this reviewer's own cache read) overlaid on the PRIYA ~7% convergence band
         -> sign + k-growth check; note the z>3.6 SIGN FLIP (LF over-predicts at high z).
Panel C: the MF resolution-correction rho vs tau0 at low z (from the on-disk vs_tau0_z txt)
         -> the tau0-dependence of a *resolution* correction is real (~4pp) and physical
            (high <F> = more resolution-sensitive small-scale structure).
Panel D: HR-sim n_s coverage vs the design + eBOSS/Planck landing
         -> the MF correction's ns in [0.86,0.98] gap relative to n_P~1.0.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py

OUT = "/home/mfho/hcd_priya/figures/analysis/review/2026-06-08-review2-cosmology.png"

# ---- Panel A data: z-attribution npz -----------------------------------------
za = np.load("/home/mfho/hcd_priya/figures/analysis/04_emulator/nsbias_z_attribution.npz",
             allow_pickle=True)
zg = za["z_global"]; bks = za["bias_z_ks_mean"]; bds = za["bias_z_desi_mean"]
total = float(za["allNs"].mean())

# ---- Panel B data: independent LF-HR filtered-clean deficit z-trend ----------
hr = h5py.File("/home/mfho/hcd_priya/hcd_analysis/_emulator_data/observables_tau0_hr.h5", "r")
lf = h5py.File("/home/mfho/hcd_priya/hcd_analysis/_emulator_data/observables_tau0_lf.h5", "r")
Pf_hr = hr["P_tier_c_filtered"][:, 0, :]; Pf_lf = lf["P_tier_c_filtered"][:, 0, :]
khr = hr["kfkms"][:]; klf = lf["kfkms"][:]
zhr = hr["z_grid"][:]; zlf = lf["z_grid"][:]
pa_hr = hr["params"][:]; pa_lf = lf["params"][:]
ai_hr = hr["alpha_idx"][:]; ai_lf = lf["alpha_idx"][:]
lf_idx = {}
for i in range(len(zlf)):
    lf_idx[(tuple(np.round(pa_lf[i], 6)), round(float(zlf[i]), 4), int(ai_lf[i]))] = i
from collections import defaultdict
ztrend = defaultdict(list)
for h in range(len(zhr)):
    k = (tuple(np.round(pa_hr[h], 6)), round(float(zhr[h]), 4), int(ai_hr[h]))
    if k not in lf_idx:
        continue
    l = lf_idx[k]; z = round(float(zhr[h]), 4)
    Phr_h = Pf_hr[h]; Plf_h = Pf_lf[l]; khr_h = khr[h]; klf_h = klf[l]
    mhr = np.isfinite(Phr_h) & (Phr_h > 0); mlf = np.isfinite(Plf_h) & (Plf_h > 0)
    if mhr.sum() < 2 or mlf.sum() < 2:
        continue
    sel = mlf & (klf_h >= khr_h[mhr].min()) & (klf_h <= khr_h[mhr].max()) & (klf_h > 0.04)
    if not sel.any():
        continue
    Phr_on = np.exp(np.interp(np.log(klf_h[sel]), np.log(khr_h[mhr]), np.log(Phr_h[mhr])))
    ztrend[z].append(((Plf_h[sel] - Phr_on) / Phr_on).mean())
zs = np.array(sorted(z for z in ztrend if z <= 4.6))
dz = np.array([100 * np.mean(ztrend[z]) for z in zs])

# ---- Panel C: rho vs tau0 (from the vs_tau0_z txt, k~0.0488 row) -------------
# numbers transcribed from figures/analysis/04_emulator/mf_rescorr_vs_tau0_z.txt
rungs = np.array([0, 9, 19]); alpha_slope = np.array([0.656, 0.976, 1.331])
rho_z22 = np.array([1.0761, 1.0562, 1.0393])   # z=2.2, k~0.049
rho_z36 = np.array([1.0126, 1.0133, 1.0079])   # z=3.6
rho_z50 = np.array([1.0257, 0.9868, 0.9444])   # z=5.0 (sign flip at high tau0)

# ---- Panel D: HR-sim n_s coverage vs design + landmarks ----------------------
hr_ns = np.array([0.859, 0.885, 0.909, 0.914, 0.972, 0.979])

# ============================ FIGURE =========================================
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Cosmology-lens review #2 (2026-06-08): physics audit of the n_s-bias resolution + MF plan",
             fontsize=12, fontweight="bold")

# Panel A
ax = axes[0, 0]
w = 0.35
ax.bar(zg - w / 2, bks, w, label="KS leg", color="crimson", alpha=0.85)
ax.bar(zg + w / 2, bds, w, label="DESI leg", color="steelblue", alpha=0.85)
ax.axhline(0, color="k", lw=0.6)
ax.axvspan(1.9, 2.3, color="grey", alpha=0.18)
ax.axvline(2.35, color="green", ls="--", lw=1.4, label="z_lo=2.4 (implemented)")
ax.axvline(2.75, color="purple", ls=":", lw=1.4, label="z_lo=2.8 (published KS cut)")
ax.text(2.0, -0.42, "z=2.0,2.2\ncarry 105%\nof bias", fontsize=8, ha="center", color="darkred")
ax.text(2.5, 0.04, "z=2.4,2.6\nKS = +0.008sig\n(clean)", fontsize=8, ha="center", color="green")
ax.set_xlabel("z"); ax.set_ylabel(r"per-z $\langle$bias$\rangle$ in $n_s$ [$\sigma$]")
ax.set_title(f"(A) n_s-bias z-attribution  (total={total:+.2f}sig; DESI={bds.sum():+.2f}sig CLEAN)",
             fontsize=10)
ax.legend(fontsize=7, loc="center right"); ax.set_xlim(1.8, 4.7); ax.set_ylim(-0.55, 0.12)

# Panel B
ax = axes[0, 1]
ax.axhspan(-7, 7, color="gold", alpha=0.15, label="PRIYA published ~7% convergence")
ax.axhline(0, color="k", lw=0.6)
ax.plot(zs, dz, "o-", color="darkorange", lw=1.6, label="LF-HR deficit (this reviewer's recompute)")
ax.axhline(-7, color="goldenrod", ls="--", lw=0.8); ax.axhline(7, color="goldenrod", ls="--", lw=0.8)
ax.fill_between([2.0, 3.4], -11, 0, color="crimson", alpha=0.08)
ax.fill_between([3.5, 4.6], 0, 3, color="steelblue", alpha=0.10)
ax.text(2.4, -10, "LF DEFICIENT\n(low-z, high-k)", fontsize=8, color="darkred")
ax.text(3.9, 1.6, "LF EXCESS\n(SIGN FLIP\nz>3.6)", fontsize=8, color="navy")
ax.set_xlabel("z"); ax.set_ylabel(r"$(P_{LF}-P_{HR})/P_{HR}$ at $k>0.04$ s/km [%]")
ax.set_title("(B) LF-HR clean-P1D deficit: sign + k-growth + z-trend", fontsize=10)
ax.legend(fontsize=7, loc="upper left"); ax.set_ylim(-11, 4)

# Panel C
ax = axes[1, 0]
ax.plot(alpha_slope, 100 * (rho_z22 - 1), "s-", color="crimson", label="z=2.2 (low-z, MF regime)")
ax.plot(alpha_slope, 100 * (rho_z36 - 1), "o-", color="green", label="z=3.6")
ax.plot(alpha_slope, 100 * (rho_z50 - 1), "^--", color="navy", label="z=5.0 (sign flip)")
ax.axhline(0, color="k", lw=0.6)
ax.annotate("", xy=(0.70, 7.6), xytext=(1.33, 3.9),
            arrowprops=dict(arrowstyle="->", color="crimson", lw=1.2))
ax.text(0.95, 8.1, "~3.7pp tau0 swing\n(high <F> = more\nresolution-sensitive)", fontsize=7.5, color="darkred")
ax.set_xlabel(r"$\alpha_{slope}$ (tau0 multiplier;  low = high $\langle F\rangle$)")
ax.set_ylabel(r"$(\rho-1)$ at k~0.049 [%]   ($\rho=P_{HR}/P_{LF}$)")
ax.set_title("(C) MF correction rho vs tau0: a real, physical tau0-dependence", fontsize=10)
ax.legend(fontsize=7.5)

# Panel D
ax = axes[1, 1]
ax.scatter(hr_ns, np.ones_like(hr_ns), s=80, color="crimson", zorder=5,
           label="6 HR sims (MF-trained)")
ax.axvspan(0.859, 0.979, color="crimson", alpha=0.12, label="HR coverage [0.86,0.98]")
ax.axvspan(0.995, 1.05, color="orange", alpha=0.18, label="sparse extension (2/60 design pts)")
ax.axvline(0.965, color="grey", ls=":", lw=1, label="Planck n_s~0.965")
ax.axvline(1.009, color="purple", ls="--", lw=1.4, label="eBOSS n_P~1.009")
ax.axvline(0.995, color="darkorange", ls="-", lw=1, label="C_emu step at 0.995")
ax.set_xlim(0.80, 1.05); ax.set_ylim(0.5, 1.5); ax.set_yticks([])
ax.set_xlabel(r"$n_s$ (primordial tilt $n_P$)")
ax.set_title("(D) HR (MF) n_s coverage gap vs the real-fit landing (~1.0)", fontsize=10)
ax.text(1.005, 1.30, "real fit lands\nHERE, in the\nMF coverage gap\n+ sparse extension",
        fontsize=7.5, color="purple", ha="center")
ax.legend(fontsize=6.5, loc="lower left")

fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT, dpi=130)
print("saved", OUT)
print(f"PanelA total={total:+.4f}  KS(z<=2.2)={bks[zg<=2.25].sum():+.4f}  DESI={bds.sum():+.4f}")
print(f"PanelB low-z deficit z=2.0: {dz[0]:+.2f}%  z=3.4: {dz[zs==3.4][0] if (zs==3.4).any() else float('nan'):+.2f}%")
