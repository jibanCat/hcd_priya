"""Linearity / template-stability check for the KS LLS ×2.5 prior boost.

The forward P_obs = P_clean + Σ_c α_c·(P_filt,c − P_clean) is EXACTLY linear in α by
construction, so boosting α_LLS to the KS level (~2.5×) is only safe if the per-class LLS
EXCESS TEMPLATE shape e_LLS(k) ≡ (P_filt,LLS − P_clean)/P_clean is INCIDENCE-INDEPENDENT —
i.e. it does not change as the LLS sightline fraction w_LLS rises. If sims with low vs high
w_LLS share the same e_LLS(k) shape, the linear ×2.5 extrapolation is faithful.

Env: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/diag_lls_template_linearity.py
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import hcd_analysis.emulator  # x64
from hcd_analysis.emulator.data import load_cache, DATA_RANGE

d = load_cache("hcd_analysis/_emulator_data/observables_tau0_lf.h5")
Pf = np.asarray(d["P_filt"])              # (R,4,K) [clean,LLS,subDLA,DLA] filtered
wc = np.asarray(d["w_c_cache"])[:, 1:]    # (R,3) LLS,subDLA,DLA sightline fraction
z  = np.asarray(d["z_grid"]); k = np.asarray(d["kfkms"])[0]   # angular k grid (same across rows)
inband = (k >= DATA_RANGE["k_min"]) & (k <= 0.06)
# z≈3 slice, finite clean+LLS
zsel = np.isclose(z, 3.0, atol=0.25)
Pclean = Pf[:, 0, :]; Plls = Pf[:, 1, :]
e_lls = np.where(Pclean > 0, (Plls - Pclean) / Pclean, np.nan)   # (R,K) fractional LLS excess
row_ok = zsel & np.isfinite(e_lls[:, inband]).all(1) & (wc[:, 0] > 0)
E = e_lls[row_ok][:, inband]; W = wc[row_ok, 0]; kk = k[inband]
print(f"rows@z~3: {row_ok.sum()}; w_LLS range [{W.min():.3f},{W.max():.3f}] median {np.median(W):.3f}")

# tertiles of w_LLS
q = np.quantile(W, [1/3, 2/3])
bins = [("low w_LLS", W <= q[0]), ("mid", (W > q[0]) & (W <= q[1])), ("high w_LLS", W > q[1])]
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
colors = ["#1f77b4", "#2ca02c", "#d62728"]
shapes = []
for (lab, m), c in zip(bins, colors):
    mean = np.nanmean(E[m], axis=0); sd = np.nanstd(E[m], axis=0)
    shapes.append(mean)
    ax[0].fill_between(kk, mean - sd, mean + sd, color=c, alpha=0.15)
    ax[0].plot(kk, mean, "-o", color=c, ms=3, label=f"{lab} (n={m.sum()}, ⟨w⟩={W[m].mean():.3f})")
ax[0].set_xscale("log"); ax[0].axhline(0, color="k", lw=0.7)
ax[0].set_xlabel("k [s/km]"); ax[0].set_ylabel(r"LLS excess  $(P_{\rm LLS}-P_{\rm clean})/P_{\rm clean}$")
ax[0].set_title("LLS excess template shape by LLS abundance tertile", fontsize=11)
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, which="both")

# shape stability: normalize each tertile-mean to unit norm and compare; report max shape diff
norm = [s / np.sqrt(np.nansum(s**2)) for s in shapes]
shape_corr_lowhigh = float(np.nansum(norm[0]*norm[2]))
# also: does the AMPLITUDE of the excess correlate with w_LLS (it should NOT if template-stable)?
amp = np.nanmean(E, axis=1)
r_amp_w = float(np.corrcoef(amp, W)[0, 1])
ax[1].scatter(W, amp, s=10, alpha=0.5, color="#555")
ax[1].set_xlabel(r"$w_{\rm LLS}$ (LLS sightline fraction)")
ax[1].set_ylabel("mean in-band LLS excess amplitude")
ax[1].set_title(f"excess amplitude vs abundance\nr={r_amp_w:+.2f} (≈0 ⇒ template incidence-independent)", fontsize=11)
ax[1].grid(alpha=0.3)
fig.suptitle(f"LLS template linearity check — shape corr(low,high w_LLS)={shape_corr_lowhigh:.4f}  "
             f"(1.0 ⇒ identical shape ⇒ ×2.5 boost safe)", fontsize=12, y=1.0)
fig.text(0.5, -0.03, "The forward is exactly linear in α; this checks the FILTERED LLS excess "
         "TEMPLATE is incidence-independent so scaling α_LLS to the KS ~2.5× level is faithful.",
         ha="center", fontsize=9, style="italic")
fig.tight_layout()
out = "figures/analysis/05_likelihood/lls_template_linearity.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"shape corr(low,high w_LLS) = {shape_corr_lowhigh:.4f}")
print(f"corr(excess amplitude, w_LLS) = {r_amp_w:+.3f}  (near 0 => template shape/amp incidence-independent)")
print("wrote", out)
