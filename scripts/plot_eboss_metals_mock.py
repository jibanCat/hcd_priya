"""eBOSS DR14 mock P1D with SiIII metal contamination — the injected-ripple validation figure.

Builds the SAME eBOSS leg + fiducial truth as the E_f6_si cert arm (fold 6, n_s≈0.966, PRIYA
τ₀ anchor) and calls make_legb_mock TWICE with the SAME PRNGKey — once with inject_a_siiii=0.045
(the cert injection level, f_SiIII/(1-⟨F⟩)≈0.045) and once clean (0.0). Because the only difference
is the McDonald SiIII factor (data_likelihood._metal_factor), differencing the two truth-on-leg
curves isolates the ripple EXACTLY (same noise draw cancels in the ratio of the noiseless truths).

Emits ONE figure (notes repo, 05_truth_validation):
  top 3 panels  — P1D(k) at z≈2.4 / 3.0 / 3.8: clean truth (dashed) vs SiIII truth (solid) vs the
                  noisy mock data points (±√diag C_data). The ±9% ripple rides the smooth forest.
  bottom panel  — the SiIII ripple (P_SiIII/P_clean − 1)·100% vs k: the MEASURED ratio (points) over
                  the ANALYTIC _metal_factor−1 (line). k-independent z → one panel; annotates Δv,
                  the in-band period count, and the sigmoid-decorrelation damping at the band top.

ENV: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
     /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/plot_eboss_metals_mock.py
"""
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/mfho/hcd_priya/scripts")
import run_stepA  # noqa: E402  (config source of truth for the E_f6_si arm)
from hcd_analysis.emulator import closure_legb as C  # noqa: E402
from hcd_analysis.emulator import data_likelihood as DL  # noqa: E402

A_SIIII = 0.045
OUTDIR = "/home/mfho/hcd_priya_notes/figures/analysis/05_truth_validation"
os.makedirs(OUTDIR, exist_ok=True)

# --- pull the E_f6_si cert arm config (faithful fold/ckpt/sim/anchor) --------------------------- #
cfg = run_stepA.build_config()
arm = next(c for c in cfg if c["id"] == "E_f6_si_c0")
fold, ckpt, sim = int(arm["fold"]), arm["ckpt"], arm["sim"]
print(f"[metals-mock] E_f6_si arm: fold={fold} sim={sim} n_s={arm['n_s']} a_SiIII_inj={A_SIIII}")

# --- build the eBOSS leg ctx + the fiducial truth (PRIYA anchor, tau0_extreme=False) ------------ #
ctx, d = C.build_legb_ctx(ckpt=ckpt, with_eboss=True, sample_metals=True)
ctx = ctx._replace(legs=[l for l in ctx.legs if l.name == "eBOSS"])
truth = C.make_truth_from_sim(d, sim, fold=fold, tau0_anchor="priya")

# --- the same noise draw, with/without the SiIII injection ------------------------------------- #
key = jax.random.PRNGKey(int(arm["seed"]))
mock_si, _, info_si = C.make_legb_mock(ctx, truth, key, inject_a_siiii=A_SIIII)
mock_cl, _, info_cl = C.make_legb_mock(ctx, truth, key, inject_a_siiii=0.0)

leg = ctx.legs[0]
k = np.asarray(leg.k)
z_row = np.asarray(leg.z_row)                           # per-row redshift (N,), not the (n_z,) grid
sig_data = np.sqrt(np.diag(np.asarray(leg.C_data)))
t_si = np.asarray(info_si["truth_on_leg"]["eBOSS"])     # noiseless SiIII truth-on-leg
t_cl = np.asarray(info_cl["truth_on_leg"]["eBOSS"])     # noiseless clean truth-on-leg
p_data = np.asarray(mock_si[0].P_data)                  # noisy SiIII mock (the cert sees this)

# the analytic McDonald SiIII factor on a fine k-grid (the forward's _metal_factor, a_SiII=0)
kfine = np.linspace(k.min(), k.max(), 400)
mfac_fine = np.asarray(DL._metal_factor(jnp.asarray(kfine), a_SiIII=A_SIIII)) - 1.0
mfac_on_k = np.asarray(DL._metal_factor(jnp.asarray(k), a_SiIII=A_SIIII)) - 1.0

# Δv and the in-band period count (for the annotation)
dv = DL.C_KMS * np.log(DL.LAMBDA_LYA / DL.LAMBDA_SiIII)
period_k = 2.0 * np.pi / dv
n_periods = (k.max() - k.min()) / period_k
# sigmoid decorrelation damping at the band edges
D_lo = 2.0 - 2.0 / (1.0 + np.exp(-k.min() / DL.K_SiIII_DEFAULT))
D_hi = 2.0 - 2.0 / (1.0 + np.exp(-k.max() / DL.K_SiIII_DEFAULT))
print(f"[metals-mock] Δv_SiIII={dv:.1f} km/s  period_k={period_k:.5f} s/km  in-band periods≈{n_periods:.2f}")
print(f"[metals-mock] sigmoid D: low-k {D_lo:.3f} → high-k {D_hi:.3f}  (ripple ±{2*A_SIIII*100:.1f}% damped to ±{2*A_SIIII*D_hi*100:.1f}% at band top)")

# choose 3 representative z (nearest available leg z to 2.4 / 3.0 / 3.8)
z_unique = np.unique(z_row)
z_targets = [2.4, 3.0, 3.8]
z_show = [z_unique[int(np.argmin(np.abs(z_unique - zt)))] for zt in z_targets]

fig = plt.figure(figsize=(13, 8))
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.85], hspace=0.32, wspace=0.28)

for ci, zz in enumerate(z_show):
    ax = fig.add_subplot(gs[0, ci])
    m = np.isclose(z_row, zz) & np.isfinite(t_cl)
    order = np.argsort(k[m])
    kk = k[m][order]
    ax.plot(kk, t_cl[m][order], ls="--", color="0.45", lw=1.8, label="clean truth")
    ax.plot(kk, t_si[m][order], ls="-", color="C3", lw=1.8, label=f"+ SiIII ($a$={A_SIIII})")
    ax.errorbar(kk, p_data[m][order], yerr=sig_data[m][order], fmt="o", ms=4, color="C0",
                ecolor="0.7", elinewidth=0.9, capsize=2, label="mock data", zorder=5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(f"$z = {zz:.1f}$", fontsize=11)
    ax.set_xlabel(r"$k\ [\mathrm{s\,km^{-1}}]$")
    if ci == 0:
        ax.set_ylabel(r"$P_{\rm 1D}(k)$")
        ax.legend(fontsize=8.5, loc="lower left")
    ax.grid(alpha=0.25, which="both")

# bottom: the SiIII ripple (k-only; measured ratio over all kept rows vs the analytic factor)
axr = fig.add_subplot(gs[1, :])
good = np.isfinite(t_cl) & np.isfinite(t_si) & (t_cl != 0)
ratio = (t_si[good] / t_cl[good] - 1.0) * 100.0
axr.axhline(0, color="0.7", lw=0.8)
axr.plot(kfine, mfac_fine * 100.0, color="C3", lw=1.6,
         label=r"analytic $\mathcal{M}(k)-1$ (McDonald SiIII, $a_{\rm SiIII}=%.3f$)" % A_SIIII)
axr.scatter(k[good], ratio, s=14, color="C0", zorder=5, label="measured (truth$_{\\rm SiIII}$/truth$_{\\rm clean}-1$)")
axr.axhline(A_SIIII ** 2 * 100.0, color="0.5", ls=":", lw=1.0,
            label=r"constant offset $a^2=%.2f\%%$" % (A_SIIII ** 2 * 100))
axr.set_xlabel(r"$k\ [\mathrm{s\,km^{-1}}]$")
axr.set_ylabel(r"$P_{\rm SiIII}/P_{\rm clean}-1\ [\%]$")
axr.set_title(r"SiIII ripple — $1+a^2+2a\cos(k\,\Delta v)\,D(k)$,  "
              rf"$\Delta v={dv:.0f}$ km/s, $\approx{n_periods:.1f}$ periods in-band, "
              rf"$D$: {D_lo:.2f}$\to${D_hi:.2f}", fontsize=10)
axr.legend(fontsize=8.5, loc="upper right", ncol=2)
axr.grid(alpha=0.25)

fig.suptitle(f"eBOSS DR14 mock P1D with SiIII metals — E_f6 fiducial (fold {fold}, $n_s$={arm['n_s']}), "
             f"injected $a_{{\\rm SiIII}}={A_SIIII}$", fontsize=12)
out = f"{OUTDIR}/eboss_metals_mock.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"[metals-mock] wrote {out}")

# a compact stats line for the doc
print(f"[metals-mock] ripple range over the eBOSS band: "
      f"{ratio.min():.2f}% … {ratio.max():.2f}%  (n_rows={good.sum()})")
