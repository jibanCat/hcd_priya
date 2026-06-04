"""Ratio-only plot: P_mine / P_PRIYA vs k, at the tight 0.1% scale.

Reads:  docs/superpowers/figs/2026-05-20-priya-p1d-consistency.npz
Writes: docs/superpowers/figs/2026-05-20-priya-p1d-ratio.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_FIGS = Path(__file__).resolve().parent.parent.parent / "docs" / "superpowers" / "figs"
d = np.load(_FIGS / "2026-05-20-priya-p1d-consistency.npz")
kp = d["priya_kfkms"]
r = d["ratio"]
ok = np.isfinite(r)

m = np.nanmedian(r)
s = np.nanstd(r)
maxabs = np.nanmax(np.abs(r[ok] - 1))

fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True,
                         gridspec_kw={"height_ratios": [1, 1], "hspace": 0.05})

# panel 1: zoomed to 0.99–1.01 to show shape across full range
ax = axes[0]
ax.semilogx(kp[ok], r[ok], "C2.-", ms=4, lw=1.0)
ax.axhline(1.0, color="k", ls="--", lw=0.8, alpha=0.6)
ax.axhline(m, color="C2", ls=":", lw=0.8, alpha=0.7,
           label=f"median = {m:.5f}  std = {s:.5f}  max|r-1| = {maxabs:.4f}")
ax.fill_between(kp[ok], 0.99, 1.01, color="grey", alpha=0.05)
ax.set_ylim(0.99, 1.01)
ax.set_ylabel(r"$P_\text{mine}\,/\,P_\text{PRIYA}$")
ax.set_title("P_current / P_PRIYA — sim 0 (ns0.80325...), z = 3.0, α (Kim slope) = 1.011183\n"
             "Pipeline: fake_spectra.flux_power + _filter_single_tau_complex + _rescale_mean_flux")
ax.legend(loc="upper center", fontsize=10)
ax.grid(True, which="both", alpha=0.2)

# panel 2: zoomed to 0.998-1.002 (the bulk of the data — see real <0.1% structure)
ax = axes[1]
ax.semilogx(kp[ok], r[ok], "C2.-", ms=4, lw=1.0)
ax.axhline(1.0, color="k", ls="--", lw=0.8, alpha=0.6)
ax.axhline(m, color="C2", ls=":", lw=0.8, alpha=0.7)
ax.set_ylim(0.998, 1.002)
ax.set_xlabel(r"$k$  [rad s km$^{-1}$, angular]")
ax.set_ylabel(r"$P_\text{mine}\,/\,P_\text{PRIYA}$ (zoomed)")
ax.grid(True, which="both", alpha=0.2)

# annotate the single edge bin
worst_idx = int(np.nanargmax(np.abs(r - 1)))
ax_top = axes[0]
ax_top.annotate(f"k={kp[worst_idx]:.3e}, ratio={r[worst_idx]:.4f}\n(low-k interpolation edge — only bin >0.5 %)",
                xy=(kp[worst_idx], r[worst_idx]),
                xytext=(0.03, 0.85), textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="C3", lw=1.0),
                fontsize=9, color="C3")

out = _FIGS / "2026-05-20-priya-p1d-ratio.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"wrote {out}")
print(f"median = {m:.6f}, std = {s:.6f}, max|r-1| = {maxabs:.4f}")
print(f"99th percentile of |r-1|: {np.nanpercentile(np.abs(r[ok]-1), 99):.5f}")
print(f"95th percentile of |r-1|: {np.nanpercentile(np.abs(r[ok]-1), 95):.5f}")
