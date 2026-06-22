"""Overlay + ratio plot for the fake_spectra-based consistency check.

Reads:  hcd_priya_notes/docs/superpowers/figs/2026-05-20-priya-p1d-consistency.npz
Writes: hcd_priya_notes/docs/superpowers/figs/2026-05-20-priya-p1d-overlay.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_FIGS = Path(__file__).resolve().parent.parent.parent / "docs" / "superpowers" / "figs"
d = np.load(_FIGS / "2026-05-20-priya-p1d-consistency.npz")

kp = d["priya_kfkms"]
Pp = d["priya_p1d"]
km = d["kf_kms"]
Pm = d["p1d_mine"]
Pm_on_p = d["p1d_mine_on_priya"]
ratio = d["ratio"]

fig, axes = plt.subplots(2, 1, figsize=(8.5, 8.5), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1.4], "hspace": 0.05})

# top: log-log P1D overlay
ax = axes[0]
finite = (km > 0) & (Pm > 0) & np.isfinite(Pm)
ax.loglog(km[finite], Pm[finite], "C0-", lw=1.0, alpha=0.7,
          label=f"mine (fake_spectra.flux_power, {finite.sum()} native bins)")
ax.loglog(kp, Pp, "C3.", ms=4, alpha=0.85,
          label="PRIYA (flux_vectors[344, z=3], 172 bins)")
ax.loglog(kp, Pm_on_p, "C0o", ms=2.5, mfc="none", alpha=0.85,
          label="mine interp → PRIYA grid")
ax.set_ylabel(r"$P_F(k)$  [s km$^{-1}$]")
ax.set_title(
    f"PRIYA vs mine — sim 0 (ns0.80325...), z=3.0, α (tau0/Kim) = {float(d['alpha']):.6f}\n"
    f"mean_flux_desired = {float(d['mean_flux_desired']):.6f}  →  "
    f"scale_inverted = {float(d['scale_inverted']):.6f}  "
    f"(filtered {int(d['n_filtered'])}/{int(d['n_total'])} sightlines)"
)
ax.legend(loc="lower left", fontsize=9)
ax.grid(True, which="both", alpha=0.2)

# bottom: ratio
ax = axes[1]
ax.semilogx(kp, ratio, "C2.-", ms=4)
ax.axhline(1.0, color="k", ls="--", lw=0.8, alpha=0.6)
m = np.nanmedian(ratio)
s = np.nanstd(ratio)
ax.axhline(m, color="C2", ls=":", lw=0.8, alpha=0.7,
           label=f"median = {m:.5f}  std = {s:.5f}")
ax.set_xlabel(r"$k$  [rad s km$^{-1}$, angular convention]")
ax.set_ylabel("mine / PRIYA")
# Tight y-range so the <0.1% structure is visible
ax.set_ylim(0.995, 1.012)
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, which="both", alpha=0.2)

out = _FIGS / "2026-05-20-priya-p1d-overlay.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"wrote {out}")
print(f"ratio median={np.nanmedian(ratio):.6f}  std={np.nanstd(ratio):.6f}  "
      f"max|r-1|={np.max(np.abs(ratio-1)):.4g}")
