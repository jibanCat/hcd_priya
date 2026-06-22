"""Ratio plot v3: fake_spectra direct call (with PRIYA's actual filter).

Reads:  hcd_priya_notes/docs/superpowers/figs/2026-05-20-priya-p1d-consistency-v3.npz
Writes: hcd_priya_notes/docs/superpowers/figs/2026-05-20-priya-p1d-ratio-v3.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_FIGS = Path(__file__).resolve().parent.parent.parent / "docs" / "superpowers" / "figs"
d = np.load(_FIGS / "2026-05-20-priya-p1d-consistency-v3.npz")
kp = d["priya_kfkms"]
r = d["ratio_direct"]

# Also overlay the v2 (with my Python port) ratio for contrast, if present.
v2 = _FIGS / "2026-05-20-priya-p1d-consistency.npz"
have_v2 = v2.exists()
if have_v2:
    d2 = np.load(v2)
    r2 = d2["ratio"]
    kp2 = d2["priya_kfkms"]

fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                         gridspec_kw={"height_ratios": [1, 1], "hspace": 0.08})

# top: zoomed-out (compare v2 port vs v3 fake_spectra direct)
ax = axes[0]
if have_v2:
    ok2 = np.isfinite(r2)
    ax.semilogx(kp2[ok2], r2[ok2], "C3.--", ms=4, lw=0.8, alpha=0.7,
                label=f"v2: my port of _filter_single_tau_complex  "
                      f"(med={np.nanmedian(r2):.5f}, σ={np.nanstd(r2):.5f})")
ax.semilogx(kp, r, "C0o-", ms=4, lw=1.0,
            label=f"v3: fake_spectra direct call  "
                  f"(med={np.median(r):.6f}, σ={np.std(r):.2e})")
ax.axhline(1.0, color="k", ls="--", lw=0.8, alpha=0.6)
ax.set_ylim(0.99, 1.015)
ax.set_ylabel(r"$P_\text{mine}\,/\,P_\text{PRIYA}$")
ax.set_title(
    "P_mine / P_PRIYA — sim 0, z=3.0, α (Kim slope) = 1.011183\n"
    "v3 uses fake_spectra._filter_single_tau_complex DIRECTLY  (was a periodic-BC port bug in v2)"
)
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, which="both", alpha=0.2)

# bottom: zoom into 1±0.00001 to show v3 is at machine precision
ax = axes[1]
ax.semilogx(kp, r, "C0o-", ms=4, lw=1.0,
            label=f"max|r-1| = {np.max(np.abs(r-1)):.2e}  (machine precision)")
ax.axhline(1.0, color="k", ls="--", lw=0.8, alpha=0.6)
ax.set_ylim(0.999995, 1.000005)
ax.set_xlabel(r"$k$  [rad s km$^{-1}$, angular]")
ax.set_ylabel(r"$P_\text{mine}\,/\,P_\text{PRIYA}$ (1e-6 zoom)")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, which="both", alpha=0.2)

out = _FIGS / "2026-05-20-priya-p1d-ratio-v3.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"wrote {out}")
print(f"median={np.median(r):.8f}  std={np.std(r):.2e}  max|r-1|={np.max(np.abs(r-1)):.2e}")
