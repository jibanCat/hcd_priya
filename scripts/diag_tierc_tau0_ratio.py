"""Per-k ratio of the unfiltered Tier-C P1D response to alpha, to separate a
flat (mean-flux-normalization) level shift from a genuine scale-dependent
response. Loads the npz dumped by diag_tierc_tau0_response.py -- no recompute.

A flat ratio vs k => pure level/normalization response.
A k-tilted ratio => scale-dependent (e.g. forest pixels at high-k respond
while damping wings at low-k stay alpha-insensitive).
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("figures/analysis/03_templates_and_p1d")
d = np.load(OUT / "tierc_tau0_response_z3.npz")
kf = d["kf"]
alphas = list(d["alphas"])            # [0.70, 1.00, 1.35]
ilo, imid, ihi = 0, 1, 2
NAMES = ["clean", "LLS", "subDLA", "DLA"]
series = {n: d[f"P_{n}"] for n in NAMES}     # each (3, n_k)
series["Tier P (total)"] = d["P_tier_p"]
panels = NAMES + ["Tier P (total)"]

m = kf > 0
k = kf[m]

fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
axes = axes.ravel()
for ax, name in zip(axes, panels):
    P = series[name]
    Plo, Pmid, Phi = P[ilo][m], P[imid][m], P[ihi][m]
    with np.errstate(invalid="ignore", divide="ignore"):
        r_lo = Plo / Pmid
        r_hi = Phi / Pmid
    ax.semilogx(k, r_hi, "C2", label=f"alpha={alphas[ihi]:.2f} / 1.00")
    ax.semilogx(k, r_lo, "C0", label=f"alpha={alphas[ilo]:.2f} / 1.00")
    ax.axhline(1.0, color="k", lw=0.7, ls=":")
    # annotate low-k vs high-k response level to read off scale-dependence
    lo_band = (k > 1e-3) & (k < 3e-3)
    hi_band = (k > 3e-2) & (k < 1e-1)
    def band_mean(r, b):
        v = r[b]; v = v[np.isfinite(v)]
        return float(np.mean(v)) if v.size else np.nan
    txt = (f"alpha=hi/mid: low-k={band_mean(r_hi,lo_band):.3f}  "
           f"high-k={band_mean(r_hi,hi_band):.3f}")
    ax.set_title(f"{name}\n{txt}", fontsize=9)
    ax.set_xlabel("k  [s/km, angular]")
    ax.set_ylabel(r"$P(k,\alpha)\,/\,P(k,\alpha{=}1)$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
axes[-1].axis("off")
fig.suptitle("Per-k alpha response ratio (unfiltered Tier-C)  —  "
             "flat = normalization; tilted = scale-dependent", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "tierc_tau0_ratio_z3.png", dpi=150)
print(f"wrote {OUT/'tierc_tau0_ratio_z3.png'}")
