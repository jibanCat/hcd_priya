"""Diagnose the residual tilt in P_mine/P_PRIYA(k).

Two hypotheses:
1. k-grid mismatch: my fake_spectra k array != PRIYA's stored kfkms for the same
   (sim, z), so log-log interp picks up a tilt.
2. P at the same mode index actually disagrees with a tilt (would be a real
   bug — finite-pixel window, mode coupling, etc.).

Output: print direct mode-by-mode ratio (no interpolation) and the k-grid
diff. Save a new ratio plot.

Reads:  docs/superpowers/figs/2026-05-20-priya-p1d-consistency.npz
Writes: docs/superpowers/figs/2026-05-20-priya-p1d-ratio-direct.png

Note: the .npz from the v2 run stores `kf_kms` (my native, 625 bins) and
`priya_kfkms` (PRIYA stored, 172 bins). We can do the no-interp comparison
directly from those arrays.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_FIGS = Path(__file__).resolve().parent.parent.parent / "docs" / "superpowers" / "figs"
d = np.load(_FIGS / "2026-05-20-priya-p1d-consistency.npz")

km = d["kf_kms"]              # mine, 625 bins, modes 1..625
Pm = d["p1d_mine"]            # mine, 625 bins
kp = d["priya_kfkms"]         # PRIYA, 172 bins
Pp = d["priya_p1d"]           # PRIYA, 172 bins

# 1. Direct mode-by-mode comparison
n = len(kp)                            # 172
print(f"=== k-grid alignment ===")
print(f"  my native k[0..4]:    {km[:5]}")
print(f"  priya k[0..4]:        {kp[:5]}")
print(f"  my native k[-4:]:     {km[-4:]}")
print(f"  my k at index 168..171: {km[168:172]}")
print(f"  priya k at -4:        {kp[-4:]}")

# absolute and relative differences between my k[0:n] and priya's k[0:n]
k_mine_first_n = km[:n]
dk_abs = k_mine_first_n - kp
dk_rel = dk_abs / kp
print(f"\n  |dk/k| stats (n={n}):  max={np.max(np.abs(dk_rel)):.3e}, "
      f"mean={np.mean(np.abs(dk_rel)):.3e}, median={np.median(np.abs(dk_rel)):.3e}")
print(f"  dk_abs at first 4: {dk_abs[:4]}")
print(f"  dk_abs at last 4:  {dk_abs[-4:]}")

# 2. Direct power ratio at matched mode index (no interpolation)
P_mine_at_priya_modes = Pm[:n]
ratio_direct = P_mine_at_priya_modes / Pp
print(f"\n=== Direct mode-by-mode ratio (NO interpolation) ===")
print(f"  median = {np.median(ratio_direct):.6f}")
print(f"  mean   = {np.mean(ratio_direct):.6f}")
print(f"  std    = {np.std(ratio_direct):.6f}")
print(f"  min, max = {ratio_direct.min():.6f}, {ratio_direct.max():.6f}")
print(f"  max|r-1| = {np.max(np.abs(ratio_direct - 1)):.4g}")

# Sample bins (matches the earlier v2 sampling)
print("\n  Sample bins (direct, no interp):")
for i in [0, 5, 20, 50, 100, 150, 170]:
    if i < n:
        print(f"    mode m={i+1:3d}  k_mine={km[i]:.4e}  k_priya={kp[i]:.4e}  "
              f"P_mine={Pm[i]:.4e}  P_priya={Pp[i]:.4e}  ratio={ratio_direct[i]:.6f}")

# Plot direct ratio (no interpolation)
fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True,
                         gridspec_kw={"height_ratios": [1, 1], "hspace": 0.05})

ax = axes[0]
ax.semilogx(kp, ratio_direct, "C0.-", ms=4, lw=1.0, label="DIRECT (no interp)")
# also overlay the old interp-based ratio for contrast
ratio_interp = d["ratio"]
ax.semilogx(kp[np.isfinite(ratio_interp)], ratio_interp[np.isfinite(ratio_interp)],
            "C2.--", ms=3, lw=0.8, alpha=0.6, label="WITH log-log interp")
ax.axhline(1.0, color="k", ls="--", lw=0.8, alpha=0.6)
ax.set_ylim(0.99, 1.012)
ax.set_ylabel(r"$P_\text{mine}\,/\,P_\text{PRIYA}$")
ax.set_title("Direct mode-by-mode ratio vs log-log-interp ratio\n"
             "If the slope vanishes in the DIRECT comparison, the v2 tilt was an interp artefact.")
ax.legend(loc="upper center", fontsize=10)
ax.grid(True, which="both", alpha=0.2)

ax = axes[1]
ax.semilogx(kp, ratio_direct, "C0.-", ms=4, lw=1.0)
ax.axhline(1.0, color="k", ls="--", lw=0.8, alpha=0.6)
ax.set_ylim(0.998, 1.002)
ax.set_xlabel(r"$k$  [rad s km$^{-1}$, angular]")
ax.set_ylabel(r"$P_\text{mine}\,/\,P_\text{PRIYA}$ (zoom)")
ax.grid(True, which="both", alpha=0.2)

out = _FIGS / "2026-05-20-priya-p1d-ratio-direct.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"\nwrote {out}")
