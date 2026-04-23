"""
Plot HiRes/LF convergence T(k) = P_hires(k) / P_LF(k) at each matched sim
and z, for both the unmasked "all" and the PRIYA-masked "no_DLA_priya"
P1D variants.  Reads convergence_ratios.npz produced by the convergence
SLURM job.

After the z-tolerance match fix in compute_convergence_ratios (default
z_tol = 0.05), each stored z-label is the HR z and the paired LF snap
is the one whose z is closest to that value.  T(k) is therefore a
genuine resolution-convergence ratio at a fixed epoch.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
HR_ROOT = Path("/scratch/cavestru_root/cavestru0/mfho/hcd_outputs/hires")
OUT = ROOT / "figures" / "analysis" / "03_templates_and_p1d"
DATA = ROOT / "figures" / "analysis" / "data"
OUT.mkdir(parents=True, exist_ok=True)

ratios_files = sorted(HR_ROOT.glob("*/convergence_ratios.npz"))
print(f"Found {len(ratios_files)} convergence_ratios files")

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
ax_all, ax_priya = axes

# All z labels available in the first file
d0 = np.load(ratios_files[0], allow_pickle=True)
z_labels = sorted({k.split("__")[0] for k in d0.files})
z_values = [float(lbl.replace("z", "").replace("p", ".")) for lbl in z_labels]
cmap = plt.cm.plasma(np.linspace(0, 1, len(z_values)))

# Determine actual k range present in the data
_d0_k = d0[f"{z_labels[0]}__all__k"]
k_cyc_min = float(_d0_k.min()); k_cyc_max = float(_d0_k.max())
k_ang_data_min = 2.0 * np.pi * k_cyc_min
k_ang_data_max = 2.0 * np.pi * k_cyc_max
# Plot range: clamp to the requested emulator range AND to the actual data
x_lo = max(0.0009, k_ang_data_min)
x_hi = min(0.20,   k_ang_data_max)

for rf in ratios_files:
    d = np.load(rf, allow_pickle=True)
    for z_idx, (zlbl, zval) in enumerate(zip(z_labels, z_values)):
        color = cmap[z_idx]
        key_k = f"{zlbl}__all__k"
        if key_k not in d.files: continue
        k = d[key_k]
        k_ang = 2.0 * np.pi * k
        sel = (k_ang >= x_lo) & (k_ang <= x_hi)
        ax_all.plot(k_ang[sel], d[f"{zlbl}__all__T_k"][sel],
                     lw=1.0, alpha=0.7, color=color,
                     label=f"z={zval:.1f}" if rf is ratios_files[0] else None)
        ax_priya.plot(k_ang[sel], d[f"{zlbl}__no_DLA_priya__T_k"][sel],
                       lw=1.0, alpha=0.7, color=color,
                       label=f"z={zval:.1f}" if rf is ratios_files[0] else None)

for ax, title in zip(
    [ax_all, ax_priya],
    ["T(k) = P1D_hires / P1D_LF,  unmasked 'all'",
     "T(k) = P1D_hires / P1D_LF,  PRIYA DLA mask"]
):
    ax.axhline(1.0, color="gray", ls="--", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("k [rad·s/km]  (PRIYA angular)")
    ax.set_ylabel("T(k)")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0.85, 1.40)
ax_all.legend(fontsize=7, loc="upper left", ncol=2)

fig.suptitle(
    f"HiRes / LF convergence ratio  ({len(ratios_files)} paired sims, "
    "z-matched within |Δz|≤0.05)"
)
fig.tight_layout()
out = OUT / "convergence_Tk.png"
fig.savefig(out, dpi=120); plt.close(fig)
print(f"wrote {out}")
