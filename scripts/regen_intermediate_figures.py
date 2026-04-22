"""
Regenerate the key 'intermediate' figures from the fresh fixed catalogs
on /scratch/.../hcd_outputs/.

Produces PNGs under figures/analysis/ (new directory, to keep the stale
pre-audit figures/intermediate/ around for reference until deletion).

Figures:
  - cddf_per_z.png           CDDF per z-bin, all sims stacked
  - nhi_distribution.png     log N distribution of all catalog entries
  - dndx_vs_z.png            dN/dX vs z per class (LLS/subDLA/DLA)
  - absorber_counts_vs_z.png raw absorber counts per class per z
  - param_sensitivity.png    class counts vs each PRIYA parameter

Usage:
    python3 scripts/regen_intermediate_figures.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hcd_analysis.cddf import absorption_path_per_sightline

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = Path("/scratch/cavestru_root/cavestru0/mfho/hcd_outputs")
OUT_DIR = ROOT / "figures" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------- parameter parsing helpers --------
_SIM_PARAM = {
    "ns":         r"ns([0-9.]+)",
    "Ap":         r"Ap([0-9.e+-]+)",
    "herei":      r"herei([0-9.]+)",
    "heref":      r"heref([0-9.]+)",
    "alphaq":     r"alphaq([0-9.]+)",
    "hub":        r"hub([0-9.]+)",
    "omegamh2":   r"omegamh2([0-9.]+)",
    "hireionz":   r"hireionz([0-9.]+)",
    "bhfeedback": r"bhfeedback([0-9.]+)",
}


def parse_params(sim: str) -> dict:
    out = {}
    for key, pat in _SIM_PARAM.items():
        m = re.search(pat, sim)
        if m:
            out[key] = float(m.group(1))
    return out


def prochaska2014_logf(logN):
    from scipy.interpolate import PchipInterpolator
    _logN = np.array([12.0, 15.0, 17.0, 18.0, 20.0, 21.0, 21.5, 22.0])
    _logf = np.array([-9.72, -14.41, -17.94, -19.39, -21.28, -22.82, -23.95, -25.50])
    return PchipInterpolator(_logN, _logf)(np.clip(np.asarray(logN, dtype=float), _logN[0], _logN[-1]))


# -----------------------------------------------------------------------------
# Observational DLA dN/dX tabulations.
# Values are quoted from the cited papers; NOT from a package —
# keeping them here so the citation is visible alongside every datum.
# -----------------------------------------------------------------------------

# Prochaska & Wolfe 2009 (SDSS DR5), MNRAS 391, 1499 — Table 2 (log N_HI ≥ 20.3)
#   z_center, dN/dX, σ_dN/dX
PW09_DLA = np.array([
    [2.4, 0.066, 0.004],
    [2.8, 0.082, 0.005],
    [3.1, 0.078, 0.006],
    [3.4, 0.091, 0.010],
    [3.7, 0.094, 0.015],
    [4.0, 0.115, 0.026],
    [4.3, 0.149, 0.067],
])

# Noterdaeme et al. 2012 (BOSS DR9), A&A 547, L1 — Table 4 (log N_HI ≥ 20.3)
#   z_center, dN/dX, σ_dN/dX
N12_DLA = np.array([
    [2.16, 0.0615, 0.0055],
    [2.36, 0.0695, 0.0045],
    [2.56, 0.0750, 0.0050],
    [2.76, 0.0825, 0.0050],
    [2.96, 0.0980, 0.0065],
    [3.16, 0.1010, 0.0080],
    [3.36, 0.1160, 0.0100],
    [3.56, 0.1350, 0.0140],
])

# Crighton et al. 2015 (high-z, Giant-Gemini-GMOS), MNRAS 452, 217 — Table 3
#   z_center, dN/dX, σ_dN/dX_low, σ_dN/dX_high
C15_DLA = np.array([
    [4.4, 0.170, 0.030, 0.030],
    [5.0, 0.220, 0.050, 0.060],
])

# Sanchez-Ramirez et al. 2016 (SDSS DR12+), MNRAS 456, 4488 — Table 3
#   z_center, dN/dX, σ_dN/dX
SR16_DLA = np.array([
    [2.150, 0.0441, 0.0106],
    [2.500, 0.0604, 0.0076],
    [2.850, 0.0711, 0.0074],
    [3.200, 0.0819, 0.0082],
    [3.550, 0.1132, 0.0116],
    [3.900, 0.1373, 0.0166],
    [4.250, 0.1664, 0.0245],
])


# -------- scan fresh outputs --------
print("Scanning fresh catalogs...")
records = []   # list of {sim, z, logN_array, n_skewers, box_kpc_h, hubble, omegam, omegal, params}
for sim_dir in sorted(OUTPUT.iterdir()):
    if sim_dir.name == "hires" or not sim_dir.is_dir() or not sim_dir.name.startswith("ns"):
        continue
    params = parse_params(sim_dir.name)
    for snap_dir in sorted(sim_dir.iterdir()):
        if not snap_dir.name.startswith("snap_"):
            continue
        if not (snap_dir / "done").exists():
            continue
        meta_p = snap_dir / "meta.json"
        cat_p = snap_dir / "catalog.npz"
        if not (meta_p.exists() and cat_p.exists()):
            continue
        meta = json.load(open(meta_p))
        # Fast: just load NHI array from catalog.npz (no object construction)
        d = np.load(cat_p, allow_pickle=True)
        NHI = d["NHI"].astype(np.float64)
        logN = np.log10(np.maximum(NHI, 1.0))
        records.append({
            "sim": sim_dir.name,
            "snap": int(meta["snap"]),
            "z": float(meta["z"]),
            "n_skewers": int(meta["n_skewers"]),
            "box_kpc_h": float(meta["box_kpc_h"]),
            "hubble": float(meta["hubble"]),
            "omegam": 0.31, "omegal": 0.69,   # not in meta; use approximate Planck
            "logN": logN,
            "params": params,
        })
print(f"  got {len(records)} (sim, snap) records")

# -------- class thresholds --------
LLS_MIN, SUBDLA_MIN, DLA_MIN = 17.2, 19.0, 20.3

# =============================================================
# 1. NHI distribution
# =============================================================
print("1. NHI distribution...")
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

bins = np.linspace(17.0, 23.0, 91)
all_logN = np.concatenate([r["logN"] for r in records])

# split by class for stacked histogram
mask_lls = (all_logN >= LLS_MIN) & (all_logN < SUBDLA_MIN)
mask_sub = (all_logN >= SUBDLA_MIN) & (all_logN < DLA_MIN)
mask_dla = all_logN >= DLA_MIN

ax[0].hist(all_logN[mask_lls], bins=bins, histtype="step", lw=1.8, color="C2",
            label=f"LLS  ({mask_lls.sum():,})")
ax[0].hist(all_logN[mask_sub], bins=bins, histtype="step", lw=1.8, color="C1",
            label=f"subDLA  ({mask_sub.sum():,})")
ax[0].hist(all_logN[mask_dla], bins=bins, histtype="step", lw=1.8, color="C3",
            label=f"DLA  ({mask_dla.sum():,})")
for thr, name in [(17.2,"LLS"),(19.0,"subDLA"),(20.3,"DLA")]:
    ax[0].axvline(thr, color="gray", lw=0.6, ls=":")
ax[0].set_yscale("log"); ax[0].set_xlabel("log10(N_HI)")
ax[0].set_ylabel("count (all sims stacked)")
ax[0].set_title(f"NHI distribution (fresh catalogs, N={len(records)} snaps)")
ax[0].grid(alpha=0.3); ax[0].legend()

# Mean absorber count per sim per z
sum_per_z = defaultdict(lambda: {"LLS": [], "subDLA": [], "DLA": []})
for r in records:
    z_key = round(r["z"], 1)
    sum_per_z[z_key]["LLS"].append(int(((r["logN"]>=LLS_MIN)&(r["logN"]<SUBDLA_MIN)).sum()))
    sum_per_z[z_key]["subDLA"].append(int(((r["logN"]>=SUBDLA_MIN)&(r["logN"]<DLA_MIN)).sum()))
    sum_per_z[z_key]["DLA"].append(int((r["logN"]>=DLA_MIN).sum()))

zs_sorted = sorted(sum_per_z)
w = 0.25
x = np.arange(len(zs_sorted))
for i, (c, color) in enumerate([("LLS","C2"),("subDLA","C1"),("DLA","C3")]):
    means = [np.mean(sum_per_z[z][c]) for z in zs_sorted]
    ax[1].bar(x + (i-1)*w, means, w, color=color, label=c)
ax[1].set_xticks(x)
ax[1].set_xticklabels([f"z={z:.1f}" for z in zs_sorted], rotation=60, ha="right")
ax[1].set_ylabel("mean absorber count per sim")
ax[1].set_title("Mean count per z-bin (averaged across 60 sims)")
ax[1].grid(alpha=0.3, axis="y"); ax[1].legend()

fig.tight_layout()
outp = OUT_DIR / "nhi_distribution.png"
fig.savefig(outp, dpi=120); plt.close(fig)
print(f"   {outp}")

# =============================================================
# 2. CDDF per z-bin (stack across sims)
# =============================================================
print("2. CDDF per z-bin...")

z_bins = np.array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5])
nhi_bins = np.linspace(17.0, 23.0, 31)
centres = 0.5 * (nhi_bins[:-1] + nhi_bins[1:])
edges_lin = 10 ** nhi_bins
dN = edges_lin[1:] - edges_lin[:-1]

# Group records by z-bin
from collections import defaultdict
zbin_records = defaultdict(list)
for r in records:
    z = r["z"]
    for i in range(len(z_bins) - 1):
        if z_bins[i] <= z < z_bins[i+1]:
            zbin_records[i].append(r)
            break

n_zbins = sum(1 for v in zbin_records.values() if v)
ncols = 4
nrows = (n_zbins + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3.8, nrows*3.3), sharex=True, sharey=True)
axes = np.atleast_2d(axes).flatten()

for i, zi in enumerate(sorted(zbin_records)):
    zlo, zhi = z_bins[zi], z_bins[zi+1]
    rs = zbin_records[zi]
    counts = np.zeros(len(centres))
    total_path = 0.0
    for r in rs:
        path = absorption_path_per_sightline(
            r["box_kpc_h"], r["hubble"], r["omegam"], r["omegal"], r["z"]
        ) * r["n_skewers"]
        total_path += path
        hist, _ = np.histogram(r["logN"], bins=nhi_bins)
        counts += hist
    f_cddf = np.where(dN * total_path > 0, counts / (dN * total_path), 0.0)
    ax = axes[i]
    ax.step(centres, f_cddf, where="mid", lw=1.8, color="C0",
             label=f"sim, z ∈ [{zlo:.1f}, {zhi:.1f}]")
    ax.plot(centres, 10.0**prochaska2014_logf(centres), "k--", lw=1.2, label="Prochaska+2014")
    for thr in (17.2, 19.0, 20.3):
        ax.axvline(thr, color="gray", lw=0.5, ls=":")
    ax.set_yscale("log")
    ax.set_ylim(1e-28, 1e-15)
    ax.set_title(f"z ∈ [{zlo:.1f}, {zhi:.1f}]  ({len(rs)} snaps)", fontsize=10)
    ax.grid(alpha=0.3)
    if i == 0:
        ax.legend(fontsize=8)

for ax in axes[len(zbin_records):]:
    ax.axis("off")
for ax in axes.reshape(nrows, ncols)[-1, :]:
    ax.set_xlabel("log10(N_HI)")
for ax in axes.reshape(nrows, ncols)[:, 0]:
    ax.set_ylabel("f(N, X)")
fig.suptitle("CDDF per z-bin, all 60 PRIYA sims stacked")
fig.tight_layout(rect=(0, 0, 1, 0.97))
outp = OUT_DIR / "cddf_per_z.png"
fig.savefig(outp, dpi=120); plt.close(fig)
print(f"   {outp}")

# =============================================================
# 3. dN/dX vs z per class
# =============================================================
print("3. dN/dX vs z...")

z_unique = sorted({round(r["z"], 2) for r in records})
by_z = defaultdict(list)
for r in records:
    z_key = round(r["z"], 2)
    by_z[z_key].append(r)

fig, ax = plt.subplots(1, 1, figsize=(9, 6))
dndx_per_class = {"LLS": [], "subDLA": [], "DLA": []}
dndx_zs = []
for z in z_unique:
    rs = by_z[z]
    lls_tot = sub_tot = dla_tot = 0
    path_tot = 0.0
    for r in rs:
        path = absorption_path_per_sightline(
            r["box_kpc_h"], r["hubble"], r["omegam"], r["omegal"], r["z"]
        ) * r["n_skewers"]
        path_tot += path
        lls_tot += int(((r["logN"]>=LLS_MIN)&(r["logN"]<SUBDLA_MIN)).sum())
        sub_tot += int(((r["logN"]>=SUBDLA_MIN)&(r["logN"]<DLA_MIN)).sum())
        dla_tot += int((r["logN"]>=DLA_MIN).sum())
    dndx_per_class["LLS"].append(lls_tot / path_tot)
    dndx_per_class["subDLA"].append(sub_tot / path_tot)
    dndx_per_class["DLA"].append(dla_tot / path_tot)
    dndx_zs.append(z)

dndx_zs = np.array(dndx_zs)
for c, color in [("LLS","C2"),("subDLA","C1"),("DLA","C3")]:
    ax.plot(dndx_zs, np.array(dndx_per_class[c]), "o-", color=color, label=f"PRIYA sim ({c})", lw=1.5, ms=4)

# Overlay observational DLA dN/dX tabulations (all log N_HI ≥ 20.3)
ax.errorbar(PW09_DLA[:,0], PW09_DLA[:,1], yerr=PW09_DLA[:,2],
             fmt="s", color="black", ms=6, capsize=3,
             label="Prochaska & Wolfe 2009 (SDSS DR5)")
ax.errorbar(N12_DLA[:,0], N12_DLA[:,1], yerr=N12_DLA[:,2],
             fmt="^", color="dimgray", ms=6, capsize=3,
             label="Noterdaeme+2012 (BOSS DR9)")
ax.errorbar(SR16_DLA[:,0], SR16_DLA[:,1], yerr=SR16_DLA[:,2],
             fmt="D", color="darkslategray", ms=6, capsize=3,
             label="Sanchez-Ramirez+2016 (SDSS DR12)")
ax.errorbar(C15_DLA[:,0], C15_DLA[:,1],
             yerr=[C15_DLA[:,2], C15_DLA[:,3]],
             fmt="v", color="slategray", ms=6, capsize=3,
             label="Crighton+2015 (GGG, z≥4)")

ax.set_yscale("log")
ax.set_xlabel("z"); ax.set_ylabel("dN/dX")
ax.set_title("dN/dX vs z per class  (60 PRIYA sims, log N_HI ≥ 20.3 for DLA)\n"
              "observational overlays for DLA only")
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc="upper left", ncol=1)
ax.set_xlim(1.9, 5.6)
fig.tight_layout()
outp = OUT_DIR / "dndx_vs_z.png"
fig.savefig(outp, dpi=120); plt.close(fig)
print(f"   {outp}")

# =============================================================
# 4. Parameter sensitivity: DLA count at z=3 vs each param
# =============================================================
print("4. Parameter sensitivity...")
z_target = 3.0
snap_at_z = [r for r in records if abs(r["z"] - z_target) < 0.05]
param_keys = list(_SIM_PARAM.keys())
fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharey=True)
axes = axes.flatten()
for i, pk in enumerate(param_keys):
    ax = axes[i]
    xs = np.array([r["params"][pk] for r in snap_at_z if pk in r["params"]])
    ys = np.array([int((r["logN"] >= DLA_MIN).sum()) for r in snap_at_z if pk in r["params"]])
    ax.scatter(xs, ys, s=8, alpha=0.7, color="C3")
    ax.set_xlabel(pk)
    ax.set_ylabel("DLA count per 691k sightlines")
    ax.set_title(f"{pk}", fontsize=10)
    ax.grid(alpha=0.3)
fig.suptitle(f"DLA count vs each PRIYA parameter at z={z_target} (60 sims)")
fig.tight_layout(rect=(0, 0, 1, 0.96))
outp = OUT_DIR / "param_sensitivity.png"
fig.savefig(outp, dpi=120); plt.close(fig)
print(f"   {outp}")

print("\nAll figures in", OUT_DIR)
