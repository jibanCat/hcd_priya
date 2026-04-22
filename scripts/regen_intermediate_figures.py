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
# Observational DLA dN/dX tabulations, taken verbatim from the sbird/dla_data
# repository: https://github.com/sbird/dla_data  (files dndx.txt and
# ho21/dndx_all.txt).  Only the papers with *published dN/dX vs z tables*
# are overlaid here; papers reporting only CDDF or Ω_DLA (Peroux+05,
# Zafar+13, Crighton+15, Ho21 CDDF-only files) are NOT included.
# -----------------------------------------------------------------------------

# Prochaska & Wolfe 2009 (SDSS DR5), arXiv:0811.2003 — from dla_data/dndx.txt.
#   columns: z_low, z_high, l_DLA, err_l_DLA, rho_HI(10^8 M_sun/Mpc^3), err_rho_HI
#   we skip the overall z=[2.2, 5.5] bin (first row) and use the 6 narrower bins.
PW09_DLA = np.array([
    # z_lo  z_hi  l_DLA  err
    [2.2, 2.4, 0.048, 0.006],
    [2.4, 2.7, 0.055, 0.005],
    [2.7, 3.0, 0.067, 0.006],
    [3.0, 3.5, 0.084, 0.006],
    [3.5, 4.0, 0.075, 0.009],
    [4.0, 5.5, 0.106, 0.018],
])

# Noterdaeme+2012 (BOSS DR9), arXiv:1210.1213 — reproduced from dla_data/dla_data.py:
#     dndz = [0.2, 0.2, 0.25, 0.29, 0.36]
#     zz   = [2.15, 2.45, 2.75, 3.05, 3.35]
#     dzdx = [3690/11625, 4509/14841, 2867/9900, 1620/5834, 789/2883]
#     dN/dX = dndz * dzdx
#     xerr fixed at 0.15 (per-bin half-width); y-error bars not given in the code
N12_Z  = np.array([2.15, 2.45, 2.75, 3.05, 3.35])
N12_DNDZ = np.array([0.20, 0.20, 0.25, 0.29, 0.36])
N12_DZDX = np.array([3690/11625., 4509/14841., 2867/9900., 1620/5834., 789/2883.])
N12_DLA = np.column_stack([N12_Z, N12_DNDZ * N12_DZDX])

# Ho et al. 2021 (SDSS DR16, CNN DLA finder).  From dla_data/ho21/dndx_all.txt.
# Row 0: z centres (18 bins from 2.08 to 4.92).  Row 1: median dN/dX.
# Rows 2-3: 68% CI lower/upper.  Rows 4-5: 95% CI lower/upper.
HO21_DNDX_Z = np.array([
    2.083, 2.250, 2.417, 2.583, 2.750, 2.917, 3.083, 3.250, 3.417, 3.583,
    3.750, 3.917, 4.083, 4.250, 4.417, 4.583, 4.750, 4.917,
])
HO21_DNDX_MEDIAN = np.array([
    0.0337, 0.0430, 0.0462, 0.0494, 0.0622, 0.0664, 0.0706, 0.0748, 0.0763,
    0.0777, 0.0630, 0.0646, 0.0577, 0.0725, 0.1015, 0.0821, 0.1033, 0.0674,
])
HO21_DNDX_68_LO = np.array([
    0.0330, 0.0421, 0.0452, 0.0482, 0.0607, 0.0647, 0.0685, 0.0722, 0.0729,
    0.0736, 0.0584, 0.0583, 0.0503, 0.0637, 0.0888, 0.0684, 0.0812, 0.0506,
])
HO21_DNDX_68_HI = np.array([
    0.0345, 0.0438, 0.0472, 0.0506, 0.0637, 0.0682, 0.0729, 0.0777, 0.0800,
    0.0822, 0.0687, 0.0717, 0.0666, 0.0857, 0.1205, 0.1049, 0.1402, 0.1180,
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

# Use bin width 0.1 dex so class boundaries (17.2, 19.0, 20.3) divide
# evenly into bin edges with NO narrow-boundary bins (unequal bin widths
# at the boundaries had been causing a visual "drop" at class edges).
# np.linspace is preferred over np.arange because the latter accumulates
# floating-point error (e.g. 20.2999999… instead of 20.3).
bin_width = 0.1
n_bins_edges = round((23.0 - 17.0) / bin_width) + 1
bins = np.linspace(17.0, 23.0, n_bins_edges)
# Sanity check: class thresholds must coincide with bin edges to within
# floating-point tolerance so classification and plotting agree exactly.
for thr in (LLS_MIN, SUBDLA_MIN, DLA_MIN):
    assert np.any(np.abs(bins - thr) < 1e-10), \
        f"{thr} is not an exact bin edge"

all_logN = np.concatenate([r["logN"] for r in records])

# Single histogram of ALL absorbers (the ground truth, continuous across
# class boundaries — this demonstrates there is no physical gap).
counts_all, _ = np.histogram(all_logN, bins=bins)
centres = 0.5 * (bins[:-1] + bins[1:])
# Colour each bin by the class of its right edge
class_colour = np.where(
    bins[1:] <= SUBDLA_MIN, "C2",   # LLS
    np.where(bins[1:] <= DLA_MIN, "C1", "C3")   # subDLA / DLA
)
# Plot as a single step line (monotonic, no gaps)
ax[0].step(centres, counts_all, where="mid", color="black", lw=0.8, alpha=0.5,
           label="all absorbers")
# Overlay per-class coloured bars at each bin
for c_lbl, c_colour, lo, hi, n_cls in [
    ("LLS",    "C2", LLS_MIN, SUBDLA_MIN, ((all_logN>=LLS_MIN)&(all_logN<SUBDLA_MIN)).sum()),
    ("subDLA", "C1", SUBDLA_MIN, DLA_MIN, ((all_logN>=SUBDLA_MIN)&(all_logN<DLA_MIN)).sum()),
    ("DLA",    "C3", DLA_MIN, 23.0,       (all_logN>=DLA_MIN).sum()),
]:
    sel = (centres >= lo) & (centres < hi)
    ax[0].bar(centres[sel], counts_all[sel], width=np.diff(bins)[sel],
              color=c_colour, alpha=0.6, edgecolor="none",
              label=f"{c_lbl}  ({n_cls:,})")
for thr, name in [(17.2,"LLS boundary"),(19.0,"subDLA"),(20.3,"DLA")]:
    ax[0].axvline(thr, color="gray", lw=0.8, ls=":")
ax[0].set_yscale("log")
ax[0].set_xlabel("log10(N_HI)")
ax[0].set_ylabel("count (all sims stacked)")
ax[0].set_title(f"NHI distribution (N={len(records)} snaps)\n"
                f"single histogram, coloured by class — no physical gap")
ax[0].grid(alpha=0.3)
ax[0].legend(loc="lower left", fontsize=8)

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

# Overlay observational DLA dN/dX tabulations (all log N_HI ≥ 20.3 where
# specified; taken from sbird/dla_data verbatim).

# PW09: bin-edge format — show as horizontal error bars in z + y error
z_mid_pw09 = 0.5*(PW09_DLA[:,0] + PW09_DLA[:,1])
z_err_pw09 = 0.5*(PW09_DLA[:,1] - PW09_DLA[:,0])
ax.errorbar(z_mid_pw09, PW09_DLA[:,2],
             xerr=z_err_pw09, yerr=PW09_DLA[:,3],
             fmt="s", color="black", ms=6, capsize=3,
             label="Prochaska & Wolfe 2009 (SDSS DR5)")

# N12: no y-errors provided; use half-bin x-error only (0.15)
ax.errorbar(N12_DLA[:,0], N12_DLA[:,1],
             xerr=0.15, fmt="^", color="dimgray", ms=6, capsize=3,
             label="Noterdaeme+2012 (BOSS DR9)")

# Ho+2021: median with asymmetric 68% CI error bars
ho21_yerr_lo = HO21_DNDX_MEDIAN - HO21_DNDX_68_LO
ho21_yerr_hi = HO21_DNDX_68_HI  - HO21_DNDX_MEDIAN
ax.errorbar(HO21_DNDX_Z, HO21_DNDX_MEDIAN,
             yerr=[ho21_yerr_lo, ho21_yerr_hi],
             fmt="D", color="darkslategray", ms=5, capsize=3, alpha=0.85,
             label="Ho+2021 (SDSS DR16)")

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
# 4. Parameter sensitivity: per-class counts at z=3 vs each param
# =============================================================
print("4. Parameter sensitivity (LLS / subDLA / DLA)...")
z_target = 3.0
snap_at_z = [r for r in records if abs(r["z"] - z_target) < 0.05]
param_keys = list(_SIM_PARAM.keys())

for cls_name, cls_color, cls_mask_fn, fname in [
    ("LLS",    "C2", lambda lN: (lN >= LLS_MIN) & (lN < SUBDLA_MIN),
        "param_sensitivity_LLS.png"),
    ("subDLA", "C1", lambda lN: (lN >= SUBDLA_MIN) & (lN < DLA_MIN),
        "param_sensitivity_subDLA.png"),
    ("DLA",    "C3", lambda lN: lN >= DLA_MIN,
        "param_sensitivity_DLA.png"),
]:
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharey=True)
    axes = axes.flatten()
    for i, pk in enumerate(param_keys):
        ax = axes[i]
        xs = np.array([r["params"][pk] for r in snap_at_z if pk in r["params"]])
        ys = np.array([int(cls_mask_fn(r["logN"]).sum()) for r in snap_at_z if pk in r["params"]])
        ax.scatter(xs, ys, s=10, alpha=0.7, color=cls_color)
        # Simple Spearman rank correlation
        if len(xs) > 3:
            from scipy.stats import spearmanr
            rho, p = spearmanr(xs, ys)
            ax.text(0.02, 0.97, f"ρ={rho:+.2f}\np={p:.2g}",
                     transform=ax.transAxes, fontsize=7, va="top",
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))
        ax.set_xlabel(pk)
        ax.set_ylabel(f"{cls_name} count / 691k sightlines")
        ax.set_title(f"{pk}", fontsize=10)
        ax.grid(alpha=0.3)
    fig.suptitle(f"{cls_name} count vs each PRIYA parameter at z={z_target}  (60 sims)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    outp = OUT_DIR / fname
    fig.savefig(outp, dpi=120); plt.close(fig)
    print(f"   {outp}")

print("\nAll figures in", OUT_DIR)
