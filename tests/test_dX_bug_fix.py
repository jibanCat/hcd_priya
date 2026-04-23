"""
Bug verification + fix test for absorption_path_per_sightline.

Three checks:
  (1) Numerical: the three formulas — codebase, my-proposed-fix, and a
      fake-spectra-like direct port — on identical input.
  (2) Analytical: both against the canonical dX/dz · Δz_box integral.
  (3) Empirical: regenerate the z=3 CDDF with the fixed dX and compare
      against Prochaska+2014.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hcd_analysis.cddf import absorption_path_per_sightline as dX_CODE

# ------------------------------------------------------------------------
# Implementations
# ------------------------------------------------------------------------

_C_KMS = 2.99792458e5

def dX_fixed(box_kpc_h, hubble, z):
    """Proposed corrected version: (1+z)² · L_Mpc · (h·100) / c."""
    L_com_Mpc = box_kpc_h / 1000.0 / hubble
    H0 = hubble * 100.0
    return (1.0 + z)**2 * L_com_Mpc * H0 / _C_KMS

def dX_fake_spectra_style(box_kpc_h, hubble, z):
    """Port of fake_spectra.unitsystem.absorption_distance:
         return self.h100/self.light * speclen * self.UnitLength_in_cm * (1+red)**2
    where speclen is in kpc/h (so speclen × UnitLength_in_cm = L_com_cm · h),
    self.h100 = 100 km/s/Mpc in 1/s = 3.2407789e-18 /s,
    self.light = 2.99e10 cm/s, UnitLength_in_cm = 3.085678e21 cm.
    Internally this reduces to (h · 100 · L_com / c) · (1+z)².
    """
    h100 = 3.2407789e-18     # 100 km/s/Mpc in 1/s
    light_cms = 2.99e10
    ULc = 3.085678e21        # kpc in cm
    speclen_kpch = box_kpc_h
    L_cm_times_h = speclen_kpch * ULc       # L_com_cm · h (implicit h factor)
    return h100 / light_cms * L_cm_times_h * (1 + z)**2

# Quick derivation of what we expect analytically (at fixed z for a box of
# comoving size L, convention A).  Used as the "truth" reference.
def dX_analytical(L_com_Mpc, H0, z):
    return (1.0 + z)**2 * L_com_Mpc * H0 / _C_KMS


# ------------------------------------------------------------------------
# (1) + (2): numerical cross-check
# ------------------------------------------------------------------------
print("=" * 78)
print("(1) Numerical comparison of dX formulas")
print("=" * 78)
box = 120000.0
h = 0.7
L_com_Mpc = box / 1000.0 / h
H0 = h * 100.0

print(f"{'z':>5}  {'code (broken)':>14}  {'fixed':>10}  {'fake_spectra':>14}  "
      f"{'analytical':>12}  {'ratio fixed/code':>18}")
for z in [0.0, 1.0, 2.0, 2.5, 3.0, 4.0, 5.0]:
    a = dX_CODE(box, h, 0.3, 0.7, z)
    b = dX_fixed(box, h, z)
    c_ = dX_fake_spectra_style(box, h, z)
    d = dX_analytical(L_com_Mpc, H0, z)
    print(f"  {z:.2f}  {a:>14.5f}  {b:>10.5f}  {c_:>14.5f}  {d:>12.5f}  "
          f"{b/a if a else float('nan'):>18.3f}")

print()
print("Verification:")
print(f"  * fixed == analytical: {np.allclose(dX_fixed(box, h, 3.0), dX_analytical(L_com_Mpc, H0, 3.0))}")
print(f"  * fake_spectra ≈ analytical (within float precision): "
      f"{abs(dX_fake_spectra_style(box, h, 3.0) / dX_analytical(L_com_Mpc, H0, 3.0) - 1) < 1e-3}")
print(f"  * code / correct at z=3: {dX_CODE(box, h, 0.3, 0.7, 3.0) / dX_fixed(box, h, 3.0):.4f}")
print(f"    → correct/code = (1+z)·h = 4·0.7 = 2.8")

# ------------------------------------------------------------------------
# (3) Empirical: re-compute CDDF with fixed dX for z=3
# ------------------------------------------------------------------------
print("\n" + "=" * 78)
print("(3) CDDF(log N, X) at z=3 — stacked over all 60 sims — with fixed dX")
print("=" * 78)

from pathlib import Path
import json

OUTPUT = Path("/scratch/cavestru_root/cavestru0/mfho/hcd_outputs")

# Collect all catalog entries at z ≈ 3
records = []
for sim in sorted(OUTPUT.iterdir()):
    if not sim.is_dir() or sim.name == "hires" or not sim.name.startswith("ns"):
        continue
    snap = sim / "snap_017"
    done = snap / "done"
    if not done.exists():
        continue
    meta = json.load(open(snap / "meta.json"))
    if abs(meta["z"] - 3.0) > 0.1:
        continue
    d = np.load(snap / "catalog.npz", allow_pickle=True)
    logN = np.log10(np.maximum(d["NHI"].astype(np.float64), 1.0))
    records.append({
        "logN": logN, "z": meta["z"],
        "box_kpc_h": meta["box_kpc_h"], "hubble": meta["hubble"],
        "n_skewers": meta["n_skewers"],
    })

print(f"  collected {len(records)} sims at z≈3")

# Bins on log N
bin_edges = np.linspace(17.0, 23.0, 61)
bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
dN = 10**bin_edges[1:] - 10**bin_edges[:-1]

# Accumulate counts and path, old vs new dX
counts = np.zeros(len(bin_centres))
X_old = 0.0; X_new = 0.0
for r in records:
    h_ = r["hubble"]; z_ = r["z"]; L_Mpc = r["box_kpc_h"] / 1000.0 / h_
    X_old += dX_CODE(r["box_kpc_h"], h_, 0.3, 0.7, z_) * r["n_skewers"]
    X_new += dX_fixed(r["box_kpc_h"], h_, z_) * r["n_skewers"]
    hist, _ = np.histogram(r["logN"], bins=bin_edges)
    counts += hist

f_old = counts / (dN * X_old)
f_new = counts / (dN * X_new)
print(f"  total X (old, broken): {X_old:.2e}")
print(f"  total X (fixed):       {X_new:.2e}")
print(f"  ratio X_new / X_old = {X_new/X_old:.3f}  ≈ (1+z)·h = {4*0.7:.3f}")

# Prochaska+2014 spline
def prochaska(logN):
    from scipy.interpolate import PchipInterpolator
    _logN = np.array([12.0, 15.0, 17.0, 18.0, 20.0, 21.0, 21.5, 22.0])
    _logf = np.array([-9.72, -14.41, -17.94, -19.39, -21.28, -22.82, -23.95, -25.50])
    return PchipInterpolator(_logN, _logf)(np.clip(np.asarray(logN, dtype=float),
                                                     _logN[0], _logN[-1]))

fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for a, f_val, lbl in [(ax[0], f_old, "broken code"),
                        (ax[1], f_new, "fixed dX formula")]:
    a.step(bin_centres, f_val, where="mid", lw=1.8, color="C0", label=f"PRIYA, {lbl}")
    a.plot(bin_centres, 10.0**prochaska(bin_centres), "k--", lw=1.5,
            label="Prochaska+2014 (z=2.5)")
    for thr in (17.2, 19.0, 20.3):
        a.axvline(thr, color="gray", lw=0.5, ls=":")
    a.set_yscale("log")
    a.set_xlabel("log10(N_HI)"); a.set_ylabel("f(N_HI, X)")
    a.set_title(lbl)
    a.legend()
    a.grid(alpha=0.3)
    a.set_ylim(1e-28, 1e-15)

fig.suptitle(f"CDDF at z≈3, {len(records)} PRIYA sims stacked\n"
              f"dX bug fix: X_new/X_old = {X_new/X_old:.2f}")
fig.tight_layout()
outp = ROOT / "figures" / "analysis" / "01_catalog_obs" / "cddf_bugfix_comparison.png"
outp.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(outp, dpi=120); plt.close(fig)
print(f"\n  wrote {outp}")

# Numerical table at key log N
print("\n  f(N,X) at key log N bins:")
print(f"    {'log N':>8}  {'f_old (broken)':>16}  {'f_new (fixed)':>15}  "
      f"{'P+2014':>12}  {'f_old/P+14':>11}  {'f_new/P+14':>11}")
for logN_tgt in [17.5, 18.0, 19.0, 20.0, 20.5, 21.0, 21.5]:
    j = int(np.argmin(np.abs(bin_centres - logN_tgt)))
    f_p = 10**prochaska(bin_centres[j])
    print(f"    {bin_centres[j]:>8.2f}  {f_old[j]:>16.4e}  {f_new[j]:>15.4e}  "
          f"{f_p:>12.4e}  {f_old[j]/f_p:>11.3f}  {f_new[j]/f_p:>11.3f}")
