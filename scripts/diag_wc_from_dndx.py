"""Verify per-class sightline fractions w_c against a Poisson prediction from
dN/dX, to size the informative consistency prior between Head A's directly
emulated w_c and the dN/dX-derived w_c.

Pure catalog/CDDF bookkeeping (no fake_spectra) on the FULL sightline set.
Uses the Tier-C boundaries (17.2 / 19.0 / 20.3) for BOTH paths so the only
difference is absorber multiplicity/clustering along sightlines.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "scripts")
import build_emulator_cache as bec
from hcd_analysis.catalog import AbsorberCatalog

SIM = "ns0.803Ap2.2e-09herei4.05heref2.67alphaq2.21hub0.735omegamh20.141hireionz7.17bhfeedback0.056"
SNAP = 17
EDGES = {"LLS": (17.2, 19.0), "subDLA": (19.0, 20.3), "DLA": (20.3, np.inf)}
ORDER = ["DLA", "subDLA", "LLS"]   # high -> low for the telescoping product

snap_dir = Path("/scratch/cavestru_root/cavestru0/mfho/hcd_outputs") / SIM / f"snap_{SNAP:03d}"
cddf = bec.read_cddf(snap_dir)
meta = bec.read_meta(snap_dir)
catalog = AbsorberCatalog.load_npz(snap_dir / "catalog.npz")

N_sl = int(cddf["n_sightlines"])
X_tot = float(cddf["total_path"])
print(f"{SIM[:24]} snap {SNAP}  z={meta['z']:.3f}  N_sightlines={N_sl}  total_dX={X_tot:.1f}\n")

def cls_of(nhi):
    for name, (lo, hi) in EDGES.items():
        if lo <= nhi < hi:
            return name
    return None

# absorber counts (dN/dX numerator) and per-sightline max class (w_c numerator)
n_abs = {"LLS": 0, "subDLA": 0, "DLA": 0}
maxnhi = np.full(N_sl, -np.inf)
for ab in catalog.absorbers:
    c = cls_of(ab.log_NHI)
    if c is not None:
        n_abs[c] += 1
    if ab.log_NHI >= 17.2 and ab.skewer_idx < N_sl and ab.log_NHI > maxnhi[ab.skewer_idx]:
        maxnhi[ab.skewer_idx] = ab.log_NHI

n_sl = {"LLS": 0, "subDLA": 0, "DLA": 0}
for v in maxnhi[np.isfinite(maxnhi)]:
    n_sl[cls_of(v)] += 1
n_sl_clean = N_sl - sum(n_sl.values())

# direct fractions
w_direct = {"clean": n_sl_clean / N_sl, **{c: n_sl[c] / N_sl for c in n_sl}}

# dN/dX and Poisson prediction
dndx = {c: n_abs[c] / X_tot for c in n_abs}
mu = {c: n_abs[c] / N_sl for c in n_abs}     # = dndx * X_tot / N_sl
w_pois = {}
w_pois["DLA"] = 1 - np.exp(-mu["DLA"])
w_pois["subDLA"] = (1 - np.exp(-mu["subDLA"])) * np.exp(-mu["DLA"])
w_pois["LLS"] = (1 - np.exp(-mu["LLS"])) * np.exp(-(mu["subDLA"] + mu["DLA"]))
w_pois["clean"] = np.exp(-(mu["LLS"] + mu["subDLA"] + mu["DLA"]))

print(f"{'class':8s} {'dN/dX':>10s} {'mu (abs/sl)':>12s} {'n_abs':>9s} {'n_sl(max)':>10s} "
      f"{'w_direct':>10s} {'w_poisson':>10s} {'reldiff':>9s}")
for c in ["clean", "LLS", "subDLA", "DLA"]:
    dd = dndx[c] if c in dndx else 0.0
    mm = mu[c] if c in mu else 0.0
    na = n_abs[c] if c in n_abs else 0
    ns = n_sl[c] if c in n_sl else n_sl_clean
    wd, wp = w_direct[c], w_pois[c]
    rd = (wp - wd) / wd if wd > 0 else np.nan
    print(f"{c:8s} {dd:10.4f} {mm:12.5f} {na:9d} {ns:10d} {wd:10.5f} {wp:10.5f} {rd:+9.2%}")

print(f"\nsum w_direct = {sum(w_direct.values()):.6f}   sum w_poisson = {sum(w_pois.values()):.6f}")
# multiplicity diagnostic: n_abs / n_sl per class (>1 => multiple absorbers per sightline)
print("\nmultiplicity n_abs / n_sl(max-class) per class:")
for c in ["LLS", "subDLA", "DLA"]:
    print(f"  {c:8s} {n_abs[c]/max(n_sl[c],1):.3f}")
