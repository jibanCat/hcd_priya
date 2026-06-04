"""Measure cross-class absorber clustering to size the dN/dX -> w_c mixing map M.

For each (sim, z): compare the directly-measured sightline-by-max fractions w_c
to the INDEPENDENT-Poisson prediction, and report cross-class clustering factors
xi = P(A & B on same sightline) / [P(A) P(B)]  (>1 => positive clustering, the
off-diagonal that makes M non-diagonal) plus the "hidden" fractions (how much of
a class's dN/dX sits behind a higher-class absorber).

Catalog-only (no fake_spectra). z-resolved over one sim's full snapshot ladder.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "scripts")
import build_emulator_cache as bec
from hcd_analysis.catalog import AbsorberCatalog

SIM = "ns0.803Ap2.2e-09herei4.05heref2.67alphaq2.21hub0.735omegamh20.141hireionz7.17bhfeedback0.056"
ROOT = Path("/scratch/cavestru_root/cavestru0/mfho/hcd_outputs") / SIM
# class code: 1=LLS [17.2,19.0), 2=subDLA [19.0,20.3), 3=DLA [20.3,inf)
EDGES = np.array([17.2, 19.0, 20.3])

snaps = []
for sd in sorted(ROOT.glob("snap_*")):
    if (sd / "catalog.npz").exists() and (sd / "done").exists():
        snaps.append(sd)

print(f"{SIM[:24]}  {len(snaps)} snaps\n")
hdr = (f"{'z':>5s} {'N_sl':>8s} | {'wLLS_m':>7s} {'wLLS_P':>7s} {'wSub_m':>7s} "
       f"{'wSub_P':>7s} {'wDLA_m':>7s} {'wDLA_P':>7s} | {'xi_S.D':>7s} {'xi_L.D':>7s} "
       f"{'xi_L.S':>7s} | {'hid_S':>6s} {'hid_L':>6s}")
print(hdr)
print("-" * len(hdr))

for sd in snaps:
    meta = bec.read_meta(sd)
    cddf = bec.read_cddf(sd)
    z = float(meta["z"])
    N_sl = int(cddf["n_sightlines"])
    cat = AbsorberCatalog.load_npz(sd / "catalog.npz")

    sk = np.fromiter((ab.skewer_idx for ab in cat.absorbers), dtype=np.int64)
    nhi = np.fromiter((ab.log_NHI for ab in cat.absorbers), dtype=np.float64)
    keep = (nhi >= EDGES[0]) & (sk < N_sl)
    sk, nhi = sk[keep], nhi[keep]
    cls = np.digitize(nhi, EDGES)        # 1=LLS, 2=subDLA, 3=DLA

    n_abs = {c: int((cls == c).sum()) for c in (1, 2, 3)}
    maxcls = np.zeros(N_sl, dtype=np.int64)
    np.maximum.at(maxcls, sk, cls)
    w_meas = {c: float((maxcls == c).mean()) for c in (1, 2, 3)}

    mu = {c: n_abs[c] / N_sl for c in (1, 2, 3)}
    wP = {}
    wP[3] = 1 - np.exp(-mu[3])
    wP[2] = (1 - np.exp(-mu[2])) * np.exp(-mu[3])
    wP[1] = (1 - np.exp(-mu[1])) * np.exp(-(mu[2] + mu[3]))

    # presence indicators per sightline
    hasc = {c: np.zeros(N_sl, dtype=bool) for c in (1, 2, 3)}
    for c in (1, 2, 3):
        hasc[c][sk[cls == c]] = True
    P = {c: float(hasc[c].mean()) for c in (1, 2, 3)}
    def xi(a, b):
        joint = float((hasc[a] & hasc[b]).mean())
        return joint / (P[a] * P[b]) if P[a] * P[b] > 0 else np.nan
    xi_SD, xi_LD, xi_LS = xi(2, 3), xi(1, 3), xi(1, 2)

    # hidden: fraction of a class's absorbers sitting behind a HIGHER max class
    hid_S = float((maxcls[sk[cls == 2]] > 2).mean()) if n_abs[2] else np.nan
    hid_L = float((maxcls[sk[cls == 1]] > 1).mean()) if n_abs[1] else np.nan

    print(f"{z:5.2f} {N_sl:8d} | {w_meas[1]:7.4f} {wP[1]:7.4f} {w_meas[2]:7.4f} "
          f"{wP[2]:7.4f} {w_meas[3]:7.4f} {wP[3]:7.4f} | {xi_SD:7.2f} {xi_LD:7.2f} "
          f"{xi_LS:7.2f} | {hid_S:6.3f} {hid_L:6.3f}")

print("\nLegend: _m=measured, _P=independent-Poisson. xi_S.D=P(subDLA&DLA)/indep, "
      "xi_L.D=LLS&DLA, xi_L.S=LLS&subDLA (>1 = clustered). hid_S=frac of subDLA "
      "absorbers on a max>subDLA (=DLA) sightline; hid_L=frac of LLS absorbers on "
      "a max>LLS sightline. Large xi / hid => M must be non-diagonal.")
