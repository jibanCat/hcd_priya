"""Is 10 alpha enough, or is 20 needed? Use the 20-alpha timing cache for one
pair. The 10 PRIYA-exact alpha are at EVEN indices; the 10 refinement alpha at
ODD indices are the exact MIDPOINTS between them (idx 19 is an extrapolation
above the top, excluded). Reconstruct the held-out midpoints from the PRIYA-10
by interpolation in tau0 (the emulator's input coordinate) and report the
fractional error. Linear interp at a midpoint is the WORST case for a smooth
curve -> conservative bound; a smooth NN trained on the 10 will do better.
"""
import h5py
import numpy as np
import sys
sys.path.insert(0, "scripts")
from hcd_analysis.priya_p1d import merge_fine_to_classes

CACHE = "/scratch/cavestru_root/cavestru0/mfho/timing_one_pair_v33.h5"
RANGES = [(0, 1), (1, 8), (8, 13), (13, 15)]
NAMES = ["clean", "LLS", "subDLA", "DLA"]
KBAND = slice(5, 120)

with h5py.File(CACHE, "r") as f:
    aidx = f["alpha_idx"][...]
    tF = f["target_F"][...]
    kf = f["kfkms"][0]
    P_tp = f["P_tier_p"][...]                 # (20, nk)
    P_tc = f["P_tier_c"][...]                  # (20, 15, nk)
    counts = f["tier_c_counts"][0]             # alpha-invariant

tau0 = -np.log(tF)                             # emulator input coordinate
order = np.argsort(aidx)                       # ensure idx order
aidx, tau0, P_tp = aidx[order], tau0[order], P_tp[order]
P_tc = P_tc[order]

even = aidx % 2 == 0                           # PRIYA-exact 10
# held-out interior midpoints = odd indices strictly inside the even range
odd_interior = (aidx % 2 == 1) & (tau0 < tau0[even].max()) & (tau0 > tau0[even].min())
print(f"alpha grid: {len(aidx)} pts; PRIYA-exact (even idx)={even.sum()}; "
      f"held-out interior midpoints={odd_interior.sum()} (idx19 extrapolation excluded)")
print(f"tau0 range [{tau0.min():.3f}, {tau0.max():.3f}], PRIYA-10 spacing "
      f"~{np.diff(np.sort(tau0[even])).mean():.4f}\n")

# build per-quantity P(tau0) curves: Total + 4 collapsed classes
series = {"Total": P_tp}
merged_per_alpha = [merge_fine_to_classes(P_tc[i], counts, RANGES) for i in range(len(aidx))]
for ci, name in enumerate(NAMES):
    series[name] = np.stack([merged_per_alpha[i][ci] for i in range(len(aidx))])  # (20, nk)

xt = tau0[even]; xs = np.argsort(xt)
xh = tau0[odd_interior]
print(f"{'quantity':8s} {'med |err|':>10s} {'95th':>9s} {'max':>9s}   (linear interp from PRIYA-10 at midpoints)")
for name, P in series.items():
    Ptrain = P[even][xs]                       # (10, nk) sorted by tau0
    Ptrue = P[odd_interior]                    # (Nmid, nk)
    errs = []
    for j, x in enumerate(xh):
        pred = np.array([np.interp(x, xt[xs], Ptrain[:, k]) for k in range(P.shape[1])])
        with np.errstate(invalid="ignore", divide="ignore"):
            e = np.abs(pred[KBAND] - Ptrue[j][KBAND]) / np.abs(Ptrue[j][KBAND])
        errs.append(e[np.isfinite(e)])
    e = np.concatenate(errs)
    print(f"{name:8s} {np.median(e):10.2%} {np.percentile(e,95):9.2%} {e.max():9.2%}")

print("\nReading: fractional error of recovering the in-between alpha from the "
      "PRIYA-10 (linear = worst case for a smooth curve). If << ~1% (emulator "
      "target) and << cosmic variance, 10 alpha is enough; a smooth NN beats this.")
