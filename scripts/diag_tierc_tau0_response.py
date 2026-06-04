"""Demonstrate that the UNFILTERED Tier-C per-class P1D carries a real tau0
(mean-flux / alpha) response, and that the response is class-ordered
(clean strongest -> DLA weakest, the saturated-core self-shielding physics).

One LF sim, snap nearest z=3, 3 alpha values, a subset of skewers for speed.
Not a bit-identity run -- a response demonstration. Run in emu-3.9 env.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "scripts")
import build_emulator_cache_tau0 as bt0
from hcd_analysis.priya_p1d import merge_fine_to_classes

SIM_PREFIX = "ns0.803"
N_SKEWERS = 40000
N_K = 172
ALPHAS = [0.70, 1.00, 1.35]          # low / mid / high, spanning PRIYA's range
# 15 fine bins -> 4 physical classes
RANGES = [(0, 1), (1, 8), (8, 13), (13, 15)]
NAMES = ["clean", "LLS", "subDLA", "DLA"]

HCD_ROOT = Path("/scratch/cavestru_root/cavestru0/mfho/hcd_outputs")
EMU_ROOT = Path("/nfs/turbo/umor-yueyingn/mfho/emu_full")

pairs = bt0.discover_tau0_pairs(HCD_ROOT, EMU_ROOT, fidelity="lf")
cand = [p for p in pairs if p[0].startswith(SIM_PREFIX)]
# pick snap whose grid-z is closest to 3.0
import json
def zmeta(sd): return float(json.load(open(sd / "meta.json"))["z"])
cand.sort(key=lambda p: abs(zmeta(p[2]) - 3.0))
sim, snap, snap_dir, raw = cand[0]
print(f"sim={sim[:24]} snap={snap} z_meta={zmeta(snap_dir):.3f} raw={raw.name}")
print(f"alphas={ALPHAS}  n_skewers={N_SKEWERS}  n_k={N_K}\n")

rows, _ = bt0.build_tau0_rows(sim, snap, snap_dir, raw, ALPHAS, N_K,
                              n_skewers=N_SKEWERS)

# counts (alpha-invariant)
counts = rows[0]["tier_c_counts"]
cls_counts = [int(counts[lo:hi].sum()) for lo, hi in RANGES]
print("class counts:", dict(zip(NAMES, cls_counts)), "\n")

# collapse 15 -> 4 per alpha
P_cls = {n: [] for n in NAMES}   # name -> list over alpha of P1D[n_k]
tF, sc = [], []
for r in rows:
    tF.append(r["target_F"]); sc.append(r["scale"])
    merged = merge_fine_to_classes(r["P_tier_c"], r["tier_c_counts"], RANGES)
    for n, P in zip(NAMES, merged):
        P_cls[n].append(np.asarray(P))

print("alpha   target_F   scale")
for a, t, s in zip(ALPHAS, tF, sc):
    print(f"{a:.3f}   {t:.4f}    {s:.4f}")
print()

# response per class: fractional change low->high alpha, averaged over a
# mid-k band (avoid the noisy highest-k tail and k=0)
klo, khi = 5, 120
print(f"{'class':8s} {'P(lo)@k50':>12s} {'P(mid)@k50':>12s} {'P(hi)@k50':>12s} "
      f"{'|hi-lo|/mid (mean over k)':>26s}")
for n in NAMES:
    Plo, Pmid, Phi = P_cls[n][0], P_cls[n][1], P_cls[n][2]
    band = slice(klo, khi)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.abs(Phi[band] - Plo[band]) / np.abs(Pmid[band])
    frac = frac[np.isfinite(frac)]
    resp = float(np.nanmean(frac)) if frac.size else float("nan")
    k50 = 50
    print(f"{n:8s} {Plo[k50]:12.4e} {Pmid[k50]:12.4e} {Phi[k50]:12.4e} {resp:26.4f}")

print("\nInterpretation: a nonzero response per class confirms unfiltered Tier-C "
      "carries tau0 variation; clean >= LLS >= subDLA >= DLA in response "
      "amplitude is the expected self-shielding ordering.")

# ---- P vs k figure ----------------------------------------------------------
kf = np.asarray(rows[0]["kfkms"])                      # (n_k,) angular s/km
P_tier_p = [np.asarray(r["P_tier_p"]) for r in rows]   # per-alpha total

OUT = Path("figures/analysis/03_templates_and_p1d")
OUT.mkdir(parents=True, exist_ok=True)
np.savez(OUT / "tierc_tau0_response_z3.npz",
         kf=kf, alphas=np.asarray(ALPHAS), target_F=np.asarray(tF),
         scale=np.asarray(sc), counts=counts,
         P_tier_p=np.stack(P_tier_p),
         **{f"P_{n}": np.stack(P_cls[n]) for n in NAMES})

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

colors = {0.70: "C0", 1.00: "C1", 1.35: "C2"}
panels = NAMES + ["Tier P (total)"]
series = {n: P_cls[n] for n in NAMES}
series["Tier P (total)"] = P_tier_p

fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
axes = axes.ravel()
m = kf > 0
for ax, name in zip(axes, panels):
    for a, P in zip(ALPHAS, series[name]):
        P = np.asarray(P)
        ax.loglog(kf[m], (kf[m] * P[m] / np.pi), color=colors[a],
                  label=f"alpha={a:.2f} (F={dict(zip(ALPHAS,tF))[a]:.3f})")
    cnt = "" if name.startswith("Tier") else f"  (n={dict(zip(NAMES,cls_counts))[name]})"
    ax.set_title(name + cnt)
    ax.set_xlabel("k  [s/km, angular]")
    ax.set_ylabel(r"$k\,P(k)/\pi$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
axes[-1].axis("off")
fig.suptitle(f"Unfiltered Tier-C tau0 response  —  {sim[:24]}  snap {snap}  z=3.0",
             fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "tierc_tau0_response_z3.png", dpi=150)
print(f"\nwrote {OUT/'tierc_tau0_response_z3.png'}")
