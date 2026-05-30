"""One-snapshot sensitivity of the unfiltered Tier-C per-class P1D to the
freeze threshold tau_freeze. Replaces the (infeasible) PART re-extraction
ground truth with: (a) how big the freeze effect is vs the uniform rescale,
(b) how the tau0 response shrinks as self-shielded gas is frozen, (c) the
frozen-pixel fraction per threshold (to anchor against the literature
self-shielding column). One LF snapshot, subset of skewers, in emu-3.9 env.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "scripts")
import build_emulator_cache_tau0 as bt0
from hcd_analysis.priya_p1d import (compute_tier_p_p1d, compute_tier_c_p1d,
                                    merge_fine_to_classes)
from hcd_analysis.io import read_header

SIM_PREFIX = "ns0.803"
N_SKEWERS = 40000
ALPHAS = [0.70, 1.00, 1.35]
TAU_FREEZES = [np.inf, 1e5, 1e4, 1e3, 3e2]      # inf = uniform (current twin)
RANGES = [(0, 1), (1, 8), (8, 13), (13, 15)]
NAMES = ["clean", "LLS", "subDLA", "DLA"]
KBAND = slice(5, 120)                            # mid-k band for the metrics

HCD_ROOT = Path("/scratch/cavestru_root/cavestru0/mfho/hcd_outputs")
EMU_ROOT = Path("/nfs/turbo/umor-yueyingn/mfho/emu_full")

import json
pairs = bt0.discover_tau0_pairs(HCD_ROOT, EMU_ROOT, fidelity="lf")
cand = [p for p in pairs if p[0].startswith(SIM_PREFIX)]
cand.sort(key=lambda p: abs(float(json.load(open(p[2] / "meta.json"))["z"]) - 3.0))
sim, snap, snap_dir, raw = cand[0]
meta = bt0.bec.read_meta(snap_dir)
z_grid = bt0._snap_z_to_priya_grid(float(meta["z"]))
vmax = int(read_header(raw).nbins) * float(meta["dv_kms"])
from hcd_analysis.catalog import AbsorberCatalog
catalog = AbsorberCatalog.load_npz(snap_dir / "catalog.npz")
tau = bt0._read_tau(raw, n_skewers=N_SKEWERS)
print(f"sim={sim[:22]} snap={snap} z_grid={z_grid} n_skewers={N_SKEWERS}\n")

# frozen-pixel fraction per threshold (alpha-independent; native tau)
print("frozen-pixel fraction (native tau > tau_freeze):")
for tf in TAU_FREEZES:
    frac = float(np.mean(tau > tf))
    print(f"  tau_freeze={tf:>8.0f}  frac={frac:.3e}")
print()

# per (alpha, tau_freeze): collapse 15->4 class P1Ds, sharing Tier-P scale/target_F
def classed(alpha, tau_freeze):
    tf_arr = tau.copy()
    _, _, target_F, scale = compute_tier_p_p1d(tf_arr, vmax, alpha_slope=alpha, z=z_grid)
    _, P_by_bin, n_by_bin, _, _ = compute_tier_c_p1d(
        tau, vmax, alpha_slope=alpha, z=z_grid, catalog=catalog,
        external_scale=scale, external_target_F=target_F, tau_freeze=tau_freeze)
    merged = merge_fine_to_classes(P_by_bin, n_by_bin, RANGES)
    return {n: np.asarray(P) for n, P in zip(NAMES, merged)}

P = {tf: {a: classed(a, tf) for a in ALPHAS} for tf in TAU_FREEZES}

def rel(a, b):
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.abs(a[KBAND] - b[KBAND]) / np.abs(b[KBAND])
    r = r[np.isfinite(r)]
    return float(np.mean(r)) if r.size else float("nan")

# (a) freeze effect at alpha=1: P_frozen vs P_uniform(inf)
print("(a) freeze effect at alpha=1.0  [mean_k |P_frozen - P_uniform| / P_uniform]:")
print(f"    {'tau_freeze':>10s} " + " ".join(f"{n:>9s}" for n in NAMES))
for tf in TAU_FREEZES:
    if tf == np.inf:
        continue
    effs = [rel(P[tf][1.0][n], P[np.inf][1.0][n]) for n in NAMES]
    print(f"    {tf:>10.0f} " + " ".join(f"{e:9.4f}" for e in effs))

# (b) tau0 response per tau_freeze: mean_k |P(hi)-P(lo)| / P(mid)
print("\n(b) tau0 response  [mean_k |P(a=1.35) - P(a=0.70)| / P(a=1.0)] per class:")
print(f"    {'tau_freeze':>10s} " + " ".join(f"{n:>9s}" for n in NAMES))
for tf in TAU_FREEZES:
    resp = [rel(P[tf][1.35][n], P[tf][0.70][n]) for n in NAMES]  # diff via |hi-lo|/mid below
    resp = []
    for n in NAMES:
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.abs(P[tf][1.35][n][KBAND] - P[tf][0.70][n][KBAND]) / np.abs(P[tf][1.0][n][KBAND])
        r = r[np.isfinite(r)]
        resp.append(float(np.mean(r)) if r.size else float("nan"))
    tag = "(uniform)" if tf == np.inf else ""
    print(f"    {tf:>10.0f} " + " ".join(f"{e:9.4f}" for e in resp) + f"  {tag}")

print("\nReading: (a) how much freezing moves each class's P1D vs uniform; (b) how "
      "the per-class tau0 response shrinks as more self-shielded gas is frozen "
      "(HCD classes should drop most; clean ~unchanged). Stability of (b) across "
      "tau_freeze = robustness of the threshold choice.")
