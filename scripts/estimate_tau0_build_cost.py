"""Cost matrix for the production tau0 build under memory strategies x alpha count.
Bill (Great Lakes, MAX_TRES): core_equiv = max(2, mem_GB/7); billed = core_equiv*wall.
Memory is the floor, so the levers are (a) request mem per-pair (nbins-tiered) not
uniform, and (b) reduce the peak by CHUNKING Tier-P's flux_power (mirror the already-
verified chunked _per_class path) so we hold 2 tau arrays instead of 4.

Anchors (measured): peak 34.3 GB and walls 1.03 h (10a) / 1.945 h (20a) at
work0 = 691200*1556 pixels. Peak ~ bytes/pixel * work.
  current monolithic Tier-P: ~31.9 B/px (4 full arrays)
  chunked Tier-P:           ~16 B/px (2 tau arrays) + ~2.5 GB fixed
"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, "scripts")
import build_emulator_cache_tau0 as bt0

WORK0 = 691200 * 1556
WALL10, WALL20 = 1.03, 1.945           # h/pair at work0 (measured)
BPP_CUR = 34.3e9 / WORK0               # bytes/pixel, monolithic
BPP_CHUNK = 16.0e9 / WORK0             # 2 float64 tau arrays
FIXED_CHUNK = 2.5e9                    # GB overhead for chunked path
MARGIN = 1.20
HCD = Path("/scratch/cavestru_root/cavestru0/mfho/hcd_outputs")
LF = Path("/nfs/turbo/umor-yueyingn/mfho/emu_full")
HR = Path("/scratch/yueyingn_root/yueyingn0/mfho/priya/emu_full_hires_2")

def works(pairs):
    w = []
    for sim, snap, sd, raw in pairs:
        m = json.load(open(sd / "meta.json"))
        w.append(int(m.get("n_skewers", 691200)) * int(m["nbins"]))
    return np.array(w, float)

lf_w = works(bt0.discover_tau0_pairs(HCD, LF, fidelity="lf"))
hr_w = works(bt0.discover_tau0_pairs(HCD, HR, fidelity="hr"))
allw = np.concatenate([lf_w, hr_w])
print(f"pairs: LF={len(lf_w)} HR={len(hr_w)}  work/work0: med={np.median(allw)/WORK0:.2f} max={allw.max()/WORK0:.2f}\n")

def total(bpp, fixed, wall_anchor, mem_strategy):
    ce, billed = [], []
    for w in allw:
        peak = bpp * w + fixed
        wall = wall_anchor * w / WORK0
        if mem_strategy == "uniform48":
            mem = 48e9
        else:  # per-pair (nbins-tiered, set mem = peak*margin)
            mem = peak * MARGIN
        c = max(2.0, mem / 7e9)
        ce.append(c); billed.append(c * wall)
    return sum(billed), np.mean(ce)

print(f"{'strategy':36s} {'10 alpha':>12s} {'20 alpha':>12s}")
for label, bpp, fixed, strat in [
    ("current (monolithic) uniform 48G", BPP_CUR, 0.0, "uniform48"),
    ("current (monolithic) nbins-tiered", BPP_CUR, 0.0, "perpair"),
    ("CHUNKED Tier-P nbins-tiered",       BPP_CHUNK, FIXED_CHUNK, "perpair"),
]:
    t10, ce10 = total(bpp, fixed, WALL10, strat)
    t20, ce20 = total(bpp, fixed, WALL20, strat)
    print(f"{label:36s} {t10:7.0f} CPU-h {t20:7.0f} CPU-h   (mean core_eq {ce10:.1f})")
print("\n(+~1k CPU-h already burned on the cancelled run.)")
