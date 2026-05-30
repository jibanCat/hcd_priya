"""Re-estimate the production tau0 build cost by scaling the ONE measured timing
point to every (sim, snap) pair via its tau-array size (n_skewers * nbins).

Measured anchor (job 51140681, LF offset-0 pair ns0.803 snap_004, z=5.4):
  n_skewers=691200, nbins=1556 -> work=1.0755e9 pixels
  wall=1.945 h, peak RSS=34.3 GB, TotalCPU=2.38 CPU-h
Memory and wall both scale ~linearly with work (tau arrays + FFT); there is a
fixed overhead, so small pairs are slightly OVER-estimated (conservative).
Billing (Great Lakes, MAX_TRES): core_equiv = max(cores, mem_GB/7.0); we drive
cores to 2 (job is ~serial), so memory floors the bill.
"""
import sys, json
from pathlib import Path
import numpy as np

sys.path.insert(0, "scripts")
import build_emulator_cache_tau0 as bt0

# anchor
W0 = 691200 * 1556          # work (pixels) of the measured pair
WALL0 = 1.945               # h
PEAK0 = 34.3                # GB
MEM_MARGIN = 1.18           # request peak*margin (the first OOM showed peak > sampled MaxRSS)
MEM_PER_CORE = 7.0          # GB per core-equiv (TRESBillingWeights cpu/mem ratio)
CORES = 2                   # minimal; build is ~serial (30% eff on 4)

HCD_ROOT = Path("/scratch/cavestru_root/cavestru0/mfho/hcd_outputs")
LF_EMU = Path("/nfs/turbo/umor-yueyingn/mfho/emu_full")
HR_EMU = Path("/scratch/yueyingn_root/yueyingn0/mfho/priya/emu_full_hires_2")

def pair_costs(pairs, label):
    rows = []
    for sim, snap, snap_dir, raw in pairs:
        m = json.load(open(snap_dir / "meta.json"))
        nbins = int(m["nbins"]); nsk = int(m.get("n_skewers", 691200))
        work = nsk * nbins
        wall = WALL0 * work / W0
        peak = PEAK0 * work / W0
        mem_req = peak * MEM_MARGIN
        core_eq = max(CORES, mem_req / MEM_PER_CORE)
        billed = core_eq * wall
        rows.append((nbins, nsk, wall, peak, core_eq, billed))
    a = np.array(rows, dtype=float)
    print(f"\n=== {label}: {len(rows)} pairs ===")
    print(f"  nbins      : min={a[:,0].min():.0f} med={np.median(a[:,0]):.0f} max={a[:,0].max():.0f}")
    print(f"  n_skewers  : med={np.median(a[:,1]):.0f}")
    print(f"  wall/pair h: min={a[:,2].min():.2f} med={np.median(a[:,2]):.2f} max={a[:,2].max():.2f}")
    print(f"  peak GB    : min={a[:,3].min():.1f} med={np.median(a[:,3]):.1f} max={a[:,3].max():.1f}")
    print(f"  core_equiv : med={np.median(a[:,4]):.2f} (mem-floored)")
    print(f"  billed/pair CPU-h: med={np.median(a[:,5]):.1f}")
    print(f"  >>> TOTAL billed: {a[:,5].sum():.0f} CPU-h   (sum wall={a[:,2].sum():.0f} pair-h)")
    return a[:,5].sum()

lf = bt0.discover_tau0_pairs(HCD_ROOT, LF_EMU, fidelity="lf")
hr = bt0.discover_tau0_pairs(HCD_ROOT, HR_EMU, fidelity="hr")
lf_tot = pair_costs(lf, "LF")
hr_tot = pair_costs(hr, "HR")
print(f"\n=== GRAND TOTAL (20 alpha): {lf_tot + hr_tot:.0f} CPU-h ===")
print(f"    at 10 alpha (~half the per-alpha loop): ~{(lf_tot + hr_tot)*0.55:.0f} CPU-h")
print("    (10-alpha factor ~0.55 not 0.5: tau-read + mean-flux solve overhead is alpha-shared)")
