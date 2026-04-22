"""
End-to-end regression check: run the pipeline's run_one_snap on one (sim, snap)
with the new production defaults (fast_mode catalog + PRIYA-only mask variants),
dump to a throwaway output dir, and print the key numbers.

Does NOT touch production data in /scratch.
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from hcd_analysis.config import load_config
from hcd_analysis.pipeline import run_one_snap
from hcd_analysis.snapshot_map import build_snapshot_map

SIM = "ns0.803Ap2.2e-09herei4.05heref2.67alphaq2.21hub0.735omegamh20.141hireionz7.17bhfeedback0.056"
TARGET_SNAP = 17
OUT_DIR = ROOT / "tests" / "out" / "sanity_run"


def main():
    cfg = load_config(str(ROOT / "config" / "default.yaml"))
    # redirect outputs to a throwaway dir
    cfg.output_root = str(OUT_DIR)
    cfg.resume = False          # force recompute
    cfg.n_workers_skewer = 8
    cfg.sim_filter = [SIM]
    print(f"Config: fast_mode={cfg.absorber.fast_mode}  output_root={cfg.output_root}")

    snap_map = build_snapshot_map(
        data_root=cfg.data_root, z_min=cfg.z_min, z_max=cfg.z_max,
        sim_filter=cfg.sim_filter,
    )
    if not snap_map:
        print("NO SIM MATCHED"); return
    ss = snap_map[0]
    entry = [e for e in ss.entries if e.snap == TARGET_SNAP]
    if not entry:
        print(f"Snap {TARGET_SNAP} not found in {ss.sim.name[:40]}"); return
    entry = entry[0]
    print(f"Running sim={ss.sim.name[:40]}...  snap={entry.snap}  z={entry.z}")
    t0 = time.time()
    res = run_one_snap(ss, entry, cfg)
    if res is None:
        print("run_one_snap returned None — error, see out/error.txt")
        err_path = OUT_DIR / ss.sim.name / f"snap_{entry.snap:03d}" / "error.txt"
        if err_path.exists():
            print(err_path.read_text())
        return
    print(f"  DONE in {time.time()-t0:.0f}s  catalog: {res.catalog.summary()}")

    # Inspect outputs
    sd = OUT_DIR / ss.sim.name / f"snap_{entry.snap:03d}"
    print(f"\nOutput files in {sd}:")
    for p in sorted(sd.iterdir()):
        print(f"  {p.name:25s}  {p.stat().st_size:>10d} bytes")

    # meta.json
    meta = json.load(open(sd / "meta.json"))
    print(f"\nmeta.json:")
    print(f"  z = {meta['z']}")
    print(f"  n_absorbers = {meta['n_absorbers']}")
    print(f"  timing = {meta['timing_s']}")

    # p1d variants
    p1d = np.load(sd / "p1d.npz")
    print(f"\np1d.npz keys: {sorted(p1d.files)}")
    for var in ["all", "no_DLA_priya"]:
        k = p1d[f"k_{var}"]; p = p1d[f"p1d_{var}"]; mf = float(p1d[f"meanF_{var}"][0])
        print(f"  {var:15s} mean_F={mf:.4f}  P1D[5]={p[5]:.4g}")

    # ratios
    k_ref = p1d["k_all"]; p_all = p1d["p1d_all"]; p_priya = p1d["p1d_no_DLA_priya"]
    with np.errstate(divide='ignore', invalid='ignore'):
        r = p_priya / p_all
    print(f"\n  no_DLA_priya/all at key k:")
    for k_tgt in [0.001, 0.005, 0.01, 0.02, 0.03, 0.04]:
        j = int(np.argmin(np.abs(k_ref - k_tgt)))
        print(f"    k={k_ref[j]:.4f}  ratio={r[j]:.4f}")


if __name__ == "__main__":
    main()
