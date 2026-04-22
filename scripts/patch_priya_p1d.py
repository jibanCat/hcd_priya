"""
Patch existing p1d.npz files to add p1d_no_DLA_priya variant.

For each completed snap dir (has p1d.npz + meta.json + done):
  1. Find the tau HDF5 file from data_root/sim_name/output/SPECTRA_NNN/
  2. Run compute_p1d_priya_masked (two-pass)
  3. Append k_no_DLA_priya, p1d_no_DLA_priya, meanF_no_DLA_priya to p1d.npz

Usage:
  python3 scripts/patch_priya_p1d.py [--n-workers N] [--limit N]
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hcd_analysis.io import spectra_files_for_sim, SimInfo, parse_sim_params
from hcd_analysis.p1d import compute_p1d_priya_masked

LF_DATA_ROOT = Path("/nfs/turbo/umor-yueyingn/mfho/emu_full")
HIRES_DATA_ROOT = Path("/nfs/turbo/lsa-cavestru/mfho/priya/emu_full_hires")
OUTPUT_ROOT = Path("/scratch/cavestru_root/cavestru0/mfho/hcd_outputs")
FORCE = False  # set via --force flag in main()


def patch_one(snap_dir: Path) -> str:
    """Patch one snap directory. Returns status string."""
    p1d_path = snap_dir / "p1d.npz"
    meta_path = snap_dir / "meta.json"

    if not (p1d_path.exists() and meta_path.exists()):
        return f"SKIP (no p1d or meta): {snap_dir}"

    with open(meta_path) as f:
        meta = json.load(f)

    existing = np.load(p1d_path, allow_pickle=False)
    if not FORCE and "p1d_no_DLA_priya" in existing.files:
        return f"ALREADY_DONE: {snap_dir.name}"

    sim_name = meta["sim_name"]
    snap_num = int(meta["snap"])
    nbins = int(meta["nbins"])
    dv_kms = float(meta["dv_kms"])

    # Detect hires vs LF by output directory location
    is_hires = (OUTPUT_ROOT / "hires") in snap_dir.parents
    data_root = HIRES_DATA_ROOT if is_hires else LF_DATA_ROOT
    sim_path = data_root / sim_name
    if not sim_path.exists():
        return f"NO_SIM_DIR ({'hires' if is_hires else 'LF'}): {sim_name}"

    params = parse_sim_params(sim_name) or {}
    sim_info = SimInfo(name=sim_name, path=sim_path, params=params)
    spectra_map = spectra_files_for_sim(sim_info, prefer_grid=True)
    if snap_num not in spectra_map:
        return f"NO_HDF5 snap={snap_num}: {sim_name}"

    hdf5_path = spectra_map[snap_num]

    try:
        k, p1d, mean_F = compute_p1d_priya_masked(
            hdf5_path, nbins=nbins, dv_kms=dv_kms,
            batch_size=4096, n_skewers=None,
            k_bins=None,  # use default k grid (same as other variants)
        )
    except Exception as e:
        return f"ERROR {snap_dir}: {e}"

    # Merge and save
    data = dict(existing)
    data["k_no_DLA_priya"] = k
    data["p1d_no_DLA_priya"] = p1d
    data["meanF_no_DLA_priya"] = np.float64(mean_F)
    np.savez(p1d_path, **data)
    return f"OK z={meta['z']:.2f}: {sim_name[:40]}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Recompute even if already patched")
    args = parser.parse_args()
    global FORCE
    FORCE = args.force

    # Collect all snap dirs with done sentinel and p1d.npz but missing priya key
    snap_dirs = []
    for done_file in sorted(OUTPUT_ROOT.rglob("done")):
        snap_dir = done_file.parent
        if (snap_dir / "p1d.npz").exists():
            snap_dirs.append(snap_dir)

    if args.limit:
        snap_dirs = snap_dirs[:args.limit]

    print(f"Found {len(snap_dirs)} completed snap dirs to check/patch")

    t0 = time.time()
    ok = skip = already = error = 0

    with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
        futs = {ex.submit(patch_one, d): d for d in snap_dirs}
        for i, fut in enumerate(as_completed(futs), 1):
            result = fut.result()
            if result.startswith("OK"):
                ok += 1
            elif result.startswith("ALREADY"):
                already += 1
            elif result.startswith("SKIP") or result.startswith("NO_"):
                skip += 1
            else:
                error += 1
            if i % 50 == 0 or i == len(snap_dirs):
                elapsed = time.time() - t0
                print(f"  [{i}/{len(snap_dirs)}] {elapsed:.0f}s  OK={ok} already={already} skip={skip} err={error}")
                print(f"    last: {result}")

    print(f"\nDone in {time.time()-t0:.0f}s: OK={ok} already={already} skip={skip} errors={error}")


if __name__ == "__main__":
    main()
