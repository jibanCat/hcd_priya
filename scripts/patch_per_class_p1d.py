"""
Patch existing snap directories under OUTPUT_ROOT by adding p1d_per_class.h5.

For each snap dir that has a `done` sentinel + `catalog.npz` + `meta.json`,
stream the tau file once and append a p1d_per_class.h5 file with four
subset P1Ds (clean / LLS / subDLA / DLA) plus their mean-F and sightline
counts, all in HDF5 with metadata attributes.

Does NOT modify any existing output; only adds p1d_per_class.h5.

Usage:
    python3 scripts/patch_per_class_p1d.py                       # all done snaps
    python3 scripts/patch_per_class_p1d.py --n-workers 4
    python3 scripts/patch_per_class_p1d.py --limit 10            # debug subset
    python3 scripts/patch_per_class_p1d.py --force               # overwrite existing
    python3 scripts/patch_per_class_p1d.py --hires               # /hires subtree only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hcd_analysis.catalog import AbsorberCatalog
from hcd_analysis.io import parse_sim_params, spectra_files_for_sim, SimInfo
from hcd_analysis.p1d import compute_p1d_per_class, save_p1d_per_class_hdf5

LF_DATA_ROOT = Path("/nfs/turbo/umor-yueyingn/mfho/emu_full")
HIRES_DATA_ROOT = Path("/nfs/turbo/lsa-cavestru/mfho/priya/emu_full_hires")
OUTPUT_ROOT = Path("/scratch/cavestru_root/cavestru0/mfho/hcd_outputs")
FORCE = False


def patch_one(snap_dir: Path) -> str:
    out_h5 = snap_dir / "p1d_per_class.h5"
    if not FORCE and out_h5.exists():
        return f"ALREADY_DONE {snap_dir.name}"
    meta_path = snap_dir / "meta.json"
    cat_path = snap_dir / "catalog.npz"
    if not (meta_path.exists() and cat_path.exists()):
        return f"SKIP (missing meta/catalog): {snap_dir}"
    with open(meta_path) as f:
        meta = json.load(f)
    sim_name = meta["sim_name"]
    snap_num = int(meta["snap"])
    nbins = int(meta["nbins"])
    dv_kms = float(meta["dv_kms"])
    z = float(meta["z"])

    is_hires = (OUTPUT_ROOT / "hires") in snap_dir.parents
    data_root = HIRES_DATA_ROOT if is_hires else LF_DATA_ROOT
    sim_path = data_root / sim_name
    if not sim_path.exists():
        return f"NO_SIM_DIR ({'hires' if is_hires else 'LF'}): {sim_name}"
    params = parse_sim_params(sim_name) or {}
    sim_info = SimInfo(name=sim_name, path=sim_path, params=params)
    spectra_map = spectra_files_for_sim(sim_info, prefer_grid=True)
    if snap_num not in spectra_map:
        return f"NO_HDF5 snap={snap_num}: {sim_name[:30]}"
    hdf5_path = spectra_map[snap_num]

    try:
        cat = AbsorberCatalog.load_npz(cat_path)
        res = compute_p1d_per_class(
            hdf5_path=hdf5_path,
            nbins=nbins, dv_kms=dv_kms,
            catalog=cat,
            batch_size=4096, n_skewers=None,
        )
        save_p1d_per_class_hdf5(
            out_h5, res,
            sim_name=sim_name, snap=snap_num, z=z, dv_kms=dv_kms,
            extra_attrs={
                "nbins": nbins,
                "mode": "hires" if is_hires else "LF",
                "tau_threshold": 100.0,
                "min_log_nhi": 17.2,
                "fast_mode": True,
                "patch_script": "patch_per_class_p1d.py",
            },
        )
    except Exception as e:
        return f"ERROR {snap_dir.name}: {e}"
    return (f"OK z={z:.2f} counts(cl/LLS/sub/DLA)={res['n_sightlines_clean']}/"
            f"{res['n_sightlines_LLS']}/{res['n_sightlines_subDLA']}/{res['n_sightlines_DLA']}: "
            f"{sim_name[:30]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--hires", action="store_true",
                        help="Only process the /hires subtree")
    parser.add_argument("--lf-only", action="store_true",
                        help="Only process the LF (non-hires) subtree")
    args = parser.parse_args()
    global FORCE
    FORCE = args.force

    if args.hires:
        scan_root = OUTPUT_ROOT / "hires"
    else:
        scan_root = OUTPUT_ROOT
    snap_dirs = []
    for done in sorted(scan_root.rglob("done")):
        snap_dir = done.parent
        if args.lf_only and "hires" in snap_dir.parts:
            continue
        if snap_dir.name.startswith("snap_") and (snap_dir / "catalog.npz").exists():
            snap_dirs.append(snap_dir)
    if args.limit:
        snap_dirs = snap_dirs[:args.limit]
    print(f"Found {len(snap_dirs)} snap dirs to check/patch")

    t0 = time.time()
    counters = {"OK": 0, "ALREADY_DONE": 0, "SKIP": 0, "ERROR": 0}
    with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
        futs = [ex.submit(patch_one, d) for d in snap_dirs]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r.startswith("OK"):                  counters["OK"] += 1
            elif r.startswith("ALREADY_DONE"):      counters["ALREADY_DONE"] += 1
            elif r.startswith(("SKIP", "NO_")):     counters["SKIP"] += 1
            else:                                   counters["ERROR"] += 1
            if i % 25 == 0 or i == len(snap_dirs):
                print(f"  [{i}/{len(snap_dirs)}] {time.time()-t0:.0f}s  {counters}  last={r}")

    print(f"\nDone in {time.time()-t0:.0f}s: {counters}")


if __name__ == "__main__":
    main()
