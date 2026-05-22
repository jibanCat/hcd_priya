"""Merge per-shard tau0 caches into one canonical observables_tau0.h5.

The production cache is built by sharding the (sim, snap) pairs across many
sbatch tasks (build_emulator_cache_tau0.py --offset/--limit), each writing its
own shard .h5. This script concatenates them:

  - per-row datasets  -> concatenated along axis 0
  - per-snap datasets -> concatenated along axis 0
  - snap_group_idx    -> remapped by the cumulative per-shard snap offset
  - top-level shared datasets (tier_c_labels, tier_c_nhi_edges, param_names,
    log_nhi_*) -> from shard 0 (numeric ones asserted identical across shards)

Usage:
    python3 scripts/merge_tau0_cache.py \
        --shards 'hcd_analysis/_emulator_data/shards/observables_tau0_*.h5' \
        --output hcd_analysis/_emulator_data/observables_tau0.h5
"""
from __future__ import annotations

import argparse
import datetime
import glob
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import build_emulator_cache as bec  # noqa: E402
import build_emulator_cache_tau0 as bt0  # noqa: E402

_TOP_LEVEL = ("tier_c_labels", "tier_c_nhi_edges", "param_names",
              "log_nhi_centres", "log_nhi_edges")
_ROW_STR = ("sim_name",)
_ROW_ARR = ("params",) + bt0._ROW_P1D_KEYS + bt0._ROW_TIERC_KEYS + bt0._ROW_COUNT_KEYS
_ROW_FLOAT = bt0._ROW_FLOAT_KEYS
_ROW_INT = bt0._ROW_INT_KEYS  # includes snap_group_idx (remapped specially)
_SNAP_STR = ("snap_sim_name",)
_SNAP_INT = ("snap_snap",)
_SNAP_FLOAT = tuple("snap_" + k for k in bt0._SNAP_FLOAT_KEYS)
_SNAP_2D = tuple("snap_" + k for k in bt0._SNAP_2D_KEYS)


def merge_shards(shard_paths, output_path):
    shard_paths = sorted(shard_paths)
    if not shard_paths:
        raise ValueError("no shard files matched")
    print(f"Merging {len(shard_paths)} shards -> {output_path}")

    # Accumulators
    row_data = {}   # name -> list of arrays
    snap_data = {}
    snap_group_idx = []
    n_snaps_so_far = 0
    top = {}
    alpha_range = None

    for si, sp in enumerate(shard_paths):
        with h5py.File(sp, "r") as f:
            n_rows = int(f.attrs["n_rows"])
            n_snaps = int(f.attrs["n_snaps"])
            print(f"  shard {si}: {Path(sp).name}  n_rows={n_rows} n_snaps={n_snaps}")
            if si == 0:
                for k in _TOP_LEVEL:
                    top[k] = f[k][...]
                alpha_range = f.attrs["alpha_range"]
                first_n_k = int(f.attrs["n_k"])
                tier_c_note = f.attrs.get("tier_c_note", None)
            else:
                for k in ("tier_c_nhi_edges", "log_nhi_centres", "log_nhi_edges"):
                    assert np.allclose(f[k][...], top[k]), f"{k} mismatch in {sp}"
                assert int(f.attrs["n_k"]) == first_n_k, f"n_k mismatch in {sp}"
                assert list(f["tier_c_labels"][...]) == list(top["tier_c_labels"]), \
                    f"tier_c_labels mismatch in {sp}"

            # per-row: remap snap_group_idx by the running snap offset
            for k in _ROW_STR + _ROW_ARR + _ROW_FLOAT + _ROW_INT:
                if k == "snap_group_idx":
                    snap_group_idx.append(f[k][...].astype(np.int64) + n_snaps_so_far)
                else:
                    row_data.setdefault(k, []).append(f[k][...])
            # per-snap
            for k in _SNAP_STR + _SNAP_INT + _SNAP_FLOAT + _SNAP_2D:
                snap_data.setdefault(k, []).append(f[k][...])
            n_snaps_so_far += n_snaps

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_rows = sum(len(a) for a in row_data["sim_name"])
    total_snaps = sum(len(a) for a in snap_data["snap_snap"])

    with h5py.File(output_path, "w") as f:
        f.attrs["created_utc"] = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
        f.attrs["git_sha"] = bec._git_sha(REPO_ROOT)
        f.attrs["n_rows"] = total_rows
        f.attrs["n_snaps"] = total_snaps
        f.attrs["cache_version"] = "3.1"
        f.attrs["n_k"] = first_n_k
        f.attrs["priya_convention"] = (
            "Kim 2013 slope-alpha (obs_mean_tau=2.3e-3(1+z)^3.65); "
            "fake_spectra _filter_single_tau_complex(tau_thresh=1e6, thresh2=0.25); "
            "flux_power window=False spec_res=0; native k-grid (first n_k FFT bins)")
        f.attrs["tau_thresh"] = 1.0e6
        f.attrs["alpha_range"] = alpha_range
        f.attrs["k_convention"] = "angular (rad*s/km) native FFT grid, PRIYA convention"
        f.attrs["merged_from_n_shards"] = len(shard_paths)
        if tier_c_note is not None:
            f.attrs["tier_c_note"] = tier_c_note

        for k in _TOP_LEVEL:
            f.create_dataset(k, data=top[k])
        for k, lst in row_data.items():
            f.create_dataset(k, data=np.concatenate(lst, axis=0))
        f.create_dataset("snap_group_idx", data=np.concatenate(snap_group_idx).astype(np.int32))
        for k, lst in snap_data.items():
            f.create_dataset(k, data=np.concatenate(lst, axis=0))

    # sanity: snap_group_idx in range
    with h5py.File(output_path, "r") as f:
        gi = f["snap_group_idx"][...]
        ns = f.attrs["n_snaps"]
        assert gi.min() >= 0 and gi.max() < ns, \
            f"snap_group_idx out of range [0,{ns}): [{gi.min()},{gi.max()}]"
        print(f"Wrote {output_path}  ({output_path.stat().st_size/1e6:.1f} MB, "
              f"{f.attrs['n_rows']} rows, {f.attrs['n_snaps']} snaps)")
        print(f"  snap_group_idx range [{gi.min()},{gi.max()}] OK")


def main():
    ap = argparse.ArgumentParser(description="Merge sharded tau0 caches.")
    ap.add_argument("--shards", required=True,
                    help="glob pattern for shard .h5 files (quote it).")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    merge_shards(glob.glob(args.shards), args.output)


if __name__ == "__main__":
    main()
