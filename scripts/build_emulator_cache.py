"""Build the HCD-emulator training cache (Phase 1).

Walks the on-disk (sim, snap) outputs and stacks their per-class P1D,
CDDF, dN/dX, and parameter-vector observables into one HDF5 file at
`hcd_analysis/_emulator_data/observables.h5`. Read-only over the
source data; no recomputation from raw spectra.

Usage:
    python3 scripts/build_emulator_cache.py \
        [--root /scratch/cavestru_root/cavestru0/mfho/hcd_outputs] \
        [--output hcd_analysis/_emulator_data/observables.h5] \
        [--limit N] [--spot-check]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from hcd_analysis.io import parse_sim_params  # noqa: E402

SNAP_DIR_RE = re.compile(r"^snap_(\d{3})$")


def discover_sim_snap_pairs(root: Path) -> list[tuple[str, int, Path]]:
    """Return [(sim_folder_name, snap_number, snap_dir_path), ...] for every
    valid (sim, snap) under `root`. A pair is valid only if `meta.json`,
    `cddf_corrected.npz`, and `p1d_per_class.h5` all exist in the snap dir.
    """
    pairs: list[tuple[str, int, Path]] = []
    for sim_dir in sorted(root.iterdir()):
        if not sim_dir.is_dir():
            continue
        if parse_sim_params(sim_dir.name) is None:
            continue
        for snap_dir in sorted(sim_dir.iterdir()):
            m = SNAP_DIR_RE.match(snap_dir.name)
            if not m or not snap_dir.is_dir():
                continue
            required = ["meta.json", "cddf_corrected.npz", "p1d_per_class.h5"]
            if not all((snap_dir / f).exists() for f in required):
                continue
            pairs.append((sim_dir.name, int(m.group(1)), snap_dir))
    return pairs


def read_meta(snap_dir: Path) -> dict:
    with open(snap_dir / "meta.json") as f:
        return json.load(f)


def read_cddf(snap_dir: Path) -> dict:
    with np.load(snap_dir / "cddf_corrected.npz") as data:
        return {key: data[key] for key in data.files}


def read_p1d_per_class(snap_dir: Path) -> dict:
    out: dict = {}
    with h5py.File(snap_dir / "p1d_per_class.h5", "r") as f:
        for key in f.keys():
            arr = f[key][...]
            out[key] = arr.item() if arr.shape == () else arr
    return out


def interp_p1d_loglog(k_src: np.ndarray, P_src: np.ndarray,
                      k_target: np.ndarray) -> np.ndarray:
    """Log-log linear interpolation of P(k) from `k_src` onto `k_target`.

    Out-of-range targets (k_target < k_src.min() or > k_src.max()) are
    set to NaN. Non-positive P values in the source are dropped before
    log-spacing the interpolation (P1D should be > 0 in practice; this
    guards against floating-point edge cases).
    """
    k_src = np.asarray(k_src, dtype=np.float64)
    P_src = np.asarray(P_src, dtype=np.float64)
    k_target = np.asarray(k_target, dtype=np.float64)

    pos = P_src > 0
    if pos.sum() < 2:
        return np.full(k_target.shape, np.nan)

    log_k_src = np.log(k_src[pos])
    log_P_src = np.log(P_src[pos])

    k_lo, k_hi = k_src[pos].min(), k_src[pos].max()
    in_range = (k_target >= k_lo) & (k_target <= k_hi)

    out = np.full(k_target.shape, np.nan)
    out[in_range] = np.exp(np.interp(np.log(k_target[in_range]), log_k_src, log_P_src))
    return out
