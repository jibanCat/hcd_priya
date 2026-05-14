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

    valid = (P_src > 0) & (k_src > 0)
    if valid.sum() < 2:
        return np.full(k_target.shape, np.nan)

    log_k_src = np.log(k_src[valid])
    log_P_src = np.log(P_src[valid])

    k_lo, k_hi = k_src[valid].min(), k_src[valid].max()
    in_range = (k_target >= k_lo) & (k_target <= k_hi)

    out = np.full(k_target.shape, np.nan)
    out[in_range] = np.exp(np.interp(np.log(k_target[in_range]), log_k_src, log_P_src))
    return out


_LLS_LOW, _LLS_HIGH = 17.2, 19.0
_SUB_LOW, _SUB_HIGH = 19.0, 20.3
_DLA_LOW = 20.3


def compute_dndx_per_class(cddf: dict) -> dict[str, float]:
    """Sum absorber counts per class and divide by total path-length dX."""
    centres = cddf["log_nhi_centres"]
    n_abs = cddf["n_absorbers"]
    total_path = float(cddf["total_path"])

    lls = float(n_abs[(centres >= _LLS_LOW) & (centres < _LLS_HIGH)].sum() / total_path)
    sub = float(n_abs[(centres >= _SUB_LOW) & (centres < _SUB_HIGH)].sum() / total_path)
    dla = float(n_abs[centres >= _DLA_LOW].sum() / total_path)
    return {"dNdX_LLS": lls, "dNdX_subDLA": sub, "dNdX_DLA": dla}


PARAM_ORDER = ("ns", "Ap", "herei", "heref", "alphaq",
               "hub", "omegamh2", "hireionz", "bhfeedback")


def build_row(sim_name: str, snap: int, snap_dir: Path,
              k_target: np.ndarray) -> dict:
    """Assemble one (sim, snap) row matching the output HDF5 schema."""
    params_dict = parse_sim_params(sim_name)
    if params_dict is None:
        raise ValueError(f"could not parse params from sim folder name: {sim_name!r}")
    params = np.array([params_dict[k] for k in PARAM_ORDER], dtype=np.float64)

    meta = read_meta(snap_dir)
    cddf = read_cddf(snap_dir)
    p1d = read_p1d_per_class(snap_dir)
    dndx = compute_dndx_per_class(cddf)

    k_src = p1d["k"]
    row = {
        "sim_name": sim_name,
        "snap": int(snap),
        "params": params,
        "z": float(meta["z"]),
        "dv_kms": float(meta["dv_kms"]),
        "nbins_native": int(meta["nbins"]),
        "n_total_sightlines": int(meta["n_skewers"]),
        "P_clean":       interp_p1d_loglog(k_src, p1d["P_clean"],       k_target),
        "P_LLS_only":    interp_p1d_loglog(k_src, p1d["P_LLS_only"],    k_target),
        "P_subDLA_only": interp_p1d_loglog(k_src, p1d["P_subDLA_only"], k_target),
        "P_DLA_only":    interp_p1d_loglog(k_src, p1d["P_DLA_only"],    k_target),
        "mean_F_clean":  float(p1d["mean_F_clean"]),
        "mean_F_LLS":    float(p1d["mean_F_LLS"]),
        "mean_F_subDLA": float(p1d["mean_F_subDLA"]),
        "mean_F_DLA":    float(p1d["mean_F_DLA"]),
        "n_sightlines_clean":  int(p1d["n_sightlines_clean"]),
        "n_sightlines_LLS":    int(p1d["n_sightlines_LLS"]),
        "n_sightlines_subDLA": int(p1d["n_sightlines_subDLA"]),
        "n_sightlines_DLA":    int(p1d["n_sightlines_DLA"]),
        "log_nhi_centres": np.asarray(cddf["log_nhi_centres"], dtype=np.float64),
        "log_nhi_edges":   np.asarray(cddf["log_nhi_edges"], dtype=np.float64),
        "f_nhi":           np.asarray(cddf["f_nhi"], dtype=np.float64),
        "n_absorbers":     np.asarray(cddf["n_absorbers"], dtype=np.int64),
        "total_path_dX":   float(cddf["total_path"]),
        **dndx,
    }
    return row
