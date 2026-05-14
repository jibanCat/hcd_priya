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

import re
import sys
from pathlib import Path

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
