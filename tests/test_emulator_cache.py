"""Tests for scripts/build_emulator_cache.py.

Run with: python3 tests/test_emulator_cache.py
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_emulator_cache as bec

HCD_ROOT = Path("/scratch/cavestru_root/cavestru0/mfho/hcd_outputs")


def test_discover_sim_snap_pairs_returns_nonempty():
    pairs = bec.discover_sim_snap_pairs(HCD_ROOT)
    assert len(pairs) >= 1, f"no (sim, snap) pairs found under {HCD_ROOT}"
    sim_name, snap, snap_dir = pairs[0]
    assert isinstance(sim_name, str) and sim_name.startswith("ns")
    assert isinstance(snap, int)
    assert snap_dir.is_dir()
    assert (snap_dir / "meta.json").exists()
    print(f"discover_sim_snap_pairs: {len(pairs)} pairs found; "
          f"first = ({sim_name}, snap_{snap:03d})")


if __name__ == "__main__":
    test_discover_sim_snap_pairs_returns_nonempty()
    print("OK")
