"""Tests for scripts/build_emulator_cache_tau0.py.

Run with: python3 tests/test_emulator_cache_tau0.py
"""
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_emulator_cache_tau0 as bt0


def test_locate_raw_tau_file_finds_grid_file():
    with tempfile.TemporaryDirectory() as tmp:
        emu_root = Path(tmp)
        sim = "ns0.8Ap2e-09herei4heref3alphaq2hub0.7omegamh20.14hireionz7bhfeedback0.03"
        spectra_dir = emu_root / sim / "output" / "SPECTRA_010"
        spectra_dir.mkdir(parents=True)
        grid = spectra_dir / "lya_forest_spectra_grid_480.hdf5"
        grid.write_bytes(b"")  # presence is all locate checks

        found = bt0.locate_raw_tau_file(emu_root, sim, 10)
        assert found == grid, f"got {found}"


def test_locate_raw_tau_file_returns_none_when_missing():
    with tempfile.TemporaryDirectory() as tmp:
        found = bt0.locate_raw_tau_file(Path(tmp), "ns0.8Ap2e-09", 10)
        assert found is None


if __name__ == "__main__":
    test_locate_raw_tau_file_finds_grid_file()
    test_locate_raw_tau_file_returns_none_when_missing()
    print("OK")
