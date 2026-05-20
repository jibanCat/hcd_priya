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


_HCD_ROOT = Path("/scratch/cavestru_root/cavestru0/mfho/hcd_outputs")
_EMU_ROOT = Path("/nfs/turbo/umor-yueyingn/mfho/emu_full")


def test_discover_tau0_pairs_returns_nonempty():
    pairs = bt0.discover_tau0_pairs(_HCD_ROOT, _EMU_ROOT)
    assert len(pairs) >= 1, "no tau0-buildable (sim, snap) pairs found"
    sim, snap, snap_dir, raw = pairs[0]
    assert isinstance(sim, str) and sim.startswith("ns")
    assert isinstance(snap, int)
    assert (snap_dir / "catalog.npz").exists()
    assert raw.exists() and raw.suffix == ".hdf5"
    print(f"discover_tau0_pairs: {len(pairs)} pairs; first = ({sim}, snap_{snap:03d})")


def test_build_tau0_rows_integration_small():
    """Real-data integration: 1 pair, 2 alpha, limited skewers."""
    from hcd_analysis.p1d import _DEFAULT_K_BINS
    pairs = bt0.discover_tau0_pairs(_HCD_ROOT, _EMU_ROOT)
    sim, snap, snap_dir, raw = pairs[0]
    k_target = 2.0 * np.pi * _DEFAULT_K_BINS
    alpha_grid = np.array([0.8, 1.2])

    rows, snap_block = bt0.build_tau0_rows(
        sim, snap, snap_dir, raw, alpha_grid, k_target,
        tau_freeze=1.0e6, n_skewers=4096,
    )

    assert len(rows) == 2
    for a_idx, row in enumerate(rows):
        assert row["sim_name"] == sim and row["snap"] == snap
        assert row["alpha_idx"] == a_idx
        assert row["params"].shape == (9,)
        for key in ("P_clean", "P_LLS_only", "P_subDLA_only", "P_DLA_only"):
            assert row[key].shape == (50,)
        # tau0 == -ln(mean_F_clean) exactly (spec sec.9 unit test)
        assert np.isclose(row["tau0"], -np.log(row["mean_F_clean"]))
    # the two alpha rows must differ (mean flux moved)
    assert not np.isclose(rows[0]["tau0"], rows[1]["tau0"])

    # one tau0-invariant CDDF block per (sim, snap)
    assert snap_block["f_nhi"].shape == (30,)
    assert snap_block["n_absorbers"].shape == (30,)
    for key in ("dNdX_LLS", "dNdX_subDLA", "dNdX_DLA"):
        assert np.isfinite(snap_block[key])
    print(f"build_tau0_rows: ({sim}, snap_{snap:03d}) "
          f"tau0={rows[0]['tau0']:.3f},{rows[1]['tau0']:.3f}")


if __name__ == "__main__":
    test_locate_raw_tau_file_finds_grid_file()
    test_locate_raw_tau_file_returns_none_when_missing()
    test_discover_tau0_pairs_returns_nonempty()
    test_build_tau0_rows_integration_small()
    print("OK")
