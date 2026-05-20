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


def _fake_row(sim, snap, a_idx, alpha, snap_group_idx):
    return {
        "sim_name": sim, "snap": snap, "alpha": alpha, "alpha_idx": a_idx,
        "snap_group_idx": snap_group_idx,
        "tau0": 2.0 + 0.1 * a_idx, "z": 3.0, "dv_kms": 10.0, "nbins_native": 1500,
        "params": np.arange(9, dtype=np.float64),
        "P_clean": np.full(50, 1.0), "P_LLS_only": np.full(50, 2.0),
        "P_subDLA_only": np.full(50, 3.0), "P_DLA_only": np.full(50, 4.0),
        "mean_F_clean": 0.3, "mean_F_LLS": 0.2,
        "mean_F_subDLA": 0.1, "mean_F_DLA": 0.05,
        "n_sightlines_clean": 600, "n_sightlines_LLS": 50,
        "n_sightlines_subDLA": 20, "n_sightlines_DLA": 10,
    }


def _fake_snap_block(sim, snap):
    return {
        "sim_name": sim, "snap": snap,
        "f_nhi": np.full(30, 1e-21), "n_absorbers": np.arange(30, dtype=np.int64),
        "log_nhi_centres": np.linspace(17.1, 22.9, 30),
        "log_nhi_edges": np.linspace(17.0, 23.0, 31),
        "total_path_dX": 1234.5,
        "dNdX_LLS": 0.5, "dNdX_subDLA": 0.2, "dNdX_DLA": 0.1,
    }


def test_write_cache_tau0_round_trip():
    sim = "ns0.8Ap2e-09herei4heref3alphaq2hub0.7omegamh20.14hireionz7bhfeedback0.03"
    k_target = np.geomspace(0.007, 0.31, 50)
    # 1 snap, 3 alpha
    rows = [_fake_row(sim, 10, i, 0.7 + 0.3 * i, snap_group_idx=0) for i in range(3)]
    snap_blocks = [_fake_snap_block(sim, 10)]

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "observables_tau0.h5"
        bt0.write_cache_tau0(rows, snap_blocks, k_target, out,
                             tier="B", tau_freeze=1.0e6,
                             alpha_range=(0.66, 1.36))
        with h5py.File(out, "r") as f:
            assert f["P_clean"].shape == (3, 50)
            assert f["alpha"].shape == (3,)
            assert f["tau0"].shape == (3,)
            assert f["snap_group_idx"].shape == (3,)
            assert np.all(f["snap_group_idx"][...] == 0)
            assert f["snap_f_nhi"].shape == (1, 30)
            assert f["snap_dNdX_DLA"].shape == (1,)
            assert f["k_target"].shape == (50,)
            assert f.attrs["rescale_tier"] == "B"
            assert f.attrs["tau_freeze"] == 1.0e6
            assert np.allclose(f["P_DLA_only"][0], 4.0)
            # per-row CDDF is reachable via snap_group_idx
            gi = f["snap_group_idx"][1]
            assert f["snap_f_nhi"][gi].shape == (30,)
    print("write_cache_tau0: round-trip OK")


if __name__ == "__main__":
    test_locate_raw_tau_file_finds_grid_file()
    test_locate_raw_tau_file_returns_none_when_missing()
    test_write_cache_tau0_round_trip()
    test_discover_tau0_pairs_returns_nonempty()
    test_build_tau0_rows_integration_small()
    print("OK")
