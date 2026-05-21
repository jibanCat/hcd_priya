"""Tests for scripts/build_emulator_cache_tau0.py (v2.0, fake_spectra-driven).

Real-data tests need the emu-3.9 + GSL env:
    export LD_LIBRARY_PATH=/sw/pkgs/arc/stacks/gcc/10.3.0/gsl/2.7/lib:/home/mfho/.conda/envs/emu-3.9/lib:$LD_LIBRARY_PATH
    /home/mfho/.conda/envs/emu-3.9/bin/python3 tests/test_emulator_cache_tau0.py
The synthetic write_cache_tau0 round-trip runs under plain python3.
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

_HCD_ROOT = Path("/scratch/cavestru_root/cavestru0/mfho/hcd_outputs")
_EMU_ROOT = Path("/nfs/turbo/umor-yueyingn/mfho/emu_full")
_PRIYA_FILE = "/home/mfho/lya_emulator_full/kodiaq_2_2_4_6-48-48/mf_emulator_flux_vectors_tau1000000.hdf5"


def test_locate_raw_tau_file_finds_grid_file():
    with tempfile.TemporaryDirectory() as tmp:
        emu_root = Path(tmp)
        sim = "ns0.8Ap2e-09herei4heref3alphaq2hub0.7omegamh20.14hireionz7bhfeedback0.03"
        spectra_dir = emu_root / sim / "output" / "SPECTRA_010"
        spectra_dir.mkdir(parents=True)
        grid = spectra_dir / "lya_forest_spectra_grid_480.hdf5"
        grid.write_bytes(b"")
        assert bt0.locate_raw_tau_file(emu_root, sim, 10) == grid


def test_locate_raw_tau_file_returns_none_when_missing():
    with tempfile.TemporaryDirectory() as tmp:
        assert bt0.locate_raw_tau_file(Path(tmp), "ns0.8Ap2e-09", 10) is None


def test_snap_z_to_priya_grid():
    assert np.isclose(bt0._snap_z_to_priya_grid(4.600013), 4.6)
    assert np.isclose(bt0._snap_z_to_priya_grid(3.0), 3.0)
    assert np.isclose(bt0._snap_z_to_priya_grid(2.199), 2.2)


def _fake_row(sim, snap, a_idx, alpha, snap_group_idx, nk):
    return {
        "sim_name": sim, "snap": snap, "alpha_slope": alpha, "alpha_idx": a_idx,
        "snap_group_idx": snap_group_idx,
        "target_F": 0.7 - 0.01 * a_idx, "scale": 0.95, "z_meta": 3.000013,
        "z_grid": 3.0, "dv_kms": 10.0, "nbins_native": 1250,
        "params": np.arange(9, dtype=np.float64),
        "P_tier_p": np.full(nk, 5.0), "P_clean": np.full(nk, 4.0),
        "P_LLS": np.full(nk, 3.0), "P_subDLA": np.full(nk, 2.0),
        "P_DLA": np.full(nk, 1.0),
        "n_clean": 600, "n_LLS": 50, "n_subDLA": 20, "n_DLA": 10,
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
    nk = 50
    k_target = np.geomspace(0.007, 0.31, nk)
    rows = [_fake_row(sim, 10, i, 0.66 + 0.05 * i, snap_group_idx=0, nk=nk) for i in range(3)]
    snap_blocks = [_fake_snap_block(sim, 10)]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "observables_tau0.h5"
        bt0.write_cache_tau0(rows, snap_blocks, k_target, out, alpha_range=(0.66, 1.36))
        with h5py.File(out, "r") as f:
            assert f["P_tier_p"].shape == (3, nk)
            assert f["P_clean"].shape == (3, nk)
            assert f["P_DLA"].shape == (3, nk)
            assert f["alpha_slope"].shape == (3,)
            assert f["target_F"].shape == (3,)
            assert f["scale"].shape == (3,)
            assert f["z_grid"].shape == (3,)
            assert f["n_DLA"].shape == (3,)
            assert f["snap_group_idx"].shape == (3,)
            assert np.all(f["snap_group_idx"][...] == 0)
            assert f["snap_f_nhi"].shape == (1, 30)
            assert f["snap_dNdX_DLA"].shape == (1,)
            assert f["k_target"].shape == (nk,)
            assert f.attrs["cache_version"] == "2.0"
            assert f.attrs["tau_thresh"] == 1.0e6
            assert np.allclose(f["P_tier_p"][0], 5.0)
            assert np.allclose(f["P_DLA"][0], 1.0)
    print("write_cache_tau0 v2.0: round-trip OK")


def _have_fake_spectra():
    try:
        import hcd_analysis.priya_p1d  # noqa: F401
        return True
    except ImportError:
        return False


def test_build_tau0_rows_tier_p_matches_priya():
    """Real-data end-to-end: build_tau0_rows at full n_skewers for sim44/snap17
    must reproduce PRIYA's flux_vector (interpolated to the same k_target)."""
    if not _have_fake_spectra():
        print("SKIP test_build_tau0_rows_tier_p_matches_priya — fake_spectra unavailable")
        return
    from hcd_analysis.p1d import _DEFAULT_K_BINS
    sim = "ns0.803Ap2.2e-09herei4.05heref2.67alphaq2.21hub0.735omegamh20.141hireionz7.17bhfeedback0.056"
    snap = 17
    snap_dir = _HCD_ROOT / sim / f"snap_{snap:03d}"
    raw = bt0.locate_raw_tau_file(_EMU_ROOT, sim, snap)
    k_target = 2.0 * np.pi * _DEFAULT_K_BINS
    alpha = 1.0111827706628471  # PRIYA row 344 alpha at this sim

    rows, snap_block = bt0.build_tau0_rows(
        sim, snap, snap_dir, raw, np.array([alpha]), k_target, n_skewers=None)
    assert len(rows) == 1
    row = rows[0]
    for key in ("P_tier_p", "P_clean", "P_LLS", "P_subDLA", "P_DLA"):
        assert row[key].shape == (len(_DEFAULT_K_BINS),)
    assert np.isclose(row["z_grid"], 3.0)
    assert (row["n_clean"] + row["n_LLS"] + row["n_subDLA"] + row["n_DLA"]) == 691200

    # Compare interpolated P_tier_p to PRIYA's flux_vector interpolated the same way.
    from build_emulator_cache import interp_p1d_loglog
    NK = 172; ZIDX = 8; ROW = 344
    with h5py.File(_PRIYA_FILE, "r") as f:
        P_priya = f["flux_vectors"][ROW, ZIDX*NK:(ZIDX+1)*NK].astype(np.float64)
        kp = f["kfkms"][ROW, ZIDX, :].astype(np.float64)
    P_priya_on_target = interp_p1d_loglog(kp, P_priya, k_target)
    both = np.isfinite(P_priya_on_target) & np.isfinite(row["P_tier_p"])
    # PRIYA's per-row kfkms tops out at ~0.086 s/km while k_target (the 50-bin
    # cyclic grid * 2pi) extends to ~0.314, so only ~24 of the 50 k_target bins
    # fall inside PRIYA's range. The load-bearing check is the <1e-3 agreement
    # in the overlap; on the NATIVE Tier-P grid all 171 PRIYA bins agree to ~1e-6.
    assert both.sum() >= 20, f"too few overlapping k bins: {both.sum()}"
    rel = np.abs(row["P_tier_p"][both] / P_priya_on_target[both] - 1)
    assert np.max(rel) < 1e-3, f"Tier-P vs PRIYA max rel diff {np.max(rel):.3e}"
    print(f"build_tau0_rows Tier-P vs PRIYA: max rel diff {np.max(rel):.3e}  "
          f"target_F={row['target_F']:.4f} scale={row['scale']:.4f} "
          f"n=({row['n_clean']},{row['n_LLS']},{row['n_subDLA']},{row['n_DLA']})")


if __name__ == "__main__":
    test_locate_raw_tau_file_finds_grid_file()
    test_locate_raw_tau_file_returns_none_when_missing()
    test_snap_z_to_priya_grid()
    test_write_cache_tau0_round_trip()
    test_build_tau0_rows_tier_p_matches_priya()
    print("OK")
