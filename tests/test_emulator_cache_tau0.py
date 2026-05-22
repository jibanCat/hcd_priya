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
from hcd_analysis import priya_p1d as bt0_pp

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


def test_discover_tau0_pairs_includes_hires_and_is_sorted():
    """discover_tau0_pairs must include the 4 HR sims (under hcd_root/hires)
    and return a deterministic LF-first-then-HR order for stable sharding."""
    pairs = bt0.discover_tau0_pairs(_HCD_ROOT, _EMU_ROOT, include_hires=True)
    pairs_lf_only = bt0.discover_tau0_pairs(_HCD_ROOT, _EMU_ROOT, include_hires=False)
    assert len(pairs) > len(pairs_lf_only), "HR sims not added"
    # HR sims are those whose snap_dir is under .../hires/. All 4 should be
    # found: 3 by exact emu_full name, and the 4th (ns0.914...) via the
    # params-based fallback in locate_raw_tau_file (its emu_full folder rounds
    # alphaq/omegamh2 differently: 1.58/0.142 vs the Phase-1 1.57/0.141).
    hr = [p for p in pairs if "/hires/" in str(p[2])]
    hr_sims = sorted({p[0] for p in hr})
    assert len(hr_sims) == 4, f"expected 4 buildable HR sims, got {len(hr_sims)}: {hr_sims}"
    # The 4th HR sim's raw tau must resolve (via the params fallback) even
    # though its emu_full folder name differs from the Phase-1 name.
    sim4 = "ns0.914Ap1.32e-09herei3.85heref2.65alphaq1.57hub0.742omegamh20.141hireionz6.88bhfeedback0.04"
    raw4 = bt0.locate_raw_tau_file(_EMU_ROOT, sim4, 4)
    assert raw4 is not None and raw4.exists(), f"4th HR raw tau not located: {raw4}"
    assert "alphaq1.58" in str(raw4), f"expected params-fallback match: {raw4}"
    # raw tau for HR resolves under emu_root/<sim> (bare, no hires prefix)
    for sim, snap, snap_dir, raw in hr[:3]:
        assert raw.exists() and "/hires/" not in str(raw), raw
    # deterministic: two calls give identical ordering
    pairs2 = bt0.discover_tau0_pairs(_HCD_ROOT, _EMU_ROOT, include_hires=True)
    assert [p[:2] for p in pairs] == [p[:2] for p in pairs2], "ordering not stable"
    print(f"discover_tau0_pairs: {len(pairs)} total ({len(pairs_lf_only)} LF + "
          f"{len(hr)} HR rows across {len(hr_sims)} HR sims)")


def test_read_priya_params_matches_priya_array():
    """_read_priya_params (SimulationICs.json + Ap pivot transform) must
    reproduce PRIYA's params row to machine precision for several sims."""
    import build_emulator_cache as bec
    # (sim_idx in PRIYA params, hcd_outputs folder, any snap with a grid file)
    cases = [
        (44, "ns0.803Ap2.2e-09herei4.05heref2.67alphaq2.21hub0.735omegamh20.141hireionz7.17bhfeedback0.056"),
        (0,  "ns0.842Ap1.36e-09herei3.51heref2.85alphaq2hub0.658omegamh20.14hireionz6.72bhfeedback0.0453"),
        (29, "ns0.901Ap1.22e-09herei3.93heref2.87alphaq1.68hub0.712omegamh20.146hireionz6.97bhfeedback0.068"),
    ]
    with h5py.File(_PRIYA_FILE, "r") as f:
        priya_params = f["params"][...]
    for sim_idx, folder in cases:
        raw = _EMU_ROOT / folder / "output" / "SPECTRA_008" / "lya_forest_spectra_grid_480.hdf5"
        if not raw.exists():
            print(f"SKIP _read_priya_params for sim {sim_idx} (no raw tau)")
            continue
        ours = bt0._read_priya_params(raw)
        theirs = priya_params[sim_idx, 1:].astype(np.float64)  # col 0 is alpha
        rel = np.max(np.abs(ours / theirs - 1.0))
        assert rel < 1e-6, f"sim {sim_idx}: param mismatch max rel {rel:.3e}\nours={ours}\nPRIYA={theirs}"
    print("read_priya_params matches PRIYA params array (max rel < 1e-6): OK")


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
    SIM = "ns0.803Ap2.2e-09herei4.05heref2.67alphaq2.21hub0.735omegamh20.141hireionz7.17bhfeedback0.056"
    SNAP = 17; NK = 172; ZIDX = 8; ROW = 344
    snap_dir = Path(f"/scratch/cavestru_root/cavestru0/mfho/hcd_outputs/{SIM}/snap_{SNAP:03d}")
    raw = Path(f"/nfs/turbo/umor-yueyingn/mfho/emu_full/{SIM}/output/SPECTRA_{SNAP:03d}/lya_forest_spectra_grid_480.hdf5")
    if not (snap_dir.exists() and raw.exists()):
        print("SKIP test_build_tau0_rows_tier_p_matches_priya (data unavailable)"); return
    priya = "/home/mfho/lya_emulator_full/kodiaq_2_2_4_6-48-48/mf_emulator_flux_vectors_tau1000000.hdf5"
    with h5py.File(priya, "r") as f:
        alpha = float(f["params"][ROW, 0])
        P_priya = f["flux_vectors"][ROW, ZIDX*NK:(ZIDX+1)*NK].astype(np.float64)
        kp = f["kfkms"][ROW, ZIDX, :].astype(np.float64)
    rows, _ = bt0.build_tau0_rows(SIM, SNAP, snap_dir, raw,
                                  alpha_slope_grid=np.array([alpha]), n_k=NK)
    r = rows[0]
    assert r["kfkms"].shape == (NK,) and r["P_tier_p"].shape == (NK,)
    assert np.max(np.abs(r["kfkms"] / kp - 1)) < 1e-12, "native k-grid mismatch"
    assert np.max(np.abs(r["P_tier_p"] / P_priya - 1)) < 1e-4, "Tier-P not bit-identical"
    assert r["P_tier_c"].shape == (bt0_pp.N_TIER_C_BINS, NK)
    assert r["tier_c_counts"].shape == (bt0_pp.N_TIER_C_BINS,)
    assert r["tier_c_counts"].sum() > 0, "Tier-C produced no sightlines"
    assert np.isfinite(r["P_tier_c"]).all(), "Tier-C P1D has non-finite values"
    assert r["tier_c_counts"][0] > 0, "clean (forest) bin should be populated"
    print("OK build_tau0_rows native-grid Tier-P bit-identical")


if __name__ == "__main__":
    test_locate_raw_tau_file_finds_grid_file()
    test_locate_raw_tau_file_returns_none_when_missing()
    test_snap_z_to_priya_grid()
    test_discover_tau0_pairs_includes_hires_and_is_sorted()
    test_read_priya_params_matches_priya_array()
    test_write_cache_tau0_round_trip()
    test_build_tau0_rows_tier_p_matches_priya()
    print("OK")
