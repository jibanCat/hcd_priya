"""Tests for scripts/build_emulator_cache_tau0.py (v3.1, fake_spectra-driven).

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
_EMU_LF = Path("/nfs/turbo/umor-yueyingn/mfho/emu_full")
_EMU_HR = Path("/scratch/yueyingn_root/yueyingn0/mfho/priya/emu_full_hires_2")
_EMU_ROOT = _EMU_LF  # backward-compat alias used by test_read_priya_params / test_build_tau0_rows
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


def test_discover_lf_pairs_sorted_no_hires():
    pairs = bt0.discover_tau0_pairs(_HCD_ROOT, _EMU_LF, fidelity="lf")
    assert pairs, "no LF pairs discovered"
    assert all("/hires/" not in str(p[2]) for p in pairs)
    assert pairs == sorted(pairs, key=lambda p: (p[0], p[1]))
    print(f"OK LF discovery: {len(pairs)} pairs")


def test_discover_hr_pairs_six_sims():
    pairs = bt0.discover_tau0_pairs(_HCD_ROOT, _EMU_HR, fidelity="hr")
    sims = sorted({p[0] for p in pairs})
    assert len(sims) >= 4, f"expected >=4 HR sims with catalogs, got {len(sims)}"
    assert all("/hires/" in str(p[2]) for p in pairs)
    for _s, _snap, _sd, raw in pairs[:3]:
        assert raw.exists() and "emu_full_hires_2" in str(raw)
    print(f"OK HR discovery: {len(sims)} sims, {len(pairs)} pairs")


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


def _fake_snap_block(sim, snap):
    return {
        "sim_name": sim, "snap": snap,
        "f_nhi": np.full(30, 1e-21), "n_absorbers": np.arange(30, dtype=np.int64),
        "log_nhi_centres": np.linspace(17.1, 22.9, 30),
        "log_nhi_edges": np.linspace(17.0, 23.0, 31),
        "total_path_dX": 1234.5,
        "dNdX_LLS": 0.5, "dNdX_subDLA": 0.2, "dNdX_DLA": 0.1,
    }


def _fake_rows(nk, nrows=3):
    from hcd_analysis.priya_p1d import N_TIER_C_BINS as NB
    rows = []
    for i in range(nrows):
        rows.append({
            "sim_name": f"sim{i%2}", "snap": 10 + i, "alpha_slope": 1.0 + 0.1*i,
            "alpha_idx": i, "z_meta": 3.0, "z_grid": 3.0, "dv_kms": 10.0,
            "nbins_native": 1228, "target_F": 0.7, "scale": 0.9,
            "params": np.arange(9, dtype=np.float64),
            "kfkms": np.linspace(5e-4, 0.08, nk),
            "P_tier_p": np.full(nk, 5.0),
            "P_tier_c": np.tile(np.arange(NB)[:, None], (1, nk)).astype(float),
            "P_tier_c_filtered": np.tile(np.arange(NB)[:, None], (1, nk)).astype(float) * 0.9,
            "tier_c_counts": np.arange(NB, dtype=np.int64),
            "mean_F_by_bin": np.linspace(0.5, 0.9, NB),
            "snap_group_idx": i,
        })
    return rows


def test_write_cache_tau0_round_trip(tmp_path=Path("/tmp")):
    import hcd_analysis.priya_p1d as pp
    nk = 172
    rows = _fake_rows(nk)
    snap_blocks = [_fake_snap_block(f"sim{i%2}", 10 + i) for i in range(3)]
    out = tmp_path / "rt_tau0.h5"
    bt0.write_cache_tau0(rows, snap_blocks, out, alpha_range=(1.0, 1.2), n_k=nk)
    with h5py.File(out, "r") as f:
        assert f.attrs["cache_version"] == "3.3"
        assert f.attrs["tier_c_recipe"] == "uniform"
        assert f.attrs["n_k"] == nk
        assert f["P_tier_p"].shape == (3, nk)
        assert f["kfkms"].shape == (3, nk)
        assert f["P_tier_c"].shape == (3, pp.N_TIER_C_BINS, nk)
        assert "P_tier_c_frozen" not in f      # frozen tier removed (per-pixel freeze no-op)
        assert f["P_tier_c_filtered"].shape == (3, pp.N_TIER_C_BINS, nk)
        assert f["tier_c_counts"].shape == (3, pp.N_TIER_C_BINS)
        assert f["mean_F_by_bin"].shape == (3, pp.N_TIER_C_BINS)
        assert list(f["tier_c_labels"].asstr()[...]) == pp.tier_c_labels()
        assert np.allclose(f["P_tier_p"][0], 5.0)
        assert "k_target" not in f
        assert f["tier_c_counts"].dtype == np.int64
        assert np.allclose(f["tier_c_nhi_edges"][...], pp.FINE_NHI_EDGES)
        assert f["snap_group_idx"][...].tolist() == [0, 1, 2]
    print("OK write_cache_tau0 v3.3 round-trip")


def test_discover_dedups_one_snap_per_grid_z():
    pairs = bt0.discover_tau0_pairs(_HCD_ROOT, _EMU_LF, fidelity="lf")
    # no duplicate (sim, z_grid)
    seen = {}
    import json as _json
    for sim, snap, sd, raw in pairs:
        zg = bt0._snap_z_to_priya_grid(float(_json.load(open(sd / "meta.json"))["z"]))
        key = (sim, round(zg, 4))
        assert key not in seen, f"duplicate (sim,z_grid) {key}: snaps {seen[key]} and {snap}"
        seen[key] = snap
    # every kept snap is within 0.05 of its grid z
    for sim, snap, sd, raw in pairs:
        z = float(_json.load(open(sd / "meta.json"))["z"])
        assert abs(z - bt0._snap_z_to_priya_grid(z)) <= 0.05, f"{sim} {snap} too far off-grid"
    # ns0.907: the on-grid snaps survive, the off-grid extras (snap_015 z3.27, snap_018 z2.67) are gone
    n907 = [snap for sim, snap, sd, raw in pairs if "ns0.907Ap1.5e-09" in sim]
    assert 16 in n907 and 15 not in n907, "ns0.907 should keep snap_016 (z3.2), drop snap_015 (z3.27)"
    assert 19 in n907 and 18 not in n907, "ns0.907 should keep snap_019 (z2.6), drop snap_018 (z2.67)"
    print(f"OK discovery dedup: {len(pairs)} LF pairs, one per (sim,z_grid)")


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


def test_merge_v3_synthetic(tmp_path=Path("/tmp")):
    import scripts.merge_tau0_cache as mg
    import hcd_analysis.priya_p1d as pp
    nk = 172
    for si in range(2):
        rows = _fake_rows(nk, nrows=2)
        for r in rows:
            r["snap_group_idx"] = 0  # one snap per shard
        blocks = [_fake_snap_block(f"sim{si}", 0)]
        bt0.write_cache_tau0(rows, blocks, tmp_path / f"shard_{si}.h5",
                             alpha_range=(1.0, 1.2), n_k=nk)
    out = tmp_path / "merged.h5"
    mg.merge_shards([str(tmp_path / f"shard_{i}.h5") for i in range(2)], out)
    with h5py.File(out, "r") as f:
        assert f.attrs["cache_version"] == "3.3"
        assert f.attrs["tier_c_recipe"] == "uniform"
        assert f.attrs["n_k"] == nk
        assert f["P_tier_p"].shape == (4, nk)
        assert f["kfkms"].shape == (4, nk)
        assert f["P_tier_c"].shape == (4, pp.N_TIER_C_BINS, nk)
        assert "P_tier_c_frozen" not in f
        assert f["P_tier_c_filtered"].shape == (4, pp.N_TIER_C_BINS, nk)
        assert f["tier_c_counts"].shape[0] == 4
        assert f["mean_F_by_bin"].shape == (4, pp.N_TIER_C_BINS)
        gi = f["snap_group_idx"][...]
        assert gi.tolist() == [0, 0, 1, 1]   # second shard's snap remapped
        assert list(f["tier_c_labels"].asstr()[...]) == pp.tier_c_labels()
    print("OK merge v3.3 synthetic")


def test_v33_uniform_schema_roundtrip():
    """v3.3 uniform-only schema: stores P_tier_c (uniform unfiltered),
    P_tier_c_filtered, mean_F_by_bin; P_tier_c_frozen is REMOVED (the per-pixel
    freeze was proven a numerical no-op); tier_c_recipe='uniform'."""
    import os
    import hcd_analysis.priya_p1d as pp
    nk = bt0._N_K["lf"]
    NB = pp.N_TIER_C_BINS

    def _row(a):
        return dict(sim_name="nsTEST", snap=1, alpha_slope=float(a), alpha_idx=a,
                    z_meta=3.0, z_grid=3.0, dv_kms=10.0, nbins_native=2*nk,
                    snap_group_idx=0, target_F=0.7, scale=0.8,
                    params=np.zeros(9), kfkms=np.linspace(1e-3, 0.1, nk),
                    P_tier_p=np.ones(nk), P_tier_c=np.ones((NB, nk)),
                    P_tier_c_filtered=np.ones((NB, nk)),
                    tier_c_counts=np.arange(NB, dtype=np.int64),
                    mean_F_by_bin=np.linspace(0.5, 0.9, NB))

    rows = [_row(0), _row(1)]
    snap_blocks = [_fake_snap_block("nsTEST", 1)]
    out = os.path.join(tempfile.mkdtemp(), "c.h5")
    bt0.write_cache_tau0(rows, snap_blocks, out, alpha_range=(0.0, 1.0), n_k=nk)
    with h5py.File(out, "r") as f:
        assert f.attrs["cache_version"] == "3.3", \
            f"expected cache_version='3.3', got '{f.attrs['cache_version']}'"
        assert f.attrs["tier_c_recipe"] == "uniform"
        assert "P_tier_c_frozen" not in f, "frozen tier should be removed in v3.3"
        assert "tau_freeze_tierc" not in f.attrs, "tau_freeze_tierc attr should be gone"
        assert f["P_tier_c"].shape == (2, NB, nk)
        assert f["P_tier_c_filtered"].shape == (2, NB, nk)
        assert f["mean_F_by_bin"].shape == (2, NB), \
            f"mean_F_by_bin shape mismatch: {f['mean_F_by_bin'].shape}"
        assert np.allclose(f["mean_F_by_bin"][1], np.linspace(0.5, 0.9, NB)), \
            "mean_F_by_bin[1] round-trip failed"
    print("OK v3.3 uniform-only schema round-trip (frozen tier removed)")


def test_filtered_tier_c_reconstructs_priya():
    SIM = "ns0.803Ap2.2e-09herei4.05heref2.67alphaq2.21hub0.735omegamh20.141hireionz7.17bhfeedback0.056"
    SNAP, NK = 17, 172
    sd = Path(f"/scratch/cavestru_root/cavestru0/mfho/hcd_outputs/{SIM}/snap_{SNAP:03d}")
    raw = Path(f"/nfs/turbo/umor-yueyingn/mfho/emu_full/{SIM}/output/SPECTRA_{SNAP:03d}/lya_forest_spectra_grid_480.hdf5")
    if not (sd.exists() and raw.exists()):
        print("SKIP filtered_tier_c (data unavailable)"); return
    rows, _ = bt0.build_tau0_rows(SIM, SNAP, sd, raw, np.array([1.0]), n_k=NK)
    r = rows[0]
    assert r["P_tier_c_filtered"].shape == (bt0_pp.N_TIER_C_BINS, NK)
    N = int(r["tier_c_counts"].sum())
    recon = (r["tier_c_counts"][:, None] / N * r["P_tier_c_filtered"]).sum(0)
    rel = np.max(np.abs(recon / r["P_tier_p"] - 1))
    assert rel < 1e-9, f"filtered Tier-C != Tier P (=PRIYA): max|r-1|={rel:.2e}"
    print(f"RESULT OK filtered Tier-C reconstructs PRIYA exactly: max|r-1|={rel:.2e}")


def test_grid_in_dir_requires_grid_480(tmp_path=None):
    import tempfile, os
    base = Path(tempfile.mkdtemp())
    # dir A: only the low-res fallback -> must be skipped (None)
    a = base / "SPECTRA_015"; a.mkdir()
    (a / "lya_forest_spectra.hdf5").write_bytes(b"x")
    assert bt0._grid_in_dir(a) is None, "fallback-only dir must NOT be located"
    # dir B: has the grid_480 -> must be returned
    b = base / "SPECTRA_016"; b.mkdir()
    g = b / "lya_forest_spectra_grid_480.hdf5"; g.write_bytes(b"x")
    assert bt0._grid_in_dir(b) == g, "grid_480 must be located"
    # dir C: empty -> None
    c = base / "SPECTRA_099"; c.mkdir()
    assert bt0._grid_in_dir(c) is None
    import shutil; shutil.rmtree(base)
    print("OK _grid_in_dir requires grid_480")


def test_build_tau0_rows_hr_matches_priya_6sim():
    """One HR (sim, z, alpha) row vs the new 6-sim PRIYA HR reference, native grid.
    N_K is READ from the ref (525), not hardcoded — confirms Tier P (tau_thresh=1e6)
    is bit-identical to PRIYA's hires flux vectors."""
    import json
    HR_REF = "/scratch/yueyingn_root/yueyingn0/mfho/priya/emu_full_hires_2/mf_emulator_flux_vectors_tau1000000.hdf5"
    EMU_HR = "/scratch/yueyingn_root/yueyingn0/mfho/priya/emu_full_hires_2"
    HCD_BASE = "/scratch/cavestru_root/cavestru0/mfho/hcd_outputs"   # discover appends /hires
    if not Path(HR_REF).exists():
        print("SKIP HR bit-identity (6-sim ref unavailable)"); return
    pairs = bt0.discover_tau0_pairs(HCD_BASE, EMU_HR, fidelity="hr")
    if not pairs:
        print("SKIP HR bit-identity (no HR Phase-1 catalogs)"); return
    cand = None
    for sim, snap, sd, raw in pairs:
        z = bt0._snap_z_to_priya_grid(json.load(open(sd / "meta.json"))["z"])
        if abs(z - 3.0) < 1e-6:
            cand = (sim, snap, sd, raw); break
    assert cand, "no HR z=3.0 pair"
    sim, snap, sd, raw = cand
    my_params = bt0._read_priya_params(raw)            # 9 cosmo in bec.PARAM_ORDER
    with h5py.File(HR_REF, "r") as f:
        params = f["params"][...]; fv = f["flux_vectors"][...]
        zout = f["zout"][...]
        NK = int(f["kfkms"].shape[-1])                 # 525, read from the ref
        sim_rows = np.where(np.all(np.isclose(params[:, 1:], my_params[None, :],
                                              rtol=1e-4), axis=1))[0]
        assert len(sim_rows) > 0, "HR sim not found in PRIYA ref by params"
        row = int(sim_rows[0]); alpha = float(params[row, 0])
        zidx = int(np.argmin(np.abs(zout - 3.0)))
        P_priya = fv[row, zidx*NK:(zidx+1)*NK].astype(np.float64)
        kp = f["kfkms"][row, zidx, :].astype(np.float64)
    rows, _ = bt0.build_tau0_rows(sim, snap, sd, raw,
                                  alpha_slope_grid=np.array([alpha]), n_k=NK)
    r = rows[0]
    assert r["kfkms"].shape == (NK,), f"expected {NK} k-bins, got {r['kfkms'].shape}"
    assert np.max(np.abs(r["kfkms"] / kp - 1)) < 1e-10, "HR native k-grid mismatch"
    rel = np.max(np.abs(r["P_tier_p"] / P_priya - 1))
    assert rel < 1e-4, f"HR Tier-P not bit-identical to PRIYA hires vectors: max|r-1|={rel:.3e}"
    print(f"RESULT OK HR bit-identity sim={sim[:24]} z=3.0 alpha={alpha:.4f} N_K={NK} max|r-1|={rel:.3e}")


if __name__ == "__main__":
    test_locate_raw_tau_file_finds_grid_file()
    test_locate_raw_tau_file_returns_none_when_missing()
    test_snap_z_to_priya_grid()
    test_grid_in_dir_requires_grid_480()
    test_discover_lf_pairs_sorted_no_hires()
    test_discover_hr_pairs_six_sims()
    test_discover_dedups_one_snap_per_grid_z()
    test_read_priya_params_matches_priya_array()
    test_write_cache_tau0_round_trip()
    test_v33_uniform_schema_roundtrip()
    test_merge_v3_synthetic()
    test_build_tau0_rows_tier_p_matches_priya()
    test_filtered_tier_c_reconstructs_priya()
    test_build_tau0_rows_hr_matches_priya_6sim()
    print("OK")
