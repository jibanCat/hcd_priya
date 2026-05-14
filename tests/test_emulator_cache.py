"""Tests for scripts/build_emulator_cache.py.

Run with: python3 tests/test_emulator_cache.py
"""
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

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


def test_per_file_readers_on_first_pair():
    pairs = bec.discover_sim_snap_pairs(HCD_ROOT)
    _, _, snap_dir = pairs[0]

    meta = bec.read_meta(snap_dir)
    assert "z" in meta and "dv_kms" in meta and "nbins" in meta and "n_skewers" in meta
    assert meta["nbins"] > 0 and 0 < meta["z"] < 10

    cddf = bec.read_cddf(snap_dir)
    for key in ("log_nhi_centres", "log_nhi_edges", "f_nhi", "n_absorbers", "total_path"):
        assert key in cddf, f"missing {key} in cddf_corrected.npz"
    assert cddf["log_nhi_centres"].shape == (30,)
    assert cddf["log_nhi_edges"].shape == (31,)
    assert cddf["f_nhi"].shape == (30,)
    assert cddf["n_absorbers"].shape == (30,)

    p1d = bec.read_p1d_per_class(snap_dir)
    for key in ("k", "P_clean", "P_LLS_only", "P_subDLA_only", "P_DLA_only",
                "mean_F_clean", "mean_F_LLS", "mean_F_subDLA", "mean_F_DLA",
                "n_sightlines_clean", "n_sightlines_LLS", "n_sightlines_subDLA",
                "n_sightlines_DLA", "n_total"):
        assert key in p1d, f"missing {key} in p1d_per_class.h5"
    assert p1d["k"].ndim == 1
    assert p1d["P_clean"].shape == p1d["k"].shape
    print(f"readers: meta z={meta['z']:.3f} nbins={meta['nbins']} | "
          f"cddf bins={len(cddf['log_nhi_centres'])} | "
          f"p1d k.shape={p1d['k'].shape}")


def test_interp_p1d_loglog_pure_power_law_recovers_input():
    # P(k) = A k^n on a fine source grid; log-log interpolation onto a
    # coarse subgrid should reproduce A k^n exactly (log-log linear interp
    # is exact for power laws).
    k_src = np.geomspace(1e-3, 5e-2, 200)
    A, n = 7.3, -1.4
    P_src = A * k_src**n

    k_target = np.geomspace(2e-3, 3e-2, 25)
    P_interp = bec.interp_p1d_loglog(k_src, P_src, k_target)
    assert np.allclose(P_interp, A * k_target**n, rtol=1e-10)


def test_interp_p1d_loglog_out_of_range_is_nan():
    k_src = np.geomspace(1e-3, 2e-2, 100)
    P_src = np.ones_like(k_src)
    k_target = np.array([5e-4, 1e-2, 5e-2])  # below, inside, above source range
    P_interp = bec.interp_p1d_loglog(k_src, P_src, k_target)
    assert np.isnan(P_interp[0])
    assert P_interp[1] == 1.0
    assert np.isnan(P_interp[2])


def test_dndx_per_class_matches_manual_sum_on_first_pair():
    pairs = bec.discover_sim_snap_pairs(HCD_ROOT)
    _, _, snap_dir = pairs[0]
    cddf = bec.read_cddf(snap_dir)

    result = bec.compute_dndx_per_class(cddf)

    centres = cddf["log_nhi_centres"]
    n_abs = cddf["n_absorbers"]
    total_path = float(cddf["total_path"])
    lls_mask = (centres >= 17.2) & (centres < 19.0)
    sub_mask = (centres >= 19.0) & (centres < 20.3)
    dla_mask = (centres >= 20.3)
    ref = {
        "dNdX_LLS":    float(n_abs[lls_mask].sum() / total_path),
        "dNdX_subDLA": float(n_abs[sub_mask].sum() / total_path),
        "dNdX_DLA":    float(n_abs[dla_mask].sum() / total_path),
    }
    for k in ref:
        assert np.isclose(result[k], ref[k], rtol=1e-12), \
            f"{k}: got {result[k]}, expected {ref[k]}"
    print(f"dNdX: LLS={result['dNdX_LLS']:.4f}  "
          f"subDLA={result['dNdX_subDLA']:.4f}  DLA={result['dNdX_DLA']:.4f}")


def test_build_row_schema_first_pair():
    from hcd_analysis.p1d import _DEFAULT_K_BINS
    pairs = bec.discover_sim_snap_pairs(HCD_ROOT)
    sim_name, snap, snap_dir = pairs[0]

    # Cache uses angular k throughout; convert here too.
    row = bec.build_row(sim_name, snap, snap_dir, 2.0 * np.pi * _DEFAULT_K_BINS)

    # scalars
    assert row["sim_name"] == sim_name
    assert row["snap"] == snap
    assert row["params"].shape == (9,)
    assert 0 < row["z"] < 10
    # P1D arrays match k_target shape
    for key in ("P_clean", "P_LLS_only", "P_subDLA_only", "P_DLA_only"):
        assert row[key].shape == (50,)
    # CDDF arrays
    assert row["f_nhi"].shape == (30,)
    assert row["n_absorbers"].shape == (30,)
    # dN/dX scalars
    for key in ("dNdX_LLS", "dNdX_subDLA", "dNdX_DLA"):
        assert key in row and np.isfinite(row[key])
    print(f"build_row: z={row['z']:.3f}  "
          f"P_clean finite frac = {np.isfinite(row['P_clean']).mean():.2f}")


def test_write_cache_two_row_round_trip():
    from hcd_analysis.p1d import _DEFAULT_K_BINS
    pairs = bec.discover_sim_snap_pairs(HCD_ROOT)[:2]
    k_target = 2.0 * np.pi * _DEFAULT_K_BINS  # angular, project convention
    rows = [bec.build_row(s, sn, sd, k_target) for s, sn, sd in pairs]

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "observables.h5"
        bec.write_cache(rows, k_target, out)
        assert out.exists()

        with h5py.File(out, "r") as f:
            assert f["params"].shape == (2, 9)
            assert f["P_clean"].shape == (2, 50)
            assert f["k_target"].shape == (50,)
            assert list(f["param_names"][...].astype(str)) == list(bec.PARAM_ORDER)
            assert float(f["z"][0]) == rows[0]["z"]
            assert np.array_equal(f["params"][0], rows[0]["params"])
            assert np.allclose(f["P_clean"][0], rows[0]["P_clean"], equal_nan=True)
            assert "git_sha" in f.attrs
    print("write_cache: 2-row round-trip OK")


def test_verify_round_trip_against_source_first_pair():
    from hcd_analysis.p1d import _DEFAULT_K_BINS
    pairs = bec.discover_sim_snap_pairs(HCD_ROOT)[:1]
    k_target = 2.0 * np.pi * _DEFAULT_K_BINS  # angular, project convention
    rows = [bec.build_row(s, sn, sd, k_target) for s, sn, sd in pairs]

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "observables.h5"
        bec.write_cache(rows, k_target, out)
        bec.verify_round_trip(out, sim=pairs[0][0], snap=pairs[0][1], root=HCD_ROOT)
    print("verify_round_trip: OK")


if __name__ == "__main__":
    test_discover_sim_snap_pairs_returns_nonempty()
    test_per_file_readers_on_first_pair()
    test_interp_p1d_loglog_pure_power_law_recovers_input()
    test_interp_p1d_loglog_out_of_range_is_nan()
    test_dndx_per_class_matches_manual_sum_on_first_pair()
    test_build_row_schema_first_pair()
    test_write_cache_two_row_round_trip()
    test_verify_round_trip_against_source_first_pair()
    print("OK")
