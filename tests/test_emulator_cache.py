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


if __name__ == "__main__":
    test_discover_sim_snap_pairs_returns_nonempty()
    test_per_file_readers_on_first_pair()
    print("OK")
