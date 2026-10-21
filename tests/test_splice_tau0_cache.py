"""Cache splicing and the historical-vs-repaired delta audit (S4/F8 repair; spec hcd_priya_notes
docs/superpowers/emulator-debug-2026-10/gateB/S4_REPAIR_SPEC.md sections 3-4). Synthetic caches with the production key
layout (the merge script's key lists); no fake_spectra."""
import json
import sys

import h5py
import numpy as np
import pytest

from hcd_analysis.paths import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import merge_tau0_cache as mg  # noqa: E402
import splice_tau0_cache as sp  # noqa: E402

K, NB, NH = 6, 15, 30


def _write(path, groups, *, edges_shift=0.0, attrs=None):
    """groups: list of (sim, snap, z_grid, n_alpha, seed). Rows grouped by group in the given order."""
    rows, blocks = [], []
    for gi, (sim, snap, zg, na, seed) in enumerate(groups):
        rng = np.random.default_rng(seed)
        blocks.append(dict(snap_sim_name=sim, snap_snap=snap, snap_total_path_dX=rng.random(),
                           snap_dNdX_LLS=rng.random(), snap_dNdX_subDLA=rng.random(), snap_dNdX_DLA=rng.random(),
                           snap_f_nhi=rng.random(NH), snap_n_absorbers=rng.integers(0, 9, NH)))
        for a in range(na):
            rows.append(dict(sim_name=sim, params=rng.random(9), alpha_slope=0.6 + 0.1 * a, target_F=rng.random(),
                             scale=rng.random(), z_meta=zg + 1e-6, z_grid=zg, dv_kms=10.0 + rng.random(),
                             snap=snap, alpha_idx=a, nbins_native=1400, snap_group_idx=gi,
                             kfkms=rng.random(K), P_tier_p=rng.random(K), P_tier_c=rng.random((NB, K)),
                             P_tier_c_filtered=rng.random((NB, K)), tier_c_counts=rng.integers(0, 99, NB),
                             mean_F_by_bin=rng.random(NB)))
    with h5py.File(path, "w") as f:
        for k, v in dict(created_utc="t0", git_sha="abc", n_rows=len(rows), n_snaps=len(blocks), cache_version="3.3",
                         n_k=K, tau_thresh=1e6, tier_c_recipe="uniform", alpha_range=np.array([0.6, 0.6 + 0.1 * 3]),
                         merged_from_n_shards=3, k_convention="angular", **(attrs or {})).items():
            f.attrs[k] = v
        f.create_dataset("tier_c_labels", data=np.array([f"c{i}" for i in range(NB)], dtype=h5py.string_dtype()))
        f.create_dataset("tier_c_nhi_edges", data=np.linspace(17, 22, NB - 1))
        f.create_dataset("param_names", data=np.array([f"p{i}" for i in range(9)], dtype=h5py.string_dtype()))
        f.create_dataset("log_nhi_centres", data=np.linspace(17.1, 22.9, NH) + edges_shift)
        f.create_dataset("log_nhi_edges", data=np.linspace(17, 23, NH + 1))
        f.create_dataset("sim_name", data=np.array([r["sim_name"] for r in rows], dtype=h5py.string_dtype()))
        for k in mg._ROW_ARR:
            f.create_dataset(k, data=np.stack([r[k] for r in rows]))
        for k in mg._ROW_FLOAT:
            f.create_dataset(k, data=np.array([r[k] for r in rows], dtype=np.float64))
        for k in mg._ROW_INT:
            f.create_dataset(k, data=np.array([r[k] for r in rows], dtype=np.int32))
        f.create_dataset("snap_sim_name", data=np.array([b["snap_sim_name"] for b in blocks], dtype=h5py.string_dtype()))
        f.create_dataset("snap_snap", data=np.array([b["snap_snap"] for b in blocks], dtype=np.int32))
        for k in mg._SNAP_FLOAT:
            f.create_dataset(k, data=np.array([b[k] for b in blocks], dtype=np.float64))
        for k in mg._SNAP_2D:
            f.create_dataset(k, data=np.stack([b[k] for b in blocks]))
    return path


BASE = [("simA", 16, 3.2, 3, 1), ("simA", 17, 2.8, 3, 2), ("simA", 19, 2.6, 3, 3), ("simB", 5, 4.0, 3, 4)]
SHARD = [("simA", 17, 3.0, 3, 20), ("simA", 18, 2.8, 3, 21)]


def _splice(tmp_path, base=BASE, shard=SHARD, remove=(("simA", 17, 2.8),), **kw):
    b = _write(tmp_path / "base.h5", base)
    s = _write(tmp_path / "shard.h5", shard)
    out = tmp_path / "out.h5"
    sp.splice_cache(b, [s], list(remove), out, record={"spec": "test"}, **kw)
    return b, s, out


def _read(path):
    with h5py.File(path, "r") as f:
        d = {k: f[k][...] for k in f.keys()}
        return d, dict(f.attrs)


def test_splice_replaces_and_inserts_in_canonical_order(tmp_path):
    b, s, out = _splice(tmp_path)
    d, a = _read(out)
    ids = [(x.decode(), int(n)) for x, n in zip(d["snap_sim_name"], d["snap_snap"])]
    assert ids == [("simA", 16), ("simA", 17), ("simA", 18), ("simA", 19), ("simB", 5)]
    assert a["n_rows"] == 15 and a["n_snaps"] == 5 and "merged_from_n_shards" not in a
    rec = json.loads(a["repair_record"])
    assert rec["spec"] == "test" and rec["removed"] == [["simA", 17, 2.8]] and len(rec["base_sha256"]) == 64
    assert rec["added"] == [["simA", 17, 3.0], ["simA", 18, 2.8]]
    sim, snap = d["sim_name"].astype(str), d["snap"]
    for r in range(sim.size):                                  # every row points at its own block
        g = d["snap_group_idx"][r]
        assert (sim[r], int(snap[r])) == (d["snap_sim_name"][g].decode(), int(d["snap_snap"][g]))
    zg = {(s_, int(n)): z for s_, n, z in zip(sim, snap, d["z_grid"])}
    assert zg[("simA", 17)] == 3.0 and zg[("simA", 18)] == 2.8


def test_splice_keeps_untouched_groups_bit_identical_and_the_key_set(tmp_path):
    b, s, out = _splice(tmp_path)
    rep = sp.cache_delta(b, out)
    assert rep["key_set_equal"] and rep["static_equal"] and rep["dtypes_equal"]
    # row identity = (sim, z_grid, alpha_idx): the z = 2.8 rows exist on both sides (rebuilt), the z = 3.0 rows are new
    assert rep["rows_only_hist"] == []
    assert rep["rows_only_new"] == [("simA", 3.0, a) for a in range(3)]
    assert set(rep["rows_changed"]) == {("simA", 2.8, a) for a in range(3)}
    assert "snap" in rep["rows_changed"][("simA", 2.8, 0)] and "P_tier_c" in rep["rows_changed"][("simA", 2.8, 0)]
    untouched = [i for i in rep["row_identities_common"] if i[0] == "simB" or i[1] in (3.2, 2.6)]
    assert len(untouched) == 9 and all(i not in rep["rows_changed"] for i in untouched)
    # block identity = (sim, snap): (simA, 17) now holds the shard's block, (simA, 18) is new
    assert rep["blocks_only_new"] == [("simA", 18)] and rep["blocks_only_hist"] == []
    assert set(rep["blocks_changed"]) == {("simA", 17)}


def test_cache_delta_catches_any_change_outside_the_footprint(tmp_path):
    b, s, out = _splice(tmp_path)
    with h5py.File(out, "r+") as f:                               # perturb one untouched row's P_tier_c by 1e-9
        i = int(np.where((f["sim_name"].asstr()[...] == "simB"))[0][1])
        f["P_tier_c"][i, 3, 2] = f["P_tier_c"][i, 3, 2] * (1 + 1e-9)
    rep = sp.cache_delta(b, out)
    assert ("simB", 4.0, 1) in rep["rows_changed"] and "P_tier_c" in rep["rows_changed"][("simB", 4.0, 1)]


def test_cache_delta_catches_block_and_static_changes(tmp_path):
    b, s, out = _splice(tmp_path)
    with h5py.File(out, "r+") as f:
        f["snap_dNdX_DLA"][0] = f["snap_dNdX_DLA"][0] + 1e-12
        f["log_nhi_edges"][3] = f["log_nhi_edges"][3] + 1e-12
    rep = sp.cache_delta(b, out)
    assert ("simA", 16) in rep["blocks_changed"] and not rep["static_equal"]


def test_splice_refuses_a_group_present_in_both(tmp_path):
    with pytest.raises(ValueError, match="already present"):
        _splice(tmp_path, remove=())


def test_splice_refuses_unknown_removal(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        _splice(tmp_path, remove=(("simA", 17, 2.8), ("simC", 1, 2.2)))


def test_splice_refuses_mismatched_static_tables(tmp_path):
    b = _write(tmp_path / "base.h5", BASE)
    s = _write(tmp_path / "shard.h5", SHARD, edges_shift=1e-9)
    with pytest.raises(ValueError, match="log_nhi_centres"):
        sp.splice_cache(b, [s], [("simA", 17, 2.8)], tmp_path / "out.h5", record={})


def test_splice_refuses_a_base_not_in_canonical_order(tmp_path):
    with pytest.raises(ValueError, match="canonical"):
        _splice(tmp_path, base=[BASE[1], BASE[0], BASE[2], BASE[3]])


def test_splice_refuses_to_overwrite_its_inputs(tmp_path):
    b = _write(tmp_path / "base.h5", BASE)
    s = _write(tmp_path / "shard.h5", SHARD)
    with pytest.raises(ValueError, match="overwrite"):
        sp.splice_cache(b, [s], [("simA", 17, 2.8)], b, record={})
