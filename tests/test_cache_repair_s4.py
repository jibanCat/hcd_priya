"""S4/F8 repair: historical vs repaired LF cache, the PRE-SPECIFIED expected delta (hcd_priya_notes
docs/superpowers/emulator-debug-2026-10/gateB/S4_REPAIR_SPEC.md section 4, recorded in PU-0041 before any build).

observables_tau0_lf_hist.h5 = the historical cache (hashed campaign input, untouched); observables_tau0_lf.h5 = the
repaired cache (the production LF cache from the repair on). Any difference outside the footprint fails here."""
import json
import sys

import h5py
import numpy as np
import pytest

from hcd_analysis.paths import REPO_ROOT
from tests.gate_helpers import real_cache_path, require_real_cache

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import splice_tau0_cache as sp  # noqa: E402

NS0907 = "ns0.907Ap1.5e-09herei3.75heref2.77alphaq2.04hub0.662omegamh20.144hireionz7.47bhfeedback0.0347"
HIST, NEW = real_cache_path("lf_hist"), real_cache_path("lf")
ALLOWED_ROW_CHANGES = {"snap", "z_meta", "dv_kms", "kfkms", "P_tier_p", "P_tier_c", "P_tier_c_filtered",
                       "tier_c_counts", "mean_F_by_bin", "snap_group_idx", "target_F", "scale"}
ALLOWED_ATTR_CHANGES = {"created_utc", "git_sha", "n_rows", "n_snaps", "merged_from_n_shards", "repair_record"}


@pytest.fixture(scope="module")
def delta():
    require_real_cache(HIST)
    require_real_cache(NEW)
    return sp.cache_delta(HIST, NEW)


def _rows(path, sim, zg):
    with h5py.File(path, "r") as f:
        s, z, a = f["sim_name"].asstr()[...], f["z_grid"][...], f["alpha_idx"][...]
        idx = np.where((s == sim) & np.isclose(z, zg, atol=1e-9))[0]
        idx = idx[np.argsort(a[idx])]
        return {k: f[k][...][idx] for k in ("snap", "z_meta", "dv_kms", "nbins_native", "kfkms", "P_tier_p",
                                             "P_tier_c_filtered", "tier_c_counts", "target_F", "scale", "params",
                                             "alpha_slope", "alpha_idx")}


def test_layout_static_tables_and_attributes(delta):
    assert delta["key_set_equal"] and delta["dtypes_equal"] and delta["static_equal"]
    assert set(delta["attrs_changed"]) <= ALLOWED_ATTR_CHANGES, delta["attrs_changed"]


def test_rows_outside_the_footprint_are_identical(delta):
    assert delta["rows_only_hist"] == []
    assert delta["rows_only_new"] == [(NS0907, 3.0, a) for a in range(20)]
    assert set(delta["rows_changed"]) == {(NS0907, 2.8, a) for a in range(20)}
    assert len(delta["row_identities_common"]) == 21440 and 21440 - len(delta["rows_changed"]) == 21420


def test_footprint_rows_change_only_in_the_expected_keys(delta):
    for ident, diff in delta["rows_changed"].items():
        assert set(diff) <= ALLOWED_ROW_CHANGES, (ident, sorted(set(diff) - ALLOWED_ROW_CHANGES))
        for k in ("target_F", "scale"):                   # same tau, same target: identical up to rounding
            assert k not in diff or diff[k] <= 1e-12, (ident, k, diff[k])
        assert diff.get("snap_group_idx", "").endswith(f"({NS0907!r}, 18)"), diff.get("snap_group_idx")


def test_snapshot_blocks_outside_the_footprint_are_identical(delta):
    assert delta["blocks_only_hist"] == []
    assert delta["blocks_only_new"] == [(NS0907, 18)]
    assert delta["blocks_changed"] == {}                  # (ns0.907, 17) is the same directory's CDDF, now on z = 3.0


def test_rebuilt_z28_rows_relations():
    h, n = _rows(HIST, NS0907, 2.8), _rows(NEW, NS0907, 2.8)
    assert len(h["snap"]) == len(n["snap"]) == 20
    assert set(h["snap"]) == {17} and set(n["snap"]) == {18}
    assert np.allclose(h["z_meta"], 2.799998) and np.allclose(n["z_meta"], 2.8, atol=1e-12)
    for k in ("params", "alpha_slope", "alpha_idx", "nbins_native"):
        assert np.array_equal(h[k], n[k]), k
    r = h["dv_kms"] / n["dv_kms"]                          # vmax_hist / vmax_new (same nbins)
    assert np.allclose(r, r[0], rtol=0, atol=0) and abs(r[0] - 1) > 9e-5
    assert np.allclose(n["kfkms"] / h["kfkms"], r[0], rtol=1e-10, atol=0)
    assert np.allclose(n["P_tier_p"] / h["P_tier_p"], 1 / r[0], rtol=1e-10, atol=0)
    assert not np.array_equal(h["tier_c_counts"], n["tier_c_counts"])
    for c in (h, n):                                       # Tier P = count-weighted sum of the filtered pieces
        w = c["tier_c_counts"] / c["tier_c_counts"].sum(1, keepdims=True)
        assert np.allclose(np.einsum("rc,rck->rk", w, c["P_tier_c_filtered"]), c["P_tier_p"], rtol=1e-10, atol=0)


def test_restored_z30_rows_and_their_block():
    n = _rows(NEW, NS0907, 3.0)
    assert len(n["snap"]) == 20 and set(n["snap"]) == {17} and np.allclose(n["z_meta"], 3.0, atol=1e-12)
    w = n["tier_c_counts"] / n["tier_c_counts"].sum(1, keepdims=True)
    assert np.allclose(np.einsum("rc,rck->rk", w, n["P_tier_c_filtered"]), n["P_tier_p"], rtol=1e-10, atol=0)


SHARDS = "/nfs/turbo/umor-yueyingn/mfho/hcd/emulator_v2/caches/s4_build_shards"
# Built with 2 threads, as the historical production build (PU-0044): fake_spectra's mean-flux scale solve is
# thread-count dependent at ~1e-9 (the first, 4-thread shards shard_03f03b8_task{2..5}.h5 differ by that much in scale
# and the keys computed with it; kept as the recorded evidence).
REPRO = {4: ("ns0.803Ap2.2e-09herei4.05heref2.67alphaq2.21hub0.735omegamh20.141hireionz7.17bhfeedback0.056", 17),
         5: ("ns0.959Ap2.34e-09herei3.81heref2.99alphaq1.77hub0.725omegamh20.144hireionz6.83bhfeedback0.0467", 20)}


@pytest.mark.parametrize("task", sorted(REPRO))
def test_reproducibility_groups_rebuilt_today_equal_the_historical_rows(task):
    """Pre-specified (PU-0041): groups the repair must not change, rebuilt from source with the repaired builder, equal
    the historical cache to <= 1e-12 on every float key and exactly on every int/string key; their snapshot block
    equals the historical block exactly. This shows the repaired rows use the historical recipe."""
    import os
    shard = f"{SHARDS}/shard_03f03b8_2threads_task{task}.h5"
    if not os.path.exists(shard):
        if os.environ.get("HCD_GATE_RUN") == "1":
            pytest.fail(f"shard absent: {shard}")
        pytest.skip(f"shard absent: {shard}")
    require_real_cache(HIST)
    sim, snap = REPRO[task]
    with h5py.File(shard, "r") as s, h5py.File(HIST, "r") as h:
        assert list(s["snap_sim_name"].asstr()[...]) == [sim] and list(s["snap_snap"][...]) == [snap]
        hs, hn = h["sim_name"].asstr()[...], h["snap"][...]
        hi = np.where((hs == sim) & (hn == snap))[0]
        hi = hi[np.argsort(h["alpha_idx"][...][hi])]
        assert hi.size == s["sim_name"].shape[0] == 20
        for k in sp.ROW_KEYS:
            if k == "snap_group_idx":
                continue
            a = s[k].asstr()[...] if k in sp.STR_KEYS else s[k][...]
            b = h[k].asstr()[...][hi] if k in sp.STR_KEYS else h[k][...][hi]
            if a.dtype.kind == "f":
                assert sp._rel(a, b) <= 1e-12 and np.array_equal(np.isnan(a), np.isnan(b)), (k, sp._rel(a, b))
            else:
                assert np.array_equal(a, b), k
        hb = int(np.where((h["snap_sim_name"].asstr()[...] == sim) & (h["snap_snap"][...] == snap))[0][0])
        for k in sp.SNAP_KEYS:
            a = s[k].asstr()[...][0] if k in sp.STR_KEYS else s[k][...][0]
            b = h[k].asstr()[...][hb] if k in sp.STR_KEYS else h[k][...][hb]
            assert np.array_equal(a, b), k


def test_repair_record_names_the_inputs():
    require_real_cache(NEW)
    require_real_cache(HIST)
    with h5py.File(NEW, "r") as f:
        rec = json.loads(f.attrs["repair_record"])
    assert rec["base_sha256"] == sp.sha256(HIST)
    assert rec["removed"] == [[NS0907, 17, 2.8]]
    assert sorted((s, n, round(z, 9)) for s, n, z in rec["added"]) == [(NS0907, 17, 3.0), (NS0907, 18, 2.8)]
    assert rec["n_rows"] == [21440, 21460] and rec["n_snaps"] == [1072, 1073]
    assert "S4_REPAIR_SPEC" in rec["spec"]
