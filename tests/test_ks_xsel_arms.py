"""TDD for the X-battery-2 registry (scripts/ks_xsel_arms.py): arm matrix pins, X4 profile
math, mixture arithmetic, the fail-loud truth-table loader, the registry signature (truth-
table sha INCLUDED), and the swap-off per-mock-record identity harness. Pure numpy + tmp
files; no ctx build, no fits. New test file only (frozen tests untouched)."""
import hashlib
import json
import os

import numpy as np
import pytest

import scripts.ks_xsel_arms as XA


# ---------------------------------------------------------------- synthetic truth table

def make_synth_table(path, *, n_zbins=2, n_k=5, tamper=None):
    """Write a contract-conformant synthetic truth table; returns (path, sha256hex).
    ``tamper`` mutates the payload BEFORE writing (for fail-loud tests)."""
    z_bins = np.linspace(2.4, 3.0, n_zbins)
    leg_z = np.repeat(z_bins, n_k)
    leg_k = np.tile(np.linspace(0.005, 0.06, n_k), n_zbins)
    rng = np.random.default_rng(7)
    payload = dict(
        leg_k=leg_k, leg_z=leg_z,
        ratio_rows_X1_dla100=0.90 + 0.02 * rng.random(leg_k.size),
        ratio_rows_X1b_dla100_diluted=0.63 + 0.02 * rng.random(leg_k.size),
        ratio_rows_X2_sub100=1.30 + 0.05 * rng.random(leg_k.size),
        ratio_rows_X3_lls100=1.20 + 0.05 * rng.random(leg_k.size),
        convention_json=json.dumps(dict(XA.REQUIRED_CONVENTION, source="synthetic-test")),
        gate_power_json=json.dumps({a: dict(D=0.5, sigma_pair_expected=0.1,
                                            p_part1_fail_null=0.01)
                                    for a in XA.GATE_POWER_ARMS}),
    )
    if tamper:
        tamper(payload)
    np.savez(path, **payload)
    with open(path, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    return str(path), sha


@pytest.fixture()
def table(tmp_path):
    return make_synth_table(tmp_path / "xsel_truth_tables.npz")


# ---------------------------------------------------------------- registry pins

def test_arm_matrix_pinned():
    assert set(XA.ARMS) == {"X1_dla100", "X2_sub100", "X3_lls100", "X4_prof",
                            "K8a_eps_hi", "K8b_eps_lo", "K8c_kap_hi"}
    for a in ("X1_dla100", "X2_sub100", "X3_lls100", "X4_prof"):
        assert XA.ARMS[a]["n_mocks"] == 16
    for a in ("K8a_eps_hi", "K8b_eps_lo", "K8c_kap_hi"):
        assert XA.ARMS[a]["n_mocks"] == 8
    assert XA.part1_arm_ids() == ["X1_dla100", "X2_sub100", "X3_lls100"]
    assert not XA.ARMS["X4_prof"]["part1"] and not XA.ARMS["K8a_eps_hi"]["part1"]
    assert XA.ARMS["X1_dla100"]["kind"] == "data_swap"
    assert XA.ARMS["X2_sub100"]["cls"] == 1 and XA.ARMS["X3_lls100"]["cls"] == 0
    assert XA.ARMS["K8a_eps_hi"]["sigma_expect"] == 0.5310
    assert XA.ARMS["K8b_eps_lo"]["n_sigma"] == -1.0
    assert XA.ARMS["K8c_kap_hi"]["site"] == "kappa_lls"
    assert XA.ARMS["K8c_kap_hi"]["sigma_expect"] == 0.6681
    assert XA.OVERLAY_ARMS == ("X1b_dla100_diluted",)
    assert XA.PAIR_SEED == 20260615          # the reused K0 shards' stamped seed (annex OQ5)


def test_batch_cells_and_cost():
    cells = XA.batch_cells()
    assert len(cells) == 88
    assert cells[0] == ("X1_dla100", 0) and cells[15] == ("X1_dla100", 15)
    assert cells[16] == ("X2_sub100", 0) and cells[48] == ("X4_prof", 0)
    assert cells[64] == ("K8a_eps_hi", 0) and cells[87] == ("K8c_kap_hi", 7)
    assert len(set(cells)) == 88
    cost = XA.campaign_cost_cpuh()
    assert cost["fits"] == 88 and cost["nominal"] == 440.0
    assert cost["nominal_x123"] == 240.0     # proposal Sec 7 lines
    assert cost["nominal_x4"] == 80.0
    assert cost["nominal_k8"] == 120.0
    assert cost["worst_case"] == 660.0


def test_shard_pkl_name():
    assert XA.shard_pkl_name("X2_sub100", 3) == "ks_xsel_X2_sub100_shard_003.pkl"
    assert XA.shard_pkl_name("K8a_eps_hi", 0, smoke=True) == \
        "ks_xsel_K8a_eps_hi_shard_000.smoke.pkl"
    with pytest.raises(AssertionError):
        XA.shard_pkl_name("K6_inv_u", 0)     # legacy arms are not this registry's


# ---------------------------------------------------------------- X4 profile math

def test_f_sel_peak_and_edges():
    # peak at z = 3.364 (the retired K6 shape, peak-normalized), f == 1 there
    assert abs(XA.X4_PEAK_Z - 3.364) < 1e-3
    assert abs(XA.f_sel(XA.X4_PEAK_Z) - 1.0) < 1e-12
    # band edges 0.254 (proposal Sec 4)
    assert round(XA.f_sel(2.4), 3) == 0.254
    assert round(XA.f_sel(4.6), 3) == 0.254
    # f_sel in (0, 1] everywhere on a dense band grid, max at the peak
    z = np.linspace(2.2, 4.8, 801)
    f = XA.f_sel(z)
    assert np.all(f > 0) and np.all(f <= 1.0)
    assert abs(z[np.argmax(f)] - XA.X4_PEAK_Z) < 5e-3
    # scalar in -> float out
    assert isinstance(XA.f_sel(3.0), float)


def test_f_sel_requires_negative_eta2():
    with pytest.raises(AssertionError):
        XA.f_sel(3.0, eta1=3.83, eta2=+22.0)


# ---------------------------------------------------------------- mixture arithmetic

def test_corner_and_profile_rows():
    rows = XA.corner_alpha_rows(4, 1)
    assert rows.shape == (4, 3)
    assert np.array_equal(rows[:, 1], np.ones(4)) and rows[:, [0, 2]].sum() == 0
    with pytest.raises(AssertionError):
        XA.corner_alpha_rows(4, 2)           # the DLA corner is data-side (X1), never mixture
    z = np.array([2.4, 3.0, 4.6])
    prows = XA.profile_alpha_rows(z, 0)
    assert np.allclose(prows[:, 0], XA.f_sel(z)) and prows[:, 1:].sum() == 0


def test_mixture_ratio_identity():
    r = np.array([0.8, 1.0, 1.5])
    assert np.allclose(XA.mixture_ratio(0.0, r), 1.0)          # f=0 -> clean
    assert np.allclose(XA.mixture_ratio(1.0, r), r)            # f=1 -> the corner
    assert np.allclose(XA.mixture_ratio(0.5, r), 1 + 0.5 * (r - 1))


def test_truth_admissible():
    good = np.array([[0.2, 0.05, 0.01], [0.0, 0.0, 0.0]])
    assert XA.truth_admissible(good)
    assert XA.truth_admissible(good, strict_interior=True)
    corner = XA.corner_alpha_rows(3, 0)
    assert XA.truth_admissible(corner)                          # sum == 1 admissible
    assert not XA.truth_admissible(corner, strict_interior=True)
    assert not XA.truth_admissible(np.array([[0.7, 0.4, 0.0]]))  # sum > 1
    assert not XA.truth_admissible(np.array([[-0.1, 0.0, 0.0]]))
    assert not XA.truth_admissible(np.array([[np.nan, 0.0, 0.0]]))


# ---------------------------------------------------------------- truth-table loader

def test_loader_unpinned_fails_loud(table, monkeypatch):
    """Stage V PINNED the table (commit 1778ce0), so the module-level sha is now a live
    64-hex pin; the unpinned fail-loud path must still fire when the pin is absent."""
    path, _sha = table
    assert isinstance(XA.XSEL_TRUTH_TABLE_SHA256, str)
    assert len(XA.XSEL_TRUTH_TABLE_SHA256) == 64
    # with the pin LIVE, a foreign table + expect_sha=None must never be silently accepted:
    # it falls back to the pin and fails on the mismatch.
    with pytest.raises(AssertionError, match="sha256"):
        XA.load_truth_tables(path, expect_sha=None)
    # and with the pin removed, the NOT-PINNED refusal is still the contract.
    monkeypatch.setattr(XA, "XSEL_TRUTH_TABLE_SHA256", None)
    with pytest.raises(RuntimeError, match="NOT PINNED"):
        XA.load_truth_tables(path, expect_sha=None)


def test_loader_missing_file_fails_loud(tmp_path):
    with pytest.raises(FileNotFoundError, match="stage-V"):
        XA.load_truth_tables(tmp_path / "nope.npz", expect_sha="0" * 64)


def test_loader_sha_mismatch_fails_loud(table):
    path, _sha = table
    with pytest.raises(AssertionError, match="sha256"):
        XA.load_truth_tables(path, expect_sha="0" * 64)


def test_loader_good_table(table):
    path, sha = table
    tt = XA.load_truth_tables(path, expect_sha=sha)
    assert tt["sha256"] == sha and tt["band"] is None
    assert set(tt["ratios"]) == {"ratio_rows_X1_dla100", "ratio_rows_X1b_dla100_diluted",
                                 "ratio_rows_X2_sub100", "ratio_rows_X3_lls100"}
    assert tt["convention"]["x1_fork"] == "dilution_corrected"
    assert set(tt["gate_power"]) == set(XA.GATE_POWER_ARMS)
    assert tt["leg_k"].shape == tt["leg_z"].shape


@pytest.mark.parametrize("mutate,msg", [
    (lambda p: p.pop("ratio_rows_X2_sub100"), "missing required keys"),
    (lambda p: p.update(convention_json=json.dumps(
        dict(XA.REQUIRED_CONVENTION, x1_fork="dilution_included"))), "convention"),
    (lambda p: p.update(gate_power_json=json.dumps(
        {a: dict(D=0.5, sigma_pair_expected=0.1, p_part1_fail_null=0.01)
         for a in XA.GATE_POWER_ARMS if a != "K8c_kap_hi"})), "gate_power"),
    (lambda p: p.update(ratio_rows_X3_lls100=p["ratio_rows_X3_lls100"] * -1.0),
     "non-finite/non-positive"),
    (lambda p: p.update(ratio_rows_X1_dla100=p["ratio_rows_X1_dla100"][:-1]), "shape"),
])
def test_loader_payload_validation(tmp_path, mutate, msg):
    path, sha = make_synth_table(tmp_path / "bad.npz", tamper=mutate)
    with pytest.raises(AssertionError, match=msg):
        XA.load_truth_tables(path, expect_sha=sha)


def test_loader_band_requires_both_rows(tmp_path):
    def _t(p):
        p["band_lo_rows_X1_dla100"] = p["ratio_rows_X1_dla100"] * 0.95
    path, sha = make_synth_table(tmp_path / "band.npz", tamper=_t)
    with pytest.raises(AssertionError, match="BOTH lo and hi"):
        XA.load_truth_tables(path, expect_sha=sha)


# ---------------------------------------------------------------- registry signature

def test_registry_signature_pins_table_sha(tmp_path):
    p1, s1 = make_synth_table(tmp_path / "t1.npz")
    sig1 = XA.registry_signature(p1, expect_sha=s1)
    sig1b = XA.registry_signature(p1, expect_sha=s1)
    assert sig1 == sig1b and len(sig1) == 64 and int(sig1, 16) >= 0
    # a DIFFERENT table (one ratio nudged) must move the signature (sha is in the payload)
    p2, s2 = make_synth_table(
        tmp_path / "t2.npz",
        tamper=lambda p: p.update(ratio_rows_X2_sub100=p["ratio_rows_X2_sub100"] * 1.001))
    assert XA.registry_signature(p2, expect_sha=s2) != sig1
    # payload is canonical JSON (fully serializable)
    json.dumps(XA._registry_payload(p1, expect_sha=s1), sort_keys=True)


def test_registry_signature_refuses_unpinned(tmp_path, monkeypatch):
    p, _s = make_synth_table(tmp_path / "t.npz")
    monkeypatch.setattr(XA, "XSEL_TRUTH_TABLE_SHA256", None)   # pin live since 1778ce0
    with pytest.raises(RuntimeError, match="NOT PINNED"):
        XA.registry_signature(p, expect_sha=None)


def test_signature_differs_from_legacy_registry():
    """The X signature can never pool with the frozen legacy campaign registry."""
    import scripts.ks_selboost_arms as AR
    # legacy signature computes from the pinned K5 table (must exist on this tree)
    assert os.path.exists(AR.K5_TABLE_PATH)
    legacy = AR.registry_signature()
    # cannot compute the X signature without a pinned table, and even a synthetic pin can
    # never equal the legacy one (different payload structure); compare against a synthetic
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p, s = make_synth_table(os.path.join(td, "t.npz"))
        assert XA.registry_signature(p, expect_sha=s) != legacy


# ---------------------------------------------------------------- swap-off record harness

def _rec(seed=0):
    rng = np.random.default_rng(seed)
    return dict(sim="leg_a_prior",
                truth_vec=rng.random(7),
                draws=rng.random((10, 7)),
                L=10, ll_true=float(rng.random()),
                ll_draws=rng.random(10),
                names=["ns", "Ap", "t0", "t1", "alpha_lls", "alpha_subdla", "alpha_dla"],
                kept_global=np.ones(3, bool),
                dropped={"KS": []},
                sites_extra={"tau0_amp": dict(draws=rng.random(10), truth=0.5),
                             "eps_lls": dict(draws=rng.random(10), truth=float("nan"))},
                n_div=0,
                truth_alpha_hcd_z=rng.random((3, 3)) * 0.1)


def test_compare_records_identical():
    import copy
    a = _rec()
    assert XA.compare_per_mock_records(a, copy.deepcopy(a)) == []


def test_compare_records_detects_array_bit_flip():
    import copy
    a = _rec()
    b = copy.deepcopy(a)
    b["draws"][3, 2] = np.nextafter(b["draws"][3, 2], 1.0)
    mism = XA.compare_per_mock_records(a, b)
    assert any("draws" in m and "bytes differ" in m for m in mism)


def test_compare_records_detects_dtype_and_keys():
    import copy
    a = _rec()
    b = copy.deepcopy(a)
    b["truth_vec"] = b["truth_vec"].astype(np.float32)
    b["extra_key"] = 1
    del b["ll_draws"]
    mism = XA.compare_per_mock_records(a, b)
    assert any("dtype" in m for m in mism)
    assert any("'extra_key'" in m for m in mism)
    assert any("'ll_draws'" in m for m in mism)


def test_compare_records_nested_and_nan():
    import copy
    a = _rec()
    b = copy.deepcopy(a)
    assert XA.compare_per_mock_records(a, b) == []      # NaN truth == NaN truth allowed
    b["sites_extra"]["tau0_amp"]["truth"] = 0.6
    mism = XA.compare_per_mock_records(a, b)
    assert any("tau0_amp" in m for m in mism)


def test_compare_records_scalar_float_nan_rules():
    a = {"x": float("nan"), "y": 1.0}
    assert XA.compare_per_mock_records(a, {"x": float("nan"), "y": 1.0}) == []
    assert XA.compare_per_mock_records(a, {"x": 0.0, "y": 1.0})
