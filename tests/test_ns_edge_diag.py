"""NO-PEEK tests for scripts/ns_edge_diag.py (PI #17 sec-12 bounded diagnostic).

Every test runs on SYNTHETIC data. No test reads a certification pkl, so the whole file can
be run before the prereg is unblinded without any risk of inspecting the real association.

The two that matter most:
  * test_size_under_null            -- the permutation test is correctly sized
  * test_seed_driven_retry_does_not_inflate  -- the joint permutation handles the
    truth/base_seed confound; an independent-per-arm null would be anti-conservative here
"""
import os
import pickle
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))
import ns_edge_diag as D  # noqa: E402


# --------------------------------------------------------------------------- #
# coordinate helpers
# --------------------------------------------------------------------------- #
def test_unit_physical_roundtrip():
    u = np.linspace(0, 1, 11)
    assert np.allclose(D.u_of_phys(D.phys_of_u(u)), u)
    assert D.phys_of_u(0.0) == pytest.approx(0.8)
    assert D.phys_of_u(1.0) == pytest.approx(1.05)


def test_design_hull_constant():
    # theta_unit of the design-hull top, from the pinned cache value.
    assert D.DESIGN_MAX_U == pytest.approx((1.0395833 - 0.8) / 0.25)
    assert 0.95 < D.DESIGN_MAX_U < 0.96


def test_d_ns_matches_implemented_formula():
    """d_ns = max(max(ns - 0.98, 0.86 - ns), 0)  -- data_likelihood.py:868."""
    for ns in (0.80, 0.86, 0.90, 0.98, 0.99, 1.05):
        u = D.u_of_phys(ns)
        want = max(max(ns - 0.98, 0.86 - ns), 0.0)
        assert D.d_ns_of_u(u) == pytest.approx(want, abs=1e-12)
    # zero strictly inside the box, positive strictly outside
    assert D.d_ns_of_u(D.u_of_phys(0.92)) == 0.0
    assert D.d_ns_of_u(D.u_of_phys(1.00)) > 0.0
    assert D.d_ns_of_u(D.u_of_phys(0.84)) > 0.0


def test_d_ns_uses_frozen_box_not_the_rejected_comment():
    """PI #17 sec-12 forbids the 0.995 comment value. Guard against a silent revival."""
    assert D.NS_BOX == (0.86, 0.98)
    assert 0.995 not in D.NS_BOX


# --------------------------------------------------------------------------- #
# the statistic
# --------------------------------------------------------------------------- #
def test_stat_is_plain_mean_over_retried_units():
    score = {"A": np.arange(10.0), "B": np.arange(10.0) * 100}
    retried = {"A": [1, 2], "B": [3]}
    got = D.stat_mean_over_retried(score, retried)
    assert got == pytest.approx((1 + 2 + 300) / 3)


def test_shared_index_counted_once_per_arm():
    """A mock index retried in two arms contributes TWICE (two execution units)."""
    score = {"A": np.zeros(10), "B": np.zeros(10)}
    score["A"][4] = 1.0
    score["B"][4] = 1.0
    got = D.stat_mean_over_retried(score, {"A": [4], "B": [4]})
    assert got == pytest.approx(1.0)          # mean of [1,1]
    got2 = D.stat_mean_over_retried(score, {"A": [4], "B": [0]})
    assert got2 == pytest.approx(0.5)         # mean of [1,0]


def test_campaign_multiplicities_and_unit_count():
    """The prereg's stated weights: index 4,5,7 have multiplicity 2; total 17 units."""
    w = np.zeros(D.N_SHARED)
    n_units = 0
    for arm, ms in D.RETRIED.items():
        for m in ms:
            n_units += 1
            if m < D.N_SHARED:
                w[m] += 1
    assert n_units == 17
    assert w.sum() == 15                       # 15 in the shared block
    assert w[4] == 2 and w[5] == 2 and w[7] == 2
    for m in (1, 8, 10, 14, 15, 25, 37, 39, 41):
        assert w[m] == 1
    ks_only = [m for ms in D.RETRIED.values() for m in ms if m >= D.N_SHARED]
    assert sorted(ks_only) == [49, 73]


def test_permuted_statistic_matches_manual_application():
    score = {"A": np.arange(48.0) + 1000, "B": np.arange(48.0) + 1000}
    retried = {"A": [0, 1], "B": [2]}
    P_sh = np.array([np.roll(np.arange(48), 1)])          # single permutation
    P_ks = np.zeros((1, 0), int)
    got = D.stat_mean_over_retried(score, retried, P_sh, P_ks)
    want = (score["A"][P_sh[0, 0]] + score["A"][P_sh[0, 1]] + score["B"][P_sh[0, 2]]) / 3
    assert got[0] == pytest.approx(want)


# --------------------------------------------------------------------------- #
# Holm
# --------------------------------------------------------------------------- #
def test_holm_known_values():
    p = np.array([0.01, 0.04, 0.03])
    adj = D.holm(p)
    assert adj[0] == pytest.approx(0.03)       # 3 * 0.01
    assert adj[2] == pytest.approx(0.06)       # max(0.03, 2 * 0.03)
    assert adj[1] == pytest.approx(0.06)       # monotone, max(0.06, 1 * 0.04)


def test_holm_is_monotone_and_bounded():
    rng = np.random.default_rng(0)
    p = rng.random(25)
    adj = D.holm(p)
    assert np.all(adj <= 1.0)
    assert np.all(adj >= p)
    order = np.argsort(p)
    assert np.all(np.diff(adj[order]) >= -1e-12)


# --------------------------------------------------------------------------- #
# calibration of the null -- the load-bearing tests
# --------------------------------------------------------------------------- #
def _null_scores(rng):
    """Truths shared on 0..47, independent on the KS-only block -- as in the campaign."""
    sh = rng.random(D.N_SHARED)
    ks = rng.random(48)
    return {"A1c_eBOSS": sh.copy(), "A2c_DESI": sh.copy(),
            "A3c_KS": np.concatenate([sh, ks])}


def test_size_under_null():
    """p-values must be ~Uniform when retry is independent of truth."""
    rng = np.random.default_rng(11)
    ps = []
    for _ in range(300):
        sc = _null_scores(rng)
        ps.append(D.perm_test(sc, D.RETRIED, rng, B=2000, side="upper")["p"])
    ps = np.array(ps)
    assert 0.02 <= np.mean(ps <= 0.05) <= 0.11, np.mean(ps <= 0.05)
    assert 0.05 <= np.mean(ps <= 0.10) <= 0.18, np.mean(ps <= 0.10)
    assert 0.40 <= np.median(ps) <= 0.60


def test_power_against_planted_edge_effect():
    """Planting retries in the top decile must be detected most of the time."""
    rng = np.random.default_rng(12)
    hits = 0
    n = 120
    for _ in range(n):
        sc = _null_scores(rng)
        for arm, ms in D.RETRIED.items():
            for m in ms:
                v = 1.0 - 0.10 * rng.random()
                if m < D.N_SHARED:
                    for a in sc:
                        sc[a][m] = v          # shared block: same truth in every arm
                else:
                    sc["A3c_KS"][m] = v
        if D.perm_test(sc, D.RETRIED, rng, B=2000, side="upper")["p"] <= 0.05:
            hits += 1
    assert hits / n > 0.90, hits / n


def test_seed_driven_retry_does_not_inflate():
    """THE CONFOUND TEST. If retry is driven by the shared base_seed (so the SAME indices
    retry in every arm) and is independent of truth, the joint permutation must stay
    correctly sized. An independent-per-arm null would over-reject here."""
    rng = np.random.default_rng(13)
    ps = []
    for _ in range(300):
        sc = _null_scores(rng)
        # a seed-driven retry pattern: one shared set of "unlucky" indices for all arms
        unlucky = rng.choice(D.N_SHARED, size=6, replace=False)
        retried = {"A1c_eBOSS": list(unlucky), "A2c_DESI": list(unlucky),
                   "A3c_KS": list(unlucky)}
        ps.append(D.perm_test(sc, retried, rng, B=2000, side="upper")["p"])
    ps = np.array(ps)
    assert np.mean(ps <= 0.05) <= 0.12, np.mean(ps <= 0.05)


def test_p_is_never_zero():
    rng = np.random.default_rng(14)
    sc = {a: np.zeros(D.ARMS[a][1]) for a in D.ARMS}
    for a in sc:
        for m in D.RETRIED[a]:
            sc[a][m] = 1e9                     # maximally extreme
    r = D.perm_test(sc, D.RETRIED, rng, B=1000, side="upper")
    assert r["p"] > 0.0
    assert r["p"] == pytest.approx(1 / 1001)


def test_two_sided_is_not_smaller_than_one_sided_when_effect_is_upper():
    rng = np.random.default_rng(15)
    sc = _null_scores(rng)
    for a in sc:
        for m in D.RETRIED[a]:
            if m < len(sc[a]):
                sc[a][m] = 0.99
    up = D.perm_test(sc, D.RETRIED, np.random.default_rng(1), B=4000, side="upper")["p"]
    tw = D.perm_test(sc, D.RETRIED, np.random.default_rng(1), B=4000, side="two")["p"]
    assert tw >= up - 1e-9


# --------------------------------------------------------------------------- #
# loader + refusal
# --------------------------------------------------------------------------- #
def _write_fake_arm(tmpdir, arm, n, rng, shared_u=None):
    d, _ = D.ARMS[arm]
    os.makedirs(os.path.join(tmpdir, d), exist_ok=True)
    names = ["ns", "Ap"] + [f"x{i}" for i in range(23)]
    for m in range(n):
        u = shared_u[m] if (shared_u is not None and m < len(shared_u)) else rng.random()
        truth = np.concatenate([[u, rng.random()], rng.random(23)])
        draws = np.column_stack([np.clip(rng.normal(u, 0.05, 150), 0, 1),
                                 rng.random((150, 24))])
        extra = {k: dict(truth=float(rng.random()), draws=rng.random(120))
                 for k in ("tau0_amp", "dtau0", "f_res_amp", "f_res_slope")}
        extra[f"own_{arm}"] = dict(truth=float(rng.random()), draws=rng.random(120))
        with open(os.path.join(tmpdir, d, f"mock_{m:04d}.pkl"), "wb") as fh:
            pickle.dump(dict(names=names, truth_vec=truth, draws=draws, n_div=0, L=150,
                             sites_extra=extra), fh)


def test_loader_extracts_ns_by_name_not_position(tmp_path):
    rng = np.random.default_rng(3)
    _write_fake_arm(str(tmp_path), "A2c_DESI", 48, rng)
    got = D.load_arm("A2c_DESI", root=str(tmp_path))
    assert len(got) == 48
    assert got[0]["i_ns"] == 0
    m0 = got[0]
    assert 0.0 <= m0["rank"] <= 1.0
    assert m0["post_sd"] > 0


def test_run_refuses_when_truths_are_not_shared(tmp_path):
    """The joint-permutation null is only valid if the shared block really is shared."""
    rng = np.random.default_rng(4)
    for arm, n in (("A1c_eBOSS", 48), ("A2c_DESI", 48), ("A3c_KS", 96)):
        _write_fake_arm(str(tmp_path), arm, n, rng)     # independent truths -> NOT shared
    with pytest.raises(SystemExit, match="REFUSE"):
        D.run(root=str(tmp_path), B=200)


def test_run_end_to_end_on_shared_synthetic(tmp_path):
    rng = np.random.default_rng(5)
    shared = rng.random(48)
    for arm, n in (("A1c_eBOSS", 48), ("A2c_DESI", 48), ("A3c_KS", 96)):
        _write_fake_arm(str(tmp_path), arm, n, rng, shared_u=shared)
    res = D.run(root=str(tmp_path), B=500)
    assert res["_meta"]["shared_truth_verified"] is True
    assert res["_meta"]["all_final_n_div_zero"] is True
    for key in ("primary", "S1_per_arm", "S2_control_family", "S2b_arm_specific_extra",
                "S3_two_sided", "S4_KS_kink", "S5_calibration", "S6_boundary",
                "S7_concordance", "per_unit", "init_strategy"):
        assert key in res
    # S2 pools only the sites common to ALL arms; arm-specific ones go to S2b
    assert res["S2_common_extra_sites"] == ["dtau0", "f_res_amp", "f_res_slope", "tau0_amp"]
    assert len(res["S2_control_family"]) == 25 + 4
    for a in D.ARMS:
        assert list(res["S2b_arm_specific_extra"][a]) == [f"own_{a}"]
    assert 0 < res["primary"]["p"] <= 1
    assert len(res["per_unit"]["A3c_KS"]) == 96
    assert sum(r["retried"] for r in res["per_unit"]["A1c_eBOSS"]) == 8
    assert sum(r["retried"] for r in res["per_unit"]["A3c_KS"]) == 4


def test_retry_sets_match_the_frozen_record():
    assert D.RETRIED["A1c_eBOSS"] == [1, 4, 7, 10, 15, 37, 39, 41]
    assert D.RETRIED["A2c_DESI"] == [4, 5, 7, 8, 14]
    assert D.RETRIED["A3c_KS"] == [5, 25, 49, 73]
    assert sum(len(v) for v in D.RETRIED.values()) == 17


# --------------------------------------------------------------------------- #
# initialization distribution (PI #17 sec-12 items 3 and 11)
# --------------------------------------------------------------------------- #
def test_init_distribution_is_beta_8_8():
    """init_to_median(num_samples=15) on a Uniform site -> 8th order statistic of 15
    iid Uniform(0,1) -> Beta(8,8) exactly. Confirmed against Monte Carlo."""
    a = D.init_dist_analysis()
    assert a["mean"] == pytest.approx(0.5, abs=1e-12)
    assert a["sd"] == pytest.approx(0.12126781, abs=1e-6)
    rng = np.random.default_rng(7)
    mc = np.sort(rng.random((200000, 15)), axis=1)[:, 7]
    assert mc.mean() == pytest.approx(0.5, abs=0.005)
    assert mc.std() == pytest.approx(a["sd"], abs=0.005)


def test_init_cannot_reach_the_upper_region():
    a = D.init_dist_analysis()
    # design hull: astronomically unlikely over the whole campaign
    assert a["tail_prob"]["u_0.9583_design_hull_top"] < 1e-6
    assert a["expected_count_over_campaign"]["u_0.9583_design_hull_top"] < 1e-3
    # and the monotone ordering of the tail probabilities must hold
    tp = a["tail_prob"]
    assert (tp["u_0.7200_ns_0.98_KS_ns_box_top"] > tp["u_0.7800_ns_0.995_REJECTED_comment_value"]
            > tp["u_0.8000_ns_1.00"] > tp["u_0.9583_design_hull_top"]
            > tp["u_0.9900_ns_1.0475"])


def test_init_median_commutes_with_the_uniform_bijector():
    """The logit is monotone per component, so median-then-map == map-then-median."""
    rng = np.random.default_rng(8)
    s = rng.random((15, 9))
    unc = np.log(s / (1 - s))
    assert np.allclose(1 / (1 + np.exp(-np.median(unc, axis=0))), np.median(s, axis=0))
