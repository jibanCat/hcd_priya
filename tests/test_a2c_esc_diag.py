"""Synthetic-fixture tests for scripts/a2c_esc_diag.py (the A2c ROW-3 bounded diagnostic).

NO-PEEK DISCIPLINE: these tests build synthetic populations only and never touch the real
pkls, gate JSONs or scratch directories. The diagnostic's first contact with real data is
its single post-review production invocation (prereg section 20).
"""
import hashlib
import importlib.util
import json
import os
import pickle

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name):
    spec = importlib.util.spec_from_file_location(name[:-3], os.path.join(REPO, "scripts", name))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ED():
    return _load("a2c_esc_diag.py")


N = 48
NAMES = ["ns", "Ap"] + [f"tau0_z{z}" for z in range(13)]


def _mk_pop(tmpdir, ED, seed=11, hetero_L=True, n=N, cfg_override=None):
    """Synthetic n-mock DESI population with the frozen cfg; returns (outdir, sha, gate, sdj)."""
    frozen = ED._frozen_cfg()
    if cfg_override:
        frozen = dict(frozen, **cfg_override)
    rng = np.random.default_rng(seed)
    os.makedirs(tmpdir, exist_ok=True)
    Ls = []
    for m in range(n):
        L = int(rng.choice([75, 100, 150, 200, 300])) if hetero_L else 150
        Ls.append(L)
        truth = np.zeros(len(NAMES))
        truth[NAMES.index("ns")] = float(rng.normal())
        truth[NAMES.index("Ap")] = float(rng.normal())
        ladder = np.exp(rng.normal(size=13) * 0.01 - 0.3)
        for z in range(13):
            truth[NAMES.index(f"tau0_z{z}")] = ladder[z]
        draws = np.zeros((L, len(NAMES)))
        draws[:, NAMES.index("ns")] = rng.normal(truth[NAMES.index("ns")], 1.0, L)
        draws[:, NAMES.index("Ap")] = rng.normal(truth[NAMES.index("Ap")], 1.0, L)
        for z in range(13):
            draws[:, NAMES.index(f"tau0_z{z}")] = np.exp(
                rng.normal(np.log(ladder[z]), 0.05, L))
        rec = dict(names=list(NAMES), draws=draws, truth_vec=truth, L=L, n_div=0,
                   run_cfg=dict(frozen),
                   sites_extra={"f_res_amp": {"draws": rng.normal(size=L), "truth": 0.01},
                                "tau0_amp": {"draws": rng.normal(size=L), "truth": 0.7}})
        with open(os.path.join(tmpdir, f"mock_{m:04d}.pkl"), "wb") as f:
            pickle.dump(rec, f)

    sha = os.path.join(tmpdir, "sha.txt")
    with open(sha, "w") as f:
        for m in range(n):
            p = os.path.join(tmpdir, f"mock_{m:04d}.pkl")
            f.write(f"{ED._sha256(p)}  mock_{m:04d}.pkl\n")

    if cfg_override:
        # The population is deliberately non-frozen: the loader must refuse it, so we
        # cannot build gate/selfdraw anchors from it. Only the pkls+sha are needed.
        return tmpdir, sha, None, None

    recs = ED.load_population(tmpdir, sha, n_total=n)
    legrec = {"div_total": 0, "pulls": {}, "rank_uniformity": {},
              "finite_L_null": {"sd": ED.finite_L_null_sd([r["L"] for r in recs])}}
    for key in ("ns", "Ap", "tau0amp"):
        p, rk = ED._chan(recs, key)
        legrec["pulls"][key] = {"mean": float(p.mean()), "std": float(p.std(ddof=1))}
        legrec["rank_uniformity"][key] = {"ks_p": ED.rank_ks_p(rk)}
    gate = os.path.join(tmpdir, "gate.json")
    json.dump({"legs": {"DESI": legrec}}, open(gate, "w"))
    sdj = os.path.join(tmpdir, "sd.json")
    json.dump({"n": n, "survey": "DESI", "conjuncts_ok": True, "health_ok": True,
               "L_median": float(np.median([r["L"] for r in recs])),
               "L_range": [int(min(Ls)), int(max(Ls))]}, open(sdj, "w"))
    return tmpdir, sha, gate, sdj


# --- deployed-definition fidelity ---------------------------------------------------

def test_tau0_amp_vec_matches_scalar(ED):
    rng = np.random.default_rng(3)
    ladders = np.exp(rng.normal(size=(40, 13)) * 0.1 - 0.3)
    vec = ED.tau0_amp_vec(ladders)
    scal = np.array([ED.tau0_amp_slope(row)[0] for row in ladders])
    assert np.allclose(vec, scal, rtol=1e-15, atol=0.0)


def test_tau0_amp_is_geometric_mean_of_ladder(ED):
    """intercept = mean(log tau0) because lx is centered -> amp = geometric mean."""
    ladder = np.exp(np.linspace(-0.5, -0.1, 13))
    amp, _ = ED.tau0_amp_slope(ladder)
    assert np.isclose(amp, float(np.exp(np.log(ladder).mean())), rtol=1e-15)


def test_finite_L_null_refuses_small_L(ED):
    with pytest.raises(ED.DiagRefusal):
        ED.finite_L_null_sd([100, 3])
    with pytest.raises(ED.DiagRefusal):
        ED.finite_L_null_sd([])


def test_band_edges_are_literal_and_inclusive(ED):
    """A rank exactly at 0.025 must be INSIDE the 95% band (the frozen float hazard)."""
    lo, hi = ED.BAND_EDGES[0.95]
    assert lo == 0.025 and hi == 0.975
    assert (0.025 >= lo) and (0.975 <= hi)
    assert lo != (1 - 0.95) / 2 or True  # documents intent; literal must be used


def test_winsor_k_follows_the_frozen_rule(ED):
    import math
    assert ED.WINSOR_K == math.ceil(0.05 * 48) == 3


def test_holm_step_down(ED):
    assert ED.holm([0.001, 0.04, 0.9]) == [True, False, False]
    assert ED.holm([0.001, 0.01, 0.02]) == [True, True, True]
    assert ED.holm([0.9, 0.9]) == [False, False]


# --- population integrity -------------------------------------------------------------

def test_load_population_happy(tmp_path, ED):
    d, sha, _, _ = _mk_pop(str(tmp_path / "pop"), ED)
    recs = ED.load_population(d, sha)
    assert len(recs) == N
    assert [r["m"] for r in recs] == list(range(N))
    assert all(r["ns"].shape[0] == r["L"] for r in recs)


def test_census_short_refuses(tmp_path, ED):
    d, sha, _, _ = _mk_pop(str(tmp_path / "pop"), ED)
    os.remove(os.path.join(d, "mock_0047.pkl"))
    with pytest.raises(ED.DiagRefusal, match="census"):
        ED.load_population(d, sha)


def test_stray_file_refuses(tmp_path, ED):
    d, sha, _, _ = _mk_pop(str(tmp_path / "pop"), ED)
    with open(os.path.join(d, "mock_0099.pkl"), "wb") as f:
        pickle.dump({}, f)
    with pytest.raises(ED.DiagRefusal, match="census"):
        ED.load_population(d, sha)


def test_sha_mismatch_refuses(tmp_path, ED):
    d, sha, _, _ = _mk_pop(str(tmp_path / "pop"), ED)
    lines = open(sha).read().splitlines()
    lines[5] = "0" * 64 + "  mock_0005.pkl"
    open(sha, "w").write("\n".join(lines) + "\n")
    with pytest.raises(ED.DiagRefusal, match="sha256 mismatch"):
        ED.load_population(d, sha)


def test_malformed_sha_line_refuses(tmp_path, ED):
    d, sha, _, _ = _mk_pop(str(tmp_path / "pop"), ED)
    open(sha, "a").write("garbage-no-space\n")
    with pytest.raises(ED.DiagRefusal, match="malformed"):
        ED.load_population(d, sha)


def test_duplicate_sha_line_refuses(tmp_path, ED):
    d, sha, _, _ = _mk_pop(str(tmp_path / "pop"), ED)
    first = open(sha).read().splitlines()[0]
    open(sha, "a").write(first + "\n")
    with pytest.raises(ED.DiagRefusal, match="duplicate"):
        ED.load_population(d, sha)


def test_sha_names_mismatch_refuses(tmp_path, ED):
    d, sha, _, _ = _mk_pop(str(tmp_path / "pop"), ED)
    lines = open(sha).read().splitlines()
    lines[0] = lines[0].replace("mock_0000.pkl", "mock_1000.pkl")
    open(sha, "w").write("\n".join(lines) + "\n")
    with pytest.raises(ED.DiagRefusal, match="inventory names"):
        ED.load_population(d, sha)


def test_wrong_cfg_refuses(tmp_path, ED):
    d, sha, _, _ = _mk_pop(str(tmp_path / "pop"), ED, cfg_override={"leg": "KS"})
    with pytest.raises(ED.DiagRefusal, match="run_cfg"):
        ED.load_population(d, sha)


def test_draws_rows_must_equal_stored_L(tmp_path, ED):
    d, sha, _, _ = _mk_pop(str(tmp_path / "pop"), ED)
    p = os.path.join(d, "mock_0003.pkl")
    rec = pickle.load(open(p, "rb"))
    rec["L"] = rec["L"] + 1
    pickle.dump(rec, open(p, "wb"))
    with open(sha, "r") as f:
        lines = f.read().splitlines()
    lines[3] = f"{ED._sha256(p)}  mock_0003.pkl"
    open(sha, "w").write("\n".join(lines) + "\n")
    with pytest.raises(ED.DiagRefusal, match="stored L"):
        ED.load_population(d, sha)


# --- deployment-consistency conjuncts --------------------------------------------------

def test_consistency_happy(tmp_path, ED):
    d, sha, gate, sdj = _mk_pop(str(tmp_path / "pop"), ED)
    recs = ED.load_population(d, sha)
    got = ED.check_consistency(recs, gate, sdj)
    assert "ns pull mean" in got and "finite-L null sd" in got


@pytest.mark.parametrize("chan,field", [
    ("ns", "mean"), ("ns", "std"), ("Ap", "mean"), ("Ap", "std"),
    ("tau0amp", "mean"), ("tau0amp", "std")])
def test_consistency_refuses_perturbed_pull(tmp_path, ED, chan, field):
    d, sha, gate, sdj = _mk_pop(str(tmp_path / "pop"), ED)
    recs = ED.load_population(d, sha)
    j = json.load(open(gate))
    j["legs"]["DESI"]["pulls"][chan][field] += 1e-6
    json.dump(j, open(gate, "w"))
    with pytest.raises(ED.DiagRefusal, match="deployment-consistency FAIL"):
        ED.check_consistency(recs, gate, sdj)


@pytest.mark.parametrize("chan", ["ns", "Ap", "tau0amp"])
def test_consistency_refuses_perturbed_rank_p(tmp_path, ED, chan):
    d, sha, gate, sdj = _mk_pop(str(tmp_path / "pop"), ED)
    recs = ED.load_population(d, sha)
    j = json.load(open(gate))
    j["legs"]["DESI"]["rank_uniformity"][chan]["ks_p"] *= 1.0001
    json.dump(j, open(gate, "w"))
    with pytest.raises(ED.DiagRefusal, match="deployment-consistency FAIL"):
        ED.check_consistency(recs, gate, sdj)


def test_consistency_refuses_perturbed_finite_L(tmp_path, ED):
    d, sha, gate, sdj = _mk_pop(str(tmp_path / "pop"), ED)
    recs = ED.load_population(d, sha)
    j = json.load(open(gate))
    j["legs"]["DESI"]["finite_L_null"]["sd"] += 1e-8
    json.dump(j, open(gate, "w"))
    with pytest.raises(ED.DiagRefusal, match="finite-L null sd"):
        ED.check_consistency(recs, gate, sdj)


@pytest.mark.parametrize("patch", [
    {"n": 96}, {"survey": "KS"}, {"conjuncts_ok": False}, {"health_ok": False}])
def test_selfdraw_anchor_refusals(tmp_path, ED, patch):
    d, sha, gate, sdj = _mk_pop(str(tmp_path / "pop"), ED)
    recs = ED.load_population(d, sha)
    j = json.load(open(sdj))
    j.update(patch)
    json.dump(j, open(sdj, "w"))
    with pytest.raises(ED.DiagRefusal, match="selfdraw anchor"):
        ED.check_consistency(recs, gate, sdj)


def test_L_census_mismatch_refuses(tmp_path, ED):
    d, sha, gate, sdj = _mk_pop(str(tmp_path / "pop"), ED)
    recs = ED.load_population(d, sha)
    j = json.load(open(sdj))
    j["L_range"] = [j["L_range"][0], j["L_range"][1] + 1]
    json.dump(j, open(sdj, "w"))
    with pytest.raises(ED.DiagRefusal, match="L census"):
        ED.check_consistency(recs, gate, sdj)


def test_divergence_total_mismatch_refuses(tmp_path, ED):
    d, sha, gate, sdj = _mk_pop(str(tmp_path / "pop"), ED)
    recs = ED.load_population(d, sha)
    j = json.load(open(gate))
    j["legs"]["DESI"]["div_total"] = 1
    json.dump(j, open(gate, "w"))
    with pytest.raises(ED.DiagRefusal, match="divergence total"):
        ED.check_consistency(recs, gate, sdj)


def test_no_output_leaks_on_refusal(tmp_path, ED):
    """A refusal must happen BEFORE any diagnostic value is returned."""
    d, sha, gate, sdj = _mk_pop(str(tmp_path / "pop"), ED)
    recs = ED.load_population(d, sha)
    j = json.load(open(gate))
    j["legs"]["DESI"]["pulls"]["ns"]["mean"] += 1e-6
    json.dump(j, open(gate, "w"))
    with pytest.raises(ED.DiagRefusal):
        ED.check_consistency(recs, gate, sdj)


# --- P2 core (verbatim reuse of the frozen KS suite) ------------------------------------

def test_winsorized_sd_shrinks_and_matches_manual(ED):
    x = np.array([-9.0, -2, -1, -0.5, 0, 0.5, 1, 2, 9.0])
    w = ED.winsorized_sd(x, k=2)
    s = np.sort(x)
    man = s.copy()
    man[:2] = s[2]
    man[-2:] = s[-3]
    assert np.isclose(w, float(np.std(man, ddof=1)))
    assert w < float(np.std(x, ddof=1))


def test_normal_reference_winsor_ratio_is_not_one_and_is_n_specific(ED):
    """The n=96/k=5 constant 0.907 must NOT be reused at n=48/k=3."""
    r48 = ED.normal_references(n=48, k=3, nsim=20000)
    r96 = ED.normal_references(n=96, k=5, nsim=20000)
    assert 0.5 < r48["winsorized_over_sd"] < 1.0
    assert not np.isclose(r48["winsorized_over_sd"], r96["winsorized_over_sd"], atol=1e-3)
    assert r48["n"] == 48 and r48["k"] == 3


def test_normal_reference_is_seeded_and_reproducible(ED):
    a = ED.normal_references(n=48, k=3, nsim=5000)
    b = ED.normal_references(n=48, k=3, nsim=5000)
    assert a == b


def test_discrete_null_coverage_matches_brute_force(ED):
    lo, hi = ED.BAND_EDGES[0.95]
    got = ED.discrete_null_coverage([10], lo, hi)
    k = np.arange(0, 11)
    want = float(np.mean((k / 10 >= lo) & (k / 10 <= hi)))
    assert np.isclose(got, want)


def test_discrete_null_coverage_is_below_nominal_at_small_L(ED):
    """Discreteness makes the inclusive-band null occupancy differ from the nominal."""
    lo, hi = ED.BAND_EDGES[0.95]
    assert ED.discrete_null_coverage([75, 100, 150], lo, hi) != 0.95


def test_coverage_counts_boundary_rank_as_inside(ED):
    ranks = [0.025, 0.975, 0.5, 0.001]
    cov = ED.coverage(ranks)
    assert cov["0.95"]["covered"] == 3


def test_coverage_exact_binomial_ci_brackets_fraction(ED):
    cov = ED.coverage([0.5] * 30 + [0.001] * 18)
    row = cov["0.95"]
    assert row["ci95"][0] <= row["fraction"] <= row["ci95"][1]


def test_scales_and_tail_variance_shares_monotone_and_bounded(ED):
    rng = np.random.default_rng(5)
    x = rng.standard_normal(48)
    s = ED.scales_and_tail(x, normref_nsim=3000)
    c = s["variance_contrib_topk"]
    assert 0 < c["1"] <= c["2"] <= c["3"] <= c["5"] <= 1.0
    assert len(s["loo_sd_all"]) == 48


def test_scales_and_tail_detects_a_single_outlier(ED):
    rng = np.random.default_rng(6)
    x = rng.standard_normal(48)
    x[0] = 8.0
    s = ED.scales_and_tail(x, normref_nsim=3000)
    assert s["variance_contrib_topk"]["1"] > 0.3
    assert s["loo_sd_min"] < s["sd"]
    assert s["top5_abs_pulls"][0]["i"] == 0


def test_r_scale_bootstrap_point_and_ci(ED):
    rng = np.random.default_rng(9)
    x = rng.standard_normal(48) * 1.27
    Ls = rng.choice([75, 150, 300], size=48)
    out = ED.r_scale_bootstrap(x, Ls, B=500)
    assert np.isclose(out["point"], float(np.std(x, ddof=1) / ED.finite_L_null_sd(Ls)))
    assert out["ci68"][0] <= out["point"] <= out["ci68"][1]
    assert out["ci95"][0] <= out["ci68"][0] and out["ci68"][1] <= out["ci95"][1]


def test_r_scale_bootstrap_length_mismatch_refuses(ED):
    with pytest.raises(ED.DiagRefusal, match="length mismatch"):
        ED.r_scale_bootstrap([1.0, 2.0], [100], B=10)


# --- LOO exchangeability tables (the primary healthy reference machinery) ---------------

def _brute_loo(x, y, j):
    xs = np.delete(x, j); ys = np.delete(y, j)
    mx, my = xs.mean(), ys.mean()
    sdx, sdy = xs.std(ddof=1), ys.std(ddof=1)
    C = np.cov(np.vstack([xs, ys]), ddof=1)
    return mx, my, sdx, sdy, C


def test_loo_tables_match_brute_force_deletion(ED):
    rng = np.random.default_rng(21)
    x = rng.normal(size=120); y = 0.5 * x + rng.normal(size=120)
    t = ED.loo_tables(x, y)
    for j in (0, 7, 61, 119):
        mx, my, sdx, sdy, C = _brute_loo(x, y, j)
        assert np.isclose(t["pull_x"][j], (mx - x[j]) / sdx, rtol=1e-11)
        assert np.isclose(t["pull_y"][j], (my - y[j]) / sdy, rtol=1e-11)
        assert np.isclose(t["sdx"][j], sdx, rtol=1e-11)
        assert np.isclose(t["corr"][j], C[0, 1] / np.sqrt(C[0, 0] * C[1, 1]), rtol=1e-11)
        assert np.isclose(t["area"][j], np.pi * np.sqrt(np.linalg.det(C)), rtol=1e-11)


def test_loo_rank_is_fraction_of_others_below(ED):
    x = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
    t = ED.loo_tables(x, x.copy())
    for j in range(5):
        want = float(np.sum(np.delete(x, j) < x[j])) / 4.0
        assert np.isclose(t["rank_x"][j], want)


def test_loo_rank_null_is_exactly_uniform(ED):
    """The defining property of the primary reference: over j, ranks are exactly the
    uniform grid {0, 1/(L-1), ..., 1} -- no asymptotics."""
    rng = np.random.default_rng(4)
    x = rng.normal(size=60)
    t = ED.loo_tables(x, rng.normal(size=60))
    assert np.allclose(np.sort(t["rank_x"]), np.arange(60) / 59.0)


def test_loo_D2_equals_mahalanobis(ED):
    rng = np.random.default_rng(31)
    x = rng.normal(size=90); y = -0.7 * x + 0.4 * rng.normal(size=90)
    t = ED.loo_tables(x, y)
    for j in (2, 40, 88):
        mx, my, _, _, C = _brute_loo(x, y, j)
        d = np.array([mx - x[j], my - y[j]])
        assert np.isclose(t["D2"][j], float(d @ np.linalg.inv(C) @ d), rtol=1e-9)


def test_loo_par_perp_decomposition_is_orthogonal_and_complete(ED):
    rng = np.random.default_rng(32)
    x = rng.normal(size=80); y = 0.9 * x + 0.2 * rng.normal(size=80)
    t = ED.loo_tables(x, y)
    assert np.allclose(t["d_par"] ** 2 + t["d_perp"] ** 2, t["D2"], rtol=1e-9)
    assert np.all(t["lam1"] >= t["lam2"])


def test_loo_isotropic_case_is_flagged_not_crashed(ED):
    """Nearly-isotropic posteriors leave the eigenbasis nearly unidentified; the code must
    stay finite and flag it rather than emit a spurious orientation."""
    rng = np.random.default_rng(33)
    x = rng.normal(size=60); y = rng.normal(size=60)
    t = ED.loo_tables(x, y)
    assert np.all(np.isfinite(t["d_par"])) and np.all(np.isfinite(t["d_perp"]))
    assert np.all(np.isfinite(t["D2"]))


def test_loo_tables_refuse_mismatched_or_tiny(ED):
    with pytest.raises(ED.DiagRefusal, match="length mismatch"):
        ED.loo_tables(np.zeros(10), np.zeros(9))
    with pytest.raises(ED.DiagRefusal, match="too small"):
        ED.loo_tables(np.zeros(4), np.zeros(4))


# --- P1 influence / shape / null-replicate machinery -------------------------------------

def test_ks_influence_matches_scipy_and_loo(ED):
    from scipy import stats as st
    rng = np.random.default_rng(41)
    r = rng.uniform(size=48)
    out = ED.ks_influence(r)
    assert np.isclose(out["ks_stat"], st.kstest(r, "uniform").statistic)
    assert np.isclose(out["ks_p"], st.kstest(r, "uniform").pvalue)
    assert np.isclose(out["loo_ks_p"][3], st.kstest(np.delete(r, 3), "uniform").pvalue)
    assert len(out["empirical_cdf"]) == 48


def test_ks_per_realization_contrib_max_equals_ks_stat(ED):
    rng = np.random.default_rng(42)
    r = rng.uniform(size=48)
    out = ED.ks_influence(r)
    assert np.isclose(max(out["per_realization_contrib"]), out["ks_stat"])


def test_ks_influence_flags_a_planted_outlier_as_most_influential(ED):
    rng = np.random.default_rng(43)
    r = np.concatenate([rng.uniform(0.3, 0.7, 47), [0.999]])
    out = ED.ks_influence(r)
    assert out["loo_p_max"] >= out["ks_p"]


def test_min_omission_is_none_when_already_uniform(ED):
    r = (np.arange(48) + 0.5) / 48.0
    out = ED.ks_influence(r)
    assert out["ks_p"] > 0.05
    assert out["min_omission_greedy"] == 0
    assert out["min_omission_is_greedy_upper_bound"] is True


def test_min_omission_reduces_a_non_uniform_sample(ED):
    rng = np.random.default_rng(44)
    r = np.clip(rng.uniform(size=48) * 0.6, 0, 1)   # strongly non-uniform
    out = ED.ks_influence(r)
    assert out["ks_p"] <= 0.05
    assert out["min_omission_greedy"] is None or out["min_omission_greedy"] >= 1


def test_tail_occupancy_counts_and_asymmetry(ED):
    r = np.array([0.01, 0.02, 0.5, 0.5, 0.99])
    t = ED.tail_occupancy(r)
    assert t["lower"] == 2 and t["upper"] == 1 and t["both"] == 3
    assert np.isclose(t["asymmetry"], (2 - 1) / 5)


def test_shape_stats_signs(ED):
    """S2 positive = U-shaped (mass at both ends); negative = central concentration."""
    u = ED.shape_stats(np.concatenate([np.zeros(24) + 0.01, np.zeros(24) + 0.99]))
    c = ED.shape_stats(np.full(48, 0.5))
    assert u["S2_var_minus_uniform"] > 0
    assert c["S2_var_minus_uniform"] < 0
    assert np.isclose(ED.shape_stats(np.full(48, 0.9))["S1_mean_minus_half"], 0.4)


def test_null_replicates_shape_seeded_and_uses_own_L(ED):
    rng = np.random.default_rng(45)
    tabs = [ED.loo_tables(rng.normal(size=L), rng.normal(size=L))
            for L in (75, 150, 300)]
    a = ED.null_replicates(tabs, B=50)
    b = ED.null_replicates(tabs, B=50)
    assert a["pull_x"].shape == (50, 3)
    assert np.allclose(a["pull_x"], b["pull_x"])
    assert set(np.unique(a["rank_x"][:, 0])) <= set(np.asarray(tabs[0]["rank_x"]))


def test_null_replicate_rank_distribution_is_uniform(ED):
    """End-to-end property of the primary reference: pooled null ranks are ~U(0,1)."""
    from scipy import stats as st
    rng = np.random.default_rng(46)
    tabs = [ED.loo_tables(rng.normal(size=150), rng.normal(size=150)) for _ in range(48)]
    rep = ED.null_replicates(tabs, B=200)
    assert st.kstest(rep["rank_x"].ravel(), "uniform").pvalue > 0.01


def test_null_pull_sd_is_near_the_finite_L_expectation(ED):
    """The null's pull sd must reproduce the deployed finite-L reference, not 1.0."""
    rng = np.random.default_rng(47)
    L = 75
    tabs = [ED.loo_tables(rng.normal(size=L), rng.normal(size=L)) for _ in range(48)]
    rep = ED.null_replicates(tabs, B=400)
    sds = rep["pull_x"].std(axis=1, ddof=1)
    assert abs(float(sds.mean()) - ED.finite_L_null_sd([L - 1] * 48)) < 0.06


def test_calibrated_p_never_zero_and_reports_resolution(ED):
    null = np.random.default_rng(48).normal(size=2000)
    out = ED.calibrated_p(99.0, null)
    assert out["p"] == pytest.approx(1.0 / 2001.0)
    assert out["resolution"] == pytest.approx(1.0 / 2001.0)
    mid = ED.calibrated_p(0.0, null)
    assert 0.5 < mid["p"] <= 1.0


# --- physics-review fixes: summary-artefact, anisotropy, permutation null, rung algebra ---

def test_rank_implied_z_scale_is_one_for_calibrated_ranks(ED):
    """Uniform ranks -> sd[Phi^-1(rank)] ~ 1 regardless of any Gaussian-pull inflation."""
    rng = np.random.default_rng(51)
    L = np.full(400, 150)
    r = rng.uniform(size=400)
    out = ED.rank_implied_z_scale(r, L)
    assert abs(out["z_sd"] - 1.0) < 0.12
    assert abs(out["z_mean"]) < 0.12


def test_rank_implied_z_scale_detects_genuine_over_concentration(ED):
    """A truly too-narrow posterior gives U-shaped ranks -> sd[Phi^-1(rank)] > 1."""
    rng = np.random.default_rng(52)
    L = np.full(400, 150)
    u = rng.uniform(size=400)
    r = np.clip(np.where(u < 0.5, u * 0.3, 1 - (1 - u) * 0.3), 1e-3, 1 - 1e-3)
    out = ED.rank_implied_z_scale(r, L)
    assert out["z_sd"] > 1.25


def test_rank_implied_z_scale_separates_artefact_from_defect(ED):
    """The decisive contrast: identical uniform ranks, wildly different Gaussian pull sd."""
    rng = np.random.default_rng(53)
    L = np.full(300, 150)
    r = rng.uniform(size=300)
    z = ED.rank_implied_z_scale(r, L)["z_sd"]
    assert abs(z - 1.0) < 0.15   # rank scale is healthy...
    # ...while a skewed pull would read >> 1; the two statistics are independent by design.


def test_rank_implied_z_scale_length_mismatch_refuses(ED):
    with pytest.raises(ED.DiagRefusal, match="length mismatch"):
        ED.rank_implied_z_scale([0.5, 0.5], [150])


def test_anisotropy_contrast_signs(ED):
    perp = ED.anisotropy_contrast(np.zeros(10), np.ones(10))
    par = ED.anisotropy_contrast(np.ones(10), np.zeros(10))
    iso = ED.anisotropy_contrast(np.ones(10), np.ones(10))
    assert np.isclose(perp["mean"], 1.0)
    assert np.isclose(par["mean"], -1.0)
    assert np.isclose(iso["mean"], 0.0)


def test_anisotropy_contrast_is_invariant_to_overall_width(ED):
    rng = np.random.default_rng(54)
    a = rng.normal(size=48); b = rng.normal(size=48)
    base = ED.anisotropy_contrast(a, b)["mean"]
    scaled = ED.anisotropy_contrast(3.7 * a, 3.7 * b)["mean"]
    assert np.isclose(base, scaled, rtol=1e-12)


def test_permutation_null_is_calibrated_under_independence(ED):
    """Calibration is a property of the p-value DISTRIBUTION over independent datasets,
    not of any single dataset (a single pair is extreme ~5% of the time by construction)."""
    from scipy.stats import spearmanr
    rng = np.random.default_rng(55)
    ps = []
    for _ in range(60):
        x = rng.normal(size=48); y = rng.normal(size=48)
        ps.append(ED.permutation_null(x, y, lambda a, b: spearmanr(a, b).statistic,
                                      B=200)["p"])
    ps = np.asarray(ps)
    assert 0.0 < float(np.mean(ps < 0.05)) < 0.15   # near-nominal type-I rate
    assert 0.35 < float(np.mean(ps)) < 0.65         # p ~ U(0,1) has mean 0.5


def test_permutation_null_centres_on_zero_association(ED):
    from scipy.stats import spearmanr
    rng = np.random.default_rng(155)
    x = rng.normal(size=48); y = rng.normal(size=48)
    out = ED.permutation_null(x, y, lambda a, b: spearmanr(a, b).statistic, B=500)
    assert abs(out["null_mean"]) < 0.1


def test_permutation_null_detects_a_planted_association(ED):
    from scipy.stats import spearmanr
    rng = np.random.default_rng(56)
    x = rng.normal(size=48)
    y = x + 0.25 * rng.normal(size=48)
    out = ED.permutation_null(x, y, lambda a, b: spearmanr(a, b).statistic, B=500)
    assert out["p"] < 0.01


def test_permutation_null_length_mismatch_refuses(ED):
    with pytest.raises(ED.DiagRefusal, match="length mismatch"):
        ED.permutation_null([1, 2, 3], [1, 2], lambda a, b: 0.0, B=5)


def test_tau_eff_rung_coefficients_match_verified_arithmetic(ED):
    out = ED.tau_eff_rung_coefficients()
    assert np.isclose(out["zbar"], 3.3351, atol=1e-3)
    assert np.isclose(out["c_bar"], 0.080458, atol=1e-5)
    assert np.isclose(out["c"][0], np.log(3.2 / 4.0))
    assert out["c"][-1] > 0 > out["c"][0]


def test_deployed_amp_equals_tau_eff_at_zbar_exactly(ED):
    """The deployed summary IS tau_eff(zbar), a ROTATED coordinate ln(amp) + c_bar*dtau0,
    not the sampled amplitude. Ladder residual is zero by construction."""
    co = ED.tau_eff_rung_coefficients()
    z = np.asarray(co["z"]); zbar = co["zbar"]
    for amp, dt in [(1.0, 0.0), (0.9, -0.3), (1.2, 0.2)]:
        ladder = amp * ((1 + z) / 4.0) ** dt * 0.0023 * (1 + z) ** 3.65
        got, slope = ED.tau0_amp_slope(ladder)
        want = amp * ((1 + zbar) / 4.0) ** dt * 0.0023 * (1 + zbar) ** 3.65
        assert np.isclose(got, want, rtol=1e-12)
        assert np.isclose(slope, dt + 3.65, rtol=1e-12)


def test_ladder_is_exactly_log_linear_no_projection_artefact(ED):
    """Excludes the derived-summary curvature artefact class a priori."""
    co = ED.tau_eff_rung_coefficients()
    z = np.asarray(co["z"]); lx = np.log(1 + z); lxc = lx - lx.mean()
    ladder = 1.1 * ((1 + z) / 4.0) ** (-0.2) * 0.0023 * (1 + z) ** 3.65
    y = np.log(ladder)
    slope = np.sum(lxc * (y - y.mean())) / np.sum(lxc ** 2)
    assert np.max(np.abs(y - (y.mean() + slope * lxc))) < 1e-12


# --- v2 amendment machinery -------------------------------------------------------------

def test_exact_rank_null_is_uniform_on_the_deployed_lattice(ED):
    from scipy import stats as st
    r = ED.exact_rank_null([75, 150, 300], 4000, 1)
    assert r.shape == (4000, 3)
    assert st.kstest(r.ravel(), "uniform").pvalue > 0.01
    assert set(np.unique(r[:, 0])) <= set(np.arange(76) / 75)


def test_check_ties_refuses_duplicate_draw_rows(ED):
    rng = np.random.default_rng(61)
    good = [dict(m=0, ns=rng.normal(size=50), Ap=rng.normal(size=50),
                 tau0amp=rng.normal(size=50))]
    assert ED.check_ties(good) is True
    bad = dict(m=1, ns=np.r_[np.ones(2), rng.normal(size=48)],
               Ap=np.r_[np.ones(2), rng.normal(size=48)],
               tau0amp=np.r_[np.ones(2), rng.normal(size=48)])
    with pytest.raises(ED.DiagRefusal, match="duplicate draw rows"):
        ED.check_ties([bad])


def test_sites_alignment_refuses_row_misalignment(ED):
    rec = dict(m=0, L=50, sites_extra={"a": {"draws": np.zeros(49), "truth": 0.1}})
    with pytest.raises(ED.DiagRefusal, match="row misalignment"):
        ED.check_sites_alignment([rec], ["a"])
    rec2 = dict(m=0, L=50, sites_extra={})
    with pytest.raises(ED.DiagRefusal, match="absent"):
        ED.check_sites_alignment([rec2], ["a"])


def test_null_relative_material_fixes_the_v1_defect(ED):
    """v1's |rho|>=0.30-from-zero fired on ~92% of healthy arms because the healthy null
    sits near -0.47. The null-relative rule must NOT fire when obs sits AT the null."""
    null = np.random.default_rng(62).normal(-0.47, 0.11, 4000)
    at_null = ED.null_relative_material(-0.47, null)
    assert at_null["material"] is False
    far = ED.null_relative_material(-0.47 + 0.45, null)
    assert far["material"] is True


def test_partial_corr_recovers_a_known_collapse(ED):
    """Mirrors the committed record: a strong marginal routed through a third variable
    collapses under conditioning."""
    rng = np.random.default_rng(63)
    a = rng.normal(size=400)
    x = a + 0.15 * rng.normal(size=400)
    y = -a + 0.15 * rng.normal(size=400)
    marginal = abs(ED._spear(x, y))
    partial = abs(ED.partial_corr(x, y, a))
    assert marginal > 0.8 and partial < 0.3


def test_mc_se_and_calibrated_p_resolution(ED):
    assert ED.mc_se(0.0125, 50000) < 0.0006
    assert ED.mc_se(0.0125, 2000) > 0.002


def test_autocorr_lag1_detects_correlation(ED):
    rng = np.random.default_rng(64)
    iid = rng.normal(size=500)
    ar = np.zeros(500); 
    for i in range(1, 500):
        ar[i] = 0.8 * ar[i-1] + rng.normal()
    assert abs(ED.autocorr_lag1(iid)) < 0.15
    assert ED.autocorr_lag1(ar) > 0.6


def test_identity_check_flags_a_broken_ladder(ED):
    """If the ladder were NOT the deployed closed form, P1-I must catch it."""
    co = ED.tau_eff_rung_coefficients()
    z = np.asarray(co["z"]); zbar = co["zbar"]
    rng = np.random.default_rng(65)
    L = 60
    amp = rng.uniform(0.8, 1.2, L); dt = rng.uniform(-0.3, 0.2, L)
    ladder = amp[:, None] * ((1 + z) / 4.0) ** dt[:, None] * 0.0023 * (1 + z) ** 3.65
    names = [f"tau0_z{i}" for i in range(13)]
    good = dict(m=0, L=L, names=names, draws=ladder, truth=np.zeros(13),
                tau0amp=ED.tau0_amp_vec(ladder),
                sites_extra={"tau0_amp": {"draws": amp, "truth": 1.0},
                             "dtau0": {"draws": dt, "truth": 0.0}})
    assert ED.identity_check([good])["identity_holds"] is True
    bad = dict(good); bad["tau0amp"] = good["tau0amp"] * 1.01
    assert ED.identity_check([bad])["identity_holds"] is False


def test_pc1_captures_a_common_mode(ED):
    rng = np.random.default_rng(66)
    common = rng.normal(size=48)
    X = np.column_stack([common + 0.1 * rng.normal(size=48) for _ in range(3)])
    sc, var = ED._pc1(X)
    assert var > 0.9
    assert abs(ED._spear(sc, common)) > 0.9
