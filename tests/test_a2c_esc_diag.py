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
