"""TDD for the X-battery-2 readout (scripts/analyze_xsel.py): gate arithmetic on synthetic
inputs, sigma_post pooling, ingest refusals (registry signature, truth-table sha, K0
cross-signature allowance, pair identity, duplicates, r6 refusal), the per-arm truth
contracts (corner / profile / K8 map image / X1 swap identity), and the driver-vs-analyzer
K8 map-mirror agreement. Synthetic pkls + synthetic sha-pinned table; NO fits, NO ctx build.
New test file only (frozen tests untouched)."""
import copy
import os
import pickle

import numpy as np
import pytest

import scripts.ks_xsel_arms as XA
import scripts.analyze_xsel as AX
from tests.test_ks_xsel_arms import make_synth_table

Z_GLOBAL = np.round(np.arange(2.2, 4.61, 0.2), 10)
NAMES = ["ns", "Ap", "tau0_z0", "tau0_z1", "alpha_lls", "alpha_subdla", "alpha_dla"]
SITES = ("tau0_amp", "dtau0", "eps_lls", "kappa_lls", "m_sub", "t_sub", "dla_raw", "t_dla")
REF_Z = (np.array([0.014, 0.0045, 0.0012])[None, :]
         * (1.0 + 0.05 * np.arange(Z_GLOBAL.size))[:, None])
XBAR_Z = 28.0 + 0.5 * np.arange(Z_GLOBAL.size)
PC = dict(hcd_parameterization="dndx_mapped_v2",
          ks_dndx_sigma_eps=0.5310, ks_dndx_sigma_kappa=0.6681,
          ks_dndx_ref_z=REF_Z.tolist(), ks_xbar_z=XBAR_Z.tolist(),
          lls_frac_sigma_ks=0.40, lls_survey_boost_ks=2.5)
FORWARD = dict(forward_signature="f" * 40, hcd_prior_signature="h" * 40)
RUN_KW = dict(n_warmup=250, n_samples=300, seed=XA.PAIR_SEED, max_tree_depth=10,
              dense_mass=True)
L_DRAWS = 24


# ---------------------------------------------------------------- synthetic battery

def _clean_rec(m, n_rows):
    """The shared clean draw of mock m (the K0 record and every arm's underlying draw)."""
    rng = np.random.default_rng(1000 + m)
    truth = np.concatenate([rng.random(4), [0.30, 0.06, 0.01]])
    draws = truth[None, :] + rng.normal(0.0, 0.05, (L_DRAWS, len(NAMES)))
    sites = {nm: dict(draws=rng.normal(0, 1, L_DRAWS), truth=float(rng.normal(0, 0.4)))
             for nm in SITES}
    rows = np.abs(rng.normal(0, 0.02, (Z_GLOBAL.size, 3))) + \
        np.array([0.25, 0.05, 0.008])[None, :]
    return dict(sim="leg_a_prior", truth_vec=truth, draws=draws, L=L_DRAWS,
                ll_true=0.0, ll_draws=np.zeros(L_DRAWS), names=list(NAMES),
                kept_global=np.ones(Z_GLOBAL.size, bool), dropped={"KS": []},
                sites_extra=sites, n_div=0, truth_alpha_hcd_z=rows)


def _arm_rec(arm, m, tt_ratios, delta_shift=0.0):
    """An arm record sharing mock m's clean draw; ``delta_shift`` moves the ns/Ap posterior
    mean (in raw units) to exercise the gate."""
    e = XA.ARMS[arm]
    rec = _clean_rec(m, len(tt_ratios["ratio_rows_X1_dla100"]))
    rec["draws"] = rec["draws"].copy()
    rec["draws"][:, :2] += delta_shift
    tv = rec["truth_vec"].copy()
    if e["kind"] == "mixture_corner":
        piv = np.zeros(3)
        piv[e["cls"]] = 1.0
        tv[4:] = piv
        rec["truth_alpha_hcd_z"] = XA.corner_alpha_rows(Z_GLOBAL.size, e["cls"])
    elif e["kind"] == "mixture_profile":
        piv = np.zeros(3)
        piv[e["cls"]] = float(XA.f_sel(float(XA.Z_PIVOT)))
        tv[4:] = piv
        rec["truth_alpha_hcd_z"] = XA.profile_alpha_rows(Z_GLOBAL, e["cls"])
    elif e["kind"] == "dndx_displaced":
        st = {nm: rec["sites_extra"][nm]["truth"] for nm in SITES}
        sig = PC["ks_dndx_sigma_eps"] if e["site"] == "eps_lls" else PC["ks_dndx_sigma_kappa"]
        st[e["site"]] = st[e["site"]] + e["n_sigma"] * sig
        rec["sites_extra"][e["site"]]["truth"] = st[e["site"]]
        sites = {nm: st[nm] for nm in AX._K8_SITES}
        rec["truth_alpha_hcd_z"] = AX.k8_rows_from_sites(sites, REF_Z, XBAR_Z, Z_GLOBAL)
        tv[4:] = rec["truth_alpha_hcd_z"][np.argmin(np.abs(Z_GLOBAL - 3.0))]
    else:                                        # data_swap (X1): identical clean truth
        rng = np.random.default_rng(5000 + m)
        P_clean = 1.0 + rng.random(len(tt_ratios["ratio_rows_X1_dla100"]))
        rec["xsel_swap"] = dict(P_clean_truth=P_clean,
                                P_swap_truth=P_clean * tt_ratios["ratio_rows_X1_dla100"],
                                eps=rng.normal(0, 0.01, P_clean.size),
                                fork="dilution_corrected")
    rec["truth_vec"] = tv
    return rec


def _meta(arm, reg_sig, sha, table_path):
    e = XA.ARMS.get(arm, {})
    return dict(n_mocks=(XA.K0_N_MOCKS if arm == XA.K0_ARM_ID else e["n_mocks"]),
                run_kw=dict(RUN_KW), forward=dict(FORWARD),
                prior_constants=copy.deepcopy(PC),
                span=dict(cache_alpha_max=[0.5252, 0.5488, 0.1490],
                          truth_alpha_max=([1.0, 0.0, 0.0]
                                           if e.get("kind") == "mixture_corner"
                                           else [0.3, 0.06, 0.01]),
                          out_of_span_classes=(["lls"] if e.get("kind") == "mixture_corner"
                                               and e.get("cls") == 0 else []),
                          ),
                arm_stamp=dict(
                    arm_id=arm, kind=e.get("kind"), campaign=XA.CAMPAIGN,
                    registry_signature=reg_sig,
                    truth_table=dict(path=str(table_path), sha256=sha),
                    z_global=Z_GLOBAL.tolist(),
                    f_sel_z_global=(XA.f_sel(Z_GLOBAL).tolist() if arm == "X4_prof"
                                    else None),
                    displacement=(dict(site=e["site"], n_sigma=e["n_sigma"],
                                       sigma_deployed=e["sigma_expect"])
                                  if e.get("kind") == "dndx_displaced" else None)))


ARMS_IN_TEST = ("X1_dla100", "X2_sub100", "X4_prof", "K8a_eps_hi")


def build_battery(tmp_path, arms=ARMS_IN_TEST, delta_shift=0.0):
    """Write a synthetic (table, X shard dir, K0 dir); returns (shard_dir, k0_dir, path, sha)."""
    table_path, sha = make_synth_table(tmp_path / "tt.npz")
    tt = XA.load_truth_tables(table_path, expect_sha=sha)
    reg_sig = XA.registry_signature(table_path, expect_sha=sha)
    import scripts.ks_selboost_arms as AR
    legacy_sig = AR.registry_signature()
    shard_dir = tmp_path / "xsel"
    k0_dir = tmp_path / "k0"
    shard_dir.mkdir()
    k0_dir.mkdir()
    for m in range(XA.K0_N_MOCKS):
        meta = _meta(XA.K0_ARM_ID, legacy_sig, sha, table_path)
        with open(k0_dir / f"ks_selboost_clean_shard_{m:03d}.pkl", "wb") as fh:
            pickle.dump(dict(arm=XA.K0_ARM_ID, survey="ks", mode="clean", idxs=[m],
                             per_mock=[_clean_rec(m, tt["leg_k"].size)], meta=meta), fh)
    for arm in arms:
        for m in range(XA.ARMS[arm]["n_mocks"]):
            meta = _meta(arm, reg_sig, sha, table_path)
            with open(shard_dir / XA.shard_pkl_name(arm, m), "wb") as fh:
                pickle.dump(dict(arm=arm, survey="ks", mode="x", idxs=[m],
                                 per_mock=[_arm_rec(arm, m, tt["ratios"],
                                                    delta_shift=delta_shift)],
                                 meta=meta), fh)
    return str(shard_dir), str(k0_dir), str(table_path), sha


def _tamper_one(shard_dir, fname, fn):
    p = os.path.join(shard_dir, fname)
    with open(p, "rb") as fh:
        d = pickle.load(fh)
    fn(d)
    with open(p, "wb") as fh:
        pickle.dump(d, fh)


# ---------------------------------------------------------------- gate arithmetic units

def test_gate_stats_arithmetic():
    # three-outcome revised pre-registration (2026-07-26 memo): t-quantile intervals
    gs = AX.gate_stats([0.1, 0.2, 0.3, 0.2])
    assert abs(gs["mean"] - 0.2) < 1e-12
    assert abs(gs["se"] - np.std([0.1, 0.2, 0.3, 0.2], ddof=1) / 2.0) < 1e-12
    assert abs(gs["ub2"] - (0.2 + 2 * gs["se"])) < 1e-12            # legacy key kept
    assert abs(gs["sigma_pair"] - np.std([0.1, 0.2, 0.3, 0.2], ddof=1)) < 1e-12
    assert abs(gs["ub_t"] - (0.2 + gs["t975"] * gs["se"])) < 1e-12
    assert abs(gs["lb_t"] - (0.2 - gs["t975"] * gs["se"])) < 1e-12
    # ub_t 0.33 > 0.30 and lb_t 0.07 < 0.30 -> the t-interval straddles the budget
    # (under the OLD z=2 gate this case was a PASS: the t-quantile fix is visible)
    assert gs["verdict"] == "UNDER-RESOLVED" and gs["ub_t"] > gs["ub2"]
    # tight case: interval inside the budget -> PROTECTED
    assert AX.gate_stats([0.05, 0.06, 0.04, 0.05])["verdict"] == "PROTECTED"
    # zero-scatter above budget -> UNPROTECTED (se = 0, inf t handled)
    gs0 = AX.gate_stats([0.5, 0.5, 0.5, 0.5])
    assert gs0["verdict"] == "UNPROTECTED" and gs0["p_t"] == 0.0
    assert AX.gate_stats([-0.2, -0.4, -0.3])["ub2"] > 0.3          # sign-symmetric
    with pytest.raises(AssertionError):
        AX.gate_stats([0.1])


def test_part2_disclosure_arithmetic():
    gs = AX.gate_stats([0.1, 0.2, 0.3, 0.2])
    p2 = AX.part2_disclosure(gs, dict(D=0.5, sigma_pair_expected=0.1,
                                      p_part1_fail_null=0.02))
    assert abs(p2["S"] - gs["ub2"] / 0.5) < 1e-12
    assert abs(p2["noise_floor_2se_over_D"] - 2 * gs["se"] / 0.5) < 1e-12
    assert p2["p_part1_fail_null"] == 0.02


def test_sigma_post_pooled():
    rng = np.random.default_rng(0)
    recs = {m: dict(names=["ns"], draws=rng.normal(0, 0.1, (200, 1)),
                    truth_vec=np.zeros(1), sites_extra={}) for m in range(4)}
    want = np.sqrt(np.mean([np.asarray(recs[m]["draws"][:, 0]).var(ddof=1)
                            for m in range(4)]))
    assert abs(AX.sigma_post_pooled(recs, "ns") - want) < 1e-12


# ---------------------------------------------------------------- round trip + gates

def test_summarize_round_trip_pass(tmp_path):
    shard_dir, k0_dir, tpath, sha = build_battery(tmp_path)
    res = AX.summarize(shard_dir, k0_dir, tpath, expect_sha=sha)
    assert set(res["arms"]) == set(ARMS_IN_TEST)
    assert set(res["binding"]) == {"X1_dla100", "X2_sub100"}       # X4/K8a non-binding
    for a in res["binding"]:
        for p in AX.GATE_PARAMS:
            # shared draws => deltas identically (mean_x - mean_k0)/sigma; shift 0
            # => zero-width interval inside the budget => PROTECTED
            assert res["binding"][a][p]["verdict"] == "PROTECTED", (a, p)
    for p in AX.PAIRED_PARAMS:
        assert res["sigma_post"][p] > 0
    # Part-2 disclosure carries the stage-V gate-power inputs
    p2 = res["part2"]["X2_sub100"]["ns"]
    assert p2["D"] == 0.5 and p2["p_part1_fail_null"] == 0.01
    assert not res["any_binding_fail"]


def test_summarize_detects_binding_fail(tmp_path):
    # a large raw-units shift on ns/Ap in every arm rec => bias detected above budget
    # => UNPROTECTED on the binding arms (feeds the corner-failure protocol)
    shard_dir, k0_dir, tpath, sha = build_battery(tmp_path, delta_shift=0.5)
    res = AX.summarize(shard_dir, k0_dir, tpath, expect_sha=sha)
    assert res["any_binding_fail"]
    assert res["binding"]["X2_sub100"]["ns"]["verdict"] == "UNPROTECTED"


# ---------------------------------------------------------------- ingest refusals

def test_refuses_stamped_table_sha_mismatch(tmp_path):
    shard_dir, k0_dir, tpath, sha = build_battery(tmp_path)
    _tamper_one(shard_dir, XA.shard_pkl_name("X2_sub100", 3),
                lambda d: d["meta"]["arm_stamp"]["truth_table"].update(sha256="0" * 64))
    with pytest.raises(AssertionError, match="truth-table sha"):
        AX.summarize(shard_dir, k0_dir, tpath, expect_sha=sha)


def test_refuses_registry_signature_drift(tmp_path):
    shard_dir, k0_dir, tpath, sha = build_battery(tmp_path)
    _tamper_one(shard_dir, XA.shard_pkl_name("X1_dla100", 0),
                lambda d: d["meta"]["arm_stamp"].update(registry_signature="a" * 64))
    with pytest.raises(AssertionError, match="registry_signature"):
        AX.summarize(shard_dir, k0_dir, tpath, expect_sha=sha)


def test_refuses_run_kw_seed_drift(tmp_path):
    shard_dir, k0_dir, tpath, sha = build_battery(tmp_path)
    _tamper_one(shard_dir, XA.shard_pkl_name("X2_sub100", 1),
                lambda d: d["meta"]["run_kw"].update(seed=20260724))
    with pytest.raises(AssertionError, match="core run_kw"):
        AX.summarize(shard_dir, k0_dir, tpath, expect_sha=sha)


def test_refuses_r6_override_k0(tmp_path):
    shard_dir, k0_dir, tpath, sha = build_battery(tmp_path)
    _tamper_one(k0_dir, "ks_selboost_clean_shard_000.pkl",
                lambda d: d["meta"]["prior_constants"].update(r6_override=True))
    with pytest.raises(AssertionError, match="r6_override"):
        AX.summarize(shard_dir, k0_dir, tpath, expect_sha=sha)


def test_refuses_k0_with_wrong_registry_signature(tmp_path):
    shard_dir, k0_dir, tpath, sha = build_battery(tmp_path)
    _tamper_one(k0_dir, "ks_selboost_clean_shard_002.pkl",
                lambda d: d["meta"]["arm_stamp"].update(registry_signature="b" * 64))
    with pytest.raises(AssertionError, match="reuse allowance"):
        AX.summarize(shard_dir, k0_dir, tpath, expect_sha=sha)


def test_refuses_broken_pair_theta(tmp_path):
    shard_dir, k0_dir, tpath, sha = build_battery(tmp_path)

    def _break(d):
        d["per_mock"][0]["truth_vec"] = np.asarray(d["per_mock"][0]["truth_vec"]).copy()
        d["per_mock"][0]["truth_vec"][0] += 1e-6
    _tamper_one(shard_dir, XA.shard_pkl_name("X2_sub100", 5), _break)
    with pytest.raises(AssertionError, match="pairing broken"):
        AX.summarize(shard_dir, k0_dir, tpath, expect_sha=sha)


def test_refuses_corner_contract_violation(tmp_path):
    shard_dir, k0_dir, tpath, sha = build_battery(tmp_path)

    def _break(d):
        rows = np.asarray(d["per_mock"][0]["truth_alpha_hcd_z"]).copy()
        rows[2, 1] = 0.97                       # not the pure corner any more
        d["per_mock"][0]["truth_alpha_hcd_z"] = rows
    _tamper_one(shard_dir, XA.shard_pkl_name("X2_sub100", 0), _break)
    with pytest.raises(AssertionError, match="arm contract"):
        AX.summarize(shard_dir, k0_dir, tpath, expect_sha=sha)


def test_refuses_x1_swap_identity_violation(tmp_path):
    shard_dir, k0_dir, tpath, sha = build_battery(tmp_path)

    def _break(d):
        sw = d["per_mock"][0]["xsel_swap"]
        sw["P_swap_truth"] = np.asarray(sw["P_swap_truth"]).copy() * 1.001
    _tamper_one(shard_dir, XA.shard_pkl_name("X1_dla100", 7), _break)
    with pytest.raises(AssertionError, match="swap identity"):
        AX.summarize(shard_dir, k0_dir, tpath, expect_sha=sha)


def test_refuses_k8_displacement_mismatch(tmp_path):
    shard_dir, k0_dir, tpath, sha = build_battery(tmp_path)
    _tamper_one(shard_dir, XA.shard_pkl_name("K8a_eps_hi", 4),
                lambda d: d["per_mock"][0]["sites_extra"]["eps_lls"].update(
                    truth=d["per_mock"][0]["sites_extra"]["eps_lls"]["truth"] + 0.01))
    with pytest.raises(AssertionError, match="displaced eps_lls|frozen-map image"):
        AX.summarize(shard_dir, k0_dir, tpath, expect_sha=sha)


def test_refuses_incomplete_arm(tmp_path):
    shard_dir, k0_dir, tpath, sha = build_battery(tmp_path)
    os.remove(os.path.join(shard_dir, XA.shard_pkl_name("X4_prof", 9)))
    with pytest.raises(AssertionError, match="incomplete"):
        AX.summarize(shard_dir, k0_dir, tpath, expect_sha=sha)


def test_refuses_duplicate_mock(tmp_path):
    shard_dir, k0_dir, tpath, sha = build_battery(tmp_path)
    src = os.path.join(shard_dir, XA.shard_pkl_name("X2_sub100", 2))
    with open(src, "rb") as fh:
        d = pickle.load(fh)
    # a second pkl claiming the same (arm, mock)
    with open(os.path.join(shard_dir, "ks_xsel_X2_sub100_shard_099.pkl"), "wb") as fh:
        pickle.dump(d, fh)
    with pytest.raises(AssertionError, match="duplicate mock"):
        AX.summarize(shard_dir, k0_dir, tpath, expect_sha=sha)


def test_smoke_pkls_never_pool(tmp_path):
    shard_dir, k0_dir, tpath, sha = build_battery(tmp_path)
    # a smoke pkl with a poisoned payload must be invisible to the glob
    with open(os.path.join(shard_dir, "ks_xsel_X2_sub100_shard_000.smoke.pkl"), "wb") as fh:
        pickle.dump({"poison": True}, fh)
    res = AX.summarize(shard_dir, k0_dir, tpath, expect_sha=sha)
    assert set(res["arms"]) == set(ARMS_IN_TEST)


# ---------------------------------------------------------------- K8 mirror agreement

def test_k8_driver_vs_analyzer_map_mirror():
    """The driver's jnp mirror (run_xsel_shard.k8_alpha_from_raw) and the analyzer's numpy
    mirror (analyze_xsel.k8_rows_from_sites) are INDEPENDENT reimplementations sharing only
    the frozen w_c_corrected; they must agree to float precision (displaced and not)."""
    RX = pytest.importorskip("scripts.run_xsel_shard")
    raw = dict(eps_lls=0.21, kappa_lls=-0.4, m_sub=0.1, t_sub=0.3, dla_raw=0.6, t_dla=-0.2)
    rows_drv, piv_drv = RX.k8_alpha_from_raw(
        raw, REF_Z, XBAR_Z, Z_GLOBAL, REF_Z[np.argmin(np.abs(Z_GLOBAL - 3.0))],
        XBAR_Z[np.argmin(np.abs(Z_GLOBAL - 3.0))], d_eps=0.5310)
    sites = dict(raw, eps_lls=raw["eps_lls"] + 0.5310)
    rows_ana = AX.k8_rows_from_sites(sites, REF_Z, XBAR_Z, Z_GLOBAL)
    assert np.allclose(rows_drv, rows_ana, rtol=0, atol=1e-12)
    assert XA.truth_admissible(rows_drv, strict_interior=True)
    assert piv_drv.shape == (3,) and np.all(piv_drv > 0) and piv_drv.sum() < 1


def test_driver_truth_builders_copy_semantics():
    RX = pytest.importorskip("scripts.run_xsel_shard")
    tp = dict(theta9=np.arange(9.0), tau0_global=np.ones(13),
              alpha_hcd=np.array([0.3, 0.06, 0.01]),
              alpha_hcd_z=np.full((13, 3), 0.05), a_siiii=0.0,
              kept_global_z=np.ones(13, bool), raw={})
    out = RX.build_corner_truth(tp, 1, 13)
    assert np.array_equal(out["alpha_hcd"], [0.0, 1.0, 0.0])
    assert np.array_equal(out["alpha_hcd_z"], XA.corner_alpha_rows(13, 1))
    assert np.array_equal(tp["alpha_hcd"], [0.3, 0.06, 0.01])      # input NOT mutated
    assert np.all(tp["alpha_hcd_z"] == 0.05)
    out4 = RX.build_profile_truth(tp, 0, Z_GLOBAL)
    assert np.allclose(out4["alpha_hcd_z"][:, 0], XA.f_sel(Z_GLOBAL))
    assert out4["alpha_hcd"][0] == XA.f_sel(3.0) and out4["alpha_hcd"][1:].sum() == 0
    assert np.all(tp["alpha_hcd_z"] == 0.05)


# ------------------------------------------- pre-registered disclosures (2026-07-26 memo)

def _rec_with_alpha(vals, cls=0):
    """Minimal record whose alpha_<cls> draws are ``vals`` (span-occupancy unit)."""
    names = ["ns", "Ap", "alpha_lls", "alpha_subdla", "alpha_dla"]
    n = len(vals)
    draws = np.zeros((n, len(names)))
    draws[:, 2 + cls] = np.asarray(vals, float)
    return dict(names=names, draws=draws, truth_vec=np.zeros(len(names)), L=n,
                sites_extra={}, n_div=0)


SPAN = dict(cache_alpha_max=[0.50, 0.55, 0.15], truth_alpha_max=[1.0, 0.0, 0.0],
            out_of_span_classes=["lls"])


def test_span_occupancy_separates_edge_pileup_from_scatter():
    pinned = AX.span_occupancy(_rec_with_alpha([0.48, 0.49, 0.50, 0.499]), SPAN, 0)
    assert pinned["frac_at_edge"] == 1.0          # every draw above 0.9 x 0.50
    assert pinned["frac_above_span"] == 0.0       # but none ABOVE the span
    assert pinned["cls"] == "lls" and pinned["cache_alpha_max"] == 0.50
    scattered = AX.span_occupancy(_rec_with_alpha([0.10, 0.20, 0.25, 0.30]), SPAN, 0)
    assert scattered["frac_at_edge"] == 0.0
    over = AX.span_occupancy(_rec_with_alpha([0.60, 0.10]), SPAN, 0)
    assert over["frac_above_span"] == 0.5


def test_arm_span_occupancy_aggregates_and_carries_out_of_span():
    recs = {0: _rec_with_alpha([0.48, 0.49]), 1: _rec_with_alpha([0.10, 0.20])}
    spans = {0: SPAN, 1: dict(SPAN, out_of_span_classes=[])}
    agg = AX.arm_span_occupancy(recs, spans, 0)
    assert agg["mean_frac_at_edge"] == 0.5 and agg["max_frac_at_edge"] == 1.0
    assert agg["truth_out_of_span_classes"] == ["lls"]      # union over the arm's fits
    assert agg["per_fit_frac_at_edge"] == [1.0, 0.0]


def test_pilot_excluded_sensitivity_only_for_piloted_arms(tmp_path):
    shard_dir, k0_dir, tp, sha = build_battery(tmp_path)
    res = AX.summarize(shard_dir, k0_dir, tp, sha)
    for a in ("X1_dla100", "X2_sub100", "X4_prof"):
        gs = res["nopilot"][a]["ns"]
        assert gs is not None and gs["n"] == XA.ARMS[a]["n_mocks"] - len(AX.PILOT_MOCKS)
    # K8 was never piloted: no exclusion is owed, and its mocks 0-3 must NOT be dropped.
    assert res["nopilot"]["K8a_eps_hi"]["ns"] is None
    assert res["stats"]["K8a_eps_hi"]["ns"]["n"] == XA.ARMS["K8a_eps_hi"]["n_mocks"]


def test_span_occupancy_reaches_the_arm_outputs(tmp_path):
    shard_dir, k0_dir, tp, sha = build_battery(tmp_path)
    res = AX.summarize(shard_dir, k0_dir, tp, sha)
    out = tmp_path / "readout"
    AX._write_arm_outputs(res, "X2_sub100", str(out))
    z = np.load(out / "xsel_X2_sub100.npz", allow_pickle=False)
    assert str(z["span_class"]) == "subdla"
    assert 0.0 <= float(z["span_mean_frac_at_edge"]) <= 1.0
    txt = (out / "xsel_X2_sub100.txt").read_text()
    assert "span occupancy" in txt and "PILOT-EXCLUDED SENSITIVITY" in txt
