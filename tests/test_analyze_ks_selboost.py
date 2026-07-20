"""TDD suite for scripts/analyze_ks_selboost.py — the KS selection-boost campaign analyzer
(spec Sec 6 + A2.6-A2.7, PI-signed 2026-07-19). Conventions follow analyze_dla_selfdraw
post-1876a14 (by-name indexing, exact t-quantile pooling, smoke filtering) with the KS deltas:

  * ARM-RESOLVED fail-loud ingest: clean set (K0) loaded once, every boosted arm grouped and
    CROSS-PKL paired against it by mock index. Per-pair asserts: identical forward_signature /
    prior-constants stamp / registry_signature / run_kw / seed; theta9 truth identity (the
    pairing certificate); boosted pivot alpha_lls truth == B_arm(3.0) x clean (rtol 1e-12);
    row-level truth_alpha_hcd_z LLS column == B_arm(z_global) x clean column (the new
    row contract; subDLA column too for K5); all non-boosted truth entries bit-identical;
    per-arm completeness {0..N_arm-1}; smoke pkls filtered.
  * design coordinates: WLS of ln B_a(z) on {1, x, x^2} over the KS-band z rows (A2.6),
    r_a = weighted RMS residual after the quadratic; D_a = sqrt(mean lnB^2)/sigma_frac from
    the STAMPED deployed width (never hard-coded).
  * per-arm Part-1 gates at 0.30 with the exact t-quantile ub (reused pool_deltas);
  * pooled 3-vector response surface (S_A, S_eta1, S_eta2) through the origin, jackknife over
    mocks; Part-2 S = max over mis-centered arms of (|mean| + 2 SE)/D_a vs 0.50; Part-3
    projection lines at the K4 coordinates.

Env: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_analyze_ks_selboost.py -q
"""
import pickle

import numpy as np
import pytest

import scripts.ks_selboost_arms as AR
import scripts.analyze_ks_selboost as AN

SIG = "68f71a3d" + "0" * 56
NAMES = ["ns", "Ap", "tau0_z0", "alpha_lls", "alpha_subdla", "alpha_dla"]
ZG = [2.2, 2.4, 3.0, 3.6, 4.4]          # includes one sub-KS-band row (the DESI-union row)
PRIOR = dict(lls_survey_boost_ks=2.5, lls_frac_sigma_ks=0.40, lit_over_sim_lls=0.995,
             lit_over_sim_slope_lls=0.764, lls_base_frac_sigma=0.287,
             lls_hedge2x_frac_sigma=0.574, lls_realfit_zslope=2.127,
             hcd_prior_signature="p" * 64, alpha_lls_center_built=0.4303,
             alpha_lls_sigma_built=0.17212)
RUN_KW = dict(n_warmup=250, n_samples=300, seed=20260615, dense_mass=True, max_tree_depth=10)
REG_SIG = AR.registry_signature()

# small synthetic campaign: clean control + two boosted arms (one flat, one shaped) + K5
TEST_ARMS = {
    "K2_flat_hi": {"lls_truth_boost": {"b_pivot": 1.667, "eta1": 0.0, "eta2": 0.0}},
    "K3_rising": {"lls_truth_boost": {"b_pivot": 1.25, "eta1": 2.7, "eta2": 0.0}},
    "K5_joint_meas": {
        "lls_truth_boost": {"z": [2.0, 3.0, 5.0], "boost": [1.12, 1.048, 1.03],
                            "interp": "loglog"},
        "subdla_truth_boost": {"z": [2.0, 3.0, 5.0], "boost": [1.29, 1.162, 1.07],
                               "interp": "loglog"},
    },
}
N_BY_ARM = {"K0_clean": 3, "K2_flat_hi": 3, "K3_rising": 3, "K5_joint_meas": 2}


def _profiles_B(arm, z):
    spec = TEST_ARMS.get(arm)
    if spec is None:
        return {"lls_truth_boost": np.ones(len(np.atleast_1d(z))),
                "subdla_truth_boost": np.ones(len(np.atleast_1d(z)))}
    out = {}
    for cls, prof in spec.items():
        out[cls] = np.atleast_1d(AR.eval_profile(prof, np.asarray(z, float)))
    return out


def _rec(truth, mean_shift=0.0, L=90, seed=0, alpha_z=None):
    rng = np.random.default_rng(seed)
    truth = np.asarray(truth, float)
    draws = rng.normal(truth + mean_shift, 0.05, size=(L, len(NAMES)))
    return {
        "sim": "leg_a_prior", "truth_vec": truth, "draws": draws, "L": L,
        "names": list(NAMES), "kept_global": np.ones(len(ZG), bool), "n_div": 0,
        "truth_alpha_hcd_z": np.asarray(alpha_z, float),
        "sites_extra": {
            "tau0_amp": {"draws": rng.normal(1.0, 0.02, L), "truth": 1.0},
            "dtau0": {"draws": rng.normal(0.0, 0.05, L), "truth": 0.0},
            "f_res_amp": {"draws": rng.normal(0.0, 0.1, L), "truth": np.nan},
        },
    }


def _truths(mock):
    """Deterministic clean truth for one mock index: theta-block (2 entries here), tau0 row,
    alpha pivots + z-resolved alpha rows."""
    rng = np.random.default_rng(100 + mock)
    tv = np.array([0.4, 0.8, 0.3, 0.18, 0.06, 0.004]) * (1.0 + 0.05 * rng.uniform())
    alpha_z = np.stack([tv[3:6] * (1.0 + 0.1 * i) for i in range(len(ZG))])
    return tv, alpha_z


def _boosted(tv, alpha_z, arm):
    B3 = {cls: AR.eval_profile(prof, 3.0) for cls, prof in TEST_ARMS[arm].items()}
    Bz = _profiles_B(arm, ZG)
    tvb, azb = tv.copy(), alpha_z.copy()
    tvb[3] *= B3["lls_truth_boost"]
    azb[:, 0] *= Bz["lls_truth_boost"]
    if "subdla_truth_boost" in TEST_ARMS[arm]:
        tvb[4] *= B3["subdla_truth_boost"]
        azb[:, 1] *= Bz["subdla_truth_boost"]
    return tvb, azb


def _meta(arm, n_mocks, **over):
    m = dict(
        arm_id=arm, n_mocks=n_mocks, seed=20260615, mode=("clean" if arm == "K0_clean"
                                                          else "boost"),
        run_kw=dict(RUN_KW, n_mocks=n_mocks),
        forward=dict(res_corr_on=False, fix_alpha_res=True, sample_res=True,
                     f_res_amp_sigma=0.15, metal_prior="uniform", dla_forward_frac=0.0,
                     forward_signature=SIG, hcd_prior_signature=PRIOR["hcd_prior_signature"]),
        prior_constants=dict(PRIOR),
        arm_stamp=dict(arm_id=arm, profiles=TEST_ARMS.get(arm),
                       B_pivot={cls: AR.eval_profile(p, 3.0)
                                for cls, p in (TEST_ARMS.get(arm) or {}).items()},
                       B_z_global={cls: list(v) for cls, v in _profiles_B(arm, ZG).items()
                                   if TEST_ARMS.get(arm) and cls in TEST_ARMS[arm]},
                       z_global=list(ZG), registry_signature=REG_SIG,
                       envelope=("in" if arm != "K1_flat_lo" else "out"), k5_table=None),
        span=dict(cache_alpha_max=[1.0, 1.0, 1.0], truth_alpha_max=[0.5, 0.2, 0.05],
                  out_of_span_classes=[]),
    )
    m.update(over)
    return m


def _pkl(path, arm, mock, rec, n_mocks, **meta_over):
    payload = dict(arm=arm, survey="ks", mode=("clean" if arm == "K0_clean" else "boost"),
                   idxs=[mock], per_mock=[rec], meta=_meta(arm, n_mocks, **meta_over))
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def _write_campaign(tmp_path, bias=None, mutate=None):
    """Write a complete consistent synthetic campaign. ``bias``: {arm: {param_idx: shift}}
    posterior-mean shifts on the boosted arms (paired deltas). ``mutate``: hook(payload_dict,
    path_name) applied before pickling for violation tests."""
    bias = bias or {}
    for arm, n in N_BY_ARM.items():
        for m in range(n):
            tv, az = _truths(m)
            if arm == "K0_clean":
                rec = _rec(tv, seed=m, alpha_z=az)
                name = f"ks_selboost_clean_shard_{m:03d}.pkl"
            else:
                tvb, azb = _boosted(tv, az, arm)
                shift = np.zeros(len(NAMES))
                for j, s in (bias.get(arm) or {}).items():
                    shift[j] = s
                rec = _rec(tvb, mean_shift=shift, seed=m, alpha_z=azb)
                name = f"ks_selboost_{arm}_shard_{m:03d}.pkl"
            payload = dict(arm=arm, survey="ks",
                           mode=("clean" if arm == "K0_clean" else "boost"),
                           idxs=[m], per_mock=[rec], meta=_meta(arm, n))
            if mutate is not None:
                mutate(payload, name)
            with open(tmp_path / name, "wb") as f:
                pickle.dump(payload, f)


# --------------------------------------------------------------------------------------------- #
#  Ingest: the pass case + each violation class rejected.
# --------------------------------------------------------------------------------------------- #
def test_ingest_pass(tmp_path):
    _write_campaign(tmp_path)
    clean, arms, meta = AN.load_campaign(str(tmp_path))
    assert set(clean) == {0, 1, 2}
    assert set(arms) == {"K2_flat_hi", "K3_rising", "K5_joint_meas"}
    assert set(arms["K5_joint_meas"]) == {0, 1}


def test_ingest_filters_smoke(tmp_path):
    _write_campaign(tmp_path)
    # a stray smoke pkl with a WRONG signature must be ignored, not poison the pool
    with open(tmp_path / "ks_selboost_K3_rising_shard_000.smoke.pkl", "wb") as f:
        pickle.dump({"arm": "K3_rising", "meta": {}}, f)
    clean, arms, meta = AN.load_campaign(str(tmp_path))
    assert len(arms["K3_rising"]) == 3


@pytest.mark.parametrize("field,frag", [
    ("forward_signature", "signature"),
    ("registry_signature", "registry"),
    ("prior", "prior"),
    ("run_kw", "run_kw"),
])
def test_ingest_rejects_mixed_stamps(tmp_path, field, frag):
    def mutate(payload, name):
        if "K3_rising_shard_001" not in name:
            return
        if field == "forward_signature":
            payload["meta"]["forward"]["forward_signature"] = "f" * 64
        elif field == "registry_signature":
            payload["meta"]["arm_stamp"]["registry_signature"] = "r" * 64
        elif field == "prior":
            payload["meta"]["prior_constants"]["lls_frac_sigma_ks"] = 0.29
        elif field == "run_kw":
            payload["meta"]["run_kw"]["n_samples"] = 600
    _write_campaign(tmp_path, mutate=mutate)
    with pytest.raises(AssertionError, match=frag):
        AN.load_campaign(str(tmp_path))


def test_ingest_rejects_theta_pairing_violation(tmp_path):
    def mutate(payload, name):
        if "K2_flat_hi_shard_000" in name:
            payload["per_mock"][0]["truth_vec"][0] += 1e-6      # theta drifts vs clean partner
    _write_campaign(tmp_path, mutate=mutate)
    with pytest.raises(AssertionError, match="theta"):
        AN.load_campaign(str(tmp_path))


def test_ingest_rejects_pivot_contract_violation(tmp_path):
    def mutate(payload, name):
        if "K2_flat_hi_shard_000" in name:
            payload["per_mock"][0]["truth_vec"][3] *= 1.01      # pivot != B(3) x clean
    _write_campaign(tmp_path, mutate=mutate)
    with pytest.raises(AssertionError, match="pivot"):
        AN.load_campaign(str(tmp_path))


def test_ingest_rejects_row_contract_violation(tmp_path):
    def mutate(payload, name):
        if "K3_rising_shard_002" in name:
            payload["per_mock"][0]["truth_alpha_hcd_z"][2, 0] *= 1.001
    _write_campaign(tmp_path, mutate=mutate)
    with pytest.raises(AssertionError, match="row"):
        AN.load_campaign(str(tmp_path))


def test_ingest_rejects_nonboosted_truth_drift(tmp_path):
    def mutate(payload, name):
        if "K2_flat_hi_shard_001" in name:
            payload["per_mock"][0]["truth_alpha_hcd_z"][1, 1] *= 1.001   # subDLA row (not K5)
    _write_campaign(tmp_path, mutate=mutate)
    with pytest.raises(AssertionError):
        AN.load_campaign(str(tmp_path))


def test_ingest_k5_subdla_row_contract_enforced(tmp_path):
    def mutate(payload, name):
        if "K5_joint_meas_shard_001" in name:
            payload["per_mock"][0]["truth_alpha_hcd_z"][3, 1] *= 1.001
    _write_campaign(tmp_path, mutate=mutate)
    with pytest.raises(AssertionError):
        AN.load_campaign(str(tmp_path))


def test_ingest_rejects_incomplete_arm(tmp_path):
    _write_campaign(tmp_path)
    (tmp_path / "ks_selboost_K3_rising_shard_002.pkl").unlink()
    with pytest.raises(AssertionError, match="incomplete|missing"):
        AN.load_campaign(str(tmp_path))


def test_ingest_rejects_registry_drift(tmp_path):
    """A campaign whose pkls were stamped under a DIFFERENT registry than the current module
    (homogeneous among themselves) must be rejected: gate membership is read from the live
    registry, so silent drift would change the binding gate set."""
    def mutate(payload, name):
        payload["meta"]["arm_stamp"]["registry_signature"] = "d" * 64   # homogeneous but stale
    _write_campaign(tmp_path, mutate=mutate)
    with pytest.raises(AssertionError, match="CURRENT|registry"):
        AN.load_campaign(str(tmp_path))


def test_part2_noise_floor_flag(tmp_path):
    """Small-displacement arms (K5 at the measured truth) whose 2SE/D noise floor alone
    exceeds the 0.50 threshold are FLAGGED noise-dominated (consistency-review FLAG A; a PI
    disposition item, not a silent analyzer fix)."""
    _write_campaign(tmp_path)
    res = AN.summarize_campaign(str(tmp_path))
    pa = res["part2"]["per_arm"]["K5_joint_meas"]
    assert "noise_floor_2se_over_D" in pa and "noise_dominated" in pa
    # K5's D at these tables ~ 0.13 prior-sigma; with per-mock delta noise its floor exceeds
    # the smallest arms' by construction, and the flag list is exposed for the report
    assert "noise_dominated_arms" in res["part2"]


def test_part2_excludes_null_bound_k5_from_binding_max(tmp_path):
    """FLAG A resolution (PI-signed A2.8-c), landed in code 2026-07-20: the null-bound arm
    K5 (registry part2_binding=False) is EXCLUDED from the binding Part-2 max and reported
    on a separate null-bound consistency line; the raw all-arms max is kept for transparency
    only. Guards against the binding S silently reverting to the ill-conditioned K5 term."""
    _write_campaign(tmp_path)
    res = AN.summarize_campaign(str(tmp_path))
    p2 = res["part2"]
    # K5 is marked non-binding and never the argmax of the binding S
    assert p2["per_arm"]["K5_joint_meas"]["part2_binding"] is False
    assert p2["argmax"] is None or p2["argmax"][0] != "K5_joint_meas"
    # null-bound line carries K5's raw sensitivity (reported, not gated)
    assert "K5_joint_meas" in p2["null_bound_line"]
    for par in ("ns", "Ap"):
        assert par in p2["null_bound_line"]["K5_joint_meas"]
    # transparency field exists and is >= the binding S (raw includes K5)
    assert p2["S_raw_allarms"] >= p2["S"] - 1e-12
    # every binding arm's registry flag is honored
    for a, e in p2["per_arm"].items():
        assert e["part2_binding"] == bool(AR.ARMS[a].get("part2_binding", True))


def test_ingest_rejects_stamped_B_mismatch(tmp_path):
    # the stamped evaluated vector must match the profile re-evaluated at ingest
    def mutate(payload, name):
        if "K3_rising_shard_000" in name:
            payload["meta"]["arm_stamp"]["B_z_global"]["lls_truth_boost"][2] *= 1.01
    _write_campaign(tmp_path, mutate=mutate)
    with pytest.raises(AssertionError, match="B_z|stamped"):
        AN.load_campaign(str(tmp_path))


# --------------------------------------------------------------------------------------------- #
#  Design coordinates + displacement scalars (hand-computable profiles).
# --------------------------------------------------------------------------------------------- #
def test_design_coords_flat_and_slope_and_quad():
    z = np.array([2.4, 2.8, 3.2, 3.6, 4.0, 4.4])
    x = np.log((1 + z) / 4.0)
    # flat
    dA, de1, de2, r = AN.design_coords(np.log(1.667) * np.ones(z.size), z)
    assert dA == pytest.approx(np.log(1.667), rel=1e-12)
    assert de1 == pytest.approx(0.0, abs=1e-12) and de2 == pytest.approx(0.0, abs=1e-12)
    assert r == pytest.approx(0.0, abs=1e-12)
    # pure log-quadratic member recovered exactly (basis-exact)
    lnB = np.log(1.25) + 2.7 * x + 13.0 * x * x
    dA, de1, de2, r = AN.design_coords(lnB, z)
    assert (dA, de1, de2) == (pytest.approx(np.log(1.25), rel=1e-9),
                              pytest.approx(2.7, rel=1e-9), pytest.approx(13.0, rel=1e-9))
    assert r == pytest.approx(0.0, abs=1e-9)


def test_displacement_scalar_flat():
    z = np.linspace(2.4, 4.6, 12)
    D = AN.displacement_D(np.log(1.667) * np.ones(z.size), 0.40)
    assert D == pytest.approx(abs(np.log(1.667)) / 0.40, rel=1e-12)   # = 1.278


def test_arm_coordinates_restrict_to_ks_band(tmp_path):
    _write_campaign(tmp_path)
    clean, arms, meta = AN.load_campaign(str(tmp_path))
    c = AN.arm_coordinates(meta["K3_rising"])
    # z rows restricted to [2.4, 4.6] (the 2.2 union row dropped): 4 of the 5 ZG rows
    assert c["n_z"] == 4
    assert c["dA"] == pytest.approx(np.log(1.25), rel=1e-6)
    assert c["de1"] == pytest.approx(2.7, rel=1e-6)
    assert c["de2"] == pytest.approx(0.0, abs=1e-6)
    assert c["D"] > 0


# --------------------------------------------------------------------------------------------- #
#  Pooled response surface + Part-2 S + collapse check on synthetic linear responses.
# --------------------------------------------------------------------------------------------- #
def test_pooled_surface_recovers_linear_response(tmp_path):
    # inject a pure amplitude response on ns: delta = S_A * dA  (S_A = 0.5/ln-unit)
    S_A = 0.5
    d2 = S_A * np.log(1.667) * 0.05      # posterior sd is 0.05 -> mean shift in raw units
    d3 = S_A * np.log(1.25) * 0.05
    _write_campaign(tmp_path, bias={"K2_flat_hi": {0: d2}, "K3_rising": {0: d3}})
    res = AN.summarize_campaign(str(tmp_path))
    S = res["surface"]["ns"]
    # with two arms and three coefficients the fit is underdetermined in de2 -> the analyzer
    # must restrict to identifiable columns; S_A recovered within MC noise
    assert S["S_A"] == pytest.approx(S_A, abs=0.2)


def test_part1_gate_table_and_part2_max(tmp_path):
    _write_campaign(tmp_path)
    res = AN.summarize_campaign(str(tmp_path))
    g = res["part1"]
    # gate set: envelope="in" mis-centered arms present (K2, K3; K5 via the if-in-envelope rule)
    assert "K2_flat_hi" in g and "K3_rising" in g
    for armres in g.values():
        for p in ("ns", "Ap"):
            assert "ub" in armres[p] and "verdict" in armres[p]
    # null campaign: everything passes the 0.30 gate comfortably at sd 0.05 / L 90
    assert all(armres[p]["verdict"] == "PASS" for armres in g.values() for p in ("ns", "Ap"))
    s2 = res["part2"]
    assert s2["S"] >= 0 and s2["threshold"] == 0.50
    assert set(s2["per_arm"]) == {"K2_flat_hi", "K3_rising", "K5_joint_meas"}


def test_part2_uses_displacement_normalization(tmp_path):
    _write_campaign(tmp_path)
    res = AN.summarize_campaign(str(tmp_path))
    pa = res["part2"]["per_arm"]["K2_flat_hi"]
    # (|mean| + 2 SE)/D with D = ln(1.667)/0.40 from the STAMPED width
    D = abs(np.log(1.667)) / 0.40
    assert pa["D"] == pytest.approx(D, rel=1e-6)


def test_report_runs_and_mentions_tau0(tmp_path, capsys):
    _write_campaign(tmp_path)
    AN.main_report(str(tmp_path))
    out = capsys.readouterr().out
    assert "tau0_amp" in out and "dtau0" in out          # mandatory tau0/dtau0 reporting
    assert "PART 1" in out and "PART 2" in out and "PART 3" in out
    assert "alpha_lls" in out
