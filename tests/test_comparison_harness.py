"""Cross-emulator comparison harness (hcd_analysis/comparison): schema, conventions, metrics, outcomes.

Synthetic records only (a made-up smooth P(k) on 8 box modes); no emulator or reference is loaded. Each test names
the mistake it guards against: units, coordinates, parameter order, masking, versions, outcome accounting."""
import copy
import json

import numpy as np
import pytest

from hcd_analysis.comparison import cases as CS
from hcd_analysis.comparison import conventions as C
from hcd_analysis.comparison import metrics as M
from hcd_analysis.comparison import schema as S
from hcd_analysis.comparison.adapters import lace

L, MODES, Z = 120.0, list(range(1, 9)), [2.2, 3.0, 4.6]
PARAMS = {"ns": 0.95, "Ap": 1.9e-9, "herei": 4.0, "heref": 2.7, "alphaq": 2.1, "hub": 0.7, "omegamh2": 0.143,
          "hireionz": 7.2, "bhfeedback": 0.05}
MF = {"tau0_amp": 1.0, "dtau0": 0.0, "z_pivot": 3.0}


def _case(cid="c1", family="I", params=PARAMS, mf=MF, scale=1.0):
    kc = C.k_com_modes(np.asarray(MODES), L)
    K = [C.k_skm_from_kcom(kc, z, params["hub"], params["omegamh2"]).tolist() for z in Z]
    P = [(scale * (1 + z) * 10.0 / (1 + (np.asarray(k) / 0.01) ** 2)).tolist() for z, k in zip(Z, K)]
    return {"case_id": cid, "family": family, "z": list(Z), "state": {"params": dict(params), "mean_flux": dict(mf)},
            "support": {"status": "COMPARABLE", "reason": ""}, "P": P, "k_skm": K, "native": {}}


def _rec(cases=None, cid="T"):
    return {"schema": S.SCHEMA, "observable": "O1", "L_box_hmpc": L, "modes": list(MODES),
            "comparator": {"id": cid, "name": "synthetic", "repo": "none", "commit": "abc", "implementation": {},
                           "artifacts": [{"path": "x", "sha256": "0" * 64}], "fidelity": "LF", "training": "none",
                           "env": {}},
            "cases": cases if cases is not None else [_case()]}


# --------------------------------------------------------------------------------------------- conventions
def test_k_and_P_conversions_round_trip_and_h_cancels():
    kc = C.k_com_modes(np.arange(1, 173), L)
    for z in (2.2, 3.0, 4.6):
        ks = C.k_skm_from_kcom(kc, z, 0.7, 0.143)
        assert np.allclose(C.kcom_from_kskm(ks, z, 0.7, 0.143), kc, rtol=1e-14, atol=0)
        P = np.linspace(1.0, 5.0, 172)
        assert np.allclose(C.P_hmpc_from_P_kms(C.P_kms_from_P_hmpc(P, z, 0.7, 0.143), z, 0.7, 0.143), P,
                           rtol=1e-14, atol=0)
        # only Om = omegamh2 / hub^2 enters: scaling hub and omegamh2 together at fixed Om leaves k_skm unchanged
        assert np.allclose(C.k_skm_from_kcom(kc, z, 0.6, 0.143 * (0.6 / 0.7) ** 2), ks, rtol=1e-14, atol=0)
    assert C.k_com_modes(np.array([1]), L)[0] == pytest.approx(2 * np.pi / L, rel=1e-15)
    with pytest.raises(C.ConventionError):
        C.k_com_modes(np.array([0.5]), L)


def test_mean_flux_conventions_are_explicit():
    assert C.alpha_priya(3.0, {"tau0_amp": 1.1, "dtau0": 0.3, "z_pivot": 3.0}) == pytest.approx(1.1, rel=1e-15)
    t = C.hcd_priya_tau0_input(3.0, {"tau0_amp": 1.1, "dtau0": 0.3, "z_pivot": 3.0})
    assert t == pytest.approx(1.1 * 0.0023 * 4.0 ** 3.65, rel=1e-14)      # an optical depth, not PRIYA's tau0
    with pytest.raises(C.ConventionError):
        C.alpha_priya(3.0, {"tau0": 1.1, "dtau0": 0.0, "z_pivot": 3.0})     # the ambiguous name is refused
    with pytest.raises(C.ConventionError):
        C.check_mean_flux({"tau0_amp": 1.0, "dtau0": 0.0})                  # the pivot must be explicit


def test_reference_vector_is_built_by_name_not_position():
    index = {n: i for i, n in enumerate(C.PARAM_NAMES)}
    shuffled = dict(zip(C.PARAM_NAMES, [3, 0, 8, 1, 7, 2, 6, 4, 5]))
    v1, v2 = C.vector_from_names(PARAMS, index), C.vector_from_names(PARAMS, shuffled)
    assert v1[index["ns"]] == v2[shuffled["ns"]] == PARAMS["ns"]
    assert v1[index["Ap"]] == v2[shuffled["Ap"]] == PARAMS["Ap"]
    assert C.params_from_vector(v2, sorted(shuffled, key=shuffled.get)) == PARAMS
    with pytest.raises(C.ConventionError):
        C.vector_from_names(PARAMS, {**index, "n_p": 9})                    # a LaCE name is not a PRIYA name
    with pytest.raises(C.ConventionError):
        C.vector_from_names({**PARAMS, "n_s": 0.96}, index)


# --------------------------------------------------------------------------------------------- schema
def test_valid_record_round_trips_exactly():
    rec = _rec()
    assert S.loads(S.dumps(rec)) == json.loads(json.dumps(rec))


@pytest.mark.parametrize("mutate,match", [
    (lambda r: r.update(schema="hcd_priya.compare/0"), "schema"),
    (lambda r: r.update(observable="P1D_unfiltered"), "observable"),
    (lambda r: r["comparator"].update(fidelity="medium"), "fidelity"),
    (lambda r: r["comparator"].update(artifacts=[{"path": "x", "sha256": "abc"}]), "sha256"),
    (lambda r: r.update(modes=[2, 1, 3]), "modes"),
    (lambda r: r["cases"][0]["state"]["params"].update(tau0=1.0), "parameters must be exactly"),
    (lambda r: r["cases"][0]["state"]["params"].pop("bhfeedback"), "parameters must be exactly"),
    (lambda r: r["cases"][0]["state"]["mean_flux"].pop("z_pivot"), "mean-flux"),
    (lambda r: r["cases"][0]["P"][1].__setitem__(3, float("nan")), "non-finite"),
    (lambda r: r["cases"][0]["P"][1].__setitem__(3, -1.0), "non-positive"),
    (lambda r: r["cases"][0]["P"].pop(), "P shape"),
    (lambda r: r["cases"][0].update(support={"status": "BLOCKED", "reason": " "}), "needs a reason"),
    (lambda r: r["cases"].append(copy.deepcopy(r["cases"][0])), "duplicate"),
])
def test_validator_refuses(mutate, match):
    rec = _rec()
    mutate(rec)
    with pytest.raises(S.ComparisonSchemaError, match=match):
        S.validate_record(rec)


@pytest.mark.parametrize("wrong_k", [
    lambda kc, z, p: kc,                                                              # h/Mpc labelled as s/km
    lambda kc, z, p: C.k_skm_from_kcom(kc / (2 * np.pi), z, p["hub"], p["omegamh2"]),  # cyclic k (missing 2 pi)
    lambda kc, z, p: C.k_skm_from_kcom(kc, z, p["hub"], 0.15),                        # wrong cosmology
    lambda kc, z, p: C.k_skm_from_kcom(kc * p["hub"], z, p["hub"], p["omegamh2"]),    # 1/Mpc instead of h/Mpc
])
def test_validator_refuses_inconsistent_k_coordinates(wrong_k):
    rec = _rec()
    kc = C.k_com_modes(np.asarray(MODES), L)
    c = rec["cases"][0]
    c["k_skm"] = [wrong_k(kc, z, c["state"]["params"]).tolist() for z in c["z"]]
    with pytest.raises(S.ComparisonSchemaError, match="does not follow"):
        S.validate_record(rec)


def test_declared_cases_need_no_arrays():
    c = _case()
    for k in ("P", "k_skm"):
        c.pop(k)
    c["support"] = {"status": "NOT COMPARABLE", "reason": "alpha outside the reference rungs"}
    S.validate_record(_rec([c]))


# --------------------------------------------------------------------------------------------- comparison
def test_identical_records_pass_with_zero_residual():
    out = S.compare_records(_rec(), _rec(cid="R"), 0.02)
    assert out["verdict"] == "PASS" and out["max_abs"] == 0.0 and out["points"] == len(Z) * len(MODES)


def test_small_offset_passes_and_one_bad_point_fails_without_clipping():
    assert S.compare_records(_rec([_case(scale=1.015)]), _rec(), 0.02)["verdict"] == "PASS"
    bad = _rec()
    bad["cases"][0]["P"][2][5] *= 1.03
    out = S.compare_records(bad, _rec(), 0.02)
    assert out["verdict"] == "FAIL" and out["max_abs"] == pytest.approx(0.03, rel=1e-12)
    assert out["summary"]["exceed_frac"] == pytest.approx(1 / (len(Z) * len(MODES)))


def test_unit_mistake_in_power_is_caught_as_fail():
    wrong = _rec()
    c = wrong["cases"][0]
    p = c["state"]["params"]
    c["P"] = [C.P_hmpc_from_P_kms(P, z, p["hub"], p["omegamh2"]).tolist() for z, P in zip(c["z"], c["P"])]
    assert S.compare_records(wrong, _rec(), 0.02)["verdict"] == "FAIL"


def test_parameter_order_mistake_is_refused_not_compared():
    swapped = dict(PARAMS, ns=PARAMS["Ap"], Ap=PARAMS["ns"])
    with pytest.raises(S.ComparisonSchemaError, match="different physical states"):
        S.compare_records(_rec([_case(params=swapped)]), _rec(), 0.02)
    with pytest.raises(S.ComparisonSchemaError, match="different physical states"):
        S.compare_records(_rec([_case(mf={"tau0_amp": 1.0, "dtau0": 0.1, "z_pivot": 3.0})]), _rec(), 0.02)


def test_mismatched_nodes_or_observables_are_refused_not_intersected():
    short = _rec()
    short["modes"] = MODES[:-1]
    for c in short["cases"]:
        c["P"] = [row[:-1] for row in c["P"]]
        c["k_skm"] = [row[:-1] for row in c["k_skm"]]
    with pytest.raises(S.ComparisonSchemaError, match="nodes differ"):
        S.compare_records(short, _rec(), 0.02)
    other = _rec()
    S.OBSERVABLES["O_test"] = dict(S.OBSERVABLES["O1"])
    try:
        other["observable"] = "O_test"
        with pytest.raises(S.ComparisonSchemaError, match="observables differ"):
            S.compare_records(other, _rec(), 0.02)
    finally:
        S.OBSERVABLES.pop("O_test")


def test_outcome_accounting_never_turns_missing_into_pass():
    nc = _case("c2")
    nc["support"] = {"status": "NOT COMPARABLE", "reason": "outside the rungs"}
    out = S.compare_records(_rec([_case("c1"), nc, _case("c3")]), _rec([_case("c1"), _case("c2")]), 0.02)
    assert out["cases"] == {"c1": "PASS", "c2": "NOT COMPARABLE", "c3": "NOT RUN"}
    assert out["verdict"] == "PASS" and out["points"] == len(Z) * len(MODES)    # only the comparable case counts
    only_missing = S.compare_records(_rec([_case("c9")]), _rec([_case("c1")]), 0.02)
    assert only_missing["verdict"] == "NOT RUN" and only_missing["max_abs"] is None
    blocked = lace.blocked_record([_case("c1")], pins={"name": "LaCE", "repo": "r", "commit": "c", "reason": "pickle"},
                                  n_modes=len(MODES), L_box_hmpc=L)
    assert S.compare_records(_rec(), blocked, 0.05)["verdict"] == "BLOCKED"


def test_metrics_on_known_values():
    r = np.array([0.01, -0.02, 0.03, 0.0])
    s = M.summarize(r, 0.02)
    assert s["max_abs"] == pytest.approx(0.03) and s["mean"] == pytest.approx(0.005)
    assert s["median_abs"] == pytest.approx(0.015) and s["exceed_frac"] == pytest.approx(0.25)
    assert s["rms"] == pytest.approx(np.sqrt((1 + 4 + 9) / 4) * 0.01)
    assert M.verdict(r, 0.02) == "FAIL" and M.verdict(r[:2], 0.02) == "PASS" and M.verdict([], 0.02) == "NOT RUN"
    assert list(M.band_index([0.0, 0.002, 0.003, 0.04, 0.2], [0.0, 0.003, 0.01, 0.02, 0.05])) == [0, 0, 1, 3, 4]
    with pytest.raises(ValueError):
        M.ratio_residual([1.0], [0.0])


# --------------------------------------------------------------------------------------------- frozen config
def test_frozen_config_matches_the_spec():
    cfg = CS.load_config("phase0_v1")
    assert cfg["schema"] == S.SCHEMA and cfg["observable"] in S.OBSERVABLES
    assert cfg["thresholds"]["L2_priya"] == 0.02 and cfg["thresholds"]["L3_lace"] == 0.05
    assert cfg["z_grid"] == [round(2.2 + 0.2 * i, 1) for i in range(13)] and cfg["z_pivot"] == 3.0
    assert cfg["families"]["D"]["ks_lf_sims"] == [24, 34, 42] and cfg["families"]["D"]["rung_indices"] == [1, 5, 8]
    assert cfg["families"]["I"]["sobol"]["seed"] == 20261022
    assert cfg["comparators"]["R2"]["status"] == cfg["comparators"]["R3"]["status"] == "BLOCKED"


def test_domain_check_declares_out_of_rung_states():
    cfg = CS.load_config("phase0_v1")
    lim = {n: (0.0, 10.0) for n in C.PARAM_NAMES}
    case = {"z": cfg["z_grid"], "state": {"params": dict(PARAMS), "mean_flux": dict(MF)}}
    assert CS.domain_support(case, cfg, lim)["status"] == "COMPARABLE"
    case["state"]["mean_flux"] = {"tau0_amp": 1.25, "dtau0": 0.25, "z_pivot": 3.0}
    assert CS.domain_support(case, cfg, lim)["status"] == "NOT COMPARABLE"
    case["state"]["mean_flux"] = dict(MF)
    case["state"]["params"]["ns"] = 20.0
    assert CS.domain_support(case, cfg, lim)["status"] == "NOT COMPARABLE"
