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
    assert S.loads(S.dumps(rec)) == rec
    bad = _rec()
    bad["cases"][0]["native"] = {"std": [float("nan")]}
    with pytest.raises(ValueError):                                          # no non-standard JSON (NaN) is written
        S.dumps(bad)


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


def _nc(cid):
    c = _case(cid)
    c["support"] = {"status": "NOT COMPARABLE", "reason": "outside the rungs"}
    return c


def test_outcome_accounting_never_turns_missing_or_blocked_into_pass():
    # a case declared NOT COMPARABLE on both sides is excluded; the rest passes
    out = S.compare_records(_rec([_case("c1"), _nc("c2")]), _rec([_case("c1"), _nc("c2")]), 0.02)
    assert out["cases"] == {"c1": "PASS", "c2": "NOT COMPARABLE"} and out["verdict"] == "PASS" and out["complete"]
    assert out["points"] == len(Z) * len(MODES)                                  # only the comparable case counts
    # a case missing from one side makes the level NOT RUN, even when everything present passes
    out = S.compare_records(_rec([_case("c1"), _case("c3")]), _rec([_case("c1")]), 0.02)
    assert out["cases"]["c3"] == "NOT RUN" and out["verdict"] == "NOT RUN" and not out["complete"]
    # ... but a real failure is never hidden by incompleteness
    bad = _rec([_case("c1", scale=1.05), _case("c3")])
    assert S.compare_records(bad, _rec([_case("c1")]), 0.02)["verdict"] == "FAIL"
    only_missing = S.compare_records(_rec([_case("c9")]), _rec([_case("c1")]), 0.02)
    assert only_missing["verdict"] == "NOT RUN" and only_missing["max_abs"] is None
    # BLOCKED on either side blocks the level
    rb = _rec([_case("c1"), _case("c2")])
    rb["cases"][1]["support"] = {"status": "BLOCKED", "reason": "asset missing"}
    out = S.compare_records(_rec([_case("c1"), _case("c2")]), rb, 0.02)
    assert out["cases"]["c2"] == "BLOCKED" and out["verdict"] == "BLOCKED"


def test_one_sided_exclusion_is_refused():
    with pytest.raises(S.ComparisonSchemaError, match="support differs"):
        S.compare_records(_rec([_case("c1")]), _rec([_nc("c1")]), 0.02)


def test_lace_stub_is_blocked_with_an_undefined_observable():
    pins = {"name": "LaCE", "repo": "r", "commit": "c", "reason": "pickle"}
    blocked = lace.blocked_record([_case("c1")], pins=pins, n_modes=len(MODES), L_box_hmpc=L)
    assert blocked["observable"] == S.UNDEFINED and "HCD" in blocked["cases"][0]["support"]["reason"]
    out = S.compare_records(_rec(), blocked, 0.05)
    assert out["verdict"] == "BLOCKED" and out["cases"] == {"c1": "BLOCKED"} and out["points"] == 0
    claimed = _rec()
    claimed["observable"] = S.UNDEFINED                                     # an UNDEFINED record cannot carry data
    with pytest.raises(S.ComparisonSchemaError, match="UNDEFINED"):
        S.validate_record(claimed)


def test_metrics_on_known_values():
    r = np.array([0.01, -0.02, 0.03, 0.0])
    s = M.summarize(r, 0.02)
    assert s["max_abs"] == pytest.approx(0.03) and s["mean"] == pytest.approx(0.005)
    assert s["median_abs"] == pytest.approx(0.015) and s["exceed_frac"] == pytest.approx(0.25)
    assert s["rms"] == pytest.approx(np.sqrt((1 + 4 + 9) / 4) * 0.01)
    assert M.verdict(r, 0.02) == "FAIL" and M.verdict(r[:2], 0.02) == "PASS" and M.verdict([], 0.02) == "NOT RUN"
    assert list(M.band_index([0.0, 0.002, 0.003, 0.04, 0.2], [0.0, 0.003, 0.01, 0.02, 0.05])) == [0, 0, 1, 3, 4]
    with pytest.raises(ValueError):
        M.band_index([-1e-3], [0.0, 0.003])
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


# --------------------------------------------------------------------------------------------- case matrix builders
def test_family_D_reads_the_reference_design_by_name():
    cfg = CS.load_config("phase0_v1")
    rng = np.random.default_rng(5)
    design = rng.uniform(1.0, 2.0, size=(50, 9))
    shuffled = dict(zip(C.PARAM_NAMES, [4, 2, 0, 8, 6, 1, 3, 5, 7]))
    rungs = np.linspace(0.6, 1.3, 10)
    cases = CS.family_D(cfg, design, shuffled, rungs)
    assert len(cases) == 9 and {c["inputs"]["ks_lf_index"] for c in cases} == {24, 34, 42}
    c = next(c for c in cases if c["inputs"]["ks_lf_index"] == 34 and c["inputs"]["rung_index"] == 5)
    for n, col in shuffled.items():
        assert c["state"]["params"][n] == design[34, col]
    assert c["state"]["mean_flux"] == {"tau0_amp": rungs[5], "dtau0": 0.0, "z_pivot": 3.0}


def test_family_I_points_are_deterministic_and_inside_the_mapped_box():
    cfg = CS.load_config("phase0_v1")
    lim = {n: (float(i), float(i) + 2.0) for i, n in enumerate(C.PARAM_NAMES)}
    a, b = CS.family_I(cfg, lim), CS.family_I(cfg, lim)
    assert a == b and len(a) == 15
    i2 = next(c for c in a if c["case_id"] == "I2-M1")["state"]["params"]
    assert all(i2[n] == lim[n][0] + 1.0 for n in C.PARAM_NAMES)                 # the box centre
    for c in a:
        if c["case_id"][:2] in ("I3", "I4", "I5"):
            u = [(c["state"]["params"][n] - lim[n][0]) / 2.0 for n in C.PARAM_NAMES]
            assert all(0.15 - 1e-12 <= x <= 0.85 + 1e-12 for x in u)
    assert C.params_from_vector([lim[n][0] for n in C.PARAM_NAMES]) == {n: lim[n][0] for n in C.PARAM_NAMES}


def test_domain_check_refuses_off_grid_redshifts():
    cfg = CS.load_config("phase0_v1")
    lim = {n: (0.0, 10.0) for n in C.PARAM_NAMES}
    case = {"z": cfg["z_grid"] + [2.3], "state": {"params": dict(PARAMS), "mean_flux": dict(MF)}}
    assert CS.domain_support(case, cfg, lim) == {"status": "NOT COMPARABLE", "reason": "z not on the reference grid"}


# --------------------------------------------------------------------------------------------- reference assembly
class _StubGP:
    """Mimics lyaemu SingleBinAR1.predict: a mean that encodes (redshift slot, column 0, fidelity)."""
    def __init__(self, i, nk=4):
        self.i, self.nk, self.zbin = i, nk, None

    def predict(self, zp, res=1):
        return np.full((1, self.nk), 100.0 * self.i + zp[0][0] + 10.0 * res), np.full((1, self.nk), 0.5)


def test_reference_assembly_and_z_order_from_descending_zout():
    from hcd_analysis.comparison.adapters import lyaemu_priya as A
    zout = np.linspace(4.6, 2.2, 13)
    gp = type("G", (), {})()
    gp.gpemu = type("E", (), {"gps": [_StubGP(i) for i in range(13)]})()
    fac = ((1 + zout) / 4.0) ** 0.2
    mean, std = A.assemble(gp, np.array([1.1, 0.9, 2.0]), fac, res=0)
    assert np.allclose(mean[:, 0], 100.0 * np.arange(13) + 1.1 * fac, rtol=0, atol=1e-12)    # column 0 x factor
    assert np.allclose(A.assemble(gp, np.array([1.1]), fac, res=1)[0] - mean, 10.0)
    order = A.z_order(zout, [2.2, 3.0, 4.6])
    assert order == [12, 8, 0] and np.allclose(zout[order], [2.2, 3.0, 4.6])
    with pytest.raises(A.ReferenceRefused, match="not on the reference grid"):
        A.z_order(zout, [2.3])


# --------------------------------------------------------------------------------------------- declared cases, groups
def test_declared_case_list_binds_the_comparison():
    declared = {"c1": "COMPARABLE", "c2": "COMPARABLE"}
    out = S.compare_records(_rec([_case("c1")]), _rec([_case("c1")]), 0.02, declared=declared)
    assert out["cases"]["c2"] == "NOT RUN" and out["verdict"] == "NOT RUN"     # dropped by both adapters
    with pytest.raises(S.ComparisonSchemaError, match="not declared"):
        S.compare_records(_rec([_case("c1"), _case("c9")]), _rec([_case("c1")]), 0.02, declared=declared)
    with pytest.raises(S.ComparisonSchemaError, match="differs from the declared"):
        S.compare_records(_rec([_nc("c1")]), _rec([_nc("c1")]), 0.02, declared={"c1": "COMPARABLE"})
    both = S.compare_records(_rec([_case("c1"), _case("c2")]), _rec([_case("c1"), _case("c2")]), 0.02,
                             declared=declared)
    assert both["verdict"] == "PASS" and both["complete"]


def test_group_verdicts_use_the_level_precedence():
    outcomes = {"d1": "PASS", "d2": "NOT RUN", "i1": "PASS", "i2": "FAIL", "b1": "BLOCKED", "b2": "BLOCKED",
                "n1": "NOT COMPARABLE"}
    group_of = {"d1": "D", "d2": "D", "i1": "I", "i2": "I", "b1": "B", "b2": "B", "n1": "N", "x1": "X"}
    g = S.group_verdicts(outcomes, group_of)
    assert g["D"]["verdict"] == "NOT RUN" and not g["D"]["complete"]           # a missing case never passes
    assert g["I"]["verdict"] == "FAIL"
    assert g["B"] == {"verdict": "BLOCKED", "complete": False, "outcomes": {"BLOCKED": 2}}   # reported, not dropped
    assert g["N"]["verdict"] == "NOT COMPARABLE" and g["N"]["complete"]
    assert g["X"]["verdict"] == "NOT RUN"                                        # declared but absent from outcomes
    assert S.level_verdict([]) == "NOT COMPARABLE"
