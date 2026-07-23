"""TDD suite for the KS selection-boost z-PROFILE truth-boost hook extension (spec
2026-07-18-ks-selection-mock-challenge-spec.md Sec 4 + A2.7, PI-signed 2026-07-19).

API under test (closure_legb): the three inject_spec keys ("lls_truth_boost",
"subdla_truth_boost", "dla_truth_boost") keep their names; the VALUE becomes scalar-or-dict.

  * scalar        -> the existing code path, byte-identical (regression-tested here);
  * dict          -> a z-profile spec, exactly one of two key-sets:
                     {"b_pivot", "eta1", "eta2"}       log-quadratic (Amendment 1/2 canonical
                                                       family; b(z) = b_pivot * exp(eta1*x +
                                                       eta2*x^2), x = ln((1+z)/4), pivot z=3
                                                       HARD-CODED to inference.HCD_Z_PIVOT), or
                     {"z", "boost", "interp"}          tabulated, interp="loglog", NO silent
                                                       extrapolation (table must cover the grid).

PIVOT CONVENTION (convention B): truth_pack["alpha_hcd"][c] is multiplied by the profile
EVALUATED AT z=3.0 exactly (== b_pivot for ANY (eta1, eta2) since x(3)=0 -- explicit A2.7 test),
never the nearest z-grid row; truth_pack["alpha_hcd_z"][:, c] is multiplied row-wise by B(z_grid).
z is NEVER inferred from array length -- _apply_truth_boosts gains a z kwarg threaded from
run_legb (ctx.z_global); a profile without z, or against a pivot-only truth (alpha_hcd_z=None),
is a HARD ValueError, not a degraded scalar fallback.

Env: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_ks_selboost_hooks.py -q
"""
import math

import numpy as np
import pytest

import hcd_analysis.emulator  # noqa: F401  (x64)
import hcd_analysis.emulator.closure_legb as LB
from hcd_analysis.emulator.inference import HCD_Z_PIVOT

# z grid deliberately NOT containing 3.0: the pivot slot must use the profile evaluated at
# exactly z=3, never the nearest row (2.9 or 3.1) -- the convention-B invariant.
ZG = np.array([2.4, 2.9, 3.1, 3.7, 4.4])


def _toy_truth_pack(nZg=len(ZG)):
    rng = np.random.default_rng(0)
    return dict(
        theta9=rng.uniform(0.2, 0.8, 9),
        tau0_global=rng.uniform(0.9, 1.1, nZg),
        alpha_hcd=np.array([0.7, 0.3, 0.05]),                 # (3,) [LLS, subDLA, DLA] pivot
        alpha_hcd_z=np.stack([np.array([0.7, 0.3, 0.05]) * (1.0 + 0.1 * i) for i in range(nZg)]),
        a_siiii=0.02,
        kept_global_z=np.ones(nZg, bool))


def _B_quad(b0, e1, e2, z):
    x = np.log((1.0 + np.asarray(z, float)) / (1.0 + HCD_Z_PIVOT))
    return b0 * np.exp(e1 * x + e2 * x * x)


# --------------------------------------------------------------------------------------------- #
#  (1) SBC-neutrality certificate: spec None/absent is a no-op (identity object; the deployed
#      production-SBC path never passes inject_spec, so this is the neutrality anchor), and the
#      run_legb call site threads z=ctx.z_global (source-level guard: z is never inferred).
# --------------------------------------------------------------------------------------------- #
def test_none_and_empty_spec_are_identity():
    tp = _toy_truth_pack()
    assert LB._apply_truth_boosts(tp, None) is tp
    assert LB._apply_truth_boosts(tp, {}) is tp
    assert LB._apply_truth_boosts(tp, None, z=ZG) is tp


def test_run_legb_threads_z_global_into_truth_boosts():
    import inspect
    src = inspect.getsource(LB.run_legb)
    assert "_apply_truth_boosts(truth_pack, inject_spec, z=" in src, \
        "run_legb must thread ctx.z_global into _apply_truth_boosts (never infer z from length)"


def test_run_legb_records_truth_alpha_hcd_z():
    import inspect
    src = inspect.getsource(LB.run_legb)
    assert "truth_alpha_hcd_z" in src, \
        "run_legb per-mock record must carry truth_alpha_hcd_z (the analyzer row-level contract)"


# --------------------------------------------------------------------------------------------- #
#  (2) Scalar path bit-identical to the pre-extension implementation.
# --------------------------------------------------------------------------------------------- #
@pytest.mark.parametrize("boost", [1.0, 1.5])
def test_scalar_path_bit_identical(boost):
    tp = _toy_truth_pack()
    for fn, c in ((LB.apply_lls_truth_boost, 0), (LB.apply_subdla_truth_boost, 1),
                  (LB.apply_dla_truth_boost, 2)):
        out = fn(tp, boost)
        # the reference IS the old implementation: fresh array, a[c] *= float(boost)
        a_ref = np.array(tp["alpha_hcd"], float)
        a_ref[c] = a_ref[c] * float(boost)
        az_ref = np.array(tp["alpha_hcd_z"], float)
        az_ref[:, c] = az_ref[:, c] * float(boost)
        assert np.array_equal(out["alpha_hcd"], a_ref)
        assert np.array_equal(out["alpha_hcd_z"], az_ref)


def test_scalar_via_apply_truth_boosts_bit_identical():
    tp = _toy_truth_pack()
    out = LB._apply_truth_boosts(tp, {"lls_truth_boost": 1.5}, z=ZG)
    ref = LB.apply_lls_truth_boost(tp, 1.5)
    assert np.array_equal(out["alpha_hcd"], ref["alpha_hcd"])
    assert np.array_equal(out["alpha_hcd_z"], ref["alpha_hcd_z"])


# --------------------------------------------------------------------------------------------- #
#  (3) Flat dict == scalar exactly (A2.7: test 3 is {b_pivot: b, eta1: 0, eta2: 0} == scalar b).
# --------------------------------------------------------------------------------------------- #
def test_flat_profile_equals_scalar_exactly():
    tp = _toy_truth_pack()
    prof = {"b_pivot": 1.7, "eta1": 0.0, "eta2": 0.0}
    out_p = LB.apply_lls_truth_boost(tp, prof, z=ZG)
    out_s = LB.apply_lls_truth_boost(tp, 1.7)
    assert np.array_equal(out_p["alpha_hcd"], out_s["alpha_hcd"])
    assert np.array_equal(out_p["alpha_hcd_z"], out_s["alpha_hcd_z"])


# --------------------------------------------------------------------------------------------- #
#  (4) Pivot/row consistency + the x(3)=0 identity.
# --------------------------------------------------------------------------------------------- #
def test_eval_boost_profile_pivot_returns_b_pivot_for_any_etas():
    # x(3) = 0 => B(3) == b_pivot EXACTLY for ANY (eta1, eta2)  (A2.7 explicit unit test)
    for e1, e2 in ((0.0, 0.0), (2.7, 0.0), (-0.35, 13.0), (3.83, -22.0)):
        B = LB.eval_boost_profile({"b_pivot": 1.25, "eta1": e1, "eta2": e2}, HCD_Z_PIVOT)
        assert float(B) == 1.25


def test_pivot_and_rows_boosted_consistently():
    tp = _toy_truth_pack()
    prof = {"b_pivot": 1.3, "eta1": 0.8, "eta2": 0.4}
    out = LB.apply_lls_truth_boost(tp, prof, z=ZG)
    # pivot slot: profile at exactly z=3 (== b_pivot), NEVER the nearest grid row (2.9/3.1)
    np.testing.assert_allclose(out["alpha_hcd"][0], tp["alpha_hcd"][0] * 1.3, rtol=1e-15)
    # rows: B(z_g) per row, hand-computed
    Bz = _B_quad(1.3, 0.8, 0.4, ZG)
    np.testing.assert_allclose(out["alpha_hcd_z"][:, 0], tp["alpha_hcd_z"][:, 0] * Bz, rtol=1e-15)
    # one row fully by hand (z=2.4): x = ln(3.4/4)
    x = math.log(3.4 / 4.0)
    b_hand = 1.3 * math.exp(0.8 * x + 0.4 * x * x)
    np.testing.assert_allclose(out["alpha_hcd_z"][0, 0], tp["alpha_hcd_z"][0, 0] * b_hand,
                               rtol=1e-15)


def test_tabulated_loglog_interpolation_and_pivot():
    tp = _toy_truth_pack()
    zt = [2.0, 3.0, 5.0]
    bt = [2.0, 1.0, 4.0]
    prof = {"z": zt, "boost": bt, "interp": "loglog"}
    out = LB.apply_lls_truth_boost(tp, prof, z=ZG)
    # pivot: 3.0 is a node -> exactly 1.0
    np.testing.assert_allclose(out["alpha_hcd"][0], tp["alpha_hcd"][0], rtol=1e-15)
    # rows: log-linear in ln(1+z) of ln(B), hand-computed at z=2.4
    lz = np.log1p([2.0, 3.0, 5.0])
    lb = np.log([2.0, 1.0, 4.0])
    w = (math.log(3.4) - lz[0]) / (lz[1] - lz[0])
    b_hand = math.exp(lb[0] * (1 - w) + lb[1] * w)
    np.testing.assert_allclose(out["alpha_hcd_z"][0, 0], tp["alpha_hcd_z"][0, 0] * b_hand,
                               rtol=1e-12)


def test_tabulated_pivot_off_node():
    """Reviewer gap-closure: a table WITHOUT a z=3.0 node — the pivot slot must be the loglog
    interpolation AT exactly 3.0, never snapped to the nearest table node."""
    tp = _toy_truth_pack()
    prof = {"z": [2.0, 2.8, 3.4, 5.0], "boost": [2.0, 1.4, 1.1, 0.9], "interp": "loglog"}
    out = LB.apply_lls_truth_boost(tp, prof, z=ZG)
    lz = np.log1p([2.8, 3.4])
    lb = np.log([1.4, 1.1])
    wgt = (math.log(4.0) - lz[0]) / (lz[1] - lz[0])
    b3_hand = math.exp(lb[0] * (1 - wgt) + lb[1] * wgt)
    assert abs(b3_hand - 1.4) > 1e-3 and abs(b3_hand - 1.1) > 1e-3   # off both nodes
    np.testing.assert_allclose(out["alpha_hcd"][0], tp["alpha_hcd"][0] * b3_hand, rtol=1e-12)


# --------------------------------------------------------------------------------------------- #
#  (5) Other columns / fields untouched; input never mutated.
# --------------------------------------------------------------------------------------------- #
def test_profile_touches_only_its_class_and_never_mutates_input():
    tp = _toy_truth_pack()
    a0 = np.array(tp["alpha_hcd"])
    az0 = np.array(tp["alpha_hcd_z"])
    prof = {"b_pivot": 1.25, "eta1": 2.7, "eta2": 0.0}
    out = LB.apply_lls_truth_boost(tp, prof, z=ZG)
    # subDLA + DLA columns untouched
    np.testing.assert_array_equal(out["alpha_hcd"][1:], a0[1:])
    np.testing.assert_array_equal(out["alpha_hcd_z"][:, 1:], az0[:, 1:])
    # theta9 / tau0 / a_siiii untouched
    np.testing.assert_array_equal(out["theta9"], tp["theta9"])
    np.testing.assert_array_equal(out["tau0_global"], tp["tau0_global"])
    assert float(out["a_siiii"]) == float(tp["a_siiii"])
    # input unmutated, outputs are fresh arrays
    np.testing.assert_array_equal(tp["alpha_hcd"], a0)
    np.testing.assert_array_equal(tp["alpha_hcd_z"], az0)
    assert out["alpha_hcd"] is not tp["alpha_hcd"]
    assert out["alpha_hcd_z"] is not tp["alpha_hcd_z"]


# --------------------------------------------------------------------------------------------- #
#  (6) Fail-loud validation list (spec Sec 4, all ValueError).
# --------------------------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [
    {"b_pivot": 1.0, "eta": 2.7},                          # the superseded Sec-4 key-set (A2.7)
    {"b_pivot": 1.0, "eta1": 2.7},                         # missing eta2
    {"b_pivot": 1.0, "eta1": 0.0, "eta2": 0.0, "x": 1},    # extra key
    {"z": [2.0, 5.0], "boost": [1.0, 1.0]},                # missing interp
    {},                                                     # empty dict
])
def test_unknown_value_keyset_raises(bad):
    with pytest.raises(ValueError, match="profile"):
        LB.eval_boost_profile(bad, ZG)


@pytest.mark.parametrize("b0,e1,e2", [
    (0.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (float("nan"), 0.0, 0.0),
    (1.0, float("nan"), 0.0), (1.0, 0.0, float("inf")),
])
def test_nonpositive_or_nonfinite_params_raise(b0, e1, e2):
    with pytest.raises(ValueError):
        LB.eval_boost_profile({"b_pivot": b0, "eta1": e1, "eta2": e2}, ZG)


def test_tabulated_validation_raises():
    ok = {"z": [2.0, 3.0, 5.0], "boost": [1.0, 1.0, 1.0], "interp": "loglog"}
    with pytest.raises(ValueError):   # z not strictly increasing
        LB.eval_boost_profile({**ok, "z": [2.0, 2.0, 5.0]}, ZG)
    with pytest.raises(ValueError):   # decreasing
        LB.eval_boost_profile({**ok, "z": [5.0, 3.0, 2.0]}, ZG)
    with pytest.raises(ValueError):   # boost <= 0
        LB.eval_boost_profile({**ok, "boost": [1.0, -1.0, 1.0]}, ZG)
    with pytest.raises(ValueError):   # non-finite boost
        LB.eval_boost_profile({**ok, "boost": [1.0, float("nan"), 1.0]}, ZG)
    with pytest.raises(ValueError, match="cover"):   # non-covering table: NO silent extrapolation
        LB.eval_boost_profile({"z": [2.5, 3.0, 4.0], "boost": [1.0, 1.0, 1.0],
                               "interp": "loglog"}, ZG)
    with pytest.raises(ValueError):   # unknown interp
        LB.eval_boost_profile({**ok, "interp": "linear"}, ZG)


def test_evaluated_overflow_raises():
    # a huge eta2 overflows exp -> non-finite B must fail loud, never propagate
    with pytest.raises(ValueError):
        LB.eval_boost_profile({"b_pivot": 1.0, "eta1": 0.0, "eta2": 1e6}, ZG)


def test_profile_without_z_grid_raises():
    tp = _toy_truth_pack()
    prof = {"b_pivot": 1.5, "eta1": 0.0, "eta2": 0.0}
    with pytest.raises(ValueError, match="z"):
        LB.apply_lls_truth_boost(tp, prof)            # no z kwarg
    with pytest.raises(ValueError, match="z"):
        LB._apply_truth_boosts(tp, {"lls_truth_boost": prof})   # z not threaded


def test_profile_against_pivot_only_truth_raises():
    tp = _toy_truth_pack()
    tp["alpha_hcd_z"] = None
    prof = {"b_pivot": 1.5, "eta1": 0.0, "eta2": 0.0}
    with pytest.raises(ValueError, match="alpha_hcd_z"):
        LB.apply_lls_truth_boost(tp, prof, z=ZG)


def test_z_length_mismatch_raises():
    tp = _toy_truth_pack()                            # 5 rows
    prof = {"b_pivot": 1.5, "eta1": 0.0, "eta2": 0.0}
    with pytest.raises(ValueError, match="mismatch|length"):
        LB.apply_lls_truth_boost(tp, prof, z=np.array([2.4, 3.0, 4.4]))


# --------------------------------------------------------------------------------------------- #
#  (7) Top-level typo guard unchanged; joint compositions (incl. the K5_meas DICT+DICT form).
# --------------------------------------------------------------------------------------------- #
def test_top_level_unknown_key_still_raises():
    with pytest.raises(ValueError, match="unknown inject_spec key"):
        LB._apply_truth_boosts(_toy_truth_pack(), {"lls_truthboost": 1.5}, z=ZG)


def test_joint_dict_plus_scalar_composition():
    tp = _toy_truth_pack()
    prof = {"b_pivot": 1.3, "eta1": 0.8, "eta2": 0.4}
    out = LB._apply_truth_boosts(tp, {"lls_truth_boost": prof, "subdla_truth_boost": 1.5}, z=ZG)
    Bz = _B_quad(1.3, 0.8, 0.4, ZG)
    np.testing.assert_allclose(out["alpha_hcd"][0], tp["alpha_hcd"][0] * 1.3, rtol=1e-15)
    np.testing.assert_allclose(out["alpha_hcd_z"][:, 0], tp["alpha_hcd_z"][:, 0] * Bz, rtol=1e-15)
    np.testing.assert_allclose(out["alpha_hcd"][1], tp["alpha_hcd"][1] * 1.5, rtol=1e-15)
    np.testing.assert_allclose(out["alpha_hcd_z"][:, 1], tp["alpha_hcd_z"][:, 1] * 1.5, rtol=1e-15)
    np.testing.assert_array_equal(out["alpha_hcd_z"][:, 2], tp["alpha_hcd_z"][:, 2])


def test_joint_dict_plus_dict_composition_k5_form():
    """A2.7: the K5_meas form -- LLS table + subDLA table, applied independently per class at
    the row level."""
    tp = _toy_truth_pack()
    t_lls = {"z": [2.0, 3.0, 5.0], "boost": [1.12, 1.05, 1.03], "interp": "loglog"}
    t_sub = {"z": [2.0, 3.0, 5.0], "boost": [1.29, 1.16, 1.07], "interp": "loglog"}
    out = LB._apply_truth_boosts(tp, {"lls_truth_boost": t_lls, "subdla_truth_boost": t_sub},
                                 z=ZG)
    B_lls = np.asarray(LB.eval_boost_profile(t_lls, ZG))
    B_sub = np.asarray(LB.eval_boost_profile(t_sub, ZG))
    np.testing.assert_allclose(out["alpha_hcd_z"][:, 0], tp["alpha_hcd_z"][:, 0] * B_lls,
                               rtol=1e-15)
    np.testing.assert_allclose(out["alpha_hcd_z"][:, 1], tp["alpha_hcd_z"][:, 1] * B_sub,
                               rtol=1e-15)
    np.testing.assert_array_equal(out["alpha_hcd_z"][:, 2], tp["alpha_hcd_z"][:, 2])
    np.testing.assert_allclose(out["alpha_hcd"][0],
                               tp["alpha_hcd"][0] * LB.eval_boost_profile(t_lls, HCD_Z_PIVOT),
                               rtol=1e-15)


# --------------------------------------------------------------------------------------------- #
#  (8) B(z)=1 profile is a numerical identity.
# --------------------------------------------------------------------------------------------- #
def test_unit_profile_is_identity():
    tp = _toy_truth_pack()
    for prof in ({"b_pivot": 1.0, "eta1": 0.0, "eta2": 0.0},
                 {"z": [2.0, 5.0], "boost": [1.0, 1.0], "interp": "loglog"}):
        out = LB.apply_lls_truth_boost(tp, prof, z=ZG)
        np.testing.assert_array_equal(out["alpha_hcd"], tp["alpha_hcd"])
        np.testing.assert_array_equal(out["alpha_hcd_z"], tp["alpha_hcd_z"])
