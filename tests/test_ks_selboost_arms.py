"""TDD suite for scripts/ks_selboost_arms.py — the KS selection-boost campaign arm registry
(spec 2026-07-18 + Amendment 2 A2.1-A2.2, PI A2.8 sign-offs 2026-07-19: "A-F all default",
second ask APPROVED => RESTORED-N battery: K3/K4 at N=16, K6 at N=12, K7-falling promoted to a
default arm at N=12; K5_meas null-bound arm at N=8 on the MEASURED conditional-incidence
vector).

Contract under test:
  * the A2.1 ENVELOPE MACHINE-CHECK: every envelope="in" arm satisfies 0.8 <= B(z) <= 3.2
    (relative units) on a dense grid over z in [2.4, 4.6]; K1 (mirror arm) and C2 (adversarial
    corner) are explicitly labeled envelope="out" and genuinely violate it;
  * the signed arm matrix (profiles + Ns) recorded verbatim; total default fits = 108;
  * K5_meas built from the measured conditional-incidence table (LLS + subDLA columns at
    L=120, pooled-suite rows), loglog-interp tabulated specs, provenance sha256-pinned;
  * contingencies C1/C2 registered but trigger-gated (default_run=False);
  * the registry profile evaluator agrees with the closure_legb hook evaluator (single-truth
    consistency certificate for the analyzer's login-node numpy path);
  * registry_signature is a stable sha256 covering profiles + Ns + envelope labels.

Env: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_ks_selboost_arms.py -q
"""
import numpy as np
import pytest

import scripts.ks_selboost_arms as AR

Z_DENSE = np.linspace(2.4, 4.6, 441)


# --------------------------------------------------------------------------------------------- #
#  The signed arm matrix (A2.2 + the A2.8-b restored Ns), recorded verbatim.
# --------------------------------------------------------------------------------------------- #
def test_signed_arm_matrix_verbatim():
    want = {
        "K0_clean":      (None,                 16, "in",  True),
        "K1_flat_lo":    ((0.714, 0.0, 0.0),    12, "out", True),
        "K2_flat_hi":    ((1.667, 0.0, 0.0),    16, "in",  True),
        "K3_rising":     ((1.25, 2.7, 0.0),     16, "in",  True),
        "K4_u_paper":    ((0.81, -0.35, 13.0),  16, "in",  True),
        "K6_inv_u":      ((2.67, 3.83, -22.0),  12, "in",  True),
        "K7_falling":    ((2.00, -2.7, 0.0),    12, "in",  True),
        "K5_joint_meas": ("measured",            8, "in",  True),
        "C1_falling":    ((2.00, -2.7, 0.0),    12, "in",  False),
        "C2_adversarial": ((1.30, 6.5, 0.0),     8, "out", False),
    }
    assert set(AR.ARMS) == set(want)
    for aid, (prof, n, env, default) in want.items():
        e = AR.ARMS[aid]
        assert e["n_mocks"] == n, aid
        assert e["envelope"] == env, aid
        assert e["default_run"] is default, aid
        if prof is None:
            assert e["quad"] is None
        elif prof == "measured":
            assert e["quad"] == "measured"
        else:
            assert e["quad"] == prof, aid


def test_default_campaign_totals_108_fits():
    total = sum(AR.ARMS[a]["n_mocks"] for a in AR.default_arm_ids())
    assert total == 108                       # 16 clean + 92 boosted (second-ask restored Ns)
    assert "C1_falling" not in AR.default_arm_ids()
    assert "C2_adversarial" not in AR.default_arm_ids()
    # nominal cost bookkeeping at the planning number (5 CPU-h/fit): 540 nominal, 850 worst-case
    assert AR.campaign_cost_cpuh()["nominal"] == pytest.approx(540.0)
    assert AR.campaign_cost_cpuh()["worst_case"] == pytest.approx(850.0)


# --------------------------------------------------------------------------------------------- #
#  A2.1 envelope machine-check: a TESTED property of the campaign, not prose.
# --------------------------------------------------------------------------------------------- #
def test_envelope_machine_check_all_in_arms():
    lo, hi = AR.ENVELOPE_REL
    assert (lo, hi) == (0.8, 3.2)
    for aid, e in AR.ARMS.items():
        if e["envelope"] != "in" or e["quad"] is None:
            continue
        for B in AR.arm_boost_B(aid, Z_DENSE).values():
            assert np.all(B >= lo - 1e-12) and np.all(B <= hi + 1e-12), \
                f"{aid}: B(z) leaves the envelope [{lo},{hi}]: [{B.min()}, {B.max()}]"


def test_envelope_out_arms_actually_violate():
    # K1 mirror arm sits below the floor; C2 adversarial corner exceeds the ceiling. Both are
    # LABELED out and thereby exempt from the in-envelope check (and from Part 1 gates).
    B1 = AR.arm_boost_B("K1_flat_lo", Z_DENSE)["lls_truth_boost"]
    assert B1.max() < 0.8
    B2 = AR.arm_boost_B("C2_adversarial", Z_DENSE)["lls_truth_boost"]
    assert B2.max() > 3.2


def test_envelope_extremal_placement():
    # the battery corners run floor-to-near-ceiling (A2.2 "min cosmic 2.00, max 7.89")
    mins, maxs = [], []
    for aid in ("K3_rising", "K4_u_paper", "K6_inv_u", "K7_falling"):
        B = AR.arm_boost_B(aid, Z_DENSE)["lls_truth_boost"]
        mins.append(B.min()); maxs.append(B.max())
    assert min(mins) == pytest.approx(0.8014, abs=2e-3)
    assert max(maxs) == pytest.approx(3.1543, abs=2e-3)


# --------------------------------------------------------------------------------------------- #
#  Inject-spec construction + hook-evaluator consistency.
# --------------------------------------------------------------------------------------------- #
def test_arm_inject_spec_shapes():
    assert AR.arm_inject_spec("K0_clean") is None
    s3 = AR.arm_inject_spec("K3_rising")
    assert set(s3) == {"lls_truth_boost"}
    assert s3["lls_truth_boost"] == {"b_pivot": 1.25, "eta1": 2.7, "eta2": 0.0}
    s5 = AR.arm_inject_spec("K5_joint_meas")
    assert set(s5) == {"lls_truth_boost", "subdla_truth_boost"}
    for cls in s5:
        assert s5[cls]["interp"] == "loglog"
        assert len(s5[cls]["z"]) == len(s5[cls]["boost"]) >= 2


def test_unknown_arm_raises():
    with pytest.raises(KeyError):
        AR.arm_inject_spec("K9_nope")


def test_registry_evaluator_matches_hook_evaluator():
    """The analyzer/registry numpy evaluator and the deployed hook evaluator must agree to
    float precision on every registered profile — the single-truth certificate that keeps the
    login-node analyzer honest against the cluster-side hook."""
    import hcd_analysis.emulator  # noqa: F401
    import hcd_analysis.emulator.closure_legb as LB
    z = np.linspace(2.4, 4.6, 45)
    for aid in AR.arm_ids():
        spec = AR.arm_inject_spec(aid)
        if spec is None:
            continue
        for cls, prof in spec.items():
            np.testing.assert_allclose(AR.eval_profile(prof, z),
                                       np.asarray(LB.eval_boost_profile(prof, z)),
                                       rtol=1e-13, err_msg=f"{aid}/{cls}")
            assert AR.eval_profile(prof, 3.0) == pytest.approx(
                float(LB.eval_boost_profile(prof, 3.0)), rel=1e-13)


def test_pivot_convention_matches_deployed_constant():
    from hcd_analysis.emulator.inference import HCD_Z_PIVOT
    assert AR.Z_PIVOT == float(HCD_Z_PIVOT) == 3.0


# --------------------------------------------------------------------------------------------- #
#  K5 measured-vector ingest: pooled-suite L=120 rows, provenance-hashed.
# --------------------------------------------------------------------------------------------- #
def test_k5_table_ingest_and_hash():
    t = AR.load_k5_measured_tables()
    assert t["sha256"] == AR.K5_TABLE_SHA256          # pinned provenance: a swapped table fails
    z = np.asarray(t["z"])
    assert np.all(np.diff(z) > 0)
    # covers the KS band with explicit rows (the hook forbids silent extrapolation) AND the
    # z_global union grid (DESI rows reach below 2.4): pooled rows run 2.0..5.4
    assert z[0] <= 2.2 and z[-1] >= 4.6
    # pooled-suite rows only: the single-sim rows 2.01/2.39/2.67/3.27 (nsnap=1, zero suite
    # scatter) are excluded per the table caveat
    for bad in (2.01, 2.39, 2.67, 3.27):
        assert not np.any(np.isclose(z, bad))
    # spot values from the committed table (L=120 columns)
    i24 = int(np.argmin(np.abs(z - 2.4)))
    i30 = int(np.argmin(np.abs(z - 3.0)))
    assert t["B_lls"][i24] == pytest.approx(1.116, abs=1e-9)
    assert t["B_subdla"][i24] == pytest.approx(1.285, abs=1e-9)
    assert t["B_lls"][i30] == pytest.approx(1.048, abs=1e-9)
    assert t["B_subdla"][i30] == pytest.approx(1.162, abs=1e-9)


def test_k5_spec_pivot_values():
    # loglog interp at z=3.0 lands exactly on the 3.00 table node
    s5 = AR.arm_inject_spec("K5_joint_meas")
    assert AR.eval_profile(s5["lls_truth_boost"], 3.0) == pytest.approx(1.048, abs=1e-9)
    assert AR.eval_profile(s5["subdla_truth_boost"], 3.0) == pytest.approx(1.162, abs=1e-9)


def test_k5_wrong_table_hash_fails_loud(tmp_path):
    p = tmp_path / "conditional_incidence_table.txt"
    p.write_text("tampered\n")
    with pytest.raises(AssertionError, match="sha256"):
        AR.load_k5_measured_tables(str(p))


# --------------------------------------------------------------------------------------------- #
#  Registry signature (stamped into every shard pkl; analyzer re-asserts homogeneity).
# --------------------------------------------------------------------------------------------- #
def test_registry_signature_stable_and_sensitive():
    s1 = AR.registry_signature()
    s2 = AR.registry_signature()
    assert isinstance(s1, str) and len(s1) == 64 and s1 == s2
    # covers the K5 table content: signature must differ under a different table hash
    payload = AR._registry_payload()
    assert AR.K5_TABLE_SHA256 in str(payload)
