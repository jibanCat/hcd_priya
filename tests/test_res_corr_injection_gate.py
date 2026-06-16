"""res_corr injection-recovery GATE — the PAIRED analysis estimator (TDD, Task 2.1-SETUP).

These tests pin the *analysis contract* for the decisive Phase-2 gate (spec §4.2): after the
NUTS injection arm runs (a CLEAN control + an INJECTED arm at the SAME (sim, fold, seed) so the
mocks share byte-identical base truth + cosmic noise, differing ONLY by the out-of-span exp(b1)
res_corr misspecification on the z>=2.8 truth), this estimator forms the per-mock PAIRED shift and
gates it. PASS => the marginalized alpha_res nuisance absorbs the worst out-of-span res_corr
misspecification WITHOUT leaking into cosmology (A_p, n_s).

The function under test is the pure core of scripts/analyze_res_corr_injection.py:

    paired_injection_gate(clean_means, inj_means, sigma_ref)
      -> {delta_mean, delta_se, stat, sigma_ref, passed}

where, for paired mock i (shared seed/noise),

    Delta_i  = inj_mean_i - clean_mean_i              # paired -> shared noise cancels
    stat     = |mean_i Delta_i| + 2 * SE(Delta_i)     # the confidence-bound gate statistic
    passed   = stat < 0.3 * sigma_ref

THE TWO LOAD-BEARING DESIGN CHOICES these tests pin (mirrors of analyze_dnuis_bias.py's structure,
but in FIXED-reference units instead of the per-record bias_z):

  1. The estimator works on RAW POSTERIOR MEANS (post_mean) and an EXTERNALLY-SUPPLIED, FIXED
     sigma_ref -- the anchored / alpha-FIXED n_s posterior sigma (or the eBOSS-anchor sigma). It
     does NOT divide each pair by that pair's own (alpha-inflated) post_sd the way bias_z does.
     Reason (spec §4.2, "the masking trap"): with alpha_res free, the per-fit post_sd inflates by
     ~1.3-1.4x, so a bias_z = Delta(mean-truth)/post_sd statistic would mechanically deflate toward
     PASS WITHOUT de-biasing. Test (c) pins that the threshold is a fixed external yardstick:
     scaling sigma_ref does not touch Delta, only the bar.

  2. The SE is the WITHIN-PAIR Delta SD / sqrt(N) (paired), NOT the unpaired between-arm SD. The
     shared cosmic noise cancels in Delta_i, collapsing the unpaired SE (~0.2 sigma) to the
     contaminant-only SE, which is what makes N>=8 paired mocks enough to certify <0.3 sigma_ref.
     Test (d) pins that the paired SE is much smaller than the naive unpaired SE on data where the
     two arms share a large common noise component.

Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_res_corr_injection_gate.py -q
"""
import importlib.util
import os

import numpy as np
import pytest


# The estimator lives in a SCRIPT (scripts/analyze_res_corr_injection.py), not an importable
# package module -- load it by path so the test does not depend on scripts/ being on sys.path.
# Until the CS partner writes the script this import fails with FileNotFoundError -> the tests
# fail for the RIGHT reason (function not yet implemented), as required by Task 2.1-SETUP step 5.
_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "analyze_res_corr_injection.py")


def _load_gate_fn():
    spec = importlib.util.spec_from_file_location("analyze_res_corr_injection", _SCRIPT)
    if spec is None or not os.path.exists(_SCRIPT):
        raise FileNotFoundError(
            f"{_SCRIPT} not yet implemented (CS partner Task 2.1). "
            f"Expected a pure paired_injection_gate(clean_means, inj_means, sigma_ref).")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "paired_injection_gate"):
        raise AttributeError(
            "scripts/analyze_res_corr_injection.py must expose paired_injection_gate("
            "clean_means, inj_means, sigma_ref) -> dict(delta_mean, delta_se, stat, "
            "sigma_ref, passed).")
    return mod.paired_injection_gate


# Gate threshold from spec §4.2: stat < 0.3 * sigma_ref (the same 0.30 sigma budget as the
# dnuis bias gate, but here measured against an EXTERNAL fixed sigma_ref).
GATE_FRAC = 0.30


# --------------------------------------------------------------------------------------------- #
#  Synthetic paired-mock generator: a shared cosmic-noise component (cancels in Delta) + a small
#  per-arm independent jitter + an optional systematic injection offset. This mimics the real
#  CLEAN/INJECTED pair: same (seed,noise) base mock, the injected arm differs by the contaminant.
# --------------------------------------------------------------------------------------------- #
def _paired_means(n, *, offset=0.0, shared_sd=0.20, indep_sd=0.01, seed=0):
    """Return (clean_means, inj_means) for n paired mocks.

    shared_sd : the common cosmic-noise spread shared by both arms of a pair (CANCELS in Delta).
    indep_sd  : a tiny arm-specific jitter (the residual the paired SE actually measures).
    offset    : a systematic shift applied to the injected arm only (the leaked bias to detect).
    """
    rng = np.random.default_rng(seed)
    shared = rng.normal(0.0, shared_sd, n)             # per-mock truth+noise realization
    clean = shared + rng.normal(0.0, indep_sd, n)
    inj = shared + offset + rng.normal(0.0, indep_sd, n)
    return clean, inj


# --------------------------------------------------------------------------------------------- #
#  (a) CLEAN case: inj == clean exactly -> Delta==0 -> stat==0 -> PASS.
# --------------------------------------------------------------------------------------------- #
def test_clean_pairs_stat_zero_and_pass():
    gate = _load_gate_fn()
    rng = np.random.default_rng(1)
    means = rng.normal(0.0, 0.2, 12)                   # arbitrary post means
    sigma_ref = 0.01                                   # a tight fixed yardstick
    out = gate(means.copy(), means.copy(), sigma_ref)  # inj IS clean -> every Delta == 0
    assert out["delta_mean"] == pytest.approx(0.0, abs=1e-12)
    assert out["delta_se"] == pytest.approx(0.0, abs=1e-12)
    assert out["stat"] == pytest.approx(0.0, abs=1e-12)
    assert out["sigma_ref"] == pytest.approx(sigma_ref)
    assert out["passed"] is True                       # 0 < 0.3*sigma_ref for any sigma_ref>0


# --------------------------------------------------------------------------------------------- #
#  (b) BIASED case: a systematic offset of 0.5*sigma_ref on the injected arm -> |mean Delta| alone
#      already exceeds 0.3*sigma_ref -> FAIL. (The leak alpha_res failed to absorb.)
# --------------------------------------------------------------------------------------------- #
def test_biased_pairs_exceed_threshold_and_fail():
    gate = _load_gate_fn()
    sigma_ref = 0.01
    offset = 0.5 * sigma_ref                            # half a sigma_ref leak
    clean, inj = _paired_means(16, offset=offset, shared_sd=0.20, indep_sd=1e-4, seed=2)
    out = gate(clean, inj, sigma_ref)
    # mean Delta ~ +0.5*sigma_ref; |mean|+2SE must be > 0.3*sigma_ref -> FAIL
    assert out["delta_mean"] == pytest.approx(offset, abs=3e-4)
    assert out["stat"] > GATE_FRAC * sigma_ref
    assert out["passed"] is False


# --------------------------------------------------------------------------------------------- #
#  (c) FIXED-REFERENCE units (the alpha-inflation masking trap): scaling sigma_ref up scales ONLY
#      the threshold, NOT Delta/stat. So a bias that FAILS at the true (alpha-fixed) sigma_ref must
#      not be turned into a PASS by feeding a larger (alpha-inflated) sigma. The statistic is
#      computed from raw post means; sigma_ref is purely the external bar.
# --------------------------------------------------------------------------------------------- #
def test_gates_in_fixed_reference_units_not_inflated_sd():
    gate = _load_gate_fn()
    sigma_ref = 0.01
    offset = 0.5 * sigma_ref
    clean, inj = _paired_means(16, offset=offset, shared_sd=0.20, indep_sd=1e-4, seed=3)

    out_true = gate(clean, inj, sigma_ref)             # the honest, alpha-FIXED yardstick
    out_infl = gate(clean, inj, 1.4 * sigma_ref)       # an alpha-INFLATED yardstick (the trap)

    # Delta and stat depend ONLY on the means -> identical regardless of sigma_ref.
    assert out_infl["delta_mean"] == pytest.approx(out_true["delta_mean"], abs=0, rel=0)
    assert out_infl["stat"] == pytest.approx(out_true["stat"], abs=0, rel=0)
    # The threshold scales with sigma_ref; the reported sigma_ref is exactly what was passed.
    assert out_true["sigma_ref"] == pytest.approx(sigma_ref)
    assert out_infl["sigma_ref"] == pytest.approx(1.4 * sigma_ref)
    # The honest gate FAILS this 0.5-sigma_ref leak.
    assert out_true["passed"] is False
    # And inflating sigma_ref by 1.4x is NOT enough to flip 0.5*sigma_ref past 0.3*sigma_ref:
    #   stat ~ 0.5*sigma_ref ;  0.3 * 1.4 * sigma_ref = 0.42 * sigma_ref < 0.5*sigma_ref -> still FAIL.
    assert out_infl["passed"] is False


# --------------------------------------------------------------------------------------------- #
#  (d) PAIRING actually cancels the shared noise: the within-pair SE must be MUCH smaller than the
#      naive unpaired SE of either arm when the two arms share a large common noise component. This
#      is the property that makes N>=8 paired mocks sufficient to certify <0.3 sigma_ref.
# --------------------------------------------------------------------------------------------- #
def test_paired_se_much_smaller_than_unpaired():
    gate = _load_gate_fn()
    n = 16
    shared_sd, indep_sd = 0.30, 0.01                   # big shared noise, tiny arm jitter
    clean, inj = _paired_means(n, offset=0.0, shared_sd=shared_sd, indep_sd=indep_sd, seed=4)
    out = gate(clean, inj, sigma_ref=0.01)

    # Naive UNPAIRED SE (treating the two arms as independent samples): dominated by shared_sd.
    se_unpaired = np.sqrt(clean.std(ddof=1) ** 2 / n + inj.std(ddof=1) ** 2 / n)
    # The paired SE the gate reports is the within-pair Delta SD / sqrt(n) -> ~indep-only.
    se_paired = out["delta_se"]

    # Independent check that the gate's paired SE matches the within-pair Delta SD / sqrt(n).
    delta = np.asarray(inj) - np.asarray(clean)
    se_paired_expected = delta.std(ddof=1) / np.sqrt(n)
    assert se_paired == pytest.approx(se_paired_expected, rel=1e-9)

    # The whole point: pairing collapses the shared noise -> paired SE << unpaired SE.
    assert se_paired < 0.2 * se_unpaired
    # And with no injected offset the no-leak pair PASSES (stat well under 0.3*sigma_ref... here
    # sigma_ref is tiny, but delta_mean ~ 0 and se_paired ~ indep_sd/sqrt(n) -> stat tiny).
    assert out["delta_mean"] == pytest.approx(0.0, abs=4 * se_paired)


# --------------------------------------------------------------------------------------------- #
#  (e) Monotone-threshold sanity: an offset of exactly 0.3*sigma_ref sits AT the boundary so
#      |mean|+2SE (>= |mean|) FAILS; a tiny offset (<< 0.3 sigma_ref, with negligible SE) PASSES.
#      Pins the comparison sense (strict <) and that A_p is gated by the SAME function (param-agnostic).
# --------------------------------------------------------------------------------------------- #
def test_threshold_sense_strict_less_than():
    gate = _load_gate_fn()
    sigma_ref = 0.02
    # well-under-threshold, negligible SE -> PASS
    clean, inj = _paired_means(20, offset=0.02 * sigma_ref, shared_sd=0.1, indep_sd=1e-5, seed=5)
    assert gate(clean, inj, sigma_ref)["passed"] is True
    # exactly at 0.3*sigma_ref mean offset -> |mean|+2SE >= 0.3*sigma_ref -> FAIL (strict <)
    clean2, inj2 = _paired_means(20, offset=0.30 * sigma_ref, shared_sd=0.1, indep_sd=1e-5, seed=6)
    assert gate(clean2, inj2, sigma_ref)["passed"] is False


if __name__ == "__main__":
    # Allow a quick `python tests/test_res_corr_injection_gate.py` smoke run.
    for fn in (test_clean_pairs_stat_zero_and_pass, test_biased_pairs_exceed_threshold_and_fail,
               test_gates_in_fixed_reference_units_not_inflated_sd,
               test_paired_se_much_smaller_than_unpaired, test_threshold_sense_strict_less_than):
        fn()
    print("[res_corr-injection-gate] all tests OK")
