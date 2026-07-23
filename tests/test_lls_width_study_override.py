"""Unit tests for the LLS-width hedge-form study's RUNTIME prior-override plumbing
(scripts/run_lls_width_study_shard.py; PI 2026-07-18 hedge-form sensitivity study).

Covers the three hazards the study brief names:
  1. the override actually LANDS (no silent no-op) -- verified on the EXACT function reference
     closure_legb.build_legb_ctx consumes (CL.hcd_incidence_prior, the from-import binding),
     which reads inference.HCD_LLS_SURVEY_FRAC_SIGMA at call time;
  2. the effective-prior signature is stamped (hcd_prior_signature changes under the override,
     returns to baseline on restore, and distinguishes the two hedge candidates);
  3. NO deployed-constant mutation leaks (the original dict OBJECT is unmutated, the HEDGE2X
     dict / KS entry / HCD_PRIOR_FRAC_SIGMA are untouched, restore is exact).

The full-ctx propagation (ctx.alpha_hcd_sigma/mu == width) is additionally asserted at runtime
inside the runner itself before any NUTS (fail-loud in-job; a full build_legb_ctx needs the
production ensemble + cache and is too heavy for a login-node unit test).

Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu \
     python -m pytest tests/test_lls_width_study_override.py -q
"""
import copy

import numpy as np
import pytest

import hcd_analysis.emulator  # noqa: F401  (x64 before jax)
import hcd_analysis.emulator.inference as INF
from hcd_analysis.emulator import closure_legb as CL

from scripts.run_lls_width_study_shard import (
    DESI_FAMILY_KEYS, override_lls_width, restore_lls_width, width_tag)

W_C_FID = (0.2004, 0.05, 0.02)      # a plausible z=3 (w_LLS, w_subDLA, w_DLA) fiducial
STUDY_WIDTHS = (0.416, 0.574)


def _lls_ratio(survey="DESI"):
    """sigma/mu of the LLS slot through the EXACT binding build_legb_ctx calls
    (closure_legb.hcd_incidence_prior, imported from inference at module load)."""
    mu, sd = CL.hcd_incidence_prior(np.asarray(W_C_FID), survey=survey)
    return float(np.asarray(sd)[0] / np.asarray(mu)[0])


@pytest.fixture()
def deployed_state():
    """Snapshot the deployed prior constants; restore + verify after each test."""
    orig_obj = INF.HCD_LLS_SURVEY_FRAC_SIGMA
    orig_val = copy.deepcopy(orig_obj)
    hedge_obj = INF.HCD_LLS_SURVEY_FRAC_SIGMA_HEDGE2X
    hedge_val = copy.deepcopy(hedge_obj)
    frac_val = tuple(INF.HCD_PRIOR_FRAC_SIGMA)
    yield dict(orig_obj=orig_obj, orig_val=orig_val, hedge_val=hedge_val, frac_val=frac_val)
    INF.HCD_LLS_SURVEY_FRAC_SIGMA = orig_obj
    assert dict(INF.HCD_LLS_SURVEY_FRAC_SIGMA) == orig_val
    assert dict(INF.HCD_LLS_SURVEY_FRAC_SIGMA_HEDGE2X) == hedge_val
    assert tuple(INF.HCD_PRIOR_FRAC_SIGMA) == frac_val


def test_override_lands_in_prior_function(deployed_state):
    """The override propagates through the closure_legb-bound hcd_incidence_prior (the ctx
    builder's call site) for every study width; restore returns the deployed 0.287."""
    base = deployed_state["orig_val"]["DESI"]
    assert abs(_lls_ratio() - base) < 1e-12          # deployed baseline before override
    for w in STUDY_WIDTHS:
        orig, _, _ = override_lls_width(w)
        try:
            assert abs(_lls_ratio() - w) < 1e-12, f"width {w} did not land (silent no-op)"
            # THE 53945595 FAILURE MODE: build_legb_ctx:718 reads the FROM-IMPORTED name in
            # closure_legb's namespace, not inference's global — the override must rebind
            # BOTH modules (from-import rebinding trap; caught on-node by the runner assert)
            assert abs(CL.HCD_LLS_SURVEY_FRAC_SIGMA.get("DESI") - w) < 1e-12, (
                f"width {w} not visible through closure_legb's from-import binding "
                f"(the exact read at closure_legb build_legb_ctx:718)")
            # subDLA/DLA slots untouched by the override (LLS-only knob):
            mu, sd = CL.hcd_incidence_prior(np.asarray(W_C_FID), survey="DESI")
            r = np.asarray(sd) / np.asarray(mu)
            assert abs(float(r[1]) - INF.HCD_PRIOR_FRAC_SIGMA[1]) < 1e-12
        finally:
            restore_lls_width(orig)
        assert abs(_lls_ratio() - base) < 1e-12      # exact restore


def test_ks_and_closure_paths_unaffected(deployed_state):
    """KS keeps its own selection-driven width; the survey=None closure/SBC path keeps
    HCD_PRIOR_FRAC_SIGMA -- the override may move ONLY the DESI-family real-fit width."""
    ks_before = _lls_ratio(survey="KS")
    closure_before = _lls_ratio(survey=None)
    orig, _, _ = override_lls_width(0.574)
    try:
        assert INF.HCD_LLS_SURVEY_FRAC_SIGMA["KS"] == deployed_state["orig_val"]["KS"]
        assert abs(_lls_ratio(survey="KS") - ks_before) < 1e-12
        assert abs(_lls_ratio(survey=None) - closure_before) < 1e-12
        for k in DESI_FAMILY_KEYS:
            assert INF.HCD_LLS_SURVEY_FRAC_SIGMA[k] == 0.574
    finally:
        restore_lls_width(orig)


def test_no_deployed_constant_mutation(deployed_state):
    """The override REBINDS the module attribute; the ORIGINAL dict object and the deployed
    HEDGE2X dict must never be mutated (the constant-leak tripwire)."""
    orig_obj = deployed_state["orig_obj"]
    orig, _, _ = override_lls_width(0.416)
    try:
        assert orig is orig_obj                                   # helper returned the original
        assert dict(orig_obj) == deployed_state["orig_val"]       # object contents untouched
        assert INF.HCD_LLS_SURVEY_FRAC_SIGMA is not orig_obj      # rebound, not edited in place
        assert dict(INF.HCD_LLS_SURVEY_FRAC_SIGMA_HEDGE2X) == deployed_state["hedge_val"]
        assert tuple(INF.HCD_PRIOR_FRAC_SIGMA) == deployed_state["frac_val"]
    finally:
        restore_lls_width(orig)
    assert INF.HCD_LLS_SURVEY_FRAC_SIGMA is orig_obj              # restore = the same object


def test_signature_stamps_and_distinguishes_arms(deployed_state):
    """hcd_prior_signature changes under each override (the pkl meta stamp), differs BETWEEN
    the two hedge candidates (no silent arm mixing), and returns to baseline on restore."""
    sig_base = INF.hcd_prior_signature()
    sigs = {}
    for w in STUDY_WIDTHS:
        orig, sig_before, sig_after = override_lls_width(w)
        try:
            assert sig_before == sig_base
            assert sig_after != sig_base, f"signature blind to the width override ({w})"
            assert sig_after == INF.hcd_prior_signature()
            sigs[w] = sig_after
        finally:
            restore_lls_width(orig)
        assert INF.hcd_prior_signature() == sig_base
    assert sigs[0.416] != sigs[0.574], "signature cannot distinguish the two hedge arms"


def test_width_tag():
    assert width_tag(0.416) == "0p416"
    assert width_tag(0.574) == "0p574"
    # matches the bash-side tag (echo $WIDTH | tr '.' 'p') used for skip-if-exists:
    assert width_tag(0.416) == "0.416".replace(".", "p")
