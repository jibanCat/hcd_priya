"""Option-b spectral-resolution nuisance f_res: a forward-only 2-param b_res(z) = f_res_amp *
((1+z)/(1+z_p))^f_res_slope threaded through _resolution_factor = exp(2 b_res k^2 R_z^2). Cloned from
the alpha_res forward-only pattern. Default OFF (ctx.sample_res=False) -> byte-identical golden.

4-lens-verified design (2026-07-02): keep EXP; TIGHT prior on physics grounds (amp ~ Normal(0,0.02));
per-z-block rank-1 cov surgery for option-b; forward-only (not in truth -> no closure cancellation).

Env: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_legb_resolution.py -q
"""
import os
import numpy as np
import pytest
import hcd_analysis.emulator  # x64 before jax
import jax
from numpyro import handlers
from numpyro.infer.util import constrain_fn
from hcd_analysis.emulator import closure_legb as C

_CACHE = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/observables_tau0_lf.h5"
_CKPT0 = "/home/mfho/hcd_priya/checkpoints/final_fold0.eqx"
_DESI_NPZ = "/home/mfho/data/desi_dr1_p1d/desi_dr1_p1d.npz"
_have = all(os.path.exists(p) for p in (_CACHE, _CKPT0, _DESI_NPZ))


def _priors_only_sites(ctx):
    tr = handlers.trace(handlers.seed(lambda: C._legb_priors_only(ctx), jax.random.PRNGKey(0))).get_trace()
    return [n for n, s in tr.items() if s["type"] == "sample"]


def test_bres_of_z_formula():
    """b_res(z) = f_res_amp * ((1+z)/(1+z_pivot))^f_res_slope (mirrors tau0_alpha_priya). amp=0 is the
    no-op (b_res=0 for ANY slope) -> _resolution_factor = 1 -> golden identity."""
    from hcd_analysis.emulator.closure_legb import _bres_of_z, F_RES_PIVOT_Z
    assert float(F_RES_PIVOT_Z) == 3.0
    z = np.array([2.2, 3.0, 4.2])
    amp, slope = 0.02, 0.5
    expect = amp * ((1.0 + z) / (1.0 + 3.0)) ** slope
    np.testing.assert_allclose(np.asarray(_bres_of_z(z, amp, slope)), expect, rtol=1e-12)
    # at the pivot z=3, b_res == amp regardless of slope
    assert np.isclose(float(_bres_of_z(3.0, amp, 1.7)), amp, rtol=1e-12)
    # amp=0 -> 0 for any slope (the golden no-op)
    assert np.allclose(np.asarray(_bres_of_z(z, 0.0, 3.0)), 0.0)


def test_fres_prior_constants_tight():
    """The f_res amplitude prior is TIGHT (physics: DESI deconvolved to ~few%), NOT cup1d's wide
    [-0.5,0.5] code default. 4-lens verdict C1."""
    from hcd_analysis.emulator.closure_legb import F_RES_AMP_SIGMA, F_RES_SLOPE_SIGMA
    assert 0.01 <= float(F_RES_AMP_SIGMA) <= 0.05, "f_res amp prior must be tight (~0.02), not wide"
    assert 0.0 < float(F_RES_SLOPE_SIGMA) <= 1.0


@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI not present")
def test_fres_sites_present_ordered_and_golden_off():
    """sample_res=True adds f_res_amp/f_res_slope to _legb_priors_only IN ORDER AFTER alpha_res (the
    constrain_fn mirror invariant _legb_model must match); sample_res=False (golden) adds NO f_res sites."""
    ctx, _d = C.build_legb_ctx(use_xclass=True, desi_kwargs=dict(z_lo=0.0, z_hi=2.6))
    on = _priors_only_sites(ctx._replace(sample_res=True))
    off = _priors_only_sites(ctx._replace(sample_res=False))
    assert "f_res_amp" in on and "f_res_slope" in on, "sample_res=True must sample both f_res sites"
    assert on.index("f_res_slope") == on.index("f_res_amp") + 1, "amp THEN slope (fixed order)"
    if "alpha_res" in on:
        assert on.index("f_res_amp") > on.index("alpha_res"), "f_res must come AFTER alpha_res"
    # GOLDEN-safe: default off -> the site set is unchanged (no f_res_*)
    assert "f_res_amp" not in off and "f_res_slope" not in off
    assert set(off) == set(on) - {"f_res_amp", "f_res_slope"}, "f_res is the ONLY site-set change"


@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI not present")
def test_cov_b_resolution_float_spd_and_golden():
    """option-b: load_desi_leg(resolution_float=True) removes the per-z-block rank-1 resolution mode
    (cov_b = cov - sum_z outer(e_res|z)) -> SPD + diagonal reduced by the resolution variance, and flips
    resolution_on True. Default (resolution_float=False) is byte-identical to option-a (golden)."""
    from hcd_analysis.emulator.data_likelihood import load_desi_leg
    leg_a = load_desi_leg(z_lo=0.0, z_hi=2.6)                          # option-a: resolution IN cov
    leg_b = load_desi_leg(z_lo=0.0, z_hi=2.6, resolution_float=True)   # option-b: cov_b + float f_res
    leg_off = load_desi_leg(z_lo=0.0, z_hi=2.6, resolution_float=False)
    assert leg_a.resolution_on is False and leg_b.resolution_on is True
    Ca, Cb = np.asarray(leg_a.C_data, float), np.asarray(leg_b.C_data, float)
    np.linalg.cholesky(Cb)                                            # cov_b is SPD (the correct removal)
    dA, dB = np.diag(Ca), np.diag(Cb)
    assert np.all(dB <= dA + 1e-9) and np.any(dB < dA - 1e-12), "resolution variance removed from diagonal"
    # golden: resolution_float=False is byte-identical to option-a
    np.testing.assert_array_equal(np.asarray(leg_off.C_data, float), Ca)
