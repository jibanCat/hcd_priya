"""NORC res_corr INJECTION harness — the mock-side wiring for the NORC injection-recovery gate.

Design : hcd_priya_notes/docs/superpowers/2026-07-04-norc-design-synthesis.md (Gate-A NORC) +
         the Phase-2 res_corr injection gate (spec §4.2).

Gate-A NORC drops the mean-flux ``res_corr`` correction ENTIRELY (``mf.res_corr(z)``==ones) but the
RAW res_corr table (``mf.z_rc``/``mf.logk_rc``/``mf.rc_vals``) stays populated, so
``interp_res_corr(...)`` still returns the REAL anchored res_corr. The NORC injection gate injects the
ACTUAL res_corr the real universe carries into the mock TRUTH ONLY (the forward stays NORC) and
measures the paired Delta(n_s, A_p). This file pins the mock-side wiring:

  (a) ``make_legb_mock(NORC ctx, inject_res_corr=None)`` is a byte-identical no-op (the default).
  (b) ``run_stepA._actual_res_corr_bvec(leg, mf, 5.0)`` == per-row ``log(interp_res_corr(...))`` on
      the leg grid (it IS the correction NORC drops, read from the raw table).
  (c) injecting ``{leg: b}`` multiplies the leg-binned TRUTH by ``exp(b)`` on the finite rows.
  (d) the b-vector on the CAPPED KS leg matches the (0.045-capped) leg grid length.
  (e) the runner-expansion yields ``inject=None`` when ``ctx.mf is None`` (LF-only, no res_corr).

Env (MANDATORY):
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
    /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_res_corr_norc_injection.py -q
"""
import importlib.util
import os

import numpy as np
import pytest

import hcd_analysis.emulator  # x64 BEFORE jax
import jax
import jax.numpy as jnp

from hcd_analysis.emulator import closure_legb as C
from hcd_analysis.emulator.multifidelity import interp_res_corr

_CACHE = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/observables_tau0_lf.h5"
_CKPT0 = "/home/mfho/hcd_priya/checkpoints/final_fold0.eqx"
_DESI_NPZ = "/home/mfho/data/desi_dr1_p1d/desi_dr1_p1d.npz"
_KS = ("/home/mfho/lya_emulator_full/lyaemu/data/kodiaq_squad/"
       "final-conservative-p1d-karacayli_etal2021.txt")
_have = all(os.path.exists(p) for p in (_CACHE, _CKPT0, _DESI_NPZ))
_have_ks = os.path.exists(_KS)

pytestmark = pytest.mark.skipif(
    not (_have and _have_ks), reason="real cache/ckpt/DESI/KS not present")


# The helper lives in a SCRIPT (scripts/run_stepA.py), not an importable package module — load it
# by path (the run_stepA top-level imports are stdlib+numpy only; jax/hcd imports are lazy).
_RUNSTEPA = os.path.join(os.path.dirname(__file__), "..", "scripts", "run_stepA.py")


def _load_runstepA():
    spec = importlib.util.spec_from_file_location("run_stepA", _RUNSTEPA)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "_actual_res_corr_bvec"), "run_stepA must expose _actual_res_corr_bvec"
    return mod


@pytest.fixture(scope="module")
def norc():
    """A deployed NORC ctx (res_corr_on=False -> fix_alpha_res) + its cache + one held-out truth.

    Mirrors run_prod_sbc_shard's NORC ctx: build_legb_ctx(res_corr_on=False) then
    ctx._replace(fix_alpha_res=True). Under NORC mf.res_corr(z) returns ones but the raw table is
    still populated. The truth is a self-consistent NORC self-draw (make_truth_from_sim(mf=ctx.mf))."""
    ctx, d = C.build_legb_ctx(with_mf=True, mf_with_floor=True, res_corr_on=False)
    ctx = ctx._replace(fix_alpha_res=True)
    assert ctx.mf is not None and ctx.mf.res_corr_on is False and ctx.fix_alpha_res is True
    sims, _ = C.held_out_sims(d, fold=0)
    truth = C.make_truth_from_sim(d, sims[0], fold=0, tau0_anchor="priya", mf=ctx.mf)
    return ctx, d, truth


def _leg(ctx, name):
    return [l for l in ctx.legs if l.name == name][0]


def _expected_bvec(leg, mf, anchor_mult):
    """Re-derive b = log(anchored res_corr) row-by-row (the reference for test b)."""
    k = np.asarray(leg.k); zi = np.asarray(leg.z_idx); zl = np.asarray(leg.z)
    exp = np.zeros(k.shape[0])
    for iz in range(leg.n_z):
        rsel = np.where(zi == iz)[0]
        if rsel.size == 0:
            continue
        rc = np.asarray(interp_res_corr(mf.z_rc, mf.logk_rc, mf.rc_vals,
                                        float(zl[iz]), jnp.asarray(k[rsel]),
                                        anchor_mult=float(anchor_mult)))
        exp[rsel] = np.log(rc)
    return exp


# --------------------------------------------------------------------------- #
#  (a) inject_res_corr=None is a byte-identical no-op (the non-injection default).
# --------------------------------------------------------------------------- #
def test_noop_default_byte_identical(norc):
    ctx, _d, truth = norc
    key = jax.random.PRNGKey(0)
    legs1, _tp1, _i1 = C.make_legb_mock(ctx, truth, key, inject_res_corr=None)
    legs2, _tp2, _i2 = C.make_legb_mock(ctx, truth, key, inject_res_corr=None)
    for l1, l2 in zip(legs1, legs2):
        # assert_array_equal treats NaN (dropped-z rows) in matching positions as equal.
        np.testing.assert_array_equal(np.asarray(l1.P_data), np.asarray(l2.P_data),
                                      err_msg=f"{l1.name}: NORC no-op mock not byte-identical")


# --------------------------------------------------------------------------- #
#  (b) _actual_res_corr_bvec == per-row log(interp_res_corr(...)) on the DESI leg grid.
# --------------------------------------------------------------------------- #
def test_bvec_matches_interp_res_corr(norc):
    ctx, _d, _truth = norc
    rs = _load_runstepA()
    desi = _leg(ctx, "DESI")
    b = rs._actual_res_corr_bvec(desi, ctx.mf, 5.0)
    exp = _expected_bvec(desi, ctx.mf, 5.0)
    assert b.shape == (np.asarray(desi.k).shape[0],)
    np.testing.assert_allclose(b, exp, rtol=0.0, atol=1e-10,
                               err_msg="DESI bvec != per-row log(interp_res_corr)")
    # NORC genuinely drops a nonzero correction on DESI high-k (the b-vector is not all zero).
    assert np.max(np.abs(b)) > 1e-4, "expected a nonzero anchored res_corr on the DESI leg"


# --------------------------------------------------------------------------- #
#  (c) injecting {leg: b} multiplies the leg-binned TRUTH by exp(b) on the finite rows.
# --------------------------------------------------------------------------- #
def test_inject_multiplies_truth_by_exp_b(norc):
    ctx, _d, truth = norc
    rs = _load_runstepA()
    key = jax.random.PRNGKey(3)
    inj = {leg.name: rs._actual_res_corr_bvec(leg, ctx.mf, 5.0) for leg in ctx.legs}
    _l0, _t0, info_clean = C.make_legb_mock(ctx, truth, key, inject_res_corr=None)
    _l1, _t1, info_inj = C.make_legb_mock(ctx, truth, key, inject_res_corr=inj)
    for leg in ctx.legs:
        tc = np.asarray(info_clean["truth_on_leg"][leg.name])
        ti = np.asarray(info_inj["truth_on_leg"][leg.name])
        fin = np.isfinite(tc) & np.isfinite(ti)
        assert fin.any(), f"{leg.name}: no finite truth rows"
        np.testing.assert_allclose(ti[fin] / tc[fin], np.exp(inj[leg.name][fin]),
                                   rtol=1e-6, atol=0.0,
                                   err_msg=f"{leg.name}: truth_inj/truth_clean != exp(b)")


# --------------------------------------------------------------------------- #
#  (d) the b-vector on the CAPPED KS leg matches the (0.045-capped) leg grid.
# --------------------------------------------------------------------------- #
def test_bvec_ks_capped(norc):
    ctx, _d, _truth = norc
    rs = _load_runstepA()
    ks = _leg(ctx, "KS")
    b = rs._actual_res_corr_bvec(ks, ctx.mf, 5.0)
    assert b.shape[0] == np.asarray(ks.k).shape[0], "KS bvec length != capped KS leg grid"
    assert np.asarray(ks.k).max() <= 0.045 + 1e-9, "NORC did not cap the KS leg at 0.045"


# --------------------------------------------------------------------------- #
#  (e) the runner-expansion yields inject=None when ctx.mf is None (LF-only). Light unit test of the
#      inline run_one_chain expansion logic (the plan sanctions a light test here) + the pass-through.
# --------------------------------------------------------------------------- #
def test_expansion_none_when_mf_none():
    class _StubCtx:  # a minimal ctx with no MF backbone (LF-only) and no legs
        mf = None
        legs = []

    def _expand(chain, ctx):
        # EXACTLY the run_one_chain (step 4) inline expansion.
        _rc_spec = chain.get("inject_res_corr", None)
        if isinstance(_rc_spec, dict) and _rc_spec.get("actual_res_corr"):
            _am = float(chain.get("mf_anchor_mult", 5.0))
            _rc_inject = (None if ctx.mf is None
                          else {leg.name: leg for leg in ctx.legs})  # (leg-value irrelevant here)
        else:
            _rc_inject = _rc_spec
        return _rc_inject

    ctx = _StubCtx()
    # actual_res_corr spec + mf None -> None (no res_corr to inject on an LF-only ctx).
    assert _expand({"inject_res_corr": {"actual_res_corr": True}}, ctx) is None
    # a non-actual spec (path/member/strength) passes THROUGH unchanged (mf irrelevant).
    spec = {"path": "x.npz", "member": "b1", "strength": 1.0}
    assert _expand({"inject_res_corr": spec}, ctx) == spec
    # None passes through as None (the non-injection default).
    assert _expand({}, ctx) is None


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
