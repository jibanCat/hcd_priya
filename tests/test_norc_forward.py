"""Gate-A NORC: the res_corr_on forward toggle + KS k_max=0.045 cap (TDD).

Design : hcd_priya_notes/docs/superpowers/2026-07-04-norc-design-synthesis.md

NORC drops the mean-flux ``res_corr`` particle-convergence correction ENTIRELY and caps the KS
leg at k<=0.045. The toggle lives on the ``MultiFidelity`` object (``mf.res_corr_on``), the single
chokepoint BOTH the likelihood forward AND the self-draw truth reach (``mf.res_corr(z)``), so the
drop is self-consistent for the SBC. Threaded ``build_legb_ctx(res_corr_on=) -> build_mf_correction
-> build_multifidelity -> MultiFidelity``.

Contract this file pins:
  * res_corr_on=True (DEFAULT) is BYTE-IDENTICAL to the pre-NORC forward (the MF golden, rtol 1e-10).
  * res_corr_on=False => ``mf.res_corr(z)`` is all-ones => the forward loses exactly the ``+log_rc``
    factor: P_off/P_on == exp(-log_rc) per leg row (NONZERO on KS high-k, ~0 on DESI anchored low-k).
  * A NORC ctx (res_corr_on=False) caps KS at k<=0.045; an explicit ks_kwargs k_max overrides it.

Env (MANDATORY):
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
    /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_norc_forward.py -q
"""
import os
import numpy as np
import pytest

import hcd_analysis.emulator  # x64 BEFORE jax
import jax
import jax.numpy as jnp

from hcd_analysis.emulator import closure_legb as C
from hcd_analysis.emulator import data_likelihood as DL
from hcd_analysis.emulator import meanflux_prior as MF

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden")
MF_GOLDEN = os.path.join(GOLDEN_DIR, "legb_mf_golden.npz")
_CACHE = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/observables_tau0_lf.h5"
_CKPT0 = "/home/mfho/hcd_priya/checkpoints/final_fold0.eqx"
_DESI_NPZ = "/home/mfho/data/desi_dr1_p1d/desi_dr1_p1d.npz"
_KS = ("/home/mfho/lya_emulator_full/lyaemu/data/kodiaq_squad/"
       "final-conservative-p1d-karacayli_etal2021.txt")
_have = all(os.path.exists(p) for p in (_CACHE, _CKPT0, _DESI_NPZ))
_have_ks = os.path.exists(_KS)

pytestmark = pytest.mark.skipif(
    not (_have and _have_ks and os.path.exists(MF_GOLDEN)),
    reason="real cache/ckpt/DESI/KS or MF golden not present")


def _fwd(ctx, leg, theta9, alpha3, **kw):
    """predict_P_obs_on_leg on one leg through the MF forward -> P_model (np). Mirrors
    tests/test_res_corr_alpha._fwd (the production golden wiring)."""
    tau0_vec = MF.becker13_tau0(jnp.asarray(leg.z))
    szb = ctx.sigma_zb_per_leg.get(leg.name)
    rzb = ctx.rho_zb_per_leg.get(leg.name) if ctx.rho_zb_per_leg is not None else None
    core = jnp.mean(jnp.asarray(ctx.dla_core_leg[leg.name]), axis=0)
    P_model, _ = DL.predict_P_obs_on_leg(
        ctx.model, theta9, tau0_vec, alpha3, pf_stats=ctx.pf_stats,
        dla_core=core, cache_k=ctx.cache_k, leg=leg, sigma_zb=szb,
        alpha_centres=ctx.alpha_centres, cemu_inflate=ctx.cemu_inflate, rho_zb=rzb,
        mf=ctx.mf, **kw)
    return np.asarray(P_model)


def _expected_log_rc(ctx, leg):
    """log res_corr(z,k) per flat leg row from ctx.mf (the res_corr-ON anchored table)."""
    z_row = np.asarray(leg.z)[np.asarray(leg.z_idx)]
    k_row = np.asarray(leg.k)
    out = np.zeros(z_row.shape[0])
    for i in range(z_row.shape[0]):
        rc_cache = np.asarray(ctx.mf.res_corr(float(z_row[i])))
        rc_k = float(np.interp(k_row[i], np.asarray(ctx.cache_k), rc_cache))
        out[i] = np.log(rc_k)
    return out


# --------------------------------------------------------------------------- #
#  1. Default res_corr_on=True is BYTE-IDENTICAL (golden untouched by the field).
# --------------------------------------------------------------------------- #
def test_default_reproduces_mf_golden():
    """build_legb_ctx() default (res_corr_on=True) reproduces the Task-1.2 MF golden on
    DESI+KS to rtol 1e-10 -- i.e. adding the res_corr_on field is a no-op at the default."""
    g = np.load(MF_GOLDEN, allow_pickle=True)
    ctx, _ = C.build_legb_ctx(with_mf=True, mf_with_floor=True)
    assert ctx.mf is not None and ctx.mf.res_corr_on is True
    theta9 = jnp.asarray(g["theta9"]); alpha3 = jnp.asarray(g["alpha3"])
    for leg in ctx.legs:
        P = _fwd(ctx, leg, theta9, alpha3)
        np.testing.assert_allclose(P, g[f"{leg.name}_P"], rtol=1e-10, atol=0.0,
                                   err_msg=f"{leg.name}: default != MF golden (field perturbed it)")


# --------------------------------------------------------------------------- #
#  2. res_corr_on=False => mf.res_corr(z) is all ones (the correction is dropped).
# --------------------------------------------------------------------------- #
def test_res_corr_off_returns_ones():
    ctx_on, _ = C.build_legb_ctx(with_mf=True, mf_with_floor=True)
    ctx_off, _ = C.build_legb_ctx(with_mf=True, mf_with_floor=True, res_corr_on=False,
                                  ks_kwargs={"k_max": 0.069})  # keep KS grid for later reuse
    for z in (2.4, 3.0, 3.6, 4.2):
        rc_off = np.asarray(ctx_off.mf.res_corr(float(z)))
        np.testing.assert_array_equal(rc_off, np.ones_like(rc_off),
                                      err_msg=f"res_corr_on=False must return ones at z={z}")
        rc_on = np.asarray(ctx_on.mf.res_corr(float(z)))
        assert np.max(np.abs(rc_on - 1.0)) > 1e-3, f"res_corr_on table ~1 at z={z} (no signal)"


# --------------------------------------------------------------------------- #
#  3. Forward: P_off/P_on == exp(-log_rc); NONZERO on KS high-k, ~0 on DESI low-k.
# --------------------------------------------------------------------------- #
def test_norc_forward_drops_log_rc():
    """On the SAME KS grid (ks_kwargs k_max=0.069 override), turning res_corr off multiplies
    P_obs by exp(-log_rc) per row: genuinely different on KS high-k, negligible on DESI low-k."""
    g = np.load(MF_GOLDEN, allow_pickle=True)
    theta9 = jnp.asarray(g["theta9"]); alpha3 = jnp.asarray(g["alpha3"])
    ctx_on, _ = C.build_legb_ctx(with_mf=True, mf_with_floor=True)
    ctx_off, _ = C.build_legb_ctx(with_mf=True, mf_with_floor=True, res_corr_on=False,
                                  ks_kwargs={"k_max": 0.069})
    on_legs = {l.name: l for l in ctx_on.legs}
    off_legs = {l.name: l for l in ctx_off.legs}

    # KS: nonzero, tracks exp(-log_rc)
    ks_on, ks_off = on_legs["KS"], off_legs["KS"]
    assert np.asarray(ks_on.k).shape == np.asarray(ks_off.k).shape, "KS grid drifted (override failed)"
    P_on = _fwd(ctx_on, ks_on, theta9, alpha3)
    P_off = _fwd(ctx_off, ks_off, theta9, alpha3)
    log_rc_KS = _expected_log_rc(ctx_on, ks_on)
    assert np.max(np.abs(P_off / P_on - 1.0)) > 1e-4, "KS: NORC did not change P (res_corr not dropped)"
    np.testing.assert_allclose(P_off / P_on, np.exp(-log_rc_KS), rtol=1e-5, atol=2e-5,
                               err_msg="KS: P_off/P_on != exp(-log_rc)")

    # DESI: the SAME exp(-log_rc) mechanism holds; the shift is concentrated at DESI's high-k
    # bins (res_corr < 1 ABOVE the 5*kbox anchor, ~2-3%) and is ~0 at the anchored low-k bins.
    d_on, d_off = on_legs["DESI"], off_legs["DESI"]
    Pd_on = _fwd(ctx_on, d_on, theta9, alpha3)
    Pd_off = _fwd(ctx_off, d_off, theta9, alpha3)
    log_rc_DESI = _expected_log_rc(ctx_on, d_on)
    np.testing.assert_allclose(Pd_off / Pd_on, np.exp(-log_rc_DESI), rtol=1e-5, atol=2e-5,
                               err_msg="DESI: P_off/P_on != exp(-log_rc)")
    # the anchored low-k DESI bins (res_corr ~ 1 => log_rc ~ 0) are negligibly shifted (the
    # design's "near-no-op on DESI's constraining low-k" claim); high-k CAN move a few percent.
    lowk = np.abs(log_rc_DESI) < 1e-3
    assert lowk.any(), "expected some anchored (log_rc~0) DESI low-k bins"
    assert np.max(np.abs((Pd_off / Pd_on - 1.0)[lowk])) < 1e-3, "DESI anchored low-k moved"


# --------------------------------------------------------------------------- #
#  4. A NORC ctx caps KS at k<=0.045; explicit ks_kwargs k_max overrides.
# --------------------------------------------------------------------------- #
def test_norc_ks_kmax_cap():
    # loader-level: the cut exists and is strictly tighter than the 0.069 default
    ks045 = DL.load_ks_leg(k_max=0.045)
    ks069 = DL.load_ks_leg()  # default CACHE_KMAX=0.069
    assert np.asarray(ks045.k).max() <= 0.045 + 1e-9
    assert np.asarray(ks045.k).size < np.asarray(ks069.k).size

    # ctx-level: NORC applies 0.045 automatically; res_corr_on=True keeps 0.069
    ctx_on, _ = C.build_legb_ctx(with_mf=True, mf_with_floor=True)
    ctx_norc, _ = C.build_legb_ctx(with_mf=True, mf_with_floor=True, res_corr_on=False)
    ks_on = [l for l in ctx_on.legs if l.name == "KS"][0]
    ks_norc = [l for l in ctx_norc.legs if l.name == "KS"][0]
    assert np.asarray(ks_on.k).max() > 0.045, "res_corr_on=True should keep the 0.069 KS cap"
    assert np.asarray(ks_norc.k).max() <= 0.045 + 1e-9, "NORC did not cap KS at 0.045"

    # explicit override wins over the NORC default
    ctx_ov, _ = C.build_legb_ctx(with_mf=True, mf_with_floor=True, res_corr_on=False,
                                 ks_kwargs={"k_max": 0.069})
    ks_ov = [l for l in ctx_ov.legs if l.name == "KS"][0]
    assert np.asarray(ks_ov.k).max() > 0.045, "explicit ks_kwargs k_max should override the NORC cap"
