"""res_corr AMPLITUDE nuisance α(z) — forward-only marginalization (Task 1.3, TDD).

Plan : docs/superpowers/plans/2026-06-16-res-corr-anchor-marginalize-plan.md (Task 1.3)
Spec : docs/superpowers/specs/2026-06-16-res-corr-anchor-marginalize-design.md (§3.2/§5)

The multi-fidelity ``res_corr`` correction (``mf.res_corr``, the HF→n512 particle-convergence
factor) is currently a FIXED, zero-uncertainty deterministic factor whose high-k amplitude is
~0.65–0.70 aligned with the n_s response — so any res_corr error leaks ~directly into n_s. This
task makes its AMPLITUDE a sampled, marginalized nuisance applied FORWARD-ONLY:

    log res_corr  →  α(z) · log res_corr ,   α(z) = α₀ · ((1+z)/(1+z_p))^s ,   z_p = 3.0

at the production chokepoint ``data_likelihood._mf_corr_on_cache`` (currently returns
``g + log_rc`` with ``log_rc = jnp.log(mf.res_corr(z_phys))[None,:]``; becomes
``g + alpha_z[None,:] * log_rc``). α is sampled in ``closure_legb._legb_model`` (and its
prior-only twin ``_legb_priors_only`` in the SAME site order) and threaded forward-only:
``_legb_model → _data_loglik_legcore → predict_P_obs_on_leg → _predict_P_obs_mf →
_mf_corr_on_cache``. The MATCHED-CLOSURE TRUTH path ``make_truth_from_sim`` keeps α≡1 (else the
nuisance cancels in the closure and is unconstrained).

============================== CONTRACT (for the CS implementer) ==============================
* ``predict_P_obs_on_leg(..., alpha_res=None)``  — NEW keyword-only argument.
    - ``alpha_res`` is a ``(alpha0, s)`` tuple (the res_corr amplitude + z-slope).
    - ``alpha_res=None`` (the DEFAULT) ⇒ α≡1 ⇒ the forward is BIT-IDENTICAL to the current
      code (the Task-1.2 MF golden, rtol 1e-10, must still pass; all other callers unaffected).
    - ``alpha_res=(1.0, 0.0)`` must be bit-identical to ``alpha_res=None``.
* Threaded FORWARD-ONLY, signatures gain ``alpha_res`` (default None):
      predict_P_obs_on_leg(..., alpha_res=None)
        → _predict_P_obs_mf(..., alpha_res=None)
          → _mf_corr_on_cache(mf, theta9, z_unit, tau0, alpha_res=None)
* At the chokepoint ``_mf_corr_on_cache`` (data_likelihood.py:~373):
      z_phys  = z_unit*(Z_LIMITS[1]-Z_LIMITS[0]) + Z_LIMITS[0]
      log_rc  = jnp.log(mf.res_corr(z_phys))[None, :]            # (1, Kc)
      alpha_z = alpha0 * ((1+z_phys)/(1+Z_PIVOT))**s             # scalar (per z), 1.0 if alpha_res is None
      return  g + alpha_z * log_rc                               # (4, Kc); == g+log_rc when alpha_z==1
  with module constant ``Z_PIVOT = 3.0``.
* ``make_truth_from_sim`` (closure_legb.py:~509) keeps α≡1 (do NOT thread the sampled α in).
* ``_legb_model`` / ``_legb_priors_only`` gain TWO sample sites in IDENTICAL order:
      alpha_res        ~ TruncatedNormal(loc=1.0, scale=SIGMA_A0=1.0, low=0.0)   (P_MF>0)
      alpha_res_slope  ~ Normal(0.0, SIGMA_S=0.5)
  placed AFTER the existing nuisances (relative position must match between the two models). The
  model threads ``alpha_res=(alpha_res, alpha_res_slope)`` into ``_data_loglik_legcore`` →
  ``predict_P_obs_on_leg`` so the forward picks them up.
=============================================================================================

Env (MANDATORY):
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
    /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_res_corr_alpha.py -q
"""
import os
import numpy as np
import pytest

import hcd_analysis.emulator  # x64 BEFORE jax
import jax
import jax.numpy as jnp
import numpyro
from numpyro import handlers

from hcd_analysis.emulator import closure_legb as C
from hcd_analysis.emulator import data_likelihood as DL
from hcd_analysis.emulator import meanflux_prior as MF
from hcd_analysis.emulator.data import Z_LIMITS

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden")
MF_GOLDEN = os.path.join(GOLDEN_DIR, "legb_mf_golden.npz")
_CACHE = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/observables_tau0_lf.h5"
_CKPT0 = "/home/mfho/hcd_priya/checkpoints/final_fold0.eqx"
_DESI_NPZ = "/home/mfho/data/desi_dr1_p1d/desi_dr1_p1d.npz"
_KS = ("/home/mfho/lya_emulator_full/lyaemu/data/kodiaq_squad/"
       "final-conservative-p1d-karacayli_etal2021.txt")
_have = all(os.path.exists(p) for p in (_CACHE, _CKPT0, _DESI_NPZ))
_have_ks = os.path.exists(_KS)

# pivot used by α(z) = α₀·((1+z)/(1+z_p))^s. Pinned by the spec/plan (z_p = 3.0). The
# implementation exposes it as data_likelihood.Z_PIVOT; we mirror it here so the test states
# the contract value independently of the implementation (a guard against a silent pivot drift).
Z_PIVOT = 3.0


# --------------------------------------------------------------------------- #
#  Shared production-MF context (the SAME path the MF golden / real fit use).
# --------------------------------------------------------------------------- #
def _mf_ctx():
    """The production MF ctx (anchored res_corr attached) on the real DESI+KS legs — IDENTICAL
    config to ``test_legb_golden.test_legb_mf_golden`` so the α==1 no-op reproduces that golden."""
    ctx, d = C.build_legb_ctx(with_mf=True, mf_with_floor=True)
    assert ctx.mf is not None, "build_legb_ctx(with_mf=True) did not attach ctx.mf"
    return ctx, d


def _fwd(ctx, leg, theta9, alpha3, **kw):
    """``predict_P_obs_on_leg`` on one leg with the MF forward, returning P_model (np).

    ``kw`` carries the contract kwarg (e.g. ``alpha_res=(1.0, 0.0)`` or omitted for the default).
    becker13 τ₀ + the same szb/rzb/core wiring as the golden so this is the production forward.
    """
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


def _alpha_z(z, alpha0, s):
    """α(z) = α₀·((1+z)/(1+z_p))^s — the contract z-slope, mirrored for the assertions."""
    return alpha0 * ((1.0 + np.asarray(z)) / (1.0 + Z_PIVOT)) ** s


# --------------------------------------------------------------------------- #
#  1. α==1 reproduces the MF golden; default (None) == (1.0, 0.0) (the no-op).
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not (_have and _have_ks and os.path.exists(MF_GOLDEN)),
                    reason="real cache/ckpt/DESI/KS or MF golden not present")
def test_alpha_one_reproduces_golden():
    """With α₀=1, s=0 (α(z)≡1) the forward reproduces the Task-1.2 MF golden P_model to rtol
    1e-10 on DESI+KS (the no-op baseline), AND the DEFAULT (alpha_res omitted/None) is
    bit-identical to alpha_res=(1.0, 0.0)."""
    g = np.load(MF_GOLDEN, allow_pickle=True)
    ctx, _ = _mf_ctx()
    theta9 = jnp.asarray(g["theta9"]); alpha3 = jnp.asarray(g["alpha3"])
    for leg in ctx.legs:
        P_alpha1 = _fwd(ctx, leg, theta9, alpha3, alpha_res=(1.0, 0.0))
        P_default = _fwd(ctx, leg, theta9, alpha3)                      # alpha_res omitted -> None
        P_ref = g[f"{leg.name}_P"]
        # (a) α(z)≡1 reproduces the golden (the marginalization is a faithful no-op at α=1).
        np.testing.assert_allclose(P_alpha1, P_ref, rtol=1e-10, atol=0.0,
                                   err_msg=f"{leg.name}: alpha_res=(1,0) != MF golden")
        # (b) the DEFAULT (None) is BIT-IDENTICAL to (1.0, 0.0) — back-compat / golden untouched.
        np.testing.assert_array_equal(
            P_default, P_alpha1,
            err_msg=f"{leg.name}: default alpha_res=None != alpha_res=(1.0, 0.0)")


# --------------------------------------------------------------------------- #
#  2. α₀-response is forward-only and ~ the anchored log_rc shape; high-k only.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not (_have and _have_ks and os.path.exists(MF_GOLDEN)),
                    reason="real cache/ckpt/DESI/KS or MF golden not present")
def test_alpha_response_forward_only():
    """d log P_model / d α₀ at the fiducial (α₀=1, s=0):
      * NONZERO on KS (the high-k leg) and ~ the anchored log_rc shape, since
        P_model ∝ exp(α(z)·log_rc) ⇒ ∂ log P_model/∂α₀ = ((1+z)/(1+z_p))^s · log_rc = log_rc (s=0);
      * ≈0 on DESI's anchored (low-k) bins (res_corr→1 there ⇒ log_rc≈0 ⇒ no α leverage),
        matching the Phase-0 cos(α, n_s)_DESI≈0 (DESI is anchored away from the α handle).

    METHOD: jax.grad of log P_model w.r.t. α₀ (analytic, exact), compared to the independently
    computed anchored log_rc = log(mf.res_corr(z_phys)) per leg row.

    NOTE on tolerance: the forward applies α·log_rc at the CACHE grid then linearly interpolates
    P_obs (not logP) to leg-k, so the exact response is interp(P·log_rc)/interp(P), which differs
    from interp(log_rc) by the interp-in-P nonlinearity (~5e-6 on KS, since every KS leg-k point
    falls strictly between cache nodes). We therefore compare to atol=2e-5 — this still verifies the
    response IS the anchored log_rc shape (O(0.035) on KS, ~0 on DESI); a real α bug would be O(1).
    """
    g = np.load(MF_GOLDEN, allow_pickle=True)
    ctx, _ = _mf_ctx()
    theta9 = jnp.asarray(g["theta9"]); alpha3 = jnp.asarray(g["alpha3"])

    def _grad_logP_wrt_alpha0(leg):
        tau0_vec = MF.becker13_tau0(jnp.asarray(leg.z))
        szb = ctx.sigma_zb_per_leg.get(leg.name)
        rzb = ctx.rho_zb_per_leg.get(leg.name) if ctx.rho_zb_per_leg is not None else None
        core = jnp.mean(jnp.asarray(ctx.dla_core_leg[leg.name]), axis=0)

        def logP_sum_per_row(a0):
            P, _ = DL.predict_P_obs_on_leg(
                ctx.model, theta9, tau0_vec, alpha3, pf_stats=ctx.pf_stats,
                dla_core=core, cache_k=ctx.cache_k, leg=leg, sigma_zb=szb,
                alpha_centres=ctx.alpha_centres, cemu_inflate=ctx.cemu_inflate, rho_zb=rzb,
                mf=ctx.mf, alpha_res=(a0, 0.0))
            return jnp.log(P)                                   # (N,) per-row log P_model

        # jacobian of the (N,) log P_model vector w.r.t. the scalar α₀ → (N,) per-row response.
        jac = jax.jacfwd(logP_sum_per_row)(jnp.asarray(1.0))
        return np.asarray(jac)                                  # (N,) d logP_row / dα₀

    def _expected_log_rc(leg):
        """The anchored log res_corr per leg row (the EXACT analytic response at s=0)."""
        z_row = np.asarray(leg.z)[np.asarray(leg.z_idx)]        # (N,) z of each flat row
        k_row = np.asarray(leg.k)                               # (N,) k of each flat row
        out = np.zeros(z_row.shape[0])
        for i in range(z_row.shape[0]):
            rc_cache = np.asarray(ctx.mf.res_corr(float(z_row[i])))   # (Kc,) anchored res_corr
            rc_k = float(np.interp(k_row[i], np.asarray(ctx.cache_k), rc_cache))
            out[i] = np.log(rc_k)
        return out

    legs = {leg.name: leg for leg in ctx.legs}

    # ---- KS (high-k): response is NONZERO and tracks the anchored log_rc ----
    ks = legs["KS"]
    dKS = _grad_logP_wrt_alpha0(ks)
    log_rc_KS = _expected_log_rc(ks)
    assert np.max(np.abs(dKS)) > 1e-4, "KS α₀-response is ~0 — the nuisance has no high-k handle"
    # ∂ logP/∂α₀ ≈ log_rc (s=0): tracks the anchored log_rc shape to the interp-in-P artifact (~5e-6).
    np.testing.assert_allclose(dKS, log_rc_KS, rtol=1e-5, atol=2e-5,
                               err_msg="KS: d logP/dα₀ does not track the anchored log_rc shape")

    # ---- DESI (anchored low-k): response is ~0 where res_corr is anchored to 1 ----
    desi = legs["DESI"]
    dDESI = _grad_logP_wrt_alpha0(desi)
    log_rc_DESI = _expected_log_rc(desi)
    # consistency: the DESI response equals its log_rc too (forward-only, same mechanism)...
    np.testing.assert_allclose(dDESI, log_rc_DESI, rtol=1e-5, atol=2e-5,
                               err_msg="DESI: d logP/dα₀ does not track the anchored log_rc shape")
    # ...and the ANCHOR zeros DESI's response at low-k (where the +6% bump was): res_corr→1 ⇒ log_rc→0.
    # k < 0.008 s/km is deep in the 5×k_box anchor zone for every z (5·k_box ≈ 0.017–0.021 over z).
    desi_k = np.asarray(desi.k)
    low_k = desi_k < 0.008
    assert low_k.sum() > 0, "no DESI bins below the anchor zone — check k grid"
    assert np.max(np.abs(dDESI[low_k])) < 1e-3, \
        (f"DESI low-k α-response (max |{np.max(np.abs(dDESI[low_k])):.3g}|) is not anchored to ~0 — "
         "the 5×k_box anchor should zero res_corr (and thus the α handle) at low k")
    # NOTE: DESI's RESIDUAL α-handle lives at its HIGH-k bins (k≳0.02, incl. the He-II z≥2.8 deficit),
    # which sit above the anchor and so are NOT zero (max|dDESI|≈0.025). DESI n_s is protected by the
    # WHITENED Fisher (Phase-0: cos(α,n_s)_DESI≈0, σ(n_s) inflation 1.00× — C_data down-weights DESI's
    # high-k edge and it is n_s-orthogonal), NOT by the raw per-bin response being small. The Phase-2
    # injection/cert gate is the real arbiter of DESI-n_s protection.


# --------------------------------------------------------------------------- #
#  3. The matched-closure TRUTH is invariant to α (forward-only, no cancellation).
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI not present")
def test_truth_unchanged_by_alpha():
    """``make_truth_from_sim`` truth ``P_obs_true`` is INVARIANT to α (it keeps α≡1), while the
    FORWARD moves with α. Proves forward-only / no closure cancellation: if the truth tracked the
    sampled α, the nuisance would cancel in ΔP and be unconstrained.

    Two checks:
      (a) make_truth_from_sim takes NO alpha argument and its truth is fixed (it is built with the
          fixed res_corr at α≡1) — the truth construction has no α handle at all;
      (b) the FORWARD P_model at α₀=1.5 differs from α₀=1.0 on the high-k leg, but the TRUTH does
          not — the asymmetry between truth and forward is the forward-only property.
    """
    ctx, d = _mf_ctx() if _have_ks else (lambda r: r)(C.build_legb_ctx(with_mf=True))
    sims, _ = C.held_out_sims(d, fold=0)
    # the matched MF truth (α≡1 by construction — make_truth_from_sim applies the FIXED res_corr).
    truth = C.make_truth_from_sim(d, sims[0], fold=0, mf=ctx.mf)
    P_truth = np.asarray(truth["P_obs_true"]).copy()           # (nZs, Kc) fixed truth

    # (a) the truth function exposes NO α knob (the sampled α never reaches the truth path).
    import inspect
    truth_params = set(inspect.signature(C.make_truth_from_sim).parameters)
    assert not (truth_params & {"alpha_res", "alpha0", "alpha_res_amp", "alpha_res_slope"}), \
        ("make_truth_from_sim must NOT accept an α argument — threading the sampled α into the "
         "truth would cancel it in the closure (unconstrained nuisance).")

    # (b) the FORWARD moves with α (high-k leg); the truth above is untouched by the same change.
    if _have_ks:
        ks = [l for l in ctx.legs if l.name == "KS"][0]
        theta9 = jnp.full(9, 0.5)
        alpha3 = jnp.asarray([0.27, 0.09, 0.016])
        P_fwd_1 = _fwd(ctx, ks, theta9, alpha3, alpha_res=(1.0, 0.0))
        P_fwd_15 = _fwd(ctx, ks, theta9, alpha3, alpha_res=(1.5, 0.0))
        assert np.max(np.abs(P_fwd_15 / P_fwd_1 - 1.0)) > 1e-4, \
            "the FORWARD did not move when α₀: 1→1.5 — α is not threaded into the forward"
    # the truth array is, and remains, the α≡1 construction (re-load to prove determinism / no
    # hidden α state mutated it).
    truth2 = C.make_truth_from_sim(d, sims[0], fold=0, mf=ctx.mf)
    np.testing.assert_array_equal(
        P_truth, np.asarray(truth2["P_obs_true"]),
        err_msg="make_truth_from_sim truth is not invariant/deterministic (α leaked into truth)")


# --------------------------------------------------------------------------- #
#  4. α(z) z-slope shape: pivot at z=3, monotone increasing in z for s>0.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not (_have and os.path.exists(MF_GOLDEN)),
                    reason="real cache/ckpt/DESI or MF golden not present")
def test_alpha_z_zslope():
    """α(z) = α₀·((1+z)/(1+3.0))^s : at z=3 α(z)==α₀ (the pivot), and s>0 makes α(z) larger at
    z>3 (and smaller at z<3). Probed THROUGH the forward: the response ∂logP/∂α₀ at row z equals
    α(z)/α₀ · log_rc only when α(z) carries the z-slope; here we pin the slope shape directly via
    the forward P_model ratio between two s values at fixed α₀ on a z>3 vs z<3 high-k bin.

    The cleanest, implementation-independent check is the algebraic identity itself (the contract
    formula), plus a forward consistency check that the z-slope enters multiplicatively in the
    res_corr exponent.
    """
    # --- pure algebra of the contract formula (no model) ---
    assert _alpha_z(3.0, alpha0=1.3, s=0.7) == pytest.approx(1.3), "α(z=z_p) must equal α₀ (pivot)"
    assert _alpha_z(4.0, alpha0=1.0, s=0.5) > _alpha_z(3.0, alpha0=1.0, s=0.5) > \
        _alpha_z(2.2, alpha0=1.0, s=0.5), "s>0 must make α(z) increase with z"
    assert _alpha_z(4.0, alpha0=1.0, s=-0.5) < _alpha_z(3.0, alpha0=1.0, s=-0.5), \
        "s<0 must make α(z) decrease with z"

    # --- forward consistency: the z-slope enters as exp((α(z)-1)·log_rc) per z row ---
    g = np.load(MF_GOLDEN, allow_pickle=True)
    ctx, _ = _mf_ctx() if _have_ks else (lambda r: r)(C.build_legb_ctx(with_mf=True))
    theta9 = jnp.asarray(g["theta9"]); alpha3 = jnp.asarray(g["alpha3"])
    # use the leg with the widest z-coverage AND high-k leverage (KS if present, else DESI).
    leg = ([l for l in ctx.legs if l.name == "KS"] or
           [l for l in ctx.legs if l.name == "DESI"])[0]
    z_row = np.asarray(leg.z)[np.asarray(leg.z_idx)]
    k_row = np.asarray(leg.k)

    s_val = 0.6
    P_s0 = _fwd(ctx, leg, theta9, alpha3, alpha_res=(1.0, 0.0))     # α(z)≡1 everywhere
    P_s = _fwd(ctx, leg, theta9, alpha3, alpha_res=(1.0, s_val))    # α(z)=((1+z)/(1+z_p))^s
    # expected per-row ratio = exp((α(z)-1)·log_rc(z,k)); α₀=1 so α(z)=((1+z)/(1+z_p))^s.
    az = _alpha_z(z_row, alpha0=1.0, s=s_val)                       # (N,)
    log_rc = np.zeros_like(z_row)
    for i in range(z_row.shape[0]):
        rc_cache = np.asarray(ctx.mf.res_corr(float(z_row[i])))
        log_rc[i] = np.log(float(np.interp(k_row[i], np.asarray(ctx.cache_k), rc_cache)))
    expected_ratio = np.exp((az - 1.0) * log_rc)
    np.testing.assert_allclose(P_s / P_s0, expected_ratio, rtol=1e-6, atol=1e-9,
                               err_msg="the z-slope does not enter as exp((α(z)-1)·log_rc) per z")


# --------------------------------------------------------------------------- #
#  5. The α sample sites appear in BOTH models, same relative position.
# --------------------------------------------------------------------------- #
def _sites(model, *a, seed=1):
    tr = handlers.trace(handlers.seed(model, jax.random.PRNGKey(seed))).get_trace(*a)
    return [k for k, v in tr.items() if v["type"] == "sample" and not v.get("is_observed")]


@pytest.mark.skipif(not (_have and _have_ks), reason="real cache/ckpt/DESI/KS not present")
def test_alpha_sites_in_both_models_same_order():
    """The α nuisance sites (``alpha_res`` amplitude + ``alpha_res_slope``) must appear in BOTH
    ``_legb_model`` and ``_legb_priors_only`` at the SAME relative position — the fast-postprocess
    constrain_fn traces ``_legb_priors_only``; a site-order mismatch corrupts it."""
    ctx, d = C.build_legb_ctx(desi_kwargs=dict(z_lo=0.0, z_hi=2.6), use_xclass=True)
    ctx = ctx._replace(legs=[l for l in ctx.legs if l.name == "DESI"])
    sims, _ = C.held_out_sims(d, fold=0)
    truth = C.make_truth_from_sim(d, sims[0], fold=0)
    mock_legs, _, _ = C.make_legb_mock(ctx, truth, jax.random.PRNGKey(0))
    core = C._mock_core_per_leg(ctx, truth)

    s_model = _sites(C._legb_model, ctx, mock_legs, core)
    s_prior = _sites(C._legb_priors_only, ctx)

    # (a) BOTH α sites present in BOTH models.
    for nm in ("alpha_res", "alpha_res_slope"):
        assert nm in s_model, f"{nm!r} missing from _legb_model sample sites: {s_model}"
        assert nm in s_prior, f"{nm!r} missing from _legb_priors_only sample sites: {s_prior}"

    # (b) the amplitude site precedes the slope site in both (sampled adjacently, this order).
    assert s_model.index("alpha_res") < s_model.index("alpha_res_slope"), \
        f"_legb_model: alpha_res must precede alpha_res_slope: {s_model}"
    assert s_prior.index("alpha_res") < s_prior.index("alpha_res_slope"), \
        f"_legb_priors_only: alpha_res must precede alpha_res_slope: {s_prior}"

    # (c) IDENTICAL site order across the two models (the constrain_fn invariant) — this is the
    # load-bearing assertion; it subsumes the per-site relative-position check.
    assert s_model == s_prior, \
        f"site-order mismatch model {s_model} vs priors {s_prior} — constrain_fn would corrupt"

    # (d) the two α sites are at the SAME relative position within the shared ordering.
    assert s_model.index("alpha_res") == s_prior.index("alpha_res")
    assert s_model.index("alpha_res_slope") == s_prior.index("alpha_res_slope")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
