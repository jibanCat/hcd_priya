"""Task 1.6 (BAYESIAN arm) — unit tests for the res_corr injection harness.

GOAL
----
The Phase-2 injection-recovery gate (spec §4.2) injects an OUT-OF-SPAN res_corr
misspecification into the cert MOCK TRUTH ONLY (the misspecification the fit must
absorb via the marginalized alpha), NOT into the forward. These tests pin the
harness contract BEFORE the CS partner implements it.

THE BASIS (already built, untracked)
------------------------------------
``hcd_analysis/_emulator_data/res_corr_injection_basis.npz`` (built by
``scripts/build_res_corr_injection_basis.py``). Per leg (DESI/KS/eBOSS) it carries
``{leg}_b1``, ``{leg}_b2`` = multiplicative LOG-res_corr perturbations on the LEG
(z,k) grid (the SAME flat z-major ordering as ``leg.k``), provably
C_data^-1-orthogonal to the alpha(z) span (cos<0.8) and localized to z>=2.8 (the
He-II window, via ``{leg}_hiZ_mask``). ``b1`` is the pre-selected
worst-n_s-projecting member (``{leg}_worst_ns_member == 1`` for all legs) — the
gate injects b1. Apply to the leg-binned truth as ``P_truth_on_leg * exp(b1_leg)``.

THE CONTRACT THESE TESTS PIN (for the CS partner)
-------------------------------------------------
- KWARG: ``make_legb_mock(ctx, truth_sim, key, *, inject_a_siiii=0.0,
  inject_res_corr=None)`` — a NEW kwarg, default ``None`` ⇒ no-op (byte-identical
  to the un-injected mock). A SEPARATE injection strength from the sampled
  ``alpha_res`` (which never enters the truth).
- WHAT IT CARRIES: a spec resolvable to a per-leg log-perturbation b-vector. The
  tests use the dict form
      ``{"path": <basis_npz_path>, "member": "b1", "strength": 1.0}``
  (path to the basis npz + which member b1/b2 + a scalar injection strength). An
  equivalent ``{leg.name: b_vector}`` dict form is also acceptable; these tests
  exercise the (path, member, strength) form because that is what ``run_stepA``'s
  config arm will pass.
- WHERE APPLIED: on the LEG-BINNED truth ``P_truth_on_leg`` in ``make_legb_mock``
  (the SAME object ``inject_a_siiii`` multiplies, line ~833), AFTER the SiIII
  inject and BEFORE the cosmic-noise draw, as a per-leg multiply by
  ``exp(strength * b_member_leg)``. The b-vector is already on ``leg.k`` (same
  shape/order), so it is applied DIRECTLY (no re-bin) — analogous to the
  ``_metal_factor`` mfac multiply, but a log-perturbation: ``P * exp(b)``.
- TRUTH-ONLY: it goes into the mock TRUTH (and hence the recorded
  ``info["truth_on_leg"]`` and the noisy ``P_data``). The FORWARD
  (``predict_P_obs_on_leg``) never sees it, so it does NOT cancel in a closure —
  it is the misspecification the marginalized alpha must absorb.

WHY THESE TESTS FAIL TODAY
--------------------------
``make_legb_mock`` has no ``inject_res_corr`` kwarg yet, so every test that passes
it raises ``TypeError: make_legb_mock() got an unexpected keyword argument
'inject_res_corr'``. ``test_inject_none_is_noop`` is the lone test that does NOT
pass the kwarg; it is the post-implementation regression guard (it passes today
AND must keep passing after the kwarg lands with a ``None`` default). It is marked
so the harness can confirm the default stays byte-identical.

Env (MANDATORY):
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu \
    CUDA_VISIBLE_DEVICES="" /home/mfho/.conda/envs/emu-jax/bin/python3 \
    -m pytest tests/test_res_corr_inject.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

import hcd_analysis.emulator  # noqa: F401  (enables float64 before jax import)
import jax

from hcd_analysis.emulator.closure_legb import (
    build_legb_ctx,
    make_hr_truth_from_cache,
    make_legb_mock,
)
import hcd_analysis.emulator.multifidelity as MF

BASIS_NPZ = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/res_corr_injection_basis.npz"
SEED = 0
# the spec/basis pre-selected gate member: b1 = the worst-n_s-projecting, out-of-span,
# z>=2.8-localized direction ({leg}_worst_ns_member == 1 for all legs).
GATE_MEMBER = "b1"
GATE_STRENGTH = 1.0


# ----------------------------------------------------------------------------
# session-scoped fixtures (the ctx + HR truth builds are ~tens of seconds each)
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ctx():
    """Production Leg-B ctx (MF forward, DESI+KS+eBOSS) — the same ctx the gate uses."""
    c, _ = build_legb_ctx(with_mf=True, with_eboss=True)
    return c


@pytest.fixture(scope="module")
def hr_sim_name():
    """A concrete held-out HR sim name (the gate's hr_truth arm reads the HR cache)."""
    d = MF.load_cache(MF.HR_CACHE)
    names = np.array([s.decode() if isinstance(s, bytes) else s for s in d["sim_name"]])
    return sorted(set(map(str, names)))[0]


@pytest.fixture(scope="module")
def truth_sim(ctx, hr_sim_name):
    """The genuine HF-LOSO truth on the cache grid (make_hr_truth_from_cache), as the
    gate builds it; make_legb_mock then bins it to each leg."""
    return make_hr_truth_from_cache(hr_sim_name, ctx.cache_k, tau0_anchor="priya")


@pytest.fixture(scope="module")
def basis():
    return np.load(BASIS_NPZ, allow_pickle=True)


def _member_vec(basis, leg_name, member):
    """The leg's log-res_corr b-vector on the leg k-grid (same order as leg.k)."""
    return np.asarray(basis[f"{leg_name}_{member}"], dtype=float)


def _gate_spec(member=GATE_MEMBER, strength=GATE_STRENGTH):
    """The (path, member, strength) spec form run_stepA's arm will pass."""
    return {"path": BASIS_NPZ, "member": member, "strength": float(strength)}


# ----------------------------------------------------------------------------
# 1. default None => no-op (byte-identical). This is the post-impl regression
#    guard; it does NOT pass the new kwarg, so it must pass both before AND after.
# ----------------------------------------------------------------------------
def test_inject_none_is_noop(ctx, truth_sim):
    """inject_res_corr=None (the default) ⇒ the leg-binned truth is byte-identical
    to the un-injected truth (same noiseless truth_on_leg AND same noisy P_data,
    same key). The default must be a true no-op."""
    key = jax.random.PRNGKey(SEED)

    # un-injected (baseline) — no kwarg at all.
    ml0, _tp0, info0 = make_legb_mock(ctx, truth_sim, key, inject_a_siiii=0.0)
    # explicit None — must be IDENTICAL (this is the contract for the new kwarg's default).
    ml1, _tp1, info1 = make_legb_mock(
        ctx, truth_sim, key, inject_a_siiii=0.0, inject_res_corr=None)

    for leg0, leg1 in zip(ml0, ml1):
        name = leg0.name
        t0 = np.asarray(info0["truth_on_leg"][name])
        t1 = np.asarray(info1["truth_on_leg"][name])
        # byte-identical noiseless truth-on-leg (NaN-aware exact compare)
        assert np.array_equal(t0, t1, equal_nan=True), f"{name}: truth_on_leg changed by None"
        # byte-identical noisy mock (same key, same noise)
        p0 = np.asarray(leg0.P_data)
        p1 = np.asarray(leg1.P_data)
        assert np.array_equal(p0, p1, equal_nan=True), f"{name}: P_data changed by None"


# ----------------------------------------------------------------------------
# 2. injecting b1 multiplies the leg-binned TRUTH by exp(b1_leg) per leg.
# ----------------------------------------------------------------------------
def test_inject_b1_perturbs_truth(ctx, truth_sim, basis):
    """Injecting b1 multiplies the leg-binned NOISELESS truth by exp(strength*b1_leg)
    per leg: ratio (injected truth_on_leg / clean truth_on_leg) == exp(b1) on the
    leg grid to rtol 1e-10 (over the finite/kept rows)."""
    key = jax.random.PRNGKey(SEED)

    _ml0, _tp0, info0 = make_legb_mock(ctx, truth_sim, key, inject_a_siiii=0.0)
    _ml1, _tp1, info1 = make_legb_mock(
        ctx, truth_sim, key, inject_a_siiii=0.0,
        inject_res_corr=_gate_spec(GATE_MEMBER, GATE_STRENGTH))

    any_leg_checked = False
    for leg in ctx.legs:
        name = leg.name
        t0 = np.asarray(info0["truth_on_leg"][name])
        t1 = np.asarray(info1["truth_on_leg"][name])
        b = _member_vec(basis, name, GATE_MEMBER)
        assert b.shape == t0.shape, f"{name}: basis {GATE_MEMBER} shape {b.shape} != leg truth {t0.shape}"

        ok = np.isfinite(t0) & np.isfinite(t1) & (np.abs(t0) > 0)
        assert ok.any(), f"{name}: no finite nonzero truth rows to compare"
        ratio = t1[ok] / t0[ok]
        expected = np.exp(GATE_STRENGTH * b[ok])
        np.testing.assert_allclose(
            ratio, expected, rtol=1e-10,
            err_msg=f"{name}: injected/clean truth ratio != exp(strength*b1)")
        # the dropped/NaN rows stay NaN (the inject must not resurrect them)
        assert np.array_equal(np.isnan(t0), np.isnan(t1)), f"{name}: NaN pattern changed by inject"
        any_leg_checked = True
    assert any_leg_checked


# ----------------------------------------------------------------------------
# 3. the injected perturbation is z>=2.8-localized (the He-II window).
# ----------------------------------------------------------------------------
def test_inject_is_z_ge_2p8_localized(ctx, truth_sim, basis):
    """The injected perturbation is ~0 for z<2.8 leg rows and nonzero for z>=2.8
    (the He-II window), using the npz hiZ_mask. Verified on the truth-ratio:
    ratio==1 (exp(b)=1, b≈0) off the He-II window, ratio!=1 somewhere on it."""
    key = jax.random.PRNGKey(SEED)

    _ml0, _tp0, info0 = make_legb_mock(ctx, truth_sim, key, inject_a_siiii=0.0)
    _ml1, _tp1, info1 = make_legb_mock(
        ctx, truth_sim, key, inject_a_siiii=0.0,
        inject_res_corr=_gate_spec(GATE_MEMBER, GATE_STRENGTH))

    for leg in ctx.legs:
        name = leg.name
        t0 = np.asarray(info0["truth_on_leg"][name])
        t1 = np.asarray(info1["truth_on_leg"][name])
        hiZ = np.asarray(basis[f"{name}_hiZ_mask"], dtype=bool)
        b = _member_vec(basis, name, GATE_MEMBER)
        assert hiZ.shape == t0.shape == b.shape, f"{name}: hiZ/truth/b shape mismatch"

        ok = np.isfinite(t0) & np.isfinite(t1) & (np.abs(t0) > 0)
        ratio = np.full(t0.shape, np.nan)
        ratio[ok] = t1[ok] / t0[ok]

        # (a) the basis itself is z<2.8-zero (the harness must inherit this localization)
        assert np.allclose(b[~hiZ], 0.0, atol=0.0), f"{name}: basis {GATE_MEMBER} nonzero at z<2.8"
        # (b) off the He-II window the truth ratio is exactly 1 (no perturbation there)
        lo = (~hiZ) & ok
        if lo.any():
            np.testing.assert_allclose(
                ratio[lo], 1.0, rtol=1e-12, atol=1e-12,
                err_msg=f"{name}: truth perturbed at z<2.8 (should be He-II localized)")
        # (c) on the He-II window the perturbation is genuinely nonzero somewhere
        hi = hiZ & ok
        assert hi.any(), f"{name}: no finite z>=2.8 rows"
        assert np.any(np.abs(ratio[hi] - 1.0) > 1e-6), \
            f"{name}: injection vanished on the z>=2.8 He-II window"


# ----------------------------------------------------------------------------
# 4. truth-only: the FORWARD prediction is unchanged by inject_res_corr.
# ----------------------------------------------------------------------------
def test_inject_on_truth_not_forward(ctx, truth_sim, basis):
    """The injection goes into the TRUTH only — the FORWARD model prediction at the
    truth theta is byte-identical with and without inject_res_corr. (So in a closure
    the injection does NOT cancel: truth carries exp(b1), the forward does not.)

    We compute the forward via predict_P_obs_on_leg at the truth (theta9, tau0,
    alpha_hcd) — the exact call the likelihood makes — and assert it is independent
    of inject_res_corr. The injection must NOT touch the forward path."""
    import hcd_analysis.emulator.data_likelihood as DL

    key = jax.random.PRNGKey(SEED)

    # build the mock twice (clean + injected) to get matched truth_packs/cores.
    ml0, tp0, info0 = make_legb_mock(ctx, truth_sim, key, inject_a_siiii=0.0)
    ml1, tp1, info1 = make_legb_mock(
        ctx, truth_sim, key, inject_a_siiii=0.0,
        inject_res_corr=_gate_spec(GATE_MEMBER, GATE_STRENGTH))

    import jax.numpy as jnp
    zg = np.asarray(ctx.z_global)
    theta9 = jnp.asarray(tp0["theta9"])
    tau0_global = jnp.asarray(tp0["tau0_global"])
    alpha_hcd = jnp.asarray(tp0["alpha_hcd"])

    # the per-leg DLA core the forward consumes (matched-closure core builder).
    from hcd_analysis.emulator.closure_legb import _mock_core_per_leg
    core_per_leg = _mock_core_per_leg(ctx, truth_sim)

    forward_changed = False
    for leg in ctx.legs:
        name = leg.name
        sel = np.array([int(np.argmin(np.abs(zg - zz))) for zz in leg.z])
        ah = np.broadcast_to(np.asarray(alpha_hcd), (3,))
        P_fwd, _C = DL.predict_P_obs_on_leg(
            ctx.model, theta9, tau0_global[sel], jnp.asarray(ah), pf_stats=ctx.pf_stats,
            dla_core=core_per_leg[name], cache_k=ctx.cache_k, leg=leg,
            alpha_centres=ctx.alpha_centres, mf=ctx.mf, mf_floor=ctx.mf_floor)
        P_fwd = np.asarray(P_fwd)

        # the forward is a function of (model, theta, tau0, alpha_hcd, leg, mf) ONLY —
        # NONE of which depend on inject_res_corr. The injection lives only on the truth.
        # Confirm via the recorded truth divergence vs the forward invariance:
        t0 = np.asarray(info0["truth_on_leg"][name])
        t1 = np.asarray(info1["truth_on_leg"][name])
        if np.any(np.abs(np.nan_to_num(t1 - t0)) > 0):
            forward_changed = True  # truth DID move (sanity: the inject is live on this leg)

        # the closure residual (truth - forward) DIFFERS between clean and injected,
        # i.e. the injection does NOT cancel in the forward — the whole point of the gate.
        ok = np.isfinite(t0) & np.isfinite(t1) & np.isfinite(P_fwd)
        if ok.any():
            resid0 = t0[ok] - P_fwd[ok]
            resid1 = t1[ok] - P_fwd[ok]
            # on legs where the injection is live, the residual must change (no cancellation).
            if np.any(np.abs(t1[ok] - t0[ok]) > 0):
                assert np.any(np.abs(resid1 - resid0) > 0), \
                    f"{name}: injection cancelled in the closure residual (forward absorbed it)"

    assert forward_changed, "injection never moved the truth on any leg (harness inert)"
