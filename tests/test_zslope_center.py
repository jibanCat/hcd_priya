"""Regression guard for the FORWARD HCD z-slope center (dN/dX wrong-slope fix).

The forward incidence weight is α_c(z) = α_pivot · ((1+z)/(1+z_p))^s_c, where s_c is the
per-class z-slope returned by ``closure_legb._zslope_sites`` on the NON-2D path (fixed OR
marginalize_zslope; the 2D-tilt path is a different site and is covered by test_legb_2d_tilt).

s_c MUST be centered on the SIM incidence-weight slope ``HCD_INCIDENCE_SLOPE`` (≈2.4 — the
slope the held-out-sim mock TRUTH's native w_c(z) carries), NOT on the lit/sim RATIO slope
``HCD_LIT_OVER_SIM_SLOPE`` (≈0.95, a DIFFERENT object: d ln(lit/sim)/d ln(1+z), evaluated at
the z=3 pivot only). Centering on 0.95 makes the predicted dN/dX(z) FALL with z (truth rises)
and puts the mock-truth slope 2.9–6σ off-center — the bug.

These tests are lightweight (SimpleNamespace ctx; no cache/ckpt/data) — they trace ONLY the
HCD z-slope sample sites of ``_zslope_sites``.

Env (MANDATORY):
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
    /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_zslope_center.py -q
"""
import numpy as np
import hcd_analysis.emulator  # noqa: F401  (x64 before jax)
import jax
import jax.numpy as jnp
from types import SimpleNamespace
from numpyro import handlers

from hcd_analysis.emulator import closure_legb as CL
from hcd_analysis.emulator.inference import (HCD_LIT_OVER_SIM_SLOPE, HCD_Z_PIVOT,
                                             lit_over_sim_at_z, HCD_LLS_REALFIT_ZSLOPE)
import pytest


def _fake_ctx(marginalize_zslope=False, zslope_mu=None, zslope_sigma=None):
    return SimpleNamespace(
        marginalize_zslope=marginalize_zslope,
        zslope_mu=zslope_mu,
        zslope_sigma=zslope_sigma,
    )


def test_btilt_site_runtime_guard_catches_reverted_2d_center():
    """Gap-1 (adversarial-recurrence review): the 2D-tilt RUNTIME exponent site _btilt_site must
    ALSO trip the guard on a 0.95 reversion of hcd_btilt_mu — not only the build-time guard — so a
    future ctx._replace(hcd_btilt_mu=...) / new builder cannot silently revert the 2D forward slope.
    The guard is _btilt_site's first line, so it raises before numpyro.sample (no trace needed)."""
    bad = SimpleNamespace(hcd_btilt_mu=float(HCD_LIT_OVER_SIM_SLOPE[0]),   # 0.95, the wrong object
                          hcd_btilt_sigma=0.5, hcd_dslope=[0.0, 0.293, -0.099])
    with pytest.raises(AssertionError):
        CL._btilt_site(bad)
    # the correct incidence-weight center passes the guard
    CL._assert_forward_zslope_center(float(CL.HCD_INCIDENCE_SLOPE[0]), "_btilt_site")


def test_fixed_branch_returns_incidence_slope_not_ratio_slope():
    """marginalize_zslope=False → _zslope_sites returns the SIM incidence slope
    HCD_INCIDENCE_SLOPE (the mock-truth w_c(z) slope), NOT the lit/sim ratio HCD_LIT_OVER_SIM_SLOPE."""
    ctx = _fake_ctx(marginalize_zslope=False)
    s_c = np.asarray(CL._zslope_sites(ctx))
    incidence = np.asarray(CL.HCD_INCIDENCE_SLOPE)
    ratio = np.asarray(HCD_LIT_OVER_SIM_SLOPE)
    np.testing.assert_allclose(s_c, incidence, rtol=0, atol=0,
                               err_msg=f"fixed forward s_c {s_c} must be the incidence slope "
                                       f"{incidence}, not the lit/sim ratio {ratio}")
    # and it must NOT be the ratio slope (guard against the bug recurring).
    assert not np.allclose(s_c, ratio), \
        f"fixed forward s_c is the lit/sim RATIO slope {ratio} — the wrong-object bug"


def test_marginalized_default_center_is_incidence_slope():
    """marginalize_zslope=True, zslope_mu=None → the Normal prior CENTER (loc) of the s_lls/
    s_subdla/s_dla sites is HCD_INCIDENCE_SLOPE, NOT HCD_LIT_OVER_SIM_SLOPE."""
    ctx = _fake_ctx(marginalize_zslope=True, zslope_mu=None)
    tr = handlers.trace(handlers.seed(
        lambda: CL._zslope_sites(ctx), jax.random.PRNGKey(0))).get_trace()
    incidence = np.asarray(CL.HCD_INCIDENCE_SLOPE)
    ratio = np.asarray(HCD_LIT_OVER_SIM_SLOPE)
    locs = np.array([float(tr[nm]["fn"].loc) for nm in ("s_lls", "s_subdla", "s_dla")])
    np.testing.assert_allclose(locs, incidence, rtol=0, atol=0,
                               err_msg=f"marginalized None-default center {locs} must be the "
                                       f"incidence slope {incidence}, not the ratio {ratio}")
    assert not np.allclose(locs, ratio), \
        f"marginalized None-default center is the lit/sim RATIO slope {ratio} — the bug"


def test_marginalized_explicit_mu_is_respected():
    """An explicit ctx.zslope_mu still overrides (the fix only changes the None-DEFAULT center).
    Use a center above the forward-z-slope guard floor (incidence-slope-like) — an explicit center
    is a forward EXPONENT center too, so it must still be a legitimate incidence slope (>1.5)."""
    custom = jnp.asarray([2.6, 2.7, 2.5])
    ctx = _fake_ctx(marginalize_zslope=True, zslope_mu=custom)
    tr = handlers.trace(handlers.seed(
        lambda: CL._zslope_sites(ctx), jax.random.PRNGKey(0))).get_trace()
    locs = np.array([float(tr[nm]["fn"].loc) for nm in ("s_lls", "s_subdla", "s_dla")])
    np.testing.assert_allclose(locs, np.asarray(custom), rtol=0, atol=0)


# --------------------------------------------------------------------------- #
#  PI re-determination 2026-06-17: the REAL-FIT (survey) path uses litWLS γ_LLS  #
#  = 2.127 for the LLS forward z-slope; the CLOSURE (survey=None) stays 2.465.   #
# --------------------------------------------------------------------------- #
def test_realfit_litwls_lls_slope_constant_and_guard():
    """The real-fit LLS forward z-slope is the literature slope γ_LLS=2.127 — KEPT under the
    corrected-law re-derivation (PI 2026-07-18 decision 1c: the corrected K1a free-gamma fit
    2.137 is consistent, so the deployed LLS law is CONSTRAINED to 2.127 and the test-enforced
    identity HCD_LLS_REALFIT_ZSLOPE == HCD_LIT_DNDX_LAW['LLS'][1] holds exactly). NOT the sim
    incidence slope 2.465 the closure carries, NOR the lit/sim RATIO slope. ABOVE the
    forward-z-slope guard floor (1.5) so _assert_forward_zslope_center PASSES it."""
    from hcd_analysis.emulator import inference as INF
    assert HCD_LLS_REALFIT_ZSLOPE == pytest.approx(2.127, abs=1e-6), \
        "the real-fit LLS forward z-slope must be the literature γ_LLS=2.127"
    # test-enforced equality with the deployed corrected law's slope (spec sec 5.2)
    assert HCD_LLS_REALFIT_ZSLOPE == INF.HCD_LIT_DNDX_LAW["LLS"][1], \
        "HCD_LLS_REALFIT_ZSLOPE must equal HCD_LIT_DNDX_LAW['LLS'][1] (constrained-fit identity)"
    # distinct from the closure sim-truth slope AND the lit/sim ratio slope
    assert HCD_LLS_REALFIT_ZSLOPE != pytest.approx(float(CL.HCD_INCIDENCE_SLOPE[0]))   # != 2.465
    assert HCD_LLS_REALFIT_ZSLOPE > float(HCD_LIT_OVER_SIM_SLOPE[0])                   # != ratio slope
    # the litWLS LLS slope passes the guard (2.127 > 1.5)
    CL._assert_forward_zslope_center(HCD_LLS_REALFIT_ZSLOPE, "litWLS LLS slope")
    # and the full real-fit zslope_mu vector (litWLS LLS, sim subDLA/DLA) passes the LLS-slot guard
    realfit_mu = np.array([HCD_LLS_REALFIT_ZSLOPE, CL.HCD_INCIDENCE_SLOPE[1], CL.HCD_INCIDENCE_SLOPE[2]])
    CL._assert_forward_zslope_center(realfit_mu, "real-fit litWLS zslope_mu")


def test_realfit_survey_path_centers_lls_on_litwls():
    """Trace the marginalize-zslope sites with the REAL-FIT zslope_mu = (2.127, sim_subDLA, sim_DLA):
    s_lls centers on the litWLS slope 2.127 while subDLA/DLA stay on the SIM incidence slope. This
    is the real-fit survey-path center build_legb_ctx(survey=…) now plumbs (the CLOSURE survey=None
    keeps the None-default center HCD_INCIDENCE_SLOPE — test_marginalized_default above)."""
    realfit_mu = jnp.asarray([HCD_LLS_REALFIT_ZSLOPE,
                              float(CL.HCD_INCIDENCE_SLOPE[1]), float(CL.HCD_INCIDENCE_SLOPE[2])])
    ctx = _fake_ctx(marginalize_zslope=True, zslope_mu=realfit_mu)
    tr = handlers.trace(handlers.seed(
        lambda: CL._zslope_sites(ctx), jax.random.PRNGKey(0))).get_trace()
    locs = np.array([float(tr[nm]["fn"].loc) for nm in ("s_lls", "s_subdla", "s_dla")])
    assert locs[0] == pytest.approx(2.127, abs=1e-6), "real-fit s_lls center must be litWLS 2.127"
    assert locs[1] == pytest.approx(float(CL.HCD_INCIDENCE_SLOPE[1]), abs=1e-6)   # subDLA stays sim
    assert locs[2] == pytest.approx(float(CL.HCD_INCIDENCE_SLOPE[2]), abs=1e-6)   # DLA stays sim
    # and the closure None-default is the SIM LLS slope 2.465 — the two paths DIFFER on the LLS slot.
    assert locs[0] != pytest.approx(float(CL.HCD_INCIDENCE_SLOPE[0]))


# --------------------------------------------------------------------------- #
#  5b. The 2D-tilt B_hcd center is HCD_INCIDENCE_SLOPE[0] (cache-free anchor).   #
# --------------------------------------------------------------------------- #
def test_2d_btilt_center_is_incidence_slope_lls():
    """The 2D AMPLITUDE×TILT global-tilt center B_hcd_mu = HCD_INCIDENCE_SLOPE[0] (the LLS incidence
    slope ~2.46), and the class-differential δs_c = HCD_INCIDENCE_SLOPE − HCD_INCIDENCE_SLOPE[0] so
    that at B_hcd = center the per-class slopes equal HCD_INCIDENCE_SLOPE — NOT the lit/sim ratio.
    (build_legb_ctx wires these from real cache; here we pin the CONSTANTS the wiring uses, so this
    runs without the cache — test_legb_2d_tilt covers the full ctx build when the cache is present.)"""
    incid = np.asarray(CL.HCD_INCIDENCE_SLOPE)
    ratio = np.asarray(HCD_LIT_OVER_SIM_SLOPE)
    btilt_mu = float(incid[0])
    assert btilt_mu == pytest.approx(2.465, abs=1e-9), f"B_hcd center {btilt_mu} != HCD_INCIDENCE_SLOPE[0]"
    assert not np.isclose(btilt_mu, float(ratio[0])), \
        f"B_hcd center is the lit/sim RATIO LLS slope {ratio[0]} — the wrong-object bug"
    # and the guard would PASS this center but FAIL the ratio center.
    CL._assert_forward_zslope_center(btilt_mu, "test 2D center")
    with pytest.raises(AssertionError):
        CL._assert_forward_zslope_center(float(ratio[0]), "test ratio center")


# --------------------------------------------------------------------------- #
#  5c. SIGN test — the forward dN/dX(z) RISES with z (the test that catches the  #
#      bug: with the 0.95 ratio slope it FALLS).                                 #
# --------------------------------------------------------------------------- #
def _dndx_z_at_slope(slope, z, xbar):
    """Forward dN/dX_c(z) = alpha_to_dndx(α_pivot·((1+z)/(1+z_p))^s_c, Xbar(z), z) — the (APPROXIMATE,
    renorm-ignoring; fine for this sign test) map
    scripts/scratch_hcd_dndx_loso_vs_lit.py uses. ``xbar`` is Xbar(z) (mean absorption path per
    sightline), which RISES with z in PRIYA (the cache fit goes ~0.45→1.08 over z 2.4→4.2). That
    rising denominator is what makes the SIGN of d ln dN/dX/d ln(1+z) discriminating: at the
    incidence slope (~2.4) the (1+z)^s_c growth WINS → dN/dX rises; at the ratio slope (~0.95) the
    rising Xbar WINS → dN/dX FALLS (the bug). Representative, illustrative Xbar(z) — matches the
    real cache trend without loading the 1 GB cache."""
    from hcd_analysis.emulator.dndx_wc import alpha_to_dndx
    alpha_pivot = np.array([0.27, 0.09, 0.004])      # representative LLS/subDLA/DLA pivot weights
    zp = float(HCD_Z_PIVOT)
    s = np.asarray(slope, float)
    out = np.empty((len(z), 3))
    for i, zz in enumerate(z):
        a_z = alpha_pivot * ((1.0 + zz) / (1.0 + zp)) ** s
        out[i] = np.asarray(alpha_to_dndx(jnp.asarray(a_z), jnp.asarray(float(xbar[i])),
                                          jnp.asarray(float(zz))))
    return out


# Xbar(z) RISING with z, matching the PRIYA cache fit (build_xbar) over the in-range z window.
_Z_GRID = np.array([2.4, 3.0, 3.6, 4.2])
_XBAR_Z = np.array([0.454, 0.632, 0.839, 1.076])      # ≈ the real cache Xbar(z) at these z


def test_forward_dndx_rises_with_z_at_incidence_slope():
    """The forward dN/dX_c(z) built at the INCIDENCE slope HCD_INCIDENCE_SLOPE RISES with z for ALL
    three HCD classes (d ln dN/dX / d ln(1+z) > 0) — matching the literature + mock truth. This is
    the SIGN test that would have caught the wrong-object bug (and catches any future reversion).
    Uses the rising Xbar(z) of the real cache so the sign is the figure's sign (see _dndx_z_at_slope)."""
    dndx = _dndx_z_at_slope(CL.HCD_INCIDENCE_SLOPE, _Z_GRID, _XBAR_Z)
    lz = np.log(1.0 + _Z_GRID)
    for j, cls in enumerate(("LLS", "subDLA", "DLA")):
        slope_fit = np.polyfit(lz, np.log(dndx[:, j]), 1)[0]   # d ln dN/dX / d ln(1+z)
        assert slope_fit > 0.0, (
            f"forward dN/dX_{cls}(z) FALLS with z (d ln dN/dX/d ln(1+z)={slope_fit:.3f} ≤ 0) at the "
            f"incidence slope — a sign the forward z-exponent reverted to the lit/sim RATIO slope "
            f"(~0.95). See hcd-dndx-zslope-bug.")


def test_forward_dndx_falls_with_z_at_ratio_slope_documents_the_bug():
    """The COMPLEMENT (documents the bug's symptom): at the WRONG lit/sim RATIO slope
    HCD_LIT_OVER_SIM_SLOPE the forward dN/dX(z) FALLS with z for ALL three classes (the rising Xbar(z)
    overwhelms the shallow (1+z)^0.95 growth) — exactly the failure the figure showed. Pins WHY the
    incidence slope is required, and that the sign FLIPS between the two slope objects."""
    dndx = _dndx_z_at_slope(HCD_LIT_OVER_SIM_SLOPE, _Z_GRID, _XBAR_Z)
    lz = np.log(1.0 + _Z_GRID)
    for j, cls in enumerate(("LLS", "subDLA", "DLA")):
        slope_fit = np.polyfit(lz, np.log(dndx[:, j]), 1)[0]
        assert slope_fit < 0.0, (
            f"sanity: with the 0.95 ratio slope dN/dX_{cls}(z) should FALL (got {slope_fit:+.3f}); if "
            f"this fails the bug's symptom changed — revisit the forward map / Xbar(z) regime.")


# --------------------------------------------------------------------------- #
#  5d. HCD_LIT_OVER_SIM_SLOPE is consumed ONLY at the z=3 pivot (slope cancels). #
# --------------------------------------------------------------------------- #
def test_ratio_slope_only_acts_off_pivot_not_at_z3():
    """lit_over_sim_at_z(z=3) is UNCHANGED if the ratio slope is zeroed — i.e. HCD_LIT_OVER_SIM_SLOPE
    has NO effect at the z=3 PIVOT (its only legitimate consumption point); it acts ONLY off-pivot.
    This pins that the ratio slope is a z=3-pivot prior-center quantity, never the forward exponent."""
    zp = float(HCD_Z_PIVOT)
    at_pivot = np.asarray(lit_over_sim_at_z(jnp.asarray(zp)))
    at_pivot_zeroed = np.asarray(lit_over_sim_at_z(jnp.asarray(zp), slope=jnp.zeros(3)))
    np.testing.assert_allclose(at_pivot, at_pivot_zeroed, rtol=0, atol=1e-12,
                               err_msg="lit_over_sim_at_z(z=3) depends on the ratio slope — it must "
                                       "cancel at the pivot (the slope's only consumption point).")
    # off-pivot, the slope DOES matter (so the test above isn't vacuous):
    z_off = zp + 1.0
    full = np.asarray(lit_over_sim_at_z(jnp.asarray(z_off)))
    zeroed = np.asarray(lit_over_sim_at_z(jnp.asarray(z_off), slope=jnp.zeros(3)))
    assert not np.allclose(full, zeroed), "ratio slope must act OFF the pivot (else the test is vacuous)"


# --------------------------------------------------------------------------- #
#  Guard fires on a 0.95 reversion of the forward-exponent CENTER.              #
# --------------------------------------------------------------------------- #
def test_guard_fires_on_ratio_slope_reversion():
    """The runtime guard _assert_forward_zslope_center RAISES if the forward z-slope center is
    reverted to the lit/sim RATIO slope (~0.95), and PASSES on the incidence slope (~2.4)."""
    CL._assert_forward_zslope_center(CL.HCD_INCIDENCE_SLOPE, "incidence ok")
    with pytest.raises(AssertionError, match="hcd-dndx-zslope-bug"):
        CL._assert_forward_zslope_center(HCD_LIT_OVER_SIM_SLOPE, "ratio reversion")


def test_fixed_branch_guard_catches_a_reverted_constant(monkeypatch):
    """If a future edit reverts the module constant HCD_INCIDENCE_SLOPE to the ratio slope, the
    _zslope_sites FIXED branch raises via the guard (reversion fails LOUDLY at trace time)."""
    monkeypatch.setattr(CL, "HCD_INCIDENCE_SLOPE", tuple(HCD_LIT_OVER_SIM_SLOPE))
    ctx = _fake_ctx(marginalize_zslope=False)
    with pytest.raises(AssertionError, match="hcd-dndx-zslope-bug"):
        CL._zslope_sites(ctx)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
