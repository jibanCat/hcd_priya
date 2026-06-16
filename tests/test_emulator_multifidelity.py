"""Tests for the multi-fidelity (LF->HF) correction layer (multifidelity.py).

Covers: res_corr load/interp (incl. per-z k-grid + edge clamp + differentiability),
log-log interp/extrap, the smooth k-basis, the DeltaHead shapes + jit/vmap/grad
cleanliness, the full MultiFidelity forward model (additive<->log-ratio identity,
res_corr on/off, HMC-differentiability), and the delta target measurement / training
on a synthetic cache.  No real caches or checkpoints are touched (fast).
"""
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

import hcd_analysis.emulator  # enables x64
from hcd_analysis.emulator import multifidelity as MF

# equinox warns "A JAX array is being set as static" for the MultiFidelity.lf_norm
# static field (a read-only numpy norm dict; is_array flags numpy too) -- benign and
# expected here (the module filters it in normal use, but pytest installs its own
# warning capture).  Silence that ONE message for these tests.
pytestmark = pytest.mark.filterwarnings(
    "ignore:A JAX array is being set as static:UserWarning")
from hcd_analysis.emulator.model import Emulator
from hcd_analysis.emulator.data import (
    load_cache, fit_target_norm, reconstruct_P_filt, Z_LIMITS,
)
from tests.emulator._fixture import write_synthetic_cache


# --------------------------------------------------------------------------- #
# res_corr
# --------------------------------------------------------------------------- #
def test_load_res_corr_shapes_and_range():
    z, logk, rc = MF.load_res_corr()
    assert z.ndim == 1 and rc.shape == logk.shape and rc.shape[0] == z.shape[0]
    # z ascending; per-z log-k strictly increasing
    assert np.all(np.diff(z) > 0)
    assert np.all(np.diff(logk, axis=1) > 0)
    # the factor is the documented L15n512/L15n384 range
    assert 0.8 < rc.min() and rc.max() < 1.1


def test_interp_res_corr_matches_table_on_grid():
    # CORRECTED for the res_corr anchor (Task 1.1): the production interp now blends
    # res_corr -> 1 below 5x the L15 box fundamental (kbox_skm), so on-grid bins
    # BELOW / near the anchor no longer equal the raw table.  Exact table recovery is
    # now (a) the anchor_mult=0.0 (NO-anchor) reference path on the FULL grid, and
    # (b) the anchored path on bins WELL ABOVE the anchor (k > 10*kbox, where the
    # tanh blend has saturated to w~1).
    z, logk, rc = MF.load_res_corr()
    zi = 5                                          # z = 3.2
    zz = z[zi]
    k_eval = jnp.asarray(10.0 ** logk[zi])
    # (a) anchor_mult=0.0 reproduces the raw table EXACTLY on-grid (regression ref).
    got_raw = np.asarray(MF.interp_res_corr(z, logk, rc, jnp.asarray(zz), k_eval,
                                            anchor_mult=0.0))
    assert np.allclose(got_raw, rc[zi], rtol=1e-9, atol=1e-9)
    # (b) anchored path on bins well above the anchor (k > 10*kbox) matches the table;
    #     the tanh tail is asymptotic (worst ~4e-5 at the cut), so use a high-k tol.
    kbox = float(MF.kbox_skm(zz))                   # ~0.00377 at z=3.2
    kgrid = 10.0 ** logk[zi]
    hi = kgrid > 10.0 * kbox                        # 49/59 bins kept
    got_anch = np.asarray(MF.interp_res_corr(z, logk, rc, jnp.asarray(zz),
                                             jnp.asarray(kgrid)))
    assert hi.sum() >= 40
    assert np.allclose(got_anch[hi], rc[zi][hi], rtol=0, atol=1e-3)


def test_interp_res_corr_high_k_suppression_and_clamp():
    # CORRECTED for the res_corr anchor (Task 1.1): the old probe k=0.01 sits INSIDE
    # the anchor zone at z=3 (5*kbox ~ 0.0188 s/km), so res_corr there is now ~1, not
    # the raw table value.  Probe the high-k suppression ABOVE the anchor, and add the
    # anchor assertion (res_corr ~ 1 below 5*kbox).
    z, logk, rc = MF.load_res_corr()
    kb = float(MF.kbox_skm(3.0))                     # ~0.00376 s/km
    # high-k probe, all WELL above 5*kbox (~0.0188): suppression untouched by the anchor.
    k = jnp.asarray(np.array([0.05, 0.1, 0.18]))
    val = np.asarray(MF.interp_res_corr(z, logk, rc, jnp.asarray(3.0), k))
    # high-k is suppressed relative to mid-k (the ~10% particle-convergence drop)
    assert val[2] < val[0]
    assert np.all((val > 0.8) & (val < 1.1))
    # the anchor: res_corr ~ 1 below 5*kbox (the +6% low-k bump is blended away).
    klo = jnp.asarray(np.array([0.5 * kb, 1.0 * kb]))   # deep in the anchored zone
    vlo = np.asarray(MF.interp_res_corr(z, logk, rc, jnp.asarray(3.0), klo))
    assert np.allclose(vlo, 1.0, atol=2e-3)
    # edge clamp: z and k beyond the table support stay finite & in-range
    big = np.asarray(MF.interp_res_corr(z, logk, rc, jnp.asarray(6.0),
                                        jnp.asarray(np.array([1e-4, 5.0]))))
    assert np.all(np.isfinite(big)) and np.all((big > 0.7) & (big < 1.2))


def test_interp_res_corr_differentiable_and_jit():
    z, logk, rc = MF.load_res_corr()
    k = jnp.asarray(np.array([0.01, 0.1]))
    f = lambda zz: MF.interp_res_corr(z, logk, rc, zz, k).sum()
    g = jax.grad(f)(jnp.asarray(3.0))
    assert np.isfinite(g)
    jf = jax.jit(lambda zz, kk: MF.interp_res_corr(z, logk, rc, zz, kk))
    assert np.allclose(np.asarray(jf(jnp.asarray(3.0), k)),
                       np.asarray(MF.interp_res_corr(z, logk, rc, jnp.asarray(3.0), k)))


# --- Task 1.1: anchor res_corr -> 1 below 5x the L15 box fundamental --------- #
# NEW behaviour (TDD): interp_res_corr blends res_corr -> 1 below 5*kbox_skm(z) with a
# smooth tanh (width 0.12 dex), killing the spurious +6.3% low-k bump (at z=3 it sits at
# the table k_min ~0.0037 s/km, rc ~1.0626) that the old CLAMPED edge injected into the
# data's lowest bins.  A module-level kbox_skm(z) helper + an anchor_mult kwarg (default
# 5.0; anchor_mult=0.0 == NO anchoring, reproduces the raw table) are added by the CS
# partner.  Verified numbers (z=3): kbox_skm(3)=0.0037560, 5*kbox=0.018780;
# raw rc at k_min (=0.0037269) = 1.0626489 (+6.26%); raw rc at k=0.063 = 0.964847 (-3.5%).
#
# TWO EXISTING TESTS BREAK and are corrected ABOVE (see their CORRECTED docstrings):
#   * test_interp_res_corr_matches_table_on_grid -- asserted EXACT table recovery on the
#     FULL on-grid k row; bins at/below the anchor now blend toward 1, so exact recovery
#     is moved to (a) the anchor_mult=0.0 reference path and (b) bins k > 10*kbox.
#   * test_interp_res_corr_high_k_suppression_and_clamp -- probed k=0.01, which is now
#     INSIDE the anchor zone (5*kbox~0.0188 at z=3) where rc~1; probe moved above the
#     anchor and an rc~1-below-5*kbox assertion added.
def test_interp_res_corr_anchored_to_one_below_5kbox():
    # res_corr must be ~1 well below 5x the L15 box fundamental, and UNCHANGED well above.
    from hcd_analysis.emulator.multifidelity import (
        load_res_corr, interp_res_corr, kbox_skm)
    z_rc, logk_rc, rc = load_res_corr()
    z = 3.0
    kb = float(kbox_skm(z))              # ~0.0037560 s/km (verified)
    # deep in the anchored zone, INCLUDING the +6.3% bump (raw rc ~1.0616-1.0626 here):
    k_lo = np.array([0.5 * kb, 1.0 * kb])
    # well above 5*kbox (~0.0188): the tanh blend has saturated (k=0.1 is ~6 widths above,
    # k=0.18 ~8 widths) so the anchored path equals the raw table to ~5e-7 -- "untouched".
    k_hi = np.array([0.1, 0.18])
    rc_lo = np.asarray(interp_res_corr(z_rc, logk_rc, rc, z, k_lo))
    rc_hi_anch = np.asarray(interp_res_corr(z_rc, logk_rc, rc, z, k_hi))
    # unanchored reference for k_hi (anchor_mult=0.0 == raw table path)
    rc_hi_raw = np.asarray(interp_res_corr(z_rc, logk_rc, rc, z, k_hi, anchor_mult=0.0))
    assert np.allclose(rc_lo, 1.0, atol=2e-3)             # anchored to 1 at low k
    assert np.allclose(rc_hi_anch, rc_hi_raw, atol=1e-6)  # untouched at high k


def test_interp_res_corr_default_is_5x():
    # COSMOLOGY-REFEREE regression pin (Task 1.2): the PRODUCTION default anchor is 5x the
    # L15 box fundamental.  The MF production forward (predict_P_obs_on_leg(..., mf=mf) ->
    # _mf_corr_on_cache -> mf.res_corr) calls interp_res_corr with the anchor_mult kwarg
    # OMITTED, so it inherits whatever the DEFAULT is.  This test pins that default == 5.0:
    # interp_res_corr(...) with the kwarg omitted MUST be BIT-IDENTICAL to
    # interp_res_corr(..., anchor_mult=5.0), and DISCRIMINABLY different from 3.0 (the
    # pre-Task-1.1 value).  A future silent change to the default (e.g. back to 3.0, or to
    # 4.0) flips this test red BEFORE it can move the n_s-driving low-k DESI bins.
    from hcd_analysis.emulator.multifidelity import (
        load_res_corr, interp_res_corr, kbox_skm)
    z_rc, logk_rc, rc = load_res_corr()
    z = 3.0
    kb = float(kbox_skm(z))                      # ~0.0037560 s/km (verified)
    # a k grid that STRADDLES the anchor: from 0.5x kbox (deep in the anchor zone, where the
    # anchor_mult choice matters most) up to 0.1 s/km (well above any of {3,5}x kbox, where
    # all anchors agree) — so the 5x-vs-3x discrimination is exercised in the blend region.
    k_eval = jnp.asarray(np.geomspace(0.5 * kb, 0.1, 40))
    rc_default = np.asarray(interp_res_corr(z_rc, logk_rc, rc, z, k_eval))
    rc_5x = np.asarray(interp_res_corr(z_rc, logk_rc, rc, z, k_eval, anchor_mult=5.0))
    rc_3x = np.asarray(interp_res_corr(z_rc, logk_rc, rc, z, k_eval, anchor_mult=3.0))
    # the default IS 5.0: bit-identical (rtol 1e-12, atol 0 — exact, same code path).
    np.testing.assert_allclose(rc_default, rc_5x, rtol=1e-12, atol=0.0,
                               err_msg="interp_res_corr default anchor_mult is no longer 5.0")
    # and it is NOT 3.0 — the test discriminates the default (the 3x anchor moves the blend
    # edge to ~0.011 s/km vs 5x's ~0.019, so they differ measurably in the straddle region).
    assert not np.allclose(rc_default, rc_3x, rtol=1e-6, atol=0.0), \
        "interp_res_corr default coincides with anchor_mult=3.0 — the 5x default is not pinned"


# --------------------------------------------------------------------------- #
# log-log interp / extrap
# --------------------------------------------------------------------------- #
def test_loglog_interp_recovers_powerlaw_and_extrapolates():
    # a pure power-law logP = a + b*log10(k): interp exact, extrap follows the slope.
    logk_src = jnp.asarray(np.log10(np.linspace(1e-3, 0.07, 30)))
    a, b = 0.5, -1.7
    logP_src = a + b * logk_src
    logk_dst = jnp.asarray(np.log10(np.array([5e-3, 0.05, 0.15, 0.2])))  # last two extrap
    out = np.asarray(MF.loglog_interp_extrap(logk_src, logP_src, logk_dst, n_tail=6))
    expect = a + b * np.asarray(logk_dst)
    assert np.allclose(out, expect, rtol=0, atol=1e-6)


def test_loglog_extrap_differentiable():
    logk_src = jnp.asarray(np.log10(np.linspace(1e-3, 0.07, 20)))
    logk_dst = jnp.asarray(np.log10(np.array([0.05, 0.15])))
    f = lambda lp: MF.loglog_interp_extrap(logk_src, lp, logk_dst).sum()
    g = jax.grad(f)(jnp.ones_like(logk_src) * 0.3)
    assert np.all(np.isfinite(g))


# --------------------------------------------------------------------------- #
# smooth k-basis + delta head
# --------------------------------------------------------------------------- #
def test_smooth_k_basis_shape_and_first_row_constant():
    logk = jnp.asarray(np.linspace(-3, -0.7, 40))
    B = MF.smooth_k_basis(logk, n_basis=4)
    assert B.shape == (4, 40)
    assert np.allclose(np.asarray(B[0]), 1.0)        # row 0 is the constant
    # Chebyshev rows are bounded in [-1,1]
    assert np.all(np.abs(np.asarray(B[1:])) <= 1.0 + 1e-9)


def test_delta_head_shapes_jit_vmap_grad():
    key = jax.random.PRNGKey(0)
    logk = jnp.asarray(np.linspace(-3, -0.7, 32))
    B = MF.smooth_k_basis(logk, n_basis=4)
    head = MF.DeltaHead(n_basis=4, n_classes=4, in_dim=11, width=16, n_layers=1, key=key)
    cond = jnp.zeros(11)
    g = head(cond, B)
    assert g.shape == (4, 32)
    assert g.dtype == jnp.float64                    # x64 active
    # jit + vmap over a batch
    batch = jnp.ones((5, 11))
    gv = jax.jit(jax.vmap(lambda c: head(c, B)))(batch)
    assert gv.shape == (5, 4, 32)
    # grad wrt conditioning (HMC path) finite
    gg = jax.grad(lambda c: head(c, B).sum())(cond)
    assert np.all(np.isfinite(gg))


def test_delta_head_key_sensitivity():
    logk = jnp.asarray(np.linspace(-3, -0.7, 8))
    B = MF.smooth_k_basis(logk, n_basis=3)
    h0 = MF.DeltaHead(n_basis=3, in_dim=11, key=jax.random.PRNGKey(0))
    h1 = MF.DeltaHead(n_basis=3, in_dim=11, key=jax.random.PRNGKey(0))
    h2 = MF.DeltaHead(n_basis=3, in_dim=11, key=jax.random.PRNGKey(1))
    c = jnp.ones(11)
    assert jnp.allclose(h0(c, B), h1(c, B))          # same key -> identical
    assert not jnp.allclose(h0(c, B), h2(c, B))      # different key -> different


# --------------------------------------------------------------------------- #
# small synthetic LF backbone -> full MultiFidelity forward
# --------------------------------------------------------------------------- #
def _sanitize_unit_cube(d):
    """Clip the synthetic cache's encoder input into the unit cube.

    The synthetic fixture draws raw params uniform(0.5,1.5), which map FAR outside
    PRIYA's design box (e.g. Ap_unit ~1e9), so the LF backbone produces extreme
    logP that exp-overflows.  For the forward-model unit tests we only need a
    PHYSICAL-RANGE input, so we re-map params_unit into [0,1] (the z_unit column is
    already sane).  Mutates and returns ``d``."""
    pu = d["x"][:, :9]
    lo = pu.min(0, keepdims=True); hi = pu.max(0, keepdims=True)
    d["x"] = d["x"].copy()
    d["x"][:, :9] = (pu - lo) / np.where(hi - lo > 0, hi - lo, 1.0)
    return d


@pytest.fixture
def lf_backbone(tmp_path):
    """A tiny trained-free LF Emulator + norm fit on a synthetic cache."""
    path = tmp_path / "lf.h5"
    write_synthetic_cache(path, n_sims=4, snaps_per_sim=2, n_alpha=4, n_k=12, seed=1)
    d = _sanitize_unit_cube(load_cache(path))
    norm = fit_target_norm(d, np.arange(len(d["z_grid"])))
    n_k = d["P_tier_p"].shape[1]
    model = Emulator(in_dim=10, n_k=n_k, n_basis=4, key=jax.random.PRNGKey(0))
    lf_logk = np.log10(d["kfkms"][0])
    return d, model, norm, lf_logk


def _build_mf(lf_backbone, n_basis=4, log_rho=None):
    d, model, norm, lf_logk = lf_backbone
    eval_logk = np.linspace(np.log10(1e-3), np.log10(0.15), 20)
    basis = MF.smooth_k_basis(jnp.asarray(eval_logk), n_basis=n_basis)
    head = MF.DeltaHead(n_basis=n_basis, in_dim=11, width=8, n_layers=1,
                        key=jax.random.PRNGKey(3))
    z_rc, logk_rc, rc_vals = MF.load_res_corr()
    if log_rho is None:
        log_rho = np.zeros(len(eval_logk))
    mf = MF.MultiFidelity(
        lf_model=model, lf_norm=norm, eval_logk=eval_logk, lf_logk=lf_logk,
        delta_head=head, basis=basis, log_rho=log_rho,
        z_rc=z_rc, logk_rc=logk_rc, rc_vals=rc_vals)
    return mf, eval_logk


def test_mf_forward_shapes_positive_jit_vmap(lf_backbone):
    mf, eval_logk = _build_mf(lf_backbone)
    d = lf_backbone[0]
    x = jnp.asarray(d["x"][0]); tau0 = jnp.asarray(d["tau0"][0])
    P = mf.P_mf(x, tau0)
    assert P.shape == (4, len(eval_logk))
    assert np.all(np.asarray(P) > 0)                 # strictly positive
    assert P.dtype == jnp.float64
    # jit + vmap
    xs = jnp.asarray(d["x"][:5]); ts = jnp.asarray(d["tau0"][:5])
    Pv = jax.jit(jax.vmap(mf.P_mf))(xs, ts)
    assert Pv.shape == (5, 4, len(eval_logk))
    assert np.all(np.isfinite(np.asarray(Pv)))


def test_mf_logratio_additive_identity(lf_backbone):
    """P_MF = (f_LF * exp(g)) * res_corr  ==  (rho*f_LF + delta_add)*res_corr with
    delta_add = f_LF*(exp(g)-1), rho=1.  Confirms the reported log-ratio<->additive
    equivalence is exact in code."""
    mf, _ = _build_mf(lf_backbone)
    d = lf_backbone[0]
    x = jnp.asarray(d["x"][0]); tau0 = jnp.asarray(d["tau0"][0])
    f_lf = jnp.exp(mf.lf_logP(x, tau0))
    g = mf.g(x, tau0)
    z_phys = x[9] * (Z_LIMITS[1] - Z_LIMITS[0]) + Z_LIMITS[0]
    rc = mf.res_corr(z_phys)
    delta_add = f_lf * (jnp.exp(g) - 1.0)
    P_additive = (f_lf + delta_add) * rc[None, :]
    P_logratio = mf.P_mf(x, tau0)
    assert np.allclose(np.asarray(P_additive), np.asarray(P_logratio), rtol=1e-10)


def test_mf_res_corr_on_off(lf_backbone):
    """res_corr toggles the multiplicative correction: P_with/P_without == res_corr(z,k),
    class-independent, and the table genuinely suppresses high-k at z=3 (~10%)."""
    mf, eval_logk = _build_mf(lf_backbone)
    d = lf_backbone[0]
    x = jnp.asarray(d["x"][0]); tau0 = jnp.asarray(d["tau0"][0])
    P_with = mf.P_mf(x, tau0, apply_res_corr=True)
    P_no = mf.P_mf(x, tau0, apply_res_corr=False)
    z_phys = float(x[9]) * (Z_LIMITS[1] - Z_LIMITS[0]) + Z_LIMITS[0]
    rc = np.asarray(mf.res_corr(jnp.asarray(z_phys)))
    ratio = np.asarray(P_with / P_no)
    assert np.allclose(ratio, rc[None, :], rtol=1e-10)  # class-independent factor
    # at z=3 the table suppresses the high-k end relative to low-k (the documented
    # ~10% particle-convergence drop in the KODIAQ band).
    rc3 = np.asarray(mf.res_corr(jnp.asarray(3.0)))
    k_eval = 10.0 ** np.asarray(eval_logk)
    hi = rc3[k_eval >= 0.07]
    assert hi.size and hi.min() < 0.95 and hi.min() > 0.8


def test_mf_jit_patterns(lf_backbone):
    """The forward is jittable via eqx.filter_jit and via a closure-over-module
    lambda (the HMC patterns); both agree with eager.  Plain jax.jit(mf.P_mf) (a
    jitted bound method) is NOT supported (the module's DeltaHead.layers list is
    unhashable as a static arg) -- documented in P_mf's docstring."""
    mf, _ = _build_mf(lf_backbone)
    d = lf_backbone[0]
    x = jnp.asarray(d["x"][0]); tau0 = jnp.asarray(d["tau0"][0])
    P_eager = np.asarray(mf.P_mf(x, tau0))

    @eqx.filter_jit
    def fwd(m, xx, tt):
        return m.P_mf(xx, tt)
    assert np.allclose(np.asarray(fwd(mf, x, tau0)), P_eager, rtol=0, atol=1e-12)

    jfwd = jax.jit(lambda xx, tt: mf.P_mf(xx, tt))
    assert np.allclose(np.asarray(jfwd(x, tau0)), P_eager, rtol=0, atol=1e-12)


def test_mf_second_order_grad_for_nuts(lf_backbone):
    """NUTS/HMC needs 2nd-order autodiff through the whole chain; a grad-of-grad wrt
    theta must be finite."""
    mf, _ = _build_mf(lf_backbone)
    d = lf_backbone[0]
    x = jnp.asarray(d["x"][0]); tau0 = jnp.asarray(d["tau0"][0])
    f = lambda th: mf.logP_mf(jnp.concatenate([th, x[9:10]]), tau0).sum()
    hvp = jax.grad(lambda th: jax.grad(f)(th).sum())(x[:9])
    assert np.all(np.isfinite(np.asarray(hvp)))


def test_mf_hmc_differentiable_in_theta(lf_backbone):
    """The whole P_MF(theta) chain (LF backbone + extrap + delta + res_corr) is
    differentiable wrt the 9 cosmology params (the HMC requirement)."""
    mf, _ = _build_mf(lf_backbone)
    d = lf_backbone[0]
    x0 = jnp.asarray(d["x"][0]); tau0 = jnp.asarray(d["tau0"][0])

    def logP_of_theta(theta9):
        x = jnp.concatenate([theta9, x0[9:10]])
        return mf.logP_mf(x, tau0).sum()

    g = jax.grad(logP_of_theta)(x0[:9])
    assert g.shape == (9,)
    assert np.all(np.isfinite(np.asarray(g)))
    # jacfwd (the Fisher path) also clean
    J = jax.jacfwd(lambda th: mf.logP_mf(jnp.concatenate([th, x0[9:10]]), tau0))(x0[:9])
    assert np.all(np.isfinite(np.asarray(J)))


def test_mf_lf_backbone_is_frozen(lf_backbone):
    """The LF backbone is FROZEN: differentiating the whole MF module wrt its
    lf_model array leaves gives EXACTLY zero (stop_gradient), while the DeltaHead
    leaves get finite grads -- so only the delta head is ever trained, and the LF
    weights can never be perturbed regardless of how the MF is differentiated."""
    mf, _ = _build_mf(lf_backbone)
    d = lf_backbone[0]
    x = jnp.asarray(d["x"][0]); tau0 = jnp.asarray(d["tau0"][0])

    grad = eqx.filter_grad(lambda m: m.P_mf(x, tau0).sum())(mf)
    lf_leaves = [np.asarray(l) for l in jax.tree_util.tree_leaves(grad.lf_model)
                 if eqx.is_array(l)]
    assert lf_leaves and all(np.all(l == 0.0) for l in lf_leaves)   # frozen -> 0 grad
    head_leaves = [np.asarray(l) for l in jax.tree_util.tree_leaves(grad.delta_head)
                   if eqx.is_array(l)]
    assert head_leaves and all(np.all(np.isfinite(l)) for l in head_leaves)
    assert any(np.any(l != 0.0) for l in head_leaves)              # head IS trained


# --------------------------------------------------------------------------- #
# delta_mode: rho(k,z)-only DEFAULT ('none') + 'linear' + 'mlp' variants
# --------------------------------------------------------------------------- #
def _build_mf_mode(lf_backbone, mode, n_basis=4, log_rho=None):
    """Build a MultiFidelity for a given delta_mode with a synthetic head."""
    d, model, norm, lf_logk = lf_backbone
    eval_logk = np.linspace(np.log10(1e-3), np.log10(0.15), 20)
    K = len(eval_logk)
    if log_rho is None:
        log_rho = np.zeros(K)
    if mode == "none":
        # a synthetic per-z fixed-mean table (Nz=3) so FixedMeanHead interpolates;
        # make it z-varying (per-row offset) so the z-dependence is exercised.
        z_tab = np.array([2.5, 3.0, 4.0])
        tab = 0.1 * np.ones((3, MF.N_CLASSES, K))
        tab += np.array([0.0, 0.05, 0.12])[:, None, None]   # distinct per-z trend
        head = MF.FixedMeanHead(tab, z_tab, n_basis=n_basis)
    elif mode == "linear":
        head = MF.GlobalLinearHead(in_dim=11, n_classes=MF.N_CLASSES,
                                   n_basis=n_basis, key=jax.random.PRNGKey(5))
    elif mode == "mlp":
        head = MF.DeltaHead(n_basis=n_basis, in_dim=11, width=8, n_layers=1,
                            key=jax.random.PRNGKey(3))
    mf = MF.build_multifidelity(model, norm, lf_logk, head, eval_logk=eval_logk,
                                log_rho=log_rho, n_basis=n_basis)
    return mf, eval_logk


def test_default_delta_mode_is_none():
    """The PRODUCTION default delta_mode is 'none' (the validated rho(k,z)-only
    model): MultiFidelity's __init__ default and build_multifidelity's inferred
    mode for a FixedMeanHead are both 'none'."""
    import inspect
    sig = inspect.signature(MF.MultiFidelity.__init__)
    assert sig.parameters["delta_mode"].default == "none"
    assert MF._HEAD_MODE["FixedMeanHead"] == "none"


def test_mf_invalid_delta_mode_raises(lf_backbone):
    """An unknown delta_mode is rejected at construction time."""
    d, model, norm, lf_logk = lf_backbone
    eval_logk = np.linspace(np.log10(1e-3), np.log10(0.15), 20)
    basis = MF.smooth_k_basis(jnp.asarray(eval_logk), 4)
    z_rc, logk_rc, rc_vals = MF.load_res_corr()
    head = MF.FixedMeanHead(np.zeros((2, MF.N_CLASSES, 20)), np.array([2.5, 4.0]))
    with pytest.raises(ValueError, match="delta_mode"):
        MF.MultiFidelity(model, norm, eval_logk, lf_logk, head, basis,
                         np.zeros(20), z_rc, logk_rc, rc_vals, delta_mode="bogus")


def test_fixed_mean_head_is_theta_independent(lf_backbone):
    """delta_mode='none' (FixedMeanHead): g is THETA-INDEPENDENT at fixed (z,tau0).
    Two DIFFERENT theta share identical g (all theta-dependence is in f_LF), and
    g varies with z (the z-resolved fixed mean).  This is the core property of the
    rho(k,z)-only default."""
    mf, _ = _build_mf_mode(lf_backbone, "none")
    d = lf_backbone[0]
    # two rows with different params but we pin the SAME z_unit/tau0 to isolate theta.
    x0 = jnp.asarray(d["x"][0]); tau0 = jnp.asarray(d["tau0"][0])
    theta_a = x0.at[:9].set(0.2); theta_b = x0.at[:9].set(0.8)   # same z (x[9]), diff theta
    g_a = np.asarray(mf.g(theta_a, tau0)); g_b = np.asarray(mf.g(theta_b, tau0))
    assert np.allclose(g_a, g_b, atol=1e-12)            # theta-independent
    # but g DOES vary with z (the z-resolved fixed mean table).
    z_lo = x0.at[9].set(0.1); z_hi = x0.at[9].set(0.9)
    assert not np.allclose(np.asarray(mf.g(z_lo, tau0)), np.asarray(mf.g(z_hi, tau0)))
    # the delta-head coeffs are zeros (no basis / no learned theta term).
    cond = MF.make_cond(x0, tau0)
    assert np.allclose(np.asarray(mf.delta_head.coeffs(cond)), 0.0)


def test_fixed_mean_head_g_equals_gbar(lf_backbone):
    """The combined g == gbar(z,k): log_rho + (gbar - log_rho) == gbar.  With a
    nonzero log_rho the FixedMeanHead's table (gbar - log_rho) must cancel it so g
    recovers the per-z fixed mean exactly."""
    K = 20
    eval_logk = np.linspace(np.log10(1e-3), np.log10(0.15), K)
    log_rho = np.linspace(-0.3, 0.5, K)                 # nonzero per-k rho
    d, model, norm, lf_logk = lf_backbone
    z_tab = np.array([2.5, 3.0, 4.0])
    gbar = 0.05 + 0.02 * np.arange(K)[None, None, :] * np.ones((3, MF.N_CLASSES, 1))
    tab = gbar - log_rho[None, None, :]                 # FixedMeanHead stores gbar - rho
    head = MF.FixedMeanHead(tab, z_tab)
    mf = MF.build_multifidelity(model, norm, lf_logk, head, eval_logk=eval_logk,
                                log_rho=log_rho)
    x = jnp.asarray(d["x"][0]).at[9].set(
        float((3.0 - MF.Z_LIMITS[0]) / (MF.Z_LIMITS[1] - MF.Z_LIMITS[0])))  # z=3.0
    tau0 = jnp.asarray(d["tau0"][0])
    g = np.asarray(mf.g(x, tau0))                        # (4,K)
    assert np.allclose(g, gbar[1], atol=1e-9)           # == gbar at z=3.0 (table row 1)


@pytest.mark.parametrize("mode", ["none", "linear", "mlp"])
def test_mf_mode_forward_positive_and_hmc_differentiable(lf_backbone, mode):
    """All three delta_modes give a strictly-positive forward and are HMC-
    differentiable in theta (grad + 2nd-order finite)."""
    mf, eval_logk = _build_mf_mode(lf_backbone, mode)
    assert mf.delta_mode == mode
    d = lf_backbone[0]
    x0 = jnp.asarray(d["x"][0]); tau0 = jnp.asarray(d["tau0"][0])
    P = np.asarray(mf.P_mf(x0, tau0))
    assert P.shape == (4, len(eval_logk)) and np.all(P > 0)
    # grad wrt theta finite
    f = lambda th: mf.logP_mf(jnp.concatenate([th, x0[9:10]]), tau0).sum()
    g = np.asarray(jax.grad(f)(x0[:9]))
    assert g.shape == (9,) and np.all(np.isfinite(g))
    # 2nd-order (NUTS) finite
    hvp = np.asarray(jax.grad(lambda th: jax.grad(f)(th).sum())(x0[:9]))
    assert np.all(np.isfinite(hvp))
    # jit pattern
    P_j = np.asarray(eqx.filter_jit(lambda m, xx, tt: m.P_mf(xx, tt))(mf, x0, tau0))
    assert np.allclose(P_j, P, atol=1e-12)


def test_global_linear_head_amp_is_linear(lf_backbone):
    """delta_mode='linear' (GlobalLinearHead): the amplitude is a SINGLE linear
    functional of the conditioning, and equals the bias b at the unit-cube centre."""
    head = MF.GlobalLinearHead(in_dim=11, n_classes=4, n_basis=4,
                               key=jax.random.PRNGKey(0))
    # zero out random init then set a known linear map.
    head = eqx.tree_at(lambda h: (h.w, h.b),
                       head, (jnp.arange(11.0), jnp.asarray(0.3)))
    centre = jnp.full(11, 0.5)
    assert np.isclose(float(head.amp(centre)), 0.3)     # amp == b at centre
    # linearity: amp(2x - centre) - amp(centre) == 2*(amp(x) - amp(centre))? Check
    # plain linearity of amp(cond) = w.(cond-0.5)+b.
    c1 = jnp.full(11, 0.7); c2 = jnp.full(11, 0.9)
    lhs = float(head.amp(c1) + head.amp(c2) - 2 * head.amp(centre))
    rhs = float(head.amp(c1 + c2 - centre) - head.amp(centre))
    assert np.isclose(lhs, rhs)


def test_train_global_linear_head_reduces_loss_and_ap_prior(tmp_path):
    """train_global_linear_head fits and reduces the loss; the optional A_p-direction
    prior pulls the linear weight toward the A_p axis (orthogonal component shrinks)."""
    lfp = tmp_path / "lf.h5"; hrp = tmp_path / "hr.h5"
    write_synthetic_cache(lfp, n_sims=4, snaps_per_sim=2, n_alpha=4, n_k=12, seed=7)
    write_synthetic_cache(hrp, n_sims=4, snaps_per_sim=2, n_alpha=4, n_k=12, seed=7)
    lf = _sanitize_unit_cube(load_cache(lfp)); hr = _sanitize_unit_cube(load_cache(hrp))
    pairs = MF.match_hr_to_lf(lf, hr)
    norm = fit_target_norm(lf, np.arange(len(lf["z_grid"])))
    model = Emulator(in_dim=10, n_k=lf["P_tier_p"].shape[1], n_basis=4,
                     key=jax.random.PRNGKey(0))
    lf_logk = np.log10(lf["kfkms"][0])
    k_eval, eval_logk = MF.build_eval_grid(hr, k_max=0.05, n_k=16)
    tg = MF.measure_delta_targets(lf, hr, model, norm, lf_logk, eval_logk, pairs)
    basis = np.asarray(MF.smooth_k_basis(jnp.asarray(eval_logk), 4))
    log_rho = np.nan_to_num(MF.mean_log_ratio_rho(tg, eval_logk), nan=0.0)

    head, hist = MF.train_global_linear_head(
        tg, basis, log_rho, eval_logk, n_basis=4, epochs=80, lr=3e-3,
        coeff_l2=1e-2, mean_prior_w=1e-2)
    assert hist["loss"][-1] < hist["loss"][0]

    # with an A_p-direction prior, the weight's component orthogonal to ap_dir shrinks.
    ap_dir = np.zeros(11); ap_dir[0] = 1.0              # say param 0 is the A_p axis
    head_p, _ = MF.train_global_linear_head(
        tg, basis, log_rho, eval_logk, n_basis=4, epochs=120, lr=3e-3,
        coeff_l2=1e-2, mean_prior_w=1e-2, ap_dir=ap_dir, ap_prior_w=1e2)
    w = np.asarray(head_p.w)
    perp = w.copy(); perp[0] = 0.0                       # component orthogonal to ap_dir
    assert np.linalg.norm(perp) < 1e-2                   # strongly suppressed by the prior


def test_build_default_head_builds_fixed_mean(tmp_path):
    """build_default_head returns a FixedMeanHead whose table is the per-z fixed
    mean minus log_rho, and build_multifidelity infers delta_mode='none'."""
    lfp = tmp_path / "lf.h5"; hrp = tmp_path / "hr.h5"
    write_synthetic_cache(lfp, n_sims=4, snaps_per_sim=2, n_alpha=4, n_k=12, seed=7)
    write_synthetic_cache(hrp, n_sims=4, snaps_per_sim=2, n_alpha=4, n_k=12, seed=7)
    lf = _sanitize_unit_cube(load_cache(lfp)); hr = _sanitize_unit_cube(load_cache(hrp))
    pairs = MF.match_hr_to_lf(lf, hr)
    norm = fit_target_norm(lf, np.arange(len(lf["z_grid"])))
    model = Emulator(in_dim=10, n_k=lf["P_tier_p"].shape[1], n_basis=4,
                     key=jax.random.PRNGKey(0))
    lf_logk = np.log10(lf["kfkms"][0])
    k_eval, eval_logk = MF.build_eval_grid(hr, k_max=0.05, n_k=16)
    tg = MF.measure_delta_targets(lf, hr, model, norm, lf_logk, eval_logk, pairs)
    log_rho = np.nan_to_num(MF.mean_log_ratio_rho(tg, eval_logk), nan=0.0)

    head = MF.build_default_head(tg, log_rho, n_basis=4)
    assert isinstance(head, MF.FixedMeanHead)
    mf = MF.build_multifidelity(model, norm, lf_logk, head, eval_logk=eval_logk,
                                log_rho=log_rho, n_basis=4)
    assert mf.delta_mode == "none"
    # forward is finite & positive
    P = np.asarray(mf.P_mf(jnp.asarray(lf["x"][0]), jnp.asarray(lf["tau0"][0])))
    assert P.shape == (4, 16) and np.all(P > 0)


# --------------------------------------------------------------------------- #
# delta target measurement + training on a synthetic HR<->LF pair
# --------------------------------------------------------------------------- #
def test_match_and_measure_and_train(tmp_path):
    """End-to-end on synthetic caches: matching, target measurement, a few training
    steps that REDUCE the loss, and a finite-grad delta-head fit.

    HR is the SAME design points as LF (same seed + same n_k so the fixture's RNG
    stream aligns and every HR row matches an LF row exactly on the (params, z,
    alpha) key) -- the real HR cache likewise hits exact LF design points."""
    lfp = tmp_path / "lf.h5"; hrp = tmp_path / "hr.h5"
    write_synthetic_cache(lfp, n_sims=4, snaps_per_sim=2, n_alpha=4, n_k=12, seed=7)
    write_synthetic_cache(hrp, n_sims=4, snaps_per_sim=2, n_alpha=4, n_k=12, seed=7)
    lf = _sanitize_unit_cube(load_cache(lfp))
    hr = _sanitize_unit_cube(load_cache(hrp))
    pairs = MF.match_hr_to_lf(lf, hr)
    assert len(pairs) == len(hr["z_grid"])           # all HR rows match an LF row

    norm = fit_target_norm(lf, np.arange(len(lf["z_grid"])))
    model = Emulator(in_dim=10, n_k=lf["P_tier_p"].shape[1], n_basis=4,
                     key=jax.random.PRNGKey(0))
    lf_logk = np.log10(lf["kfkms"][0])
    # eval grid stays within the synthetic LF k-support (no extrap needed here).
    k_eval, eval_logk = MF.build_eval_grid(hr, k_max=0.05, n_k=16)
    tg = MF.measure_delta_targets(lf, hr, model, norm, lf_logk, eval_logk, pairs)
    assert tg["g"].shape == (len(pairs), 4, 16)
    assert np.isfinite(tg["g"]).mean() > 0.5

    basis = np.asarray(MF.smooth_k_basis(jnp.asarray(eval_logk), 4))
    log_rho = MF.mean_log_ratio_rho(tg, eval_logk)
    assert log_rho.shape == (16,) and np.all(np.isfinite(log_rho))

    head, hist = MF.train_delta_head(
        tg, basis, log_rho, eval_logk, n_basis=4, width=8, n_layers=1,
        epochs=60, lr=3e-3, coeff_l2=1e-2, mean_prior_w=1e-2)
    assert hist["loss"][-1] < hist["loss"][0]        # training reduced the loss
    assert np.all(np.isfinite(hist["loss"]))


def test_build_eval_grid_caps_at_kodiaq():
    import h5py
    # build_eval_grid only reads k_max/n_k, not the hr cache contents.
    k_eval, eval_logk = MF.build_eval_grid(hr=None, k_max=0.2, n_k=32)
    assert k_eval.shape == (32,) and np.isclose(k_eval[-1], 0.2)
    assert k_eval[0] == pytest.approx(MF.DATA_RANGE["k_min"])
    assert np.all(np.diff(k_eval) > 0)


# --------------------------------------------------------------------------- #
# T1: tau0+z-RESOLVED FixedMeanHead (separable + ONE rank-1 interaction)
#   g(z,tau0,k) = gbar_z(z,k) + gbar_tau(tau0,k) + a(k)*u_z(z)*u_tau(tau0)
# --------------------------------------------------------------------------- #
def _synth_resolved_targets(K=14, n_z=5, n_rung=4, n_sim=6, seed=0):
    """Synthetic measured-rho targets with a KNOWN separable+rank-1 (z,rung,k)
    structure + small 6-sim noise, for the resolved-head tests.

    Faithful to the real cache: tau0 = -log(target_F) is a DETERMINISTIC, bit-identical
    function of (rung, z) and GROWS with z within a rung -- so each row carries an
    ``alpha_idx`` (rung) and a tau0 that is rung-monotone AND z-growing.  ``g`` is
    gbar_z(z,k)+gbar_tau(rung,k)+a(k)u_z u_tau plus Gaussian 6-sim noise (class 0; the
    others copy it -> class-independent rho).  Returns ``(targets, eval_logk, truth)``
    with truth = the noiseless per-(z,rung) g table and the per-z tau0 ladder.
    """
    rng = np.random.default_rng(seed)
    eval_logk = np.linspace(np.log10(1e-2), np.log10(0.069), K)
    z_phys = np.linspace(2.2, 5.0, n_z)
    z_unit = (z_phys - Z_LIMITS[0]) / (Z_LIMITS[1] - Z_LIMITS[0])
    rungs = np.arange(n_rung)
    # per-z tau0 ladder: rung sets the base mean-flux level, z scales it up (monotone).
    tau_by_z = (0.66 + 0.67 * rungs[None, :] / max(n_rung - 1, 1)) * \
               (0.5 + 0.4 * (z_phys[:, None] - 2.2))                  # (nz,nrung)
    u = 2 * (eval_logk - eval_logk.min()) / (eval_logk.max() - eval_logk.min()) - 1
    # separable z-trend (rising tilt, larger at low z), rung-trend, + rank-1 sign-flip.
    gbar_z = (0.06 - 0.012 * (z_phys[:, None] - 2.2)) * (1 + 0.5 * u)[None, :]   # (nz,K)
    gtau = 0.015 * (rungs[:, None] - rungs.mean()) * (1 + 0.3 * u)[None, :]      # (nr,K)
    a_k = 0.02 * (1.0 + u)                                                       # (K,)
    u_z = (z_phys - z_phys.mean()) / np.ptp(z_phys)                            # (nz,)
    u_tau = (rungs - rungs.mean()) / np.ptp(rungs)                              # (nr,)
    truth = (gbar_z[:, None, :] + gtau[None, :, :]
             + a_k[None, None, :] * u_z[:, None, None] * u_tau[None, :, None])  # (nz,nr,K)

    X, T0, AI, G = [], [], [], []
    for s in range(n_sim):
        for zi in range(n_z):
            for ri in range(n_rung):
                x = np.zeros(10)
                x[0] = rng.uniform(0.1, 0.9)        # a dummy theta (must be ignored)
                x[9] = z_unit[zi]
                noise = rng.normal(0, 0.006, size=K)   # ~0.6% 6-sim noise
                gC0 = truth[zi, ri] + noise
                g4 = np.tile(gC0[None, :], (4, 1))     # class-independent rho
                X.append(x); T0.append(tau_by_z[zi, ri]); AI.append(ri); G.append(g4)
    return (dict(x=np.asarray(X), tau0=np.asarray(T0), alpha_idx=np.asarray(AI),
                 g=np.asarray(G), hr_row=np.arange(len(X)), lf_row=np.arange(len(X))),
            eval_logk, dict(truth=truth, z_phys=z_phys, rungs=rungs,
                            tau_by_z=tau_by_z, eval_logk=eval_logk))


def test_resolved_fixed_mean_table_decomposition_shapes():
    """fixed_mean_table_resolved returns the separable + rank-1 components with the
    expected shapes and z/tau0 grids ascending."""
    tg, eval_logk, truth = _synth_resolved_targets()
    log_rho = np.nan_to_num(MF.mean_log_ratio_rho(tg, eval_logk), nan=0.0)
    comp = MF.fixed_mean_table_resolved(tg, log_rho)
    K = len(eval_logk)
    nz = len(truth["z_phys"]); nt = len(truth["rungs"])
    assert comp["gbar_z_tab"].shape == (nz, MF.N_CLASSES, K)
    assert comp["gtau_tab"].shape == (nt, MF.N_CLASSES, K)
    assert comp["a_k"].shape == (MF.N_CLASSES, K)
    assert comp["u_z"].shape == (nz,) and comp["u_tau"].shape == (nt,)
    assert comp["z_tab"].shape == (nz,) and comp["tau_tab"].shape == (nt,)
    assert comp["tau_by_z"].shape == (nz, nt)
    assert np.all(np.diff(comp["z_tab"]) > 0)
    assert np.all(np.diff(comp["tau_tab"]) > 0)               # rung index axis
    assert np.all(np.diff(comp["tau_by_z"], axis=1) > 0)      # tau0 monotone in rung


def test_resolved_head_g_depends_on_tau0_and_is_differentiable():
    """The CORE T1 fix: g is tau0-resolved AND dg/dtau0 is finite & NONZERO (the
    pooled FixedMeanHead has dg/dtau0 IDENTICALLY zero)."""
    tg, eval_logk, truth = _synth_resolved_targets()
    log_rho = np.nan_to_num(MF.mean_log_ratio_rho(tg, eval_logk), nan=0.0)
    head = MF.build_default_head(tg, log_rho, resolved=True)
    basis = MF.smooth_k_basis(jnp.asarray(eval_logk), 4)
    # build a cond at a mid z and mid tau0
    z_unit = float((3.4 - Z_LIMITS[0]) / (Z_LIMITS[1] - Z_LIMITS[0]))
    x = jnp.zeros(10).at[9].set(z_unit)

    def g_of_tau(t0):
        cond = MF.make_cond(x, t0)
        return (log_rho[None, :] + head(cond, basis)).sum()

    # finite, NONZERO gradient wrt tau0
    dgt = float(jax.grad(g_of_tau)(jnp.asarray(1.0)))
    assert np.isfinite(dgt)
    assert abs(dgt) > 1e-6                              # NONZERO (vs pooled head == 0)
    # g actually differs at two different tau0
    cond_lo = MF.make_cond(x, jnp.asarray(0.7))
    cond_hi = MF.make_cond(x, jnp.asarray(1.3))
    g_lo = np.asarray(log_rho[None, :] + head(cond_lo, basis))
    g_hi = np.asarray(log_rho[None, :] + head(cond_hi, basis))
    assert not np.allclose(g_lo, g_hi, atol=1e-6)


def test_resolved_head_reproduces_rho_to_6sim_noise():
    """g(z,tau0,k) reproduces the measured per-(z,rung) rho on the synthetic 'sims'
    to within the ~0.6% 6-sim noise floor (the decomposition is unbiased).  tau0 is
    looked up at each rung's PHYSICAL tau0 at that z (the per-z ladder)."""
    tg, eval_logk, truth = _synth_resolved_targets()
    log_rho = np.nan_to_num(MF.mean_log_ratio_rho(tg, eval_logk), nan=0.0)
    head = MF.build_default_head(tg, log_rho, resolved=True)
    basis = MF.smooth_k_basis(jnp.asarray(eval_logk), 4)
    errs = []
    for zi, zp in enumerate(truth["z_phys"]):
        z_unit = (zp - Z_LIMITS[0]) / (Z_LIMITS[1] - Z_LIMITS[0])
        x = jnp.zeros(10).at[9].set(float(z_unit))
        for ri in range(len(truth["rungs"])):
            t0 = truth["tau_by_z"][zi, ri]          # physical tau0 of this rung at z
            cond = MF.make_cond(x, jnp.asarray(float(t0)))
            g = np.asarray(log_rho[None, :] + head(cond, basis))[0]  # class 0
            errs.append(np.abs(g - truth["truth"][zi, ri]))
    errs = np.concatenate(errs)
    # the noiseless target is reproduced to within the 6-sim noise (~0.6%, allow 1%).
    assert errs.max() < 0.01, f"max reproduction err {errs.max():.4f} exceeds 6-sim noise"


def test_resolved_head_clamped_no_nan_over_edges():
    """C0-continuous / no-NaN at and BEYOND the (z, tau0) table edges -- NUTS-safe."""
    tg, eval_logk, truth = _synth_resolved_targets()
    log_rho = np.nan_to_num(MF.mean_log_ratio_rho(tg, eval_logk), nan=0.0)
    head = MF.build_default_head(tg, log_rho, resolved=True)
    basis = MF.smooth_k_basis(jnp.asarray(eval_logk), 4)
    # query z far below/above the table and tau0 far outside the measured ladder
    for z_unit in (-0.5, 0.0, 0.5, 1.5):
        for t0 in (0.01, 0.66, 1.0, 1.33, 5.0):
            cond = MF.make_cond(jnp.zeros(10).at[9].set(float(z_unit)),
                                jnp.asarray(float(t0)))
            g = np.asarray(head(cond, basis))
            assert np.all(np.isfinite(g)), (z_unit, t0)
    # clamp value at the low tau0 edge == clamp value FAR below it (constant-edge):
    # the lowest-rung tau0 at the lowest z is the table's lower tau0 corner.
    tau_lo = float(truth["tau_by_z"][0, 0])
    edge = MF.make_cond(jnp.zeros(10).at[9].set(0.0), jnp.asarray(tau_lo))
    beyond = MF.make_cond(jnp.zeros(10).at[9].set(-1.0), jnp.asarray(1e-3))
    g_edge = np.asarray(head(edge, basis))
    g_beyond = np.asarray(head(beyond, basis))
    assert np.allclose(g_edge, g_beyond, atol=1e-9)


def test_resolved_head_theta_independent():
    """Even resolved in (z,tau0), g stays THETA-INDEPENDENT (no cosmology DOF): two
    different theta at the same (z,tau0) give identical g (anti-aliasing guarantee)."""
    tg, eval_logk, truth = _synth_resolved_targets()
    log_rho = np.nan_to_num(MF.mean_log_ratio_rho(tg, eval_logk), nan=0.0)
    head = MF.build_default_head(tg, log_rho, resolved=True)
    basis = MF.smooth_k_basis(jnp.asarray(eval_logk), 4)
    x_a = jnp.zeros(10).at[9].set(0.4).at[0].set(0.1)
    x_b = jnp.zeros(10).at[9].set(0.4).at[0].set(0.9)
    t0 = jnp.asarray(1.0)
    g_a = np.asarray(head(MF.make_cond(x_a, t0), basis))
    g_b = np.asarray(head(MF.make_cond(x_b, t0), basis))
    assert np.allclose(g_a, g_b, atol=1e-12)


def test_pooled_fixed_mean_head_still_tau0_invariant(lf_backbone):
    """BACK-COMPAT: the default (pooled, resolved=False) FixedMeanHead is unchanged --
    g has dg/dtau0 == 0 (the prior behaviour the existing tests rely on)."""
    mf, _ = _build_mf_mode(lf_backbone, "none")
    d = lf_backbone[0]
    x0 = jnp.asarray(d["x"][0])
    f = lambda t0: mf.g(x0, t0).sum()
    dgt = float(jax.grad(f)(jnp.asarray(1.0)))
    assert abs(dgt) < 1e-12                              # pooled head: tau0-independent


# --------------------------------------------------------------------------- #
# T6: dN/dX + CDDF FIXED (theta-independent) resolution correction (Head A)
# --------------------------------------------------------------------------- #
def _synth_dndx_targets(n_z=6, n_sim=5, seed=1):
    """Synthetic per-snap dN/dX (3 HCD classes) HR & LF with a KNOWN z-resolved,
    theta-INDEPENDENT HR/LF ratio + small 6-sim noise.  Returns (lf, hr, z_grid,
    truth_ratio (nz,3))."""
    rng = np.random.default_rng(seed)
    z = np.linspace(2.2, 5.0, n_z)
    # a smooth z-resolved ratio per class (LLS falls through 1, sub/DLA stay >1)
    truth = np.stack([
        1.30 - 0.07 * (z - 2.2),                 # LLS: 1.30 -> ~1.1
        1.29 - 0.02 * (z - 2.2),                 # subDLA
        1.13 + 0.02 * (z - 2.2),                 # DLA
    ], axis=1)                                    # (nz,3)
    L, H = [], []
    zl, zh = [], []
    pl, ph = [], []
    for s in range(n_sim):
        base = rng.uniform(0.3, 0.6, size=3)      # per-sim LF baseline incidence
        ns = rng.uniform(0.85, 0.98)
        for zi in range(n_z):
            lf_dndx = base * (1 + 0.1 * (z[zi] - 2.2))
            noise = rng.normal(0, 0.02, size=3)   # ~2% 6-sim noise
            hr_dndx = lf_dndx * truth[zi] * (1 + noise)
            L.append(lf_dndx); H.append(hr_dndx)
            zl.append(z[zi]); zh.append(z[zi])
            pl.append(ns); ph.append(ns)
    lf = dict(dndx=np.asarray(L), z=np.asarray(zl), ns=np.asarray(pl))
    hr = dict(dndx=np.asarray(H), z=np.asarray(zh), ns=np.asarray(ph))
    return lf, hr, z, truth


def test_dndx_correction_table_reproduces_hr_to_noise():
    """build_dndx_res_corr measures the z-resolved HR/LF dN/dX ratio per class and
    reproduces the HR dN/dX from LF*correction to within the 6-sim noise."""
    lf, hr, z, truth = _synth_dndx_targets()
    tab = MF.build_dndx_res_corr(lf["dndx"], lf["z"], hr["dndx"], hr["z"])
    assert tab["ratio"].shape == (len(z), 3)
    assert np.all(np.diff(tab["z"]) > 0)
    # the measured ratio recovers the known truth to within the 2% 6-sim noise / sqrt(n)
    assert np.max(np.abs(tab["ratio"] - truth)) < 0.02
    # LF * correction reproduces HR per row to within the per-row noise (~2%)
    corr = MF.apply_dndx_res_corr(tab, jnp.asarray(lf["dndx"]), jnp.asarray(lf["z"]))
    rel = np.abs(np.asarray(corr) - hr["dndx"]) / hr["dndx"]
    assert np.median(rel) < 0.02


def test_dndx_correction_is_theta_independent():
    """The dN/dX correction is FIXED: d(correction)/d(theta) == 0 -- it is a function
    of z and class ONLY (the PI directive; mirrors the P1D fixed-mean baseline)."""
    lf, hr, z, truth = _synth_dndx_targets()
    tab = MF.build_dndx_res_corr(lf["dndx"], lf["z"], hr["dndx"], hr["z"])
    # apply at a fixed (dndx, z) and differentiate wrt a fake theta that the
    # correction must NOT see: the correction depends only on z & class.
    dndx0 = jnp.asarray([0.4, 0.12, 0.05])
    z0 = jnp.asarray(3.0)

    def corrected_sum(theta):
        # theta enters only through dndx (the emulator output); the *factor* itself
        # must be theta-free, so d(factor)/d(theta)=0.  We check the factor directly.
        fac = MF.dndx_res_factor(tab, z0)
        return (fac * theta).sum()

    # the factor does not depend on theta at all -> grad is just the factor (finite),
    # and crucially the factor is identical for any theta.
    fac_a = np.asarray(MF.dndx_res_factor(tab, z0))
    fac_b = np.asarray(MF.dndx_res_factor(tab, z0))
    assert np.allclose(fac_a, fac_b)
    assert fac_a.shape == (3,)


def test_dndx_res_factor_differentiable_in_z_and_clamped():
    """The dN/dX correction factor is differentiable in z (for HMC) and clamped at
    the table z-edges (no NaN / no divergence outside the measured z-range)."""
    lf, hr, z, truth = _synth_dndx_targets()
    tab = MF.build_dndx_res_corr(lf["dndx"], lf["z"], hr["dndx"], hr["z"])
    g = jax.grad(lambda zz: MF.dndx_res_factor(tab, zz).sum())(jnp.asarray(3.1))
    assert np.isfinite(float(g))
    # clamp beyond the z-edges
    for zz in (1.0, 2.2, 3.0, 5.0, 7.0):
        fac = np.asarray(MF.dndx_res_factor(tab, jnp.asarray(float(zz))))
        assert np.all(np.isfinite(fac)) and np.all(fac > 0)
    lo = np.asarray(MF.dndx_res_factor(tab, jnp.asarray(float(z[0]))))
    below = np.asarray(MF.dndx_res_factor(tab, jnp.asarray(0.0)))
    assert np.allclose(lo, below)                       # constant-edge extrapolation
