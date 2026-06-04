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
    z, logk, rc = MF.load_res_corr()
    # evaluate exactly at a table z and its own k-grid -> recover the table row.
    zi = 5
    k_eval = jnp.asarray(10.0 ** logk[zi])
    got = np.asarray(MF.interp_res_corr(z, logk, rc, jnp.asarray(z[zi]), k_eval))
    assert np.allclose(got, rc[zi], rtol=1e-9, atol=1e-9)


def test_interp_res_corr_high_k_suppression_and_clamp():
    z, logk, rc = MF.load_res_corr()
    k = jnp.asarray(np.array([0.01, 0.1, 0.18]))
    val = np.asarray(MF.interp_res_corr(z, logk, rc, jnp.asarray(3.0), k))
    # high-k is suppressed relative to low-k (the ~10% particle-convergence drop)
    assert val[2] < val[0]
    assert np.all((val > 0.8) & (val < 1.1))
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
