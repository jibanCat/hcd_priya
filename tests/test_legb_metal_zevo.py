"""Model C — free 2-node flat-log f metal z-evolution + blind-safe metal_zevo injection arms (TDD).

Builds on A1 (commit 5e7f258). A NEW static ``metal_prior`` value ``"flatlog2node"`` gates ALL Model-C
behaviour; ``"uniform"`` and ``"flatlog"`` stay BYTE-EXACT (their goldens are NEVER regenerated).

Model C samples, PER metals_on leg IN ORDER, PER ion, TWO node values of the metal flux decrement f at
z-nodes ``ctx.metal_node_z`` ~ ``dist.LogUniform(metal_fnode_lo, metal_fnode_hi)``; SiII only on legs in
``ctx.metal_siII_legs``. Per z the amplitude is ``a(z) = f(z)/(1-<F>(z))`` with f(z) log-interp'd
(log10 f linear in log10(1+z) = power-law-exact) and ``<F>(z)=exp(-tau0_vec[iz])`` the SAMPLED mean flux.

These tests pin (TDD): the flatlog2node site inventory + order + constrain_fn mirror parity; the
log-interp; the per-z f->a map == metal_inject parity; the scalar back-compat byte-exactness; the ev_z
variance transform reuses the SAME per-z mfac; a flatlog2node forward golden; the metal_inject extensions
(a_*_direct / damp='ma' / form='ma2025'); the 3 injection arms (decreasing / increasing / Ma+2025,
eBOSS SiIII-only, DESI SiIII+SiII, de-double-count clean truth metal==1); downstream-reader robustness;
and a short flatlog2node NUTS smoke.

Env (MANDATORY):
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
    /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_legb_metal_zevo.py -q
"""
import os
import numpy as np
import pytest

import hcd_analysis.emulator  # x64 BEFORE jax
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer.util import constrain_fn, initialize_model

from hcd_analysis.emulator import closure_legb as C
from hcd_analysis.emulator import data_likelihood as DL
from hcd_analysis.emulator import meanflux_prior as MF

_CACHE = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/observables_tau0_lf.h5"
_CKPT0 = "/home/mfho/hcd_priya/checkpoints/final_fold0.eqx"
_DESI_NPZ = "/home/mfho/data/desi_dr1_p1d/desi_dr1_p1d.npz"
_EBOSS_NPZ = "/home/mfho/data/eboss_dr14_p1d/eboss_dr14_p1d.npz"
_have = all(os.path.exists(p) for p in (_CACHE, _CKPT0, _DESI_NPZ))
_have_eboss = _have and os.path.exists(_EBOSS_NPZ)


# --------------------------------------------------------------------------- #
#  Helpers.
# --------------------------------------------------------------------------- #
def _sites(model, *a, seed=1):
    tr = handlers.trace(handlers.seed(model, jax.random.PRNGKey(seed))).get_trace(*a)
    return [k for k, v in tr.items() if v["type"] == "sample" and not v.get("is_observed")]


def _trace(model, *a, seed=1):
    return handlers.trace(handlers.seed(model, jax.random.PRNGKey(seed))).get_trace(*a)


def _ctx_2node(with_eboss=False, metals_on=True, desi_z_hi=2.6):
    """A Leg-B ctx in the flatlog2node Model-C mode. DESI is restricted to a narrow z so the
    forward is light; ``with_eboss`` adds the SiIII-only eBOSS leg (z up to 4.6 > the z=4.2 node)."""
    ctx, d = C.build_legb_ctx(
        use_xclass=True, sample_metals=True, metals_on=metals_on, with_eboss=with_eboss,
        metal_prior="flatlog2node", desi_kwargs=dict(z_lo=0.0, z_hi=desi_z_hi))
    return ctx, d


def _core(ctx, d):
    return {k: np.nanmean(np.asarray(v), axis=0)
            for k, v in C._fiducial_dla_core_per_leg(d, ctx.legs, ctx.cache_k).items()}


def _desi_leg(ctx):
    return next(l for l in ctx.legs if l.name == "DESI")


# =========================================================================== #
#  1. flatlog2node site inventory + order (with_eboss=True).
# =========================================================================== #
@pytest.mark.skipif(not _have_eboss, reason="real cache/ckpt/DESI/eBOSS not present")
def test_flatlog2node_site_inventory_and_order():
    """Per metals_on leg IN ORDER: f_SiIII_<leg>_z0/z1 (always) + f_SiII_<leg>_z0/z1 (only legs in
    metal_siII_legs). DESI -> SiIII+SiII, eBOSS -> SiIII-only, KS (metals_off) -> none. The metal
    block order is f_SiIII_DESI_z0,_z1, f_SiII_DESI_z0,_z1, f_SiIII_eBOSS_z0,_z1."""
    ctx, d = _ctx_2node(with_eboss=True, metals_on=True)
    assert ctx.metal_prior == "flatlog2node"
    assert ctx.metal_node_z == (2.2, 4.2)
    assert tuple(ctx.metal_siII_legs) == ("DESI",)
    core = _core(ctx, d)
    s = _sites(C._legb_model, ctx, ctx.legs, core)
    metal_sites = [n for n in s if n.startswith("f_SiIII") or n.startswith("f_SiII_")]
    assert metal_sites == [
        "f_SiIII_DESI_z0", "f_SiIII_DESI_z1", "f_SiII_DESI_z0", "f_SiII_DESI_z1",
        "f_SiIII_eBOSS_z0", "f_SiIII_eBOSS_z1"], f"metal site order wrong: {metal_sites}"
    # KS (metals_off) contributes no metal site; the legacy scalar sites are GONE under flatlog2node.
    assert "a_SiIII" not in s and "a_SiII" not in s, f"scalar metal sites present under flatlog2node: {s}"
    assert not any("KS" in n for n in metal_sites), "KS (metals_off) must contribute no metal site"
    # each metal site is dist.LogUniform(fnode_lo, fnode_hi).
    trm = _trace(C._legb_model, ctx, ctx.legs, core)
    for nm in metal_sites:
        fn = trm[nm]["fn"]
        assert isinstance(fn, dist.LogUniform), f"{nm} not LogUniform: {type(fn).__name__}"
        assert float(fn.low) == pytest.approx(float(ctx.metal_fnode_lo), rel=1e-12)
        assert float(fn.high) == pytest.approx(float(ctx.metal_fnode_hi), rel=1e-12)


# =========================================================================== #
#  2. constrain_fn MIRROR parity (the load-bearing invariant).
# =========================================================================== #
@pytest.mark.skipif(not _have_eboss, reason="real cache/ckpt/DESI/eBOSS not present")
def test_flatlog2node_constrain_fn_mirror_parity():
    """_legb_model and _legb_priors_only trace the IDENTICAL ordered sample-site list under
    flatlog2node (with_eboss=True), and constrain_fn round-trips a node into (fnode_lo, fnode_hi)."""
    ctx, d = _ctx_2node(with_eboss=True, metals_on=True)
    core = _core(ctx, d)
    s_model = _sites(C._legb_model, ctx, ctx.legs, core)
    s_prior = _sites(C._legb_priors_only, ctx)
    assert s_model == s_prior, f"flatlog2node mirror mismatch:\n model={s_model}\n prior={s_prior}"
    lo, hi = float(ctx.metal_fnode_lo), float(ctx.metal_fnode_hi)
    init = initialize_model(jax.random.PRNGKey(2), lambda: C._legb_priors_only(ctx))
    z0 = dict(init.param_info.z)
    assert "f_SiIII_DESI_z0" in z0 and "f_SiII_DESI_z0" in z0 and "f_SiIII_eBOSS_z0" in z0
    for u in (-20.0, 0.0, 20.0):
        z = dict(z0); z["f_SiIII_DESI_z0"] = jnp.asarray(u)
        c = constrain_fn(lambda: C._legb_priors_only(ctx), (), {}, z, return_deterministic=False)
        f = float(c["f_SiIII_DESI_z0"])
        assert lo < f < hi, f"constrain f_SiIII_DESI_z0 {f:.4g} not in ({lo},{hi})"


# =========================================================================== #
#  3. log-interp: log10 f(z) linear in log10(1+z) (power-law-exact) + clamp.
# =========================================================================== #
def test_log_interp_is_power_law_exact_and_clamps():
    """``_metal_f_of_z`` reproduces a single power-law f = f0*(1+z)^p to 1e-10 between the nodes
    (log10 f LINEAR in log10(1+z)), and CLAMPS to the node value outside [node_z[0], node_z[1]]."""
    node_z = (2.2, 4.2)
    f0, p = 7.3e-3, 1.41
    f_nodes = (f0 * (1.0 + node_z[0]) ** p, f0 * (1.0 + node_z[1]) ** p)
    for z in np.linspace(2.2, 4.2, 17):
        got = float(C._metal_f_of_z(z, node_z, f_nodes))
        exp = f0 * (1.0 + z) ** p
        assert got == pytest.approx(exp, rel=1e-10), f"z={z}: {got} != power-law {exp}"
    # CLAMP: below the low node and above the high node hold the node value (eBOSS z=4.6 -> f(4.2)).
    assert float(C._metal_f_of_z(1.5, node_z, f_nodes)) == pytest.approx(f_nodes[0], rel=1e-12)
    assert float(C._metal_f_of_z(4.6, node_z, f_nodes)) == pytest.approx(f_nodes[1], rel=1e-12)
    assert float(C._metal_f_of_z(5.4, node_z, f_nodes)) == pytest.approx(f_nodes[1], rel=1e-12)
    # at the geometric midpoint of log10(1+z) the value is sqrt(f0*f1) when nodes are equal-spaced.
    # (sanity: midpoint z s.t. log10(1+z)=mean -> f=sqrt(f_lo*f_hi) only for a generic power law it is
    # the log-linear interpolant, checked above; here just assert monotone for an increasing power law)
    zs = np.linspace(2.2, 4.2, 9)
    fs = np.array([float(C._metal_f_of_z(z, node_z, f_nodes)) for z in zs])
    assert np.all(np.diff(fs) > 0), "increasing power law must give monotone-increasing f(z)"


# =========================================================================== #
#  4. per-z f->a map == metal_inject parity (machine precision) — the closure self-consistency.
#     Also the NEW flatlog2node forward GOLDEN (pin P_model at a fixed f-node draw, rtol 1e-10).
# =========================================================================== #
@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI not present")
def test_per_z_map_equals_metal_inject_and_forward_golden():
    """predict_P_obs_on_leg with f-nodes equals, per z, the clean forward times metal_inject(form=
    'desi_full', f_SiII_SiII=0) applied with the SAME per-z f and mean_flux=exp(-tau0_vec[iz]) (the
    in-class arms-1/2 contaminant) -> machine precision. This pins the Model-C forward (the golden)."""
    ctx, _ = _ctx_2node(with_eboss=False, metals_on=True)
    leg = _desi_leg(ctx)
    tau0_vec = MF.becker13_tau0(jnp.asarray(leg.z))
    alpha3 = jnp.asarray([0.20, 0.08, 0.015])
    node_z = (2.2, 4.2)
    f3_nodes = jnp.asarray([0.012, 0.006])     # a FIXED, blind-safe f-node draw (decreasing-ish)
    f2_nodes = jnp.asarray([0.007, 0.004])
    common = dict(pf_stats=ctx.pf_stats, dla_core=jnp.mean(jnp.asarray(ctx.dla_core_leg[leg.name]), 0),
                  cache_k=ctx.cache_k, leg=leg, sigma_zb=None, alpha_centres=None)
    # Model-C forward with the f-nodes.
    P_nodes, _ = DL.predict_P_obs_on_leg(ctx.model, jnp.asarray(MF_THETA), tau0_vec, alpha3,
                                         f_SiIII_nodes=f3_nodes, f_SiII_nodes=f2_nodes,
                                         metal_node_z=node_z, **common)
    # the CLEAN forward (no metal).
    P_clean, _ = DL.predict_P_obs_on_leg(ctx.model, jnp.asarray(MF_THETA), tau0_vec, alpha3,
                                         a_SiIII=0.0, a_SiII=0.0, **common)
    P_nodes = np.asarray(P_nodes); P_clean = np.asarray(P_clean)
    k = np.asarray(leg.k); z_idx = np.asarray(leg.z_idx)
    expect = P_clean.copy()
    for iz in range(leg.n_z):
        rows = np.where(z_idx == iz)[0]
        if rows.size == 0:
            continue
        f3_z = float(C._metal_f_of_z(float(leg.z[iz]), node_z, np.asarray(f3_nodes)))
        f2_z = float(C._metal_f_of_z(float(leg.z[iz]), node_z, np.asarray(f2_nodes)))
        Fbar = float(np.exp(-np.asarray(tau0_vec)[iz]))
        # REGENERATED Model C+ golden: the forward now (a) includes the SiIII-SiII cross (cross=True)
        # and (b) uses the scalar decorrelation k_SiIII=k_SiII=0.05 here (no k-nodes passed).
        expect[rows] = C.metal_inject(P_clean[rows], k[rows], Fbar, form="desi_full",
                                      f_SiIII=f3_z, f_SiII=f2_z, f_SiII_SiII=0.0,
                                      k_decorr=DL.K_SiIII_DEFAULT, k_SiII=DL.K_SiII_DEFAULT,
                                      r_doublet=DL.R_SiII_DOUBLET, cross=True)
    np.testing.assert_allclose(P_nodes, expect, rtol=1e-10, atol=0.0,
                               err_msg="Model C+ per-z f->a forward != metal_inject(cross=True) parity")
    # the metal factor is non-trivial (the nodes actually contaminate).
    assert not np.allclose(P_nodes, P_clean), "f-nodes did not contaminate P_model"


# =========================================================================== #
#  5. BACK-COMPAT: predict_P_obs_on_leg f_SiIII_nodes=None == the scalar path byte-exact.
# =========================================================================== #
@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI not present")
def test_predict_nodes_none_is_scalar_path_byte_exact():
    """f_SiIII_nodes=None (the default) reproduces the legacy scalar a_SiIII forward byte-for-byte."""
    ctx, _ = _ctx_2node(with_eboss=False, metals_on=True)
    leg = _desi_leg(ctx)
    tau0_vec = MF.becker13_tau0(jnp.asarray(leg.z))
    alpha3 = jnp.asarray([0.20, 0.08, 0.015])
    szb = ctx.sigma_zb_per_leg.get(leg.name)
    rzb = ctx.rho_zb_per_leg.get(leg.name) if ctx.rho_zb_per_leg is not None else None
    common = dict(pf_stats=ctx.pf_stats, dla_core=jnp.mean(jnp.asarray(ctx.dla_core_leg[leg.name]), 0),
                  cache_k=ctx.cache_k, leg=leg, sigma_zb=szb, alpha_centres=ctx.alpha_centres,
                  cemu_inflate=ctx.cemu_inflate, rho_zb=rzb)
    # scalar metal forward (a_SiIII != 0) with nodes=None (default).
    P0, C0 = DL.predict_P_obs_on_leg(ctx.model, jnp.asarray(MF_THETA), tau0_vec, alpha3,
                                     a_SiIII=0.03, a_SiII=0.01, **common)
    # passing metal_node_z explicitly but nodes=None must NOT change anything (the branch is gated).
    P1, C1 = DL.predict_P_obs_on_leg(ctx.model, jnp.asarray(MF_THETA), tau0_vec, alpha3,
                                     a_SiIII=0.03, a_SiII=0.01, f_SiIII_nodes=None,
                                     metal_node_z=(2.2, 4.2), **common)
    np.testing.assert_array_equal(np.asarray(P0), np.asarray(P1))
    np.testing.assert_array_equal(np.asarray(C0), np.asarray(C1))


# =========================================================================== #
#  6. ev_z (C_emu) variance transform uses the SAME per-z mfac as P_z.
# =========================================================================== #
@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI not present")
def test_cemu_variance_transform_uses_same_per_z_mfac():
    """The C_emu diagonal under Model C equals the clean C_emu diagonal scaled by the SAME per-row
    metal factor squared (no stale scalar a): emu_metal == emu_clean * (P_metal/P_clean)^2."""
    ctx, _ = _ctx_2node(with_eboss=False, metals_on=True)
    leg = _desi_leg(ctx)
    tau0_vec = MF.becker13_tau0(jnp.asarray(leg.z))
    alpha3 = jnp.asarray([0.20, 0.08, 0.015])
    szb = ctx.sigma_zb_per_leg.get(leg.name)
    rzb = ctx.rho_zb_per_leg.get(leg.name) if ctx.rho_zb_per_leg is not None else None
    common = dict(pf_stats=ctx.pf_stats, dla_core=jnp.mean(jnp.asarray(ctx.dla_core_leg[leg.name]), 0),
                  cache_k=ctx.cache_k, leg=leg, sigma_zb=szb, alpha_centres=ctx.alpha_centres,
                  cemu_inflate=ctx.cemu_inflate, rho_zb=rzb)
    f3_nodes = jnp.asarray([0.013, 0.007]); f2_nodes = jnp.asarray([0.008, 0.004])
    P_m, C_m = DL.predict_P_obs_on_leg(ctx.model, jnp.asarray(MF_THETA), tau0_vec, alpha3,
                                       f_SiIII_nodes=f3_nodes, f_SiII_nodes=f2_nodes,
                                       metal_node_z=(2.2, 4.2), **common)
    P_c, C_c = DL.predict_P_obs_on_leg(ctx.model, jnp.asarray(MF_THETA), tau0_vec, alpha3,
                                       a_SiIII=0.0, a_SiII=0.0, **common)
    Cd = np.asarray(leg.C_data)
    emu_m = np.diag(np.asarray(C_m)) - np.diag(Cd)        # C_total = C_data + diag(emu_var)
    emu_c = np.diag(np.asarray(C_c)) - np.diag(Cd)
    mfac = np.asarray(P_m) / np.asarray(P_c)              # the per-row metal factor used on P_z
    np.testing.assert_allclose(emu_m, emu_c * mfac ** 2, rtol=1e-9, atol=0.0,
                               err_msg="C_emu transform did not reuse the P_z per-z metal factor")


# =========================================================================== #
#  7. jnp.interp clamp through the FORWARD: eBOSS z>4.2 stays finite (no NaN/Inf).
# =========================================================================== #
@pytest.mark.skipif(not _have_eboss, reason="real cache/ckpt/DESI/eBOSS not present")
def test_eboss_high_z_clamp_forward_finite():
    """eBOSS reaches z=4.6 > the z=4.2 high node; jnp.interp default-clamps f flat beyond 4.2 so the
    Model-C forward stays finite (bounded a) on the extrapolated high-z bins."""
    ctx, _ = _ctx_2node(with_eboss=True, metals_on=True)
    leg = next(l for l in ctx.legs if l.name == "eBOSS")
    assert float(np.max(leg.z)) > 4.2, "expected an eBOSS z-bin above the z=4.2 node"
    tau0_vec = MF.becker13_tau0(jnp.asarray(leg.z))
    alpha3 = jnp.asarray([0.20, 0.08, 0.015])
    P, Ctot = DL.predict_P_obs_on_leg(
        ctx.model, jnp.asarray(MF_THETA), tau0_vec, alpha3,
        f_SiIII_nodes=jnp.asarray([0.01, 0.01]), f_SiII_nodes=None, metal_node_z=(2.2, 4.2),
        pf_stats=ctx.pf_stats, dla_core=jnp.mean(jnp.asarray(ctx.dla_core_leg[leg.name]), 0),
        cache_k=ctx.cache_k, leg=leg, sigma_zb=None, alpha_centres=None)
    assert np.all(np.isfinite(np.asarray(P))), "eBOSS high-z clamp produced non-finite P_model"


# =========================================================================== #
#  8. metal_inject EXTENSIONS: defaults byte-exact, a_*_direct bypass, damp='ma', form='ma2025'.
# =========================================================================== #
_K = np.array([1e-3, 5e-3, 0.01, 0.02, 0.04])
_P = np.ones_like(_K) * 10.0
_FBAR = 0.7


def test_metal_inject_direct_amplitude_bypasses_f_over_one_minus_F():
    """a_SiIII_direct sets the oscillation amplitude DIRECTLY (skips f/(1-<F>)). desi_full with
    a_SiIII_direct=A reproduces desi_full with f_SiIII=A*(1-<F>) (the same A)."""
    A = 0.05
    direct = C.metal_inject(_P, _K, _FBAR, form="desi_full", a_SiIII_direct=A, f_SiII=0.0,
                            f_SiII_SiII=0.0)
    viaf = C.metal_inject(_P, _K, _FBAR, form="desi_full", f_SiIII=A * (1.0 - _FBAR), f_SiII=0.0,
                          f_SiII_SiII=0.0)
    np.testing.assert_allclose(direct, viaf, rtol=1e-12, err_msg="a_SiIII_direct != f/(1-<F>) bypass")


def test_metal_inject_damp_ma_replaces_decorrelation():
    """damp='ma' applies exp(k/k_cross) on the SiIII cross term instead of the default decorrelation;
    k_cross<0 -> the oscillation DECAYS with k (vs the sigmoid/no-decorr default)."""
    f = 0.01
    sig = C.metal_inject(_P, _K, _FBAR, form="eboss", f_SiIII=f)                       # default
    ma = C.metal_inject(_P, _K, _FBAR, form="eboss", f_SiIII=f, damp="ma", k_cross=-0.02)
    assert not np.allclose(sig, ma), "damp='ma' did not change the eboss factor"
    aa = f / (1.0 - _FBAR)
    dv = DL.C_KMS * np.log(DL.LAMBDA_LYA / DL.LAMBDA_SiIII)
    ref = _P * (1.0 + aa ** 2 + 2.0 * aa * np.cos(dv * _K) * np.exp(_K / -0.02))
    np.testing.assert_allclose(ma, ref, rtol=1e-12, err_msg="damp='ma' != exp(k/k_cross) form")


def test_metal_inject_ma2025_form():
    """form='ma2025' (Arm 3): SiIII-only, DIRECT amplitude, Ma damping exp(k/k_cross). Finite; at
    k=0 the cross term is undamped -> factor=(1+a)^2; at high k (k_cross<0) the oscillation decays
    toward the DC limit 1+a^2."""
    a = 0.014
    kc = -1.5e-2
    out = C.metal_inject(_P, _K, _FBAR, form="ma2025", a_SiIII_direct=a, k_cross=kc, damp="ma")
    assert np.all(np.isfinite(out)), "ma2025 factor not finite"
    dv = DL.C_KMS * np.log(DL.LAMBDA_LYA / DL.LAMBDA_SiIII)
    ref = _P * (1.0 + a ** 2 + 2.0 * a * np.cos(dv * _K) * np.exp(_K / kc))
    np.testing.assert_allclose(out, ref, rtol=1e-12, err_msg="ma2025 != 1+a^2+2a cos(dv k) exp(k/k_cross)")
    # k=0 limit: cos=1, exp=1 -> (1+a)^2 (the cross term is present/undamped at k->0).
    out0 = C.metal_inject(np.array([1.0]), np.array([0.0]), _FBAR, form="ma2025",
                          a_SiIII_direct=a, k_cross=kc, damp="ma")
    assert float(out0[0]) == pytest.approx((1.0 + a) ** 2, rel=1e-12)
    # high-k: exp(k/k_cross) -> 0 so the factor approaches the DC term 1+a^2.
    outhi = C.metal_inject(np.array([1.0]), np.array([5.0]), _FBAR, form="ma2025",
                           a_SiIII_direct=a, k_cross=kc, damp="ma")
    assert float(outhi[0]) == pytest.approx(1.0 + a ** 2, abs=1e-6)


def test_metal_inject_defaults_byte_exact_to_legacy():
    """Defaults (a_*_direct=None, damp='sigmoid', k_cross=None) reproduce the legacy metal_inject
    exactly (the test_metal_inject golden) on BOTH forms."""
    for form in ("desi_full", "eboss"):
        out = C.metal_inject(_P, _K, _FBAR, form=form, f_SiIII=0.009, f_SiII=0.004, f_SiII_SiII=0.002)
        # recompute the legacy expression inline (no new kwargs).
        omf = max(1.0 - _FBAR, 1e-3); A3 = 0.009 / omf
        dvA = DL.C_KMS * np.log(DL.LAMBDA_LYA / DL.LAMBDA_SiIII)
        if form == "eboss":
            ref = _P * (1.0 + A3 ** 2 + 2.0 * A3 * np.cos(dvA * _K))
        else:
            dva = DL.C_KMS * np.log(DL.LAMBDA_LYA / DL.LAMBDA_SiII)
            dvb = DL.C_KMS * np.log(DL.LAMBDA_LYA / C._LAMBDA_SiIIb)
            dvd = DL.C_KMS * np.log(C._LAMBDA_SiIIb / DL.LAMBDA_SiII)
            D = 2.0 - 2.0 / (1.0 + np.exp(-_K / 0.05)); A2 = 0.004 / omf
            CL3 = A3 ** 2 + 2.0 * A3 * np.cos(dvA * _K) * D
            CL2 = A2 ** 2 * (1.0 + 0.25) + 2.0 * A2 * (np.cos(dvb * _K) + 0.5 * np.cos(dva * _K)) * D
            CSS = 0.002 * (1.0 + 0.25 + 2.0 * 0.5 * np.cos(dvd * _K)) * np.exp(-(_K / 0.05) ** 2)
            ref = _P * (1.0 + CL3 + CL2 + CSS)
        np.testing.assert_allclose(out, ref, rtol=1e-12, atol=0.0, err_msg=f"{form} default drifted")


# =========================================================================== #
#  9. The 3 metal_zevo INJECTION ARMS (de-double-count; eBOSS SiIII-only, DESI SiIII+SiII).
# =========================================================================== #
@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI not present")
def test_arms_registry_amplitude_trends():
    """C.METAL_ZEVO_ARMS: arm1 a(2.2)>a(4.2) (decreasing via 1/(1-<F>)); arm2 a(2.2)<a(4.2)
    (increasing, the f-rise overcomes 1/(1-<F>)); arm3 is the Ma+2025 direct power law."""
    arms = C.METAL_ZEVO_ARMS
    node_z = (2.2, 4.2)
    # ⟨F⟩(z) from Kim07 at the fiducial center: 1-<F>(2.2) << 1-<F>(4.2) so a=f/(1-<F>) falls with z.
    Fbar = {z: float(np.exp(-np.asarray(MF.becker13_tau0(jnp.asarray(z))))) for z in (2.2, 4.2)}
    for arm, decreasing in (("arm1_decreasing", True), ("arm2_increasing", False)):
        spec = arms[arm]; fn = spec["f_SiIII_nodes"]
        a_lo = C._metal_f_of_z(2.2, node_z, fn) / (1.0 - Fbar[2.2])
        a_hi = C._metal_f_of_z(4.2, node_z, fn) / (1.0 - Fbar[4.2])
        if decreasing:
            assert a_lo > a_hi, f"{arm}: expected a(2.2) > a(4.2), got {a_lo:.4g} vs {a_hi:.4g}"
        else:
            assert a_lo < a_hi, f"{arm}: expected a(2.2) < a(4.2), got {a_lo:.4g} vs {a_hi:.4g}"
    assert arms["arm3_ma2025"]["form"] == "ma2025"


@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI not present")
def test_arm_injection_de_double_count_and_desi_siII():
    """make_leg_a_legmock under flatlog2node: the CLEAN self-draw truth carries NO metal (a_siiii=0,
    f-nodes NOT forwarded) so the injected arm is the SOLE metal signal (de-double-count); on DESI the
    injected/clean ratio matches metal_inject(form='desi_full') (SiIII+SiII) and NOT the SiIII-only
    eboss form; KS (metals_off) is untouched."""
    ctx, d = _ctx_2node(with_eboss=False, metals_on=True)
    core = _core(ctx, d)
    tp = C.draw_leg_a_leg_truth(ctx, jax.random.PRNGKey(3))
    assert float(tp["a_siiii"]) == 0.0, "flatlog2node clean truth must carry no scalar metal"
    key = jax.random.PRNGKey(4)
    base_legs, base_info = C.make_leg_a_legmock(ctx, core, tp, key)
    arm = C.METAL_ZEVO_ARMS["arm1_decreasing"]
    inj_legs, inj_info = C.make_leg_a_legmock(ctx, core, tp, key, inject_metal_misspec=dict(arm))
    node_z = arm["node_z"]
    for leg in ctx.legs:
        b = np.asarray(base_info["truth_on_leg"][leg.name])
        j = np.asarray(inj_info["truth_on_leg"][leg.name])
        assert np.all(np.isfinite(j))
        if not leg.metals_on:                                   # KS untouched
            np.testing.assert_array_equal(b, j)
            continue
        assert not np.allclose(b, j), f"{leg.name}: arm did not contaminate the truth"
        Fbar = C._meanflux_on_leg(ctx, leg, tp)
        k = np.asarray(leg.k); z_idx = np.asarray(leg.z_idx)
        exp_full = b.copy()                                     # DESI: SiIII+SiII
        for iz in range(leg.n_z):
            rows = np.where(z_idx == iz)[0]
            if rows.size == 0:
                continue
            f3 = float(C._metal_f_of_z(float(leg.z[iz]), node_z, arm["f_SiIII_nodes"]))
            f2 = float(C._metal_f_of_z(float(leg.z[iz]), node_z, arm["f_SiII_nodes"]))
            # Model C+ arm: in-class desi_full + cross=True, decorrelation = the arm's k_SiIII/k_SiII.
            exp_full[rows] = C.metal_inject(b[rows], k[rows], float(Fbar[iz]), form="desi_full",
                                            f_SiIII=f3, f_SiII=f2, f_SiII_SiII=0.0,
                                            k_decorr=arm["k_SiIII"], k_SiII=arm["k_SiII"],
                                            r_doublet=DL.R_SiII_DOUBLET, cross=True)
        # de-double-count: applied EXACTLY once (== metal_inject of the metal-free clean truth).
        np.testing.assert_allclose(j, exp_full, rtol=1e-9, atol=0.0,
                                   err_msg=f"{leg.name}: arm metal not applied exactly once")


@pytest.mark.skipif(not _have_eboss, reason="real cache/ckpt/DESI/eBOSS not present")
def test_arm_injection_eboss_is_siIII_only():
    """4-lens fix (a): on eBOSS (not in metal_siII_legs) the zevo arm injects the SiIII-ONLY
    contaminant IN-CLASS via desi_full+f_SiII=0 (no SiII doublet, no cross — they are ∝ a_SiII) with
    the decorrelation matched to the forward, NOT the OLD undamped eboss form."""
    ctx, d = _ctx_2node(with_eboss=True, metals_on=True)
    core = _core(ctx, d)
    tp = C.draw_leg_a_leg_truth(ctx, jax.random.PRNGKey(7))
    key = jax.random.PRNGKey(8)
    base_legs, base_info = C.make_leg_a_legmock(ctx, core, tp, key)
    arm = C.METAL_ZEVO_ARMS["arm2_increasing"]
    inj_legs, inj_info = C.make_leg_a_legmock(ctx, core, tp, key, inject_metal_misspec=dict(arm))
    leg = next(l for l in ctx.legs if l.name == "eBOSS")
    b = np.asarray(base_info["truth_on_leg"][leg.name]); j = np.asarray(inj_info["truth_on_leg"][leg.name])
    Fbar = C._meanflux_on_leg(ctx, leg, tp)
    k = np.asarray(leg.k); z_idx = np.asarray(leg.z_idx); node_z = arm["node_z"]
    exp_inclass = b.copy(); exp_eboss = b.copy()
    for iz in range(leg.n_z):
        rows = np.where(z_idx == iz)[0]
        if rows.size == 0:
            continue
        f3 = float(C._metal_f_of_z(float(leg.z[iz]), node_z, arm["f_SiIII_nodes"]))
        exp_inclass[rows] = C.metal_inject(b[rows], k[rows], float(Fbar[iz]), form="desi_full",
                                           f_SiIII=f3, f_SiII=0.0, f_SiII_SiII=0.0,
                                           k_decorr=arm["k_SiIII"], k_SiII=arm["k_SiII"], cross=True)
        exp_eboss[rows] = C.metal_inject(b[rows], k[rows], float(Fbar[iz]), form="eboss", f_SiIII=f3)
    np.testing.assert_allclose(j, exp_inclass, rtol=1e-9, atol=0.0,
                               err_msg="eBOSS arm != in-class desi_full(f_SiII=0)")
    assert not np.allclose(j, exp_eboss), "eBOSS arm still uses the OLD undamped eboss form"


@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI not present")
def test_arm3_ma2025_injection_finite_and_siII_off():
    """Arm 3 (Ma+2025) injects the SiIII-only ma2025 form with a per-z direct amplitude + per-z
    k_cross; finite, contaminates DESI, no SiII doublet (SiII off)."""
    ctx, d = _ctx_2node(with_eboss=False, metals_on=True)
    core = _core(ctx, d)
    tp = C.draw_leg_a_leg_truth(ctx, jax.random.PRNGKey(9))
    key = jax.random.PRNGKey(10)
    base_legs, base_info = C.make_leg_a_legmock(ctx, core, tp, key)
    arm = C.METAL_ZEVO_ARMS["arm3_ma2025"]
    inj_legs, inj_info = C.make_leg_a_legmock(ctx, core, tp, key, inject_metal_misspec=dict(arm))
    leg = _desi_leg(ctx)
    b = np.asarray(base_info["truth_on_leg"][leg.name]); j = np.asarray(inj_info["truth_on_leg"][leg.name])
    assert np.all(np.isfinite(j)) and not np.allclose(b, j)
    k = np.asarray(leg.k); z_idx = np.asarray(leg.z_idx)
    exp_ma = b.copy()
    for iz in range(leg.n_z):
        rows = np.where(z_idx == iz)[0]
        if rows.size == 0:
            continue
        z = float(leg.z[iz])
        a_z = 0.014 * ((1.0 + z) / 4.0) ** 0.79
        kc = -1.58e-2 * ((1.0 + z) / 4.0) ** 1.15
        exp_ma[rows] = C.metal_inject(b[rows], k[rows], 0.0, form="ma2025",
                                      a_SiIII_direct=a_z, k_cross=kc, damp="ma")
    np.testing.assert_allclose(j, exp_ma, rtol=1e-9, atol=0.0,
                               err_msg="arm3 ma2025 injection mismatch")


# =========================================================================== #
#  10. Downstream by-name readers stay robust (no crash) under flatlog2node.
# =========================================================================== #
def test_downstream_readers_robust_without_scalar_metal():
    """_draws_matrix / _packed_names_for skip the absent scalar a_SiIII/a_SiII cleanly under
    flatlog2node (the node-aware re-score is DEFERRED to the combined re-SBC)."""
    L, nzg = 4, 3
    samples = dict(
        theta_unit=np.zeros((L, 9)), tau0_vec=np.zeros((L, nzg)),
        alpha_lls=np.full(L, 0.2), alpha_subdla=np.full(L, 0.08), alpha_dla=np.full(L, 0.015),
        f_SiIII_DESI_z0=np.full(L, 0.01), f_SiIII_DESI_z1=np.full(L, 0.008))
    kept = np.array([True, True, False])
    names = C._packed_names_for(samples, kept)
    mat = C._draws_matrix(samples, kept)
    assert "a_SiIII" not in names and "a_SiII" not in names
    assert mat.shape == (L, len(names)), "draws-matrix width != packed-names length under flatlog2node"


# =========================================================================== #
#  11. NUTS smoke (flatlog2node, with_eboss) — 0 div, finite, posterior off rails.
# =========================================================================== #
@pytest.mark.slow
@pytest.mark.skipif(not _have_eboss, reason="real cache/ckpt/DESI/eBOSS not present")
def test_flatlog2node_nuts_smoke():
    """A short flatlog2node NUTS run (with_eboss): 0 divergences, finite draws, all 6 metal node
    sites present and interior to (fnode_lo, fnode_hi)."""
    ctx, d = _ctx_2node(with_eboss=True, metals_on=True)
    core = _core(ctx, d)
    tp = C.draw_leg_a_leg_truth(ctx, jax.random.PRNGKey(11))
    mock_legs, _info = C.make_leg_a_legmock(
        ctx, core, tp, jax.random.PRNGKey(12),
        inject_metal_misspec=dict(C.METAL_ZEVO_ARMS["arm1_decreasing"]))
    samples, n_div = C._run_nuts_legb(ctx, mock_legs, core, n_warmup=15, n_samples=15, seed=0,
                                      target_accept=0.9, dense_mass=False, max_tree_depth=8)
    assert n_div == 0, f"flatlog2node smoke: {n_div} divergence(s)"
    lo, hi = float(ctx.metal_fnode_lo), float(ctx.metal_fnode_hi)
    for nm in ("f_SiIII_DESI_z0", "f_SiIII_DESI_z1", "f_SiII_DESI_z0", "f_SiII_DESI_z1",
               "f_SiIII_eBOSS_z0", "f_SiIII_eBOSS_z1"):
        assert nm in samples, f"node site {nm} missing from the posterior"
        a = np.asarray(samples[nm])
        assert np.all(np.isfinite(a)), f"{nm} not finite"
        assert np.all((a > lo) & (a < hi)), f"{nm} outside the LogUniform support"


# A fiducial in-box theta_unit for the forward tests (interior to the unit cube).
MF_THETA = np.full(9, 0.5)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q", "-m", "not slow"]))
