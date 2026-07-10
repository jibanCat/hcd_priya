"""Model C+ — float the SiIII/SiII decorrelation scales + add the SiIII-SiII metal-metal cross term.

EXTENDS Model C (commit 0600919, the ``metal_prior=="flatlog2node"`` path). The cup1d cross-check
(notes 2026-06-30-{desi-metal-model-vs-modelc, cup1d-likelihood-crosscheck}) found Model C is an
amplitude-only SUBSET of the DESI Eq-4.9 metal model. Model C+ closes the two un-amplitude-absorbable
gaps, matching cup1d ``contaminants/si_mult.py`` (class SiMult):

  (i)  FLOAT the sigmoid decorrelation scales k_SiIII / k_SiII (cup1d s_Lya_SiIII / s_Lya_SiII) per
       metals_on leg as 2-node LogUniform sites (the un-amplitude-absorbable OSCILLATORY residual =
       the n_s-leak channel). The forward stops hardcoding k=0.05.
  (ii) ADD the SiIII-SiII metal-metal CROSS term Cmm (cup1d si_mult.py ~:299-314), amplitude TIED to
       a_SiIII·a_SiII (NO new DOF), OFF when a_SiII=0 (eBOSS / back-compat byte-exact).

Plus the 4-lens wiring fixes (review wqfadip0j): (4a) eBOSS zevo arms truly in-class (inject via
form="desi_full", f_SiII=0, decorrelation matched to the forward — NOT the undamped form="eboss");
(4b) a 4th OUT-of-class arm (decreasing-trend metal with a GAUSS damping the sigmoid forward cannot
reproduce); (4c) the runner records tau0_amp AND dtau0 jointly with n_s/A_p.

uniform / flatlog stay BYTE-EXACT (test_legb_metal_flatlog.py is NOT regenerated). The flatlog2node
forward golden in test_legb_metal_zevo.py IS regenerated (the forward legitimately gains the cross
term + the floated decorrelation) — documented there.

Env (MANDATORY):
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
    /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_legb_metal_modelcplus.py -q
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

MF_THETA = np.full(9, 0.5)


# --------------------------------------------------------------------------- #
#  Helpers.
# --------------------------------------------------------------------------- #
def _sites(model, *a, seed=1):
    tr = handlers.trace(handlers.seed(model, jax.random.PRNGKey(seed))).get_trace(*a)
    return [k for k, v in tr.items() if v["type"] == "sample" and not v.get("is_observed")]


def _trace(model, *a, seed=1):
    return handlers.trace(handlers.seed(model, jax.random.PRNGKey(seed))).get_trace(*a)


def _ctx_2node(with_eboss=False, metals_on=True, desi_z_hi=2.6):
    ctx, d = C.build_legb_ctx(
        use_xclass=True, sample_metals=True, metals_on=metals_on, with_eboss=with_eboss,
        metal_prior="flatlog2node", desi_kwargs=dict(z_lo=0.0, z_hi=desi_z_hi))
    return ctx, d


def _core(ctx, d):
    return {k: np.nanmean(np.asarray(v), axis=0)
            for k, v in C._fiducial_dla_core_per_leg(d, ctx.legs, ctx.cache_k).items()}


def _desi_leg(ctx):
    return next(l for l in ctx.legs if l.name == "DESI")


def _logk_interp(z, node_z, nodes):
    """numpy twin of the per-z log-interp (log10 nodes LINEAR in log10(1+z), clamped)."""
    return 10.0 ** np.interp(np.log10(1.0 + np.asarray(z, float)),
                             np.log10(1.0 + np.asarray(node_z, float)), np.log10(np.asarray(nodes, float)))


# =========================================================================== #
#  1. k-node site inventory + order (the NEW floated decorrelation sites).
# =========================================================================== #
@pytest.mark.skipif(not _have_eboss, reason="real cache/ckpt/DESI/eBOSS not present")
def test_knode_site_inventory_and_order():
    """flatlog2node now ALSO samples, per metals_on leg IN ORDER, per ion, 2 node values of the
    sigmoid decorrelation SCALE k_SiIII (and k_SiII on metal_siII_legs) ~ LogUniform(knode_lo,
    knode_hi). The per-leg block is f_SiIII(2), [f_SiII(2)], k_SiIII(2), [k_SiII(2)]."""
    ctx, d = _ctx_2node(with_eboss=True, metals_on=True)
    assert hasattr(ctx, "metal_knode_lo") and hasattr(ctx, "metal_knode_hi")
    assert float(ctx.metal_knode_lo) < float(ctx.metal_knode_hi)
    core = _core(ctx, d)
    s = _sites(C._legb_model, ctx, ctx.legs, core)
    metal = [n for n in s if n.startswith("f_SiIII") or n.startswith("f_SiII_")
             or n.startswith("k_SiIII") or n.startswith("k_SiII_")]
    assert metal == [
        "f_SiIII_DESI_z0", "f_SiIII_DESI_z1", "f_SiII_DESI_z0", "f_SiII_DESI_z1",
        "k_SiIII_DESI_z0", "k_SiIII_DESI_z1", "k_SiII_DESI_z0", "k_SiII_DESI_z1",
        "f_SiIII_eBOSS_z0", "f_SiIII_eBOSS_z1",
        "k_SiIII_eBOSS_z0", "k_SiIII_eBOSS_z1"], f"metal site order wrong: {metal}"
    trm = _trace(C._legb_model, ctx, ctx.legs, core)
    for nm in ("k_SiIII_DESI_z0", "k_SiII_DESI_z1", "k_SiIII_eBOSS_z0"):
        fn = trm[nm]["fn"]
        assert isinstance(fn, dist.LogUniform), f"{nm} not LogUniform: {type(fn).__name__}"
        assert float(fn.low) == pytest.approx(float(ctx.metal_knode_lo), rel=1e-12)
        assert float(fn.high) == pytest.approx(float(ctx.metal_knode_hi), rel=1e-12)
    # eBOSS (not in metal_siII_legs) gets NO k_SiII nodes.
    assert not any(n.startswith("k_SiII_eBOSS") for n in metal)


# =========================================================================== #
#  2. constrain_fn MIRROR parity with the new k-nodes.
# =========================================================================== #
@pytest.mark.skipif(not _have_eboss, reason="real cache/ckpt/DESI/eBOSS not present")
def test_knode_constrain_fn_mirror_parity():
    """_legb_model and _legb_priors_only trace the IDENTICAL ordered site list (now incl. k-nodes),
    and constrain_fn round-trips a k-node into (knode_lo, knode_hi)."""
    ctx, d = _ctx_2node(with_eboss=True, metals_on=True)
    core = _core(ctx, d)
    s_model = _sites(C._legb_model, ctx, ctx.legs, core)
    s_prior = _sites(C._legb_priors_only, ctx)
    assert s_model == s_prior, f"mirror mismatch:\n model={s_model}\n prior={s_prior}"
    lo, hi = float(ctx.metal_knode_lo), float(ctx.metal_knode_hi)
    init = initialize_model(jax.random.PRNGKey(2), lambda: C._legb_priors_only(ctx))
    z0 = dict(init.param_info.z)
    assert "k_SiIII_DESI_z0" in z0 and "k_SiII_DESI_z0" in z0 and "k_SiIII_eBOSS_z0" in z0
    for u in (-15.0, 0.0, 15.0):
        z = dict(z0); z["k_SiIII_DESI_z0"] = jnp.asarray(u)
        c = constrain_fn(lambda: C._legb_priors_only(ctx), (), {}, z, return_deterministic=False)
        kk = float(c["k_SiIII_DESI_z0"])
        assert lo < kk < hi, f"constrain k_SiIII_DESI_z0 {kk:.4g} not in ({lo},{hi})"


# =========================================================================== #
#  3. SiIII-SiII cross term matches cup1d si_mult.py Cmm (hardcoded cup1d numbers).
# =========================================================================== #
def test_siIII_siII_cross_matches_cup1d():
    """The new _metal_factor cross term, extracted as factor(cross=True) − factor(cross=False) at a
    large decorrelation scale (D→1), reproduces cup1d si_mult.py Cmm (the metal-metal SiIII-SiII
    cross) to within the documented r=0.5 vs ra3/rb3=0.4806 doublet-ratio approximation.

    cup1d Cmm (G_SiII_SiIII=1, f_SiIIa_SiIII=1, off SiIIc=0): 2·aSiIII·aSiII·(cos(dv_SiIII_SiIIb·k)
    + (ra3/rb3)·cos(dv_SiIII_SiIIa·k)).  The HARDCODED cup1d values below were computed from the
    exact cup1d wav/osc constants (1206.51 / 1193.28 / 1190.42; 1.67 / 0.575 / 0.277)."""
    A3, A2 = 0.045, 0.010
    k = np.array([0.0, 1e-3, 5e-3, 0.01, 0.02, 0.04, 0.06])
    # cup1d Cmm @ (A3=0.045, A2=0.010) — HARDCODED from cup1d si_mult.py exact constants.
    cup1d_cmm = np.array([0.0013325261, -0.0011624041, -0.000488139, -0.0004207679,
                          -0.0007279319, 0.0005575495, -0.0012219672])
    BIG = 1e9    # k_SiII huge → sigmoid D→1 (compare the UNDAMPED Cmm form to cup1d, which has no D)
    full = np.asarray(DL._metal_factor(jnp.asarray(k), a_SiIII=A3, a_SiII=A2,
                                       k_SiIII=BIG, k_SiII=BIG, cross=True))
    nocross = np.asarray(DL._metal_factor(jnp.asarray(k), a_SiIII=A3, a_SiII=A2,
                                          k_SiIII=BIG, k_SiII=BIG, cross=False))
    our_cross = full - nocross
    np.testing.assert_allclose(our_cross, cup1d_cmm, atol=5e-5, rtol=0.0,
                               err_msg="SiIII-SiII cross does not match cup1d si_mult.py Cmm")
    # the velocity separations are the SiIII-SiII atomic constants (match cup1d to <0.1%).
    dv_b = abs(DL.C_KMS * np.log(DL.LAMBDA_SiIIb / DL.LAMBDA_SiIII))
    dv_a = abs(DL.C_KMS * np.log(DL.LAMBDA_SiII / DL.LAMBDA_SiIII))
    assert dv_b == pytest.approx(3305.53, rel=1e-3), f"dv_SiIII_SiIIb {dv_b} != cup1d 3305.53"
    assert dv_a == pytest.approx(4024.93, rel=1e-3), f"dv_SiIII_SiIIa {dv_a} != cup1d 4024.93"
    # amplitude TIED to a_SiIII·a_SiII (no new DOF): doubling a_SiII doubles the cross.
    full2 = np.asarray(DL._metal_factor(jnp.asarray(k), a_SiIII=A3, a_SiII=2 * A2,
                                        k_SiIII=BIG, k_SiII=BIG, cross=True))
    nocross2 = np.asarray(DL._metal_factor(jnp.asarray(k), a_SiIII=A3, a_SiII=2 * A2,
                                           k_SiIII=BIG, k_SiII=BIG, cross=False))
    np.testing.assert_allclose(full2 - nocross2, 2.0 * our_cross, rtol=1e-9, atol=0.0,
                               err_msg="cross not linear in a_SiII (amplitude not tied to a_SiIII·a_SiII)")


# =========================================================================== #
#  4. cross is OFF when a_SiII=0 (eBOSS / back-compat byte-exact).
# =========================================================================== #
def test_cross_off_when_a_siII_zero():
    """With a_SiII=0 the cross term vanishes, so _metal_factor(cross=True) == _metal_factor(cross=
    False) byte-for-byte (eBOSS is SiIII-only; and the default cross=False keeps the legacy output)."""
    k = jnp.asarray([1e-3, 5e-3, 0.01, 0.02, 0.04, 0.06])
    on = np.asarray(DL._metal_factor(k, a_SiIII=0.04, a_SiII=0.0, k_SiIII=0.05, k_SiII=0.05, cross=True))
    off = np.asarray(DL._metal_factor(k, a_SiIII=0.04, a_SiII=0.0, k_SiIII=0.05, k_SiII=0.05, cross=False))
    np.testing.assert_array_equal(on, off)
    # default (no cross kwarg) is byte-exact to cross=False for ANY a_SiII (the gate is the flag).
    default = np.asarray(DL._metal_factor(k, a_SiIII=0.04, a_SiII=0.02, k_SiIII=0.05, k_SiII=0.05))
    explicit_off = np.asarray(DL._metal_factor(k, a_SiIII=0.04, a_SiII=0.02, k_SiIII=0.05,
                                               k_SiII=0.05, cross=False))
    np.testing.assert_array_equal(default, explicit_off)


# =========================================================================== #
#  5. cross forward-formula GOLDEN — _metal_factor(cross=True) == the spec formula.
# =========================================================================== #
def test_cross_forward_formula_golden():
    """_metal_factor(cross=True) == 1 + C_LyaSiIII + C_LyaSiII + C_cross, with
    C_cross = 2·a_SiIII·a_SiII·(cos(dv_b·k) + r·cos(dv_a·k))  (UNDAMPED, matching cup1d Cmm;
    r=R_SiII_DOUBLET, dv_X = c·ln(λ_SiIIX/λ_SiIII)). Pins the implementation to the spec."""
    A3, A2, k3, k2 = 0.045, 0.012, 0.012, 0.006
    k = np.array([1e-3, 5e-3, 0.01, 0.02, 0.04, 0.06])
    out = np.asarray(DL._metal_factor(jnp.asarray(k), a_SiIII=A3, a_SiII=A2,
                                      k_SiIII=k3, k_SiII=k2, cross=True))
    dvA = DL.C_KMS * np.log(DL.LAMBDA_LYA / DL.LAMBDA_SiIII)
    dva = DL.C_KMS * np.log(DL.LAMBDA_LYA / DL.LAMBDA_SiII)
    dvb = DL.C_KMS * np.log(DL.LAMBDA_LYA / DL.LAMBDA_SiIIb)
    dvcb = DL.C_KMS * np.log(DL.LAMBDA_SiIIb / DL.LAMBDA_SiIII)     # SiIII-SiII line b
    dvca = DL.C_KMS * np.log(DL.LAMBDA_SiII / DL.LAMBDA_SiIII)      # SiIII-SiII line a
    r = DL.R_SiII_DOUBLET
    D3 = 2.0 - 2.0 / (1.0 + np.exp(-k / k3))
    D2 = 2.0 - 2.0 / (1.0 + np.exp(-k / k2))
    CL3 = A3 ** 2 + 2.0 * A3 * np.cos(dvA * k) * D3
    CL2 = A2 ** 2 * (1.0 + r ** 2) + 2.0 * A2 * (np.cos(dvb * k) + r * np.cos(dva * k)) * D2
    Ccross = 2.0 * A3 * A2 * (np.cos(dvcb * k) + r * np.cos(dvca * k))   # UNDAMPED (cup1d Cmm)
    ref = 1.0 + CL3 + CL2 + Ccross
    np.testing.assert_allclose(out, ref, rtol=1e-10, atol=0.0,
                               err_msg="_metal_factor(cross=True) != spec formula")


# =========================================================================== #
#  6. metal_inject: cross term + separate k_SiII + damp='gauss'; defaults byte-exact.
# =========================================================================== #
_K = np.array([1e-3, 5e-3, 0.01, 0.02, 0.04, 0.06])
_P = np.ones_like(_K) * 10.0
_FBAR = 0.7


def test_metal_inject_cross_and_kSiII():
    """metal_inject(form='desi_full', cross=True) adds the SiIII-SiII cross (∝ f_SiIII·f_SiII);
    cross=False (default) is byte-exact; a separate k_SiII decorrelation is honoured (default →
    k_decorr, byte-exact)."""
    f3, f2 = 0.012, 0.006
    base = C.metal_inject(_P, _K, _FBAR, form="desi_full", f_SiIII=f3, f_SiII=f2, f_SiII_SiII=0.0)
    crossed = C.metal_inject(_P, _K, _FBAR, form="desi_full", f_SiIII=f3, f_SiII=f2,
                             f_SiII_SiII=0.0, cross=True)
    assert not np.allclose(base, crossed), "cross=True did not change metal_inject"
    # the cross delta matches the spec form, with a3=f3/(1-F), a2=f2/(1-F); UNDAMPED (cup1d Cmm).
    omf = 1.0 - _FBAR
    a3, a2 = f3 / omf, f2 / omf
    dvcb = DL.C_KMS * np.log(DL.LAMBDA_SiIIb / DL.LAMBDA_SiIII)
    dvca = DL.C_KMS * np.log(DL.LAMBDA_SiII / DL.LAMBDA_SiIII)
    Ccross = 2.0 * a3 * a2 * (np.cos(dvcb * _K) + DL.R_SiII_DOUBLET * np.cos(dvca * _K))
    np.testing.assert_allclose(crossed - base, _P * Ccross, rtol=1e-10, atol=0.0,
                               err_msg="metal_inject cross delta != spec Cmm")
    # cross with f_SiII=0 is a no-op (eBOSS-style).
    z = C.metal_inject(_P, _K, _FBAR, form="desi_full", f_SiIII=f3, f_SiII=0.0,
                       f_SiII_SiII=0.0, cross=True)
    z0 = C.metal_inject(_P, _K, _FBAR, form="desi_full", f_SiIII=f3, f_SiII=0.0, f_SiII_SiII=0.0)
    np.testing.assert_array_equal(z, z0)
    # separate k_SiII: passing k_SiII != k_decorr changes ONLY the SiII/cross decorrelation.
    diff_k = C.metal_inject(_P, _K, _FBAR, form="desi_full", f_SiIII=f3, f_SiII=f2,
                            f_SiII_SiII=0.0, cross=True, k_SiII=0.02)
    assert not np.allclose(diff_k, crossed), "k_SiII override had no effect"
    # k_SiII=None default reproduces k_SiII=k_decorr byte-exact.
    same_k = C.metal_inject(_P, _K, _FBAR, form="desi_full", f_SiIII=f3, f_SiII=f2,
                            f_SiII_SiII=0.0, cross=True, k_SiII=0.05)
    np.testing.assert_array_equal(crossed, same_k)


def test_metal_inject_damp_gauss():
    """damp='gauss' applies a GAUSSIAN envelope exp(-(k/k_cross)^2) on the SiIII cross term (the
    out-of-class arm-4 damping the sigmoid forward cannot reproduce)."""
    a = 0.014
    kc = 0.02
    out = C.metal_inject(_P, _K, _FBAR, form="ma2025", a_SiIII_direct=a, k_cross=kc, damp="gauss")
    dv = DL.C_KMS * np.log(DL.LAMBDA_LYA / DL.LAMBDA_SiIII)
    ref = _P * (1.0 + a ** 2 + 2.0 * a * np.cos(dv * _K) * np.exp(-(_K / kc) ** 2))
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=0.0, err_msg="damp='gauss' != exp(-(k/kc)^2)")
    # ma (default) stays exp(k/k_cross) — distinct from gauss.
    ma = C.metal_inject(_P, _K, _FBAR, form="ma2025", a_SiIII_direct=a, k_cross=-kc, damp="ma")
    assert not np.allclose(out, ma)


def test_metal_inject_defaults_still_byte_exact():
    """The legacy metal_inject golden (cross omitted) is UNCHANGED on both forms (cross defaults OFF
    even at the nonzero default f_SiII=0.004)."""
    for form in ("desi_full", "eboss"):
        out = C.metal_inject(_P, _K, _FBAR, form=form, f_SiIII=0.009, f_SiII=0.004, f_SiII_SiII=0.002)
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
#  7. per-z DECORRELATION interp through the forward + the regenerated golden.
# =========================================================================== #
@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI not present")
def test_per_z_decorrelation_interp_and_forward_golden():
    """predict_P_obs_on_leg with k_SiIII_nodes/k_SiII_nodes interpolates k per z (log10 k LINEAR in
    log10(1+z)) and reproduces, per z, clean × metal_inject(form='desi_full', cross=True) at the
    per-z f AND per-z k — to machine precision (the Model C+ forward golden). A z-varying k-node pair
    differs from a constant one (the decorrelation is genuinely per-z)."""
    ctx, _ = _ctx_2node(with_eboss=False, metals_on=True)
    leg = _desi_leg(ctx)
    tau0_vec = MF.becker13_tau0(jnp.asarray(leg.z))
    alpha3 = jnp.asarray([0.20, 0.08, 0.015])
    node_z = (2.2, 4.2)
    f3_nodes = jnp.asarray([0.012, 0.006]); f2_nodes = jnp.asarray([0.007, 0.004])
    k3_nodes = jnp.asarray([0.012, 0.06]);  k2_nodes = jnp.asarray([0.008, 0.04])
    common = dict(pf_stats=ctx.pf_stats, dla_core=jnp.mean(jnp.asarray(ctx.dla_core_leg[leg.name]), 0),
                  cache_k=ctx.cache_k, leg=leg, sigma_zb=None, alpha_centres=None)
    P_nodes, _ = DL.predict_P_obs_on_leg(ctx.model, jnp.asarray(MF_THETA), tau0_vec, alpha3,
                                         f_SiIII_nodes=f3_nodes, f_SiII_nodes=f2_nodes,
                                         k_SiIII_nodes=k3_nodes, k_SiII_nodes=k2_nodes,
                                         metal_node_z=node_z, **common)
    P_clean, _ = DL.predict_P_obs_on_leg(ctx.model, jnp.asarray(MF_THETA), tau0_vec, alpha3,
                                         a_SiIII=0.0, a_SiII=0.0, **common)
    P_nodes = np.asarray(P_nodes); P_clean = np.asarray(P_clean)
    k = np.asarray(leg.k); z_idx = np.asarray(leg.z_idx)
    expect = P_clean.copy()
    for iz in range(leg.n_z):
        rows = np.where(z_idx == iz)[0]
        if rows.size == 0:
            continue
        z = float(leg.z[iz]); Fbar = float(np.exp(-np.asarray(tau0_vec)[iz]))
        f3_z = float(C._metal_f_of_z(z, node_z, np.asarray(f3_nodes)))
        f2_z = float(C._metal_f_of_z(z, node_z, np.asarray(f2_nodes)))
        k3_z = float(_logk_interp(z, node_z, np.asarray(k3_nodes)))
        k2_z = float(_logk_interp(z, node_z, np.asarray(k2_nodes)))
        expect[rows] = C.metal_inject(P_clean[rows], k[rows], Fbar, form="desi_full",
                                      f_SiIII=f3_z, f_SiII=f2_z, f_SiII_SiII=0.0,
                                      k_decorr=k3_z, k_SiII=k2_z,
                                      r_doublet=DL.R_SiII_DOUBLET, cross=True)
    np.testing.assert_allclose(P_nodes, expect, rtol=1e-9, atol=0.0,
                               err_msg="Model C+ per-z f/k forward != metal_inject(cross=True) parity")
    # the per-z k genuinely matters: a CONSTANT k-node pair gives a different forward.
    P_const, _ = DL.predict_P_obs_on_leg(ctx.model, jnp.asarray(MF_THETA), tau0_vec, alpha3,
                                         f_SiIII_nodes=f3_nodes, f_SiII_nodes=f2_nodes,
                                         k_SiIII_nodes=jnp.asarray([0.05, 0.05]),
                                         k_SiII_nodes=jnp.asarray([0.05, 0.05]),
                                         metal_node_z=node_z, **common)
    assert not np.allclose(np.asarray(P_const), P_nodes), "floated per-z k had no forward effect"


@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI not present")
def test_knodes_none_uses_scalar_decorr_default():
    """k_SiIII_nodes=None (with f-nodes given) falls back to the scalar k_SiIII/k_SiII=0.05 default
    (back-compat for the Model-C forward that floats only f); the cross is still applied (a2!=0)."""
    ctx, _ = _ctx_2node(with_eboss=False, metals_on=True)
    leg = _desi_leg(ctx)
    tau0_vec = MF.becker13_tau0(jnp.asarray(leg.z))
    alpha3 = jnp.asarray([0.20, 0.08, 0.015])
    common = dict(pf_stats=ctx.pf_stats, dla_core=jnp.mean(jnp.asarray(ctx.dla_core_leg[leg.name]), 0),
                  cache_k=ctx.cache_k, leg=leg, sigma_zb=None, alpha_centres=None)
    f3_nodes = jnp.asarray([0.012, 0.006]); f2_nodes = jnp.asarray([0.007, 0.004])
    P_noknodes, _ = DL.predict_P_obs_on_leg(ctx.model, jnp.asarray(MF_THETA), tau0_vec, alpha3,
                                            f_SiIII_nodes=f3_nodes, f_SiII_nodes=f2_nodes,
                                            metal_node_z=(2.2, 4.2), **common)
    P_const05, _ = DL.predict_P_obs_on_leg(ctx.model, jnp.asarray(MF_THETA), tau0_vec, alpha3,
                                           f_SiIII_nodes=f3_nodes, f_SiII_nodes=f2_nodes,
                                           k_SiIII_nodes=jnp.asarray([0.05, 0.05]),
                                           k_SiII_nodes=jnp.asarray([0.05, 0.05]),
                                           metal_node_z=(2.2, 4.2), **common)
    np.testing.assert_allclose(np.asarray(P_noknodes), np.asarray(P_const05), rtol=1e-9, atol=0.0,
                               err_msg="k_SiIII_nodes=None != scalar k=0.05 fallback")


# =========================================================================== #
#  8. The injection arms registry (Model C+): arms 1/2 carry k + cross; arm4 is OUT-of-class.
# =========================================================================== #
def test_arms_registry_modelcplus():
    """METAL_ZEVO_ARMS: arms 1/2 carry a decorrelation scale (k_SiIII/k_SiII) the forward CAN float
    to (inside the prior bracket); arm4 is a DECREASING-trend OUT-of-class metal with a GAUSS damping
    the sigmoid forward cannot reproduce."""
    arms = C.METAL_ZEVO_ARMS
    assert "arm4_decreasing_ooc" in arms, "the 4th (out-of-class) arm is missing"
    for a in ("arm1_decreasing", "arm2_increasing"):
        assert "k_SiIII" in arms[a] and "k_SiII" in arms[a], f"{a} lacks a matched decorrelation scale"
        assert 1e-3 <= arms[a]["k_SiIII"] <= 0.1, f"{a} k_SiIII outside the float bracket"
    a4 = arms["arm4_decreasing_ooc"]
    assert a4.get("damp") == "gauss", "arm4 must use the GAUSS (out-of-class) damping"
    # arm4 amplitude DECREASES with z (survey-standard low-z-dominated metal).
    node_z = (2.2, 4.2)
    a_lo = a4["a0"] * ((1.0 + 2.2) / 4.0) ** a4["p"]
    a_hi = a4["a0"] * ((1.0 + 4.2) / 4.0) ** a4["p"]
    assert a_lo > a_hi, f"arm4 must DECREASE with z: a(2.2)={a_lo:.4g} !> a(4.2)={a_hi:.4g}"


@pytest.mark.skipif(not _have_eboss, reason="real cache/ckpt/DESI/eBOSS not present")
def test_arm_eboss_in_class_to_machine_precision():
    """4-lens fix (a): the eBOSS zevo arm is now injected IN-CLASS via form='desi_full', f_SiII=0,
    with the decorrelation matched to the forward (NOT the undamped form='eboss'). The injected eBOSS
    truth equals, per z, metal_inject(form='desi_full', f_SiII=0, cross=True, k matched) to machine
    precision — and is DISTINCT from the old undamped eboss form."""
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
                               err_msg="eBOSS arm != in-class desi_full(f_SiII=0) form")
    assert not np.allclose(j, exp_eboss), "eBOSS arm still uses the OLD undamped eboss form (not in-class)"


@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI not present")
def test_arm4_out_of_class_decreasing_gauss():
    """4-lens fix (b): arm4 injects a DECREASING-trend SiIII metal with a GAUSS damping (out-of-class:
    the sigmoid forward cannot reproduce exp(-(k/k_d)^2)). Finite, contaminates DESI, SiII off."""
    ctx, d = _ctx_2node(with_eboss=False, metals_on=True)
    core = _core(ctx, d)
    tp = C.draw_leg_a_leg_truth(ctx, jax.random.PRNGKey(9))
    key = jax.random.PRNGKey(10)
    base_legs, base_info = C.make_leg_a_legmock(ctx, core, tp, key)
    arm = C.METAL_ZEVO_ARMS["arm4_decreasing_ooc"]
    inj_legs, inj_info = C.make_leg_a_legmock(ctx, core, tp, key, inject_metal_misspec=dict(arm))
    leg = _desi_leg(ctx)
    b = np.asarray(base_info["truth_on_leg"][leg.name]); j = np.asarray(inj_info["truth_on_leg"][leg.name])
    assert np.all(np.isfinite(j)) and not np.allclose(b, j)
    k = np.asarray(leg.k); z_idx = np.asarray(leg.z_idx)
    Fbar = C._meanflux_on_leg(ctx, leg, tp)
    exp = b.copy()
    for iz in range(leg.n_z):
        rows = np.where(z_idx == iz)[0]
        if rows.size == 0:
            continue
        z = float(leg.z[iz])
        a_z = arm["a0"] * ((1.0 + z) / 4.0) ** arm["p"]
        exp[rows] = C.metal_inject(b[rows], k[rows], float(Fbar[iz]), form="ma2025",
                                   a_SiIII_direct=a_z, k_cross=arm["k_cross"], damp="gauss")
    np.testing.assert_allclose(j, exp, rtol=1e-9, atol=0.0, err_msg="arm4 gauss injection mismatch")


@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI not present")
def test_arm_desi_in_class_carries_cross():
    """The DESI arms-1/2 in-class injection now carries the SiIII-SiII cross (cross=True) so the
    Model C+ forward (which has the cross) can reproduce it exactly (de-double-count clean truth)."""
    ctx, d = _ctx_2node(with_eboss=False, metals_on=True)
    core = _core(ctx, d)
    tp = C.draw_leg_a_leg_truth(ctx, jax.random.PRNGKey(3))
    assert float(tp["a_siiii"]) == 0.0
    key = jax.random.PRNGKey(4)
    base_legs, base_info = C.make_leg_a_legmock(ctx, core, tp, key)
    arm = C.METAL_ZEVO_ARMS["arm1_decreasing"]
    inj_legs, inj_info = C.make_leg_a_legmock(ctx, core, tp, key, inject_metal_misspec=dict(arm))
    leg = _desi_leg(ctx); node_z = arm["node_z"]
    b = np.asarray(base_info["truth_on_leg"][leg.name]); j = np.asarray(inj_info["truth_on_leg"][leg.name])
    Fbar = C._meanflux_on_leg(ctx, leg, tp)
    k = np.asarray(leg.k); z_idx = np.asarray(leg.z_idx)
    exp = b.copy(); exp_nocross = b.copy()
    for iz in range(leg.n_z):
        rows = np.where(z_idx == iz)[0]
        if rows.size == 0:
            continue
        f3 = float(C._metal_f_of_z(float(leg.z[iz]), node_z, arm["f_SiIII_nodes"]))
        f2 = float(C._metal_f_of_z(float(leg.z[iz]), node_z, arm["f_SiII_nodes"]))
        kw = dict(form="desi_full", f_SiIII=f3, f_SiII=f2, f_SiII_SiII=0.0,
                  k_decorr=arm["k_SiIII"], k_SiII=arm["k_SiII"])
        exp[rows] = C.metal_inject(b[rows], k[rows], float(Fbar[iz]), cross=True, **kw)
        exp_nocross[rows] = C.metal_inject(b[rows], k[rows], float(Fbar[iz]), **kw)
    np.testing.assert_allclose(j, exp, rtol=1e-9, atol=0.0, err_msg="DESI arm != in-class cross=True")
    assert not np.allclose(j, exp_nocross), "DESI arm omitted the SiIII-SiII cross"


# =========================================================================== #
#  9. (4c) the runner records tau0_amp AND dtau0 jointly with n_s/A_p.
# =========================================================================== #
@pytest.mark.slow
@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI not present")
def test_run_legb_records_tau0_amp_and_dtau0():
    """run_legb's per-mock record carries the tau0_amp + dtau0 SITES (draws + truth) — not only the
    deterministic tau0_z ladder — so the metal_zevo gate can report the mean-flux bias jointly with
    n_s/A_p (memory feedback-report-tau0-dtau0-bias)."""
    ctx, d = _ctx_2node(with_eboss=False, metals_on=True)
    arm = dict(C.METAL_ZEVO_ARMS["arm1_decreasing"])
    per_mock = C.run_legb(ctx, d, n_mocks=1, mock_indices=[0], return_per_mock=True, leg_a=True,
                          n_warmup=12, n_samples=12, seed=0, dense_mass=False, max_tree_depth=8,
                          inject_spec={"metal_misspec": arm}, verbose=False)
    rec = per_mock[0]
    assert "sites_extra" in rec, "per-mock record lacks sites_extra (tau0_amp/dtau0)"
    for nm in ("tau0_amp", "dtau0"):
        assert nm in rec["sites_extra"], f"{nm} not recorded in sites_extra"
        e = rec["sites_extra"][nm]
        assert np.isfinite(e["truth"]), f"{nm} truth not finite"
        dr = np.asarray(e["draws"])
        assert dr.shape[0] == rec["L"], f"{nm} draws len {dr.shape[0]} != L {rec['L']}"
        assert np.all(np.isfinite(dr)), f"{nm} draws not finite"


# =========================================================================== #
#  9b. run_legb ALSO records the Model C+ metal f/k node sites in sites_extra so the f_SiIII ceiling
#      check is possible from the shard pkls. Fast unit test of the packer + slow e2e confirmation.
# =========================================================================== #
def test_metal_node_sites_extra_helper():
    """_metal_node_sites_extra packs ONLY the f_/k_ metal-node sites (thinned by step, first L) with
    the injected-arm truth for an in-class 'zevo' arm, ignores theta/tau0/a_SiIII keys, and is EMPTY
    when no metal-node sites exist (golden-safe under uniform/flatlog/metals-off)."""
    ctx, d = _ctx_2node(with_eboss=False, metals_on=True)
    arm = dict(C.METAL_ZEVO_ARMS["arm1_decreasing"])   # f_SiIII=(.010,.010), f_SiII=(.006,.006), k=.05
    samples = {"theta_unit": np.zeros((10, 9)), "tau0_amp": np.arange(10.0), "a_SiIII": np.arange(10.0),
               "f_SiIII_DESI_z0": np.arange(10.0), "f_SiII_DESI_z1": np.arange(10.0),
               "k_SiIII_DESI_z1": np.arange(10.0)}
    out = C._metal_node_sites_extra(samples, step=2, L=3, inject_spec={"metal_misspec": arm},
                                    ctx=ctx, leg_a=True)
    assert set(out) == {"f_SiIII_DESI_z0", "f_SiII_DESI_z1", "k_SiIII_DESI_z1"}, (
        "must pack ONLY the f_/k_ metal nodes (not theta/tau0/a_SiIII)")
    np.testing.assert_array_equal(out["f_SiIII_DESI_z0"]["draws"], np.arange(10.0)[::2][:3])
    assert np.isclose(out["f_SiIII_DESI_z0"]["truth"], 0.010, atol=1e-9)
    assert np.isclose(out["f_SiII_DESI_z1"]["truth"], 0.006, atol=1e-9)
    assert np.isclose(out["k_SiIII_DESI_z1"]["truth"], 0.05, atol=1e-9)
    # golden-safe: no metal-node keys -> empty dict (uniform/flatlog/metals-off produce no f_/k_ sites)
    assert C._metal_node_sites_extra({"theta_unit": np.zeros((10, 9)), "a_SiIII": np.arange(10.0)},
                                     step=1, L=5, inject_spec=None, ctx=ctx, leg_a=True) == {}
    # clean run (inject_spec=None): node draws still stored, truth is NaN (undefined for a self-draw)
    clean = C._metal_node_sites_extra({"f_SiIII_DESI_z0": np.arange(10.0)}, step=1, L=5,
                                      inject_spec=None, ctx=ctx, leg_a=True)
    assert "f_SiIII_DESI_z0" in clean and np.isnan(clean["f_SiIII_DESI_z0"]["truth"])


@pytest.mark.slow
@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI not present")
def test_run_legb_records_metal_nodes_e2e():
    """End-to-end: a real run_legb under flatlog2node records the metal f/k node sites in sites_extra
    (draws len L, injected-arm truth) alongside tau0_amp/dtau0 — the ceiling-check instrumentation."""
    ctx, d = _ctx_2node(with_eboss=False, metals_on=True)
    arm = dict(C.METAL_ZEVO_ARMS["arm1_decreasing"])
    per_mock = C.run_legb(ctx, d, n_mocks=1, mock_indices=[0], return_per_mock=True, leg_a=True,
                          n_warmup=12, n_samples=12, seed=0, dense_mass=False, max_tree_depth=8,
                          inject_spec={"metal_misspec": arm}, verbose=False)
    se = per_mock[0]["sites_extra"]; L = per_mock[0]["L"]
    for nm, truth in (("f_SiIII_DESI_z0", 0.010), ("f_SiII_DESI_z1", 0.006),
                      ("k_SiIII_DESI_z0", 0.05), ("k_SiII_DESI_z1", 0.05)):
        assert nm in se, f"{nm} not in sites_extra (metal-node instrumentation not wired)"
        assert np.asarray(se[nm]["draws"]).shape[0] == L and np.all(np.isfinite(se[nm]["draws"]))
        assert np.isclose(se[nm]["truth"], truth, atol=1e-9)
    assert "tau0_amp" in se and "dtau0" in se, "tau0 sites regressed"


# =========================================================================== #
#  10. NUTS smoke (Model C+, with_eboss) — 0 div, all f+k node sites present + in support.
# =========================================================================== #
@pytest.mark.slow
@pytest.mark.skipif(not _have_eboss, reason="real cache/ckpt/DESI/eBOSS not present")
def test_modelcplus_nuts_smoke():
    """A short Model C+ NUTS run (with_eboss, arm1 injected): 0 divergences, finite draws, all metal
    f-node AND k-node sites present and interior to their LogUniform support."""
    ctx, d = _ctx_2node(with_eboss=True, metals_on=True)
    core = _core(ctx, d)
    tp = C.draw_leg_a_leg_truth(ctx, jax.random.PRNGKey(11))
    mock_legs, _info = C.make_leg_a_legmock(
        ctx, core, tp, jax.random.PRNGKey(12),
        inject_metal_misspec=dict(C.METAL_ZEVO_ARMS["arm1_decreasing"]))
    samples, n_div = C._run_nuts_legb(ctx, mock_legs, core, n_warmup=15, n_samples=15, seed=0,
                                      target_accept=0.9, dense_mass=False, max_tree_depth=8)
    assert n_div == 0, f"Model C+ smoke: {n_div} divergence(s)"
    flo, fhi = float(ctx.metal_fnode_lo), float(ctx.metal_fnode_hi)
    klo, khi = float(ctx.metal_knode_lo), float(ctx.metal_knode_hi)
    for nm in ("f_SiIII_DESI_z0", "f_SiII_DESI_z1", "f_SiIII_eBOSS_z0"):
        a = np.asarray(samples[nm]); assert np.all(np.isfinite(a)) and np.all((a > flo) & (a < fhi))
    for nm in ("k_SiIII_DESI_z0", "k_SiII_DESI_z1", "k_SiIII_eBOSS_z0"):
        a = np.asarray(samples[nm]); assert np.all(np.isfinite(a)) and np.all((a > klo) & (a < khi))


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q", "-m", "not slow"]))
