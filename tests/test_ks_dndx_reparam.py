"""KS-leg LLS prior reparameterization in dN/dX space (V-A, KS-ONLY) — W2 TDD battery.

THE CHANGE (signed PI decision 1, executed structurally 2026-07-22): the deployed KS
alpha-space prior (pivot TruncatedNormals × power-law z-shape, LLS boost 2.5 POST-map) put
~55% of its mass outside the occupancy simplex (sum alpha > 1 ⇒ a negative clean-sightline
coefficient) and its CENTRE crossed sum(alpha)=1 at z≈4.49. survey="KS" single-leg builds now
sample 6 sites (eps_lls, kappa_lls, m_sub, t_sub, dla_raw, t_dla) around a deterministic
dN/dX reference — the KS 2.5× acts in dN/dX space PRE-map on the lit LLS law — and push them
through the exact occupancy map ``w_c_corrected``, so every draw is STRUCTURALLY inside the
simplex. DESI / eBOSS / DESI+KS / survey=None are byte-unchanged; the OLD KS prior stays
invokable via ``build_legb_ctx(..., ks_legacy_alpha_param=True)`` (the matched paired arm R6).

Battery (the W2 brief's items 1-8):
  1. reference-curve pins (orchestrator-pinned rows, through the ctx path);
  2. byte-safety: DESI/eBOSS/None prior vectors bit-pinned; the KS legacy override == OLD;
  3. structural simplex bound at jointly-extreme draws;
  4. twin parity (site names+order+values) + reconstruct == model deterministics;
  5. induced-geometry MC vs the deployed KS prior (numbers in the docstring);
  6. payload/signature moved + forward_stamp carries hcd_parameterization;
  7. guards: forbidden combos, mapped pivot band, assert_known_survey untouched;
  8. model smoke on the REAL production KS ctx (log-density + grad finite; NO NUTS).

Env (MANDATORY):
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
    /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_ks_dndx_reparam.py -q
"""
import itertools
import os
import time

import numpy as np
import pytest

import hcd_analysis.emulator  # x64 BEFORE jax
import jax
import jax.numpy as jnp
from numpyro import handlers

from hcd_analysis.emulator import closure_legb as C
from hcd_analysis.emulator import inference as INF
from hcd_analysis.emulator.dndx_wc import w_c_corrected

_CACHE = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/observables_tau0_lf.h5"
_CKPT0 = "/home/mfho/hcd_priya/checkpoints/final_fold0.eqx"
_DESI_NPZ = "/home/mfho/data/desi_dr1_p1d/desi_dr1_p1d.npz"
_KS = ("/home/mfho/lya_emulator_full/lyaemu/data/kodiaq_squad/"
       "final-conservative-p1d-karacayli_etal2021.txt")
_MANIFEST = "/home/mfho/hcd_priya/checkpoints/production_ensemble_manifest.json"
_have = all(os.path.exists(p) for p in (_CACHE, _CKPT0, _DESI_NPZ, _KS))
_have_prod = _have and os.path.exists(_MANIFEST)

# The PRE-W2 signature literal (recorded; the W2 payload additions MOVE it — intended).
OLD_HCD_PRIOR_SIGNATURE = "bba3da8868fa498e50bc69f3ff85c192be1fae45058af8ad1211cc28a53102ae"

# ORCHESTRATOR-PINNED V-A KS reference alpha (LLS, subDLA, DLA) at sites=0 (6-dp literals;
# atol 5e-7 covers the literal quantization, rtol 1e-4 is the brief's tolerance).
REF_PINS = {
    2.4: (0.232049, 0.039804, 0.003008),
    3.0: (0.393236, 0.062419, 0.004426),
    4.2: (0.697056, 0.128396, 0.008215),
    4.6: (0.745217, 0.157059, 0.009761),
}

# BIT-LEVEL byte-safety pins of _survey_alpha_prior outputs (captured on THIS tree; these are
# the deployed alpha-space prior vectors the reparam must NOT move — incl. the KS row, which
# the mapped branch keeps as the legacy/audit alpha-space prior).
SURVEY_PRIOR_PINS = {
    None:    (np.array([0.18811862702546295, 0.06218315972222222, 0.004409670138888889]),
              np.array([0.02821779405381944, 0.02487326388888889, 0.0022048350694444446]),
              None),
    "DESI":  (np.array([0.17211216067355312, 0.06218315972222222, 0.004409670138888889]),
              np.array([0.04939619011330974, 0.02487326388888889, 0.0022048350694444446]),
              np.array([2.127, 2.758, 2.366])),
    "eBOSS": (np.array([0.17211216067355312, 0.06218315972222222, 0.004409670138888889]),
              np.array([0.04939619011330974, 0.02487326388888889, 0.0022048350694444446]),
              np.array([2.127, 2.758, 2.366])),
    "KS":    (np.array([0.4302804016838828, 0.06218315972222222, 0.004409670138888889]),
              np.array([0.17211216067355314, 0.02487326388888889, 0.0022048350694444446]),
              np.array([2.127, 2.758, 2.366])),
}

MAPPED_SITES = ["eps_lls", "kappa_lls", "m_sub", "t_sub", "dla_raw", "t_dla"]
LEGACY_KS_SITES = ["theta_unit", "tau0_amp", "dtau0",
                   "alpha_lls", "alpha_subdla", "alpha_dla_raw",
                   "s_lls", "s_subdla", "s_dla",
                   "alpha_res", "alpha_res_slope"]
MAPPED_KS_SITES = ["theta_unit", "tau0_amp", "dtau0"] + MAPPED_SITES + \
                  ["alpha_res", "alpha_res_slope"]


@pytest.fixture(scope="session")
def ks_setup():
    """FULL-grid mapped KS ctx (default cuts so z_global carries the pinned rows up to 4.6)."""
    ctx, d = C.build_legb_ctx(use_xclass=True, survey="KS")
    core = {leg.name: jnp.asarray(np.asarray(ctx.dla_core_leg[leg.name]).mean(axis=0))
            for leg in ctx.legs}
    return ctx, d, core


@pytest.fixture(scope="session")
def ks_legacy_setup():
    """The R6 legacy-override KS ctx (OLD alpha-space parameterization, byte-identical)."""
    ctx, d = C.build_legb_ctx(use_xclass=True, survey="KS", ks_legacy_alpha_param=True)
    core = {leg.name: jnp.asarray(np.asarray(ctx.dla_core_leg[leg.name]).mean(axis=0))
            for leg in ctx.legs}
    return ctx, d, core


def _sites(model, *a, seed=1):
    tr = handlers.trace(handlers.seed(model, jax.random.PRNGKey(seed))).get_trace(*a)
    return [k for k, v in tr.items() if v["type"] == "sample" and not v.get("is_observed")]


def _trace(model, *a, seed=1):
    return handlers.trace(handlers.seed(model, jax.random.PRNGKey(seed))).get_trace(*a)


# --------------------------------------------------------------------------- #
#  1. Reference-curve pins (through the ctx path).
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI/KS not present")
def test_reference_curve_pins(ks_setup):
    """The mapped KS reference alpha (sites at 0) reproduces the orchestrator-pinned rows,
    computed THROUGH the ctx fields (ks_dndx_ref/ks_xbar_z built at ctx build)."""
    ctx, d, _ = ks_setup
    assert ctx.ks_dndx_mapped is True
    zg = np.asarray(ctx.z_global)
    a_ref = np.asarray(w_c_corrected(jnp.asarray(ctx.ks_dndx_ref),
                                     jnp.asarray(ctx.ks_xbar_z), jnp.asarray(zg)))[..., 1:]
    for zt, pin in REF_PINS.items():
        rows = np.isclose(zg, zt, atol=1e-6)
        assert rows.sum() == 1, f"pinned z={zt} not on the KS build z_global {zg}"
        np.testing.assert_allclose(
            a_ref[rows][0], np.asarray(pin), rtol=1e-4, atol=5e-7,
            err_msg=f"mapped KS reference alpha at z={zt} drifted from the pinned row")
    # the pivot fields go through the identical construction (z=3 row == pivot map output)
    a_piv = np.asarray(w_c_corrected(jnp.asarray(ctx.ks_dndx_ref_pivot),
                                     jnp.asarray(float(ctx.ks_xbar_pivot)),
                                     jnp.asarray(3.0)))[..., 1:]
    np.testing.assert_allclose(a_piv, np.asarray(REF_PINS[3.0]), rtol=1e-4, atol=5e-7)
    # documented centre moves: KS LLS pivot 0.4302804 (legacy) -> 0.393236 (mapped);
    # deployed subDLA pivot preserved to +0.38%.
    assert a_piv[0] == pytest.approx(0.393236, abs=5e-6)
    assert a_piv[1] / 0.06218315972222222 == pytest.approx(1.0038, abs=2e-4)


@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI/KS not present")
def test_reference_lls_component_is_boosted_lit_law(ks_setup):
    """dndx_ref LLS column == 2.5 x A_LLS x (1+z)^gamma_LLS exactly (dndx_premap boost);
    sub/DLA columns come from the exact inverse of the boost-1.0 triple (finite, positive)."""
    ctx, d, _ = ks_setup
    zg = np.asarray(ctx.z_global)
    A, g = INF.HCD_LIT_DNDX_LAW["LLS"]
    want = INF.HCD_LLS_SURVEY_BOOST["KS"] * A * (1.0 + zg) ** g
    np.testing.assert_allclose(np.asarray(ctx.ks_dndx_ref)[:, 0], want, rtol=0, atol=1e-15)
    ref = np.asarray(ctx.ks_dndx_ref)
    assert np.isfinite(ref).all() and (ref > 0).all()


# --------------------------------------------------------------------------- #
#  2. Byte-safety: DESI/eBOSS/None prior vectors + the KS legacy override.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI/KS not present")
def test_survey_alpha_prior_vectors_bit_unchanged(ks_setup):
    """_survey_alpha_prior outputs are BIT-EQUAL to the pre-W2 pins for every survey key
    (incl. KS: the alpha-space prior vector is retained for the legacy override/audit; the
    mapped sites never read it)."""
    _, d, _ = ks_setup
    w_c_med, Xbar_z3 = C.hcd_pivot_wc_and_xbar(d)
    for sv, (mu_pin, sd_pin, zmu_pin) in SURVEY_PRIOR_PINS.items():
        mu, sd, zmu = C._survey_alpha_prior(w_c_med, Xbar_z3, sv)
        np.testing.assert_array_equal(np.asarray(mu), mu_pin,
                                      err_msg=f"{sv} prior CENTER moved (bit-level)")
        np.testing.assert_array_equal(np.asarray(sd), sd_pin,
                                      err_msg=f"{sv} prior WIDTH moved (bit-level)")
        if zmu_pin is None:
            assert zmu is None
        else:
            np.testing.assert_array_equal(np.asarray(zmu), zmu_pin)


@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI/KS not present")
def test_non_ks_builds_have_no_mapped_fields():
    """DESI (narrow, cheap) and survey=None builds keep the dormant defaults — the mapped
    branch cannot activate off-KS (byte-safety by the static dispatch)."""
    for sv in ("DESI", None):
        ctx, _ = C.build_legb_ctx(desi_kwargs=dict(z_lo=0.0, z_hi=2.6), use_xclass=True,
                                  survey=sv)
        assert ctx.ks_dndx_mapped is False
        assert ctx.ks_dndx_ref is None and ctx.ks_xbar_z is None
        assert ctx.ks_dndx_ref_pivot is None and ctx.ks_xbar_pivot is None


@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI/KS not present")
def test_ks_legacy_override_reproduces_old_prior(ks_legacy_setup, ks_setup):
    """ks_legacy_alpha_param=True: the OLD KS parameterization exactly — legacy site list,
    alpha_lls TruncatedNormal at the pinned boosted centre/width, no mapped fields; and the
    ctx-level alpha_hcd_mu/sigma are IDENTICAL between the legacy and mapped builds (the
    mapped branch does not touch the stored alpha-space prior)."""
    ctx_l, _, core_l = ks_legacy_setup
    ctx_m, _, _ = ks_setup
    assert ctx_l.ks_dndx_mapped is False and ctx_l.ks_dndx_ref is None
    s = _sites(C._legb_model, ctx_l, ctx_l.legs, core_l)
    assert s == LEGACY_KS_SITES, s
    tr = _trace(C._legb_model, ctx_l, ctx_l.legs, core_l)
    fn = tr["alpha_lls"]["fn"]
    mu_pin, sd_pin, _ = SURVEY_PRIOR_PINS["KS"]
    # TruncatedNormal exposes loc/scale on the base_dist
    loc = float(getattr(fn, "base_dist", fn).loc)
    scale = float(getattr(fn, "base_dist", fn).scale)
    assert loc == mu_pin[0] and scale == sd_pin[0]
    np.testing.assert_array_equal(np.asarray(ctx_l.alpha_hcd_mu), np.asarray(ctx_m.alpha_hcd_mu))
    np.testing.assert_array_equal(np.asarray(ctx_l.alpha_hcd_sigma),
                                  np.asarray(ctx_m.alpha_hcd_sigma))


# --------------------------------------------------------------------------- #
#  3. Structural simplex bound at jointly-extreme draws.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI/KS not present")
def test_structural_bound_extreme_draws(ks_setup):
    """All 64 sign combinations of jointly-extreme sites (eps ±8σ, kappa ±6σ, m_sub ±8σ,
    t_sub ±6σ, dla_raw ±8 latent-σ, t_dla ±6σ) keep sum(alpha_hcd_z) < 1 at EVERY z_global
    row and alpha >= 0 (float ties: sum == 1.0 with clean coef exactly 0.0 acceptable,
    never negative)."""
    ctx, _, _ = ks_setup
    ext = dict(eps_lls=8.0 * INF.KS_DNDX_SIGMA_EPS,
               kappa_lls=6.0 * INF.KS_DNDX_SIGMA_KAPPA,
               m_sub=8.0 * INF.KS_DNDX_SIGMA_MSUB,
               t_sub=6.0 * C.ZSLOPE_PRIOR_SIGMA[1],
               dla_raw=8.0,
               t_dla=6.0 * C.ZSLOPE_PRIOR_SIGMA[2])
    worst = -np.inf
    for signs in itertools.product((-1.0, 1.0), repeat=6):
        sub = {nm: jnp.asarray(sg * ext[nm] + (INF.KS_DNDX_DLA_RAW_MU0 if nm == "dla_raw"
                                               else 0.0))
               for nm, sg in zip(MAPPED_SITES, signs)}
        tr = handlers.trace(handlers.substitute(
            handlers.seed(C._legb_priors_only, jax.random.PRNGKey(0)), sub)).get_trace(ctx)
        az = np.asarray(tr["alpha_hcd_z"]["value"])
        s = az.sum(axis=1)
        assert (az >= 0.0).all(), f"negative alpha at signs={signs}"
        assert (s <= 1.0).all() and not (s > 1.0).any(), \
            f"sum(alpha) > 1 at signs={signs}: max {s.max()!r}"
        assert (s < 1.0).all() or np.isclose(s.max(), 1.0), s.max()
        worst = max(worst, float(s.max()))
    print(f"[structural-bound] worst sum(alpha_hcd_z) over 64 extreme corners: {worst:.15f}")


# --------------------------------------------------------------------------- #
#  4. Twin parity + reconstruction == model deterministics.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI/KS not present")
def test_twin_parity_sites_and_values(ks_setup):
    """_legb_model and _legb_priors_only produce IDENTICAL sample-site names in IDENTICAL
    order on the mapped KS ctx, and the seeded-trace sample VALUES agree bitwise."""
    ctx, _, core = ks_setup
    s_model = _sites(C._legb_model, ctx, ctx.legs, core, seed=7)
    s_prior = _sites(C._legb_priors_only, ctx, seed=7)
    assert s_model == MAPPED_KS_SITES, s_model
    assert s_prior == MAPPED_KS_SITES, s_prior
    tr_m = _trace(C._legb_model, ctx, ctx.legs, core, seed=7)
    tr_p = _trace(C._legb_priors_only, ctx, seed=7)
    for nm in MAPPED_KS_SITES:
        np.testing.assert_array_equal(
            np.asarray(tr_m[nm]["value"]), np.asarray(tr_p[nm]["value"]),
            err_msg=f"twin sample values diverged at site {nm!r} (fixed seed)")


@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI/KS not present")
def test_reconstruct_matches_model_deterministics(ks_setup):
    """_legb_reconstruct_deterministics rebuilds alpha_lls/alpha_subdla/alpha_dla (mapped
    pivot values) + alpha_hcd_z + tau0_vec from the raw mapped sites, matching the model's
    deterministic trace (the fast-postprocess contract)."""
    ctx, _, core = ks_setup
    tr = _trace(C._legb_model, ctx, ctx.legs, core, seed=11)
    raw = {nm: jnp.asarray(tr[nm]["value"])[None] for nm in MAPPED_KS_SITES}
    rec = C._legb_reconstruct_deterministics(ctx, raw)
    for nm in ("tau0_vec", "alpha_hcd_z", "alpha_lls", "alpha_subdla", "alpha_dla"):
        np.testing.assert_allclose(
            np.asarray(rec[nm])[0], np.asarray(tr[nm]["value"]), rtol=1e-14, atol=0,
            err_msg=f"reconstructed {nm!r} deviates from the model deterministic")


@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI/KS not present")
def test_self_draw_truth_pack_on_mapped_ctx(ks_setup):
    """draw_leg_a_leg_truth works on the mapped ctx: alpha_hcd (pivot) and alpha_hcd_z come
    from the mapped reconstruct; the raw dict carries the 6 mapped sites (Wave-2 SBC plumbing)."""
    ctx, _, _ = ks_setup
    tp = C.draw_leg_a_leg_truth(ctx, jax.random.PRNGKey(5))
    assert set(MAPPED_SITES) <= set(tp["raw"])
    assert tp["alpha_hcd"].shape == (3,) and np.isfinite(tp["alpha_hcd"]).all()
    az = np.asarray(tp["alpha_hcd_z"])
    assert az.shape == (len(ctx.z_global), 3)
    assert (az >= 0).all() and (az.sum(axis=1) < 1.0).all()


# --------------------------------------------------------------------------- #
#  5. Induced-geometry sanity (MC).
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI/KS not present")
def test_induced_geometry_vs_deployed_prior(ks_setup):
    """MC (N=2e4, fixed seed) induced alpha_LLS PIVOT quantiles, mapped vs deployed KS prior.

    MEASURED (seed 20260722, recorded for the PI packet):
      mapped  alpha_LLS pivot quantiles (5,16,50,84,95)% = 0.1913 0.2562 0.3904 0.5613 0.6778
      deployed TruncatedNormal(0.43028, 0.17211, low=0)  = 0.1566 0.2628 0.4316 0.6021 0.7139
    Documented direction: the mapped median (~0.39) sits BELOW the deployed centre (~0.43)
    — the dndx-premap 2.5x through the saturating occupancy map lands lower than the legacy
    post-map multiply; every mapped draw is inside the simplex (frac sum>1 = 0.0)."""
    ctx, _, _ = ks_setup
    rng = np.random.default_rng(20260722)
    N = 20000
    eps = rng.normal(0.0, INF.KS_DNDX_SIGMA_EPS, N)
    msub = rng.normal(0.0, INF.KS_DNDX_SIGMA_MSUB, N)
    draw = rng.normal(INF.KS_DNDX_DLA_RAW_MU0, 1.0, N)
    dla_amp = np.log1p(np.exp(draw)) / np.log1p(np.exp(INF.KS_DNDX_DLA_RAW_MU0))
    fac = np.stack([np.exp(eps), np.exp(msub), dla_amp], axis=-1)
    dndx = np.asarray(ctx.ks_dndx_ref_pivot)[None, :] * fac
    a = np.asarray(w_c_corrected(jnp.asarray(dndx),
                                 jnp.asarray(np.full(N, float(ctx.ks_xbar_pivot))),
                                 jnp.asarray(np.full(N, 3.0))))[:, 1:]
    q = np.quantile(a[:, 0], [0.05, 0.16, 0.5, 0.84, 0.95])
    np.testing.assert_allclose(q, [0.1913, 0.2562, 0.3904, 0.5613, 0.6778], atol=2e-3)
    from scipy.stats import truncnorm
    mu_pin, sd_pin, _ = SURVEY_PRIOR_PINS["KS"]
    loc, sc = float(mu_pin[0]), float(sd_pin[0])
    tn = truncnorm(a=(0.0 - loc) / sc, b=np.inf, loc=loc, scale=sc)
    qd = tn.ppf([0.05, 0.16, 0.5, 0.84, 0.95])
    assert q[2] < qd[2], "mapped median must sit below the deployed KS centre (~0.39 vs ~0.43)"
    assert q[2] == pytest.approx(0.39, abs=0.01)
    assert qd[2] == pytest.approx(0.4316, abs=1e-3)
    # every mapped draw inside the simplex (the point of the reparameterization)
    assert float((a.sum(axis=1) >= 1.0).mean()) == 0.0


# --------------------------------------------------------------------------- #
#  6. Payload / signature / forward_stamp.
# --------------------------------------------------------------------------- #
def test_signature_moved_and_payload_carries_new_constants():
    sig = INF.hcd_prior_signature()
    assert sig != OLD_HCD_PRIOR_SIGNATURE, \
        "hcd_prior_signature did NOT move — the W2 payload additions are missing"
    p = INF.hcd_prior_constants_payload()
    for k in ("HCD_ALPHA_PARAMETERIZATION", "HCD_LLS_BOOST_SPACE", "KS_DNDX_SIGMA_EPS",
              "KS_DNDX_SIGMA_KAPPA", "KS_DNDX_SIGMA_MSUB", "KS_DNDX_WIDTH_CONVENTION",
              "KS_HEADROOM_SOURCE", "KS_DNDX_DLA_RAW_MU0", "KS_DNDX_MAPPED_PIVOT_BAND"):
        assert k in p, f"payload missing the new constant {k!r}"
    assert p["HCD_ALPHA_PARAMETERIZATION"]["KS"] == "dndx_mapped_v2"
    assert p["HCD_LLS_BOOST_SPACE"]["KS"] == "dndx_premap"
    assert p["KS_DNDX_SIGMA_EPS"] == 0.5310
    assert p["KS_DNDX_SIGMA_KAPPA"] == 0.6681
    assert p["KS_DNDX_SIGMA_MSUB"] == 0.4130
    assert p["KS_HEADROOM_SOURCE"] == \
        "exact-inverse of the boost-1.0 deployed real-fit centre triple"
    assert set(p["KS_DNDX_WIDTH_CONVENTION"]) == set(MAPPED_SITES)
    # JSON round-trip stability (the signature is canonical-JSON over this payload)
    import json
    json.dumps(p, sort_keys=True)


def test_forward_stamp_carries_hcd_parameterization():
    """forward_stamp resolves hcd_parameterization from the BUILT ctx (getattr default keeps
    pre-feature stubs buildable): mapped KS -> dndx_mapped_v2, everything else -> the legacy
    id. Adding the key makes analyze_dnuis_bias pool pre-land vs post-land shards fail-loud
    (intended freeze-handoff behaviour: regenerate the group)."""
    from types import SimpleNamespace
    stub = dict(res_corr_on=False, fix_alpha_res=True, sample_res=True,
                f_res_amp_sigma=0.15, metal_prior="uniform", metal_node_z=(2.2, 4.2))
    leg = SimpleNamespace(dla_cov_reduced=False, dla_forward_frac=0.0, use_snr3=False,
                          cv_floor_on=False, cv_floor_rank1=False)
    st_old = C.forward_stamp(SimpleNamespace(**stub), leg)          # no ks_dndx_mapped attr
    assert st_old["hcd_parameterization"] == "alpha_pivot_powerlaw_v1"
    st_new = C.forward_stamp(SimpleNamespace(ks_dndx_mapped=True, **stub), leg)
    assert st_new["hcd_parameterization"] == "dndx_mapped_v2"


# --------------------------------------------------------------------------- #
#  7. Guards.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI/KS not present")
def test_forbidden_combinations_raise(ks_setup):
    with pytest.raises(ValueError, match="incompatible"):
        C.build_legb_ctx(desi_kwargs=dict(z_lo=0.0, z_hi=2.6), use_xclass=True,
                         survey="KS", hierarchical_hcd=True)
    with pytest.raises(ValueError, match="incompatible"):
        C.build_legb_ctx(desi_kwargs=dict(z_lo=0.0, z_hi=2.6), use_xclass=True,
                         survey="KS", hierarchical_hcd=True, hcd_2d_tilt=True)
    # a hand-built forbidden ctx (ctx._replace) fails loudly at trace time too
    ctx, _, _ = ks_setup
    for bad in (dict(per_leg_alpha=True), dict(hierarchical_hcd=True),
                dict(hcd_2d_tilt=True)):
        ctx_bad = ctx._replace(**bad)
        with pytest.raises(ValueError, match="incompatible"):
            _sites(C._legb_priors_only, ctx_bad)


def test_mapped_pivot_band_guard():
    """assert_ks_mapped_pivot: 0.393236 passes; the legacy-band edges and the ~0.727
    pre-map-equivalent of an all-z-median-based centre fail."""
    INF.assert_ks_mapped_pivot(0.393236, "test")
    for bad in (0.30, 0.50, 0.727, 0.4302804 * 2.5):
        with pytest.raises(AssertionError, match="mapped band"):
            INF.assert_ks_mapped_pivot(bad, "test")
    # band literal + derivation pin (0.83/1.21 relative margins around the mapped centre)
    assert INF.KS_DNDX_MAPPED_PIVOT_BAND == (0.326, 0.476)


def test_assert_known_survey_untouched():
    INF.assert_known_survey("KS", "w2-test")
    INF.assert_known_survey("DESI", "w2-test")
    with pytest.raises(AssertionError, match="valid"):
        INF.assert_known_survey("KODIAQ", "w2-test")


# --------------------------------------------------------------------------- #
#  8. Model smoke on the REAL production KS ctx (NO NUTS).
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _have_prod, reason="production ensemble manifest/data not present")
def test_model_smoke_real_ks_ctx():
    """Build the real KS ctx EXACTLY as run_real_fit.build_real_ctx('ks') does (manifest-
    pinned ensemble, NORC, KS f_res float), draw 20 prior samples through
    _legb_priors_only/Predictive, and evaluate the _legb_model log-density AND its jax.grad
    at each — all finite, no NaN. Wall time printed for the W2 report."""
    from scripts.run_real_fit import build_real_ctx
    from numpyro.infer import Predictive
    from numpyro.infer.util import log_density

    t0 = time.time()
    ctx, d, members = build_real_ctx("ks")
    t_build = time.time() - t0
    assert ctx.ks_dndx_mapped is True
    leg = ctx.legs[0]
    assert leg.name == "KS"
    fid = np.asarray(ctx.dla_core_leg[leg.name])
    core = {leg.name: jnp.asarray(fid.mean(axis=0) if fid.ndim == 2 else fid)}

    pred = Predictive(lambda: C._legb_priors_only(ctx), num_samples=20)
    draws = pred(jax.random.PRNGKey(20260722))
    # restrict to the SAMPLE sites (Predictive also returns the deterministics the mapped
    # branch emits; log_density params must be the latent sites only)
    snames = _sites(C._legb_priors_only, ctx)
    assert set(MAPPED_SITES) <= set(snames)

    def logdens(params):
        ld, _ = log_density(lambda: C._legb_model(ctx, ctx.legs, core), (), {}, params)
        return ld

    t1 = time.time()
    n_bad = 0
    for i in range(20):
        params = {k: jnp.asarray(draws[k])[i] for k in snames}
        val = logdens(params)
        grads = jax.grad(logdens)(params)
        ok = np.isfinite(float(val)) and all(
            np.isfinite(np.asarray(g)).all() for g in jax.tree_util.tree_leaves(grads))
        n_bad += (not ok)
    t_eval = time.time() - t1
    assert n_bad == 0, f"{n_bad}/20 prior draws gave non-finite log-density or gradient"
    print(f"[model-smoke] real KS ctx build {t_build:.1f}s; 20x (logp+grad) {t_eval:.1f}s "
          f"({t_eval/20:.2f}s/draw); members={len(members)}")


if __name__ == "__main__":
    test_signature_moved_and_payload_carries_new_constants()
    test_forward_stamp_carries_hcd_parameterization()
    test_mapped_pivot_band_guard()
    test_assert_known_survey_untouched()
    print("[ks-dndx-reparam] constant-level checks OK (run pytest for the ctx-backed battery).")
