"""Per-leg HCD alpha sites (joint-fit capability, PI decision 2c) — TDD battery.

Spec of record: hcd_priya_notes/docs/superpowers/2026-07-20-per-leg-alpha-sites-spec.md
(Sec 1 items 1-12, Sec 5 TDD order). The capability lands DORMANT (per_leg_alpha=False
default); every deployed single-leg path must stay BYTE-IDENTICAL. This file carries:

  T1  seeded-trace golden (site name order + bitwise values, both twins, deployed DESI
      config) vs tests/golden/legb_sites_golden.npz — generated at the PRE-FEATURE tree
      by scripts/make_legb_sites_golden.py (generating commit recorded in the npz).
  T2  short-NUTS bitwise golden (5+5 draws, mtd 5, fixed key) vs
      tests/golden/legb_shortnuts_golden.npz (env-pinned; regeneration protocol in the
      generator script).
  T3  priors-only mirror parity under per_leg_alpha=True (and per_leg_zslope=True).
  T4  joint trace has exactly the suffixed names in ctx.legs order and NO unsuffixed.
  T5  1-leg joint == single-leg equivalence modulo the name map (bitwise target; if not
      bitwise the mechanism is identified and pinned — see the test docstring).
  T6  per-leg prior dicts bitwise == each build_legb_ctx(survey=L) prior (extraction
      drift tripwire for _survey_alpha_prior).
  T7  every forbidden-combination guard raises.
  T8  both signature literals unmoved (the FIRST test written; green on the parent tree).
  T9  joint_stamp contents + mixed-signature rejection + the DLA prior-only label.
  T10 joint deterministic reconstruction == full-model replay.
  T11 _draws_matrix/_packed_names_for presence-keyed back-compat.
  T12 dict-loglik == sum of per-leg-array logliks (threading cross-check).

Env (MANDATORY):
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
    /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_legb_per_leg_alpha.py -q
"""
import os
import numpy as np
import pytest

import hcd_analysis.emulator  # x64 BEFORE jax
import jax
import jax.numpy as jnp
from numpyro import handlers

from hcd_analysis.emulator import closure_legb as C
from hcd_analysis.emulator import inference as INF

_CACHE = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/observables_tau0_lf.h5"
_CKPT0 = "/home/mfho/hcd_priya/checkpoints/final_fold0.eqx"
_DESI_NPZ = "/home/mfho/data/desi_dr1_p1d/desi_dr1_p1d.npz"
_KS = ("/home/mfho/lya_emulator_full/lyaemu/data/kodiaq_squad/"
       "final-conservative-p1d-karacayli_etal2021.txt")
_have = all(os.path.exists(p) for p in (_CACHE, _CKPT0, _DESI_NPZ))
_have_ks = os.path.exists(_KS)

_GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden")
SITES_GOLDEN = os.path.join(_GOLDEN_DIR, "legb_sites_golden.npz")
NUTS_GOLDEN = os.path.join(_GOLDEN_DIR, "legb_shortnuts_golden.npz")


# --------------------------------------------------------------------------- #
#  T8 (written FIRST, choreography step 1): both freeze-audit signature literals
#  are PINNED EXACTLY. This build must not touch either payload — any drift in
#  PROD_FORWARD_BY_LEG / PROD_RES_CORR_ON / DESI_DLA_COV_REDUCE or in the HCD
#  prior-constants payload flips a hex and fails here. Forward literal read from
#  the UNTOUCHED tree at HEAD 54db658 (2026-07-20). The HCD prior literal was
#  re-pinned by W2 (KS dN/dX-mapped reparam, 2026-07-22): the payload additions
#  (HCD_ALPHA_PARAMETERIZATION / HCD_LLS_BOOST_SPACE / KS_DNDX_*) INTENTIONALLY
#  moved hcd_prior_signature. Old value (recorded in the W2 commit message +
#  tests/test_ks_dndx_reparam.OLD_HCD_PRIOR_SIGNATURE):
#  bba3da8868fa498e50bc69f3ff85c192be1fae45058af8ad1211cc28a53102ae
# --------------------------------------------------------------------------- #
FORWARD_SIGNATURE_PIN = "68f71a3d45d6e49f036c03e46a5f7953fbc932d943c7d2479746cfe08d1af09b"
HCD_PRIOR_SIGNATURE_PIN = "50befc941edfc4c789286d2054d0107f1eb19a5672bcb86131426fb101eea216"


def test_signatures_unmoved():
    """forward_signature() and hcd_prior_signature() equal the pinned literals (the per-leg
    alpha build extended STAMPS only; the ONE sanctioned hcd_prior move since is the W2 KS
    reparam re-pin above — any further drift fails here)."""
    assert C.forward_signature() == FORWARD_SIGNATURE_PIN, (
        "forward_signature moved — neither the per-leg alpha build nor the W2 KS reparam "
        "may edit the forward decision payload (PROD_FORWARD_BY_LEG / PROD_RES_CORR_ON / "
        "DESI_DLA_COV_REDUCE)")
    assert INF.hcd_prior_signature() == HCD_PRIOR_SIGNATURE_PIN, (
        "hcd_prior_signature moved off the W2-pinned literal — an HCD prior constant "
        "changed without a sanctioned re-pin (record old->new in the commit message and "
        "update this pin only with a signed decision)")


# --------------------------------------------------------------------------- #
#  Golden fixtures — ctxs/mocks built ONCE per session through the GENERATOR's
#  own builders (scripts/make_legb_sites_golden), so the test config can never
#  drift from the config the goldens were generated with.
# --------------------------------------------------------------------------- #
from scripts import make_legb_sites_golden as G


@pytest.fixture(scope="session")
def trace_setup():
    ctx, d = G.build_trace_ctx()
    mock_legs, core = G.mock_for(ctx, d)
    return ctx, d, mock_legs, core


@pytest.fixture(scope="session")
def nuts_setup():
    ctx, d = G.build_nuts_ctx()
    mock_legs, core = G.mock_for(ctx, d)
    return ctx, d, mock_legs, core


# --------------------------------------------------------------------------- #
#  T1 — seeded-trace golden: site NAME ORDER + bitwise values, both twins,
#  deployed DESI config. Stays green through every source edit (the dormant
#  default must be byte-identical).
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not (_have and _have_ks), reason="real cache/ckpt/DESI/KS not present")
@pytest.mark.skipif(not os.path.exists(SITES_GOLDEN), reason="sites golden not generated")
def test_trace_matches_sites_golden(trace_setup):
    """Both twins reproduce the pre-feature seeded trace EXACTLY: same site names in the
    same order, bitwise-equal sample/deterministic values, bitwise-equal loglik factor."""
    ctx, d, mock_legs, core = trace_setup
    g = np.load(SITES_GOLDEN, allow_pickle=True)
    model_rec, model_factor, prior_rec = G.trace_records(ctx, mock_legs, core)

    assert [n for n, _, _ in model_rec] == list(g["model_names"]), \
        "_legb_model site NAME ORDER drifted vs the pre-feature golden"
    assert [k for _, k, _ in model_rec] == list(g["model_kinds"])
    assert [n for n, _, _ in prior_rec] == list(g["prior_names"]), \
        "_legb_priors_only site NAME ORDER drifted vs the pre-feature golden"
    for n, _, v in model_rec:
        np.testing.assert_array_equal(v, g[f"model::{n}"],
                                      err_msg=f"model site {n!r} value drifted (bitwise)")
    for n, _, v in prior_rec:
        np.testing.assert_array_equal(v, g[f"prior::{n}"],
                                      err_msg=f"priors-only site {n!r} value drifted (bitwise)")
    np.testing.assert_array_equal(np.asarray(model_factor), g["loglik_factor"],
                                  err_msg="loglik factor value drifted (bitwise)")


# --------------------------------------------------------------------------- #
#  T2 — short-NUTS bitwise golden (5+5, mtd 5, fixed key): catches mass-ordering /
#  flatten-order effects a single trace cannot. Env-pinned (regeneration protocol
#  in scripts/make_legb_sites_golden.py).
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not (_have and _have_ks), reason="real cache/ckpt/DESI/KS not present")
@pytest.mark.skipif(not os.path.exists(NUTS_GOLDEN), reason="short-NUTS golden not generated")
def test_short_nuts_matches_golden(nuts_setup):
    """The deployed single-leg 5+5 NUTS run reproduces the pre-feature samples dict
    BITWISE (every key, every draw)."""
    ctx, d, mock_legs, core = nuts_setup
    g = np.load(NUTS_GOLDEN, allow_pickle=True)
    samples, n_div = G.run_short_nuts(ctx, mock_legs, core)
    assert list(samples.keys()) == list(g["sample_keys"]), \
        "samples-dict key set/order drifted vs the pre-feature golden"
    assert int(n_div) == int(g["n_div"])
    for k in samples:
        np.testing.assert_array_equal(np.asarray(samples[k]), g[f"nuts::{k}"],
                                      err_msg=f"NUTS draws for {k!r} drifted (bitwise)")


# --------------------------------------------------------------------------- #
#  Joint-ctx fixtures (narrow-z DESI legs where possible — the prior math is
#  z-cut independent, so the cheap builds prove the same identities).
# --------------------------------------------------------------------------- #
_NARROW = dict(z_lo=0.0, z_hi=2.6)


@pytest.fixture(scope="session")
def joint2_setup():
    """DESI+KS joint ctx (default flags: no metals, no f_res) + a deterministic mock."""
    ctx, d = C.build_legb_joint_ctx({"DESI": "DESI", "KS": "KS"},
                                    desi_kwargs=dict(_NARROW), use_xclass=True)
    mock_legs, core = G.mock_for(ctx, d)
    return ctx, d, mock_legs, core


@pytest.fixture(scope="session")
def joint1_setup():
    """The 1-leg joint ctx (PI C7: allowed — the T5 rename-equivalence lever), built with
    EXACTLY the scripts/make_legb_sites_golden.build_nuts_ctx flag set so the single-leg
    reference is the pre-feature short-NUTS golden itself."""
    ctx, d = C.build_legb_joint_ctx(
        {"DESI": "DESI"}, desi_kwargs=dict(_NARROW), use_xclass=True,
        metals_on=True, sample_metals=True, sample_res=True, f_res_amp_sigma=0.02,
        metal_prior="flatlog2node")
    mock_legs, core = G.mock_for(ctx, d)
    return ctx, d, mock_legs, core


@pytest.fixture(scope="session")
def ks_single_setup():
    """A single-leg survey='KS' build (prior reference for T6; ctx only)."""
    ctx, d = C.build_legb_ctx(desi_kwargs=dict(_NARROW), use_xclass=True, survey="KS")
    return ctx, d


def _sites(model, *a, seed=1):
    tr = handlers.trace(handlers.seed(model, jax.random.PRNGKey(seed))).get_trace(*a)
    return [k for k, v in tr.items() if v["type"] == "sample" and not v.get("is_observed")]


def _trace(model, *a, seed=1):
    return handlers.trace(handlers.seed(model, jax.random.PRNGKey(seed))).get_trace(*a)


# --------------------------------------------------------------------------- #
#  Dormant-default + builder-field tests (ctx fields, poison, survey_by_leg).
# --------------------------------------------------------------------------- #
def test_ctx_fields_default_off():
    """The new LegBCtx fields exist, are APPENDED after res_corr_on, and default OFF/None
    (dormant static branch — positional construction must not shift)."""
    f = C.LegBCtx._fields
    i = f.index("res_corr_on")
    assert f[i + 1:i + 6] == ("per_leg_alpha", "alpha_hcd_mu_by_leg",
                              "alpha_hcd_sigma_by_leg", "survey_by_leg", "per_leg_zslope")
    d = C.LegBCtx._field_defaults
    assert d["per_leg_alpha"] is False and d["per_leg_zslope"] is False
    assert d["alpha_hcd_mu_by_leg"] is None and d["alpha_hcd_sigma_by_leg"] is None
    assert d["survey_by_leg"] is None


@pytest.mark.skipif(not (_have and _have_ks), reason="real cache/ckpt/DESI/KS not present")
def test_joint_builder_fields_and_poison(joint2_setup):
    """build_legb_joint_ctx sets the per-leg fields, POISONS the shared alpha_hcd_mu/sigma
    to None (any legacy consumer fails loud), and restricts legs in build order."""
    ctx, d, _, _ = joint2_setup
    assert ctx.per_leg_alpha is True and ctx.per_leg_zslope is False
    assert ctx.alpha_hcd_mu is None and ctx.alpha_hcd_sigma is None      # POISON
    assert [l.name for l in ctx.legs] == ["DESI", "KS"]                  # build (list) order
    assert ctx.survey_by_leg == {"DESI": "DESI", "KS": "KS"}
    assert set(ctx.alpha_hcd_mu_by_leg) == {"DESI", "KS"}
    assert set(ctx.alpha_hcd_sigma_by_leg) == {"DESI", "KS"}
    # zslope center = the litWLS real-fit vector (survey-independent across legs).
    zmu = np.asarray(ctx.zslope_mu)
    assert zmu[0] == pytest.approx(INF.HCD_LLS_REALFIT_ZSLOPE)
    np.testing.assert_allclose(zmu[1:], np.asarray(C.HCD_INCIDENCE_SLOPE)[1:])


# --------------------------------------------------------------------------- #
#  T6 — per-leg prior dicts bitwise == each build_legb_ctx(survey=L) prior
#  (the _survey_alpha_prior extraction-drift tripwire).
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not (_have and _have_ks), reason="real cache/ckpt/DESI/KS not present")
def test_per_leg_priors_bitwise_match_single_leg_builds(joint2_setup, nuts_setup,
                                                        ks_single_setup):
    """Each leg's joint prior (mu, sigma) is BITWISE the prior that leg's single-leg
    build_legb_ctx(survey=L) deploys (coherence property: per-leg blind posteriors and
    the joint fit are prior-commensurable). Also the zslope center matches."""
    ctx_j, _, _, _ = joint2_setup
    ctx_desi = nuts_setup[0]                     # survey="DESI" single build
    ctx_ks = ks_single_setup[0]                  # survey="KS" single build
    for name, ref in (("DESI", ctx_desi), ("KS", ctx_ks)):
        np.testing.assert_array_equal(
            np.asarray(ctx_j.alpha_hcd_mu_by_leg[name]), np.asarray(ref.alpha_hcd_mu),
            err_msg=f"{name} joint prior CENTER != single-leg build (bitwise)")
        np.testing.assert_array_equal(
            np.asarray(ctx_j.alpha_hcd_sigma_by_leg[name]), np.asarray(ref.alpha_hcd_sigma),
            err_msg=f"{name} joint prior WIDTH != single-leg build (bitwise)")
        np.testing.assert_array_equal(np.asarray(ctx_j.zslope_mu), np.asarray(ref.zslope_mu),
                                      err_msg=f"{name} joint zslope center != single-leg")


# --------------------------------------------------------------------------- #
#  T4 — joint trace: exactly the suffixed names in ctx.legs order, NO unsuffixed.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not (_have and _have_ks), reason="real cache/ckpt/DESI/KS not present")
def test_joint_trace_site_names_and_order(joint2_setup):
    ctx, d, mock_legs, core = joint2_setup
    s = _sites(C._legb_model, ctx, mock_legs, core)
    assert s == ["theta_unit", "tau0_amp", "dtau0",
                 "alpha_lls_DESI", "alpha_subdla_DESI", "alpha_dla_raw_DESI",
                 "alpha_lls_KS", "alpha_subdla_KS", "alpha_dla_raw_KS",
                 "s_lls", "s_subdla", "s_dla",
                 "alpha_res", "alpha_res_slope"]
    tr = _trace(C._legb_model, ctx, mock_legs, core)
    all_names = set(tr)
    # NO unsuffixed alpha names anywhere (sample OR deterministic): every legacy by-name
    # reader must fail loudly with KeyError instead of silently reading one leg.
    for nm in ("alpha_lls", "alpha_subdla", "alpha_dla_raw", "alpha_dla", "alpha_hcd_z"):
        assert nm not in all_names, f"joint trace leaked the unsuffixed site {nm!r}"
    for nm in ("alpha_dla_DESI", "alpha_dla_KS", "alpha_hcd_z_DESI", "alpha_hcd_z_KS"):
        assert nm in all_names and tr[nm]["type"] == "deterministic"
    # per-z deterministic shape: (n_zg, 3) per leg.
    nzg = int(np.asarray(ctx.z_global).shape[0])
    assert np.asarray(tr["alpha_hcd_z_DESI"]["value"]).shape == (nzg, 3)


@pytest.mark.skipif(not (_have and _have_ks), reason="real cache/ckpt/DESI/KS not present")
def test_joint_trace_per_leg_zslope_switch(joint2_setup):
    """C1 plumbing: per_leg_zslope=True traces per-leg slope sites (legs order, classes
    within), with the SAME prior constants per leg; default False stays shared."""
    ctx, d, mock_legs, core = joint2_setup
    ctx_z = ctx._replace(per_leg_zslope=True)
    s = _sites(C._legb_model, ctx_z, mock_legs, core)
    assert s == ["theta_unit", "tau0_amp", "dtau0",
                 "alpha_lls_DESI", "alpha_subdla_DESI", "alpha_dla_raw_DESI",
                 "alpha_lls_KS", "alpha_subdla_KS", "alpha_dla_raw_KS",
                 "s_lls_DESI", "s_subdla_DESI", "s_dla_DESI",
                 "s_lls_KS", "s_subdla_KS", "s_dla_KS",
                 "alpha_res", "alpha_res_slope"]
    for nm in ("s_lls", "s_subdla", "s_dla"):
        assert nm not in s, f"per_leg_zslope leaked the SHARED slope site {nm!r}"
    # identical per-leg slope priors (same constants; spec Sec 2).
    tr = _trace(C._legb_model, ctx_z, mock_legs, core)
    fD, fK = tr["s_lls_DESI"]["fn"], tr["s_lls_KS"]["fn"]
    assert float(fD.loc) == float(fK.loc) and float(fD.scale) == float(fK.scale)


# --------------------------------------------------------------------------- #
#  T3 — priors-only mirror parity under per_leg_alpha=True (ZERO edits to the
#  priors-only twin: the dispatch inside _hcd_sites/_zslope_sites covers both).
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not (_have and _have_ks), reason="real cache/ckpt/DESI/KS not present")
@pytest.mark.parametrize("per_leg_zslope", [False, True])
def test_mirror_parity_per_leg(joint2_setup, per_leg_zslope):
    ctx, d, mock_legs, core = joint2_setup
    ctx = ctx._replace(per_leg_zslope=per_leg_zslope)
    s_model = _sites(C._legb_model, ctx, mock_legs, core)
    s_prior = _sites(C._legb_priors_only, ctx)
    assert s_model == s_prior, f"mirror desync: model {s_model} vs priors {s_prior}"


# --------------------------------------------------------------------------- #
#  T12 — dict-loglik == sum of per-leg-array logliks (threading cross-check)
#  + the key-set assertion fires both ways.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not (_have and _have_ks), reason="real cache/ckpt/DESI/KS not present")
def test_dict_loglik_equals_per_leg_sum(joint2_setup):
    ctx, d, mock_legs, core = joint2_setup
    zg = jnp.asarray(ctx.z_global)
    nzg = int(zg.shape[0])
    th = jnp.full(9, 0.5)
    from hcd_analysis.emulator.meanflux_prior import becker13_tau0
    tau0 = becker13_tau0(zg)
    shape_zg = ((1.0 + zg)[:, None] / (1.0 + C.HCD_Z_PIVOT)) ** jnp.asarray(C.HCD_INCIDENCE_SLOPE)
    a_dict = {"DESI": jnp.asarray([0.20, 0.07, 0.015])[None, :] * shape_zg,
              "KS": jnp.asarray([0.45, 0.09, 0.012])[None, :] * shape_zg}
    ll_dict = C._data_loglik_legcore(ctx, th, tau0, a_dict, mock_legs, core)
    ll_sum = 0.0
    for leg in mock_legs:
        ll_sum = ll_sum + C._data_loglik_legcore(
            ctx, th, tau0, a_dict[leg.name], [leg], core)
    assert float(ll_dict) == float(ll_sum), \
        f"dict loglik {float(ll_dict)!r} != per-leg-array sum {float(ll_sum)!r}"
    # key set != leg set must FAIL LOUD, both directions.
    with pytest.raises(AssertionError, match="missing"):
        C._data_loglik_legcore(ctx, th, tau0, {"DESI": a_dict["DESI"]}, mock_legs, core)
    extra = dict(a_dict); extra["eBOSS"] = a_dict["DESI"]
    with pytest.raises(AssertionError, match="extra"):
        C._data_loglik_legcore(ctx, th, tau0, extra, mock_legs, core)
    # a z-FLAT (3,) per-leg entry is the recurring z-flat bug: refused UNCONDITIONALLY in
    # the dict branch (consistency-audit hardening: under require_zresolved=False, jnp
    # fancy-indexing a (3,) with z-bin indices CLIPS silently and returns a finite wrong
    # loglik; a (3,) dict entry has no legitimate meaning).
    flat = dict(a_dict); flat["DESI"] = jnp.asarray([0.20, 0.07, 0.015])
    with pytest.raises(AssertionError):
        C._data_loglik_legcore(ctx, th, tau0, flat, mock_legs, core)
    with pytest.raises(AssertionError, match="z-RESOLVED"):
        C._data_loglik_legcore(ctx, th, tau0, flat, mock_legs, core,
                               require_zresolved=False)
    assert nzg == a_dict["DESI"].shape[0]


# --------------------------------------------------------------------------- #
#  T10 — joint deterministic reconstruction == full-model replay (fast vs slow
#  postprocess on a short joint NUTS run).
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not (_have and _have_ks), reason="real cache/ckpt/DESI/KS not present")
@pytest.mark.parametrize("per_leg_zslope", [False, True])
def test_joint_reconstruction_matches_replay(joint2_setup, per_leg_zslope):
    ctx, d, mock_legs, core = joint2_setup
    ctx = ctx._replace(per_leg_zslope=per_leg_zslope)
    kw = dict(n_warmup=5, n_samples=5, seed=13, max_tree_depth=5, dense_mass=False)
    s_fast, _ = C._run_nuts_legb(ctx, mock_legs, core, fast_postprocess=True, **kw)
    s_slow, _ = C._run_nuts_legb(ctx, mock_legs, core, fast_postprocess=False, **kw)
    for nm in ("tau0_vec", "alpha_dla_DESI", "alpha_dla_KS",
               "alpha_hcd_z_DESI", "alpha_hcd_z_KS"):
        assert nm in s_fast, f"fast postprocess must reconstruct {nm!r}"
        assert nm in s_slow, f"slow replay must yield {nm!r}"
        np.testing.assert_array_equal(
            np.asarray(s_fast[nm]), np.asarray(s_slow[nm]),
            err_msg=f"{nm}: fast reconstruction != in-model replay (bitwise)")
    for nm in ("alpha_lls", "alpha_subdla", "alpha_dla", "alpha_hcd_z"):
        assert nm not in s_fast and nm not in s_slow, \
            f"joint samples leaked the unsuffixed key {nm!r}"
    if per_leg_zslope:
        assert "s_lls_DESI" in s_fast and "s_lls_KS" in s_fast


@pytest.mark.skipif(not (_have and _have_ks), reason="real cache/ckpt/DESI/KS not present")
def test_reconstruct_rejects_shared_samples_on_per_leg_zslope_ctx(joint2_setup):
    """Consistency-audit finding H5: a per_leg_zslope=True ctx fed a SHARED-slope samples
    dict must raise (KeyError with the pairing diagnosis), never silently reconstruct
    with the shared slopes."""
    ctx, d, mock_legs, core = joint2_setup
    kw = dict(n_warmup=5, n_samples=5, seed=13, max_tree_depth=5, dense_mass=False)
    s_shared, _ = C._run_nuts_legb(ctx, mock_legs, core, fast_postprocess=True, **kw)
    assert "s_lls" in s_shared and "s_lls_DESI" not in s_shared
    with pytest.raises(KeyError, match="per_leg_zslope"):
        C._legb_reconstruct_deterministics(ctx._replace(per_leg_zslope=True), s_shared)
    # panel required-fix 2: the LEGAL per_leg_zslope=True + marginalize_zslope=False combo
    # (fixed slopes, no slope sites) must NOT raise; it reconstructs with the fixed
    # HCD_INCIDENCE_SLOPE, mirroring _zslope_sites_per_leg's dispatch.
    s_noslope = {k: v for k, v in s_shared.items() if not k.startswith("s_")}
    ctx_fix = ctx._replace(per_leg_zslope=True, marginalize_zslope=False)
    out = C._legb_reconstruct_deterministics(ctx_fix, s_noslope)
    zg = np.asarray(ctx.z_global)
    shape_fix = ((1.0 + zg)[:, None] / 4.0) ** np.asarray(C.HCD_INCIDENCE_SLOPE)
    for nm in ("DESI", "KS"):
        piv = np.stack([np.asarray(s_noslope[f"alpha_lls_{nm}"]),
                        np.asarray(s_noslope[f"alpha_subdla_{nm}"]),
                        np.asarray(out[f"alpha_dla_{nm}"])], axis=-1)
        np.testing.assert_allclose(np.asarray(out[f"alpha_hcd_z_{nm}"]),
                                   piv[:, None, :] * shape_fix[None, :, :], rtol=1e-12)


def test_joint_driver_imports():
    """The driver must import cleanly (consistency-audit gap: grep-tested only). Importing
    must not execute a fit (main guard)."""
    import importlib
    mod = importlib.import_module("scripts.run_joint_fit")
    assert hasattr(mod, "main")


# --------------------------------------------------------------------------- #
#  T5 — 1-leg joint == single-leg BITWISE modulo the name map. The single-leg
#  side is the PRE-FEATURE short-NUTS golden (same flags, same mock, same key),
#  so this is simultaneously the rename-equivalence proof and the cross-feature
#  byte certificate.
# --------------------------------------------------------------------------- #
_T5_NAME_MAP = {"alpha_lls": "alpha_lls_DESI", "alpha_subdla": "alpha_subdla_DESI",
                "alpha_dla_raw": "alpha_dla_raw_DESI", "alpha_dla": "alpha_dla_DESI",
                "alpha_hcd_z": "alpha_hcd_z_DESI"}


@pytest.mark.skipif(not (_have and _have_ks), reason="real cache/ckpt/DESI/KS not present")
@pytest.mark.skipif(not os.path.exists(NUTS_GOLDEN), reason="short-NUTS golden not generated")
def test_one_leg_joint_bitwise_rename_equivalence(joint1_setup):
    """BITWISE: the 1-leg joint NUTS run reproduces the single-leg golden draws exactly
    under the name map. This works because (a) numpyro consumes PRNG in SITE (call) order,
    which is identical, and (b) the flattened latent layout (ravel_pytree = keys sorted
    alphabetically) is POSITIONALLY identical under the _DESI suffix for this site set —
    asserted below as a precondition so a future site whose suffix REORDERS the sorted
    layout fails here with the mechanism named, not with a silent allclose."""
    ctx, d, mock_legs, core = joint1_setup
    g = np.load(NUTS_GOLDEN, allow_pickle=True)
    golden_keys = list(g["sample_keys"])
    # precondition: the name map preserves the SORTED (flatten) order of the latents.
    latents = [k for k in golden_keys if k not in ("tau0_vec", "alpha_dla", "alpha_hcd_z")]
    mapped = [_T5_NAME_MAP.get(k, k) for k in latents]
    order_single = [latents.index(k) for k in sorted(latents)]
    order_joint = [mapped.index(k) for k in sorted(mapped)]
    assert order_single == order_joint, (
        "the _DESI suffix REORDERS the sorted flatten layout — bitwise equivalence cannot "
        f"hold via ravel_pytree; single {sorted(latents)} vs joint {sorted(mapped)}")
    samples, n_div = G.run_short_nuts(ctx, mock_legs, core)
    assert list(samples.keys()) == [_T5_NAME_MAP.get(k, k) for k in golden_keys], \
        "joint samples-dict key order != name-mapped single-leg golden order"
    assert int(n_div) == int(g["n_div"])
    for k in golden_keys:
        np.testing.assert_array_equal(
            np.asarray(samples[_T5_NAME_MAP.get(k, k)]), g[f"nuts::{k}"],
            err_msg=f"T5 bitwise equivalence failed at site {k!r} (joint "
                    f"{_T5_NAME_MAP.get(k, k)!r})")


# --------------------------------------------------------------------------- #
#  T7 — every forbidden combination raises (guards are PRE-BUILD where static,
#  so these are cheap).
# --------------------------------------------------------------------------- #
def test_guard_survey_kwarg_forbidden():
    with pytest.raises(ValueError, match="survey"):
        C.build_legb_joint_ctx({"DESI": "DESI"}, survey="DESI")


def test_guard_hierarchical_forbidden():
    with pytest.raises(ValueError, match="hierarchical_hcd"):
        C.build_legb_joint_ctx({"DESI": "DESI"}, hierarchical_hcd=True)


def test_guard_2d_tilt_forbidden():
    with pytest.raises(ValueError, match="hcd_2d_tilt"):
        C.build_legb_joint_ctx({"DESI": "DESI"}, hcd_2d_tilt=True)


def test_guard_none_survey_value_forbidden():
    with pytest.raises(ValueError, match="None"):
        C.build_legb_joint_ctx({"DESI": None, "KS": "KS"})


def test_guard_empty_survey_by_leg_forbidden():
    with pytest.raises(ValueError):
        C.build_legb_joint_ctx({})


def test_guard_unknown_survey_key_fails_loud():
    with pytest.raises(AssertionError, match="valid"):
        C.build_legb_joint_ctx({"DESI": "desi"})


def test_guard_cross_survey_assignment_forbidden():
    """Panel required-fix 4: survey<->leg cross-assignment and the blended legacy key are
    rejected statically (pre-build, no data load)."""
    with pytest.raises(ValueError, match="cross-assignment"):
        C.build_legb_joint_ctx({"KS": "DESI"})
    with pytest.raises(ValueError, match="DESI\\+KS"):
        C.build_legb_joint_ctx({"DESI": "DESI+KS"})
    # the blended key is forbidden even under the deliberate-override kwarg
    with pytest.raises(ValueError, match="DESI\\+KS"):
        C.build_legb_joint_ctx({"DESI": "DESI+KS"}, allow_cross_survey=True)


def test_guard_unknown_leg_name_fails_loud():
    with pytest.raises(ValueError, match="NOPE"):
        C.build_legb_joint_ctx({"NOPE": "DESI"})


def test_guard_multi_leg_sample_res_not_built():
    """PI C4: per-leg f_res is NOT built — >1 resolution-floated leg with sample_res
    raises NotImplementedError naming the follow-up."""
    with pytest.raises(NotImplementedError, match="per-leg f_res"):
        C.build_legb_joint_ctx({"DESI": "DESI", "KS": "KS"}, sample_res=True,
                               ks_kwargs=dict(resolution_float=True, k_max=0.065))


@pytest.mark.skipif(not (_have and _have_ks), reason="real cache/ckpt/DESI/KS not present")
def test_guard_per_leg_zslope_requires_per_leg_alpha(nuts_setup):
    ctx, d, mock_legs, core = nuts_setup
    ctx_bad = ctx._replace(per_leg_zslope=True)          # per_leg_alpha stays False
    with pytest.raises(ValueError, match="per_leg_alpha"):
        _sites(C._legb_model, ctx_bad, mock_legs, core)


# --------------------------------------------------------------------------- #
#  T9 — joint_stamp contents, DLA prior-only label (PI C3), mixed-signature
#  rejection; forward_stamp alpha_mode.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not (_have and _have_ks), reason="real cache/ckpt/DESI/KS not present")
def test_joint_stamp_contents(joint2_setup):
    ctx, d, _, _ = joint2_setup
    st = C.joint_stamp(ctx)
    assert st["alpha_mode"] == "per_leg"
    assert st["legs"] == ["DESI", "KS"]
    assert st["per_leg_zslope"] is False
    for name in ("DESI", "KS"):
        p = st["per_leg"][name]
        assert p["survey"] == name
        assert p["lls_boost"] == pytest.approx(INF.HCD_LLS_SURVEY_BOOST[name])
        assert p["lls_frac_sigma"] == pytest.approx(INF.HCD_LLS_SURVEY_FRAC_SIGMA[name])
        np.testing.assert_allclose(p["alpha_hcd_mu"],
                                   np.asarray(ctx.alpha_hcd_mu_by_leg[name]))
        np.testing.assert_allclose(p["alpha_hcd_sigma"],
                                   np.asarray(ctx.alpha_hcd_sigma_by_leg[name]))
        # embedded per-leg forward_stamp with BOTH signatures + the per_leg alpha_mode.
        fw = p["forward"]
        assert fw["forward_signature"] == FORWARD_SIGNATURE_PIN
        assert fw["hcd_prior_signature"] == HCD_PRIOR_SIGNATURE_PIN
        assert fw["alpha_mode"] == "per_leg"
        # DLA sampled-width attestation carried per leg verbatim.
        assert p["dla_raw_latent_scale"] == 1.0
        assert "1.0" in p["dla_width_note"]
    # PI C3: dla_forward_frac=0 legs are LABELED prior-only (not data-constrained).
    assert st["per_leg"]["KS"]["dla_site_prior_only"] is True
    assert st["per_leg"]["DESI"]["dla_site_prior_only"] is False
    assert "PRIOR-ONLY" in st["per_leg"]["KS"]["dla_width_note"]
    # panel required-fix 1b: per-leg data-cut honesty in the stamp. The joint KS leg
    # without an R_z float sits under the NORC 0.045 cap, NOT the certified 0.065 band.
    for name in ("DESI", "KS"):
        p = st["per_leg"][name]
        leg = next(l for l in ctx.legs if l.name == name)
        assert p["k_max_effective"] == pytest.approx(float(np.asarray(leg.k).max()))
        assert p["resolution_float"] == bool(getattr(leg, "resolution_ready", False))
    # the NORC 0.045 KS cap binds only on the NORC forward with no R_z float (the v1
    # driver config); this fixture is not NORC, so only assert the conditional form.
    if (not ctx.res_corr_on) and (not st["per_leg"]["KS"]["resolution_float"]):
        assert st["per_leg"]["KS"]["k_max_effective"] <= 0.045 + 1e-9


@pytest.mark.skipif(not (_have and _have_ks), reason="real cache/ckpt/DESI/KS not present")
def test_joint_stamp_rejects_mixed_signatures(joint2_setup):
    ctx, d, _, _ = joint2_setup
    st = C.joint_stamp(ctx)
    doctored = {n: dict(p["forward"]) for n, p in st["per_leg"].items()}
    doctored["KS"]["forward_signature"] = "deadbeef" * 8
    with pytest.raises(ValueError, match="mixed"):
        C._assert_joint_signatures_uniform(doctored)
    doctored2 = {n: dict(p["forward"]) for n, p in st["per_leg"].items()}
    doctored2["DESI"]["hcd_prior_signature"] = "deadbeef" * 8
    with pytest.raises(ValueError, match="mixed"):
        C._assert_joint_signatures_uniform(doctored2)


def test_forward_stamp_alpha_mode_via_getattr():
    """forward_stamp gains alpha_mode ('shared'|'per_leg') via getattr default — buildable
    on pre-feature ctx stubs that lack the field entirely."""
    from types import SimpleNamespace
    stub = dict(res_corr_on=False, fix_alpha_res=True, sample_res=True,
                f_res_amp_sigma=0.02, metal_prior="flatlog2node", metal_node_z=(2.2, 4.2))
    leg = SimpleNamespace(dla_cov_reduced=True, dla_forward_frac=1.0, use_snr3=False,
                          cv_floor_on=False, cv_floor_rank1=False)
    st_old = C.forward_stamp(SimpleNamespace(**stub), leg)          # no per_leg_alpha attr
    assert st_old["alpha_mode"] == "shared"
    st_new = C.forward_stamp(SimpleNamespace(per_leg_alpha=True, **stub), leg)
    assert st_new["alpha_mode"] == "per_leg"


# --------------------------------------------------------------------------- #
#  T11 — _draws_matrix / _packed_names_for presence-keyed back-compat.
# --------------------------------------------------------------------------- #
def _legacy_samples(L=7, nzg=4):
    rng = np.random.default_rng(0)
    return {"theta_unit": rng.random((L, 9)), "tau0_vec": rng.random((L, nzg)),
            "alpha_lls": rng.random(L), "alpha_subdla": rng.random(L),
            "alpha_dla": rng.random(L)}


def _per_leg_samples(L=7, nzg=4, legs=("DESI", "KS")):
    rng = np.random.default_rng(1)
    s = {"theta_unit": rng.random((L, 9)), "tau0_vec": rng.random((L, nzg))}
    for nm in legs:
        s[f"alpha_lls_{nm}"] = rng.random(L)
        s[f"alpha_subdla_{nm}"] = rng.random(L)
        s[f"alpha_dla_{nm}"] = rng.random(L)
    return s


def test_draws_matrix_legacy_path_verbatim():
    s = _legacy_samples()
    kept = np.ones(4, bool)
    m = C._draws_matrix(s, kept)
    assert m.shape == (7, 9 + 4 + 3)
    np.testing.assert_array_equal(m[:, -3], s["alpha_lls"])
    np.testing.assert_array_equal(m[:, -2], s["alpha_subdla"])
    np.testing.assert_array_equal(m[:, -1], s["alpha_dla"])
    names = C._packed_names_for(s, kept)
    assert names[-3:] == ["alpha_lls", "alpha_subdla", "alpha_dla"]


def test_draws_matrix_per_leg_columns_in_legs_order():
    s = _per_leg_samples()
    kept = np.ones(4, bool)
    m = C._draws_matrix(s, kept)
    assert m.shape == (7, 9 + 4 + 6)
    names = C._packed_names_for(s, kept)
    assert names[-6:] == ["alpha_lls_DESI", "alpha_subdla_DESI", "alpha_dla_DESI",
                          "alpha_lls_KS", "alpha_subdla_KS", "alpha_dla_KS"]
    for j, nm in enumerate(names[-6:]):
        np.testing.assert_array_equal(m[:, 13 + j], s[nm], err_msg=nm)


def test_draws_matrix_neither_alpha_family_raises():
    s = {"theta_unit": np.zeros((3, 9)), "tau0_vec": np.zeros((3, 2))}
    kept = np.ones(2, bool)
    with pytest.raises((KeyError, ValueError)):
        C._draws_matrix(s, kept)
    with pytest.raises((KeyError, ValueError)):
        C._packed_names_for(s, kept)


# --------------------------------------------------------------------------- #
#  Thin joint driver (scripts/run_joint_fit.py): source-level wiring checks
#  (the repo idiom for driver tripwires; the driver is post-unblind machinery).
# --------------------------------------------------------------------------- #
def test_joint_driver_wiring_source_checks():
    src_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "run_joint_fit.py")
    assert os.path.exists(src_path), "scripts/run_joint_fit.py missing"
    src = open(src_path).read()
    assert "assert_env_data_flags_unset(" in src, "F2 env tripwire missing from the joint driver"
    assert "build_legb_joint_ctx(" in src, "joint driver must build via build_legb_joint_ctx"
    assert "export_getdist(" in src, "joint chains must go out in cobaya/GetDist format"
    assert "joint_stamp(" in src, "joint driver must record the joint_stamp"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
