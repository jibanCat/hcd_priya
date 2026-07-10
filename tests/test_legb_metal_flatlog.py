"""Task A1 — opt-in flat-log metal-amplitude prior on the BLINDED Leg-B forward (TDD).

The metal oscillation amplitudes ``a_SiIII`` / ``a_SiII`` (SHARED scalar sites on metals_on legs)
get a SECOND, opt-in prior mode selected by the STATIC ``ctx.metal_prior``:

  "uniform" (DEFAULT, golden-safe): ``dist.Uniform(0, a_siiii_max)`` — BYTE-EXACT to the legacy code.
  "flatlog": ``dist.LogUniform(a_lo, a_hi)`` on the SAME site names, with the flat-log10(f) prior on
    the metal flux decrement f mapped to the oscillation amplitude ``a = f/(1-<F>_ref)``:
      a_lo = 10**metal_logf_lo / (1 - F_ref),  a_hi = 10**metal_logf_hi / (1 - F_ref).
    ``1 - F_ref`` (``ctx.metal_one_minus_F_ref``) is the ONE global scalar build_legb_ctx derives
    from the fiducial mean flux exp(-Kim07(z)) on the union z-grid (a constant scale on a log-uniform
    => the prior stays flat-log f). Keeping the site NAMED a_SiIII/a_SiII preserves every by-name
    downstream read (_draws_matrix / _packed_names_for / the re-score loglik / constrain_fn).

These tests pin (TDD): default-uniform site construction is unchanged; flatlog sites are LogUniform
with the expected (a_lo,a_hi); the a=f/(1-F_ref) map; constrain_fn mirror parity in BOTH modes;
the draws/packed-name columns; the a_SiII re-score threading fix; and a short flatlog NUTS smoke.

Env (MANDATORY):
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
    /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_legb_metal_flatlog.py -q
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

_CACHE = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/observables_tau0_lf.h5"
_CKPT0 = "/home/mfho/hcd_priya/checkpoints/final_fold0.eqx"
_DESI_NPZ = "/home/mfho/data/desi_dr1_p1d/desi_dr1_p1d.npz"
_have = all(os.path.exists(p) for p in (_CACHE, _CKPT0, _DESI_NPZ))


# --------------------------------------------------------------------------- #
#  Helpers.
# --------------------------------------------------------------------------- #
def _sites(model, *a, seed=1):
    tr = handlers.trace(handlers.seed(model, jax.random.PRNGKey(seed))).get_trace(*a)
    return [k for k, v in tr.items() if v["type"] == "sample" and not v.get("is_observed")]


def _trace(model, *a, seed=1):
    return handlers.trace(handlers.seed(model, jax.random.PRNGKey(seed))).get_trace(*a)


def _ctx_metal(metal_prior="uniform", sample_a_siii=True, metals_on=False):
    """A DESI-only narrow-z Leg-B ctx with the metal sites enabled (sample_metals=True), the
    metal_prior mode selected, and the SiII doublet site optionally floated. ``metals_on``
    controls whether the DESI leg FORWARD-MODELS the metal factor (needed for the re-score /
    smoke tests where a_SiIII/a_SiII must enter the loglik)."""
    ctx, d = C.build_legb_ctx(desi_kwargs=dict(z_lo=0.0, z_hi=2.6), use_xclass=True,
                              sample_metals=True, metals_on=metals_on,
                              metal_prior=metal_prior)
    ctx = ctx._replace(legs=[l for l in ctx.legs if l.name == "DESI"],
                       sample_a_siii=bool(sample_a_siii))
    return ctx, d


def _mock(ctx, d, key_seed=0, inject_a_siiii=0.0):
    sims, _ = C.held_out_sims(d, fold=0)
    truth = C.make_truth_from_sim(d, sims[0], fold=0)
    mock_legs, truth_pack, _ = C.make_legb_mock(
        ctx, truth, jax.random.PRNGKey(key_seed), inject_a_siiii=inject_a_siiii)
    core = C._mock_core_per_leg(ctx, truth)
    return mock_legs, truth_pack, core


def _bounds(ctx):
    omf = float(ctx.metal_one_minus_F_ref)
    a_lo = 10.0 ** float(ctx.metal_logf_lo) / omf
    a_hi = 10.0 ** float(ctx.metal_logf_hi) / omf
    return a_lo, a_hi, omf


# --------------------------------------------------------------------------- #
#  1. UNIFORM default — site construction byte-exact (golden-safe).
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI not present")
def test_uniform_default_site_construction_unchanged():
    """metal_prior='uniform' (the default) keeps the a_SiIII/a_SiII sites as the literal legacy
    dist.Uniform(0, a_siiii_max), and _legb_model / _legb_priors_only trace the SAME ordered
    sample-site list (the golden-safe default contract)."""
    ctx, d = _ctx_metal(metal_prior="uniform", sample_a_siii=True)
    assert ctx.metal_prior == "uniform"                     # the golden-safe default
    mock_legs, _, core = _mock(ctx, d)
    trm = _trace(C._legb_model, ctx, mock_legs, core)
    for nm in ("a_SiIII", "a_SiII"):
        fn = trm[nm]["fn"]
        assert isinstance(fn, dist.Uniform), f"{nm} not Uniform under default: {type(fn).__name__}"
        assert float(fn.low) == 0.0 and float(fn.high) == float(ctx.a_siiii_max), \
            f"{nm} Uniform bounds drifted from (0, a_siiii_max)"
    s_model = _sites(C._legb_model, ctx, mock_legs, core)
    s_prior = _sites(C._legb_priors_only, ctx)
    assert s_model == s_prior, f"uniform site-order mismatch: {s_model} vs {s_prior}"
    # a_SiIII just before a_SiII just before the res_corr block (the documented order).
    assert s_model[-4:] == ["a_SiIII", "a_SiII", "alpha_res", "alpha_res_slope"]


# --------------------------------------------------------------------------- #
#  2. FLATLOG — the sites are LogUniform with the expected (a_lo, a_hi).
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI not present")
def test_flatlog_sites_are_loguniform_with_expected_bounds():
    """metal_prior='flatlog' → a_SiIII and a_SiII are dist.LogUniform with
    low==10**logf_lo/(1-F_ref), high==10**logf_hi/(1-F_ref) (the a=f/(1-F_ref) map), on the SAME
    site names (no rename)."""
    ctx, d = _ctx_metal(metal_prior="flatlog", sample_a_siii=True)
    assert ctx.metal_prior == "flatlog"
    a_lo, a_hi, _ = _bounds(ctx)
    mock_legs, _, core = _mock(ctx, d)
    trm = _trace(C._legb_model, ctx, mock_legs, core)
    trp = _trace(C._legb_priors_only, ctx)
    for tr in (trm, trp):
        for nm in ("a_SiIII", "a_SiII"):
            fn = tr[nm]["fn"]
            assert isinstance(fn, dist.LogUniform), \
                f"{nm} not LogUniform under flatlog: {type(fn).__name__}"
            assert float(fn.low) == pytest.approx(a_lo, rel=1e-12), f"{nm} a_lo mismatch"
            assert float(fn.high) == pytest.approx(a_hi, rel=1e-12), f"{nm} a_hi mismatch"
    # the drawn amplitude lives strictly inside the bounds (LogUniform support).
    a_drawn = float(trm["a_SiIII"]["value"])
    assert a_lo < a_drawn < a_hi


# --------------------------------------------------------------------------- #
#  3. a = f/(1-F_ref) amplitude map + F_ref provenance.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI not present")
def test_amplitude_map_and_Fref_provenance():
    """The LogUniform bounds equal the f-bounds / (1-F_ref); a representative f=0.009 maps to
    a=f/(1-F_ref) interior to (a_lo,a_hi); and (1-F_ref) is the union-z fiducial mean-flux scalar
    build_legb_ctx computed (mean over z of exp(-Kim07(z)))."""
    ctx, _ = _ctx_metal(metal_prior="flatlog", sample_a_siii=True)
    a_lo, a_hi, omf = _bounds(ctx)
    # bounds == f-bounds / (1-F_ref).
    assert a_lo == pytest.approx(10.0 ** float(ctx.metal_logf_lo) / omf, rel=1e-12)
    assert a_hi == pytest.approx(10.0 ** float(ctx.metal_logf_hi) / omf, rel=1e-12)
    # a representative f maps to an interior a.
    f_rep = 0.009
    a_rep = f_rep / omf
    assert a_lo < a_rep < a_hi, f"f={f_rep} -> a={a_rep:.4g} not interior to ({a_lo:.3g},{a_hi:.3g})"
    # F_ref provenance: the SAME fiducial mean flux the forward uses at the prior center
    # (tau0_amp=1, dtau0=0 => tau0(z)=Kim07(z)), averaged over the union z-grid.
    F_z = np.exp(-np.asarray(C._kim(jnp.asarray(ctx.z_global))))
    omf_expected = float(1.0 - np.mean(F_z))
    assert omf == pytest.approx(omf_expected, rel=1e-12), \
        f"1-F_ref {omf} != union-z fiducial mean-flux scalar {omf_expected}"
    assert 0.0 < omf < 1.0


# --------------------------------------------------------------------------- #
#  4. constrain_fn MIRROR parity (load-bearing) — both modes + round-trip.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI not present")
@pytest.mark.parametrize("metal_prior", ["uniform", "flatlog"])
def test_constrain_fn_mirror_parity(metal_prior):
    """For BOTH modes _legb_model and _legb_priors_only trace the IDENTICAL ordered sample-site
    name list (the fast-postprocess constrain_fn mirror), and the constrain_fn round-trip maps an
    arbitrary unconstrained a_SiIII into (a_lo, a_hi)."""
    ctx, d = _ctx_metal(metal_prior=metal_prior, sample_a_siii=True)
    mock_legs, _, core = _mock(ctx, d)
    s_model = _sites(C._legb_model, ctx, mock_legs, core)
    s_prior = _sites(C._legb_priors_only, ctx)
    assert s_model == s_prior, f"[{metal_prior}] mirror site-order mismatch: {s_model} vs {s_prior}"
    assert s_model[-4:] == ["a_SiIII", "a_SiII", "alpha_res", "alpha_res_slope"]

    # constrain_fn round-trip through the priors-only mirror (the production postprocess path).
    a_lo, a_hi, _ = _bounds(ctx)
    init = initialize_model(jax.random.PRNGKey(2), lambda: C._legb_priors_only(ctx))
    z0 = dict(init.param_info.z)
    assert "a_SiIII" in z0 and "a_SiII" in z0
    for u in (-25.0, 0.0, 25.0):                            # arbitrary unconstrained a_SiIII
        z = dict(z0); z["a_SiIII"] = jnp.asarray(u)
        c = constrain_fn(lambda: C._legb_priors_only(ctx), (), {}, z, return_deterministic=False)
        a = float(c["a_SiIII"])
        if metal_prior == "flatlog":
            assert a_lo < a < a_hi, f"flatlog constrain a_SiIII {a:.3g} not in ({a_lo:.3g},{a_hi:.3g})"
        else:
            assert 0.0 <= a <= float(ctx.a_siiii_max)


# --------------------------------------------------------------------------- #
#  5. _draws_matrix / _packed_names_for — a_SiIII/a_SiII columns under flatlog.
# --------------------------------------------------------------------------- #
def test_draws_matrix_and_packed_names_carry_metal_columns():
    """Under flatlog the site names are UNCHANGED (a_SiIII/a_SiII), so the by-name readers
    _draws_matrix / _packed_names_for carry both columns, appended LAST in the documented order."""
    L, nz, nzg = 5, 2, 3
    samples = dict(
        theta_unit=np.zeros((L, 9)),
        tau0_vec=np.zeros((L, nzg)),
        alpha_lls=np.full(L, 0.27), alpha_subdla=np.full(L, 0.09), alpha_dla=np.full(L, 0.016),
        a_SiIII=np.full(L, 1.2e-3), a_SiII=np.full(L, 4.0e-4))      # flatlog-scale amplitudes
    kept_global = np.array([True, True, False])
    names = C._packed_names_for(samples, kept_global)
    mat = C._draws_matrix(samples, kept_global)
    assert names[-2:] == ["a_SiIII", "a_SiII"], f"metal cols not last/named: {names}"
    assert mat.shape == (L, len(names)), "draws-matrix width != packed-names length"
    # the last two columns ARE the metal amplitudes.
    np.testing.assert_array_equal(mat[:, names.index("a_SiIII")], samples["a_SiIII"])
    np.testing.assert_array_equal(mat[:, names.index("a_SiII")], samples["a_SiII"])


# --------------------------------------------------------------------------- #
#  6. a_SiII threaded into the re-score loglik (the pre-existing :2156 gap fix).
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI not present")
def test_a_siII_threaded_into_rescore_loglik():
    """_loglik_of_draws must CHANGE when samples['a_SiII'] is nonzero — the doublet nuisance now
    enters the re-score loglik (previously only a_SiIII was threaded → a wrong loglik rank for the
    floated-doublet cell). Requires metals_on so the forward applies the metal factor."""
    ctx, d = _ctx_metal(metal_prior="flatlog", sample_a_siii=True, metals_on=True)
    mock_legs, truth_pack, core = _mock(ctx, d, inject_a_siiii=0.01)
    kept_global = truth_pack["kept_global_z"]
    # one valid model draw → tile to a small L (valid theta/tau0/alpha_hcd_z/a_SiIII).
    tr = _trace(C._legb_model, ctx, mock_legs, core, seed=4)
    L = 4
    base = dict(
        theta_unit=np.tile(np.asarray(tr["theta_unit"]["value"]), (L, 1)),
        tau0_vec=np.tile(np.asarray(tr["tau0_vec"]["value"]), (L, 1)),
        alpha_hcd_z=np.tile(np.asarray(tr["alpha_hcd_z"]["value"]), (L, 1, 1)),
        a_SiIII=np.full(L, float(tr["a_SiIII"]["value"])),
        a_SiII=np.zeros(L))
    ll0 = C._loglik_of_draws(ctx, mock_legs, core, base, kept_global)
    s2 = dict(base); s2["a_SiII"] = np.full(L, 0.05)        # a clearly-nonzero doublet amplitude
    ll1 = C._loglik_of_draws(ctx, mock_legs, core, s2, kept_global)
    assert np.all(np.isfinite(ll0)) and np.all(np.isfinite(ll1))
    assert not np.allclose(ll0, ll1), \
        "_loglik_of_draws ignores samples['a_SiII'] — the doublet is not threaded into the re-score"
    # back-compat: absent a_SiII key reproduces the a_SiII=0 result exactly.
    nob = {k: v for k, v in base.items() if k != "a_SiII"}
    ll_absent = C._loglik_of_draws(ctx, mock_legs, core, nob, kept_global)
    np.testing.assert_allclose(ll_absent, ll0, rtol=1e-12, atol=0.0)


# --------------------------------------------------------------------------- #
#  7. NUTS smoke (flatlog) — finite, 0 divergences, posterior off both rails.
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.skipif(not _have, reason="real cache/ckpt/DESI not present")
def test_flatlog_nuts_smoke_offrail():
    """A short flatlog NUTS run (sample_metals + sample_a_siii) on an SiIII-injected mock: all
    draws finite, 0 divergences, and the posterior of BOTH metal sites is off both rails (interior
    to (a_lo,a_hi), data-pulled for a_SiIII, the geometric-median for the prior-led a_SiII)."""
    ctx, d = _ctx_metal(metal_prior="flatlog", sample_a_siii=True, metals_on=True)
    mock_legs, _, core = _mock(ctx, d, inject_a_siiii=0.01)
    a_lo, a_hi, _ = _bounds(ctx)
    samples, n_div = C._run_nuts_legb(
        ctx, mock_legs, core, n_warmup=20, n_samples=20, seed=0,
        target_accept=0.9, dense_mass=False, max_tree_depth=8)
    assert n_div == 0, f"flatlog metal smoke: {n_div} divergence(s)"
    for nm in ("a_SiIII", "a_SiII"):
        a = np.asarray(samples[nm])
        assert np.all(np.isfinite(a)), f"{nm} not finite"
        assert np.all((a > a_lo) & (a < a_hi)), f"{nm} outside the LogUniform support"
        med = float(np.median(a))
        # off the LOWER rail (>> a_lo) and off the UPPER rail (comfortably below a_hi).
        assert med > a_lo * 1e3, f"{nm} posterior median {med:.3g} jammed at the lower rail {a_lo:.3g}"
        assert med < a_hi * 0.8, f"{nm} posterior median {med:.3g} pressed the upper rail {a_hi:.3g}"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q", "-m", "not slow"]))
