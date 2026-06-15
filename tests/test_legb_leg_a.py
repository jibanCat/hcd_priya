"""Phase-C — Leg-A (rank-uniformity SBC) ON THE LEG GRIDS, on the production baseline.

Leg-A draws θ from the inference PRIOR, forward-models it on each leg, and adds matched-C
noise (C_mock ≡ C_like at the truth) → rank uniformity is the exact null. This certifies the
PRODUCTION baseline likelihood (2-param τ₀, lit-pinned HCD, emucoh+cross-class C_emu, MF,
metals) — unlike the cache-grid Leg-A in closure_sbc. Pins:
  (a) draw_leg_a_leg_truth draws an in-box truth from the legb priors, deterministic in (key);
  (b) make_leg_a_legmock forward-models the truth with the SAME predict_P_obs_on_leg the
      likelihood uses, so the noiseless mock == P_model(truth) AND the noise covariance ==
      the likelihood's C_total(truth) — the Leg-A self-draw property, per leg;
  (c) build_legb_ctx(ensemble_ckpts=...) carries the EnsembleEmulator.

Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu \
     /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_legb_leg_a.py -q
"""
import glob

import hcd_analysis.emulator  # noqa: F401  enables x64
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hcd_analysis.emulator import closure_legb as LB
from hcd_analysis.emulator import data_likelihood as DL
from hcd_analysis.emulator.ensemble import EnsembleEmulator

REPO = "/home/mfho/hcd_priya"
PROD = f"{REPO}/checkpoints/final_prod_seed"


@pytest.fixture(scope="module")
def ctx_d():
    try:
        return LB.build_legb_ctx(use_xclass=True)        # light baseline (xclass ρ; no MF/eBOSS)
    except Exception as e:                               # data/checkpoint absent in this env
        pytest.skip(f"build_legb_ctx unavailable: {e}")


def _predict_leg_at_truth(ctx, leg, tp, core):
    """Independently call predict_P_obs_on_leg at the truth — the (P, C) the likelihood uses."""
    zg = np.asarray(ctx.z_global)
    sel = jnp.asarray([int(np.argmin(np.abs(zg - zz))) for zz in leg.z])
    tau0_vec = jnp.asarray(tp["tau0_global"])[sel]
    alpha_leg = jnp.asarray(tp["alpha_hcd_z"])[sel]
    szb = ctx.sigma_zb_per_leg.get(leg.name) if ctx.sigma_zb_per_leg else None
    rzb = ctx.rho_zb_per_leg.get(leg.name) if ctx.rho_zb_per_leg else None
    return DL.predict_P_obs_on_leg(
        ctx.model, jnp.asarray(tp["theta9"]), tau0_vec, alpha_leg, pf_stats=ctx.pf_stats,
        dla_core=core[leg.name], cache_k=ctx.cache_k, leg=leg, sigma_zb=szb,
        alpha_centres=ctx.alpha_centres, a_SiIII=float(tp.get("a_siiii", 0.0)),
        cemu_inflate=ctx.cemu_inflate, rho_zb=rzb, mf=ctx.mf, mf_floor=ctx.mf_floor)


def test_draw_leg_a_leg_truth(ctx_d):
    ctx, d = ctx_d
    nZg = len(ctx.z_global)
    tp = LB.draw_leg_a_leg_truth(ctx, jax.random.PRNGKey(0))
    assert np.asarray(tp["theta9"]).shape == (9,)
    assert np.asarray(tp["tau0_global"]).shape == (nZg,)
    assert np.asarray(tp["alpha_hcd"]).shape == (3,)            # pivot α (for the rank truth_vec)
    assert np.asarray(tp["alpha_hcd_z"]).shape == (nZg, 3)       # z-resolved (for the forward)
    assert bool(np.all(tp["kept_global_z"]))
    # θ drawn inside the (restricted) unit box, and deterministic in the key
    _lo = getattr(ctx, "theta_unit_lo", None)
    _hi = getattr(ctx, "theta_unit_hi", None)
    lo = np.zeros(9) if _lo is None else np.asarray(_lo)
    hi = np.ones(9) if _hi is None else np.asarray(_hi)
    th = np.asarray(tp["theta9"])
    assert np.all(th >= lo - 1e-9) and np.all(th <= hi + 1e-9)
    tp2 = LB.draw_leg_a_leg_truth(ctx, jax.random.PRNGKey(0))
    assert np.array_equal(np.asarray(tp["theta9"]), np.asarray(tp2["theta9"]))
    assert not np.array_equal(np.asarray(tp["theta9"]),
                              np.asarray(LB.draw_leg_a_leg_truth(ctx, jax.random.PRNGKey(1))["theta9"]))


def test_leg_a_legmock_is_a_matched_self_draw(ctx_d):
    """The Leg-A self-draw property: noiseless mock == P_model(truth) and the noise Cholesky
    == chol(C_total(truth)) — both from the SAME predict_P_obs_on_leg the likelihood calls."""
    ctx, d = ctx_d
    # the (K,) z-mean fiducial core (matches what run_legb / the leg likelihood use)
    core = {k: np.nanmean(np.asarray(v), axis=0)
            for k, v in LB._fiducial_dla_core_per_leg(d, ctx.legs, ctx.cache_k).items()}
    tp = LB.draw_leg_a_leg_truth(ctx, jax.random.PRNGKey(3))
    mock_legs, info = LB.make_leg_a_legmock(ctx, core, tp, jax.random.PRNGKey(4))
    assert len(mock_legs) == len(ctx.legs)
    for leg in ctx.legs:
        P_ref, C_ref = _predict_leg_at_truth(ctx, leg, tp, core)
        # the stored noiseless truth-on-leg == the forward P at the truth
        assert np.allclose(np.asarray(info["truth_on_leg"][leg.name]), np.asarray(P_ref),
                           rtol=0, atol=1e-8), f"{leg.name}: noiseless mock != P_model(truth)"
        # the noise Cholesky used == chol(C_like at truth) (C_mock ≡ C_like)
        L_ref = np.asarray(LB._chol_jitter(C_ref))
        assert np.allclose(np.asarray(info["chol"][leg.name]), L_ref, rtol=0, atol=1e-10), \
            f"{leg.name}: mock noise C != likelihood C at truth"
        # the realised mock differs from the noiseless truth (noise actually added)
        assert not np.allclose(np.asarray(mock_legs[ctx.legs.index(leg)].P_data),
                               np.asarray(P_ref))


def test_run_legb_leg_a_end_to_end(ctx_d):
    """run_legb(leg_a=True) produces per-mock records compatible with aggregate_leg_a (the
    shard→merge path the pilot uses). One tiny NUTS on the single-model leg ctx."""
    from hcd_analysis.emulator.closure_sbc import aggregate_leg_a
    ctx, d = ctx_d
    recs = LB.run_legb(ctx, d, n_mocks=1, mock_indices=[0], return_per_mock=True, leg_a=True,
                       n_warmup=8, n_samples=8, seed=0, verbose=False)
    assert isinstance(recs, list) and len(recs) == 1
    for k in ("truth_vec", "draws", "ll_true", "ll_draws", "n_div"):
        assert k in recs[0]
    res = aggregate_leg_a(recs, len(ctx.z_global), prob=0.95)
    assert "ranks" in res and "names" in res and "L" in res


def test_run_legb_leg_a_metals_no_offbyone():
    """Pilot-blocker guard: with sample_metals (the pilot baseline), run_legb(leg_a) must produce
    truth_vec/draws/names that ALIGN (the a_SiIII column), and aggregate_leg_a must not crash."""
    from hcd_analysis.emulator.closure_sbc import aggregate_leg_a
    try:
        ctx, d = LB.build_legb_ctx(use_xclass=True, sample_metals=True)
    except Exception as e:
        pytest.skip(f"metals ctx unavailable: {e}")
    recs = LB.run_legb(ctx, d, n_mocks=1, mock_indices=[0], return_per_mock=True, leg_a=True,
                       n_warmup=8, n_samples=8, seed=0, verbose=False)
    assert len(recs) == 1
    r = recs[0]
    assert "a_SiIII" in r["names"]
    assert len(r["truth_vec"]) == np.asarray(r["draws"]).shape[1] == len(r["names"])
    res = aggregate_leg_a(recs, len(ctx.z_global), prob=0.95)     # must NOT raise
    assert "a_SiIII" in res["names"] and res["names"][-1] == "loglik"


@pytest.mark.skipif(len(glob.glob(PROD + "*.eqx")) < 2, reason="prod ensemble absent")
def test_build_legb_ctx_ensemble():
    paths = sorted(p[:-4] for p in glob.glob(PROD + "*.eqx"))
    ctx, d = LB.build_legb_ctx(use_xclass=True, ensemble_ckpts=paths)
    assert isinstance(ctx.model, EnsembleEmulator)
    assert len(ctx.model.members) == len(paths) >= 2
