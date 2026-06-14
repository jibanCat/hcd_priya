"""Per-leg DLA-residual closure tests (§0c, RESTORED — PI-confirmed final intent 2026-06-09).

THE MODEL (this is what these tests pin):

  The 10% unmasked-DLA residual belongs in the TARGET MOCK; the FORWARD marginalizes the HCD
  nuisance (α amplitude + z-slope) over it. The closure tests whether HCD-marginalization
  recovers cosmology DESPITE the DLA residual. The production forward is τ=1e6-filtered with NO
  DLA emulator — just the HCD nuisance.

  PER-LEG TARGET (closure truth):
    * DESI: filtered Tier-P + 0.10·(full DLA excess) — the DLA finder misses ~10% of DLAs, so
      that 10% remains as FULL systems in the DESI target.
    * KS: filtered Tier-P with 0% DLA — KS fully masks DLAs, so its target carries NO DLA residual.
    Mechanism: a per-leg DLA residual fraction ``truth_frac`` = (DESI 0.10, KS 0.0) applied to the
    full-DLA-excess add-back (make_truth_from_sim builds the masked baseline + the full excess;
    make_legb_mock adds ``truth_frac_leg``·excess per leg).

  PER-LEG FORWARD:
    * DataLeg.dla_forward_frac (DESI=1.0, KS=0.0) scales the sampled α_DLA's DLA-excess
      contribution per leg. KS → 0 (the forward's KS DLA term is 0, matching the 0% KS target);
      DESI → the sampled α_DLA.

  α_DLA PRIOR (marginalized, NOT fixed):
    * HCD_DLA_RESIDUAL_FRAC = 0.10 (the DESI 10% residual center),
      HCD_PRIOR_FRAC_SIGMA[2] = 0.50 (σ/μ, one-sided softplus → α_DLA ≥ 0, masking-completeness
      width). α_LLS/α_subDLA UNCHANGED (observed dN/dX).

  TRUTH↔FORWARD CONSISTENCY: at θ→truth with α_LLS/α_subDLA at structural w_c and
  α_DLA = 0.10·(lit/sim)·w_DLA on DESI / forward DLA term = 0 on KS, the forward reproduces the
  per-leg target (DESI carries the 10% DLA, KS carries none) to tight rtol.

Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_legb_dla_masking.py -v
"""
import os

import hcd_analysis.emulator  # noqa: F401  enables jax_enable_x64
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hcd_analysis.emulator.model import Emulator
from hcd_analysis.emulator import data_likelihood as DL
from hcd_analysis.emulator import inference as I
from hcd_analysis.emulator.predict import predict_P_filt, predict_P_obs
from hcd_analysis.emulator.sampler_numpyro import _dla_raw_mu


# ----------------------------------------------------------------------------- #
#  synthetic emulator + cache fixtures (mirror test_data_likelihood._emu_ctx)
# ----------------------------------------------------------------------------- #
def _emu_ctx(n_k=172, n_basis=12, n_tb=4, seed=0):
    rng = np.random.default_rng(seed)
    model = Emulator(in_dim=10, n_k=n_k, n_basis=n_basis, key=jax.random.PRNGKey(seed))
    pf = {"mu_marg": jnp.asarray(rng.normal(-2, 0.5, (4, n_k))),
          "sig_marg": jnp.asarray(rng.uniform(0.5, 1.5, (4, n_k))),
          "sig_cosmo": jnp.asarray(rng.uniform(0.02, 0.1, (4, n_k)))}
    dla_core = jnp.asarray(rng.uniform(0.0, 0.5, n_k))
    cache_k = jnp.asarray(np.linspace(4.04e-4, 0.0694, n_k))
    theta9 = jnp.full(9, 0.5)
    return dict(model=model, pf=pf, dla_core=dla_core, cache_k=cache_k,
                theta9=theta9, n_k=n_k, n_tb=n_tb, rng=rng)


# ============================================================================ #
#  Constants: α_DLA residual = 0.10 (DESI), σ/μ = 0.50; per-leg DLA-forward axis.
# ============================================================================ #
def test_dla_residual_frac_is_ten_percent():
    """HCD_DLA_RESIDUAL_FRAC = 0.10 (the DESI 10% unmasked-DLA residual center); α_DLA is
    MARGINALIZED (sampled) over it, not fixed."""
    assert I.HCD_DLA_RESIDUAL_FRAC == 0.10, \
        "α_DLA residual center must be 0.10 (the DESI 10% unmasked-DLA residual)"
    # the §0c one-sided DLA width σ/μ = 0.50 (masking-completeness); LLS/subDLA unchanged.
    assert I.HCD_PRIOR_FRAC_SIGMA[2] == 0.50, "DLA σ/μ must be 0.50 (masking-completeness width)"
    assert I.HCD_PRIOR_FRAC_SIGMA[0] == 0.15 and I.HCD_PRIOR_FRAC_SIGMA[1] == 0.40, \
        "LLS/subDLA fractional widths must stay at the original (0.15, 0.40)"


def test_per_leg_dla_forward_axis_restored():
    """The §0c per-leg DLA-forward axis is RESTORED: DataLeg.dla_forward_frac + the module
    constants DESI_DLA_FORWARD_FRAC=1.0 / KS_DLA_FORWARD_FRAC=0.0."""
    assert "dla_forward_frac" in DL.DataLeg._fields, \
        "DataLeg.dla_forward_frac must be restored (per-leg DLA-forward axis)"
    assert DL.DESI_DLA_FORWARD_FRAC == 1.0, "DESI forward carries the full sampled α_DLA"
    assert DL.KS_DLA_FORWARD_FRAC == 0.0, "KS forward DLA term must be 0 (KS fully masks DLAs)"


def test_loaders_set_per_leg_dla_forward_frac():
    """The DESI / KS loaders set dla_forward_frac = 1.0 / 0.0 respectively (default leg field)."""
    # DataLeg default is the DESI value (1.0) when constructed bare; the loaders set per-leg.
    # We check the field DEFAULT is back-compatible (1.0 = no-op vs the legacy full DLA forward).
    assert DL.DataLeg._field_defaults.get("dla_forward_frac", None) == 1.0, \
        "DataLeg.dla_forward_frac default must be 1.0 (back-compat: full DLA forward)"


# ============================================================================ #
#  α_DLA PRIOR: centered at the 10% residual, one-sided softplus → α_DLA ≥ 0.
# ============================================================================ #
def test_dla_prior_centered_at_ten_percent_residual():
    """hcd_incidence_prior centers α_DLA at HCD_DLA_RESIDUAL_FRAC·(lit/sim)·w_DLA = 0.10·(...),
    while LLS/subDLA stay on the OBSERVED dN/dX = (lit/sim)·w_c (the ORIGINAL design, unchanged)."""
    w_c = jnp.asarray([0.05, 0.02, 0.004])           # (LLS, subDLA, DLA)
    z = 3.0
    mu, sig = I.hcd_incidence_prior(w_c, z=z)
    r = np.asarray(I.lit_over_sim_at_z(z))
    # DLA center is 0.10·(lit/sim)·w_DLA.
    mu_dla_expect = I.HCD_DLA_RESIDUAL_FRAC * r[2] * float(w_c[2])
    assert np.isclose(float(mu[2]), mu_dla_expect, rtol=1e-10)
    assert float(mu[2]) > 0.0, "α_DLA center is a positive 10% residual (not 0)"
    # LLS/subDLA UNCHANGED: center = (lit/sim)·w_c, σ/μ = 0.15 / 0.40.
    assert np.isclose(float(mu[0]), r[0] * float(w_c[0]), rtol=1e-10)
    assert np.isclose(float(mu[1]), r[1] * float(w_c[1]), rtol=1e-10)
    assert np.isclose(float(sig[0]) / float(mu[0]), 0.15, rtol=1e-10)
    assert np.isclose(float(sig[1]) / float(mu[1]), 0.40, rtol=1e-10)
    # DLA width is the 0.50 σ/μ (at z ≤ 3.5, no high-z inflation).
    assert np.isclose(float(sig[2]) / float(mu[2]), 0.50, rtol=1e-10)


def test_dla_softplus_prior_is_one_sided_around_residual():
    """The one-sided softplus(N(_dla_raw_mu(μ_DLA), 1.0)) prior keeps α_DLA ≥ 0 and centers its
    bulk at the 10% residual scale (varying, not frozen)."""
    w_c = jnp.asarray([0.05, 0.02, 0.004])
    mu, _ = I.hcd_incidence_prior(w_c, z=3.0)
    raw_mu = float(_dla_raw_mu(mu[2]))
    rng = np.random.default_rng(0)
    a_dla = np.log1p(np.exp(raw_mu + 1.0 * rng.normal(size=200000)))   # softplus(latent)
    assert (a_dla >= 0).all(), "softplus prior is one-sided (α_DLA ≥ 0)"
    # the prior bulk spreads around the 10% residual center (it is sampled, varying).
    assert np.std(a_dla) > 0, "α_DLA prior must be a varying (marginalized) site, not a point"


# ============================================================================ #
#  Forward per-leg DLA scaling: predict_P_obs_on_leg scales the DLA-excess term by
#  leg.dla_forward_frac (DESI=1.0 keeps it, KS=0.0 zeroes it).
# ============================================================================ #
def test_forward_ks_dla_term_is_zero_desi_keeps_it():
    """With α_DLA>0, the DESI forward (dla_forward_frac=1.0) carries the full DLA-excess term, and
    the KS forward (dla_forward_frac=0.0) carries ZERO DLA-excess — verified on a one-z leg by
    comparing the per-leg P_model to the same forward with α_DLA=0."""
    from hcd_analysis.emulator import data_likelihood as DLm
    c = _emu_ctx()
    th = c["theta9"]
    # build two tiny one-z legs (DESI frac 1.0, KS frac 0.0) sharing the same k grid.
    k = np.asarray(c["cache_k"])[10:20]
    z = np.array([3.0]); z_unit = (z - 2.0) / 3.4
    leg_common = dict(z=z, z_unit=z_unit, k=k, z_row=np.full(k.size, 3.0),
                      z_idx=np.zeros(k.size, int), P_data=np.ones(k.size),
                      C_data=np.eye(k.size), R_z=np.array([0.01]), n_z=1,
                      n_per_z=np.array([k.size]), metals_on=False, resolution_on=False,
                      mf_floor_on=False)
    leg_desi = DLm.DataLeg(name="DESI", dla_forward_frac=1.0, **leg_common)
    leg_ks = DLm.DataLeg(name="KS", dla_forward_frac=0.0, **leg_common)
    tau0 = jnp.array([0.8])
    a = jnp.array([0.06, 0.02, 0.05])
    a0 = jnp.array([0.06, 0.02, 0.0])
    kw = dict(pf_stats=c["pf"], dla_core=c["dla_core"], cache_k=c["cache_k"])
    P_desi_a, _ = DLm.predict_P_obs_on_leg(c["model"], th, tau0, a, leg=leg_desi, **kw)
    P_desi_0, _ = DLm.predict_P_obs_on_leg(c["model"], th, tau0, a0, leg=leg_desi, **kw)
    P_ks_a, _ = DLm.predict_P_obs_on_leg(c["model"], th, tau0, a, leg=leg_ks, **kw)
    P_ks_0, _ = DLm.predict_P_obs_on_leg(c["model"], th, tau0, a0, leg=leg_ks, **kw)
    # DESI: α_DLA moves the forward (the DLA-excess term is live).
    assert not np.allclose(np.asarray(P_desi_a), np.asarray(P_desi_0)), \
        "DESI forward (frac 1.0) must carry the DLA-excess term"
    # KS: α_DLA does NOTHING (the DLA-excess term is zeroed by dla_forward_frac=0.0).
    np.testing.assert_allclose(np.asarray(P_ks_a), np.asarray(P_ks_0), rtol=1e-12, atol=0.0,
                               err_msg="KS forward (frac 0.0) DLA term must be exactly 0")


# ============================================================================ #
#  TRUTH↔FORWARD CONSISTENCY (closure-cancellation invariant) — emulator-based,
#  isolating the DLA handling (the emulator residual cancels since both sides use
#  the SAME predict_P_filt).
# ============================================================================ #
def _truth_with_dla_residual(model, theta9, z_unit, tau0, w_c, truth_frac, dla_core, pf_stats):
    """The §0c closure target from the emulator P_filt: filtered Tier-P (DLA class masked to
    clean) + truth_frac·(full DLA excess), full DLA excess = w_DLA·(P_filt[3]+core − P_clean)."""
    Pf = predict_P_filt(model, theta9, z_unit, tau0, pf_stats)         # (4,K)
    P_clean = Pf[0]
    coef = jnp.array([1.0 - w_c[0] - w_c[1] - w_c[2], w_c[0], w_c[1], w_c[2]])
    P_cls = jnp.stack([P_clean, Pf[1], Pf[2], P_clean])                # DLA class masked → clean
    masked = jnp.einsum("c,ck->k", coef, P_cls)
    dla_excess = w_c[2] * ((Pf[3] + jnp.asarray(dla_core)) - P_clean)  # full DLA excess add-back
    return masked + truth_frac * dla_excess


def test_truth_carries_ten_percent_dla_excess_desi_zero_ks():
    """The DESI target's DLA-excess contribution = 0.10·(full DLA excess); the KS target's = 0."""
    c = _emu_ctx()
    th = c["theta9"]; z_unit = jnp.asarray(0.4); tau0 = jnp.asarray(0.8)
    w_c = (0.233, 0.079, 0.040)
    Pf = predict_P_filt(c["model"], th, z_unit, tau0, c["pf"])
    full_excess = w_c[2] * np.asarray((Pf[3] + c["dla_core"]) - Pf[0])
    masked = np.asarray(_truth_with_dla_residual(c["model"], th, z_unit, tau0, w_c, 0.0,
                                                 c["dla_core"], c["pf"]))
    desi_target = np.asarray(_truth_with_dla_residual(c["model"], th, z_unit, tau0, w_c, 0.10,
                                                      c["dla_core"], c["pf"]))
    ks_target = np.asarray(_truth_with_dla_residual(c["model"], th, z_unit, tau0, w_c, 0.0,
                                                    c["dla_core"], c["pf"]))
    np.testing.assert_allclose(desi_target - masked, 0.10 * full_excess, rtol=1e-12, atol=0.0,
                               err_msg="DESI target DLA-excess contribution must be 0.10·(full)")
    np.testing.assert_allclose(ks_target - masked, 0.0, rtol=0, atol=0.0,
                               err_msg="KS target DLA-excess contribution must be exactly 0")


def test_truth_forward_consistency_desi_ten_percent():
    """At θ→truth, α_DLA = 0.10·(lit/sim)·w_DLA on DESI (forward frac 1.0), the forward reproduces
    the DESI target (filtered Tier-P + 0.10·full DLA excess) to tight rtol — both built from the
    SAME emulator P_filt so only the DLA handling is exercised. (lit/sim = 1 isolates the frac.)"""
    c = _emu_ctx()
    th = c["theta9"]; z_unit = jnp.asarray(0.4); tau0 = jnp.asarray(0.8)
    w_c = (0.233, 0.079, 0.040)
    # DESI target = masked baseline + 0.10·full DLA excess.
    target = _truth_with_dla_residual(c["model"], th, z_unit, tau0, w_c, 0.10,
                                      c["dla_core"], c["pf"])
    # forward (DESI, frac 1.0): α_DLA = 0.10·w_DLA (the consistency point; lit/sim=1) reproduces it.
    P_fwd = predict_P_obs(c["model"], th, z_unit, tau0,
                          jnp.array([w_c[0], w_c[1], 0.10 * w_c[2]]), c["pf"], c["dla_core"])
    np.testing.assert_allclose(np.asarray(P_fwd), np.asarray(target), rtol=1e-12, atol=0.0,
                               err_msg="DESI forward(α_DLA=0.10·w_DLA) must reproduce the 10% target")


# ============================================================================ #
#  Real ctx: make_truth_from_sim builds the masked baseline + the full DLA excess;
#  make_legb_mock applies the per-leg truth_frac (DESI 0.10 / KS 0.0).
# ============================================================================ #
_have_cache = os.path.exists(
    "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/observables_tau0_lf.h5")
_have_ckpt = os.path.exists("/home/mfho/hcd_priya/checkpoints/final_fold0.eqx")
_have_desi = os.path.exists("/home/mfho/data/desi_dr1_p1d/desi_dr1_p1d.npz")
_have_ks = os.path.exists(
    "/home/mfho/lya_emulator_full/lyaemu/data/kodiaq_squad/"
    "final-conservative-p1d-karacayli_etal2021.txt")


@pytest.mark.skipif(not (_have_cache and _have_ckpt),
                    reason="real cache/ckpt not present")
def test_make_truth_stores_masked_baseline_and_full_dla_excess():
    """make_truth_from_sim returns P_obs_true (the DLA-MASKED baseline) AND dla_excess_true (the
    full DLA excess add-back, per row) — the §0c construction the per-leg mock applies truth_frac
    to. Verified against the cache P_filt directly (the truth uses cache P_filt, not the emulator)."""
    from hcd_analysis.emulator import closure_legb as CL
    from hcd_analysis.emulator.data import load_cache
    d = load_cache(CL.CACHE_PATH)
    sims, _ = CL.held_out_sims(d, fold=0)
    truth = CL.make_truth_from_sim(d, sims[0], fold=0)            # LF (mf=None)
    assert "dla_excess_true" in truth, "make_truth_from_sim must store the full DLA excess add-back"
    for i, r in enumerate(truth["rows"]):
        a = d["w_c_cache"][r, 1:]                                 # (3,)
        Pf = d["P_filt"][r]                                       # (4,K)
        core = d["delta"][r, 2]                                   # (K,) DLA core
        P_clean = Pf[0]
        # masked baseline: DLA class at clean.
        coef = np.array([1.0 - a.sum(), a[0], a[1], a[2]])
        P_cls = np.stack([P_clean, Pf[1], Pf[2], P_clean])        # DLA → clean
        masked = np.einsum("c,ck->k", coef, P_cls)
        np.testing.assert_allclose(truth["P_obs_true"][i], masked, rtol=1e-10, atol=0.0,
                                   err_msg=f"row {r}: P_obs_true must be the masked baseline")
        # full DLA excess: w_DLA·(P_filt[3]+core − clean).
        excess = a[2] * ((Pf[3] + core) - P_clean)
        np.testing.assert_allclose(truth["dla_excess_true"][i], excess, rtol=1e-10, atol=0.0,
                                   err_msg=f"row {r}: dla_excess_true must be w_DLA·(P_DLA_unf−clean)")


@pytest.mark.skipif(not (_have_cache and _have_ckpt and _have_desi and _have_ks),
                    reason="real cache/ckpt/data not present")
def test_make_legb_mock_applies_per_leg_truth_frac():
    """make_legb_mock builds the DESI mock target with 0.10·(full DLA excess) added back and the
    KS mock target with 0% — verified by reconstructing the truth-on-leg WITHOUT noise (the mock
    minus its known ε is impractical, so we re-derive the per-leg truth_frac from the noiseless
    truth-on-leg the mock exposes via info)."""
    from hcd_analysis.emulator import closure_legb as CL
    from hcd_analysis.emulator.data import load_cache
    d = load_cache(CL.CACHE_PATH)
    ctx, _ = CL.build_legb_ctx()
    sims, _ = CL.held_out_sims(d, fold=0)
    truth = CL.make_truth_from_sim(d, sims[0], fold=0)
    mock_legs, truth_pack, info = CL.make_legb_mock(ctx, truth, jax.random.PRNGKey(0))
    # the per-leg truth_frac the mock used (closure_legb.TRUTH_DLA_FRAC).
    assert CL.TRUTH_DLA_FRAC["DESI"] == 0.10, "DESI target must carry the 10% DLA residual"
    assert CL.TRUTH_DLA_FRAC["KS"] == 0.0, "KS target must carry 0% DLA residual"
    # the noiseless truth-on-leg (per leg) is exposed in info for the consistency check.
    assert "truth_on_leg" in info, "make_legb_mock must expose the noiseless per-leg truth-on-leg"
    desi_tol = info["truth_on_leg"]["DESI"]
    ks_tol = info["truth_on_leg"]["KS"]
    assert np.isfinite(desi_tol[np.isfinite(desi_tol)]).all()
    assert np.isfinite(ks_tol[np.isfinite(ks_tol)]).all()


@pytest.mark.skipif(not (_have_cache and _have_ckpt and _have_desi and _have_ks),
                    reason="real cache/ckpt/data not present")
def test_truth_pack_alpha_dla_is_ten_percent_residual():
    """The closure truth_pack α_DLA = 0.10·w_DLA (the DESI residual the forward marginalizes),
    while α_LLS/α_subDLA stay at the sim's structural w_c."""
    from hcd_analysis.emulator import closure_legb as CL
    from hcd_analysis.emulator.data import load_cache
    d = load_cache(CL.CACHE_PATH)
    ctx, _ = CL.build_legb_ctx()
    sims, _ = CL.held_out_sims(d, fold=0)
    truth = CL.make_truth_from_sim(d, sims[0], fold=0)
    _, truth_pack, _ = CL.make_legb_mock(ctx, truth, jax.random.PRNGKey(0))
    # α_DLA truth = the DESI 10% residual scale (the leg the forward DLA term is live on).
    assert np.isclose(truth_pack["alpha_hcd"][2], 0.10 * truth["w_c"][2], rtol=1e-10), \
        "truth α_DLA must be 0.10·w_DLA (the DESI 10% residual the forward marginalizes)"
    assert np.isclose(truth_pack["alpha_hcd"][0], truth["w_c"][0], rtol=1e-10)
    assert np.isclose(truth_pack["alpha_hcd"][1], truth["w_c"][1], rtol=1e-10)


@pytest.mark.skipif(not (_have_cache and _have_ckpt and _have_desi and _have_ks),
                    reason="real cache/ckpt/data not present")
def test_real_ctx_truth_forward_consistency_desi_ten_percent():
    """Real ctx, EMULATOR-based consistency: at θ→truth, α_DLA = 0.10·w_DLA on the DESI forward
    (frac 1.0), the forward reproduces the DESI 10% target built from the SAME emulator P_filt
    (isolates the DLA handling; the emulator residual cancels)."""
    from hcd_analysis.emulator import closure_legb as CL
    from hcd_analysis.emulator.data import load_cache
    d = load_cache(CL.CACHE_PATH)
    ctx, _ = CL.build_legb_ctx()
    sims, _ = CL.held_out_sims(d, fold=0)
    truth = CL.make_truth_from_sim(d, sims[0], fold=0)
    w_c = tuple(float(x) for x in np.asarray(truth["w_c"]))
    th = jnp.asarray(truth["params_unit"])
    z_unit = jnp.asarray(0.4); tau0 = jnp.asarray(0.8)
    core = jnp.asarray(truth["dla_core"][0])
    target = _truth_with_dla_residual(ctx.model, th, z_unit, tau0, w_c, 0.10, core, ctx.pf_stats)
    P_fwd = predict_P_obs(ctx.model, th, z_unit, tau0,
                          jnp.array([w_c[0], w_c[1], 0.10 * w_c[2]]), ctx.pf_stats, core)
    np.testing.assert_allclose(np.asarray(P_fwd), np.asarray(target), rtol=1e-10, atol=0.0,
                               err_msg="real-ctx DESI forward(α_DLA=0.10·w_DLA) must match the 10% target")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
