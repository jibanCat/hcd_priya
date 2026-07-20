"""The displaced-CENTER LLS closure arm (closing-panel decision P2, PI 2026-07-19).

Mirrors tests/test_dla_selfdraw_arm.py for the LLS axis. The lls_truth_boost inject_spec key is
ALREADY threaded fail-loud through closure_legb._apply_truth_boosts (tests/test_dnuis_inject.py
part (b)), so the new pieces under test here are the runner's pure helpers:

  (a) latent_sigma_displacement -- the exact prior-sigma displacement the boost induces at the
      prior median. alpha_lls's latent IS alpha (TruncatedNormal(mu, 0.287*mu, low=0), sampled
      directly -- NOT the DLA softplus latent whose +0.408 went through softplus^-1), so the
      displacement is (boost-1)*median/sigma: boost 1.287 at frac width 0.287 = +1.0000 exactly
      at the prior center; the low=0 truncnorm median correction at mu/sigma ~ 3.48 is <1e-3.

  (b) assert_deployed_lls_width / lls_prior_stamp -- the fail-loud deployed-width guard (0.287,
      the 1x lit-anchored primary; a hedge2x ctx (0.574) or a KS ctx (0.40) must ABORT the
      campaign, not silently rescale the displacement) and the meta prior stamp.

  (c) re-assert the lls_truth_boost threading contract in this campaign's own file (pivot
      alpha_hcd[0] AND alpha_hcd_z[:,0] scaled; subDLA/DLA/theta9/tau0/a_SiIII untouched).

Env: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_lls_dispcenter_arm.py -q
"""
import numpy as np
import pytest

import hcd_analysis.emulator  # noqa: F401  (x64)
import hcd_analysis.emulator.closure_legb as LB
import scripts.run_lls_dispcenter_shard as R


def _toy_truth_pack(nZg=5):
    rng = np.random.default_rng(0)
    return dict(
        theta9=rng.uniform(0.2, 0.8, 9),
        tau0_global=rng.uniform(0.9, 1.1, nZg),
        alpha_hcd=np.array([0.7, 0.3, 0.05]),                 # (3,) [LLS, subDLA, DLA] pivot
        alpha_hcd_z=np.stack([np.array([0.7, 0.3, 0.05]) * (1.0 + 0.1 * i) for i in range(nZg)]),
        a_siiii=0.02,
        kept_global_z=np.ones(nZg, bool))


# --------------------------------------------------------------------------------------------- #
#  (a) latent_sigma_displacement -- pure helper
# --------------------------------------------------------------------------------------------- #
def test_displacement_center_is_exactly_one_sigma():
    """boost = 1 + frac_sigma => (boost-1)*mu/sigma = frac/frac = 1, for ANY mu."""
    for mu in (0.02, 0.7, 3.1):
        d = R.latent_sigma_displacement(mu, 0.287 * mu, 1.287)
        np.testing.assert_allclose(d, 1.0, rtol=1e-12)


def test_displacement_scales_linearly_in_boost():
    mu = 0.5
    d = R.latent_sigma_displacement(mu, 0.287 * mu, 1.0 + 2 * 0.287)   # +2 fractional sigma
    np.testing.assert_allclose(d, 2.0, rtol=1e-12)
    d = R.latent_sigma_displacement(mu, 0.287 * mu, 1.0 - 0.287)       # the minus-arm config flip
    np.testing.assert_allclose(d, -1.0, rtol=1e-12)


def test_displacement_truncnorm_median_correction_is_negligible():
    """low=0 truncation at mu/sigma = 1/0.287 ~ 3.48: the truncnorm median sits a hair ABOVE mu
    (mass below 0 ~ 2.5e-4 folded up), so the median displacement is >= the center one but within
    1e-3 of +1."""
    mu = 0.7
    d0 = R.latent_sigma_displacement(mu, 0.287 * mu, 1.287)
    dm = R.latent_sigma_displacement(mu, 0.287 * mu, 1.287, low=0.0)
    assert dm >= d0
    assert abs(dm - 1.0) < 1e-3


def test_displacement_unit_boost_is_zero():
    assert R.latent_sigma_displacement(0.7, 0.2, 1.0) == 0.0
    assert R.latent_sigma_displacement(0.7, 0.2, 1.0, low=0.0) == 0.0


# --------------------------------------------------------------------------------------------- #
#  (b) deployed-width guard + prior stamp
# --------------------------------------------------------------------------------------------- #
def test_assert_deployed_lls_width_passes_at_0287():
    mu = 0.6153
    R.assert_deployed_lls_width(mu, 0.287 * mu)                        # no raise


def test_assert_deployed_lls_width_rejects_hedge_and_ks_widths():
    mu = 0.6153
    with pytest.raises(AssertionError, match="LLS prior width"):
        R.assert_deployed_lls_width(mu, 0.574 * mu)                    # hedge2x ctx
    with pytest.raises(AssertionError, match="LLS prior width"):
        R.assert_deployed_lls_width(mu, 0.40 * mu)                     # KS ctx


def test_default_boost_is_plus_one_fractional_sigma():
    np.testing.assert_allclose(R.DEFAULT_BOOST, 1.0 + R.LLS_FRAC_SIGMA_DEPLOYED, rtol=0, atol=0)
    np.testing.assert_allclose(R.LLS_FRAC_SIGMA_DEPLOYED, 0.287, rtol=0, atol=0)


def test_lls_prior_stamp_contents():
    mu = 0.55
    st = R.lls_prior_stamp(mu, 0.287 * mu, 1.287)
    assert set(st) == {"alpha_lls_mu", "alpha_lls_sigma", "frac_sigma", "boost",
                       "latent_sigma_disp_center", "latent_sigma_disp_median"}
    np.testing.assert_allclose(st["alpha_lls_mu"], mu, rtol=1e-12)
    np.testing.assert_allclose(st["frac_sigma"], 0.287, rtol=1e-12)
    np.testing.assert_allclose(st["latent_sigma_disp_center"], 1.0, rtol=1e-12)
    assert abs(st["latent_sigma_disp_median"] - 1.0) < 1e-3
    assert all(isinstance(v, float) for v in st.values())              # pkl/meta-safe scalars


def test_forward_stamp_carries_both_signatures():
    """meta['forward'] must stamp BOTH freeze-audit hashes: closure_legb.forward_signature()
    (forward decision set) AND inference.hcd_prior_signature() (prior constants; the width-study
    runner precedent) -- adversarial-review requirement, so a reader can reject drifted-forward
    OR drifted-prior pkls."""
    import hcd_analysis.emulator.inference as INF

    class _Ctx:
        res_corr_on, fix_alpha_res, sample_res = False, True, True
        f_res_amp_sigma, metal_prior, metal_node_z = 0.1, "flatlog2node", (2.2, 4.2)

    class _Leg:
        dla_cov_reduced, dla_forward_frac = True, 1.0

    st = R.forward_stamp(_Ctx(), _Leg())
    assert st["forward_signature"] == LB.forward_signature()
    assert st["hcd_prior_signature"] == INF.hcd_prior_signature()
    for key in ("forward_signature", "hcd_prior_signature"):
        assert isinstance(st[key], str) and len(st[key]) == 64      # sha256 hex, non-vacuous
        int(st[key], 16)
    assert st["res_corr_on"] is False and st["fix_alpha_res"] is True
    assert st["dla_cov_reduced"] is True and st["dla_forward_frac"] == 1.0
    assert st["metal_prior"] == "flatlog2node" and st["metal_node_z"] == (2.2, 4.2)


# --------------------------------------------------------------------------------------------- #
#  (c) lls_truth_boost threading contract (campaign-local re-assert)
# --------------------------------------------------------------------------------------------- #
def test_apply_truth_boosts_lls_key_threads():
    tp = _toy_truth_pack()
    out = LB._apply_truth_boosts(tp, {"lls_truth_boost": 1.287})
    np.testing.assert_allclose(out["alpha_hcd"][0], tp["alpha_hcd"][0] * 1.287, rtol=1e-12)
    np.testing.assert_allclose(out["alpha_hcd_z"][:, 0], tp["alpha_hcd_z"][:, 0] * 1.287,
                               rtol=1e-12)
    # subDLA + DLA columns/pivots, theta9, tau0, a_SiIII UNTOUCHED; input unmutated
    np.testing.assert_allclose(out["alpha_hcd"][1:], tp["alpha_hcd"][1:], rtol=0, atol=0)
    np.testing.assert_allclose(out["alpha_hcd_z"][:, 1:], tp["alpha_hcd_z"][:, 1:], rtol=0, atol=0)
    np.testing.assert_allclose(out["theta9"], tp["theta9"], rtol=0, atol=0)
    np.testing.assert_allclose(out["tau0_global"], tp["tau0_global"], rtol=0, atol=0)
    assert float(out["a_siiii"]) == float(tp["a_siiii"])
    np.testing.assert_allclose(tp["alpha_hcd"], [0.7, 0.3, 0.05], rtol=0, atol=0)


def test_apply_truth_boosts_unknown_lls_typo_raises():
    with pytest.raises(ValueError, match="unknown inject_spec key"):
        LB._apply_truth_boosts(_toy_truth_pack(), {"lls_truthboost": 1.287})
