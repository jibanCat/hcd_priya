"""TDD suite for scripts/run_ks_selboost_shard.py + scripts/batch_ks_selboost.sh (the KS
selection-boost campaign runner; spec Sec 5 + A2, PI-signed 2026-07-19).

Covers the PURE, no-cluster pieces (the ctx-building parity asserts run inside the runner at
launch and in the SMOKE; here we test the extracted helpers with stubs, mirroring the
test_dla_selfdraw_arm.py forward_stamp-stub convention):

  * assert_prior_state: the K1a constant-swap TRIPWIRE — required --expect-lls-boost /
    --expect-lls-frac-sigma must equal the DEPLOYED inference constants; the deployed LLS
    center must equal the post-correction K1a values EXACTLY (lit/sim 0.995, slope 0.764,
    widened base sigma 0.287 / hedge 0.574, realfit z-slope 2.127); returns the 5-constant
    stamp dict the analyzer re-asserts.
  * assert_ks_forward: the KS parity block (prod_forward_config("KS") + NORC + echelle
    k_max 0.065 live (NOT the 0.045 NORC auto-cap) + metals OFF + dla_forward_frac == 0.0,
    certifying the DLA-ride-along no-op) — stub-driven trip tests.
  * arm_run_modes: K0 is clean-only, boosted arms boost-only (cross-pkl pairing).
  * pkl naming + .smoke suffix convention; batch array->(arm, mock) mapping totals 108.

Env: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_ks_selboost_runner.py -q
"""
from types import SimpleNamespace

import numpy as np
import pytest

import hcd_analysis.emulator  # noqa: F401  (x64)
import hcd_analysis.emulator.inference as INF
import scripts.ks_selboost_arms as AR
import scripts.run_ks_selboost_shard as R

RUNNER = "/home/mfho/hcd_priya/scripts/run_ks_selboost_shard.py"
BATCH = "/home/mfho/hcd_priya/scripts/batch_ks_selboost.sh"


# --------------------------------------------------------------------------------------------- #
#  Prior-state tripwire (spec 5c): forward_signature does NOT cover prior constants — this is
#  the mechanical launch gate behind the K1a swap.
# --------------------------------------------------------------------------------------------- #
def test_prior_state_passes_on_deployed_constants():
    stamp = R.assert_prior_state(INF.HCD_LLS_SURVEY_BOOST["KS"],
                                 INF.HCD_LLS_SURVEY_FRAC_SIGMA["KS"])
    # the 5 prior constants stamped into meta (analyzer re-asserts homogeneity)
    assert stamp["lls_survey_boost_ks"] == 2.5
    assert stamp["lls_frac_sigma_ks"] == 0.40
    assert stamp["lit_over_sim_lls"] == 0.995
    assert stamp["lls_base_frac_sigma"] == 0.287
    assert stamp["lls_realfit_zslope"] == 2.127
    assert stamp["hcd_prior_signature"] == INF.hcd_prior_signature()


@pytest.mark.parametrize("boost,sigma", [(1.0, 0.40), (2.5, 0.15), (3.0, 0.40), (2.5, 0.287)])
def test_prior_state_trips_on_wrong_expectations(boost, sigma):
    with pytest.raises(AssertionError):
        R.assert_prior_state(boost, sigma)


# --------------------------------------------------------------------------------------------- #
#  KS forward parity block (spec 5b) — stub-driven trip tests of each load-bearing assert.
# --------------------------------------------------------------------------------------------- #
def _stub_ctx(**kw):
    base = dict(res_corr_on=False, fix_alpha_res=True, sample_res=True, f_res_amp_sigma=0.15,
                metal_prior="uniform", sample_metals=False)
    base.update(kw)
    return SimpleNamespace(**base)


def _stub_leg(**kw):
    base = dict(name="KS", metals_on=False, dla_forward_frac=0.0,
                z=np.arange(2.4, 4.61, 0.2), k=np.linspace(0.006, 0.0649, 40),
                dla_cov_reduced=False)
    base.update(kw)
    return SimpleNamespace(**base)


def test_ks_forward_parity_passes_on_deployed_stub():
    R.assert_ks_forward(_stub_ctx(), _stub_leg())      # no raise


@pytest.mark.parametrize("ctx_kw,leg_kw,frag", [
    (dict(res_corr_on=True), {}, "NORC"),
    (dict(fix_alpha_res=False), {}, "NORC"),
    (dict(sample_res=False), {}, "f_res"),
    (dict(f_res_amp_sigma=0.02), {}, "f_res"),
    (dict(sample_metals=True), {}, "metal"),
    ({}, dict(metals_on=True), "metal"),
    ({}, dict(name="DESI"), "KS"),
    ({}, dict(dla_forward_frac=1.0), "dla_forward_frac"),
    ({}, dict(z=np.arange(2.0, 4.61, 0.2)), "z"),
    ({}, dict(k=np.linspace(0.006, 0.09, 40)), "k"),
    # the 0.045 NORC auto-cap silently replacing the certified echelle 0.065 window must trip
    ({}, dict(k=np.linspace(0.006, 0.044, 40)), "0.045"),
])
def test_ks_forward_parity_trips(ctx_kw, leg_kw, frag):
    with pytest.raises(AssertionError, match=frag):
        R.assert_ks_forward(_stub_ctx(**ctx_kw), _stub_leg(**leg_kw))


# --------------------------------------------------------------------------------------------- #
#  Modes, naming, batch mapping.
# --------------------------------------------------------------------------------------------- #
def test_arm_run_modes():
    assert R.arm_run_mode("K0_clean") == "clean"
    for aid in AR.arm_ids():
        if aid != "K0_clean":
            assert R.arm_run_mode(aid) == "boost"


def test_pkl_naming_convention():
    assert R.shard_pkl_name("K0_clean", 3, smoke=False) == "ks_selboost_clean_shard_003.pkl"
    assert R.shard_pkl_name("K3_rising", 0, smoke=False) == "ks_selboost_K3_rising_shard_000.pkl"
    assert R.shard_pkl_name("K3_rising", 0, smoke=True) == \
        "ks_selboost_K3_rising_shard_000.smoke.pkl"


def test_batch_array_mapping_totals():
    """The batch maps SLURM array ids onto (arm, mock) cells over the signed default matrix:
    108 tasks, K0 first (16), arms in registry order."""
    cells = R.batch_cells()
    assert len(cells) == 108
    assert cells[0] == ("K0_clean", 0) and cells[15] == ("K0_clean", 15)
    assert cells[16] == ("K1_flat_lo", 0)
    # per-arm counts match the registry Ns
    from collections import Counter
    counts = Counter(a for a, _ in cells)
    for aid in AR.default_arm_ids():
        assert counts[aid] == AR.ARMS[aid]["n_mocks"]


def test_runner_source_conventions():
    with open(RUNNER) as fh:
        src = fh.read()
    # stamp authority + prior tripwire + registry + span logging + smoke suffix
    assert "forward_stamp(" in src, "meta['forward'] must come from closure_legb.forward_stamp"
    assert "--expect-lls-boost" in src and "--expect-lls-frac-sigma" in src
    assert "registry_signature" in src
    assert "assert_hcd_pivot_z3" in src, "band assert on the built KS LLS prior center"
    assert "w_c_cache" in src, "emulator-span logging (risk 12) must reference the cache grid"
    assert '".smoke"' in src, "smoke mode must write a .smoke-suffixed pkl"
    assert "use_prod_forward=True" in src


def test_batch_source_conventions():
    with open(BATCH) as fh:
        src = fh.read()
    assert "--array" in src or "SLURM_ARRAY_TASK_ID" in src
    assert "cavestru1" in src
    assert "SKIP" in src or "exists" in src.lower()          # skip-if-exists
    assert "SMOKE" in src
