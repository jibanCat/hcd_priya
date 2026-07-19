"""F2 (adversarial backfill 2026-07-19): env-flag data selection must be VISIBLE + REFUSABLE.

data_likelihood resolves HCD_DESI_SNR3 / HCD_CV_FLOOR / HCD_CV_FLOOR_RANK1 from the environment
on every DESI/eBOSS leg load. Before this fix the resolved values appeared in NO signature and NO
pkl stamp: an exported env var could silently swap the DESI measurement (SNR>3 npz + its own cov)
or add the CV floor under a production driver, invisible to every audit artifact. Contract:

  (a) the resolved flags are STAMPED on the DataLeg (use_snr3 / cv_floor_on / cv_floor_rank1,
      the dla_cov_reduced pattern) and threaded into meta["forward"] by closure_legb.forward_stamp
      (covered in tests/test_dla_selfdraw_arm.py);
  (b) production drivers (run_real_fit / run_prod_sbc_shard) REFUSE to start with any of the env
      flags set unless --allow-env-data-flags is passed explicitly
      (data_likelihood.assert_env_data_flags_unset);
  (c) an explicit loader kwarg still overrides the env (the test back-door is preserved).

Env: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_env_data_flags.py -q
"""
import os

import hcd_analysis.emulator  # noqa: F401  enables jax_enable_x64
import numpy as np
import pytest

from hcd_analysis.emulator import data_likelihood as DL

DESI_NPZ = "/home/mfho/data/desi_dr1_p1d/desi_dr1_p1d.npz"
EBOSS_NPZ = "/home/mfho/data/eboss_dr14_p1d/eboss_dr14_p1d.npz"
_have_desi = os.path.exists(DESI_NPZ) and os.path.exists(DL.DESI_SNR3_NPZ)
_have_eboss = os.path.exists(EBOSS_NPZ)

REPO = "/home/mfho/hcd_priya"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Every test starts from an UNSET env (the production baseline)."""
    for n in ("HCD_DESI_SNR3", "HCD_CV_FLOOR", "HCD_CV_FLOOR_RANK1"):
        monkeypatch.delenv(n, raising=False)


def test_env_data_flags_constant():
    """The tripwire + stamp machinery must cover exactly the three env-resolved data flags."""
    assert tuple(DL.ENV_DATA_FLAGS) == ("HCD_DESI_SNR3", "HCD_CV_FLOOR", "HCD_CV_FLOOR_RANK1")


# --------------------------------------------------------------------------------------------- #
#  (a) DataLeg stamps
# --------------------------------------------------------------------------------------------- #
@pytest.mark.skipif(not _have_desi, reason="DESI npz not present")
def test_desi_default_stamps_all_false():
    leg = DL.load_desi_leg()
    assert leg.use_snr3 is False
    assert leg.cv_floor_on is False
    assert leg.cv_floor_rank1 is False


@pytest.mark.skipif(not _have_desi, reason="DESI npz not present")
def test_desi_env_snr3_is_stamped(monkeypatch):
    monkeypatch.setenv("HCD_DESI_SNR3", "1")
    leg = DL.load_desi_leg()
    assert leg.use_snr3 is True
    assert leg.cv_floor_on is False


@pytest.mark.skipif(not _have_desi, reason="DESI npz not present")
def test_desi_env_cv_floor_is_stamped(monkeypatch):
    monkeypatch.setenv("HCD_CV_FLOOR", "1")
    leg = DL.load_desi_leg()
    assert leg.cv_floor_on is True and leg.cv_floor_rank1 is False
    monkeypatch.setenv("HCD_CV_FLOOR_RANK1", "1")
    leg2 = DL.load_desi_leg()
    assert leg2.cv_floor_on is True and leg2.cv_floor_rank1 is True


@pytest.mark.skipif(not _have_desi, reason="DESI npz not present")
def test_desi_rank1_without_floor_is_not_stamped(monkeypatch):
    """HCD_CV_FLOOR_RANK1 alone is inert (never consulted when the floor is off) — the stamp
    must record what was APPLIED, not the raw env."""
    monkeypatch.setenv("HCD_CV_FLOOR_RANK1", "1")
    leg = DL.load_desi_leg()
    assert leg.cv_floor_on is False and leg.cv_floor_rank1 is False


@pytest.mark.skipif(not _have_eboss, reason="eBOSS npz not present")
def test_eboss_env_cv_floor_is_stamped(monkeypatch):
    leg0 = DL.load_eboss_leg()
    assert leg0.cv_floor_on is False and leg0.cv_floor_rank1 is False and leg0.use_snr3 is False
    monkeypatch.setenv("HCD_CV_FLOOR", "1")
    monkeypatch.setenv("HCD_CV_FLOOR_RANK1", "1")
    leg = DL.load_eboss_leg()
    assert leg.cv_floor_on is True and leg.cv_floor_rank1 is True
    assert leg.use_snr3 is False                      # SNR3 is a DESI-only concept


@pytest.mark.skipif(not _have_eboss, reason="eBOSS npz not present")
def test_explicit_kwarg_overrides_env(monkeypatch):
    """The explicit add_cv_floor=False test back-door still beats the env — and the stamp
    reflects the APPLIED value."""
    monkeypatch.setenv("HCD_CV_FLOOR", "1")
    leg = DL.load_eboss_leg(add_cv_floor=False)
    assert leg.cv_floor_on is False


@pytest.mark.skipif(not os.path.exists(
    "/home/mfho/lya_emulator_full/lyaemu/data/kodiaq_squad/final-conservative-p1d-karacayli_etal2021.txt"),
    reason="KS data not present")
def test_ks_leg_stamps_default_false():
    leg = DL.load_ks_leg()
    assert leg.use_snr3 is False and leg.cv_floor_on is False and leg.cv_floor_rank1 is False


# --------------------------------------------------------------------------------------------- #
#  (b) driver-entry refusal
# --------------------------------------------------------------------------------------------- #
def test_assert_env_data_flags_unset_passes_on_clean_env():
    DL.assert_env_data_flags_unset("test-clean")      # must not raise


def test_assert_env_data_flags_unset_refuses_set_flag(monkeypatch):
    monkeypatch.setenv("HCD_CV_FLOOR", "1")
    with pytest.raises(AssertionError, match="HCD_CV_FLOOR"):
        DL.assert_env_data_flags_unset("test-refuse")
    monkeypatch.setenv("HCD_DESI_SNR3", "yes")
    with pytest.raises(AssertionError, match="HCD_DESI_SNR3"):
        DL.assert_env_data_flags_unset("test-refuse")


def test_assert_env_data_flags_unset_allow_optin(monkeypatch):
    monkeypatch.setenv("HCD_CV_FLOOR", "1")
    DL.assert_env_data_flags_unset("test-allow", allow=True)   # explicit opt-in


def test_assert_env_data_flags_falsy_token_is_unset(monkeypatch):
    """'0'/'false' tokens are FALSE per _env_flag — they must not trip the refusal."""
    monkeypatch.setenv("HCD_CV_FLOOR", "0")
    monkeypatch.setenv("HCD_DESI_SNR3", "false")
    DL.assert_env_data_flags_unset("test-falsy")


def test_production_drivers_wire_the_tripwire():
    """run_real_fit + run_prod_sbc_shard must call the refusal at driver entry and expose the
    explicit --allow-env-data-flags opt-in (source-level wiring tripwire)."""
    for p in (f"{REPO}/scripts/run_real_fit.py", f"{REPO}/scripts/run_prod_sbc_shard.py"):
        with open(p) as fh:
            src = fh.read()
        assert "assert_env_data_flags_unset" in src, f"{p}: env-flag tripwire not wired"
        assert "allow-env-data-flags" in src, f"{p}: missing the explicit opt-in flag"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
