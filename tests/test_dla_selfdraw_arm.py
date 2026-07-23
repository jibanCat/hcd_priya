"""The displaced-truth PRIYA DLA closure arm (PI disposition 2026-07-17, Workstream A).

Covers the two pure pieces the new arm adds (mirroring tests/test_dnuis_inject.py part (b)):

  (a) apply_dla_truth_boost(truth_pack, boost) -- the DLA sibling of the LLS/subDLA truth boosts:
      scales ONLY the DLA truth (pivot alpha_hcd[2] AND the z-resolved alpha_hcd_z[:,2] column);
      returns a copy (input unmutated); boost=1 is the identity. NO e_dla mean template anywhere.

  (b) _apply_truth_boosts(truth_pack, inject_spec) -- the run_legb leg-a threading helper: applies
      any of the three *_truth_boost keys, and FAILS LOUD on an unknown inject_spec key (so a typo
      like "dla_truthboost" can never silently no-op an injection arm).

Env: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_dla_selfdraw_arm.py -q
"""
import numpy as np
import pytest

import hcd_analysis.emulator  # noqa: F401  (x64)
import hcd_analysis.emulator.closure_legb as LB


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
#  (a) apply_dla_truth_boost -- pure helper
# --------------------------------------------------------------------------------------------- #
def test_apply_dla_truth_boost_scales_only_dla():
    tp = _toy_truth_pack()
    tp0 = {k: (np.array(v) if isinstance(v, np.ndarray) else v) for k, v in tp.items()}
    boost = 1.5
    out = LB.apply_dla_truth_boost(tp, boost)
    # DLA pivot + z-resolved column scaled by boost
    np.testing.assert_allclose(out["alpha_hcd"][2], tp0["alpha_hcd"][2] * boost, rtol=1e-12)
    np.testing.assert_allclose(out["alpha_hcd_z"][:, 2], tp0["alpha_hcd_z"][:, 2] * boost, rtol=1e-12)
    # LLS + subDLA columns/pivots UNTOUCHED
    np.testing.assert_allclose(out["alpha_hcd"][:2], tp0["alpha_hcd"][:2], rtol=0, atol=0)
    np.testing.assert_allclose(out["alpha_hcd_z"][:, :2], tp0["alpha_hcd_z"][:, :2], rtol=0, atol=0)
    # theta9 / tau0 / a_siiii untouched
    np.testing.assert_allclose(out["theta9"], tp0["theta9"], rtol=0, atol=0)
    np.testing.assert_allclose(out["tau0_global"], tp0["tau0_global"], rtol=0, atol=0)
    assert float(out["a_siiii"]) == float(tp0["a_siiii"])


def test_apply_dla_truth_boost_returns_copy_input_unmutated():
    tp = _toy_truth_pack()
    a_before = np.array(tp["alpha_hcd"])
    az_before = np.array(tp["alpha_hcd_z"])
    out = LB.apply_dla_truth_boost(tp, 2.0)
    np.testing.assert_allclose(tp["alpha_hcd"], a_before, rtol=0, atol=0)
    np.testing.assert_allclose(tp["alpha_hcd_z"], az_before, rtol=0, atol=0)
    assert out["alpha_hcd"] is not tp["alpha_hcd"]
    assert out["alpha_hcd_z"] is not tp["alpha_hcd_z"]


def test_apply_dla_truth_boost_unit_is_identity():
    tp = _toy_truth_pack()
    out = LB.apply_dla_truth_boost(tp, 1.0)
    np.testing.assert_allclose(out["alpha_hcd"], tp["alpha_hcd"], rtol=0, atol=0)
    np.testing.assert_allclose(out["alpha_hcd_z"], tp["alpha_hcd_z"], rtol=0, atol=0)


# --------------------------------------------------------------------------------------------- #
#  (b) _apply_truth_boosts -- the run_legb threading helper + unknown-key fail-loud
# --------------------------------------------------------------------------------------------- #
def test_apply_truth_boosts_dla_key_threads():
    tp = _toy_truth_pack()
    out = LB._apply_truth_boosts(tp, {"dla_truth_boost": 1.5})
    np.testing.assert_allclose(out["alpha_hcd"][2], tp["alpha_hcd"][2] * 1.5, rtol=1e-12)
    np.testing.assert_allclose(out["alpha_hcd_z"][:, 2], tp["alpha_hcd_z"][:, 2] * 1.5, rtol=1e-12)
    np.testing.assert_allclose(out["alpha_hcd"][:2], tp["alpha_hcd"][:2], rtol=0, atol=0)


def test_apply_truth_boosts_all_three_compose():
    tp = _toy_truth_pack()
    out = LB._apply_truth_boosts(
        tp, {"lls_truth_boost": 2.0, "subdla_truth_boost": 3.0, "dla_truth_boost": 1.5})
    np.testing.assert_allclose(out["alpha_hcd"],
                               tp["alpha_hcd"] * np.array([2.0, 3.0, 1.5]), rtol=1e-12)
    np.testing.assert_allclose(out["alpha_hcd_z"],
                               tp["alpha_hcd_z"] * np.array([2.0, 3.0, 1.5]), rtol=1e-12)


def test_apply_truth_boosts_none_and_empty_are_identity():
    tp = _toy_truth_pack()
    assert LB._apply_truth_boosts(tp, None) is tp
    assert LB._apply_truth_boosts(tp, {}) is tp


def test_apply_truth_boosts_ignores_non_boost_keys():
    """metal_misspec / resolution are make_leg_a_legmock's keys, not boosts -- passed through
    untouched (no error, no truth change)."""
    tp = _toy_truth_pack()
    out = LB._apply_truth_boosts(tp, {"metal_misspec": {"form": "x"}, "resolution": 0.1})
    np.testing.assert_allclose(out["alpha_hcd"], tp["alpha_hcd"], rtol=0, atol=0)


def test_apply_truth_boosts_unknown_key_raises():
    tp = _toy_truth_pack()
    with pytest.raises(ValueError, match="unknown inject_spec key"):
        LB._apply_truth_boosts(tp, {"dla_truthboost": 1.5})     # the typo class, fail loud


# --------------------------------------------------------------------------------------------- #
#  (c) forward_stamp single authority (F2/F5, adversarial backfill 2026-07-19): the
#      meta["forward"] dict for shard pkls is built ONCE in closure_legb, carries BOTH freeze
#      signatures (forward + prior constants) AND the env-resolved data-selection stamps.
# --------------------------------------------------------------------------------------------- #
from types import SimpleNamespace  # noqa: E402


def _stub_ctx(f_res_amp_sigma=0.02):
    return SimpleNamespace(res_corr_on=False, fix_alpha_res=True, sample_res=True,
                           f_res_amp_sigma=f_res_amp_sigma, metal_prior="flatlog2node",
                           metal_node_z=(2.2, 4.2))


def _stub_leg(**kw):
    base = dict(dla_cov_reduced=True, dla_forward_frac=1.0,
                use_snr3=False, cv_floor_on=False, cv_floor_rank1=False)
    base.update(kw)
    return SimpleNamespace(**base)


def test_forward_stamp_authority_keys_and_signatures():
    from hcd_analysis.emulator import inference as INF
    st = LB.forward_stamp(_stub_ctx(), _stub_leg())
    for k in ("res_corr_on", "fix_alpha_res", "sample_res", "f_res_amp_sigma", "metal_prior",
              "metal_node_z", "dla_cov_reduced", "dla_forward_frac",
              "use_snr3", "cv_floor_on", "cv_floor_rank1",
              "forward_signature", "hcd_prior_signature"):
        assert k in st, k
    assert st["forward_signature"] == LB.forward_signature()
    assert st["hcd_prior_signature"] == INF.hcd_prior_signature()   # F5: the prior stamp
    assert st["sample_res"] is True and st["metal_prior"] == "flatlog2node"
    assert st["dla_cov_reduced"] is True and st["dla_forward_frac"] == 1.0
    assert st["use_snr3"] is False and st["cv_floor_on"] is False


def test_forward_stamp_threads_env_data_flags():
    st = LB.forward_stamp(_stub_ctx(), _stub_leg(use_snr3=True, cv_floor_on=True,
                                                 cv_floor_rank1=True))
    assert st["use_snr3"] is True and st["cv_floor_on"] is True and st["cv_floor_rank1"] is True


def test_forward_stamp_none_f_res_sigma_survives():
    st = LB.forward_stamp(_stub_ctx(f_res_amp_sigma=None), _stub_leg())
    assert st["f_res_amp_sigma"] is None


def test_selfdraw_runner_uses_stamp_authority_and_smoke_suffix():
    """(F5) the selfdraw runner's meta['forward'] comes from the single authority (so it now
    carries hcd_prior_signature), the pre-2026-07-19 stamp gap is documented; (F6a) smoke mode
    writes a .smoke-suffixed filename that the analyzer glob can never pool as a real shard."""
    with open("/home/mfho/hcd_priya/scripts/run_dla_selfdraw_shard.py") as fh:
        src = fh.read()
    assert "forward_stamp(" in src, "meta['forward'] must come from closure_legb.forward_stamp"
    assert "latent reconstruction" in src, \
        "the pre-2026-07-19 hcd_prior_signature stamp gap must be documented in the docstring"
    assert '".smoke"' in src, "smoke mode must write a .smoke-suffixed pkl (width-study mirror)"
