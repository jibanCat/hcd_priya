"""Task #4 — wire the certified production data-nuisance forward into BOTH drivers (TDD).

The real-fit forward (run_real_fit.build_real_ctx) and the production SBC forward
(run_prod_sbc_shard) MUST be identical: same f_res float (option-b), same flat-log 2-node metals.
Today both leave those knobs at build_legb_ctx defaults (sample_res=False, metal_prior='uniform'),
so the deployed forward carries NO resolution float and uniform-scalar metals -- not the config any
gate certified. These tests pin:

  1. prod_forward_config -- the SINGLE source of truth both drivers consume (real-fit==SBC).
  2. build_real_ctx forwards those knobs into build_legb_ctx, per leg (DESI 0.02 / eBOSS 0.05 /
     KS off), verified with a spy so no heavy ensemble/NUTS runs.
  3. The SBC run_cfg population stamp gains sample_res/f_res_amp_sigma/metal_prior so a wired pkl can
     never silently pool with a pre-wiring (uniform/no-f_res) pkl, while a default resume still loads.

Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_prod_forward_wiring.py -q
"""
import importlib.util
import os
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

REPO = "/home/mfho/hcd_priya"


def _load_script(name):
    """Load a scripts/<name>.py module by path (scripts/ is not an importable package)."""
    path = os.path.join(REPO, "scripts", f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------------------------- #
#  1. The single source of truth: prod_forward_config (closure_legb).
# --------------------------------------------------------------------------------------------- #
def test_prod_forward_config_certified_values():
    from hcd_analysis.emulator import closure_legb as CL
    desi = CL.prod_forward_config("DESI")
    assert desi["sample_res"] is True and desi["f_res_amp_sigma"] == 0.02
    assert desi["metal_prior"] == "flatlog2node" and desi["metals"] is True
    eboss = CL.prod_forward_config("eBOSS")
    assert eboss["sample_res"] is True and eboss["f_res_amp_sigma"] == 0.05
    assert eboss["metal_prior"] == "flatlog2node" and eboss["metals"] is True
    ks = CL.prod_forward_config("KS")
    assert ks["sample_res"] is True and ks["f_res_amp_sigma"] == 0.15
    assert ks["metal_prior"] == "uniform" and ks["metals"] is False
    assert ks["ks_kwargs"] == {"resolution_float": True, "k_max": 0.065}
    # DESI/eBOSS carry ks_kwargs=None (their KS leg stays the proxy default)
    assert CL.prod_forward_config("DESI")["ks_kwargs"] is None
    assert CL.prod_forward_config("eBOSS")["ks_kwargs"] is None


def test_prod_forward_config_unknown_leg_raises():
    from hcd_analysis.emulator import closure_legb as CL
    with pytest.raises(KeyError):
        CL.prod_forward_config("XQ100")


def test_prod_forward_config_returns_a_copy():
    from hcd_analysis.emulator import closure_legb as CL
    a = CL.prod_forward_config("DESI")
    a["sample_res"] = "MUTATED"
    assert CL.prod_forward_config("DESI")["sample_res"] is True   # not shared (top-level)
    ks = CL.prod_forward_config("KS")
    ks["ks_kwargs"]["k_max"] = "MUTATED"                          # mutate the NESTED dict
    assert CL.prod_forward_config("KS")["ks_kwargs"]["k_max"] == 0.065   # deepcopy protected


# --------------------------------------------------------------------------------------------- #
#  2. build_real_ctx forwards the certified knobs into build_legb_ctx (spy; no ensemble/NUTS).
# --------------------------------------------------------------------------------------------- #
class _StopBuild(Exception):
    pass


def _capture_build_real_ctx_kwargs(survey, monkeypatch):
    rf = _load_script("run_real_fit")
    captured = {}

    def _spy(**kw):
        captured.update(kw)
        raise _StopBuild()

    monkeypatch.setattr(rf, "build_legb_ctx", _spy)
    with pytest.raises(_StopBuild):
        rf.build_real_ctx(survey)
    return captured


def test_build_real_ctx_desi_forwards_certified_knobs(monkeypatch):
    kw = _capture_build_real_ctx_kwargs("desi", monkeypatch)
    assert kw["sample_res"] is True
    assert kw["f_res_amp_sigma"] == 0.02
    assert kw["metal_prior"] == "flatlog2node"
    assert kw["metals_on"] is True and kw["sample_metals"] is True


def test_build_real_ctx_eboss_width_005(monkeypatch):
    kw = _capture_build_real_ctx_kwargs("eboss", monkeypatch)
    assert kw["sample_res"] is True
    assert kw["f_res_amp_sigma"] == 0.05
    assert kw["metal_prior"] == "flatlog2node"


def test_build_real_ctx_ks_forwards_fres_and_kskwargs(monkeypatch):
    kw = _capture_build_real_ctx_kwargs("ks", monkeypatch)
    assert kw["sample_res"] is True
    assert kw["f_res_amp_sigma"] == 0.15
    assert kw["metal_prior"] == "uniform"
    assert kw["metals_on"] is False and kw["sample_metals"] is False
    assert kw["ks_kwargs"] == {"resolution_float": True, "k_max": 0.065}


def test_build_real_ctx_desi_eboss_kskwargs_none(monkeypatch):
    for survey in ("desi", "eboss"):
        kw = _capture_build_real_ctx_kwargs(survey, monkeypatch)
        assert kw["ks_kwargs"] is None            # KS leg stays proxy default (byte-identical)


# --------------------------------------------------------------------------------------------- #
#  3. The SBC run_cfg population stamp discriminates the wired forward from a pre-wiring pkl.
#     _run_mock, on the SKIP branch (pkl already exists), never runs NUTS -- it only checks cfg.
# --------------------------------------------------------------------------------------------- #
def _write_pkl(path, run_cfg):
    rec = {"n_div": 0, "mock": 0}
    if run_cfg is not None:
        rec["run_cfg"] = run_cfg
    with open(path, "wb") as f:
        pickle.dump(rec, f)


# An OLD-STAMP run_cfg = a real pre-2026-07-07 (pre-wiring) DESI-NORC pkl: it HAS the older
# discriminators but LACKS the three new forward keys (sample_res/f_res_amp_sigma/metal_prior).
def _old_stamp_cfg():
    return dict(leg_a=True, cemu_variant="current", amp_sigma=0.0, leg="DESI", fold=0,
                tau0_prior_sigma=0.0, subdla_truth_boost=1.0, res_corr_on=False)


def _wired_desi_cfg():
    # the run_cfg a WIRED DESI SBC run stamps, from the shared source (tracks the certified values).
    from hcd_analysis.emulator import closure_legb as CL
    fc = CL.prod_forward_config("DESI")
    return dict(_old_stamp_cfg(), sample_res=fc["sample_res"],
                f_res_amp_sigma=fc["f_res_amp_sigma"], metal_prior=fc["metal_prior"])


def test_wired_cfg_clashes_with_pre_stamp_pkl(tmp_path):
    sbc = _load_script("run_prod_sbc_shard")
    path = sbc._mock_path(str(tmp_path), 0)
    _write_pkl(path, None)                         # a PRE-WIRING pkl (no run_cfg stamp at all)
    with pytest.raises(RuntimeError):              # a wired (flatlog2node) request must CLASH, not load
        sbc._run_mock(None, None, 0, str(tmp_path), n_mocks=1, n_warmup=1, n_samples=1,
                      max_tree_depth=1, seed=0, run_cfg=_wired_desi_cfg())


def test_wired_cfg_clashes_with_old_stamp_pkl(tmp_path):
    # the load-bearing phantom-pooling guard: a WIRED run must CLASH with an OLD-STAMP (pre-wiring) pkl.
    sbc = _load_script("run_prod_sbc_shard")
    path = sbc._mock_path(str(tmp_path), 0)
    _write_pkl(path, _old_stamp_cfg())             # stamped, but lacks the 3 new forward keys
    with pytest.raises(RuntimeError):              # wired keys are NOT popped (non-default) -> clash
        sbc._run_mock(None, None, 0, str(tmp_path), n_mocks=1, n_warmup=1, n_samples=1,
                      max_tree_depth=1, seed=0, run_cfg=_wired_desi_cfg())


def test_default_run_loads_old_stamp_pkl_via_backcompat_pops(tmp_path):
    # a DEFAULT (pre-wiring) resume must STILL load an old-stamp pkl: the missing new keys, requested at
    # their pre-wiring defaults (False/None/uniform), are POPPED so the compare matches (no false clash).
    sbc = _load_script("run_prod_sbc_shard")
    path = sbc._mock_path(str(tmp_path), 0)
    _write_pkl(path, _old_stamp_cfg())
    default_req = dict(_old_stamp_cfg(), sample_res=False, f_res_amp_sigma=None, metal_prior="uniform")
    rec = sbc._run_mock(None, None, 0, str(tmp_path), n_mocks=1, n_warmup=1, n_samples=1,
                        max_tree_depth=1, seed=0, run_cfg=default_req)
    assert rec["n_div"] == 0                        # loaded via the back-compat pops, no clash


# --------------------------------------------------------------------------------------------- #
#  4. The gated NORC KS-cap parity assert: lifted ONLY for the echelle-floating KS leg.
# --------------------------------------------------------------------------------------------- #
def _fake_ctx(ks_kmax, resolution_ready):
    ks = SimpleNamespace(name="KS", k=np.array([0.01, ks_kmax]), resolution_ready=resolution_ready)
    return SimpleNamespace(legs=[ks])


def test_norc_ks_cap_helper_lifts_only_for_floating_ks():
    rf = _load_script("run_real_fit")
    with pytest.raises(AssertionError):                      # proxy KS at 0.065 -> RAISE
        rf._assert_norc_ks_cap(_fake_ctx(0.065, resolution_ready=False))
    rf._assert_norc_ks_cap(_fake_ctx(0.065, resolution_ready=True))   # echelle-floating -> no raise
    rf._assert_norc_ks_cap(_fake_ctx(0.045, resolution_ready=False))  # proxy at 0.045 -> fine
