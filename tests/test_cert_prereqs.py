"""Certification-plan prerequisites (2026-07-23): ARM-P stamps/pooling, era-aware analyzer,
R6 paired machinery. Cheap paths only (no ctx build, no NUTS) — the heavy pair-identity
property lives in tests/test_r6_pairing.py.

Run: /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_cert_prereqs.py -q
"""
import importlib
import os
import pickle
import sys

import numpy as np
import pytest

sys.path.insert(0, "/home/mfho/hcd_priya")

runner = importlib.import_module("scripts.run_prod_sbc_shard")
AKS = importlib.import_module("scripts.analyze_ks_selboost")
R6 = importlib.import_module("scripts.analyze_r6_pairs")
ksr = importlib.import_module("scripts.run_ks_selboost_shard")

_DUMMY = dict(n_mocks=1, n_warmup=1, n_samples=1, max_tree_depth=1, seed=0, verbose=False)
# the full modern closure-run stamp (survey=None etc. — the ARM-P-era defaults)
_CLOSURE = dict(leg_a=True, cemu_variant="current", amp_sigma=0.0, leg="KS", fold=0,
                tau0_prior_sigma=0.0, subdla_truth_boost=1.0, res_corr_on=False,
                sample_res=True, f_res_amp_sigma=0.15, metal_prior="uniform", ks_kmax=0.065,
                survey=None, hcd_parameterization="alpha_pivot_powerlaw_v1",
                hcd_prior_signature=None, single_member=False)
_ARMP_KS = dict(_CLOSURE, survey="KS", hcd_parameterization="dndx_mapped_v2",
                hcd_prior_signature="s" * 64)


def _write_stub(out_dir, m, run_cfg, marker="ondisk"):
    os.makedirs(out_dir, exist_ok=True)
    rec = dict(_marker=marker)
    if run_cfg is not None:
        rec["run_cfg"] = dict(run_cfg)
    with open(runner._mock_path(out_dir, m), "wb") as f:
        pickle.dump(rec, f)


# ------------------------------- ARM-P pkl pooling (run_prod_sbc_shard) ----------------------

def test_armp_request_clashes_with_closure_pkl(tmp_path):
    """An ARM-P run over an existing closure-prior pkl must CLASH (survey/signature differ)."""
    out = str(tmp_path / "clash_armp")
    _write_stub(out, 0, _CLOSURE)
    with pytest.raises(RuntimeError, match="config CLASH"):
        runner._run_mock(None, None, 0, out, run_cfg=dict(_ARMP_KS), **_DUMMY)


def test_closure_request_clashes_with_armp_pkl(tmp_path):
    """The reverse direction: a closure resume over an ARM-P pkl must CLASH."""
    out = str(tmp_path / "clash_closure")
    _write_stub(out, 0, _ARMP_KS)
    with pytest.raises(RuntimeError, match="config CLASH"):
        runner._run_mock(None, None, 0, out, run_cfg=dict(_CLOSURE), **_DUMMY)


def test_armp_resume_over_armp_loads(tmp_path):
    out = str(tmp_path / "armp_resume")
    _write_stub(out, 0, _ARMP_KS, marker="armp")
    rec = runner._run_mock(None, None, 0, out, run_cfg=dict(_ARMP_KS), **_DUMMY)
    assert rec["_marker"] == "armp"


def test_prestamp_pkl_resumes_under_default_new_keys(tmp_path):
    """A pkl WITHOUT the 2026-07-23 keys (survey/parameterization/signature/single_member)
    must resume under a request at those defaults — the back-compat pops."""
    out = str(tmp_path / "prestamp")
    pre = {k: v for k, v in _CLOSURE.items()
           if k not in ("survey", "hcd_parameterization", "hcd_prior_signature",
                        "single_member")}
    _write_stub(out, 0, pre, marker="old")
    rec = runner._run_mock(None, None, 0, out, run_cfg=dict(_CLOSURE), **_DUMMY)
    assert rec["_marker"] == "old"


def test_armp_request_clashes_with_prestamp_pkl(tmp_path):
    """ARM-P over a PRE-STAMP pkl must CLASH (missing key excused only at the default)."""
    out = str(tmp_path / "prestamp_armp")
    pre = {k: v for k, v in _CLOSURE.items()
           if k not in ("survey", "hcd_parameterization", "hcd_prior_signature",
                        "single_member")}
    _write_stub(out, 0, pre)
    with pytest.raises(RuntimeError, match="config CLASH"):
        runner._run_mock(None, None, 0, out, run_cfg=dict(_ARMP_KS), **_DUMMY)


def test_single_member_discriminator(tmp_path):
    out = str(tmp_path / "single_member")
    _write_stub(out, 0, _CLOSURE)
    with pytest.raises(RuntimeError, match="config CLASH"):
        runner._run_mock(None, None, 0, out, run_cfg=dict(_CLOSURE, single_member=True),
                         **_DUMMY)


def test_effective_run_cfg_agrees_with_run_mock_pops(tmp_path):
    """The consumer-side default-completion (effective_run_cfg) and the _run_mock directional
    pops must give the SAME pool/clash answer on representative vintage pairs."""
    cases = [
        # (existing pkl cfg, requested cfg, may_pool)
        ({k: v for k, v in _CLOSURE.items() if k != "survey"}, _CLOSURE, True),
        (_CLOSURE, _ARMP_KS, False),
        (_ARMP_KS, _ARMP_KS, True),
    ]
    for i, (ex, req, may_pool) in enumerate(cases):
        # consumer view
        eq = runner.effective_run_cfg(ex) == runner.effective_run_cfg(req)
        assert eq is may_pool, f"case {i}: effective_run_cfg says {eq}, expected {may_pool}"
        # _run_mock view
        out = str(tmp_path / f"agree_{i}")
        _write_stub(out, 0, ex, marker="pool")
        if may_pool:
            assert runner._run_mock(None, None, 0, out, run_cfg=dict(req),
                                    **_DUMMY)["_marker"] == "pool"
        else:
            with pytest.raises(RuntimeError, match="config CLASH"):
                runner._run_mock(None, None, 0, out, run_cfg=dict(req), **_DUMMY)


# ------------------------------- run_legb truth_fn guard -------------------------------------

def test_truth_fn_held_out_guard():
    from hcd_analysis.emulator.closure_legb import run_legb
    with pytest.raises(ValueError, match="truth_fn"):
        run_legb(None, None, n_mocks=1, n_warmup=1, n_samples=1, seed=0,
                 leg_a=False, truth_fn=lambda k: None)


# ------------------------------- era rules (analyze_ks_selboost) -----------------------------

def test_stamp_era_value_rule():
    assert AKS.stamp_era({"hcd_parameterization": "dndx_mapped_v2"}) == "mapped"
    assert AKS.stamp_era({"hcd_parameterization": "alpha_pivot_powerlaw_v1"}) == "legacy"
    assert AKS.stamp_era({}) == "legacy"          # pre-stamp 108-fit history
    with pytest.raises(AssertionError):
        AKS.stamp_era({"hcd_parameterization": "unknown_v9"})


def test_r6_shard_name_prefix():
    assert ksr.shard_pkl_name("K0_clean", 3, smoke=False) == "ks_selboost_clean_shard_003.pkl"
    assert ksr.shard_pkl_name("K0_clean", 3, smoke=False, r6_arm="legacy") == \
        "ks_r6_legacy_shard_003.pkl"
    assert ksr.shard_pkl_name("K0_clean", 0, smoke=True, r6_arm="mapped") == \
        "ks_r6_mapped_shard_000.smoke.pkl"
    # the campaign glob must never match an R6 name
    import fnmatch
    assert not fnmatch.fnmatch("ks_r6_legacy_shard_000.pkl", "ks_selboost_*_shard_*.pkl")


# ------------------------------- R6 paired readout -------------------------------------------

_R6_NAMES = ["ns", "Ap", "tau0_z0", "alpha_lls", "alpha_subdla", "alpha_dla"]


def _r6_rec(truth_vec, mean_shift=0.0, L=60, seed=0):
    rng = np.random.default_rng(seed)
    tv = np.asarray(truth_vec, float)
    return {
        "truth_vec": tv, "names": list(_R6_NAMES),
        "draws": rng.normal(tv + mean_shift, 0.05, size=(L, len(tv))),
        "truth_alpha_hcd_z": np.tile(tv[3:6], (4, 1)), "L": L, "n_div": 0,
        "sites_extra": {"tau0_amp": {"draws": rng.normal(1, 0.02, L), "truth": 1.0},
                        "dtau0": {"draws": rng.normal(0, 0.05, L), "truth": 0.0}},
    }


def _r6_meta(arm):
    return dict(
        forward=dict(forward_signature="f" * 64, hcd_prior_signature="p" * 64),
        prior_constants=dict(r6_override=True, r6_arm=arm,
                             r6_truth_source="mapped-selfdraw",
                             hcd_parameterization=("dndx_mapped_v2" if arm == "mapped"
                                                   else "alpha_pivot_powerlaw_v1")),
        run_kw=dict(n_warmup=250, n_samples=300, seed=1, max_tree_depth=10, dense_mass=True),
    )


def _write_r6(tmp_path, n_pairs=3, mapped_shift=0.02, break_pair=None):
    for arm in ("legacy", "mapped"):
        for m in range(n_pairs):
            tv = np.array([0.4, 0.8, 0.3, 0.18, 0.06, 0.004]) * (1 + 0.05 * m)
            if break_pair is not None and arm == "legacy" and m == break_pair:
                tv = tv * 1.001                      # truth mismatch => not a shared-truth pair
            rec = _r6_rec(tv, mean_shift=(mapped_shift if arm == "mapped" else 0.0), seed=m)
            payload = dict(arm="K0_clean", survey="ks", mode="clean", idxs=[m],
                           per_mock=[rec], meta=_r6_meta(arm))
            with open(tmp_path / f"ks_r6_{arm}_shard_{m:03d}.pkl", "wb") as f:
                pickle.dump(payload, f)


def test_r6_pair_report_runs(tmp_path):
    _write_r6(tmp_path, mapped_shift=0.02)
    out = R6.pair_report(str(tmp_path))
    assert set(out["mocks"]) == {0, 1, 2}
    # the injected mapped-arm posterior-mean shift is recovered as the paired delta
    assert abs(float(np.mean(out["delta_ns"])) - 0.02) < 0.02


def test_r6_refuses_broken_pair(tmp_path):
    _write_r6(tmp_path, break_pair=1)
    with pytest.raises(AssertionError, match="NOT a shared-truth pair"):
        R6.pair_report(str(tmp_path))


def test_r6_refuses_cross_arm_signature_drift(tmp_path):
    _write_r6(tmp_path)
    p = tmp_path / "ks_r6_mapped_shard_001.pkl"
    d = pickle.load(open(p, "rb"))
    d["meta"]["forward"]["hcd_prior_signature"] = "q" * 64
    pickle.dump(d, open(p, "wb"))
    with pytest.raises(AssertionError):
        R6.pair_report(str(tmp_path))


def test_campaign_analyzer_refuses_r6_pkl(tmp_path):
    """An R6 pkl RENAMED into the campaign namespace must be refused by the stamp check
    (defense in depth behind the ks_r6_ prefix)."""
    _write_r6(tmp_path, n_pairs=1)
    src = tmp_path / "ks_r6_mapped_shard_000.pkl"
    dst = tmp_path / "ks_selboost_clean_shard_000.pkl"
    os.rename(src, dst)
    with pytest.raises(AssertionError, match="r6_override"):
        AKS.load_campaign(str(tmp_path))
