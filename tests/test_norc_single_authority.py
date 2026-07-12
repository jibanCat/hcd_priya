"""NORC single-authority refactor (2026-07-12) — the deployed sim-convergence (NORC) decision
(res_corr_on=False + fix_alpha_res=True) must be sourced from ONE place, consumed identically by all
three deployed drivers (real fit / SBC / dnuis use_prod_forward), so a future flip cannot desync them.

Single authority: closure_legb.PROD_RES_CORR_ON (module constant) + closure_legb.prod_norc_forward()
(the ONLY constructor of the {res_corr_on, fix_alpha_res} pair; encodes the fix_alpha_res == (not
res_corr_on) invariant). res_corr is a GLOBAL sim-convergence correction, deliberately NOT a per-leg
map key (a per-leg flip is physically meaningless and would desync the invariant).

Config-seam SPY pattern (from tests/test_prod_forward_wiring.py / test_dla_completeness_forward_parity.py):
monkeypatch build_legb_ctx with a stub that captures kwargs and raises, so NO ensemble load (~30s x 5
members) and NO NUTS ever run. The single lever is monkeypatch CL.PROD_RES_CORR_ON, which
prod_norc_forward() reads at call time -> ALL three consumers move together.

Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_norc_single_authority.py -q
"""
import importlib.util
import os
import sys

import pytest

REPO = "/home/mfho/hcd_priya"


def _load_script(name):
    """Load a scripts/<name>.py module by path (fresh module; its build_legb_ctx is monkeypatchable)."""
    path = os.path.join(REPO, "scripts", f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _StopBuild(Exception):
    """Raised by the spy so the ensemble load / NUTS never run once the kwargs are captured."""
    pass


# =============================================================================================== #
#  (1) The helper values + INV-1 invariant (fix_alpha_res == not res_corr_on).  RED (helper absent).
# =============================================================================================== #
def test_prod_norc_forward_values():
    from hcd_analysis.emulator import closure_legb as CL
    norc = CL.prod_norc_forward()
    assert norc == {"res_corr_on": False, "fix_alpha_res": True}
    # INV-1: the pair is coupled, encoded in exactly one place.
    assert norc["fix_alpha_res"] == (not norc["res_corr_on"])
    assert CL.PROD_RES_CORR_ON is False
    # fresh dict each call (no shared mutable module state)
    a = CL.prod_norc_forward(); a["res_corr_on"] = "MUTATED"
    assert CL.prod_norc_forward()["res_corr_on"] is False


# =============================================================================================== #
#  (2) GLOBAL, not per-leg: the NORC keys must NEVER leak into PROD_FORWARD_BY_LEG / prod_forward_config
#      (guards the caveat-19 one-leg-flip foot-gun). Pins CURRENT behaviour (map untouched) -> GREEN.
# =============================================================================================== #
def test_prod_norc_is_global_not_per_leg():
    from hcd_analysis.emulator import closure_legb as CL
    for leg in ("DESI", "eBOSS", "KS"):
        fc = CL.prod_forward_config(leg)
        assert "res_corr_on" not in fc, f"res_corr_on leaked into prod_forward_config({leg!r})"
        assert "fix_alpha_res" not in fc, f"fix_alpha_res leaked into prod_forward_config({leg!r})"


# =============================================================================================== #
#  build-kwargs spies for the three deployed consumers (all capture res_corr_on before any build).
# =============================================================================================== #
def _capture_real_fit_kwargs(survey, monkeypatch):
    rf = _load_script("run_real_fit")
    captured = {}

    def _spy(**kw):
        captured.update(kw)
        raise _StopBuild()

    monkeypatch.setattr(rf, "build_legb_ctx", _spy)
    with pytest.raises(_StopBuild):
        rf.build_real_ctx(survey)
    return captured


def _capture_dnuis_prod_kwargs(survey, monkeypatch):
    from scripts import run_dnuis_bias_shard as R
    captured = {}

    def _spy(**kw):
        captured.update(kw)
        raise _StopBuild()

    monkeypatch.setattr(R, "build_legb_ctx", _spy)
    with pytest.raises(_StopBuild):
        R.build_arm_ctx("metal_misspec", survey, True, use_prod_forward=True)
    return captured


def _capture_sbc_kwargs(monkeypatch, tmp_path, extra_argv=()):
    sbc = _load_script("run_prod_sbc_shard")
    captured = {}

    def _spy(**kw):
        captured.update(kw)
        raise _StopBuild()

    monkeypatch.setattr(sbc, "build_legb_ctx", _spy)
    argv = ["run_prod_sbc_shard.py", "--shard", "0", "--n-shards", "1",
            "--out-dir", str(tmp_path)] + list(extra_argv)
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(_StopBuild):
        sbc.main()
    return captured


# =============================================================================================== #
#  (3) ONE-LEG-FLIP GUARD (load-bearing): patch the single authority CL.PROD_RES_CORR_ON and assert
#      EVERY deployed consumer's built res_corr_on moves together. Encodes the panel near-miss.
#      RED today (each holds a private NORC literal) -> GREEN after (all read prod_norc_forward()).
# =============================================================================================== #
def test_one_leg_flip_guard(monkeypatch, tmp_path):
    from hcd_analysis.emulator import closure_legb as CL
    # flip the SINGLE authority; prod_norc_forward() reads it at call time.
    monkeypatch.setattr(CL, "PROD_RES_CORR_ON", True, raising=False)
    patched = True                                            # res_corr_on the authority now dictates

    rf_desi = _capture_real_fit_kwargs("desi", monkeypatch)
    rf_ks = _capture_real_fit_kwargs("ks", monkeypatch)
    dn_desi = _capture_dnuis_prod_kwargs("desi", monkeypatch)
    sbc_default = _capture_sbc_kwargs(monkeypatch, tmp_path)   # no --res-corr-on -> tracks the authority

    assert rf_desi["res_corr_on"] is patched, "run_real_fit did not track the single authority"
    assert rf_ks["res_corr_on"] is patched, "run_real_fit (KS) did not track the single authority"
    assert dn_desi["res_corr_on"] is patched, "dnuis use_prod_forward did not track the single authority"
    assert bool(sbc_default["res_corr_on"]) is patched, "SBC default did not track the single authority"


# =============================================================================================== #
#  (4) CROSS-DRIVER fix_alpha_res seam: the resolved ctx.fix_alpha_res == prod_norc_forward()[...]
#      for build_real_ctx AND dnuis use_prod_forward. Fake-build seam (no ensemble / NUTS).
# =============================================================================================== #
class _FakeLeg:
    def __init__(self, name, metals_on=True, resolution_ready=True):
        self.name = name
        self.metals_on = metals_on
        self.resolution_ready = resolution_ready


class _FakeCtx:
    def __init__(self, **f):
        self.__dict__.update(f)

    def _replace(self, **kw):
        d = dict(self.__dict__)
        d.update(kw)
        return _FakeCtx(**d)


def test_cross_driver_norc_fix_alpha_res(monkeypatch):
    from hcd_analysis.emulator import closure_legb as CL
    want = CL.prod_norc_forward()["fix_alpha_res"]            # True under the deployed authority

    # --- dnuis use_prod_forward (minimal fake; mirrors test_dla_completeness_forward_parity) --------
    from scripts import run_dnuis_bias_shard as R

    def _fake_build_dnuis(**kw):
        return (_FakeCtx(res_corr_on=bool(kw.get("res_corr_on", True)), fix_alpha_res=False,
                         legs=[_FakeLeg("DESI"), _FakeLeg("KS")]), object())

    monkeypatch.setattr(R, "build_legb_ctx", _fake_build_dnuis)
    ctx_dn, _d, _spec = R.build_arm_ctx("metal_misspec", "desi", True, use_prod_forward=True)
    assert ctx_dn.res_corr_on is False and ctx_dn.fix_alpha_res is want

    # --- build_real_ctx (richer fake: satisfies its post-build data-nuisance parity asserts) --------
    rf = _load_script("run_real_fit")

    def _fake_build_rf(**kw):
        return (_FakeCtx(
            res_corr_on=bool(kw.get("res_corr_on", True)), fix_alpha_res=False,
            legs=[_FakeLeg("DESI", metals_on=True, resolution_ready=True)],
            sample_res=kw["sample_res"], f_res_amp_sigma=kw["f_res_amp_sigma"],
            metal_prior=kw["metal_prior"], metal_node_z=(2.2, 4.2),
            sample_metals=kw["sample_metals"]), object())

    monkeypatch.setattr(rf, "build_legb_ctx", _fake_build_rf)
    ctx_rf, _d2, _members = rf.build_real_ctx("desi")
    assert ctx_rf.res_corr_on is False and ctx_rf.fix_alpha_res is want


# =============================================================================================== #
#  (5) SBC --res-corr-on RESTORE arm still WINS (precedence: CLI > authority). It builds res_corr_on
#      True; run_cfg stamps bool(a.res_corr_on)==True (a.res_corr_on is exactly the captured build value).
# =============================================================================================== #
def test_sbc_cli_restore_still_wins(monkeypatch, tmp_path):
    captured = _capture_sbc_kwargs(monkeypatch, tmp_path, extra_argv=["--res-corr-on"])
    assert captured["res_corr_on"] is True, "--res-corr-on restore arm must win over the NORC default"


# =============================================================================================== #
#  (6) SBC DEFAULT (no flag) tracks the single authority.  RED today (helper absent -> error).
# =============================================================================================== #
def test_sbc_default_tracks_authority(monkeypatch, tmp_path):
    from hcd_analysis.emulator import closure_legb as CL
    captured = _capture_sbc_kwargs(monkeypatch, tmp_path)
    assert bool(captured["res_corr_on"]) == CL.prod_norc_forward()["res_corr_on"]


# =============================================================================================== #
#  (7) run_cfg key-set unchanged (resume-compat, caveat-9): a NORC pkl (res_corr_on=False) resumes on a
#      default NORC request with NO clash. The refactor adds NO new run_cfg key. Pins CURRENT behaviour.
# =============================================================================================== #
def _old_stamp_cfg():
    # the full current DESI-NORC run_cfg key-set (the refactor must not extend this).
    return dict(leg_a=True, cemu_variant="current", amp_sigma=0.0, leg="DESI", fold=0,
                tau0_prior_sigma=0.0, subdla_truth_boost=1.0, res_corr_on=False,
                sample_res=True, f_res_amp_sigma=0.02, metal_prior="flatlog2node", ks_kmax=None)


def test_sbc_run_cfg_no_new_key_on_resume(tmp_path):
    sbc = _load_script("run_prod_sbc_shard")
    path = sbc._mock_path(str(tmp_path), 0)
    rec = {"n_div": 0, "mock": 0, "run_cfg": _old_stamp_cfg()}
    import pickle
    with open(path, "wb") as f:
        pickle.dump(rec, f)
    # request the identical NORC cfg -> loads via the skip branch, no clash (fix_alpha_res never stamped).
    out = sbc._run_mock(None, None, 0, str(tmp_path), n_mocks=1, n_warmup=1, n_samples=1,
                        max_tree_depth=1, seed=0, run_cfg=_old_stamp_cfg())
    assert out["n_div"] == 0
    assert "fix_alpha_res" not in _old_stamp_cfg(), "fix_alpha_res must NOT enter run_cfg (100% collinear)"


# =============================================================================================== #
#  (8) OOS KS-literal drift guard (scope-F): the hard-coded {resolution_float, k_max} on the
#      use_prod_forward=False KS resolution-float path == prod_forward_config('KS')['ks_kwargs'].
#      Parity TEST only (the literal is NOT rewired -- it stays a non-deployed OOS path).
# =============================================================================================== #
def test_dnuis_oos_ks_literal_matches_map(monkeypatch):
    from hcd_analysis.emulator import closure_legb as CL
    from scripts import run_dnuis_bias_shard as R
    captured = {}

    def _spy(**kw):
        captured.update(kw)
        raise _StopBuild()

    monkeypatch.setattr(R, "build_legb_ctx", _spy)
    with pytest.raises(_StopBuild):                          # KS resolution-float, NON-deployed (default forward)
        R.build_arm_ctx("resolution", "ks", True, float_res=True)
    assert captured["ks_kwargs"] == CL.prod_forward_config("KS")["ks_kwargs"], \
        "the OOS KS ks_kwargs literal drifted from prod_forward_config('KS')"


# =============================================================================================== #
#  (9) forward_signature(): deterministic + sensitive to the NORC authority (freeze-task artifact,
#      consumed by NOTHING in this task). RED today (helper absent).
# =============================================================================================== #
def test_forward_signature_determinism_and_sensitivity(monkeypatch):
    from hcd_analysis.emulator import closure_legb as CL
    s0 = CL.forward_signature()
    assert isinstance(s0, str) and len(s0) >= 32
    assert CL.forward_signature() == s0, "forward_signature must be deterministic"
    monkeypatch.setattr(CL, "PROD_RES_CORR_ON", True, raising=False)
    assert CL.forward_signature() != s0, "forward_signature must change when the NORC authority flips"
