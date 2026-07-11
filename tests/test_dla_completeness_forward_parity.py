"""DLA-completeness DESI re-run parity: the fit forward MUST equal the DEPLOYED prod_forward_config('DESI').

The DLA-completeness data-nuisance arm (scripts/run_dla_completeness_shard.py) previously built its DESI
ctx via build_arm_ctx WITHOUT the deployed forward knobs -- so it fit on a THINNER forward than the one we
unblind on (sample_res=False, f_res_amp_sigma=None, metal_prior='uniform' vs the certified
sample_res=True/0.02/flatlog2node). A diagnostic on a different forward than the deployed inference
manufactures phantom systematics; this suite pins the fix:

  (a) build_arm_ctx(..., use_prod_forward=True) wires the DEPLOYED DESI forward into build_legb_ctx.
  (b) the DLA runner actually OPTS IN (use_prod_forward=True) -- the builder/runner drift guard.
  (c) the DEFAULT path (use_prod_forward=False) is BYTE-IDENTICAL (every other arm/caller unchanged).

Config-seam SPY pattern (from tests/test_prod_forward_wiring.py): monkeypatch build_legb_ctx / build_arm_ctx
with a stub that captures kwargs and raises, so NO ensemble load (~30s x 5 members) and NO NUTS ever run.
build_arm_ctx globs the 5 prod checkpoints (present) and hits the spy before any load.

Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_dla_completeness_forward_parity.py -q
"""
import importlib.util
import os
import sys

import pytest

REPO = "/home/mfho/hcd_priya"


def _load_script(name):
    """Load a scripts/<name>.py module by path (scripts/ is not an importable package)."""
    path = os.path.join(REPO, "scripts", f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _StopBuild(Exception):
    """Raised by the spy so the ensemble load / NUTS never run once the kwargs are captured."""
    pass


# --------------------------------------------------------------------------------------------- #
#  (a) build_arm_ctx opt-in wires the DEPLOYED DESI forward into build_legb_ctx.
#      RED before the fix -> TypeError (unexpected kwarg use_prod_forward). GREEN after.
# --------------------------------------------------------------------------------------------- #
def test_dla_desi_uses_deployed_forward(monkeypatch):
    from hcd_analysis.emulator import closure_legb as CL
    from scripts import run_dnuis_bias_shard as R
    fc = CL.prod_forward_config("DESI")
    captured = {}

    def _spy(**kw):
        captured.update(kw)
        raise _StopBuild()

    monkeypatch.setattr(R, "build_legb_ctx", _spy)
    with pytest.raises(_StopBuild):
        R.build_arm_ctx("metal_misspec", "desi", True, use_prod_forward=True)
    assert captured["sample_res"] is True and captured["sample_res"] == fc["sample_res"]
    assert captured["f_res_amp_sigma"] == 0.02 and captured["f_res_amp_sigma"] == fc["f_res_amp_sigma"]
    assert captured["metal_prior"] == "flatlog2node" and captured["metal_prior"] == fc["metal_prior"]
    assert captured["metals_on"] is True and captured["sample_metals"] is True
    assert captured["ks_kwargs"] is None                      # DESI keeps the KS proxy default (byte-identical)


# --------------------------------------------------------------------------------------------- #
#  (b) the DLA RUNNER opts into the deployed forward (builder/runner drift guard).
#      RED before run_dla_completeness_shard.py:73 gains use_prod_forward=True.
# --------------------------------------------------------------------------------------------- #
def test_dla_runner_opts_into_prod_forward(monkeypatch, tmp_path):
    dla = _load_script("run_dla_completeness_shard")
    captured = {}

    def _spy(arm, survey, with_mf, *a, **kw):
        captured.update(dict(arm=arm, survey=survey, with_mf=with_mf, **kw))
        raise _StopBuild()

    monkeypatch.setattr(dla, "build_arm_ctx", _spy)
    monkeypatch.setattr(sys, "argv",
                        ["run_dla_completeness_shard.py", "--shard", "0", "--n-shards", "1",
                         "--out-dir", str(tmp_path), "--smoke"])
    with pytest.raises(_StopBuild):
        dla.main()
    assert captured["arm"] == "metal_misspec" and captured["survey"] == "desi"
    assert captured.get("use_prod_forward") is True


# --------------------------------------------------------------------------------------------- #
#  (c) DEFAULT path (use_prod_forward=False) is BYTE-IDENTICAL: the four other arms/callers unchanged.
#      Passes BEFORE and AFTER the fix.
# --------------------------------------------------------------------------------------------- #
def test_default_arm_ctx_byte_identical(monkeypatch):
    from scripts import run_dnuis_bias_shard as R
    captured = {}

    def _spy(**kw):
        captured.update(kw)
        raise _StopBuild()

    monkeypatch.setattr(R, "build_legb_ctx", _spy)
    with pytest.raises(_StopBuild):
        R.build_arm_ctx("metal_misspec", "desi", True)        # default -> use_prod_forward=False
    assert captured["sample_res"] is False                    # old thin forward: no f_res float
    assert captured["f_res_amp_sigma"] is None
    # Resolved-level byte-identity (per brief sec.1): explicit metal_prior='uniform' == the old IMPLICIT
    # build_legb_ctx default. Before the fix metal_prior is absent (-> default 'uniform'); after the fix
    # it is passed as 'uniform'. Either way the RESOLVED forward is 'uniform'. Passes before AND after.
    assert captured.get("metal_prior", "uniform") == "uniform"
    assert captured["metals_on"] is True and captured["sample_metals"] is True   # desi was always metals-on
    assert captured["ks_kwargs"] is None
