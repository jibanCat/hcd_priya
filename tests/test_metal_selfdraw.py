"""OPTION A (PI decisions #9, 2026-07-27): the corrected prior-predictive SBC metal sector.

THE DEFECT this fixes. The Leg-A self-draw mock generator never forwarded the metal f/k nodes
into the mock forward, so on a `flatlog2node` leg the mock truth for those four fitted sites was
identically ZERO while the fit floated them under LogUniform(0.003, 0.03) / (1e-3, 0.1) --
support that contains no mass at zero. The SBC null was therefore violated by construction in
the metal sector (ARM-P eBOSS N=96: metal-node `truth=nan`, tau0 ranks 95/96 outside the band).
It was not an oversight: closure_legb's metal_zevo comment records it as a deliberate
de-double-count for the INJECTION arms, and it was never revisited when the same generator was
reused for a prior-predictive certification.

THE FIX must satisfy the PI's four binding conditions:
  1. metal truths drawn from the SAME deployed priors used in fitting,
  2. propagated through the mock forward model,
  3. fit with the unchanged deployed configuration,
  4. frozen forward / priors / lock / signatures / gate definitions preserved.

Condition 1 is free: `_legb_priors_only` (which `draw_leg_a_leg_truth` traces) ALREADY samples
`_metal_2node_sites`, so the drawn values sit in `raw` under the exact deployed prior. They were
simply discarded. These tests pin the extraction, the default-OFF byte-identity, and the
pooling separation.

Run: /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_metal_selfdraw.py -q
"""
import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, "/home/mfho/hcd_priya")

CL = importlib.import_module("hcd_analysis.emulator.closure_legb")
runner = importlib.import_module("scripts.run_prod_sbc_shard")


def _ctx(metal_prior="flatlog2node", sample_metals=True, selfdraw=True,
         legs=("DESI", "eBOSS"), siII=("DESI",)):
    """A STUB ctx carrying only what the node extractor reads. Deliberately not a real
    LegBCtx: the extractor must depend on nothing else, so that it cannot drift with the
    forward model."""
    return SimpleNamespace(
        metal_prior=metal_prior, sample_metals=sample_metals,
        selfdraw_metal_truth=selfdraw,
        legs=[SimpleNamespace(name=n, metals_on=True) for n in legs],
        metal_siII_legs=tuple(siII),
        metal_fnode_lo=0.003, metal_fnode_hi=0.03,
        metal_knode_lo=1e-3, metal_knode_hi=0.1,
        metal_node_z=(2.2, 4.2))


def _raw(legs=("DESI", "eBOSS"), siII=("DESI",)):
    """A raw prior trace as `_metal_2node_sites` would leave it, with distinct values so any
    mis-ordering or mis-keying is visible."""
    raw, v = {}, 0.004
    for n in legs:
        for i in (0, 1):
            raw[f"f_SiIII_{n}_z{i}"] = v; v += 0.001
        if n in siII:
            for i in (0, 1):
                raw[f"f_SiII_{n}_z{i}"] = v; v += 0.001
        for i in (0, 1):
            raw[f"k_SiIII_{n}_z{i}"] = 0.01 + 0.001 * i
        if n in siII:
            for i in (0, 1):
                raw[f"k_SiII_{n}_z{i}"] = 0.02 + 0.001 * i
    return raw


# ------------------------------- the node extractor -------------------------------

def test_extractor_returns_none_when_flag_off():
    """DEFAULT OFF is the byte-identity guarantee: no nodes -> make_leg_a_legmock takes the
    historical path and the 96 landed pkls stay reproducible."""
    assert CL._selfdraw_metal_nodes(_raw(), _ctx(selfdraw=False)) is None


def test_extractor_returns_none_on_the_legacy_scalar_prior():
    """Only MODEL C+ (`flatlog2node`) has node sites; the scalar a_SiIII path must be untouched."""
    assert CL._selfdraw_metal_nodes(_raw(), _ctx(metal_prior="uniform")) is None


def test_extractor_returns_none_when_metals_not_sampled():
    assert CL._selfdraw_metal_nodes(_raw(), _ctx(sample_metals=False)) is None


def test_extractor_mirrors_the_sampled_site_values_exactly():
    """CONDITION 1: the truths must BE the deployed-prior draws, not a re-draw or a centre."""
    raw, ctx = _raw(), _ctx()
    nodes = CL._selfdraw_metal_nodes(raw, ctx)
    assert set(nodes) == {"DESI", "eBOSS"}
    f3, f2, k3, k2 = nodes["eBOSS"]
    assert np.allclose(f3, [raw["f_SiIII_eBOSS_z0"], raw["f_SiIII_eBOSS_z1"]])
    assert np.allclose(k3, [raw["k_SiIII_eBOSS_z0"], raw["k_SiIII_eBOSS_z1"]])
    assert f2 is None and k2 is None, "eBOSS is not in metal_siII_legs -> no SiII doublet"
    f3d, f2d, k3d, k2d = nodes["DESI"]
    assert np.allclose(f2d, [raw["f_SiII_DESI_z0"], raw["f_SiII_DESI_z1"]])
    assert np.allclose(k2d, [raw["k_SiII_DESI_z0"], raw["k_SiII_DESI_z1"]])


def test_extractor_skips_metals_off_legs():
    """KS floats no metal nodes; it must not acquire any (and its tau0 pull is a SEPARATE open
    question -- PI #9 decision 6)."""
    ctx = _ctx(legs=("DESI", "eBOSS"))
    ctx.legs.append(SimpleNamespace(name="KS", metals_on=False))
    assert "KS" not in CL._selfdraw_metal_nodes(_raw(), ctx)


def test_extractor_shape_matches_the_fit_contract():
    """The tuple shape must be exactly what _data_loglik_legcore forwards to
    predict_P_obs_on_leg, so mock and fit apply metals through the IDENTICAL code path."""
    for leg, (f3, f2, k3, k2) in CL._selfdraw_metal_nodes(_raw(), _ctx()).items():
        assert np.asarray(f3).shape == (2,), leg
        assert np.asarray(k3).shape == (2,), leg
        for opt in (f2, k2):
            assert opt is None or np.asarray(opt).shape == (2,), leg


def test_extractor_raises_on_a_missing_site_rather_than_defaulting():
    """FAIL-LOUD: a silently-missing node would reintroduce a zero truth, i.e. the very defect
    this fixes. Never default."""
    raw = _raw(); del raw["f_SiIII_eBOSS_z1"]
    with pytest.raises(KeyError):
        CL._selfdraw_metal_nodes(raw, _ctx())


# ------------------------------- pooling / provenance -------------------------------

# ------------------------- f_res self-draw (PI #9 Q1 answer, scope expansion (b)) -------------

def _fres_ctx(sample_res=True, selfdraw=True, amp_sigma=0.05):
    return SimpleNamespace(sample_res=sample_res, fres_selfdraw_truth=selfdraw,
                           f_res_amp_sigma=amp_sigma,
                           z_global=np.array([2.2 + 0.2 * i for i in range(13)]))


def _fres_raw(amp=0.013, slope=0.42):
    return {"f_res_amp": amp, "f_res_slope": slope}


def test_fres_extractor_returns_none_when_flag_off():
    """DEFAULT OFF preserves the historical mock (no resolution distortion at all)."""
    assert CL._selfdraw_fres_bres(_fres_raw(), _fres_ctx(selfdraw=False)) is None


def test_fres_extractor_returns_none_when_res_not_sampled():
    """If the fit does not float f_res there is no truth to draw and nothing to propagate."""
    assert CL._selfdraw_fres_bres(_fres_raw(), _fres_ctx(sample_res=False)) is None


def test_fres_extractor_matches_the_deployed_bres_curve_exactly():
    """CONDITION: the mock's b_res(z) must be the SAME function of the SAME drawn sites that the
    fit builds in _legb_model, else the correction introduces a new misspecification."""
    raw, ctx = _fres_raw(), _fres_ctx()
    got = CL._selfdraw_fres_bres(raw, ctx)
    want = CL._bres_of_z(ctx.z_global, raw["f_res_amp"], raw["f_res_slope"])
    assert np.allclose(got, want, rtol=0, atol=0), "b_res(z) must be bit-identical to the fit's"
    assert np.asarray(got).shape == (13,), "b_res is defined on z_global, sliced per leg later"


def test_fres_extractor_raises_on_a_missing_site_rather_than_defaulting():
    """FAIL-LOUD: defaulting to 0 would silently reinstate the pinned-truth defect."""
    with pytest.raises(KeyError):
        CL._selfdraw_fres_bres({"f_res_amp": 0.01}, _fres_ctx())


def test_fres_selfdraw_is_in_run_cfg_defaults():
    assert runner.RUN_CFG_DEFAULTS.get("fres_selfdraw") is False


def test_revised_arm_cannot_pool_with_A1_Ad_OR_the_metals_only_A1c_build():
    """PI #9 Q1: the revised arm must be a distinct population from ALL THREE predecessors --
    including the metals-only A1c build whose smoke pkl is already on disk."""
    base = dict(runner.RUN_CFG_DEFAULTS, survey="eBOSS", hcd_prior_signature="s" * 64)
    a1 = dict(base)                                                    # invalidated, preserved
    ad = dict(base, diag_no_sample_metals=True)                        # metals-off diagnostic
    a1c_metals_only = dict(base, metal_selfdraw=True)                  # superseded build
    a1c_revised = dict(base, metal_selfdraw=True, fres_selfdraw=True)  # the arm of record
    eff = runner.effective_run_cfg
    for name, other in (("A1", a1), ("Ad", ad), ("A1c-metals-only", a1c_metals_only)):
        assert eff(a1c_revised) != eff(other), f"revised arm must not pool with {name}"
        assert eff(other) != eff(a1c_revised), f"{name} must not pool with the revised arm"


def test_metal_selfdraw_is_in_run_cfg_defaults():
    """Same lesson as the 2026-07-27 diag_no_sample_metals regression: a stamped key absent from
    the defaults makes pre-key pkls refuse to pool with post-key ones."""
    assert runner.RUN_CFG_DEFAULTS.get("metal_selfdraw") is False


def test_corrected_arm_cannot_pool_with_the_defective_arm():
    """PI #9 decision 2: the original N=96 arm is INVALIDATED as certification but PRESERVED.
    The corrected arm must be a separate population in both directions, so neither can silently
    absorb the other."""
    base = dict(runner.RUN_CFG_DEFAULTS, survey="eBOSS", hcd_prior_signature="s" * 64)
    defective = dict(base)                                   # A1, as it exists on disk
    corrected = dict(base, metal_selfdraw=True)              # A1c
    metals_off = dict(base, diag_no_sample_metals=True)      # Ad, the attribution diagnostic
    eff = runner.effective_run_cfg
    assert eff(defective) != eff(corrected)
    assert eff(corrected) != eff(metals_off)
    assert eff(defective) != eff(metals_off)
