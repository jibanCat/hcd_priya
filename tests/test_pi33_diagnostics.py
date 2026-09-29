"""PI #33 (2026-09-27): the eBOSS diagnostic refit machinery (H-1 HCD prior scale, MF-1 tau0 box, R-1 resolution off,
MF-3 third mean-flux mode). Generic machinery only; no science values.
Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_pi33_diagnostics.py -q -p no:cacheprovider
"""
import argparse
import importlib.util
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.handlers as H
import pytest

import hcd_analysis.emulator  # noqa: F401
from hcd_analysis.emulator import closure_legb as CL


def _load_driver():
    sp = importlib.util.spec_from_file_location("run_real_fit", "/home/mfho/hcd_priya/scripts/run_real_fit.py")
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


def test_apply_tau0_curv_identity_and_formula():
    z = jnp.array([2.2, 3.0, 4.6]); a = jnp.array([1.1, 1.2, 1.3])
    assert np.allclose(CL._apply_tau0_curv(a, z, None, z_pivot=3.0), a)
    x2 = np.log((1 + np.array([2.2, 3.0, 4.6])) / 4.0) ** 2
    out = CL._apply_tau0_curv(a, z, 0.7, z_pivot=3.0)
    assert np.allclose(out, np.array(a) * np.exp(0.7 * x2)) and abs(float(out[1]) - 1.2) < 1e-12   # pivot untouched
    A = jnp.stack([a, a]); c = jnp.array([0.7, -0.7])
    out2 = CL._apply_tau0_curv(A, z, c, z_pivot=3.0)
    assert out2.shape == (2, 3) and np.allclose(out2[1], np.array(a) * np.exp(-0.7 * x2))


def test_sample_tau0_curv_site_only_when_on():
    with H.seed(rng_seed=0), H.trace() as tr:
        assert CL._sample_tau0_curv(SimpleNamespace(tau0_curv_sigma=None)) is None
    assert "ctau0" not in tr
    with H.seed(rng_seed=0), H.trace() as tr2:
        v = CL._sample_tau0_curv(SimpleNamespace(tau0_curv_sigma=2.0))
    assert "ctau0" in tr2 and np.isfinite(float(v)) and float(tr2["ctau0"]["fn"].scale) == 2.0
    assert CL._sample_tau0_curv(SimpleNamespace()) is None       # old ctx objects without the field: OFF


def _dummy_ctx(**over):
    req = {f: None for f in CL.LegBCtx._fields if f not in CL.LegBCtx._field_defaults}
    req.update(alpha_hcd_sigma=jnp.array([0.05, 0.025, 0.002]), alpha_hcd_mu=jnp.array([0.17, 0.062, 0.0044]))
    return CL.LegBCtx(**req)._replace(sample_res=True, **over)


def test_apply_diagnostic_overrides_default_is_identity_and_each_knob_applies():
    M = _load_driver(); ctx = _dummy_ctx()
    c0, ap0 = M.apply_diagnostic_overrides(ctx, None); assert c0 is ctx and ap0 == {}
    c0, ap0 = M.apply_diagnostic_overrides(ctx, {}); assert c0 is ctx and ap0 == {}
    c0, ap0 = M.apply_diagnostic_overrides(ctx, dict(hcd_prior_scale=None, resolution_off=None)); assert c0 is ctx and ap0 == {}
    c1, ap1 = M.apply_diagnostic_overrides(ctx, dict(hcd_prior_scale=4.0))
    assert np.allclose(np.asarray(c1.alpha_hcd_sigma), [0.2, 0.1, 0.002]) and ap1["hcd_prior_scale"] == 4.0 and c1.tau0_amp_range == ctx.tau0_amp_range
    c2, ap2 = M.apply_diagnostic_overrides(ctx, dict(tau0_amp_max=1.5))
    assert c2.tau0_amp_range == (0.75, 1.5) and ap2["tau0_amp_range_before"] == [0.75, 1.25] and np.allclose(np.asarray(c2.alpha_hcd_sigma), np.asarray(ctx.alpha_hcd_sigma))
    with pytest.raises(SystemExit):
        M.apply_diagnostic_overrides(ctx, dict(tau0_amp_max=0.5))
    with pytest.raises(AssertionError):                                          # R1 fix: the leg must have been built without f_res
        M.apply_diagnostic_overrides(ctx, dict(resolution_off=True))
    ctx_off = ctx._replace(sample_res=False, legs=[SimpleNamespace(resolution_on=False)])
    c3, ap3 = M.apply_diagnostic_overrides(ctx_off, dict(resolution_off=True)); assert c3.sample_res is False and ap3 == dict(resolution_off=True, resolution_term_in_covariance=True)
    c4, ap4 = M.apply_diagnostic_overrides(ctx, dict(mf_curvature_sigma=2.0)); assert c4.tau0_curv_sigma == 2.0 and ctx.tau0_curv_sigma is None
    c5, ap5 = M.apply_diagnostic_overrides(ctx_off, dict(hcd_prior_scale=4.0, tau0_amp_max=1.5, resolution_off=True, mf_curvature_sigma=2.0))
    assert set(ap5) == {"hcd_prior_scale", "alpha_hcd_sigma", "alpha_hcd_sigma_before", "alpha_hcd_mu", "tau0_amp_range", "tau0_amp_range_before", "resolution_off", "resolution_term_in_covariance", "mf_curvature_sigma"}
    assert np.allclose(ap5["alpha_hcd_sigma_before"], [0.05, 0.025, 0.002]) and np.allclose(ap5["alpha_hcd_mu"], [0.17, 0.062, 0.0044])


def test_nuisance_export_keys_include_ctau0_when_present():
    M = _load_driver()
    assert M._nuisance_export_keys(dict(f_res_amp=1, f_res_slope=1, theta_unit=1)) == ["f_res_amp", "f_res_slope"]
    assert M._nuisance_export_keys(dict(f_res_amp=1, f_res_slope=1, ctau0=1)) == ["f_res_amp", "f_res_slope", "ctau0"]
    assert M._nuisance_export_keys(dict(ctau0=1)) == ["ctau0"]


def _args(**kw):
    d = dict(hcd_prior_scale=None, tau0_amp_max=None, resolution_off=False, mf_curvature_sigma=None, diag_tag=None, out_dir=None); d.update(kw)
    return argparse.Namespace(**d)


def test_diag_from_args_validation():
    M = _load_driver()
    assert M.diag_from_args(_args()) == {}
    with pytest.raises(SystemExit):
        M.diag_from_args(_args(hcd_prior_scale=4.0, out_dir="/tmp/x"))                       # override without tag
    with pytest.raises(SystemExit):
        M.diag_from_args(_args(diag_tag="H1", out_dir="/tmp/x"))                             # tag without override
    with pytest.raises(SystemExit):
        M.diag_from_args(_args(hcd_prior_scale=4.0, diag_tag="H1"))                          # no explicit out-dir
    with pytest.raises(SystemExit):
        M.diag_from_args(_args(hcd_prior_scale=4.0, diag_tag="H-1", out_dir="/tmp/x"))       # non-alphanumeric tag
    assert M.diag_from_args(_args(resolution_off=True, diag_tag="R1", out_dir="/tmp/x")) == dict(resolution_off=True)
    assert M.diag_from_args(_args(mf_curvature_sigma=2.0, tau0_amp_max=1.5, diag_tag="X", out_dir="/tmp/x")) == dict(tau0_amp_max=1.5, mf_curvature_sigma=2.0)


def test_priors_only_twin_and_model_share_the_optional_site_name():
    src = open(CL.__file__).read()
    i = src.index("def _legb_priors_only(ctx):"); j = src.index("def _legb_reconstruct_deterministics", i)
    assert "_sample_tau0_curv(ctx)" in src[i:j]                       # priors-only twin samples the same optional site
    k = src.index("def _legb_model("); m = src.index("def _legb_priors_only", k)
    assert src[k:m].count("_sample_tau0_curv(ctx)") == 1 and '"ctau0" in samples' in src[j:]   # model + reconstruction


def test_trace_site_order_and_batched_reconstruction_equals_vmap():
    """JAX review S4: the optional site follows tau0_amp, dtau0 in the trace; the (L, nZ) batched curvature equals the vmapped scalar path."""
    import jax
    ctx = SimpleNamespace(tau0_curv_sigma=2.0, tau0_amp_range=(0.75, 1.25), dtau0_range=(-0.4, 0.25), tau0_amp_gauss=None, dtau0_gauss=None)
    with H.seed(rng_seed=1), H.trace() as tr:
        CL._sample_tau0_sites(ctx); CL._sample_tau0_curv(ctx)
    assert [k for k in tr if tr[k]["type"] == "sample"] == ["tau0_amp", "dtau0", "ctau0"]
    z = jnp.array([2.2, 3.0, 4.6]); A = jnp.array([[1.1, 1.2, 1.3], [0.9, 1.0, 1.1]]); c = jnp.array([0.7, -0.3])
    batched = CL._apply_tau0_curv(A, z, c, z_pivot=3.0)
    scalar = jax.vmap(lambda a, cc: CL._apply_tau0_curv(a, z, cc, z_pivot=3.0))(A, c)
    assert np.array_equal(np.asarray(batched), np.asarray(scalar))
    with pytest.raises(AssertionError):
        CL._apply_tau0_curv(jnp.array([1.0, 1.0, 1.0]), z, jnp.array([0.1, 0.2, 0.3]), z_pivot=3.0)   # (L,) against a 1-D ladder is refused


def test_build_real_ctx_signature_has_sample_res_override():
    """R1 fix: the option-a control is decided at build time (leg loaded with resolution_float=False), not after."""
    import inspect
    M = _load_driver(); sig = inspect.signature(M.build_real_ctx)
    assert "sample_res_override" in sig.parameters and sig.parameters["sample_res_override"].default is None
    src = inspect.getsource(M.run_real_fit); assert "sample_res_override=(False if (diag or {}).get(\"resolution_off\") else None)" in src
