"""The public predictor (hcd_analysis/predictor.py) on synthetic h5/2 bundles (hcd_analysis.emulator.release_testing):
untrained networks and made-up normalizers, so the values mean nothing; the tests check the contract (loading, shapes,
the ensemble mean and spread, the w_c path through the bundled map, k placement, the mean-flux inputs, the amplitude
rules, the refusals) and that the module never imports the historical likelihood."""
import inspect
import json
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import hcd_analysis.predictor as hp
from hcd_analysis.emulator import dndx_wc as D
from hcd_analysis.emulator import kcoord as K
from hcd_analysis.emulator import release_testing as RT
from hcd_analysis.emulator import safe_h5 as H
from hcd_analysis.emulator.data import Z_LIMITS, normalize_params
from hcd_analysis.emulator.mean_flux_model import kim_tau0
from hcd_analysis.emulator.predict import predict_P_filt, reconstruct_P_filt_jax

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMS = {"ns": 0.95, "Ap": 2.0e-9, "herei": 3.8, "heref": 2.9, "alphaq": 1.8, "hub": 0.7, "omegamh2": 0.143,
          "hireionz": 7.0, "bhfeedback": 0.05}
FORBIDDEN = ("hcd_analysis.emulator.inference", "hcd_analysis.emulator.closure_legb", "hcd_analysis.emulator.lit_dndx",
             "hcd_analysis.emulator.lit_dndx_kernel", "hcd_analysis.emulator.meanflux_prior")


@pytest.fixture(scope="module")
def info(tmp_path_factory):
    return RT.write_synthetic_bundle(str(tmp_path_factory.mktemp("b") / "bundle"), n_members=3)


@pytest.fixture(scope="module")
def emu(info):
    return hp.load_bundle(info.bundle_dir, info.manifest_sha256)


def _theta_x(z):
    th = np.asarray(normalize_params(np.array([PARAMS[n] for n in hp.PARAM_NAMES])), dtype=np.float64)
    return th, (z - Z_LIMITS[0]) / (Z_LIMITS[1] - Z_LIMITS[0])


# ------------------------------------------------------------------------------------------------------- loading
def test_load_bundle_refuses_none_another_digest_an_ineligible_and_a_historical_bundle(info, tmp_path):
    with pytest.raises(H.SafeFormatError, match="None is refused"):
        hp.load_bundle(info.bundle_dir, None)
    with pytest.raises(H.SafeFormatError, match="sha256 does not match"):
        hp.load_bundle(info.bundle_dir, "f" * 64)
    bad = RT.write_synthetic_bundle(str(tmp_path / "dirty"), export_code={"commit": RT.SYNTHETIC_COMMIT, "dirty": True})
    with pytest.raises(H.SafeFormatError, match="does not establish a clean release"):
        hp.load_bundle(bad.bundle_dir, bad.manifest_sha256)
    from hcd_analysis.emulator.model import Emulator
    v1 = tmp_path / "v1"
    v1.mkdir()
    rec, ident = RT.synthetic_cache_record()
    H.export_member(str(v1 / "m0"), Emulator(**RT.SYNTHETIC_ARCH, key=jax.random.PRNGKey(0)),
                    RT.synthetic_meta(0, rec, ident), RT.synthetic_norm(), fmt=H.FORMAT_HISTORICAL)
    H.write_manifest_historical(str(v1), [str(v1 / "m0")], {})
    with pytest.raises(H.SafeFormatError, match="historical"):
        hp.load_bundle(str(v1), H.sha256_file(str(v1 / "manifest.json")))


def test_a_bundle_on_another_parameter_box_is_refused(tmp_path):
    def other_box(i, model, meta, nm):
        return model, dict(meta, param_limits=[[lo, hi * 1.01] for lo, hi in meta["param_limits"]]), nm
    b = RT.write_synthetic_bundle(str(tmp_path / "b"), member_fn=other_box)
    with pytest.raises(H.SafeFormatError, match="PARAM_LIMITS"):
        hp.load_bundle(b.bundle_dir, b.manifest_sha256)


def test_provenance_names_the_manifest_the_lineage_and_the_map(emu, info):
    pv = emu.predict(PARAMS, 3.0, tau_eff=0.4).provenance
    assert pv["manifest_sha256"] == info.manifest_sha256 == emu.manifest_sha256 and pv["format"] == H.FORMAT
    assert pv["lineage"] == info.lineage and pv["n_members"] == 3
    assert pv["wc_map_sha256"] == info.lineage["wc_map"]["sha256"]
    man = json.load(open(os.path.join(info.bundle_dir, "manifest.json")))
    assert pv["bundle_provenance"] == man["provenance"]
    pv["bundle_provenance"]["source"] = "edited"                                  # a copy, not the bundle's own
    assert emu.provenance["bundle_provenance"] == man["provenance"]


# -------------------------------------------------------------------------------------------------- the forward
def test_the_prediction_is_the_ensemble_mean_with_its_spread_template_and_w_c(emu, info, monkeypatch):
    z, tau = 3.0, 0.41
    pred = emu.predict(PARAMS, z, tau_eff=tau)
    th, zu = _theta_x(z)
    ens, metas, norm = H.load_ensemble(info.bundle_dir, expected_manifest_sha256=info.manifest_sha256)
    ref = np.asarray(predict_P_filt(ens, jnp.asarray(th), zu, tau, norm["P_filt"]))
    K_ = RT.SYNTHETIC_ARCH["n_k"]
    assert pred.P_class.shape == (4, K_) and pred.k.shape == (K_,) and pred.template.shape == (3, K_)
    assert pred.w_c.shape == (3,) and pred.dndx.shape == (3,) and pred.member_std.shape == (4, K_)
    assert np.allclose(pred.P_class, ref, rtol=1e-13, atol=0)
    x = jnp.concatenate([jnp.asarray(th), jnp.asarray([zu])])
    outs = [m(x, tau) for m in ens.members]
    per = np.stack([np.asarray(reconstruct_P_filt_jax(o["P_filt_base"], o["P_filt_resid"], norm["P_filt"]))
                    for o in outs])
    assert np.allclose(pred.member_std, per.std(axis=0, ddof=1), rtol=1e-12, atol=1e-300)
    assert np.allclose(pred.template, pred.P_class[1:] / pred.P_class[0] - 1, rtol=1e-15, atol=0)
    dn = np.mean([np.exp(np.asarray(o["dndx"]) * norm["dndx"]["std"] + norm["dndx"]["mean"]) for o in outs], axis=0)
    assert np.allclose(pred.dndx, dn, rtol=1e-13, atol=0)
    m = emu.wc_map
    w_ref = np.asarray(D.w_c_corrected(jnp.asarray(dn), jnp.asarray(np.polyval(m.xbar_coeffs, z)), jnp.asarray(z),
                                       D.DeltaCMap(m.delta_c_coeffs, *m.delta_c_z_clamp, "ref")))
    assert np.allclose(pred.w_c, w_ref[1:], rtol=1e-13, atol=0) and 0 < 1 - pred.w_c.sum() < 1
    # the historical constants are never used
    monkeypatch.setattr(D, "HISTORICAL_DELTA_C", D.HISTORICAL_DELTA_C._replace(coeffs=np.full((4, 3), np.nan)))
    again = emu.predict(PARAMS, z, tau_eff=tau)
    assert np.array_equal(again.w_c, pred.w_c) and np.array_equal(again.P_class, pred.P_class)
    assert not pred.P_class.flags.writeable


def test_native_modes_and_requested_k(emu):
    z = 3.4
    pred = emu.predict(PARAMS, z, tau_eff=0.5)
    k_com = 2 * np.pi * np.arange(1, RT.SYNTHETIC_ARCH["n_k"] + 1) / 120.0
    assert np.allclose(pred.k, np.asarray(K.k_skm_from_kcom(k_com, z, PARAMS["hub"], PARAMS["omegamh2"])),
                       rtol=1e-15, atol=0)
    kq = np.array([pred.k[0], 0.5 * (pred.k[2] + pred.k[3]), pred.k[-1]])
    q = emu.predict(PARAMS, z, tau_eff=0.5, k=kq)
    want = np.stack([np.interp(kq, pred.k, pred.P_class[c]) for c in range(4)])
    assert q.k.tolist() == kq.tolist() and np.allclose(q.P_class, want, rtol=1e-13, atol=0)
    assert np.allclose(q.member_std[:, 0], pred.member_std[:, 0], rtol=1e-13)
    for bad in ([pred.k[0] * 0.99], [pred.k[-1] * 1.01]):
        with pytest.raises(hp.PredictorInputError, match="outside the native mode span"):
            emu.predict(PARAMS, z, tau_eff=0.5, k=bad)
    with pytest.raises(hp.PredictorInputError, match="dimension"):
        emu.predict(PARAMS, z, tau_eff=0.5, k=[[0.01]])


def test_the_mean_flux_inputs(emu):
    z = 2.6
    a = emu.predict(PARAMS, z, tau0_amp=1.0, dtau0=0.0)
    assert a.tau_eff == pytest.approx(float(kim_tau0(z)), rel=1e-15)
    b = emu.predict(PARAMS, z, tau_eff=hp.tau_eff_priya(z, 1.1, 0.2))
    c = emu.predict(PARAMS, z, tau0_amp=1.1, dtau0=0.2)
    assert np.array_equal(b.P_class, c.P_class)
    for kw, match in (({}, "either tau_eff"), ({"tau_eff": 0.4, "tau0_amp": 1.0, "dtau0": 0.0}, "either tau_eff"),
                      ({"tau0_amp": 1.0}, "both"), ({"tau_eff": 0.0}, "positive"), ({"tau_eff": -0.1}, "positive"),
                      ({"tau_eff": np.nan}, "not finite"), ({"tau_eff": [0.4, 0.5]}, "2 values for 1")):
        with pytest.raises(hp.PredictorInputError, match=match):
            emu.predict(PARAMS, z, **kw)


@pytest.mark.parametrize("bad, match", [
    ({k: v for k, v in PARAMS.items() if k != "ns"}, "missing"),
    (dict(PARAMS, sigma8=0.8), "unknown"),
    (dict(PARAMS, ns=True), "real scalar"),
    (dict(PARAMS, ns=np.nan), "not finite"),
    (dict(PARAMS, ns=[0.95]), "real scalar"),
    (dict(PARAMS, ns="0.95"), "real scalar"),
    ([0.95] * 9, "dict"),
])
def test_parameter_refusals(emu, bad, match):
    with pytest.raises(hp.PredictorInputError, match=match):
        emu.predict(bad, 3.0, tau_eff=0.4)


def test_redshift_refusals_and_the_grid_tolerance(emu):
    assert emu.predict(PARAMS, 3.0 + 5e-10, tau_eff=0.4).z == 3.0
    for z in (3.1, 5.6, 1.8, 3.0 + 1e-8):
        with pytest.raises(hp.PredictorInputError, match="not a PRIYA snapshot redshift"):
            emu.predict(PARAMS, z, tau_eff=0.4)
    with pytest.raises(hp.PredictorInputError, match="repeats"):
        emu.predict_many(PARAMS, [3.0, 3.0], tau_eff=0.4)


def test_the_checked_entry_points_refuse_a_tracer(emu):
    with pytest.raises(TypeError, match="host values"):
        jax.jit(lambda z: emu.predict(PARAMS, z, tau_eff=0.4).P_class)(3.0)
    with pytest.raises(TypeError, match="host values"):
        jax.jit(lambda t: emu.predict(PARAMS, 3.0, tau_eff=t).P_class)(0.4)
    pred = emu.predict(PARAMS, 3.0, tau_eff=0.4)
    with pytest.raises(TypeError, match="host values"):
        jax.jit(lambda a: hp.add_hcd(pred, a))(jnp.zeros(3))


def test_the_stack_equals_the_single_predictions(emu):
    zs = [2.6, 3.0, 4.2]
    s = emu.predict_many(PARAMS, zs, tau0_amp=0.95, dtau0=-0.1)
    assert s.stacked and s.P_class.shape == (3, 4, RT.SYNTHETIC_ARCH["n_k"]) and s.w_c.shape == (3, 3)
    for i, z in enumerate(zs):
        one = emu.predict(PARAMS, z, tau0_amp=0.95, dtau0=-0.1)
        assert s.tau_eff[i] == one.tau_eff and np.allclose(s.k[i], one.k, rtol=0, atol=0)
        assert np.allclose(s.P_class[i], one.P_class, rtol=1e-13, atol=0)
        assert np.allclose(s.w_c[i], one.w_c, rtol=1e-13, atol=0)
        assert np.allclose(s.member_std[i], one.member_std, rtol=1e-11, atol=1e-300)
    t = emu.predict_many(PARAMS, zs, tau_eff=[0.3, 0.4, 0.9])
    assert t.tau_eff.tolist() == [0.3, 0.4, 0.9]
    kq = np.linspace(s.k[:, 0].max(), s.k[:, -1].min(), 3)              # inside the span at every z
    assert emu.predict_many(PARAMS, zs, tau_eff=0.4, k=kq).P_class.shape == (3, 4, 3)
    assert emu.predict_many(PARAMS, zs, tau_eff=0.4, k=np.tile(kq, (3, 1))).k.shape == (3, 3)
    with pytest.raises(hp.PredictorInputError, match="rows"):
        emu.predict_many(PARAMS, zs, tau_eff=0.4, k=np.tile(kq, (2, 1)))


# ---------------------------------------------------------------------------------------------------- amplitudes
def test_add_hcd_is_the_mixture_with_one_amplitude_per_class(emu):
    pred = emu.predict(PARAMS, 3.0, tau_eff=0.4)
    P = pred.P_class
    assert np.array_equal(hp.add_hcd(pred, [0, 0, 0]), P[0])
    w = pred.w_c
    total = hp.add_hcd(pred, w)
    struct = (1 - w.sum()) * P[0] + np.einsum("c,ck->k", w, P[1:])        # the simulation's filtered total
    assert np.allclose(total, struct, rtol=1e-14, atol=0)
    assert np.allclose(total, P[0] * (1 + np.einsum("c,ck->k", w, pred.template)), rtol=1e-13, atol=0)
    assert np.allclose(hp.add_hcd(pred, [0.5, 0.3, 0.2]), np.einsum("c,ck->k", [0.5, 0.3, 0.2], P[1:]), rtol=1e-13)


@pytest.mark.parametrize("A, match", [
    ([-1e-9, 0.1, 0.0], "A_LLS = .* is negative"),
    ([0.1, -0.2, 0.0], "A_subDLA = .* is negative"),
    ([0.1, 0.0, -0.01], "A_DLA = .* is negative"),
    ([0.6, 0.3, 0.2], "clean-class fraction .* is negative"),
    ([np.nan, 0.0, 0.0], "not finite"),
    ([np.inf, 0.0, 0.0], "not finite"),
    ([0.1, 0.1], r"shape \(3,\)"),
    ([[0.1, 0.1, 0.1]], r"shape \(3,\)"),
    ([True, False, False], "real numbers"),
])
def test_only_physical_class_fractions_are_accepted(emu, A, match):
    pred = emu.predict(PARAMS, 3.0, tau_eff=0.4)
    with pytest.raises(hp.PredictorInputError, match=match):
        hp.add_hcd(pred, A)


def test_amplitudes_per_redshift_on_a_stack(emu):
    zs = [2.6, 3.0, 4.2]
    s = emu.predict_many(PARAMS, zs, tau_eff=0.4)
    flat = hp.add_hcd(s, [0.2, 0.05, 0.02])
    assert flat.shape == (3, RT.SYNTHETIC_ARCH["n_k"])
    for i, z in enumerate(zs):
        assert np.allclose(flat[i], hp.add_hcd(emu.predict(PARAMS, z, tau_eff=0.4), [0.2, 0.05, 0.02]), rtol=1e-12)
    A = np.array([[0.2, 0.05, 0.02], [0.3, 0.05, 0.0], [0.0, 0.0, 0.0]])
    per = hp.add_hcd(s, A)
    assert np.array_equal(per[2], s.P_class[2, 0])
    assert np.allclose(hp.add_hcd(s, s.w_c)[1], hp.add_hcd(emu.predict(PARAMS, 3.0, tau_eff=0.4),
                                                           s.w_c[1]), rtol=1e-12)
    bad = A.copy()
    bad[1, 0] = 0.99
    with pytest.raises(hp.PredictorInputError, match="clean-class fraction .* at z = 3.0"):
        hp.add_hcd(s, bad)
    with pytest.raises(hp.PredictorInputError, match=r"\(3,\) or \(n_z, 3\)"):
        hp.add_hcd(s, np.zeros((4, 3)))
    with pytest.raises(TypeError, match="Prediction"):
        hp.add_hcd(s.P_class, [0, 0, 0])


def test_flags_and_labels(emu):
    p = emu.predict(PARAMS, 3.0, tau_eff=float(kim_tau0(3.0)))
    f = p.flags
    assert not f["outside_training_box"] and not f["outside_sampling_box"] and not f["mean_flux_off_ladder"]
    assert not f["z_downweighted"] and not f["w_c_xbar_extrapolated"] and not f["ns_ridge"]
    assert emu.predict(dict(PARAMS, ns=1.0), 3.0, tau_eff=0.4).flags["ns_ridge"]
    assert not emu.predict(dict(PARAMS, ns=0.995), 3.0, tau_eff=0.4).flags["ns_ridge"]
    assert emu.predict(dict(PARAMS, ns=1.1), 3.0, tau_eff=0.4).flags["params_outside_training_box"] == ["ns"]
    assert emu.predict(dict(PARAMS, ns=1.1), 3.0, tau_eff=0.4).flags["outside_training_box"]
    assert emu.predict(dict(PARAMS, herei=4.3), 3.0, tau_eff=0.4).flags["outside_sampling_box"]
    assert emu.predict(PARAMS, 3.0, tau_eff=2.0 * float(kim_tau0(3.0))).flags["mean_flux_off_ladder"]
    lo = emu.predict(PARAMS, 2.0, tau_eff=0.2).flags
    assert lo["z_downweighted"] and lo["w_c_xbar_extrapolated"] and lo["k_downweighted"]
    lab = p.labels
    assert "seed-spread diagnostic" in lab["member_std"] and "not a covariance" in lab["member_std"]
    assert lab["omissions"] == list(hp.OMISSIONS) and "no prior" in lab["amplitudes"]


# ------------------------------------------------------------------------------------- no prior, no survey key
def test_the_public_interface_has_no_prior_survey_or_dla_fraction():
    names = {n for f in (hp.load_bundle, hp.Bundle.predict, hp.Bundle.predict_many, hp.add_hcd)
             for n in inspect.signature(f).parameters}
    assert not {n for n in names if any(w in n.lower() for w in ("prior", "survey", "leg", "frac", "sigma"))}
    assert list(inspect.signature(hp.add_hcd).parameters) == ["pred", "A"]


def test_the_module_never_imports_the_historical_likelihood(info, tmp_path):
    import ast
    tree = ast.parse(open(hp.__file__).read())
    mods = {(n.module or "") for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | \
        {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any(m.split(".")[-1] in ("inference", "closure_legb", "meanflux_prior") or "lit_dndx" in m
                   for m in mods), mods
    code = ("import sys, json; sys.path.insert(0, sys.argv[1]);"
            "import hcd_analysis.predictor as hp;"
            "emu = hp.load_bundle(sys.argv[2], sys.argv[3]);"
            "p = emu.predict(json.loads(sys.argv[4]), 3.0, tau_eff=0.4); hp.add_hcd(p, p.w_c);"
            "s = emu.predict_many(json.loads(sys.argv[4]), [2.6, 3.0], tau0_amp=1.0, dtau0=0.0); hp.add_hcd(s, s.w_c);"
            "print(json.dumps(sorted(m for m in sys.modules if m.startswith('hcd_analysis'))))")
    r = subprocess.run([sys.executable, "-I", "-B", "-c", code, REPO, info.bundle_dir, info.manifest_sha256,
                        json.dumps(PARAMS)], capture_output=True, text=True, cwd=str(tmp_path),
                       env=dict(os.environ, JAX_PLATFORMS="cpu"))
    assert r.returncode == 0, r.stderr[-3000:]
    loaded = json.loads(r.stdout.strip().splitlines()[-1])
    assert "hcd_analysis.predictor" in loaded and "hcd_analysis.emulator.wc_map" in loaded
    assert not [m for m in loaded if m in FORBIDDEN or "lit_dndx" in m], loaded


def test_the_mean_flux_conversions_are_the_historical_ones():
    from hcd_analysis.emulator import mean_flux_model as N
    from hcd_analysis.emulator import meanflux_prior as M
    assert M.kim_tau0 is N.kim_tau0 and M.tau0_alpha_priya is N.tau0_alpha_priya and M.TAU0_PIVOT_Z == 3.0
    z = np.linspace(2.0, 5.4, 18)
    assert np.allclose(np.asarray(N.kim_tau0(z)), 0.0023 * (1 + z) ** 3.65, rtol=1e-15, atol=0)
    assert hp.tau_eff_priya(3.4, 1.1, 0.2) == pytest.approx(1.1 * (4.4 / 4.0) ** 0.2 * 0.0023 * 4.4 ** 3.65,
                                                            rel=1e-14)
