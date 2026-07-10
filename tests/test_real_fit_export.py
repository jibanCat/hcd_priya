"""Task #4 Stage 2 -- two EXPORT-ONLY fixes in scripts/run_real_fit.py (TDD).

FIX 1: run_real_fit._loglik_chain must MIRROR the deployed _legb_model call
(closure_legb.py:1856-1859) -- thread metal_nodes (flatlog2node), b_res_global (f_res),
alpha_res (fix_alpha_res pin) AND a_siiii/a_siii into _data_loglik_legcore, not just a_siiii.
We unit-test the PURE reconstruction helper _reconstruct_nuisance against SYNTHETIC per-draw
dicts (no real ctx / no NUTS), and assert the uniform/no-f_res path is byte-identical (yields
metal_nodes=None + b_res=None + the same a_siiii => the same _data_loglik_legcore call).

FIX 2: export the f_res / metal-node posteriors next to the chains so railing is visible. We
test that export_getdist writes the companion nuisance artifact (npz raw draws + json rail
summaries) and it round-trips.

These are NUISANCE posteriors, NOT the blinded A_p/n_s -- exported UNBLINDED (like the health
json). Tests are LIGHT: pure helpers on synthetic numpy dicts + a synthetic result dict.

Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_real_fit_export.py -q
"""
import importlib.util
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

import hcd_analysis.emulator  # noqa: F401  x64 BEFORE jax
from hcd_analysis.emulator import closure_legb as CL
from hcd_analysis.emulator.inference import PARAM_NAMES

REPO = "/home/mfho/hcd_priya"


def _load_script(name):
    """Load a scripts/<name>.py module by path (scripts/ is not an importable package)."""
    path = os.path.join(REPO, "scripts", f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RF = _load_script("run_real_fit")


def _leg(name, metals_on):
    return SimpleNamespace(name=name, metals_on=metals_on)


def _ctx(metal_prior="uniform", legs=(), siII=("DESI",), sample_res=False,
         fix_alpha_res=False, z_global=(2.2, 2.6, 3.0), sample_metals=True):
    return SimpleNamespace(
        metal_prior=metal_prior, legs=list(legs), metal_siII_legs=tuple(siII),
        sample_res=sample_res, fix_alpha_res=fix_alpha_res, z_global=np.asarray(z_global),
        sample_metals=sample_metals)


# --------------------------------------------------------------------------------------------- #
#  FIX 1 -- the PURE reconstruction helper _reconstruct_nuisance.
# --------------------------------------------------------------------------------------------- #
def test_reconstruct_flatlog2node_metal_nodes():
    """flatlog2node: metal_nodes = {leg: (f3, f2, k3, k2)} for EACH metals_on leg, in ctx.legs
    order; f2/k2 are the SiII nodes ONLY on metal_siII_legs (DESI), None on eBOSS. The per-node
    values are jnp.stack([z0, z1]) of the sampled sites (mirrors _metal_2node_sites). a_siiii/a_siii
    are 0.0 on this path (the scalar amplitudes are unused)."""
    ctx = _ctx(metal_prior="flatlog2node",
               legs=(_leg("DESI", True), _leg("eBOSS", True)), siII=("DESI",))
    draw = {
        "f_SiIII_DESI_z0": 0.004, "f_SiIII_DESI_z1": 0.02,
        "k_SiIII_DESI_z0": 2e-3, "k_SiIII_DESI_z1": 5e-2,
        "f_SiII_DESI_z0": 0.005, "f_SiII_DESI_z1": 0.006,
        "k_SiII_DESI_z0": 3e-3, "k_SiII_DESI_z1": 4e-3,
        "f_SiIII_eBOSS_z0": 0.007, "f_SiIII_eBOSS_z1": 0.008,
        "k_SiIII_eBOSS_z0": 6e-3, "k_SiIII_eBOSS_z1": 7e-3,
    }
    metal_nodes, b_res, alpha_res, a_siiii, a_siii = RF._reconstruct_nuisance(ctx, draw)
    assert set(metal_nodes.keys()) == {"DESI", "eBOSS"}
    # DESI: SiIII + SiII nodes
    f3, f2, k3, k2 = metal_nodes["DESI"]
    assert np.allclose(np.asarray(f3), [0.004, 0.02])
    assert np.allclose(np.asarray(k3), [2e-3, 5e-2])
    assert np.allclose(np.asarray(f2), [0.005, 0.006])
    assert np.allclose(np.asarray(k2), [3e-3, 4e-3])
    # eBOSS: SiIII only (not a metal_siII_leg) -> f2/k2 None
    f3e, f2e, k3e, k2e = metal_nodes["eBOSS"]
    assert np.allclose(np.asarray(f3e), [0.007, 0.008])
    assert np.allclose(np.asarray(k3e), [6e-3, 7e-3])
    assert f2e is None and k2e is None
    # scalar amplitudes unused on the node path; no f_res in this draw
    assert float(a_siiii) == 0.0 and float(a_siii) == 0.0
    assert b_res is None
    assert alpha_res is None   # fix_alpha_res False + no alpha_res site


def test_reconstruct_flatlog2node_without_sample_metals_is_empty():
    """flatlog2node but sample_metals=False (defense-in-depth): mirror _metal_2node_sites, which
    returns {} when not sampling. The reconstruction must NOT read the (absent) node sites (no
    KeyError) -- it yields metal_nodes={} + a_siiii=a_siii=0 (metals-off), matching the model."""
    ctx = _ctx(metal_prior="flatlog2node", legs=(_leg("DESI", True),), sample_metals=False)
    metal_nodes, b_res, alpha_res, a_siiii, a_siii = RF._reconstruct_nuisance(ctx, {})
    assert metal_nodes == {}                         # empty, mirrors _metal_2node_sites (no KeyError)
    assert float(a_siiii) == 0.0 and float(a_siii) == 0.0


def test_reconstruct_no_metal_no_fres_is_byte_identical_call():
    """uniform + no f_res + not-fixed alpha_res + only a_SiIII present: the reconstruction yields
    metal_nodes=None, b_res=None, alpha_res=None, a_siiii=a_SiIII, a_siii=0.0 -- i.e. the EXACT
    pre-fix a_siiii-only _data_loglik_legcore call (byte-identity guard)."""
    ctx = _ctx(metal_prior="uniform", legs=(_leg("KS", False),))
    draw = {"a_SiIII": 0.031}
    metal_nodes, b_res, alpha_res, a_siiii, a_siii = RF._reconstruct_nuisance(ctx, draw)
    assert metal_nodes is None
    assert b_res is None
    assert alpha_res is None
    assert float(a_siiii) == pytest.approx(0.031)
    assert float(a_siii) == 0.0


def test_reconstruct_no_a_siiii_defaults_zero():
    """uniform + no a_SiIII site (pure KS): a_siiii falls back to 0.0 (byte-exact golden)."""
    ctx = _ctx(metal_prior="uniform", legs=(_leg("KS", False),))
    metal_nodes, b_res, alpha_res, a_siiii, a_siii = RF._reconstruct_nuisance(ctx, {})
    assert metal_nodes is None and b_res is None and alpha_res is None
    assert float(a_siiii) == 0.0 and float(a_siii) == 0.0


def test_reconstruct_fix_alpha_res_pins_noop():
    """NORC deployed real fit: ctx.fix_alpha_res=True pins alpha_res=(1.0, 0.0) (the model's
    forward no-op), NOT sampled -- mirrors _legb_model:1839-1840."""
    ctx = _ctx(metal_prior="uniform", legs=(_leg("DESI", True),), fix_alpha_res=True)
    _, _, alpha_res, _, _ = RF._reconstruct_nuisance(ctx, {"a_SiIII": 0.01})
    assert alpha_res == (1.0, 0.0)


def test_reconstruct_alpha_res_from_samples_when_sampled():
    """When alpha_res IS sampled (fix_alpha_res False, sites present), read the (amp, slope)
    tuple from the draw (mirrors the else branch of _legb_model:1842-1843)."""
    ctx = _ctx(metal_prior="uniform", legs=(_leg("DESI", True),), fix_alpha_res=False)
    draw = {"alpha_res": 1.1, "alpha_res_slope": -0.2, "a_SiIII": 0.01}
    _, _, alpha_res, _, _ = RF._reconstruct_nuisance(ctx, draw)
    assert float(alpha_res[0]) == pytest.approx(1.1)
    assert float(alpha_res[1]) == pytest.approx(-0.2)


def test_reconstruct_sample_res_builds_bres():
    """sample_res=True + f_res_amp/slope in the draw: b_res_global == _bres_of_z(z_global, amp,
    slope) (the deployed option-b forward, closure_legb:1853). amp=0 => b_res identically 0."""
    ctx = _ctx(metal_prior="flatlog2node", legs=(_leg("DESI", True),), sample_res=True)
    draw = {"f_res_amp": 0.03, "f_res_slope": 0.4,
            "f_SiIII_DESI_z0": 0.004, "f_SiIII_DESI_z1": 0.02,
            "k_SiIII_DESI_z0": 2e-3, "k_SiIII_DESI_z1": 5e-2,
            "f_SiII_DESI_z0": 0.005, "f_SiII_DESI_z1": 0.006,
            "k_SiII_DESI_z0": 3e-3, "k_SiII_DESI_z1": 4e-3}
    _, b_res, _, _, _ = RF._reconstruct_nuisance(ctx, draw)
    want = np.asarray(CL._bres_of_z(np.asarray(ctx.z_global), 0.03, 0.4))
    assert np.allclose(np.asarray(b_res), want)
    # amp=0 -> b_res == 0 for any slope (the golden no-op)
    draw0 = dict(draw); draw0["f_res_amp"] = 0.0
    _, b_res0, _, _, _ = RF._reconstruct_nuisance(ctx, draw0)
    assert np.allclose(np.asarray(b_res0), 0.0)


def test_nuisance_export_keys_selects_fres_and_metal_nodes():
    """the export-key collector picks up f_res_amp/slope + every f_/k_ metal-node key, and NOTHING
    else (theta/tau0/alpha/a_SiIII are NOT nuisance-node keys here)."""
    samples = {
        "theta_unit": 1, "tau0_vec": 1, "alpha_hcd_z": 1, "alpha_lls": 1, "a_SiIII": 1,
        "f_res_amp": 1, "f_res_slope": 1,
        "f_SiIII_DESI_z0": 1, "k_SiII_DESI_z1": 1, "f_SiII_DESI_z0": 1, "k_SiIII_eBOSS_z1": 1}
    keys = set(RF._nuisance_export_keys(samples))
    assert keys == {"f_res_amp", "f_res_slope", "f_SiIII_DESI_z0", "k_SiII_DESI_z1",
                    "f_SiII_DESI_z0", "k_SiIII_eBOSS_z1"}


# --------------------------------------------------------------------------------------------- #
#  FIX 2 -- export_getdist writes the nuisance companion artifact (npz raw + json rail summary).
# --------------------------------------------------------------------------------------------- #
def _minimal_result(nuisance_chains, bounds):
    """A synthetic run_real_fit() result dict (tiny finite arrays) sufficient for export_getdist."""
    C, N = len(nuisance_chains), 5
    names = list(PARAM_NAMES) + ["tau0_z0", "alpha_lls", "alpha_subdla", "alpha_dla"]
    P = len(names)
    rng = np.random.default_rng(0)
    packed = rng.normal(size=(C, N, P))
    packed[:, :, :len(PARAM_NAMES)] = 0.5   # unit-cube theta block (finite denorm)
    battery = dict(rhat_max=1.001, ess_bulk_min=100.0, ess_tail_min=100.0, ebfmi_min=0.7,
                   n_divergent=0, treedepth_sat_frac=0.0)
    return dict(packed=packed, names=names, battery=battery, per_chain_div=[0] * C,
                members=[f"{REPO}/checkpoints/final_prod_seed0.eqx"], leg_name="DESI",
                n_real_rows=10, ll_chains=[np.zeros(N) for _ in range(C)],
                kept_global=np.array([True]),
                nuisance_chains=nuisance_chains, nuisance_bounds=bounds)


def test_export_writes_nuisance_artifact_and_roundtrips(tmp_path):
    """export_getdist writes <root>.nuisance.npz (per-chain raw arrays) + <root>.nuisance.json
    (rail summaries) with the expected site keys; the npz round-trips the arrays and the json
    carries frac_near_lo/hi for the LogUniform f-node sites."""
    N = 5
    ch0 = {"f_SiIII_DESI_z0": np.array([0.0031, 0.02, 0.0031, 0.029, 0.01]),
           "f_res_amp": np.array([0.001, -0.002, 0.0, 0.003, 0.0015])}
    ch1 = {"f_SiIII_DESI_z0": np.array([0.02, 0.02, 0.0298, 0.005, 0.006]),
           "f_res_amp": np.array([0.002, 0.0, -0.001, 0.0, 0.001])}
    assert all(len(v) == N for v in {**ch0, **ch1}.values())
    bounds = {"f": (0.003, 0.03), "k": (1e-3, 0.1)}
    result = _minimal_result([ch0, ch1], bounds)

    out_dir = str(tmp_path)
    RF.export_getdist(result, out_dir, "real_desi", offset={"ns": 0.0, "Ap": 0.0},
                      blind=False, survey="desi")

    npz_path = os.path.join(out_dir, "real_desi.nuisance.npz")
    json_path = os.path.join(out_dir, "real_desi.nuisance.json")
    assert os.path.exists(npz_path) and os.path.exists(json_path)

    z = np.load(npz_path)
    # per-chain arrays present, shape (C, N), round-trip exact
    assert set(z.files) >= {"f_SiIII_DESI_z0", "f_res_amp"}
    got = z["f_SiIII_DESI_z0"]
    assert got.shape == (2, N)
    assert np.allclose(got[0], ch0["f_SiIII_DESI_z0"])
    assert np.allclose(got[1], ch1["f_SiIII_DESI_z0"])

    with open(json_path) as f:
        summ = json.load(f)
    assert summ["survey"] == "desi"
    sites = summ["sites"]
    assert "f_SiIII_DESI_z0" in sites and "f_res_amp" in sites
    fnode = sites["f_SiIII_DESI_z0"]
    # LogUniform f-node carries rail fractions (bounds 0.003/0.03); some draws sit at each rail
    assert "frac_near_lo" in fnode and "frac_near_hi" in fnode
    assert 0.0 <= fnode["frac_near_lo"] <= 1.0 and 0.0 <= fnode["frac_near_hi"] <= 1.0
    assert fnode["frac_near_lo"] > 0.0 and fnode["frac_near_hi"] > 0.0
    # f_res is a Normal prior (no LogUniform rail) -> stats but null rail fractions
    fres = sites["f_res_amp"]
    assert "mean" in fres and "std" in fres
    assert fres["frac_near_lo"] is None and fres["frac_near_hi"] is None


def test_export_without_nuisance_is_noop(tmp_path):
    """No nuisance_chains (KS uniform / legacy result): export_getdist writes NO nuisance artifact
    (additive-only, does not disturb the existing chain columns)."""
    result = _minimal_result([{}, {}], {"f": (0.003, 0.03), "k": (1e-3, 0.1)})  # 2 chains, no sites
    out_dir = str(tmp_path)
    RF.export_getdist(result, out_dir, "real_ks", offset={"ns": 0.0, "Ap": 0.0},
                      blind=False, survey="ks")
    assert not os.path.exists(os.path.join(out_dir, "real_ks.nuisance.npz"))
    assert not os.path.exists(os.path.join(out_dir, "real_ks.nuisance.json"))
    # the normal chain export is untouched
    assert os.path.exists(os.path.join(out_dir, "real_ks.1.txt"))
