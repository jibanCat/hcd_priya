"""Tests for the pure helpers of scripts/desi_stage2_runner.py (no context build, no sampling, no real pkl)."""
import importlib.util
import os

import numpy as np
import pytest

S = importlib.util.spec_from_file_location("s2", os.path.join("/home/mfho/hcd_priya", "scripts", "desi_stage2_runner.py"))
M = importlib.util.module_from_spec(S); S.loader.exec_module(M)


def _stored(n=25, L=120):
    rng = np.random.default_rng(0)
    return dict(run_cfg=dict(leg="DESI", seed=20260614, n_warmup=250, n_samples=600), truth_vec=rng.standard_normal(n),
                sites_extra={"tau0_amp": dict(truth=0.9), "dtau0": dict(truth=-0.1), "k_SiIII_DESI_z1": dict(truth=0.03)},
                truth_alpha_hcd_z=rng.standard_normal((13, 3)), kept_global=np.ones(13, bool), dropped={"DESI": []}, ll_true=-782.96738742859,
                draws=rng.standard_normal((L, n)), L=L, n_div=0, names=[f"p{i}" for i in range(n)])


def test_identity_checks_pass_and_fail():
    s = _stored()
    rep = M.identity_checks(s, dict(s["run_cfg"]), s["truth_vec"], {k: v["truth"] for k, v in s["sites_extra"].items()}, s["truth_alpha_hcd_z"], s["kept_global"], s["dropped"], s["ll_true"])
    assert all(v for k, v in rep.items() if isinstance(v, bool))
    for bad in ("run_cfg", "truth", "site", "ll"):
        kw = dict(run_cfg=dict(s["run_cfg"]), truth_vec=s["truth_vec"].copy(), site_truths={k: v["truth"] for k, v in s["sites_extra"].items()},
                  truth_alpha_hcd_z=s["truth_alpha_hcd_z"], kept_global=s["kept_global"], dropped=s["dropped"], ll_true=s["ll_true"])
        if bad == "run_cfg": kw["run_cfg"]["n_warmup"] = 1000
        if bad == "truth": kw["truth_vec"][3] += 1e-9
        if bad == "site": kw["site_truths"]["dtau0"] = -0.2
        if bad == "ll": kw["ll_true"] = s["ll_true"] + 1e-5
        with pytest.raises(M.Stage2Refusal):
            M.identity_checks(s, **kw)


def test_compare_replica_and_pilot_gate():
    a = np.random.default_rng(1).standard_normal((120, 25))
    c = M.compare_replica(a, a.copy(), 120); assert c["bit_identical"] and c["close_1e_8"]
    c2 = M.compare_replica(a, a + 1e-6, 120); assert (not c2["bit_identical"]) and (not c2["close_1e_8"]) and c2["same_shape"]
    c3 = M.compare_replica(a, a[:100], 120); assert not c3["same_shape"] and c3["max_abs_diff"] is None
    bat = dict(rhat=np.array([1.002, 1.005]), ess_bulk=np.array([900., 700.]), ess_tail=np.array([800., 500.]), ebfmi=np.array([0.9, 0.85]), treedepth_sat_frac=0.0)
    g = M.pilot_gate(bat, [0, 0, 0, 0]); assert g["passed"]
    g2 = M.pilot_gate(dict(bat, rhat=np.array([1.02, 1.0])), [0, 0, 0, 0]); assert not g2["passed"] and not g2["conds"]["rhat"]
    g3 = M.pilot_gate(bat, [0, 1, 0, 0]); assert not g3["passed"] and not g3["conds"]["divergences"]
    g4 = M.pilot_gate(dict(bat, ess_tail=np.array([800., 300.])), [0, 0, 0, 0]); assert not g4["passed"]


def test_a2c_argv_template_is_the_frozen_invocation():
    argv = [x.format(m=45, scratch="/tmp/x") for x in M.A2C_ARGV_TEMPLATE]
    for tok in ("--deployed-prior", "--leg", "DESI", "--no-shard-pkl", "--metal-selfdraw", "--fres-selfdraw", "--n-shards", "48", "--n-mocks", "48"):
        assert tok in argv
    assert M.STRONG == dict(n_chains=4, n_warmup=1000, n_samples=600, max_tree_depth=10, target_accept=0.9, dense_mass=True)
    assert M.STORED == dict(n_warmup=250, n_samples=600, max_tree_depth=10, seed=20260614)
