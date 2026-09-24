"""Tests for the pure helpers of scripts/desi_stage2_runner.py (no context build, no sampling, no real pkl)."""
import importlib.util
import json
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
    bat = dict(rhat={"a": 1.002, "b": 1.005}, ess_bulk={"a": 900., "b": 700.}, ess_tail={"a": 800., "b": 500.}, ebfmi=np.array([0.9, 0.85, 0.8, 0.95]), treedepth_sat_frac=0.0)
    g = M.pilot_gate(bat, [0, 0, 0, 0]); assert g["passed_sampler_criteria"] and g["pilot_gate_passed"]
    g2 = M.pilot_gate(dict(bat, rhat={"a": 1.02, "b": 1.0}), [0, 0, 0, 0]); assert not g2["pilot_gate_passed"] and not g2["conds"]["rhat"]
    g3 = M.pilot_gate(bat, [0, 1, 0, 0]); assert not g3["pilot_gate_passed"] and not g3["conds"]["divergences"]
    g4 = M.pilot_gate(dict(bat, ess_tail={"a": 800., "b": 300.}), [0, 0, 0, 0]); assert not g4["pilot_gate_passed"]


def test_a2c_argv_template_is_the_frozen_invocation():
    argv = [x.format(m=45, scratch="/tmp/x") for x in M.A2C_ARGV_TEMPLATE]
    for tok in ("--deployed-prior", "--leg", "DESI", "--no-shard-pkl", "--metal-selfdraw", "--fres-selfdraw", "--n-shards", "48", "--n-mocks", "48"):
        assert tok in argv
    assert M.STRONG == dict(n_chains=4, n_warmup=1000, n_samples=600, max_tree_depth=10, target_accept=0.9, dense_mass=True)
    assert M.STORED == dict(n_warmup=250, n_samples=600, max_tree_depth=10, seed=20260614)


def test_pilot_gate_on_real_battery_output():
    """M1 (Reviewer F): the gate must accept CL.convergence_battery's dict-valued diagnostics."""
    import hcd_analysis.emulator  # noqa: F401
    from hcd_analysis.emulator import closure_legb as CL
    rng = np.random.default_rng(5)
    packed = rng.standard_normal((4, 600, 6)); names = [f"p{i}" for i in range(6)]
    energy = rng.standard_normal((4, 600)).cumsum(axis=1) * 0.01 + rng.standard_normal((4, 600))
    steps = np.full((4, 600), 15)
    bat = CL.convergence_battery(packed, names, energy=energy, num_steps=steps, max_tree_depth=10, n_div=0)
    assert isinstance(bat["rhat"], dict)
    g = M.pilot_gate(bat, [0, 0, 0, 0], n_chains=4, cpu_h=80.0, identity_ok=True, extra_sites={"tau0_amp": dict(rhat=1.001, ess=1500.0)})
    assert g["passed_sampler_criteria"] and g["cost_ok"] and g["pilot_gate_passed"] and g["ebfmi_count"] == 4
    g2 = M.pilot_gate(bat, [0, 0, 0, 0], n_chains=4, cpu_h=200.0); assert g2["passed_sampler_criteria"] and (not g2["cost_ok"]) and (not g2["pilot_gate_passed"])
    g3 = M.pilot_gate(bat, [0, 0, 0, 0], n_chains=4, extra_sites={"k_SiIII_DESI_z1": dict(rhat=1.03, ess=900.0)}); assert not g3["conds"]["c4_sites_rhat"] and not g3["pilot_gate_passed"]
    bad = dict(bat); bad["rhat"] = dict(bat["rhat"], p2=float("nan"))
    g4 = M.pilot_gate(bad, [0, 0, 0, 0], n_chains=4); assert not g4["conds"]["finite"] and not g4["passed_sampler_criteria"]
    g5 = M.pilot_gate(bat, [0, 0, 0, 0], n_chains=5); assert not g5["conds"]["finite"]        # ebfmi count != n_chains


@pytest.mark.skipif(os.environ.get("DESI_STAGE2_SMOKE") != "1", reason="end-to-end smoke on the real frozen context with tiny NUTS; set DESI_STAGE2_SMOKE=1")
def test_end_to_end_smoke_tiny_nuts(tmp_path):
    """M3 (Reviewer F): the whole path (frozen driver, identity checks, replica, 4-chain strong run, partial checkpoints, battery, gate,
    pkl + json) with _run_nuts_legb forced to n_warmup=5, n_samples=60. Reads the real stored pkl for mock 45 (allowed: Stage 2 is authorized)."""
    import hcd_analysis.emulator  # noqa: F401
    from hcd_analysis.emulator import closure_legb as CL
    real = CL._run_nuts_legb
    def tiny(*a, **kw):
        # tiny settings: an unadapted dense mass matrix with the production tree depth would saturate at 1023 leapfrogs per step,
        # so the smoke caps the tree depth too (it exercises the path, not the sampler)
        kw["n_warmup"] = 10; kw["n_samples"] = 40; kw["max_tree_depth"] = 3; return real(*a, **kw)
    CL._run_nuts_legb = tiny
    try:
        out = tmp_path / "s2"
        summ = M.run_one(45, "/scratch/cavestru_root/cavestru1/mfho/cert_2026-07/armp_DESI_corrected_v1",
                         "/home/mfho/hcd_priya_notes/docs/superpowers/a3c-artifacts/a2c_n48_sha256.txt", str(out))
    finally:
        CL._run_nuts_legb = real
    assert (out / "stage2_mock_0045.pkl").exists() and (out / "stage2_mock_0045.json").exists()
    assert all(v for k, v in summ["identity"].items() if isinstance(v, bool))
    assert summ["replica"]["compare"]["same_shape"] is False or summ["replica"]["compare"]["L_replica"] > 0     # tiny run: not the stored L
    assert "gate" in summ["strong"] and "battery" in summ["strong"] and "site_diagnostics_non_packed" in summ["strong"]
    assert set(summ["strong"]["site_diagnostics_non_packed"]) >= {"tau0_amp", "dtau0", "k_SiIII_DESI_z1"}
    part = out / "stage2_mock_0045.partial"; assert sorted(p.name for p in part.iterdir()) == ["chain_0.pkl", "chain_1.pkl", "chain_2.pkl", "chain_3.pkl", "replica.pkl"]
    j = json.load(open(out / "stage2_mock_0045.json")); assert j["strong"]["gate"]["pilot_gate_passed"] in (True, False)
