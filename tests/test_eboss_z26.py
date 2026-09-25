"""PI #28 (2026-09-25): the z >= 2.6 eBOSS product. Readout in plain mode (exported unblinded), z grid from health.z_kept,
two-sided A_P rule against the canonical chain, gate enforcement; and the 2.2 versus 2.6 descriptive comparison script.
Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
  /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_eboss_z26.py -q -p no:cacheprovider
"""
import importlib.util
import json
import os

import numpy as np
import pytest

REPO = "/home/mfho/hcd_priya"
Z13 = [2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6]
THETA = ["ns", "Ap", "herei", "heref", "alphaq", "hub", "omegamh2", "hireionz", "bhfeedback"]
ALPHA = ["alpha_lls", "alpha_subdla", "alpha_dla"]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, "scripts", f"{name}.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


CONS = _load("eboss_priya_consistency")
ZR = _load("eboss_zrange_compare")


def _product(d, root, z, *, blinded, ns=0.898, Ap=1.25e-9, tau0_amp=1.22, dtau0=-0.27, n=400, n_chains=2, seed=1, n_div=0, alphaq=2.45,
             z_kept=True, unblinded_files=False):
    d.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(seed); z = np.asarray(z, float)
    names = THETA + [f"tau0_z{i}" for i in range(z.size)] + ALPHA
    for c in range(n_chains):
        X = np.empty((n, len(names)))
        X[:, 0] = ns + 0.012 * rng.standard_normal(n)
        X[:, 1] = np.clip(Ap + 0.08e-9 * rng.standard_normal(n), 1.2e-9, 2.6e-9)
        X[:, 2] = 3.8; X[:, 3] = 2.9; X[:, 4] = np.clip(alphaq + 0.02 * rng.standard_normal(n), 1.3, 2.5); X[:, 5] = 0.70; X[:, 6] = 0.143; X[:, 7] = 7.2; X[:, 8] = 0.05
        amp = np.clip(tau0_amp + 0.01 * rng.standard_normal(n), 0.75, 1.25); dt = np.clip(dtau0 + 0.02 * rng.standard_normal(n), -0.4, 0.25)
        X[:, 9:9 + z.size] = amp[:, None] * ((1 + z)[None, :] / 4.0) ** dt[:, None] * CONS.kim_tau0(z)[None, :]
        X[:, 9 + z.size:] = [0.17, 0.06, 0.004]
        table = np.column_stack([np.ones(n), np.zeros(n), X])
        fn = f"{root}.unblinded.{c + 1}.txt" if unblinded_files else f"{root}.{c + 1}.txt"
        np.savetxt(d / fn, table, fmt=["%.10g"] * table.shape[1], header="weight  minusloglike  " + "  ".join(names))
    (d / f"{root}.paramnames").write_text("".join(f"{nm}\t{nm}\n" for nm in names))
    h = dict(rhat_max=1.003, ess_bulk_min=900, ess_tail_min=700, ebfmi_min=0.9, n_divergent=n_div, per_chain_div=[n_div] + [0] * (n_chains - 1),
             treedepth_sat_frac=0.0, n_chains=n_chains, n_draws=n, seed=seed, blinded=blinded, blind_params=["ns", "Ap"], leg="eBOSS")
    if z_kept:
        h["z_kept"] = z.tolist(); h["eboss_zlo"] = float(z[0]) if z.size < 13 else None
    (d / f"{root}.health.json").write_text(json.dumps(h))
    sites = {k: 0.01 * (1 + 0.05 * rng.standard_normal((n_chains, n))) for k in ("f_SiIII_eBOSS_z0", "k_SiIII_eBOSS_z1")}
    sites["f_res_amp"] = -0.04 + 0.006 * rng.standard_normal((n_chains, n)); sites["f_res_slope"] = 0.5 + 0.3 * rng.standard_normal((n_chains, n))   # Normal-prior sites: null rails (K2 M1)
    np.savez(d / f"{root}.nuisance.npz", **sites)
    (d / f"{root}.nuisance.json").write_text(json.dumps(dict(sites={k: (dict(frac_near_lo=None, frac_near_hi=None) if k.startswith("f_res") else dict(frac_near_lo=0.0, frac_near_hi=0.0)) for k in sites})))
    return d


def _lock(tmp):
    p = tmp / "analysis.lock"
    p.write_text(json.dumps(dict(legs=dict(eBOSS=dict(z=Z13, prior=dict(tau0_pivot_z=3.0))))))
    return str(p)


def _ref_z26(tmp):
    ref = dict(reference="test", primary_chain="chain1", descriptive_chain="chain3", chains=dict(
        chain1={"n_P": dict(median=1.009, err_plus=0.027, err_minus=0.018), "A_P_1e-9": dict(median=1.69, err_plus=0.14, err_minus=0.15),
                "tau0": dict(median=1.082, err_plus=0.018, err_minus=0.026), "dtau0": dict(median=-0.013, err_plus=0.047, err_minus=0.041),
                "alpha_q": dict(lower68=2.86), "v_scale_h": dict(median=0.70)},
        chain3={"n_P": dict(median=0.898, err_plus=0.012, err_minus=0.013), "A_P_1e-9": dict(upper68=1.33, upper95=1.44),
                "tau0": dict(median=1.221, err_plus=0.021, err_minus=0.012), "dtau0": dict(median=-0.27, err=0.029)}))
    p = tmp / "ref_z26.json"; p.write_text(json.dumps(ref)); return str(p)


@pytest.mark.parametrize("Ap,expect", [(1.69e-9, "CONSISTENT"), (1.60e-9, "CONSISTENT"), (1.30e-9, "SHIFTED"), (1.15e-9, "DISCREPANT")])
def test_plain_mode_two_sided_Ap_against_canonical(tmp_path, Ap, expect):
    d = _product(tmp_path / "z26", "real_eboss_z26", Z13[2:], blinded=False, ns=1.0, Ap=Ap, tau0_amp=1.09, dtau0=-0.02)
    out = tmp_path / "ro"
    CONS.main(["--chain-dir", str(d), "--root", "real_eboss_z26", "--reference", _ref_z26(tmp_path), "--analysis-lock", _lock(tmp_path),
               "--out", str(out), "--chain-files", "plain", "--require-gate"])
    r = json.load(open(str(out) + ".json"))
    assert r["chain_files_mode"] == "plain" and r["z_source"] == "health.z_kept" and r["z_grid"] == Z13[2:]
    assert r["tests"]["Ap"]["label"] == expect, r["tests"]["Ap"]
    assert r["tests"]["Ap"]["rule"]["consistent_abs"] == 0.15e-9 and "two-sided" in r["tests"]["Ap"]["rule_note"]
    assert r["tests"]["ns"]["label"] == "CONSISTENT" and r["tests"]["tau0_amp"]["label"] == "CONSISTENT" and r["tests"]["dtau0"]["label"] == "CONSISTENT"
    assert r["descriptive_chain"] == "chain3" and any(k.startswith("Ap_vs_chain3") for k in r["descriptive_vs_fiducial"])
    assert r["health_gate"]["label"] == "GREEN"
    assert (r["attribution"] is None) == (expect == "CONSISTENT")


def test_plain_mode_refuses_red_gate_blinded_product_and_mixed_files(tmp_path):
    lock = _lock(tmp_path); ref = _ref_z26(tmp_path)
    d = _product(tmp_path / "red", "real_eboss_z26", Z13[2:], blinded=False, n_div=2)
    with pytest.raises(SystemExit) as e:
        CONS.main(["--chain-dir", str(d), "--root", "real_eboss_z26", "--reference", ref, "--analysis-lock", lock, "--out", str(tmp_path / "x"), "--chain-files", "plain", "--require-gate"])
    assert e.value.code == 3 and not os.path.exists(str(tmp_path / "x.json"))
    d2 = _product(tmp_path / "bl", "real_eboss_z26", Z13[2:], blinded=True)
    with pytest.raises(SystemExit):
        CONS.main(["--chain-dir", str(d2), "--root", "real_eboss_z26", "--reference", ref, "--analysis-lock", lock, "--out", str(tmp_path / "y"), "--chain-files", "plain"])
    d3 = _product(tmp_path / "mix", "real_eboss_z26", Z13[2:], blinded=False)
    (d3 / "real_eboss_z26.unblinded.1.txt").write_text("# stray\n")
    with pytest.raises(SystemExit):
        CONS.main(["--chain-dir", str(d3), "--root", "real_eboss_z26", "--reference", ref, "--analysis-lock", lock, "--out", str(tmp_path / "w"), "--chain-files", "plain"])


def test_z_kept_must_be_subset_of_lock_grid_and_ladder_count_must_match(tmp_path):
    lock = _lock(tmp_path); ref = _ref_z26(tmp_path)
    d = _product(tmp_path / "bad", "real_eboss_z26", [2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.7], blinded=False)
    with pytest.raises(SystemExit):
        CONS.main(["--chain-dir", str(d), "--root", "real_eboss_z26", "--reference", ref, "--analysis-lock", lock, "--out", str(tmp_path / "x"), "--chain-files", "plain"])
    d2 = _product(tmp_path / "nokept", "real_eboss_z26", Z13[2:], blinded=False, z_kept=False)     # no z_kept -> lock grid (13) vs 11 columns
    with pytest.raises(SystemExit):
        CONS.main(["--chain-dir", str(d2), "--root", "real_eboss_z26", "--reference", ref, "--analysis-lock", lock, "--out", str(tmp_path / "y"), "--chain-files", "plain"])


def test_zrange_compare_reports_shifts_rails_and_shared_rungs(tmp_path):
    lock = _lock(tmp_path)
    full = _product(tmp_path / "full", "real_eboss", Z13, blinded=True, ns=0.897, Ap=1.21e-9, tau0_amp=1.24, dtau0=-0.26, unblinded_files=True, z_kept=False)
    sub = _product(tmp_path / "sub", "real_eboss_z26", Z13[2:], blinded=False, ns=0.95, Ap=1.50e-9, tau0_amp=1.10, dtau0=-0.05, seed=2)
    out = tmp_path / "cmp"
    ZR.main(["--full-dir", str(full), "--sub-dir", str(sub), "--analysis-lock", lock, "--out", str(out)])
    r = json.load(open(str(out) + ".json"))
    assert r["dropped_z"] == [2.2, 2.4] and r["sub"]["z"] == Z13[2:] and r["full"]["z"] == Z13
    c = r["comparison"]
    assert c["ns"]["delta_sub_minus_full"] > 0.04 and c["ns"]["delta_over_full_sd"] > 3
    assert c["Ap"]["delta_sub_minus_full"] > 0.2e-9 and c["tau0_amp"]["delta_sub_minus_full"] < -0.1 and c["dtau0"]["delta_sub_minus_full"] > 0.15
    assert "tau0(z=2.6)" in c and "tau0(z=4.6)" in c and "tau0(z=2.2)" not in c
    assert "rails" in c["Ap"] and c["Ap"]["rails"]["full"]["near_lo"] > 0.3 and c["Ap"]["rails"]["sub"]["near_lo"] < 0.05
    assert "nuis:f_SiIII_eBOSS_z0" in c
    md = open(str(out) + ".md").read(); assert "sub minus full" in md and "tau0(z=3.0)" in md
    with pytest.raises(SystemExit):
        ZR.main(["--full-dir", str(full), "--sub-dir", str(sub), "--analysis-lock", lock, "--out", str(out)])   # runs once


# --------------------------------------------------------------------------------------------- #
#  Reviewer K (S3) and PI #29: the REAL pinned reference files, the attribution path, exactly-once, gate implied in plain mode.
# --------------------------------------------------------------------------------------------- #
NOTES_LANE = "/home/mfho/hcd_priya_notes/docs/superpowers/eboss-realdata-2026-09"
REF_RELEASED = os.path.join(NOTES_LANE, "reference_fernandez2024_released_chains.json")
REF_RELEASED_FULL = os.path.join(NOTES_LANE, "reference_fernandez2024_released_chains_fullrange_primary.json")
REF_TABLE3 = os.path.join(NOTES_LANE, "reference_fernandez2024_table3.json")


@pytest.mark.skipif(not os.path.exists(REF_RELEASED), reason="notes lane not available")
def test_real_released_reference_z26_shifted_case_runs_attribution(tmp_path):
    """Plain-mode readout of a synthetic z >= 2.6 product against the REAL released-chains reference (primary = the released
    canonical z = 2.6 to 4.6 chain): a planted tau0_amp shift makes the attribution step run; the primary shift is mean-based."""
    d = _product(tmp_path / "z26", "real_eboss_z26", Z13[2:], blinded=False, ns=1.005, Ap=1.65e-9, tau0_amp=1.16, dtau0=-0.02, seed=5)
    out = tmp_path / "ro"
    CONS.main(["--chain-dir", str(d), "--root", "real_eboss_z26", "--reference", REF_RELEASED, "--analysis-lock", _lock(tmp_path), "--out", str(out), "--chain-files", "plain"])
    r = json.load(open(str(out) + ".json"))
    assert r["primary_chain"] == "chain1_released_z2.6_4.6" and r["descriptive_chain"] == "chain3_released_z2.2_4.6"
    t = r["tests"]
    assert t["tau0_amp"]["label"] in ("SHIFTED", "DISCREPANT") and t["tau0_amp"]["delta_basis"].startswith("mean minus mean")
    assert abs(t["tau0_amp"]["delta"] - (r["summaries"]["tau0_amp"]["mean"] - 1.0818806311217888)) < 1e-9
    assert abs(t["ns"]["ref_central"] - 1.0087378615010567) < 1e-12 and t["Ap"]["rule_note"].startswith("two-sided")
    assert r["attribution"] is not None and "hub" in r["attribution"]["linear_response"] and "alphaq" in r["attribution"]["linear_response"]
    assert "Ap_vs_chain3_released_z2.2_4.6" in r["descriptive_vs_fiducial"] and "Ap_vs_chain3_released_z2.2_4.6_upper68" in r["descriptive_vs_fiducial"]
    assert r["tests"]["tau0_amp"]["label_qualified"].startswith(r["tests"]["tau0_amp"]["label"]) and r["require_gate"] is True
    # exactly once
    with pytest.raises(SystemExit) as e:
        CONS.main(["--chain-dir", str(d), "--root", "real_eboss_z26", "--reference", REF_RELEASED, "--analysis-lock", _lock(tmp_path), "--out", str(out), "--chain-files", "plain"])
    assert e.value.code == 3


@pytest.mark.skipif(not os.path.exists(REF_RELEASED_FULL), reason="notes lane not available")
def test_real_released_reference_fullrange_primary_unblinded_mode(tmp_path):
    """Unblinded-mode readout of a synthetic full-range product against the released full-range chain as primary (the re-report
    reference): the one-sided A_P rule applies (upper limits) and the primary chain passes the schema check."""
    d = _product(tmp_path / "full", "real_eboss", Z13, blinded=True, ns=0.897, Ap=1.21e-9, tau0_amp=1.24, dtau0=-0.26, unblinded_files=True, z_kept=False, seed=7)
    out = tmp_path / "ro_full"
    CONS.main(["--chain-dir", str(d), "--root", "real_eboss", "--reference", REF_RELEASED_FULL, "--analysis-lock", _lock(tmp_path), "--out", str(out), "--chain-files", "unblinded"])
    r = json.load(open(str(out) + ".json"))
    assert r["primary_chain"] == "chain3_released_z2.2_4.6" and r["tests"]["Ap"]["note"].startswith("one-sided")
    assert r["tests"]["ns"]["label"] == "CONSISTENT" and abs(r["tests"]["ns"]["ref_central"] - 0.898) < 5e-4


@pytest.mark.skipif(not os.path.exists(REF_TABLE3), reason="notes lane not available")
def test_real_table3_reference_still_loads(tmp_path):
    d = _product(tmp_path / "full", "real_eboss", Z13, blinded=True, unblinded_files=True, z_kept=False, seed=9)
    out = tmp_path / "ro_t3"
    CONS.main(["--chain-dir", str(d), "--root", "real_eboss", "--reference", REF_TABLE3, "--analysis-lock", _lock(tmp_path), "--out", str(out)])
    r = json.load(open(str(out) + ".json"))
    assert r["primary_chain"] == "chain3_full_z2.2_4.6" and "ns_vs_fiducial_1.009" in r["descriptive_vs_fiducial"]


def test_plain_mode_implies_gate_and_schema_check_refuses(tmp_path):
    lock = _lock(tmp_path); ref = _ref_z26(tmp_path)
    d = _product(tmp_path / "red", "real_eboss_z26", Z13[2:], blinded=False, n_div=3)
    with pytest.raises(SystemExit) as e:                   # RED, no --require-gate typed: still refused in plain mode
        CONS.main(["--chain-dir", str(d), "--root", "real_eboss_z26", "--reference", ref, "--analysis-lock", lock, "--out", str(tmp_path / "x"), "--chain-files", "plain"])
    assert e.value.code == 3 and not os.path.exists(str(tmp_path / "x.json"))
    bad = json.load(open(ref)); del bad["chains"]["chain1"]["v_scale_h"]
    p = tmp_path / "bad.json"; p.write_text(json.dumps(bad))
    d2 = _product(tmp_path / "ok", "real_eboss_z26", Z13[2:], blinded=False)
    with pytest.raises(SystemExit) as e2:                  # schema check before any computation
        CONS.main(["--chain-dir", str(d2), "--root", "real_eboss_z26", "--reference", str(p), "--analysis-lock", lock, "--out", str(tmp_path / "y"), "--chain-files", "plain"])
    assert e2.value.code == 3 and not os.path.exists(str(tmp_path / "y.json"))


def test_zrange_compare_refuses_non_subset_and_carries_metal_rails(tmp_path):
    lock = _lock(tmp_path)
    full = _product(tmp_path / "full", "real_eboss", Z13[2:], blinded=True, unblinded_files=True)          # a full product with only 11 z
    sub = _product(tmp_path / "sub", "real_eboss_z26", Z13, blinded=False)                                  # a sub product with 13 z: not a subset
    with pytest.raises(SystemExit):
        ZR.main(["--full-dir", str(full), "--sub-dir", str(sub), "--analysis-lock", lock, "--out", str(tmp_path / "c")])
    full2 = _product(tmp_path / "full2", "real_eboss", Z13, blinded=True, unblinded_files=True, z_kept=False)
    sub2 = _product(tmp_path / "sub2", "real_eboss_z26", Z13[2:], blinded=False, seed=3)
    ZR.main(["--full-dir", str(full2), "--sub-dir", str(sub2), "--analysis-lock", lock, "--out", str(tmp_path / "c2")])
    r = json.load(open(str(tmp_path / "c2.json")))
    assert "rails" in r["comparison"]["nuis:f_SiIII_eBOSS_z0"] and r["comparison"]["ns"]["delta_over_full_sd"] is not None
    assert "delta_mean_sub_minus_full" in r["comparison"]["ns"]



def test_execution_record_integrity_and_box_limited_qualifier(tmp_path):
    """Reviewer L S1/M3: the readout refuses when its own sha256 or the reference's sha256 disagree with the execution record;
    a floor-piled A_P carries the BOX-LIMITED qualifier while the rule label stands."""
    import hashlib
    lock = _lock(tmp_path); ref = _ref_z26(tmp_path)
    me = hashlib.sha256(open(os.path.join(REPO, "scripts", "eboss_priya_consistency.py"), "rb").read()).hexdigest()
    rsha = hashlib.sha256(open(ref, "rb").read()).hexdigest()
    good = tmp_path / "rec_ok.json"; good.write_text(json.dumps(dict(records_sha256=dict(readout_script=me, reference_json_z26=rsha))))
    bad = tmp_path / "rec_bad.json"; bad.write_text(json.dumps(dict(records_sha256=dict(readout_script="0" * 64, reference_json_z26=rsha))))
    d = _product(tmp_path / "floor", "real_eboss_z26", Z13[2:], blinded=False, Ap=1.15e-9, ns=1.0, tau0_amp=1.09, dtau0=-0.02)   # clipped at the floor
    with pytest.raises(SystemExit):
        CONS.main(["--chain-dir", str(d), "--root", "real_eboss_z26", "--reference", ref, "--analysis-lock", lock, "--out", str(tmp_path / "x"), "--chain-files", "plain", "--execution-record", str(bad)])
    assert not os.path.exists(str(tmp_path / "x.json"))
    CONS.main(["--chain-dir", str(d), "--root", "real_eboss_z26", "--reference", ref, "--analysis-lock", lock, "--out", str(tmp_path / "y"), "--chain-files", "plain", "--execution-record", str(good)])
    r = json.load(open(str(tmp_path / "y.json")))
    assert r["integrity"]["reference_record_key"] == "reference_json_z26" and r["boundary_proximity_flag"]["Ap"] is True
    assert r["tests"]["Ap"]["box_limited"] is True and r["tests"]["Ap"]["label_qualified"].endswith("(BOX-LIMITED)")
    assert r["tests"]["Ap"]["label"] in ("SHIFTED", "DISCREPANT")


def test_zrange_compare_requires_readout_gate_and_reports_nested_scatter(tmp_path):
    lock = _lock(tmp_path)
    full = _product(tmp_path / "full", "real_eboss", Z13, blinded=True, unblinded_files=True, z_kept=False, seed=11)
    sub = _product(tmp_path / "sub", "real_eboss_z26", Z13[2:], blinded=False, seed=12, ns=0.92)
    red = tmp_path / "ro_red.json"; red.write_text(json.dumps(dict(health_gate=dict(label="RED"))))
    with pytest.raises(SystemExit):
        ZR.main(["--full-dir", str(full), "--sub-dir", str(sub), "--analysis-lock", lock, "--out", str(tmp_path / "c"), "--readout-json", str(red)])
    assert not os.path.exists(str(tmp_path / "c.json"))
    green = tmp_path / "ro_green.json"; green.write_text(json.dumps(dict(health_gate=dict(label="GREEN"))))
    ZR.main(["--full-dir", str(full), "--sub-dir", str(sub), "--analysis-lock", lock, "--out", str(tmp_path / "c2"), "--readout-json", str(green)])
    r = json.load(open(str(tmp_path / "c2.json")))
    assert r["readout_gate"] == "GREEN" and "delta_over_nested_scatter" in r["comparison"]["ns"] and "units" in r
