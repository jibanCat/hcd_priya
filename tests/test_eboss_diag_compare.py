"""PI #33: eboss_diag_compare.py on synthetic products (no science values). Checks root detection (unblinded export preferred),
the 2- and 3-parameter ladder inversions (exact on synthetic ladders), the ctau0 handling, shifts in baseline sd, the gate,
rails against each product's own box, and the exactly-once output refusal."""
import importlib.util
import json
import os

import numpy as np
import pytest

KIM = lambda z: 2.3e-3 * (1 + np.asarray(z)) ** 3.65


def _load():
    sp = importlib.util.spec_from_file_location("edc", "/home/mfho/hcd_priya/scripts/eboss_diag_compare.py")
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


def _write_product(d, root, z, amp, dt, c=None, ns_mean=0.9, tau0_box=None, unblinded=False, n=400, seed=0, with_fres=True):
    rng = np.random.default_rng(seed); os.makedirs(d, exist_ok=True)
    names = ["ns", "Ap", "herei", "heref", "alphaq", "hub", "omegamh2", "hireionz", "bhfeedback"] + [f"tau0_z{i}" for i in range(len(z))] + ["alpha_lls", "alpha_subdla", "alpha_dla"]
    ns = rng.normal(ns_mean, 0.01, n); Ap = rng.uniform(1.2e-9, 1.3e-9, n)
    a = rng.normal(amp, 0.01, n); s = rng.normal(dt, 0.02, n); cc = rng.normal(c, 0.1, n) if c is not None else np.zeros(n)
    x = np.log((1 + z) / 4.0); lad = a[:, None] * np.exp(s[:, None] * x[None, :] + cc[:, None] * (x ** 2)[None, :]) * KIM(z)[None, :]
    other = np.column_stack([rng.uniform(3.5, 4.1, n), rng.uniform(2.6, 3.2, n), rng.uniform(1.3, 2.5, n), rng.normal(0.7, 0.01, n), rng.normal(0.143, 0.001, n), rng.uniform(6.5, 8, n), rng.uniform(0.03, 0.07, n)])
    hcd = np.column_stack([rng.normal(0.2, 0.04, n), rng.normal(0.09, 0.02, n), rng.normal(0.005, 0.001, n)])
    X = np.column_stack([ns, Ap, other, lad, hcd]); rows = np.column_stack([np.ones(n), rng.normal(500, 3, n), X])
    croot = root + (".unblinded" if unblinded else "")
    for ci in range(2):
        np.savetxt(f"{d}/{croot}.{ci+1}.txt", rows[ci * (n // 2):(ci + 1) * (n // 2)])
    open(f"{d}/{croot}.paramnames", "w").write("\n".join(f"{nm}\t{nm}" for nm in names) + "\n")
    if unblinded:   # a blinded twin with shifted ns that must NOT be picked
        np.savetxt(f"{d}/{root}.1.txt", np.column_stack([rows[:, :2], rows[:, 2] + 0.1, rows[:, 3:]])); open(f"{d}/{root}.paramnames", "w").write("\n".join(f"{nm}\t{nm}" for nm in names) + "\n")
    nz = {}
    if with_fres: nz.update(f_res_amp=rng.normal(-0.04, 0.005, n).reshape(2, -1), f_res_slope=rng.normal(0, 0.3, n).reshape(2, -1))
    if c is not None: nz["ctau0"] = cc.reshape(2, -1)
    nz["f_SiIII_eBOSS_z0"] = np.exp(rng.uniform(np.log(0.003), np.log(0.03), n)).reshape(2, -1)
    np.savez(f"{d}/{root}.nuisance.npz", **nz)
    h = dict(rhat_max=1.002, ess_bulk_min=900.0, ess_tail_min=800.0, ebfmi_min=0.9, n_divergent=0, per_chain_div=[0, 0], treedepth_sat_frac=0.0, z_kept=z.tolist(), target_accept=0.95, seed=1)
    if tau0_box or c is not None:
        h["diagnostic"] = dict(tag="T", requested={}, applied=({"tau0_amp_range": list(tau0_box)} if tau0_box else {}) | ({"mf_curvature_sigma": 2.0} if c is not None else {}))
    json.dump(h, open(f"{d}/{root}.health.json", "w"))


def test_compare_synthetic(tmp_path):
    M = _load(); z = np.array([2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6])
    B = tmp_path / "base"; D = tmp_path / "diag"
    _write_product(str(B), "real_eboss_z26", z, 1.2, -0.1, ns_mean=0.96, unblinded=True, seed=1)
    _write_product(str(D), "real_eboss_z26_diag_T", z, 1.3, -0.1, c=0.5, ns_mean=0.99, tau0_box=(0.75, 1.5), seed=2, with_fres=False)
    PB = M.load_product(str(B)); PD = M.load_product(str(D))
    assert PB["root"] == "real_eboss_z26.unblinded" and abs(PB["cols"]["ns"].mean() - 0.96) < 0.01      # the unblinded export, not the blinded twin
    assert PB["ladder_fit"]["resid_2par"] < 1e-8 and PD["ladder_fit"]["resid_3par"] < 1e-8 and PD["ladder_fit"]["resid_2par"] > 1e-4   # curvature breaks the 2-par inversion
    assert np.allclose(PD["cols"]["ctau0_3par"], PD["cols"]["ctau0"], atol=1e-8) and abs(PD["cols"]["tau0_amp_3par"].mean() - 1.3) < 0.01
    assert np.array_equal(PD["cols"]["tau0_amp"], PD["cols"]["tau0_amp_3par"]) and "tau0_amp_2par" in PD["cols"]   # M4b: sampled amplitude = 3-par when ctau0 exported
    assert np.allclose(M.teff_curve(PD, PD["z"])["mean"], PD["ladder"].mean(0)) and np.allclose(M.teff_curve(PB, PB["z"])["mean"], PB["ladder"].mean(0))   # M1: curves from the ladder, no double counting
    assert PD["box"]["tau0_amp"] == (0.75, 1.5) and PB["box"]["tau0_amp"] == (0.75, 1.25)
    R = M.compare(PD, PB)
    assert R["gate_diag"]["label"] == "GREEN" and abs(R["shifts"]["ns"]["shift"] - 0.03) < 0.005 and R["shifts"]["ns"]["shift_over_baseline_sd"] > 2
    assert "f_res_amp" in R["only_in_baseline"] and "ctau0" in R["only_in_diag"] and "tau0_amp" in R["shifts"]
    assert R["rails"]["diag"]["tau0_amp"]["box"] == [0.75, 1.5] and R["rails"]["baseline"]["tau0_amp"]["box"] == [0.75, 1.25]
    assert R["rails"]["diag_against_baseline_box"]["tau0_amp"]["box"] == [0.75, 1.25] and R["rails"]["diag_against_baseline_box"]["tau0_amp"]["near_hi"] > 0.9   # M4a: amp 1.3 draws sit above the baseline cap
    assert len(R["teff"]["z"]) == 11 and R["teff"]["diag"]["mean"][4] > R["teff"]["baseline"]["mean"][4]   # pivot: amp 1.3 vs 1.2
    assert R["emulator_band"]["diag"]["frac_any_z_outside"] > 0.5 and R["emulator_band"]["baseline"]["frac_any_z_outside"] < 0.6   # amp 1.3 with curvature leaves [0.75, 1.25]; amp 1.2 mostly inside
    out = tmp_path / "cmp" / "T"
    assert M.main(["--diag-dir", str(D), "--baseline-dir", str(B), "--out", str(out)]) == 0
    assert (tmp_path / "cmp" / "T.json").exists() and "Shifts" in open(str(out) + ".md").read()
    with pytest.raises(SystemExit):
        M.main(["--diag-dir", str(D), "--baseline-dir", str(B), "--out", str(out)])                       # exactly once


def test_gate_red_on_divergence():
    M = _load()
    assert M.gate(dict(rhat_max=1.0, ess_bulk_min=900, ess_tail_min=900, ebfmi_min=0.9, n_divergent=1, treedepth_sat_frac=0))["label"] == "RED"
    assert M.gate(dict(rhat_max=1.02, ess_bulk_min=900, ess_tail_min=900, ebfmi_min=0.9, n_divergent=0, treedepth_sat_frac=0))["reasons"] == ["rhat"]
    assert M.gate({})["label"] == "UNKNOWN"
