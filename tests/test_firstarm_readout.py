"""First-arm readout tooling (A3c gate G1; serves A2c unchanged): TDD suite.

Two modules, one authority per input:
  * scripts/analyze_firstarm_selfdraw.py - census + self-drawn-truth conjuncts + the health
    block, computed from the pkls (per-leg registry: KS = 2 f_res sites ranks-only; DESI =
    8 metal nodes + 2 f_res with the f_res_amp scatter-ratio threshold). REFUSES on any
    conjunct failure; a health-block failure is EVIDENCE (recorded, feeds row 2), never a
    refusal.
  * scripts/firstarm_disposition.py - the pre-registered first-arm rows 1-6 from the
    deployed analyzer's gate JSON leg entry + the selfdraw JSON. Refuses partial or
    inconsistent inputs; cross-checks the health ranks against the gate JSON's
    repaired_sectors where present.

Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_firstarm_readout.py -q
"""
import importlib.util
import json
import os
import pickle

import numpy as np
import pytest

REPO = "/home/mfho/hcd_priya"


def _load_script(name):
    path = os.path.join(REPO, "scripts", name)
    spec = importlib.util.spec_from_file_location(name[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def SD():
    return _load_script("analyze_firstarm_selfdraw.py")


@pytest.fixture(scope="module")
def DP():
    return _load_script("firstarm_disposition.py")


# ---------------------------------- selfdraw fixtures -----------------------------------------
L_DRAWS = 60


def _ks_mock(m, SD, drawn=True, cfg_over=None, nsd=None):
    """CALIBRATED fixture (G2 review finding 5): the truth is a prior draw and the posterior
    draws are iid from the SAME law independent of the truth, so the rank of the truth among
    the draws is exactly uniform -- a healthy arm by construction."""
    rng = np.random.default_rng(100 + m)
    se = {}
    for i, (k, sd) in enumerate((("f_res_amp", 0.15), ("f_res_slope", 0.5))):
        tr = float(np.random.default_rng(7 + 1000 * i + m).normal(0.0, sd)) if drawn else float("nan")
        se[k] = dict(draws=rng.normal(0.0, sd, L_DRAWS), truth=tr)
    rec = dict(sites_extra=se, L=100, n_div=0,
               run_cfg=dict(SD.REGISTRY["KS"]["frozen_cfg"], **(cfg_over or {})),
               truth_site_semantics=dict(not_self_drawn=[] if nsd is None else nsd))
    return rec


def _write_ks_arm(dirpath, SD, n=48, **kw):
    os.makedirs(dirpath, exist_ok=True)
    for m in range(n):
        with open(os.path.join(dirpath, f"mock_{m:04d}.pkl"), "wb") as f:
            pickle.dump(_ks_mock(m, SD, **kw), f)


def test_selfdraw_happy_ks(SD, tmp_path):
    out = str(tmp_path / "arm")
    _write_ks_arm(out, SD)
    r = SD.run(out, 48, "KS")
    assert r["conjuncts_ok"] is True
    assert set(r["health"]) == {"f_res_amp", "f_res_slope"}
    for h in r["health"].values():
        assert set(h) >= {"rank_ks_p", "healthy", "scatter", "scatter_expected"}
    # calibrated-by-construction fixture: the health block must POSITIVELY pass (G2 finding 5)
    assert r["health_ok"] is True
    assert r["survey"] == "KS"
    assert r["n"] == 48


def test_selfdraw_refuses_census(SD, tmp_path):
    out = str(tmp_path / "arm")
    _write_ks_arm(out, SD)
    os.remove(os.path.join(out, "mock_0031.pkl"))
    with pytest.raises(SD.FirstArmError, match="census"):
        SD.run(out, 48, "KS")


def test_selfdraw_refuses_pinned_truth(SD, tmp_path):
    out = str(tmp_path / "arm")
    _write_ks_arm(out, SD)
    p = os.path.join(out, "mock_0005.pkl")
    rec = pickle.load(open(p, "rb"))
    rec["sites_extra"]["f_res_amp"]["truth"] = float("nan")
    pickle.dump(rec, open(p, "wb"))
    with pytest.raises(SD.FirstArmError, match="finite"):
        SD.run(out, 48, "KS")


def test_selfdraw_refuses_repeated_truth(SD, tmp_path):
    out = str(tmp_path / "arm")
    _write_ks_arm(out, SD)
    p5, p6 = (os.path.join(out, f"mock_{m:04d}.pkl") for m in (5, 6))
    r5, r6 = pickle.load(open(p5, "rb")), pickle.load(open(p6, "rb"))
    r6["sites_extra"]["f_res_slope"]["truth"] = r5["sites_extra"]["f_res_slope"]["truth"]
    pickle.dump(r6, open(p6, "wb"))
    with pytest.raises(SD.FirstArmError, match="distinct"):
        SD.run(out, 48, "KS")


def test_selfdraw_refuses_lawbreak(SD, tmp_path):
    out = str(tmp_path / "arm")
    _write_ks_arm(out, SD)
    for m in range(48):
        p = os.path.join(out, f"mock_{m:04d}.pkl")
        rec = pickle.load(open(p, "rb"))
        rec["sites_extra"]["f_res_amp"]["truth"] = 0.9 + 0.001 * m     # ~6 sigma cluster
        pickle.dump(rec, open(p, "wb"))
    with pytest.raises(SD.FirstArmError, match="law"):
        SD.run(out, 48, "KS")


def test_selfdraw_refuses_not_self_drawn(SD, tmp_path):
    out = str(tmp_path / "arm")
    _write_ks_arm(out, SD)
    p = os.path.join(out, "mock_0002.pkl")
    rec = pickle.load(open(p, "rb"))
    rec["truth_site_semantics"] = dict(not_self_drawn=["f_res_amp"])
    pickle.dump(rec, open(p, "wb"))
    with pytest.raises(SD.FirstArmError, match="not_self_drawn"):
        SD.run(out, 48, "KS")


def test_selfdraw_refuses_wrong_cfg(SD, tmp_path):
    out = str(tmp_path / "arm")
    _write_ks_arm(out, SD, cfg_over=dict(seed=20260724))       # the A3d seed: wrong population
    with pytest.raises(SD.FirstArmError, match="run_cfg"):
        SD.run(out, 48, "KS")


def test_selfdraw_health_failure_is_evidence_not_refusal(SD, tmp_path):
    """Centre-heavy ranks (posterior far wider than the truth spread) must NOT refuse: they
    set healthy=False on that site and health_ok=False -- row-2 evidence for the disposition."""
    out = str(tmp_path / "arm")
    os.makedirs(out, exist_ok=True)
    for m in range(48):
        rec = _ks_mock(m, SD)
        tr = rec["sites_extra"]["f_res_amp"]["truth"]
        rec["sites_extra"]["f_res_amp"]["draws"] = \
            np.random.default_rng(m).normal(tr, 3.0, L_DRAWS)   # sd >> truth spread -> ranks ~0.5
        with open(os.path.join(out, f"mock_{m:04d}.pkl"), "wb") as f:
            pickle.dump(rec, f)
    r = SD.run(out, 48, "KS")
    assert r["conjuncts_ok"] is True
    assert r["health"]["f_res_amp"]["healthy"] is False
    assert r["health_ok"] is False


def test_selfdraw_desi_registry_shape(SD):
    reg = SD.REGISTRY["DESI"]
    assert len(reg["sites"]) == 10
    assert reg["scatter_site"] == "f_res_amp"
    lo, hi = reg["scatter_band"]
    assert lo == 0.5 and hi == 2.0
    kinds = {k: v[0] for k, v in reg["sites"].items()}
    assert sum(1 for v in kinds.values() if v == "loguniform") == 8
    assert kinds["f_res_amp"] == "normal" and kinds["f_res_slope"] == "normal"


# ---------------------------------- disposition fixtures --------------------------------------
def _gate_leg(ns=(0.05, 1.02, "UNIFORM"), Ap=(0.06, 1.05, "UNIFORM"),
              tau0=(-0.05, 1.01, "UNIFORM"), n=48, survey="KS", extra_sites=None):
    def ch(mean, std, verdict):
        return dict(pull=dict(mean=mean, std=std, sem=std / np.sqrt(n), n=n),
                    rank=dict(ks_p=(0.5 if verdict == "UNIFORM" else 1e-5),
                              verdict=verdict, n=n, n_outside_band=0, mean=0.5))
    legs = dict(
        n_mocks=n,
        gate_ns=("PASS" if abs(ns[0]) <= 0.30 and ns[1] <= 1.1 else "FAIL"),
        gate_Ap=("PASS" if abs(Ap[0]) <= 0.30 and Ap[1] <= 1.1 else "FAIL"),
        gate_rank_ns=ns[2], gate_rank_Ap=Ap[2],
        pulls={"ns": ch(*ns)["pull"], "Ap": ch(*Ap)["pull"], "tau0amp": ch(*tau0)["pull"]},
        rank_uniformity={"ns": ch(*ns)["rank"], "Ap": ch(*Ap)["rank"],
                         "tau0amp": ch(*tau0)["rank"]},
        prior=dict(survey=survey),
        repaired_sectors=dict(gated=False, sites=(extra_sites or {})),
    )
    return dict(legs={survey: legs})


def _sd_json(survey="KS", n=48, conj=True, health=None):
    health = health if health is not None else {
        "f_res_amp": dict(rank_ks_p=0.4, healthy=True, scatter=0.04, scatter_expected=0.005),
        "f_res_slope": dict(rank_ks_p=0.6, healthy=True, scatter=0.08, scatter_expected=0.1),
    }
    return dict(survey=survey, n=n, conjuncts_ok=conj, health=health,
                health_ok=all(h["healthy"] for h in health.values()))


def _run_dp(DP, tmp_path, gate, sd):
    g = tmp_path / "gate.json"; s = tmp_path / "sd.json"
    g.write_text(json.dumps(gate)); s.write_text(json.dumps(sd))
    return DP.run(str(g), str(s))


def test_disposition_row1(DP, tmp_path):
    out = _run_dp(DP, tmp_path, _gate_leg(), _sd_json())
    assert out["row"] == 1


def test_disposition_row2_health(DP, tmp_path):
    health = {
        "f_res_amp": dict(rank_ks_p=1e-5, healthy=False, scatter=0.04, scatter_expected=0.005),
        "f_res_slope": dict(rank_ks_p=0.6, healthy=True, scatter=0.08, scatter_expected=0.1),
    }
    out = _run_dp(DP, tmp_path, _gate_leg(), _sd_json(health=health))
    assert out["row"] == 2


def test_disposition_row3_gated_rank(DP, tmp_path):
    out = _run_dp(DP, tmp_path, _gate_leg(ns=(0.05, 1.02, "NON-UNIFORM")), _sd_json())
    assert out["row"] == 3


def test_disposition_row3_tau0_pull(DP, tmp_path):
    out = _run_dp(DP, tmp_path, _gate_leg(tau0=(-0.46, 1.05, "UNIFORM")), _sd_json())
    assert out["row"] == 3
    assert "tau0" in out["why"]


def test_disposition_row4_mean_fail(DP, tmp_path):
    out = _run_dp(DP, tmp_path, _gate_leg(Ap=(-0.41, 0.97, "UNIFORM")), _sd_json())
    assert out["row"] == 4


def test_disposition_row6_sd_only(DP, tmp_path):
    out = _run_dp(DP, tmp_path, _gate_leg(ns=(0.05, 1.15, "UNIFORM")), _sd_json())
    assert out["row"] == 6


def test_disposition_rank_beats_sd_only(DP, tmp_path):
    """3d.2: rank non-uniformity escalates to row 3 even when the same channel also carries an
    sd-only failure (the row-6-outranks-row-3 defect must not recur)."""
    out = _run_dp(DP, tmp_path, _gate_leg(ns=(0.05, 1.15, "NON-UNIFORM")), _sd_json())
    assert out["row"] == 3


def test_disposition_mean_fail_beats_rank(DP, tmp_path):
    out = _run_dp(DP, tmp_path,
                  _gate_leg(ns=(0.05, 1.02, "NON-UNIFORM"), Ap=(-0.41, 0.97, "UNIFORM")),
                  _sd_json())
    assert out["row"] == 4


def test_disposition_refuses_conjunct_failure(DP, tmp_path):
    with pytest.raises(DP.DispositionError, match="conjunct"):
        _run_dp(DP, tmp_path, _gate_leg(), _sd_json(conj=False))


def test_disposition_refuses_partial_arm(DP, tmp_path):
    with pytest.raises(DP.DispositionError, match="n_mocks"):
        _run_dp(DP, tmp_path, _gate_leg(n=37), _sd_json(n=48))


def test_disposition_refuses_survey_mismatch(DP, tmp_path):
    with pytest.raises(DP.DispositionError, match="survey"):
        _run_dp(DP, tmp_path, _gate_leg(survey="KS"), _sd_json(survey="DESI"))


def test_disposition_refuses_inconsistent_gate_string(DP, tmp_path):
    gate = _gate_leg(ns=(0.05, 1.02, "UNIFORM"))
    gate["legs"]["KS"]["gate_ns"] = "FAIL"          # contradicts the recomputed criteria
    with pytest.raises(DP.DispositionError, match="inconsistent"):
        _run_dp(DP, tmp_path, gate, _sd_json())


def test_disposition_crosschecks_health_ranks(DP, tmp_path):
    """When the gate JSON carries repaired_sectors for the same site, the two independently
    computed rank p-values must agree within tolerance, else refuse."""
    sites = {"f_res_amp": dict(rank_ks_p=0.9, rank_verdict="UNIFORM")}
    gate = _gate_leg(extra_sites=sites)
    sd = _sd_json()                                  # selfdraw says 0.4 for the same site
    with pytest.raises(DP.DispositionError, match="cross-check"):
        _run_dp(DP, tmp_path, gate, sd)


def test_disposition_crosscheck_agreement_passes(DP, tmp_path):
    """The agreement path (G2 finding 4): equal rank p-values on both sides -> no refusal."""
    sites = {"f_res_amp": dict(rank_ks_p=0.4, rank_verdict="UNIFORM")}
    out = _run_dp(DP, tmp_path, _gate_leg(extra_sites=sites), _sd_json())
    assert out["row"] == 1


def test_disposition_row3_tau0_rank_limb(DP, tmp_path):
    """The tau0 RANK limb alone (G2 finding 4: only the pull limb was tested)."""
    out = _run_dp(DP, tmp_path, _gate_leg(tau0=(-0.05, 1.01, "NON-UNIFORM")), _sd_json())
    assert out["row"] == 3
    assert "tau0" in out["why"]


def test_disposition_tau0_beats_sd_only(DP, tmp_path):
    """G2 MUST-FIX 1: sd-only failure + a fired tau0 limb must land row 3 (the escalation),
    NOT row 6 -- the row-6-outranks-row-3 defect class must not recur through the tau0 limb.
    Both reasons are reported."""
    out = _run_dp(DP, tmp_path,
                  _gate_leg(ns=(0.05, 1.15, "UNIFORM"), tau0=(-0.46, 1.05, "UNIFORM")),
                  _sd_json())
    assert out["row"] == 3
    assert "tau0" in out["why"] and "sd-only" in out["why"]


def test_disposition_refuses_unknown_rank_verdict(DP, tmp_path):
    """G2 finding 7b: a verdict outside {UNIFORM, NON-UNIFORM} refuses, never escalates."""
    gate = _gate_leg()
    gate["legs"]["KS"]["rank_uniformity"]["tau0amp"]["verdict"] = "underpowered(N<4)"
    with pytest.raises(DP.DispositionError, match="verdict"):
        _run_dp(DP, tmp_path, gate, _sd_json())


def test_disposition_refuses_channel_n_mismatch(DP, tmp_path):
    """G2 finding 7a: a channel whose pull n dropped below the census refuses."""
    gate = _gate_leg()
    gate["legs"]["KS"]["pulls"]["Ap"]["n"] = 47
    with pytest.raises(DP.DispositionError, match="degenerate"):
        _run_dp(DP, tmp_path, gate, _sd_json())


def test_disposition_exhaustive_sweep_vs_oracle(DP, tmp_path):
    """G2 finding 4: the full truth table (8 x 8 x 4 x 2 = 512 combos) against an
    independently coded oracle of the pre-registered precedence 4 > 3 > 6 > 2 > 1 with the
    tau0 limbs inside row 3."""
    def oracle(nsc, apc, t0, health):
        mean_fail = (not nsc[0]) or (not apc[0])
        rank_fail = (not nsc[2]) or (not apc[2])
        sd_only = (nsc[0] and not nsc[1]) or (apc[0] and not apc[1])
        tau0_esc = (not t0[0]) or (not t0[1])
        if mean_fail:
            return 4
        if rank_fail:
            return 3
        if tau0_esc:
            return 3
        if sd_only:
            return 6
        if not health:
            return 2
        return 1

    def ch_vals(flags):
        mean = 0.05 if flags[0] else 0.41
        std = 1.02 if flags[1] else 1.15
        verdict = "UNIFORM" if flags[2] else "NON-UNIFORM"
        return (mean, std, verdict)

    bad_health = {
        "f_res_amp": dict(rank_ks_p=1e-5, healthy=False, scatter=0.04, scatter_expected=0.005),
        "f_res_slope": dict(rank_ks_p=0.6, healthy=True, scatter=0.08, scatter_expected=0.1),
    }
    flags3 = [(a, b, c) for a in (True, False) for b in (True, False) for c in (True, False)]
    flags2 = [(a, b) for a in (True, False) for b in (True, False)]
    n_checked = 0
    for nsc in flags3:
        for apc in flags3:
            for t0 in flags2:
                tau0 = (0.05 if t0[1] else 0.46, 1.01, "UNIFORM" if t0[0] else "NON-UNIFORM")
                for health in (True, False):
                    gate = _gate_leg(ns=ch_vals(nsc), Ap=ch_vals(apc), tau0=tau0)
                    sd = _sd_json(health=None if health else bad_health)
                    out = _run_dp(DP, tmp_path, gate, sd)
                    want = oracle(nsc, apc, t0, health)
                    assert out["row"] == want, (nsc, apc, t0, health, out["row"], want)
                    n_checked += 1
    assert n_checked == 512
