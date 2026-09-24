"""Synthetic tests for scripts/desi_stage1.py v2 (no real pkls). Two fixture layers:
  * live-set records (dict m, X, t, L, params) with a HETEROGENEOUS population: per-mock covariance scale 0.5 to 2,
    per-mock rotation in the (ns, dtau0) plane unless `coherent`, L uniform in [75, 300] (the real census range),
    20 percent of mocks with truths within 1 sd of a prior bound; planted alternatives (orientation error, k-mediation
    at two strengths, rail-driven tails with draws truncated INSIDE the bounds);
  * full pkl-layout populations (48 + 48 files with sha manifests, the frozen 25-name list and the inventory
    sites_extra) so preflight() and run() exercise the FROZEN loaders end to end.
"""
import hashlib
import importlib.util
import json
import os
import pickle

import numpy as np
import pytest

REPO = "/home/mfho/hcd_priya"
S = importlib.util.spec_from_file_location("ds", os.path.join(REPO, "scripts", "desi_stage1.py"))
M = importlib.util.module_from_spec(S); S.loader.exec_module(M)
EX = M.EX
PARAMS = ("ns", "Ap", "tau0_amp", "dtau0", "alpha_lls", "k_SiIII_DESI_z1")
LO = np.array([0.0, 0.0, 0.75, -0.40, 0.0, -3.0]); HI = np.array([1.0, 1.0, 1.25, 0.25, 1.0, -1.0])


def _recs(rng, N=48, post_corr_nd=0.0, err_corr_nd=None, k_mediates=0.0, rail_tails=False, coherent=False, near_bound_frac=0.2, L_range=(75, 300), homogeneous=False):
    """err_corr_nd None = calibrated (truth errors follow the posterior); a number = realized ns-dtau0 error correlation."""
    recs = []
    for m in range(N):
        L = int(rng.integers(L_range[0], L_range[1] + 1)) if not homogeneous else 100
        scale = 1.0 if homogeneous else float(rng.uniform(0.5, 2.0))
        sd = scale * np.array([0.05, 0.08, 0.03, 0.06, 0.05, 0.25])
        C = np.eye(6); C[0, 3] = C[3, 0] = post_corr_nd
        if not coherent and not homogeneous:
            th = rng.uniform(0, 2 * np.pi); rot = np.eye(6); rot[0, 0] = np.cos(th); rot[0, 3] = -np.sin(th); rot[3, 0] = np.sin(th); rot[3, 3] = np.cos(th)
            C = rot @ C @ rot.T
        Sig = np.outer(sd, sd) * C; R = np.linalg.cholesky(Sig)
        if err_corr_nd is None:
            R_err = R
        else:
            Ce = np.eye(6); Ce[0, 3] = Ce[3, 0] = err_corr_nd; R_err = np.linalg.cholesky(np.outer(sd, sd) * Ce)
        mu = np.array([0.5, 0.5, 1.0, -0.05, 0.15, -1.6])
        if rng.uniform() < near_bound_frac:
            j = int(rng.integers(0, 6)); side = rng.choice([-1, 1]); mu[j] = (LO[j] + 0.8 * sd[j]) if side < 0 else (HI[j] - 0.8 * sd[j])
        eps = rng.standard_normal(6)
        if k_mediates > 0:
            eps[0] = k_mediates * eps[5] + np.sqrt(1 - k_mediates ** 2) * rng.standard_normal()
            eps[3] = k_mediates * eps[5] + np.sqrt(1 - k_mediates ** 2) * rng.standard_normal()
        if rail_tails and m < 8:
            mu[5] = HI[5] - 0.6 * sd[5]; eps[0] = 3.0 * np.sign(rng.standard_normal())
        t = mu + R_err @ eps
        X = mu + (R @ rng.standard_normal((6, L))).T
        X = np.clip(X, LO + 1e-6, HI - 1e-6); t = np.clip(t, LO + 1e-6, HI - 1e-6)     # draws and truths inside the prior box
        X[:, 5] = 10.0 ** X[:, 5]; t = t.copy(); t[5] = 10.0 ** t[5]                  # k-node in linear units
        recs.append(dict(m=m, X=X, t=t, L=L, params=PARAMS))
    return recs


def _wsa_from(recs_d, recs_e=None):
    def arm(recs):
        X3, t3 = zip(*[M.coord_matrix(r, M.P3) for r in recs])
        Z = np.stack([M.whitened_z(X, t) for X, t in zip(X3, t3)]); Sm = Z.T @ Z / len(Z)
        w, V = np.linalg.eigh(Sm); vmax = EX._sign_fix(V[:, -1]); vmin = EX._sign_fix(V[:, 0])
        return dict(directions=dict(v_max=vmax.tolist(), v_min=vmin.tolist()), descriptive=dict(proj_vmax=(Z @ vmax).tolist()), S3=Sm.tolist())
    out = dict(prereg=M.WSA_PREREG, seed_root=M.WSA_SEED_ROOT, DESI=arm(recs_d))
    out["eBOSS"] = arm(recs_e if recs_e is not None else recs_d)
    return out


def _rngs(seed=0):
    k = np.random.SeedSequence(seed).spawn(6)
    return dict(null_desi=np.random.default_rng(k[0]), null_eboss=np.random.default_rng(k[1]), boot=np.random.default_rng(k[2]), dirs=np.random.default_rng(k[3]), perm_desi=np.random.default_rng(k[4]), perm_eboss=np.random.default_rng(k[5]))


def _small():
    """Shrink the Monte-Carlo sizes for the fast tests; restore after."""
    saved = (M.B_BOOT, M.B_DIRS, M.B_PERM); M.B_BOOT, M.B_DIRS, M.B_PERM = 300, 300, 2000
    return saved


def _restore(saved):
    M.B_BOOT, M.B_DIRS, M.B_PERM = saved


# ------------------------------------------------------------------ unit checks
def test_loo_e_table_matches_direct_loo():
    rng = np.random.default_rng(1); X = rng.standard_normal((50, 3))
    tab = M.loo_e_table(X, [0, 2])
    for j in (0, 7, 49):
        Y = np.delete(X, j, axis=0); e = (Y.mean(axis=0) - X[j]) / Y.std(axis=0, ddof=1)
        np.testing.assert_allclose(tab[j], e[[0, 2]], rtol=1e-10)


def test_seed_streams_are_six_and_first_five_unchanged():
    k5 = np.random.SeedSequence(M.SEED_ROOT).spawn(5); k6 = np.random.SeedSequence(M.SEED_ROOT).spawn(6)
    assert all(a.entropy == b.entropy and a.spawn_key == b.spawn_key for a, b in zip(k5, k6[:5]))
    r = M._rngs(); assert set(r) == {"null_desi", "null_eboss", "boot", "dirs", "perm_desi", "perm_eboss"}


# ------------------------------------------------------------------ D1
def test_d1_degenerate_and_generic_labels():
    saved = _small()
    recs = _recs(np.random.default_rng(13), post_corr_nd=0.5, coherent=True, homogeneous=True)
    X3 = [M.coord_matrix(r, M.P3)[0] for r in recs]; w = _wsa_from(recs)
    d1 = M.d1_coherence(X3, w["DESI"]["directions"]["v_max"], w["DESI"]["directions"]["v_min"], np.random.default_rng(14))
    assert d1["label"].startswith("DEGENERATE")
    recs2 = _recs(np.random.default_rng(15), post_corr_nd=0.5, coherent=False)
    X3b = [M.coord_matrix(r, M.P3)[0] for r in recs2]; w2 = _wsa_from(recs2)
    d1b = M.d1_coherence(X3b, w2["DESI"]["directions"]["v_max"], w2["DESI"]["directions"]["v_min"], np.random.default_rng(16))
    assert not d1b["label"].startswith("DEGENERATE") and d1b["axis_ns"]["coherence"] > 0.95
    _restore(saved)


# ------------------------------------------------------------------ D2
def test_d2_fires_on_orientation_error_and_is_quiet_when_calibrated_even_with_skewed_rho():
    # a planted orientation error is a POPULATION-level statement (shared posterior orientation); random per-mock rotations wash it out
    recs = _recs(np.random.default_rng(4), post_corr_nd=0.5, err_corr_nd=0.0, coherent=True)
    X3, t3 = map(list, zip(*[M.coord_matrix(r, M.P3) for r in recs]))
    d2 = M.d2_orientation(X3, t3, EX._shared_index(recs, np.random.default_rng(9), 1500))
    assert d2["ns_dtau0"]["fires"] and d2["ns_dtau0"]["D_orient"] < -0.25
    # calibrated with a WIDE (symmetric) spread of posterior correlations from random rotations: must not fire systematically
    fires = 0
    for s in range(12):
        recs0 = _recs(np.random.default_rng(100 + s), post_corr_nd=0.6, err_corr_nd=None, coherent=False)
        X3, t3 = map(list, zip(*[M.coord_matrix(r, M.P3) for r in recs0]))
        fires += int(M.d2_orientation(X3, t3, EX._shared_index(recs0, np.random.default_rng(200 + s), 1000))["ns_dtau0"]["fires"])
    assert fires <= 3


# ------------------------------------------------------------------ D3
def test_d3_mediation_strong_and_projection_precondition():
    saved = _small()
    recs = _recs(np.random.default_rng(7), k_mediates=0.8, coherent=True)
    X4, t4 = map(list, zip(*[M.coord_matrix(r, M.C4) for r in recs]))
    d3 = M.d3_mediation(X4, t4, EX._shared_index(recs, np.random.default_rng(11), 1500), np.random.default_rng(3))
    assert d3["c_ns_dtau0"]["p"] < 0.05 and d3["c_ns_k"]["p"] < 0.05 and d3["label"] in ("MEDIATED", "JOINT, NOT SEPARABLE")
    assert "boot68" in d3["c_ns_k"] and len(d3["c_ns_k"]["boot68"]) == 2
    # calibrated population: PROJECTION must not be assigned when c(ns, k) does not fire
    recs0 = _recs(np.random.default_rng(21))
    X4, t4 = map(list, zip(*[M.coord_matrix(r, M.C4) for r in recs0]))
    d30 = M.d3_mediation(X4, t4, EX._shared_index(recs0, np.random.default_rng(22), 1500), np.random.default_rng(4))
    assert not (d30["label"] == "PROJECTION" and d30["c_ns_k"]["p"] >= 0.05)
    _restore(saved)


# ------------------------------------------------------------------ D4
def test_d4_rail_tails_fire_with_draws_inside_bounds_and_stale_projection_refuses():
    saved = _small()
    recs = _recs(np.random.default_rng(8), rail_tails=True); w = _wsa_from(recs)
    X4, t4 = map(list, zip(*[M.coord_matrix(r, M.C4) for r in recs]))
    for X in X4:
        assert np.all(X[:, 3] < M.BOUNDS["k_SiIII_DESI_z1"][1]) and np.all(X[:, 3] > M.BOUNDS["k_SiIII_DESI_z1"][0])
    J = EX._shared_index(recs, np.random.default_rng(12), 500)
    d4 = M.d4_tail_rail(X4, t4, w["DESI"]["directions"]["v_max"], np.random.default_rng(12), J=J, coords=M.C4, frozen_proj=w["DESI"]["descriptive"]["proj_vmax"])
    assert d4["fires"] and d4["n_tail"] >= 4 and d4["p_shared_index_null"] is not None and "width_confound" in d4
    with pytest.raises(M.Stage1Refusal):
        M.d4_tail_rail(X4, t4, w["DESI"]["directions"]["v_max"], np.random.default_rng(12), coords=M.C4, frozen_proj=[0.0] * len(recs))
    # eBOSS-style control on P3 only
    X3, t3 = map(list, zip(*[M.coord_matrix(r, M.P3) for r in recs]))
    d4e = M.d4_tail_rail(X3, t3, w["DESI"]["directions"]["v_max"], np.random.default_rng(12), coords=M.P3)
    assert d4e["coords"] == list(M.P3)
    _restore(saved)


# ------------------------------------------------------------------ decision routing
def _d2(f): return {"ns_dtau0": {"fires": f}}
def _d3(label): return {"label": label}
def _d4(f): return {"fires": f}


@pytest.mark.parametrize("desi,eboss,lab,rail,expect", [
    (True, False, "UNRESOLVED", True, "RAIL-ASSOCIATED TAILS"),
    (False, False, "MEDIATED", True, "RAIL-ASSOCIATED AND k-SUFFICIENT (both fire)"),
    (False, False, "MEDIATED", False, "k-SUFFICIENT (MEDIATED BY THE k-NODE)"),
    (True, False, "JOINT, NOT SEPARABLE", False, "JOINT DIRECTION SPANNING ns, dtau0 AND k; NOT SEPARABLE"),
    (True, True, "UNRESOLVED", False, "ORIENTATION PATTERN SHARED BY THE eBOSS CONTROL (not DESI-specific)"),
    (True, False, "PROJECTION", False, "ORIENTATION ERROR WITH THE k-NODE ASSOCIATION PROJECTED ONTO THE MEAN-FLUX STRUCTURE"),
    (True, False, "UNRESOLVED", False, "ORIENTATION ERROR, NO RAIL, NO MEDIATION"),
    (True, False, "INDEPENDENT", False, "ORIENTATION ERROR, NO RAIL, NO MEDIATION"),
    (False, False, "UNRESOLVED", False, "NOT FURTHER LOCALIZABLE WITH STORED DRAWS"),
    (False, False, "PROJECTION", False, "NOT FURTHER LOCALIZABLE WITH STORED DRAWS"),
    (False, False, "JOINT, NOT SEPARABLE", True, "RAIL-ASSOCIATED TAILS"),
    (True, True, "MEDIATED", False, "k-SUFFICIENT (MEDIATED BY THE k-NODE)"),
])
def test_decide_routes_every_combination(desi, eboss, lab, rail, expect):
    dec = M.decide(_d2(desi), _d2(eboss), _d3(lab), _d4(rail))
    assert dec["outcome"] == expect
    assert (dec["stage2"].startswith("TRIGGERED")) == (expect == "ORIENTATION ERROR, NO RAIL, NO MEDIATION")


# ------------------------------------------------------------------ full pkl layout: preflight + run through the frozen loaders
def _write_population(d, arm, recs_like, rng):
    """Write 48 pkls in the frozen layout from live-set records (k-node etc. placed into sites_extra)."""
    os.makedirs(d, exist_ok=True)
    names = list(EX.MAIN_NAMES_25); se_keys = EX.expected_sites_extra(arm)
    lines = []
    for r in recs_like:
        L = r["L"]; draws = rng.uniform(0.2, 0.8, size=(L, 25)); truth = rng.uniform(0.3, 0.7, size=25)
        X = r["X"]; t = r["t"]
        draws[:, names.index("ns")] = X[:, 0]; truth[names.index("ns")] = t[0]
        draws[:, names.index("Ap")] = X[:, 1]; truth[names.index("Ap")] = t[1]
        draws[:, names.index("alpha_lls")] = X[:, 4]; truth[names.index("alpha_lls")] = t[4]
        se = {}
        for k in se_keys:
            col = rng.uniform(0.2, 0.8, size=L) * (0.01 if k.startswith(("f_", "k_")) else 1.0); tv = float(col.mean())
            if k == "tau0_amp": col = X[:, 2]; tv = float(t[2])
            if k == "dtau0": col = X[:, 3]; tv = float(t[3])
            if k == "k_SiIII_DESI_z1": col = X[:, 5]; tv = float(t[5])
            se[k] = dict(draws=col, truth=tv)
        p = os.path.join(d, f"mock_{r['m']:04d}.pkl")
        with open(p, "wb") as f:
            pickle.dump(dict(names=names, draws=draws, truth_vec=truth, L=L, sites_extra=se), f)
        lines.append(f"{hashlib.sha256(open(p, 'rb').read()).hexdigest()}  mock_{r['m']:04d}.pkl")
    sha = os.path.join(d, "..", f"{arm}_sha256.txt")
    with open(sha, "w") as f:
        f.write("\n".join(lines) + "\n")
    return d, sha


def test_preflight_and_run_on_frozen_layout(tmp_path):
    saved = _small()
    rng = np.random.default_rng(31)
    recs_d = _recs(np.random.default_rng(32)); recs_e = _recs(np.random.default_rng(33))
    dd, sd = _write_population(str(tmp_path / "desi" / "pop"), "DESI", recs_d, rng)
    de, se = _write_population(str(tmp_path / "eboss" / "pop"), "eBOSS", recs_e, rng)
    # the frozen loaders: rebuild the live-set records and derive the WS-A fixture from THOSE (bit-identical draws)
    rd = EX.load_arm_ext(dd, sd, "DESI"); re_ = EX.load_arm_ext(de, se, "eBOSS")
    w = _wsa_from(rd, re_); wp = tmp_path / "wsa.json"; wp.write_text(json.dumps(w))
    pf = M.preflight(dd, sd, de, se, str(wp))
    assert pf["v_max_reproduced"] and pf["projections_reproduced"] and pf["n_desi"] == 48 and len(pf["DESI"]["L"]) == 48
    out = tmp_path / "res.json"
    M.B_NULL_SAVE = M.B_NULL; M.B_NULL = 300
    res = M.run(dd, sd, de, se, str(wp), str(out))
    res2_path = tmp_path / "res2.json"; res2 = M.run(dd, sd, de, se, str(wp), str(res2_path))
    M.B_NULL = M.B_NULL_SAVE
    assert out.read_text() == res2_path.read_text()          # determinism across two runs
    j = json.load(open(out))
    for k in ("D1", "D2_DESI", "D2_eBOSS_control", "D3", "D4_DESI", "D4_eBOSS_control", "decision", "frozen_reference", "mock_ids_DESI", "L_eBOSS", "seed_streams", "preflight"):
        assert k in j
    assert j["mock_ids_DESI"] == list(range(48)) and len(j["seed_streams"]) == 6
    with pytest.raises(M.Stage1Refusal):
        M.run(dd, sd, de, se, str(wp), str(out))               # exactly-once
    # stale frozen projections -> preflight refuses BEFORE any output
    w2 = json.loads(wp.read_text()); w2["DESI"]["descriptive"]["proj_vmax"] = [0.0] * 48; (tmp_path / "wsa2.json").write_text(json.dumps(w2))
    with pytest.raises(M.Stage1Refusal):
        M.preflight(dd, sd, de, se, str(tmp_path / "wsa2.json"))
    w3 = json.loads(wp.read_text()); w3["seed_root"] = 1; (tmp_path / "wsa3.json").write_text(json.dumps(w3))
    with pytest.raises(M.Stage1Refusal):
        M.preflight(dd, sd, de, se, str(tmp_path / "wsa3.json"))
    _restore(saved)


# ------------------------------------------------------------------ operating characteristics (section 3; env-gated)
@pytest.mark.skipif(os.environ.get("DESI_STAGE1_OC") != "1", reason="operating-characteristics study; set DESI_STAGE1_OC=1")
def test_operating_characteristics():
    from scipy.stats import kstest
    Mpop = int(os.environ.get("DESI_STAGE1_OC_M", "100")); M.B_PERM = 2000; M.B_BOOT = 200; M.B_DIRS = 200
    ps = {k: [] for k in ("d2", "pc_nd_k", "pc_nk_mf", "d4", "d4_shared")}; fires = {k: 0 for k in ("orient", "med_strong", "med_half", "rails", "joint_strong", "med_common")}
    for m in range(Mpop):
        # NULL: calibrated heterogeneous population with skewed posterior correlations
        recs = _recs(np.random.default_rng(1000 + m), post_corr_nd=0.4, err_corr_nd=None); w = _wsa_from(recs)
        X3, t3 = map(list, zip(*[M.coord_matrix(r, M.P3) for r in recs])); X4, t4 = map(list, zip(*[M.coord_matrix(r, M.C4) for r in recs]))
        J = EX._shared_index(recs, np.random.default_rng(m), 2000)
        ps["d2"].append(M.d2_orientation(X3, t3, J)["ns_dtau0"]["p_two_sided"])
        d3 = M.d3_mediation(X4, t4, J, np.random.default_rng(m)); ps["pc_nd_k"].append(d3["pc_ns_dtau0_given_k"]["p"]); ps["pc_nk_mf"].append(d3["pc_ns_k_given_dtau0_tau0amp"]["p"])
        d4 = M.d4_tail_rail(X4, t4, w["DESI"]["directions"]["v_max"], np.random.default_rng(m), J=J[:500], coords=M.C4, frozen_proj=w["DESI"]["descriptive"]["proj_vmax"])
        ps["d4"].append(d4["p_one_sided_negative"]); ps["d4_shared"].append(d4["p_shared_index_null"])
        # ALTERNATIVES
        recs = _recs(np.random.default_rng(2000 + m), post_corr_nd=0.4, err_corr_nd=0.0, coherent=True); X3, t3 = map(list, zip(*[M.coord_matrix(r, M.P3) for r in recs]))
        fires["orient"] += int(M.d2_orientation(X3, t3, EX._shared_index(recs, np.random.default_rng(m), 2000))["ns_dtau0"]["fires"])
        for key, strength in (("med_strong", 0.8), ("med_half", 0.55)):
            recs = _recs(np.random.default_rng(3000 + m), k_mediates=strength, coherent=True); X4, t4 = map(list, zip(*[M.coord_matrix(r, M.C4) for r in recs]))
            lab = M.d3_mediation(X4, t4, EX._shared_index(recs, np.random.default_rng(m), 2000), np.random.default_rng(m))["label"]
            fires[key] += int(lab == "MEDIATED")
            if key == "med_strong": fires["joint_strong"] += int(lab == "JOINT, NOT SEPARABLE")
        recs = _recs(np.random.default_rng(4000 + m), rail_tails=True); w = _wsa_from(recs); X4, t4 = map(list, zip(*[M.coord_matrix(r, M.C4) for r in recs]))
        fires["rails"] += int(M.d4_tail_rail(X4, t4, w["DESI"]["directions"]["v_max"], np.random.default_rng(m), coords=M.C4, frozen_proj=w["DESI"]["descriptive"]["proj_vmax"])["fires"])
    rep = dict(M=Mpop, ks_p={k: float(kstest(v, "uniform").pvalue) for k, v in ps.items()}, size={k: float(np.mean(np.asarray(v) < 0.05)) for k, v in ps.items()},
               power={k: v / Mpop for k, v in fires.items()}, fixture="null: heterogeneous (scale 0.5-2, random per-mock (ns,dtau0) rotation, L in [75,300], 20 percent near-bound truths, draws inside bounds); alternatives: same but with a SHARED posterior orientation (population-level structures)")
    json.dump(rep, open("/tmp/desi_stage1_oc.json", "w"), indent=1); print(rep)
    assert all(v > 0.01 / 5 for v in rep["ks_p"].values()), rep
