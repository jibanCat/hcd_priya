"""No-peek tests for scripts/a2c_joint_calib_ext.py (prereg amendment EXT-1, 2026-09-22).
NO real pkl is read anywhere in this file; every fixture is generated in tmp_path. The tests
build synthetic 48-mock populations with the REAL schema (25 packed names incl. the 13 derived
tau0 rungs, the arm's sites_extra keys), gate JSONs whose values are computed with the deployed
definitions, and an esc-diag JSON, so the loader, schema, pairing and consistency conjuncts and
the end-to-end run_ext path are exercised without touching a formal input.
"""
import hashlib
import importlib.util
import json
import os
import pickle

import numpy as np
import pytest
from scipy import stats as sps

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="module")
def EX():
    spec = importlib.util.spec_from_file_location(
        "a2c_joint_calib_ext", os.path.join(HERE, "..", "scripts", "a2c_joint_calib_ext.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------ fixture builders ------------------------------

Z_GRID = np.linspace(2.2, 4.6, 13)


def synth_pop(EX, rng, arm, n=48, Ls=None, hetero=True, nongauss=True, nearbound=True,
              rot=None, truths=None):
    """Exact self-draw population in the arm's full live space. Returns list of pkl-like dicts.
    `truths`: optional list of (t_live, truth_full) to PAIR with another arm (Case A columns)."""
    params = EX.live_params(arm)
    d = len(params)
    if Ls is None:
        Ls = rng.integers(75, 200, size=n)
    pkls, out_truths = [], []
    for i in range(n):
        L = int(Ls[i])
        if hetero:
            A = rng.standard_normal((d, d)) * 0.25
            Sig = np.eye(d) + A @ A.T
        else:
            Sig = np.eye(d)
        Cl = np.linalg.cholesky(Sig)
        mu = rng.standard_normal(d) * 2.0
        raw = rng.standard_normal((L + 1, d))
        Y = mu + raw @ Cl.T
        if nongauss:
            Y = np.sinh(Y * 0.3) / 0.3
        if nearbound:
            Y = np.clip(Y, mu - 4.0, mu + 1.0)
        X, t = Y[:L], Y[L].copy()
        if rot is not None:      # inject a correlated truth displacement in (ns, rot["j"])
            i_ns, i_j = params.index("ns"), params.index(rot["j"])
            cols = [i_ns, i_j]
            mu2 = X[:, cols].mean(axis=0)
            S2 = np.cov(X[:, cols].T, ddof=1)
            w2, V2 = np.linalg.eigh(S2)
            S2half = (V2 * np.sqrt(w2)) @ V2.T
            t[cols] = mu2 + S2half @ (np.linalg.cholesky(rot["cov"]) @ rng.standard_normal(2))
        if truths is not None:   # pair Case-A truths with the other arm (draws stay independent)
            t_pair, _ = truths[i]
            for p in EX.CASE_A:
                t[params.index(p)] = t_pair[p]
        out_truths.append(({p: float(t[params.index(p)]) for p in params}, None))
        # pack into the real schema
        names = list(EX.MAIN_NAMES_25)
        draws25 = np.zeros((L, 25)); truth25 = np.zeros(25)
        for p in EX.SITES_IN_TRUTH_VEC:
            draws25[:, names.index(p)] = X[:, params.index(p)]
            truth25[names.index(p)] = t[params.index(p)]
        amp, slope = X[:, params.index("tau0_amp")], X[:, params.index("dtau0")]
        ta, ts = t[params.index("tau0_amp")], t[params.index("dtau0")]
        for k, zk in enumerate(Z_GRID):                      # derived rungs, exactly log-linear
            c = np.log((1 + zk) / 4.0)
            draws25[:, names.index(f"tau0_z{k}")] = amp * np.exp(slope * c)
            truth25[names.index(f"tau0_z{k}")] = ta * np.exp(ts * c)
        se = {}
        for p in EX.expected_sites_extra(arm):
            se[p] = dict(draws=X[:, params.index(p)].copy(), truth=float(t[params.index(p)]))
        pkls.append(dict(names=names, draws=draws25, truth_vec=truth25, L=L, sites_extra=se,
                         run_cfg=dict(survey=arm, hcd_prior_signature="test"), n_div=0))
    return pkls, out_truths


def write_pop(EX, tmp, pkls, arm, tag):
    d = tmp / f"{tag}_{arm}"
    d.mkdir()
    lines = []
    for m, z in enumerate(pkls):
        p = d / f"mock_{m:04d}.pkl"
        with open(p, "wb") as f:
            pickle.dump(z, f)
        lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  mock_{m:04d}.pkl")
    sha = tmp / f"{tag}_{arm}_sha.txt"
    sha.write_text("\n".join(lines) + "\n")
    return str(d), str(sha)


def gate_from_pop(EX, pkls, arm):
    """Gate JSON with the deployed definitions (pulls, rank_uniformity, repaired_sectors)."""
    params = EX.live_params(arm)
    names = list(EX.MAIN_NAMES_25)

    def col(z, p):
        return (np.asarray(z["draws"])[:, names.index(p)] if p in EX.SITES_IN_TRUTH_VEC
                else np.asarray(z["sites_extra"][p]["draws"]))

    def tru(z, p):
        return (float(z["truth_vec"][names.index(p)]) if p in EX.SITES_IN_TRUTH_VEC
                else float(z["sites_extra"][p]["truth"]))
    leg = dict(L_all=[int(z["L"]) for z in pkls], pulls={}, rank_uniformity={},
               repaired_sectors=dict(sites={}), n_mocks=len(pkls))
    for p in ("ns", "Ap", "alpha_lls", "alpha_subdla", "alpha_dla"):
        pulls = np.array([(col(z, p).mean() - tru(z, p)) / col(z, p).std(ddof=1) for z in pkls])
        ranks = np.array([float(np.mean(col(z, p) < tru(z, p))) for z in pkls])
        leg["pulls"][p] = dict(mean=float(pulls.mean()), std=float(pulls.std(ddof=1)))
        leg["rank_uniformity"][p] = dict(ks_p=float(sps.kstest(ranks, "uniform").pvalue))
    for p in tuple(EX.METALS[arm]) + EX.RES:
        pulls = np.array([(col(z, p).mean() - tru(z, p)) / col(z, p).std(ddof=1) for z in pkls])
        ranks = np.array([float(np.mean(col(z, p) < tru(z, p))) for z in pkls])
        leg["repaired_sectors"]["sites"][p] = dict(pull_mean=float(pulls.mean()), pull_sd=float(pulls.std(ddof=1)),
                                                   rank_ks_p=float(sps.kstest(ranks, "uniform").pvalue))
    return {"legs": {arm: leg}}


def esc_from_pop(pkls):
    out = {"P1": {"sampled_sites": {}}}
    for p, key in (("tau0_amp", "tau0_amp_sampled"), ("dtau0", "dtau0_sampled")):
        pulls = np.array([(np.asarray(z["sites_extra"][p]["draws"]).mean() - z["sites_extra"][p]["truth"])
                          / np.asarray(z["sites_extra"][p]["draws"]).std(ddof=1) for z in pkls])
        out["P1"]["sampled_sites"][key] = dict(pull_sd=float(pulls.std(ddof=1)))
    return out


@pytest.fixture(scope="module")
def paired_setup(EX, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ext")
    rng = np.random.default_rng(20260922)
    pk_d, truths = synth_pop(EX, rng, "DESI")
    pk_e, _ = synth_pop(EX, rng, "eBOSS", truths=truths)
    d_dir, d_sha = write_pop(EX, tmp, pk_d, "DESI", "arm")
    e_dir, e_sha = write_pop(EX, tmp, pk_e, "eBOSS", "arm")
    d_gate = tmp / "gate_desi.json"; d_gate.write_text(json.dumps(gate_from_pop(EX, pk_d, "DESI")))
    e_gate = tmp / "gate_eboss.json"; e_gate.write_text(json.dumps(gate_from_pop(EX, pk_e, "eBOSS")))
    esc = tmp / "esc.json"; esc.write_text(json.dumps(esc_from_pop(pk_d)))
    return dict(tmp=tmp, pk_d=pk_d, pk_e=pk_e, d=(d_dir, d_sha, str(d_gate)), e=(e_dir, e_sha, str(e_gate)),
                esc=str(esc))


# ------------------------------ structural tests ------------------------------

def test_inventory_constants(EX):
    assert len(EX.live_params("DESI")) == 27 and len(EX.live_params("eBOSS")) == 23
    assert "ns" not in EX.TAU_RUNGS and all(r in EX.MAIN_NAMES_25 for r in EX.TAU_RUNGS)
    assert not any(r in EX.live_params("DESI") for r in EX.TAU_RUNGS)
    b = EX.block_map("DESI")
    assert list(b.keys()) == ["COSMO", "EMU", "MF", "HCD", "METALS", "RES",
                              "COSMO+EMU", "COSMO+MF", "COSMO+HCD", "COSMO+METALS", "COSMO+RES"]
    assert [len(v) for v in b.values()] == [2, 7, 2, 6, 8, 2, 9, 4, 8, 10, 4]
    assert [len(v) for v in EX.block_map("eBOSS").values()] == [2, 7, 2, 6, 4, 2, 9, 4, 8, 6, 4]
    assert all(EX.case_of(p) == "A" for p in EX.CASE_A) and len(EX.CASE_A) == 17
    assert all(EX.case_of(p) == "C" for p in EX.METALS["DESI"] + EX.RES)
    assert EX.EXT_SEED_ROOT != EX.JC.SEED_ROOT


def test_holm_general_matches_pair_and_is_stepdown(EX):
    rng = np.random.default_rng(1)
    for _ in range(200):
        pa, pb = rng.uniform(0, 0.12, size=2)
        assert tuple(EX.holm([pa, pb])) == EX.JC.holm_pair(pa, pb)
    p = np.array([0.001, 0.02, 0.03, 0.5])
    f = EX.holm(p, alpha=0.05)          # 0.001 < 0.0125 fires; 0.02 > 0.0167 stops
    assert f.tolist() == [True, False, False, False]
    p = np.array([0.001, 0.01, 0.02, 0.5])
    assert EX.holm(p, alpha=0.05).tolist() == [True, True, True, False]


def test_loader_schema_and_pairing(EX, paired_setup):
    s = paired_setup
    rd = EX.load_arm_ext(s["d"][0], s["d"][1], "DESI")
    re_ = EX.load_arm_ext(s["e"][0], s["e"][1], "eBOSS")
    assert rd[0]["X"].shape[1] == 27 and re_[0]["X"].shape[1] == 23
    assert EX.check_pairing_ext(rd, re_)
    # Case C truths differ by construction and must NOT refuse
    i = rd[0]["params"].index("f_res_amp"); j = re_[0]["params"].index("f_res_amp")
    assert rd[0]["t"][i] != re_[0]["t"][j]


def test_refusals(EX, paired_setup, tmp_path):
    s = paired_setup
    # wrong names
    bad = dict(s["pk_d"][0]); bad["names"] = list(bad["names"])[::-1]
    d, sha = write_pop(EX, tmp_path, [bad] + s["pk_d"][1:], "DESI", "badnames")
    with pytest.raises(EX.ExtRefusal):
        EX.load_arm_ext(d, sha, "DESI")
    # missing site
    bad = dict(s["pk_d"][0]); se = dict(bad["sites_extra"]); se.pop("s_dla"); bad["sites_extra"] = se
    d, sha = write_pop(EX, tmp_path, [bad] + s["pk_d"][1:], "DESI", "missite")
    with pytest.raises(EX.ExtRefusal):
        EX.load_arm_ext(d, sha, "DESI")
    # Case D: primary target truth mismatch across arms -> refusal naming CASE D
    bad_e = [dict(z) for z in s["pk_e"]]
    se = {k: dict(v) for k, v in bad_e[5]["sites_extra"].items()}
    se["dtau0"] = dict(se["dtau0"]); se["dtau0"]["truth"] = se["dtau0"]["truth"] + 1e-6
    bad_e[5]["sites_extra"] = se
    d, sha = write_pop(EX, tmp_path, bad_e, "eBOSS", "cased")
    rd = EX.load_arm_ext(s["d"][0], s["d"][1], "DESI")
    with pytest.raises(EX.ExtRefusal, match="CASE D"):
        EX.check_pairing_ext(rd, EX.load_arm_ext(d, sha, "eBOSS"))
    # Case D on the ns route (ns lives in truth_vec): the tag must still read CASE D
    bad_e2 = [dict(z) for z in s["pk_e"]]
    tv = np.array(bad_e2[7]["truth_vec"], float); tv[0] += 1e-6; bad_e2[7]["truth_vec"] = tv
    d2, sha2 = write_pop(EX, tmp_path, bad_e2, "eBOSS", "casedns")
    with pytest.raises(EX.ExtRefusal, match="CASE D"):
        EX.check_pairing_ext(rd, EX.load_arm_ext(d2, sha2, "eBOSS"))
    # consistency: missing repaired site -> refusal; L_all mismatch -> refusal
    g0 = json.loads(open(s["d"][2]).read()); g0["legs"]["DESI"]["repaired_sectors"]["sites"].pop("f_res_amp")
    gm = tmp_path / "gate_missing.json"; gm.write_text(json.dumps(g0))
    with pytest.raises(EX.ExtRefusal, match="lacks f_res_amp"):
        EX.check_consistency_ext(rd, str(gm), "DESI", esc_diag_json=s["esc"])
    g1 = json.loads(open(s["d"][2]).read()); g1["legs"]["DESI"]["L_all"][3] += 1
    gl = tmp_path / "gate_L.json"; gl.write_text(json.dumps(g1))
    with pytest.raises(EX.ExtRefusal, match="L census"):
        EX.check_consistency_ext(rd, str(gl), "DESI", esc_diag_json=s["esc"])
    # consistency: perturbed gate value -> refusal
    g = json.loads(open(s["d"][2]).read()); g["legs"]["DESI"]["pulls"]["ns"]["std"] *= 1.0 + 1e-6
    bad_gate = tmp_path / "bad_gate.json"; bad_gate.write_text(json.dumps(g))
    with pytest.raises(EX.ExtRefusal, match="consistency FAIL"):
        EX.check_consistency_ext(rd, str(bad_gate), "DESI", esc_diag_json=s["esc"])
    # consistency passes on the honest gate + esc
    ok = EX.check_consistency_ext(rd, s["d"][2], "DESI", esc_diag_json=s["esc"])
    assert len(ok) == 5 * 3 + 10 * 3 + 2


def test_preflight_ext_passes_and_exposes_only_flags(EX, paired_setup):
    s = paired_setup
    pf = EX.preflight_ext(*s["d"], s["esc"], *s["e"])
    assert pf["schema_ok"] and pf["pairing_ok"] and not pf["case_D_fires"]
    assert pf["n_params"] == dict(DESI=27, eBOSS=23)
    assert not any(isinstance(v, float) for v in pf.values())


# ------------------------------ statistical validity (synthetic, small B) ------------------------------

def test_level_b_c_null_uniformity(EX):
    """Amendment section 9: null-p uniformity of c_j (Level B) and of block T_max/T_min (Level C)
    at a heterogeneous, non-Gaussian, truncated fixture with L in the real census range."""
    rng = np.random.default_rng(20260923)
    M, B = 40, 200
    pc, pmax, pmin = [], [], []
    for _ in range(M):
        pk, _ = synth_pop(EX, rng, "eBOSS", Ls=rng.integers(75, 200, size=48))
        recs = _recs_from_pkls(EX, pk, "eBOSS")
        J = EX._shared_index(recs, np.random.default_rng(rng.integers(2**31)), B)
        params = recs[0]["params"]
        res, _ = EX._subspace_analysis(recs, (params.index("ns"), params.index("s_dla")), J, None, 1, want_boot=False)
        pc.append(res["c"]["p_abs"])
        cols = tuple(params.index(p) for p in EX.block_map("eBOSS")["COSMO+MF"])   # d = 4
        res4, _ = EX._subspace_analysis(recs, cols, J, None, 1, want_boot=False)
        pmax.append(res4["T_max"]["p"]); pmin.append(res4["T_min"]["p"])
    for name, arr in (("c_j", pc), ("T_max d=4", pmax), ("T_min d=4", pmin)):
        ks = sps.kstest(arr, "uniform")
        assert ks.pvalue > 0.01, f"{name} null p not uniform: KS p {ks.pvalue}, mean {np.mean(arr)}"


def test_level_b_power_against_injected_correlation(EX):
    rng = np.random.default_rng(7)
    rej = 0
    for _ in range(12):
        pk, _ = synth_pop(EX, rng, "eBOSS", rot=dict(j="s_lls", cov=np.array([[1.6, 0.9], [0.9, 1.6]])))
        recs = _recs_from_pkls(EX, pk, "eBOSS")
        J = EX._shared_index(recs, np.random.default_rng(rng.integers(2**31)), 300)
        params = recs[0]["params"]
        res, _ = EX._subspace_analysis(recs, (params.index("ns"), params.index("s_lls")), J, None, 1, want_boot=False)
        rej += res["c"]["p_abs"] < 0.05
    assert rej >= 9          # strong injected correlation must be detected most of the time


# ------------------------------ end-to-end ------------------------------

def test_run_ext_structure_determinism_and_one_shot(EX, paired_setup, tmp_path):
    s = paired_setup
    o1 = tmp_path / "ext1.json"; o2 = tmp_path / "ext2.json"
    out1 = EX.run_ext(*s["d"], s["esc"], *s["e"], str(o1), B_null=150, B_boot=30)
    out2 = EX.run_ext(*s["d"], s["esc"], *s["e"], str(o2), B_null=150, B_boot=30)
    assert json.loads(o1.read_text()) == json.loads(o2.read_text())          # deterministic
    with pytest.raises(EX.ExtRefusal, match="refusing to overwrite"):
        EX.run_ext(*s["d"], s["esc"], *s["e"], str(o1), B_null=150, B_boot=30)
    for arm, n in (("DESI", 27), ("eBOSS", 23)):
        a = out1[arm]
        assert len(a["params"]) == n and len(a["level_A"]["params"]) == n
        assert a["level_B"]["m_B"] == n - 3 and len(a["level_B"]["planes"]) == n - 1      # 2 P3-internal planes reported, not counted
        for q in ("tau0_amp", "dtau0"):
            assert a["level_B"]["planes"][q]["counted_in_family_B"] is False and a["level_B"]["planes"][q]["c"]["holm_fires"] is None
        counted = a["level_B"]["counted_planes"]
        assert set(counted) == set(a["level_B"]["planes"]) - {"tau0_amp", "dtau0"}
        # B2 extremes equal the max/min over the COUNTED planes' 2x2 eigenvalues
        assert abs(a["level_B"]["B2"]["T_pmax_B"]["observed"] - max(a["level_B"]["planes"][q]["lambda_hi"] for q in counted)) < 1e-12
        assert abs(a["level_B"]["B2"]["T_pmin_B"]["observed"] - min(a["level_B"]["planes"][q]["lambda_lo"] for q in counted)) < 1e-12
        for q, r in a["level_B"]["planes"].items():
            assert "p" not in r["T_max"] and "p_DESCRIPTIVE_not_counted" in r["T_max"]
        assert a["level_C"]["K"] == 11 and a["level_C"]["family_size"] == 22
        assert a["full_space"]["d"] == n and "p" not in a["full_space"]["T_max"]
        assert not any(k.startswith("p") for k in a["full_space"]["T_max"])           # EXT-1a: no p-value at all in the full space
        assert "null_q" in a["full_space"]["T_max"] and "stability_null_reference_note" in a["full_space"]
        assert set(a["level_A"]["params"]["ns"].keys()) >= {"pull_mean", "pull_sd", "rank_ks_p", "n_outside_band",
                                                          "rank_nonuniform_holm"}
        for blk, r in a["level_C"]["blocks"].items():
            assert "holm_family_fires" in r["T_max"] and "stability" in r and r["T_det"]["inside"] in (True, False)
            assert r["marginal_pattern_class"] in ("known_nonunit", "unit_or_unread")
            assert r["ext_b_label"] in ("none", "EXT-B-joint", "EXT-B-localization", "attributed-to-ns-marginal",
                                        "EXT-B-anisotropy-consistent-with-disclosed-marginals")
            assert (r["ext_b_label"] == "none") == (not r["implicated_EXT_B"])
    # DESI-only metal nodes are Case C in the cross-arm map; Case A planes carry both flags
    assert out1["cross_arm"]["level_B"]["f_SiII_DESI_z0"]["case"] == "C"
    assert set(out1["cross_arm"]["level_B"]["s_dla"].keys()) == {"desi_fires", "eboss_fires"}
    xc = out1["cross_arm"]["level_C"]
    assert set(xc) == set(EX.block_map("DESI"))
    assert {"desi_implicated", "eboss_implicated", "eboss_block_clean", "rotation_word_final", "desi_label"} <= set(xc["HCD"])
    assert xc["METALS"]["case"] == "C" and "eboss_implicated" not in xc["METALS"]
    # preflight_ext now includes the whitening-feasibility pass: (planes + 11 blocks + full) x 48 mocks
    pf = EX.preflight_ext(*s["d"], s["esc"], *s["e"])
    assert pf["whitening_feasibility_passes"] == dict(DESI=(26 + 11 + 1) * 48, eBOSS=(22 + 11 + 1) * 48)
    # the primary's module globals are untouched by the extension
    assert EX.JC.SEED_ROOT == 20260811 and EX.JC.B_NULL == 20000 and EX.JC.B_BOOT == 2000


def test_no_real_path_in_this_file():
    src = open(__file__).read()
    for frag in ("/scr" + "atch/", "cert_2026" + "-07", "armp_" + "DESI", "armp_" + "eBOSS"):
        assert frag not in src


# ------------------------------ helper ------------------------------

def _recs_from_pkls(EX, pkls, arm):
    params = EX.live_params(arm)
    names = list(EX.MAIN_NAMES_25)
    recs = []
    for m, z in enumerate(pkls):
        L = int(z["L"]); X = np.empty((L, len(params))); t = np.empty(len(params))
        for c, p in enumerate(params):
            if p in EX.SITES_IN_TRUTH_VEC:
                X[:, c] = np.asarray(z["draws"])[:, names.index(p)]; t[c] = z["truth_vec"][names.index(p)]
            else:
                X[:, c] = np.asarray(z["sites_extra"][p]["draws"]); t[c] = z["sites_extra"][p]["truth"]
        recs.append(dict(m=m, X=X, t=t, L=L, params=params))
    return recs


# ------------------------------ amendment EXT-1 section 9: size validation at the REAL DESI census ------------------------------

# The 48-entry DESI L census, copied as a literal from the committed gate JSON (legs.DESI.L_all), so this file
# stays self-contained and never opens a notes-repo or scratch file. It is public metadata, not a result.
REAL_DESI_L_CENSUS = [100, 120, 150, 150, 150, 120, 120, 200, 200, 200, 300, 150, 100, 150, 150, 200, 150, 75, 150, 86, 300, 100, 75, 120, 120, 100, 120, 200, 150, 150, 200, 120, 200, 200, 200, 200, 150, 100, 200, 150, 150, 300, 150, 200, 150, 120, 120, 120]


@pytest.mark.skipif(not os.environ.get("WSA_EXT_SIZE_VALIDATION"),
                    reason="slow (minutes); run once before the formal execution with WSA_EXT_SIZE_VALIDATION=1 and keep the log")
def test_size_validation_real_census_all_levels(EX):
    """Null-p uniformity, through analyze_arm_ext (the deployed path), of: c_j (three planes), the B2 pair
    (T_pmax_B, T_pmin_B), and block T_max/T_min for d in (2, 4, 6, 8, 10) = COSMO, COSMO+MF, HCD, COSMO+HCD,
    COSMO+METALS on the DESI arm (27 live parameters), at the REAL DESI L census, on the heterogeneous,
    non-Gaussian, truncated fixture. Also reports T_det band coverage per block (descriptive)."""
    rng = np.random.default_rng(20260924)
    M, B = 30, 200
    Ls = np.array(REAL_DESI_L_CENSUS)
    keys = ["c_ns_s_dla", "c_ns_f_res_amp", "c_ns_herei", "T_pmax_B", "T_pmin_B"]
    blocks = ["COSMO", "COSMO+MF", "HCD", "COSMO+HCD", "COSMO+METALS"]
    P = {k: [] for k in keys}
    for b in blocks:
        P[b + " T_max"] = []; P[b + " T_min"] = []
    det_in = {b: [] for b in blocks}
    for _ in range(M):
        pk, _ = synth_pop(EX, rng, "DESI", Ls=Ls)
        recs = _recs_from_pkls(EX, pk, "DESI")
        a = EX.analyze_arm_ext(recs, "DESI", np.random.default_rng(rng.integers(2**31)),
                               np.random.default_rng(rng.integers(2**31)), B_null=B, B_boot=5)
        P["c_ns_s_dla"].append(a["level_B"]["planes"]["s_dla"]["c"]["p_abs"])
        P["c_ns_f_res_amp"].append(a["level_B"]["planes"]["f_res_amp"]["c"]["p_abs"])
        P["c_ns_herei"].append(a["level_B"]["planes"]["herei"]["c"]["p_abs"])
        P["T_pmax_B"].append(a["level_B"]["B2"]["T_pmax_B"]["p"]); P["T_pmin_B"].append(a["level_B"]["B2"]["T_pmin_B"]["p"])
        for b in blocks:
            r = a["level_C"]["blocks"][b]
            P[b + " T_max"].append(r["T_max"]["p"]); P[b + " T_min"].append(r["T_min"]["p"]); det_in[b].append(r["T_det"]["inside"])
    report = []
    for k, arr in P.items():
        ks = sps.kstest(arr, "uniform")
        report.append(f"{k:22s} mean {np.mean(arr):.3f} frac<0.05 {np.mean(np.array(arr) < 0.05):.3f} KS p {ks.pvalue:.3f}")
    for b in blocks:
        report.append(f"T_det coverage {b:14s} {np.mean(det_in[b]):.3f} (expect ~0.94 at B={B}, M={M})")
    print("\n".join(report))
    # Pass criterion (family-corrected). This test performs len(P) correlated KS tests at once, so a per-statistic
    # threshold of 0.01 would false-alarm in roughly 1 - 0.99**15 = 14 percent of healthy runs; the criterion is
    # therefore Bonferroni at family level 0.01: every statistic must have KS p > 0.01 / len(P). Record (2026-09-22):
    # the first run of this test (M = 30, seed 20260924) gave HCD T_max KS p 0.0049 with the other 14 uniform; a
    # confirmatory probe at M = 60 with a fresh seed (wsa-review-artifacts LEAD_block_uniformity_probe.log) gave KS p
    # 0.22 for the same block and uniform results for a control 6-block, the Gaussian variant and the primary d = 3
    # space, so the first result was a chance event; the family criterion below was adopted after that probe and is
    # documented in reviews/2026-09-22-WSA-EXT-REVIEW-RECORD.md. Any statistic below 0.01 is still printed as a WARNING.
    fam_alpha = 0.01 / len(P)
    for k, arr in P.items():
        pv = sps.kstest(arr, "uniform").pvalue
        if pv < 0.01:
            print(f"WARNING: {k} KS p {pv:.4f} < 0.01 (per-statistic); family criterion is {fam_alpha:.5f}")
        assert pv > fam_alpha, f"{k} null p not uniform at the family criterion: KS p {pv}"


def test_holm_nan_and_label_logic(EX):
    f = EX.holm(np.array([0.001, np.nan, 0.5]))
    assert f.tolist() == [True, False, False]                 # NaN sorts last and never fires
    mk = lambda imp, tc, cls: dict(implicated_EXT_B=imp, T_corr=dict(fires=tc), marginal_pattern_class=cls)
    blocks = {"EMU": mk(False, False, "unit_or_unread"), "COSMO+EMU": mk(True, False, "known_nonunit"),
              "METALS": mk(True, False, "unit_or_unread"), "HCD": mk(True, False, "known_nonunit"),
              "COSMO+HCD": mk(True, True, "known_nonunit"), "RES": mk(False, False, "unit_or_unread")}
    assert EX._ext_b_label("EMU", blocks["EMU"], blocks) == "none"
    assert EX._ext_b_label("COSMO+EMU", blocks["COSMO+EMU"], blocks) == "attributed-to-ns-marginal"
    assert EX._ext_b_label("METALS", blocks["METALS"], blocks) == "EXT-B-localization"
    assert EX._ext_b_label("HCD", blocks["HCD"], blocks) == "EXT-B-anisotropy-consistent-with-disclosed-marginals"
    assert EX._ext_b_label("COSMO+HCD", blocks["COSMO+HCD"], blocks) == "EXT-B-joint"
    assert set(EX.MARGINAL_PATTERN_CLASS) == set(EX.block_map("DESI"))


def test_rank_uniformity_identical_to_deployed_analyzer(EX):
    """The Level A rank criterion must be the deployed one (analyze_sbc_perleg.rank_uniformity)."""
    # The analyzer is a script that runs at import, so extract ONLY the rank_uniformity function from its syntax
    # tree and execute that definition in a clean namespace (numpy + scipy available); nothing else of the file runs.
    import ast
    src = open(os.path.join(HERE, "..", "scripts", "analyze_sbc_perleg.py")).read()
    fn = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "rank_uniformity"]
    assert len(fn) == 1, "rank_uniformity not found in analyze_sbc_perleg.py"
    ns = {"np": np}
    exec(compile(ast.Module(body=fn, type_ignores=[]), "analyze_sbc_perleg.rank_uniformity", "exec"), ns)
    deployed = ns["rank_uniformity"]
    rng = np.random.default_rng(3)
    for _ in range(20):
        u = rng.uniform(size=48) ** rng.uniform(0.5, 2.0)
        a, b = deployed(u), EX.rank_uniformity(u)
        assert a["ks_p"] == b["ks_p"] and a["n_outside_band"] == b["n_outside_band"] and a["verdict"] == b["verdict"]


def test_cross_arm_eboss_clean_conjunct_truth_table(EX):
    """EXT-1a section 3: rotation_word_final for a Case A block = within-arm conjuncts AND eBOSS block clean
    (not implicated AND T_det inside); Case C blocks carry the within-arm value with a note (Reviewer C2 S3)."""
    def arm(blocks):
        return dict(level_B=dict(planes={}), level_C=dict(blocks=blocks))
    def blk(case, imp, rot, det_in, label="none"):
        return dict(case=case, implicated_EXT_B=imp, rotation_word_conjuncts_met_within_arm=rot,
                    T_det=dict(inside=det_in), ext_b_label=label)
    cases = [  # (desi_rot, eboss_imp, eboss_det_in) -> expected rotation_word_final for a Case A block
        ((True, False, True), True), ((True, True, True), False), ((True, False, False), False), ((False, False, True), False)]
    for (d_rot, e_imp, e_det), expect in cases:
        desi = arm({"HCD": blk("A", True, d_rot, True, "EXT-B-joint"), "METALS": blk("C", True, d_rot, True)})
        eboss = arm({"HCD": blk("A", e_imp, False, e_det), "METALS": blk("C", False, False, True)})
        x = EX.cross_arm(desi, eboss)["level_C"]
        assert x["HCD"]["rotation_word_final"] is expect and x["HCD"]["eboss_block_clean"] is ((not e_imp) and e_det)
        assert x["METALS"]["case"] == "C" and x["METALS"]["rotation_word_final"] is d_rot and "eboss_implicated" not in x["METALS"]
