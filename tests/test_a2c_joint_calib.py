"""Synthetic no-peek tests for scripts/a2c_joint_calib.py (prereg v2 machinery).
NO real pkl is read anywhere in this file; every fixture is generated in tmp_path.
"""
import importlib.util
import json
import os
import pickle

import numpy as np
import pytest
from scipy import stats as sps

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def JC():
    spec = importlib.util.spec_from_file_location(
        "a2c_joint_calib", os.path.join(HERE, "scripts", "a2c_joint_calib.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------ fixture builders ------------------------------

def synth_recs(rng, n=48, Ls=None, hetero=False, nongauss=False, nearbound=False,
               rot=None):
    """Exact self-draw fixture in 7 channels: truth is one draw from the same
    distribution the stored draws come from. rot: optional (3,3) covariance for the
    P3 truth-displacement alternative (breaks calibration on purpose)."""
    if Ls is None:
        Ls = rng.integers(75, 160, size=n)
    recs = []
    for i in range(n):
        L = int(Ls[i])
        d = 7
        if hetero:
            A = rng.standard_normal((d, d)) * 0.3
            Sig = np.eye(d) + A @ A.T
        else:
            Sig = np.eye(d)
        Cl = np.linalg.cholesky(Sig)
        mu = rng.standard_normal(d) * 2.0
        raw = rng.standard_normal((L + 1, d))
        Y = mu + raw @ Cl.T
        if nongauss:
            Y = np.sinh(Y * 0.3) / 0.3          # smooth monotone, same map for all rows
        if nearbound:
            Y = np.clip(Y, mu - 4.0, mu + 1.0)  # hard one-sided truncation, all rows
        X, t = Y[:L], Y[L].copy()
        if rot is not None:
            # replace the truth's P3 block: whitened displacement ~ N(0, rot)
            i3 = [0, 2, 3]
            mu3 = X[:, i3].mean(axis=0)
            S3 = np.cov(X[:, i3].T, ddof=1)
            w3, V3 = np.linalg.eigh(S3)
            S3half = (V3 * np.sqrt(w3)) @ V3.T
            t[i3] = mu3 + S3half @ (np.linalg.cholesky(rot) @ rng.standard_normal(3))
        recs.append(dict(m=i, X=X, t=t, L=L, truth_full=np.concatenate([t, [0.0]])))
    return recs


def write_arm(tmp, recs, name):
    d = tmp / name
    d.mkdir()
    import hashlib
    lines = []
    for r in recs:
        names = ["ns", "Ap", "herei", "alpha_dla", "alpha_lls", "alpha_subdla"]
        # main draws: ns, Ap, filler, alphas ; sites_extra: tau0_amp, dtau0
        main = np.column_stack([r["X"][:, 0], r["X"][:, 1], np.ones(r["L"]),
                                r["X"][:, 4], r["X"][:, 5], r["X"][:, 6]])
        truth = np.array([r["t"][0], r["t"][1], 1.0, r["t"][4], r["t"][5], r["t"][6]])
        z = dict(names=names, draws=main, truth_vec=truth, L=r["L"],
                 sites_extra=dict(
                     tau0_amp=dict(draws=r["X"][:, 2], truth=float(r["t"][2])),
                     dtau0=dict(draws=r["X"][:, 3], truth=float(r["t"][3]))))
        p = d / f"mock_{r['m']:04d}.pkl"
        with open(p, "wb") as f:
            pickle.dump(z, f)
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        lines.append(f"{h}  {p.name}")
    sha = tmp / f"{name}_sha.txt"
    sha.write_text("\n".join(lines) + "\n")
    return str(d), str(sha)


def write_gate(tmp, recs, leg):
    pulls = {}
    for ch, c in (("ns", 0), ("Ap", 1)):
        p = np.array([(r["X"][:, c].mean() - r["t"][c]) / r["X"][:, c].std(ddof=1)
                      for r in recs])
        pulls[ch] = dict(mean=float(p.mean()), std=float(p.std(ddof=1)))
    j = tmp / f"gate_{leg}.json"
    j.write_text(json.dumps(dict(legs={leg: dict(
        L_all=[r["L"] for r in recs], pulls=pulls)})))
    return str(j)


def write_esc(tmp, recs):
    out = {}
    for ch, c, key in (("tau0_amp", 2, "tau0_amp_sampled"), ("dtau0", 3, "dtau0_sampled")):
        p = np.array([(r["X"][:, c].mean() - r["t"][c]) / r["X"][:, c].std(ddof=1)
                      for r in recs])
        out[key] = dict(pull_sd=float(p.std(ddof=1)))
    j = tmp / "esc.json"
    j.write_text(json.dumps(dict(P1=dict(sampled_sites=out))))
    return str(j)


# ------------------------------ core algebra ------------------------------

def test_loo_matches_bruteforce(JC):
    rng = np.random.default_rng(1)
    X = rng.standard_normal((41, 7)) @ (np.eye(7) + 0.2 * rng.standard_normal((7, 7)))
    cols = [0, 2, 3]
    tab = JC.loo_z_table(X, None, cols)
    for j in [0, 7, 40]:
        Y = np.delete(X[:, cols], j, axis=0)
        mu = Y.mean(axis=0)
        Sig = np.cov(Y.T, ddof=1)
        z = JC.sym_inv_sqrt(Sig) @ (mu - X[j, cols])
        assert np.allclose(tab[j], z, rtol=1e-10, atol=1e-12)


def test_sym_inv_sqrt_and_nonpd_refusal(JC):
    rng = np.random.default_rng(2)
    A = rng.standard_normal((3, 3))
    S = A @ A.T + np.eye(3)
    W = JC.sym_inv_sqrt(S)
    assert np.allclose(W @ S @ W, np.eye(3), atol=1e-10)
    with pytest.raises(JC.JointCalibRefusal):
        JC.sym_inv_sqrt(np.diag([1.0, 0.0, 2.0]))


def test_eig22_batch(JC):
    rng = np.random.default_rng(3)
    M = rng.standard_normal((50, 2, 2))
    M = (M + np.swapaxes(M, 1, 2)) / 2
    lo, hi = JC.eig22_batch(M)
    ref = np.linalg.eigvalsh(M)
    assert np.allclose(lo, ref[:, 0], rtol=1e-12)
    assert np.allclose(hi, ref[:, 1], rtol=1e-12)


def test_stats_p3_tcorr_and_det(JC):
    S = np.array([[1.0, 0.3, 0.0], [0.3, 1.0, -0.5], [0.0, -0.5, 1.0]])
    tmax, tmin, tdet, tcorr = JC.stats_p3(S)
    assert np.isclose(tcorr, 0.5)
    assert np.isclose(tdet, np.linalg.slogdet(S)[1], rtol=1e-12)
    w = np.linalg.eigvalsh(S)
    assert np.isclose(tmax, w[-1]) and np.isclose(tmin, w[0])


def test_plane_set_is_the_18_external(JC):
    assert len(JC.EXternal_PLANES) == 18
    p3 = set(JC.P3_IDX)
    for a, b in JC.EXternal_PLANES:
        assert not (a in p3 and b in p3)
    assert len(set(JC.EXternal_PLANES)) == 18
    assert len(JC.ALL_PAIRS) == 21


def test_holm_pair(JC):
    assert JC.holm_pair(0.01, 0.04) == (True, True)
    assert JC.holm_pair(0.01, 0.06) == (True, False)
    assert JC.holm_pair(0.03, 0.04) == (False, False)   # smallest fails alpha/2
    assert JC.holm_pair(0.04, 0.01) == (True, True)
    assert JC.holm_pair(0.6, 0.7) == (False, False)


def test_p_estimators_never_zero(JC):
    null = np.arange(100, dtype=float)
    assert JC.p_upper(null, 1e9) == pytest.approx(1 / 101)
    assert JC.p_lower(null, -1e9) == pytest.approx(1 / 101)


# ------------------------------ null calibration (uniformity) ------------------------------

def _null_p_tmax(JC, recs, rng, B=400):
    """P3-only null p for T_max (mirrors analyze_arm's construction)."""
    N = len(recs)
    z3 = np.stack([JC.observed_z(r["X"], r["t"], list(JC.P3_IDX)) for r in recs])
    S3 = (z3.T @ z3) / N
    tmax_o = JC.stats_p3(S3)[0]
    tabs = [JC.loo_z_table(r["X"], None, list(JC.P3_IDX)) for r in recs]
    J = np.stack([rng.integers(0, r["L"], size=B) for r in recs], axis=1)
    G = np.stack([tabs[i][J[:, i]] for i in range(N)], axis=1)
    S_b = np.einsum("bni,bnj->bij", G, G) / N
    tmax_b = np.linalg.eigvalsh(S_b)[:, -1]
    return JC.p_upper(tmax_b, tmax_o)


@pytest.mark.parametrize("hetero,nongauss,nearbound", [
    (False, False, False),
    (True, True, True),
])
def test_null_p_uniform(JC, hetero, nongauss, nearbound):
    """Prereg section 6 validation: null p-values Uniform(0,1) within MC error,
    including the heterogeneous non-Gaussian near-bound fixture."""
    rng = np.random.default_rng(20260811)
    M = 120
    ps = []
    for _ in range(M):
        recs = synth_recs(rng, n=48, Ls=np.full(48, 80),
                          hetero=hetero, nongauss=nongauss, nearbound=nearbound)
        ps.append(_null_p_tmax(JC, recs, rng, B=300))
    ks = sps.kstest(ps, "uniform")
    assert ks.pvalue > 0.01, f"null p not uniform: KS p {ks.pvalue}, ps mean {np.mean(ps)}"


def test_power_against_injected_rotation(JC):
    """Sanity: the observed-magnitude rotation alternative rejects well above size."""
    rng = np.random.default_rng(7)
    rot = np.diag([1.2699 ** 2, 1.0, 0.7878 ** 2])
    hits = 0
    M = 60
    for _ in range(M):
        recs = synth_recs(rng, n=48, Ls=np.full(48, 100), rot=rot)
        hits += _null_p_tmax(JC, recs, rng, B=300) < 0.05
    assert hits / M > 0.15, f"power {hits/M} not above size"


# ------------------------------ end-to-end on synthetic arms ------------------------------

@pytest.fixture()
def synth_setup(tmp_path, JC):
    rng = np.random.default_rng(99)
    Ls = np.concatenate([np.full(10, 75), np.full(28, 150), np.full(10, 200)])
    recs = synth_recs(rng, n=48, Ls=Ls)
    # eBOSS arm: SAME truths (truth-paired), fresh draws
    recs_e = []
    for r in recs:
        L = r["L"]
        X = r["t"] + np.random.default_rng(1000 + r["m"]).standard_normal((L, 7))
        recs_e.append(dict(m=r["m"], X=X, t=r["t"].copy(), L=L,
                           truth_full=r["truth_full"].copy()))
    d_dir, d_sha = write_arm(tmp_path, recs, "desi")
    e_dir, e_sha = write_arm(tmp_path, recs_e, "eboss")
    d_gate = write_gate(tmp_path, recs, "DESI")
    e_gate = write_gate(tmp_path, recs_e, "eBOSS")
    esc = write_esc(tmp_path, recs)
    return dict(recs=recs, recs_e=recs_e, d=(d_dir, d_sha, d_gate), e=(e_dir, e_sha, e_gate),
                esc=esc, tmp=tmp_path)


def test_run_end_to_end_structure_and_determinism(JC, synth_setup, monkeypatch):
    monkeypatch.setattr(JC, "B_NULL", 300)
    monkeypatch.setattr(JC, "B_BOOT", 80)
    s = synth_setup
    out1 = JC.run(*s["d"], s["esc"], *s["e"], str(s["tmp"] / "o1.json"))
    out2 = JC.run(*s["d"], s["esc"], *s["e"], str(s["tmp"] / "o2.json"))
    assert json.dumps(out1, sort_keys=True) == json.dumps(out2, sort_keys=True)
    for arm in ("DESI", "eBOSS"):
        a = out1[arm]
        assert set(a["primary"]) == {"T_max", "T_min", "holm"}
        assert len(a["sweep"]["planes"]) == 18
        assert a["directions"]["eigenvalues_desc"][0] >= a["directions"]["eigenvalues_desc"][2]
        for v in (a["directions"]["v_max"], a["directions"]["v_min"]):
            v = np.array(v)
            assert v[np.argmax(np.abs(v))] > 0            # frozen sign convention
        assert a["descriptive"]["near_bound_leave_out"]["note"].startswith("DESCRIPTIVE")
    assert out1["outcome"] in ("A_rotation_established", "B_anisotropy_no_rotation_word",
                               "C_sweep_only", "D_nothing_fires", "E_control_fires")
    # per-arm null streams differ (frozen SeedSequence spawn order)
    assert out1["DESI"]["primary"]["T_max"]["null_q"] != out1["eBOSS"]["primary"]["T_max"]["null_q"]


def test_refusals(JC, synth_setup, tmp_path):
    s = synth_setup
    d_dir, d_sha, d_gate = s["d"]
    # census: remove a file
    victim = os.path.join(d_dir, "mock_0047.pkl")
    os.rename(victim, victim + ".bak")
    with pytest.raises(JC.JointCalibRefusal, match="census"):
        JC.load_arm(d_dir, d_sha)
    os.rename(victim + ".bak", victim)
    # sha mismatch
    bad_sha = tmp_path / "bad_sha.txt"
    lines = open(d_sha).read().splitlines()
    parts = lines[0].split()
    lines[0] = "0" * 64 + "  " + parts[1]
    bad_sha.write_text("\n".join(lines) + "\n")
    with pytest.raises(JC.JointCalibRefusal, match="sha mismatch"):
        JC.load_arm(d_dir, str(bad_sha))
    # malformed sha line
    bad2 = tmp_path / "bad2.txt"
    bad2.write_text("justonefield\n")
    with pytest.raises(JC.JointCalibRefusal, match="malformed|absent"):
        JC.load_arm(d_dir, str(bad2))
    # gate L mismatch
    recs = JC.load_arm(d_dir, d_sha)
    bad_gate = tmp_path / "bad_gate.json"
    g = json.load(open(d_gate))
    g["legs"]["DESI"]["L_all"][0] += 1
    bad_gate.write_text(json.dumps(g))
    with pytest.raises(JC.JointCalibRefusal, match="L census"):
        JC.check_consistency(recs, str(bad_gate), "DESI")
    # deployment-consistency mismatch
    g = json.load(open(d_gate))
    g["legs"]["DESI"]["pulls"]["ns"]["std"] *= 1.001
    bad_gate.write_text(json.dumps(g))
    with pytest.raises(JC.JointCalibRefusal, match="deployment-consistency"):
        JC.check_consistency(recs, str(bad_gate), "DESI")
    # truth pairing
    recs_e = JC.load_arm(*s["e"][:2])
    recs_e[3]["truth_full"] = recs_e[3]["truth_full"] + 1e-9
    with pytest.raises(JC.JointCalibRefusal, match="truth pairing"):
        JC.check_truth_pairing(recs, recs_e)


def test_sites_extra_misalignment_refused(JC, tmp_path):
    rng = np.random.default_rng(5)
    recs = synth_recs(rng, n=48, Ls=np.full(48, 60))
    d_dir, d_sha = write_arm(tmp_path, recs, "mis")
    # corrupt one sites_extra length, refresh its sha so the ALIGNMENT check fires
    p = os.path.join(d_dir, "mock_0005.pkl")
    z = pickle.load(open(p, "rb"))
    z["sites_extra"]["dtau0"]["draws"] = z["sites_extra"]["dtau0"]["draws"][:-1]
    pickle.dump(z, open(p, "wb"))
    import hashlib
    lines = open(d_sha).read().splitlines()
    lines[5] = hashlib.sha256(open(p, "rb").read()).hexdigest() + "  mock_0005.pkl"
    open(d_sha, "w").write("\n".join(lines) + "\n")
    with pytest.raises(JC.JointCalibRefusal, match="alignment"):
        JC.load_arm(d_dir, d_sha)


def test_classify_logic(JC):
    def arm(fmax=False, fmin=False, det=True, corr=False, bmax=0.9, bmin=0.9,
            pmax=False, pmin=False, any_fire=False):
        return {"primary": {"T_max": {"holm_fires": fmax}, "T_min": {"holm_fires": fmin}},
                "T_det": {"inside": det}, "T_corr": {"fires": corr},
                "directions": {"boot_cos_max": {"median": bmax},
                               "boot_cos_min": {"median": bmin}},
                "sweep": {"T_pmax": {"holm_fires": pmax}, "T_pmin": {"holm_fires": pmin}},
                "fires_any_primary_or_det": any_fire}
    clean = arm()
    assert JC.classify(arm(fmax=True, fmin=True, corr=True), clean) == "A_rotation_established"
    assert JC.classify(arm(fmax=True, fmin=True, corr=False), clean) == "B_anisotropy_no_rotation_word"
    assert JC.classify(arm(fmax=True, fmin=True, corr=True, bmax=0.5), clean) == "B_anisotropy_no_rotation_word"
    assert JC.classify(arm(fmax=True, fmin=True, corr=True, det=False), clean) == "B_anisotropy_no_rotation_word"
    assert JC.classify(arm(pmax=True), clean) == "C_sweep_only"
    assert JC.classify(clean, clean) == "D_nothing_fires"
    assert JC.classify(arm(fmax=True, fmin=True, corr=True), arm(any_fire=True)) == "E_control_fires"
