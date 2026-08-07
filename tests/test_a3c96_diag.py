"""Synthetic-fixture tests for scripts/a3c96_diag.py (the bounded N=96 diagnostic).

NO-PEEK DISCIPLINE: these tests build synthetic populations only and never touch the
real pkls, gate JSONs or scratch directories -- the diagnostic's first contact with
real data is its single post-review production invocation (prereg DG3).
"""
import hashlib
import importlib.util
import json
import os
import pickle

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name):
    spec = importlib.util.spec_from_file_location(name[:-3], os.path.join(REPO, "scripts", name))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def DG():
    return _load("a3c96_diag.py")


N = 96
NAMES = ["ns"] + [f"tau0_z{z}" for z in range(13)]


def _mk_pop(tmpdir, DG, seed=7, hetero_L=True, sd_scale=1.0):
    """Synthetic 96-mock population with the frozen KS cfg; returns (outdir, sha, gate, sdj)."""
    frozen = DG._frozen_cfg()
    rng = np.random.default_rng(seed)
    os.makedirs(tmpdir, exist_ok=True)
    pulls, ranks, t0_pulls, Ls = [], [], [], []
    lines = []
    for m in range(N):
        L = int(rng.choice([60, 100, 150, 200, 300])) if hetero_L else 150
        truth_ns = float(rng.normal(0.5, 0.05))
        draws_ns = rng.normal(truth_ns + rng.normal(0, 0.02), 0.02 * sd_scale, L)
        tau0 = np.exp(rng.normal(-1.0, 0.05, (L, 13)))
        tau0_t = np.exp(rng.normal(-1.0, 0.05, 13))
        dr = np.column_stack([draws_ns, tau0])
        tv = np.concatenate([[truth_ns], tau0_t])
        rec = dict(names=NAMES, draws=dr, truth_vec=tv, L=L,
                   run_cfg=dict(frozen), n_div=0,
                   sites_extra={"f_res_amp": dict(truth=float(rng.normal(0, 0.15)),
                                                  draws=rng.normal(0, 0.15, L))})
        p = os.path.join(tmpdir, f"mock_{m:04d}.pkl")
        with open(p, "wb") as f:
            pickle.dump(rec, f)
        lines.append(hashlib.sha256(open(p, "rb").read()).hexdigest() + f"  mock_{m:04d}.pkl")
        col = dr[:, 0]
        pulls.append((col.mean() - truth_ns) / col.std(ddof=1))
        ranks.append(float(np.mean(col < truth_ns)))
        amps = np.array([DG.tau0_amp_slope(row[1:14])[0] for row in dr])
        amp_t = DG.tau0_amp_slope(tv[1:14])[0]
        t0_pulls.append((amps.mean() - amp_t) / amps.std(ddof=1))
        Ls.append(L)
    sha = os.path.join(tmpdir, "sha96.txt")
    open(sha, "w").write("\n".join(lines) + "\n")
    gate = dict(legs={"KS": dict(
        pulls=dict(ns=dict(mean=float(np.mean(pulls)), std=float(np.std(pulls, ddof=1))),
                   tau0amp=dict(mean=float(np.mean(t0_pulls)),
                                std=float(np.std(t0_pulls, ddof=1)))),
        rank_uniformity=dict(ns=dict(ks_p=DG.rank_ks_p(ranks))),
        finite_L_null=dict(sd=DG.finite_L_null_sd(Ls)))})
    gpath = os.path.join(tmpdir, "gate.json")
    json.dump(gate, open(gpath, "w"))
    sdj = dict(L_median=float(np.median(Ls)), L_range=[int(min(Ls)), int(max(Ls))],
               n=96, survey="KS", conjuncts_ok=True)
    spath = os.path.join(tmpdir, "sd.json")
    json.dump(sdj, open(spath, "w"))
    return tmpdir, sha, gpath, spath, np.asarray(pulls), np.asarray(ranks)


FAST = dict(normref_nsim=2000, v5_nsim=120, boot_B=500)


def test_happy_path_and_consistency(DG, tmp_path):
    outdir, sha, g, s, pulls, ranks = _mk_pop(str(tmp_path / "pop"), DG)
    out, recs, sc, variables = DG.run(outdir, g, s, sha, **FAST)
    assert len(recs) == 96 and len(out["metadata"]) == 5
    assert np.isclose(sc["sd"], np.std(pulls, ddof=1), rtol=1e-12)
    # coverage equals a direct computation on the same ranks (literal frozen edges)
    for q, (lo, hi) in DG.BAND_EDGES.items():
        k = int(np.sum((ranks >= lo) & (ranks <= hi)))
        c = out["coverage"][f"{q:.2f}"]
        assert c["covered"] == k
        # SHOULD-FIX 7: the discrete null expectation sits at or below nominal
        assert c["discrete_null_expectation"] <= q + 0.02
    assert 0 <= out["coverage"]["0.68"]["ci95"][0] <= out["coverage"]["0.68"]["ci95"][1] <= 1
    rs = out["rscale"]
    assert np.isclose(rs["r_scale"], rs["s_obs"] / rs["s_ref"], rtol=1e-12)
    # MUST-FIX 4: the material-sensitivity boolean exists at the 95% level only
    assert set(rs["sensitivity_material"]) == {"s95", "r95"}
    assert set(rs["sensitivity_shift_over_width"]) == {"s68", "s95", "r68", "r95"}
    # SHOULD-FIX 9: T4 descriptor columns present
    for r in out["metadata"]:
        assert {"field", "transform", "target"} <= set(r)
    # MUST-FIX 5: V5 carries its healthy-degeneracy band
    hb = out["metadata"][4]["healthy_band"]
    assert hb["mean_rho"] > 0.05 and hb["band_97p5"] > hb["band_2p5"]
    # JSON round-trips (plain python types)
    json.dumps(out)


def test_refuses_wrong_selfdraw_anchor(DG, tmp_path):
    outdir, sha, g, s, _, _ = _mk_pop(str(tmp_path / "pop"), DG)
    sd = json.load(open(s))
    sd["n"] = 48                              # the committed n=48 near-namesake hazard
    json.dump(sd, open(s, "w"))
    with pytest.raises(DG.DiagRefusal, match="selfdraw anchor"):
        DG.run(outdir, g, s, sha, **FAST)


def test_normal_references_sane(DG):
    nr = DG.normal_references(nsim=4000)
    assert 0.88 < nr["winsorized_over_sd"] < 0.93       # NOT 1 (MUST-FIX 3)
    assert 0.98 < nr["sd_over_mad"] < 1.06
    c = nr["variance_contrib_topk"]
    assert 0.05 < c["1"] < 0.11 < c["5"] < 0.35


def test_refuses_sha_mismatch(DG, tmp_path):
    outdir, sha, g, s, _, _ = _mk_pop(str(tmp_path / "pop"), DG)
    open(os.path.join(outdir, "mock_0003.pkl"), "ab").write(b"x")
    with pytest.raises(DG.DiagRefusal, match="sha256 mismatch"):
        DG.run(outdir, g, s, sha)


def test_refuses_census_and_cfg(DG, tmp_path):
    outdir, sha, g, s, _, _ = _mk_pop(str(tmp_path / "pop"), DG)
    os.remove(os.path.join(outdir, "mock_0095.pkl"))
    with pytest.raises(DG.DiagRefusal, match="census"):
        DG.run(outdir, g, s, sha)
    # rebuild, then corrupt one cfg (and refresh its sha so the cfg check is what fires)
    outdir2, sha2, g2, s2, _, _ = _mk_pop(str(tmp_path / "pop2"), DG, seed=8)
    p = os.path.join(outdir2, "mock_0000.pkl")
    r = pickle.load(open(p, "rb"))
    r["run_cfg"]["seed"] = 999
    pickle.dump(r, open(p, "wb"))
    lines = open(sha2).read().splitlines()
    lines[0] = hashlib.sha256(open(p, "rb").read()).hexdigest() + "  mock_0000.pkl"
    open(sha2, "w").write("\n".join(lines) + "\n")
    with pytest.raises(DG.DiagRefusal, match="run_cfg"):
        DG.run(outdir2, g2, s2, sha2)


def test_refuses_gate_json_inconsistency(DG, tmp_path):
    outdir, sha, g, s, _, _ = _mk_pop(str(tmp_path / "pop"), DG)
    gate = json.load(open(g))
    gate["legs"]["KS"]["pulls"]["ns"]["std"] *= 1.001
    json.dump(gate, open(g, "w"))
    with pytest.raises(DG.DiagRefusal, match="deployment-consistency"):
        DG.run(outdir, g, s, sha)


def test_winsorized_sd_exact(DG):
    x = np.arange(96, dtype=float)          # 0..95
    w = x.copy()
    w[:5] = 5.0                              # 5 smallest -> 6th smallest
    w[-5:] = 90.0                            # 5 largest -> 6th largest
    assert np.isclose(DG.winsorized_sd(x), np.std(w, ddof=1), rtol=1e-12)


def test_holm_known_vector(DG):
    flags = DG.holm([0.001, 0.02, 0.5, 0.9, 0.04])
    assert flags == [True, False, False, False, False]
    assert DG.holm([0.005, 0.01, 0.012, 0.9, 0.9]) == [True, True, True, False, False]


def test_loo_and_contrib(DG):
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 96)
    sc = DG.scales_and_tail(x)
    i = 17
    assert np.isclose(sc["loo_sd_all"][i], np.std(np.delete(x, i), ddof=1), rtol=1e-12)
    assert 0 < sc["variance_contrib_topk"]["1"] < sc["variance_contrib_topk"]["5"] < 1


def test_bootstrap_deterministic_and_paired(DG):
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1.1, 96)
    L = rng.choice([60, 100, 150, 300], 96)
    a = DG.rscale_bootstrap(x, L, B=2000, seed=123)
    b = DG.rscale_bootstrap(x, L, B=2000, seed=123)
    assert a["r_ci95"] == b["r_ci95"]
    # paired denominator: heterogeneous L must give a varying ref across resamples,
    # hence r intervals differ from s intervals scaled by the fixed full-sample ref
    fixed = [v / a["s_ref"] for v in a["s_obs_ci95"]]
    assert not np.allclose(a["r_ci95"], fixed, rtol=1e-6)


def test_coverage_boundary_inclusive(DG):
    ranks = np.array([0.16, 0.84, 0.159, 0.841] + [0.5] * 92)
    out = DG.coverage(ranks)
    # inclusive band [0.16, 0.84] at 68%: the two boundary points count, the two outside do not
    assert out["0.68"]["covered"] == 94


def test_coverage_95_boundary_literal_edges(DG):
    # SHOULD-FIX 8: rank exactly 1/40 = 0.025 must be COVERED at the 95% level; the float
    # expression (1-0.95)/2 sits above the double for 0.025 and would exclude it.
    ranks = np.array([1.0 / 40.0, 39.0 / 40.0, 0.0249, 0.9751] + [0.5] * 92)
    out = DG.coverage(ranks)
    assert out["0.95"]["covered"] == 94
    assert DG.BAND_EDGES[0.95] == (0.025, 0.975)
