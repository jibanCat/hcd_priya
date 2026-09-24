"""Blind-safe divergence localization readout (eBOSS prereg v1.2 amendment section 4): synthetic chain
directory tests. Run:
PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
  /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_eboss_divergence_localize.py -q -p no:cacheprovider
"""
import importlib.util
import json
import os

import numpy as np
import pytest

REPO = "/home/mfho/hcd_priya"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DL = _load("eboss_divergence_localize")
CONS = _load("eboss_priya_consistency")
Z = np.array([2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6])
NAMES = ["ns", "Ap", "herei", "heref", "alphaq", "hub", "omegamh2", "hireionz", "bhfeedback"] + \
        [f"tau0_z{i}" for i in range(13)] + ["alpha_lls", "alpha_subdla", "alpha_dla"]


def _chain_dir(tmp_path, flags, shift=0.0, alphaq_edge_rows=(), n=60, seed=1):
    """Two chains x n rows; ns/Ap shifted by `shift` (the blind offset stand-in); optional rows with
    alphaq at the upper prior edge (2.5)."""
    rng = np.random.default_rng(seed)
    C = len(flags)
    root = "real_eboss"
    d = str(tmp_path)
    fres = np.zeros((C, n)); fnode = np.zeros((C, n))
    for c in range(C):
        amp = rng.uniform(0.9, 1.1, n); dt = rng.uniform(-0.2, 0.1, n)
        ladder = amp[:, None] * ((1 + Z[None, :]) / 4.0) ** dt[:, None] * CONS.kim_tau0(Z)[None, :]
        theta = np.column_stack([rng.uniform(0.85, 1.0, n) + shift, rng.uniform(1.4e-9, 2.2e-9, n) + shift * 1e-9,
                                 rng.uniform(3.6, 4.0, n), rng.uniform(2.7, 3.1, n), rng.uniform(1.5, 2.3, n),
                                 rng.uniform(0.66, 0.74, n), rng.uniform(0.141, 0.145, n), rng.uniform(6.6, 7.9, n),
                                 rng.uniform(0.035, 0.065, n)])
        for r in alphaq_edge_rows:
            theta[r, 4] = 2.499
        hcd = rng.normal(0.1, 0.02, (n, 3))
        tab = np.column_stack([np.ones(n), rng.uniform(500, 600, n), theta, ladder, hcd])
        np.savetxt(os.path.join(d, f"{root}.{c + 1}.txt"), tab, header="weight  minusloglike  " + "  ".join(NAMES))
        fres[c] = rng.normal(-0.04, 0.005, n); fnode[c] = 10 ** rng.uniform(-2.4, -1.6, n)
    per = [int(np.sum(f)) for f in flags]
    np.savez(os.path.join(d, f"{root}.divergences.npz"), diverging=np.asarray(flags, bool), per_chain_div=np.asarray(per),
             chain_files=np.array([f"{root}.{c + 1}.txt" for c in range(C)]))
    np.savez(os.path.join(d, f"{root}.nuisance.npz"), f_res_amp=fres, f_SiIII_eBOSS_z0=fnode)
    json.dump(dict(bounds=dict(f=[0.003, 0.03], k=[0.001, 0.1]), sites={}), open(os.path.join(d, f"{root}.nuisance.json"), "w"))
    json.dump(dict(leg="eBOSS", n_chains=C, n_draws=n, n_divergent=sum(per), per_chain_div=per, blinded=True,
                   blind_params=["ns", "Ap"]), open(os.path.join(d, f"{root}.health.json"), "w"))
    return d


def _lock(tmp_path):
    p = os.path.join(str(tmp_path), "analysis.lock")
    json.dump(dict(legs=dict(eBOSS=dict(z=Z.tolist(), prior=dict(tau0_pivot_z=3.0)))), open(p, "w"))
    return p


def test_localizes_divergent_rows_with_quantiles_edges_and_clustering(tmp_path):
    n = 60
    f0 = np.zeros(n, bool); f0[[10, 11, 40]] = True
    f1 = np.zeros(n, bool)
    d = _chain_dir(tmp_path, [f0, f1], alphaq_edge_rows=(40,))
    out = os.path.join(str(tmp_path), "loc")
    assert DL.main(["--chain-dir", d, "--analysis-lock", _lock(tmp_path), "--out", out]) == 0
    res = json.load(open(out + ".json"))
    assert res["n_divergent"] == 3 and res["per_chain_div"] == [3, 0]
    rows = [(x["chain"], x["row"]) for x in res["draws"]]
    assert rows == [(0, 10), (0, 11), (0, 40)]
    flags = [x["consecutive_with_previous"] for x in res["draws"]]
    assert flags == [False, True, False]
    q = res["draws"][0]["quantiles"]
    assert set(q) >= {"ns", "Ap", "minusloglike", "tau0_amp", "dtau0", "alphaq", "f_res_amp", "f_SiIII_eBOSS_z0"}
    assert all(0.0 <= v <= 1.0 for v in q.values())
    e = res["draws"][2]["edge_distance"]
    assert "ns" not in e and "Ap" not in e                      # blinded: never an edge distance, never a value
    assert e["alphaq"] is not None and e["alphaq"] < 0.02 and "alphaq" in res["draws"][2]["near_edge"]
    assert res["draws"][2]["verdict"].startswith("near edge alphaq")
    assert res["draws"][0]["verdict"] == "interior" or res["draws"][0]["near_edge"]
    assert e["f_SiIII_eBOSS_z0"] is not None and 0.0 <= e["f_SiIII_eBOSS_z0"] <= 0.5
    md = open(out + ".md").read()
    assert "chain 0 row 40" in md and "near edge alphaq" in md


def test_output_is_invariant_under_the_blind_offset(tmp_path):
    """Shifting ns and Ap by a constant (the additive blind offset) leaves the readout byte-identical
    apart from the chain_dir path: nothing about the blinded values can leak."""
    n = 60
    f0 = np.zeros(n, bool); f0[[5, 33]] = True
    f1 = np.zeros(n, bool); f1[[7]] = True
    outs = []
    for k, shift in enumerate((0.0, 0.137)):
        sub = tmp_path / f"s{k}"; sub.mkdir()
        d = _chain_dir(sub, [f0, f1], shift=shift, seed=3)
        out = os.path.join(str(sub), "loc")
        DL.main(["--chain-dir", d, "--analysis-lock", _lock(sub), "--out", out])
        r = json.load(open(out + ".json")); r.pop("chain_dir")
        outs.append(json.dumps(r, sort_keys=True))
    assert outs[0] == outs[1]
    assert "0.137" not in outs[1]


def test_refuses_without_divergences_or_after_unblind(tmp_path):
    n = 30
    d = _chain_dir(tmp_path, [np.zeros(n, bool), np.zeros(n, bool)])
    with pytest.raises(SystemExit) as e:
        DL.main(["--chain-dir", d, "--analysis-lock", _lock(tmp_path), "--out", os.path.join(d, "x")])
    assert e.value.code == 3
    f0 = np.zeros(n, bool); f0[3] = True
    sub = tmp_path / "u"; sub.mkdir()
    d2 = _chain_dir(sub, [f0, np.zeros(n, bool)])
    open(os.path.join(d2, "UNBLINDED.stamp"), "w").write("x")
    with pytest.raises(SystemExit) as e2:
        DL.main(["--chain-dir", d2, "--analysis-lock", _lock(sub), "--out", os.path.join(d2, "x")])
    assert e2.value.code == 3
    assert not os.path.exists(os.path.join(d2, "x.json"))


def test_refuses_flag_health_mismatch(tmp_path):
    n = 30
    f0 = np.zeros(n, bool); f0[3] = True
    d = _chain_dir(tmp_path, [f0, np.zeros(n, bool)])
    h = json.load(open(os.path.join(d, "real_eboss.health.json"))); h["per_chain_div"] = [0, 1]
    json.dump(h, open(os.path.join(d, "real_eboss.health.json"), "w"))
    with pytest.raises(SystemExit) as e:
        DL.main(["--chain-dir", d, "--analysis-lock", _lock(tmp_path), "--out", os.path.join(d, "x")])
    assert e.value.code == 3
