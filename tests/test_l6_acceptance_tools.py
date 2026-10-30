"""The L6 acceptance tools: scripts/check_wc_against_cache.py (the bundle's w_c against the cache's sightline
fractions, per class tolerance) and scripts/accept_retrained_ensemble.py (the retrained members' meta, training and
validation checks). Synthetic caches, bundles and histories only; the values mean nothing, the tests check the rules."""
import importlib.util
import json
import os

import jax
import numpy as np
import pytest

from hcd_analysis import lineage as L
from hcd_analysis.emulator import release_testing as RT
from hcd_analysis.emulator import train as T
from hcd_analysis.emulator import wc_map as W
from hcd_analysis.emulator.data import fit_target_norm, load_cache
from hcd_analysis.emulator.model import Emulator
from tests import tau0_fixtures as tf
from tests.emulator._fixture import write_synthetic_cache_v34

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, "scripts", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def cache(tmp_path, monkeypatch):
    tf.clean_code(monkeypatch)
    p = tmp_path / "cache_v34.h5"
    write_synthetic_cache_v34(p, n_sims=3, snaps_per_sim=4)
    return p


# ------------------------------------------------------------------------------------------------- the w_c check
@pytest.fixture
def bundle(cache, tmp_path):
    wc = tmp_path / "wc.json"
    W.build(cache, wc)
    d = load_cache(cache)
    ident = {"sha256": d["cache_sha256"], "size": os.path.getsize(cache)}
    return RT.write_synthetic_bundle(str(tmp_path / "bundle"), cache=(d["lineage"], ident), wc_map_path=wc)


def _wc_argv(bundle, cache, out, tol=("LLS=1e9", "subDLA=1e9", "DLA=1e9"), z=("2.4", "2.8"), **over):
    a = ["--bundle", bundle.bundle_dir, "--manifest-sha256", bundle.manifest_sha256, "--cache", str(cache),
         "--cache-sha256", over.get("cache_sha256", L.sha256_file(cache)), "--out", str(out), "--z", *z]
    for t in tol:
        a += ["--tol", t]
    return a


@pytest.mark.filterwarnings("ignore::RuntimeWarning")       # the synthetic cache's parameters lie far outside the box
def test_the_w_c_check_compares_every_simulation_and_redshift(bundle, cache, tmp_path):
    C = _script("check_wc_against_cache")
    assert C.main(_wc_argv(bundle, cache, tmp_path / "r.json")) == 0
    r = json.load(open(tmp_path / "r.json"))
    assert r["passed"] and r["n_sims"] == 3 and r["n_pairs"] == 6 and not r["missing"]
    assert r["cache_sha256"] == L.sha256_file(cache) and r["wc_map_sha256"] == bundle.lineage["wc_map"]["sha256"]
    d = load_cache(cache)
    one = r["pairs"][0]
    rows = np.flatnonzero((d["sim_name"] == one["sim"]) & (np.abs(d["z_grid"] - one["z"]) < 1e-6))
    assert np.allclose(one["w_cache"], d["w_c_cache"][rows[0], 1:], rtol=1e-15)
    assert C.main(_wc_argv(bundle, cache, tmp_path / "f.json", tol=("LLS=1e-12", "subDLA=1e9", "DLA=1e9"))) == 1
    f = json.load(open(tmp_path / "f.json"))
    assert not f["passed"] and f["per_class"]["LLS"]["n_over"] == 6 and f["per_class"]["DLA"]["n_over"] == 0
    assert C.main(_wc_argv(bundle, cache, tmp_path / "m.json", z=("2.4", "4.0"))) == 1           # 4.0 is missing
    assert len(json.load(open(tmp_path / "m.json"))["missing"]) == 3


def test_the_w_c_check_refuses_another_cache_or_incomplete_tolerances(bundle, cache, tmp_path):
    C = _script("check_wc_against_cache")
    with pytest.raises(SystemExit, match="refusing"):
        C.main(_wc_argv(bundle, cache, tmp_path / "a.json", cache_sha256="0" * 64))
    other = RT.write_synthetic_bundle(str(tmp_path / "other"))                     # trained on another cache
    with pytest.raises(SystemExit, match="trained on the cache"):
        C.main(_wc_argv(other, cache, tmp_path / "b.json"))
    for tol in (("LLS=0.02", "subDLA=0.02"), ("LLS=0.02", "subDLA=0.02", "DLA=0.05", "DLA=0.05"),
                ("LLS=0.02", "subDLA=0.02", "lls=0.05")):
        with pytest.raises(SystemExit, match="refusing"):
            C.main(_wc_argv(bundle, cache, tmp_path / "c.json", tol=tol))


# --------------------------------------------------------------------------------------- the ensemble acceptance
def _member(dirpath, cache, seed, *, hist=None, **meta_over):
    d = load_cache(cache)
    os.makedirs(dirpath, exist_ok=True)
    pre = os.path.join(dirpath, f"a4lin_prod_seed{seed}")
    model = Emulator(in_dim=10, n_k=8, n_basis=4, baseline_n_layers=1, baseline_width=16, key=jax.random.PRNGKey(seed))
    T.save_checkpoint(pre, model, {}, fit_target_norm(d, np.arange(d["P_tier_p"].shape[0])), seed=seed, cache=d,
                      cache_path=str(cache), recipe={"epochs": 10}, release=True, code=dict(tf.CLEAN))
    if meta_over:
        m = json.load(open(pre + ".meta.json"))
        m.update(meta_over)
        os.remove(pre + ".meta.json")
        json.dump(m, open(pre + ".meta.json", "w"))
    h = hist or {"train_loss": [1.0, 0.5, 0.4], "val_loss": [1.0, 0.6, 0.5], "val_resid_loss": [0.9, 0.4, 0.3],
                 "stage": [1, 2, 3]}
    json.dump(h, open(pre + ".hist.json", "w"))
    return pre


def _old_hists(tmp_path, values):
    out = []
    for i, v in enumerate(values):
        p = tmp_path / f"old_seed{i}.hist.json"
        json.dump({"val_resid_loss": [1.0, v], "val_loss": [1.0, v + 0.2]}, open(p, "w"))
        out.append(str(p))
    return out


def _acc_argv(members, cache, olds, out, *extra):
    a = ["--cache", str(cache), "--cache-sha256", L.sha256_file(cache), "--out", str(out), "--metric",
         "val_resid_loss", "--metric", "val_loss", *extra]
    for m in members:
        a += ["--member", m]
    for h in olds:
        a += ["--old-hist", h]
    return a


def test_the_ensemble_acceptance(cache, tmp_path):
    A = _script("accept_retrained_ensemble")
    ens = tmp_path / "ens"
    members = [_member(ens, cache, s) for s in (0, 1)]
    olds = _old_hists(tmp_path, [0.29, 0.31, 0.30])
    hist_prefix = ["--historical-prefix", str(tmp_path / "gate" / "prod_repaired_seed0")]
    assert A.main(_acc_argv(members, cache, olds, tmp_path / "ok.json", *hist_prefix)) == 0
    r = json.load(open(tmp_path / "ok.json"))
    assert r["passed"] and r["validation"]["val_resid_loss"]["new_mean"] == pytest.approx(0.3)
    worse = _old_hists(tmp_path / "w", [0.10, 0.11, 0.12]) if os.makedirs(tmp_path / "w") is None else None
    assert A.main(_acc_argv(members, cache, worse, tmp_path / "worse.json")) == 1
    assert not json.load(open(tmp_path / "worse.json"))["checks"]["validation_val_resid_loss_within_old_seed_spread"]
    cases = {
        "git": dict(git_sha="d" * 40),
        "dirty": dict(code_dirty=True),
        "cache": dict(cache_sha256="0" * 64),
        "record": dict(cache_lineage_sha256="0" * 64),
        "path": dict(cache_path="/somewhere/cache.h5"),
    }
    for name, over in cases.items():
        bad = [members[0], _member(tmp_path / name, cache, 1, **over)]
        assert A.main(_acc_argv(bad, cache, olds, tmp_path / f"{name}.json")) == 1, name
    short = [members[0], _member(tmp_path / "short", cache, 1, hist={"train_loss": [1.0], "val_loss": [1.0],
                                                                      "val_resid_loss": [0.3], "stage": [1]})]
    assert A.main(_acc_argv(short, cache, olds, tmp_path / "short.json")) == 1
    assert not json.load(open(tmp_path / "short.json"))["checks"]["training_completed_under_the_stopping_rule"]
    same_dir = ["--historical-prefix", str(ens / "prod_repaired_seed0")]
    assert A.main(_acc_argv(members, cache, olds, tmp_path / "dir.json", *same_dir)) == 1
    with pytest.raises(SystemExit, match="does not hash"):
        A.main(["--cache", str(cache), "--cache-sha256", "0" * 64, "--out", str(tmp_path / "x.json"), "--metric", "m",
                "--member", members[0], "--old-hist", olds[0]])
