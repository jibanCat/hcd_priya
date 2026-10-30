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
    diag = r["diagnostics_not_gates"]
    assert set(diag["xbar_by_z"]) == {"2.4", "2.8"} and set(diag["map_fit_xbar"]) == {"LLS", "subDLA", "DLA"}
    m = W.read_wc_map(os.path.join(bundle.bundle_dir, "wc_map.json"))
    g = np.flatnonzero(d["snap_group_idx"] == d["snap_group_idx"][rows[0]])
    gi = int(d["snap_group_idx"][rows[0]])
    dn = d["snap_dNdX"][gi]
    xg = d["snap_total_path_dX"][gi] / d["coarse_counts"][g[0]].sum()
    assert np.allclose(one["diagnostic"]["dndx_cache"], dn, rtol=0) and one["diagnostic"]["xbar_geom"] == \
        pytest.approx(xg, rel=1e-15)
    assert np.allclose(one["diagnostic"]["w_map_fit_xbar"], m.w_c(dn, one["z"])[1:], rtol=1e-14)
    zk = str(one["z"])
    assert diag["xbar_by_z"][zk]["fit_over_geom"] == pytest.approx(
        diag["xbar_by_z"][zk]["fit"] / diag["xbar_by_z"][zk]["geom_median"])
    assert all(len(x["diagnostic"]["rel_map_geom_xbar"]) == 3 for x in r["pairs"])
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
RECIPE = {"epochs": 12, "patience": 3, "w_coh": 2.0, "all_sims": True, "holdout_sim": None, "val_seed": 12345,
          "val_frac": 0.1}


def _hist(restore=5, *, n=None, resid=0.30, recipe=RECIPE):
    """A history shaped as train.train_fold writes it (no stage key): the stop metric val_resid + w_coh val_coh
    improves until ``restore`` and stalls after it; ``n`` epochs in all (default: the early stop, restore + patience
    + 1)."""
    n = restore + recipe["patience"] + 1 if n is None else n
    vr = [resid + 0.1 * (restore - i) if i <= restore else resid + 0.01 * (i - restore) for i in range(n)]
    return {"train_loss": [1.0 / (i + 1) for i in range(n)], "val_loss": [v + 0.2 for v in vr], "val_resid_loss": vr,
            "val_coh_loss": [0.001] * n, "grad_norm": [1.0] * n, "lr": [1e-3] * n, "recipe": dict(recipe)}


def _member(dirpath, cache, seed, *, hist=None, recipe=RECIPE, **meta_over):
    d = load_cache(cache)
    os.makedirs(dirpath, exist_ok=True)
    pre = os.path.join(dirpath, f"a4lin_prod_seed{seed}")
    model = Emulator(in_dim=10, n_k=8, n_basis=4, baseline_n_layers=1, baseline_width=16, key=jax.random.PRNGKey(seed))
    T.save_checkpoint(pre, model, {}, fit_target_norm(d, np.arange(d["P_tier_p"].shape[0])), seed=seed, cache=d,
                      cache_path=str(cache), recipe=dict(recipe), release=True, code=dict(tf.CLEAN))
    if meta_over:
        m = json.load(open(pre + ".meta.json"))
        m.update(meta_over)
        os.remove(pre + ".meta.json")
        json.dump(m, open(pre + ".meta.json", "w"))
    json.dump(hist or _hist(), open(pre + ".hist.json", "w"))
    return pre


def _old_hists(tmp_path, values):
    out = []
    os.makedirs(tmp_path, exist_ok=True)
    for i, v in enumerate(values):
        p = tmp_path / f"old_seed{i}.hist.json"
        json.dump(_hist(resid=v), open(p, "w"))
        out.append(str(p))
    return out


def _acc_argv(members, cache, olds, out, *extra, old_source=None):
    src = old_source or cache                            # the old ensemble's source: the same rows by default
    a = ["--cache", str(cache), "--cache-sha256", L.sha256_file(cache), "--out", str(out), "--old-source", str(src),
         "--old-source-sha256", L.sha256_file(src), *extra]
    for m in members:
        a += ["--member", m]
    for h in olds:
        a += ["--old-hist", h]
    return a


def test_the_stopping_rule_is_the_one_train_fold_applies():
    A = _script("accept_retrained_ensemble")
    h = _hist(5)
    assert "stage" not in h and A.restored_epoch(h, RECIPE["w_coh"])[0] == 5
    assert A.stopped_by_the_rule(h, RECIPE)                                         # stopped patience after restore
    assert A.stopped_by_the_rule(_hist(11, n=12), RECIPE)                           # ran every epoch
    assert A.stopped_by_the_rule(_hist(10, n=12), RECIPE)                           # every epoch, restore earlier
    assert not A.stopped_by_the_rule(_hist(5, n=8), RECIPE)                         # stopped one epoch early
    assert not A.stopped_by_the_rule(_hist(5, n=10), RECIPE)                        # ran past the stop
    assert not A.stopped_by_the_rule(_hist(11, n=13), RECIPE)                       # more than the recipe's epochs
    bad = _hist(5)
    bad["val_loss"][2] = float("nan")
    assert not A.stopped_by_the_rule(bad, RECIPE)
    no_coh = {k: v for k, v in _hist(5).items() if k != "val_coh_loss"}
    assert not A.stopped_by_the_rule(no_coh, RECIPE)
    assert A.stopped_by_the_rule(no_coh, dict(RECIPE, w_coh=0.0))                 # the stop is val_resid alone
    flat = _hist(5)
    flat["val_resid_loss"][6] = flat["val_resid_loss"][5] - 5e-10                  # within 1e-9: no improvement
    assert A.restored_epoch(flat, RECIPE["w_coh"])[0] == 5 and A.stopped_by_the_rule(flat, RECIPE)
    assert not A.stopped_by_the_rule(_hist(5), {"epochs": 12})                     # the recipe must give the rule


def test_the_ensemble_acceptance(cache, tmp_path):
    A = _script("accept_retrained_ensemble")
    ens = tmp_path / "ens"
    members = [_member(ens, cache, s) for s in (0, 1)]
    olds = _old_hists(tmp_path / "old", [0.29, 0.31, 0.30])
    hist_prefix = ["--historical-prefix", str(tmp_path / "gate" / "prod_repaired_seed0")]
    assert A.main(_acc_argv(members, cache, olds, tmp_path / "ok.json", *hist_prefix)) == 0
    r = json.load(open(tmp_path / "ok.json"))
    v = r["validation"]
    assert r["passed"] and v["new_mean"] == pytest.approx(0.3) and v["old_threshold"] == pytest.approx(0.31)
    assert r["checks"]["training_completed_under_the_stopping_rule"] and set(v["reported"]) == {
        "val_loss", "val_coh_loss", "stop_metric"}
    worse = _old_hists(tmp_path / "w", [0.10, 0.11, 0.12])
    assert A.main(_acc_argv(members, cache, worse, tmp_path / "worse.json")) == 1
    w = json.load(open(tmp_path / "worse.json"))
    assert not w["checks"]["validation_val_resid_loss_at_restore_within_old_seed_spread"]
    assert w["validation"]["old_threshold"] == pytest.approx(0.12)
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
    short = [members[0], _member(tmp_path / "short", cache, 1, hist=_hist(5, n=7))]
    assert A.main(_acc_argv(short, cache, olds, tmp_path / "short.json")) == 1
    assert not json.load(open(tmp_path / "short.json"))["checks"]["training_completed_under_the_stopping_rule"]
    same_dir = ["--historical-prefix", str(ens / "prod_repaired_seed0")]
    assert A.main(_acc_argv(members, cache, olds, tmp_path / "dir.json", *same_dir)) == 1
    with pytest.raises(SystemExit, match="does not hash"):
        A.main(["--cache", str(cache), "--cache-sha256", "0" * 64, "--out", str(tmp_path / "x.json"),
                "--member", members[0], "--old-hist", olds[0], "--old-source", str(cache), "--old-source-sha256",
                L.sha256_file(cache)])


def test_every_metric_is_read_at_the_restored_epoch(capsys):
    """The saved model is the stop-metric argmin, not the val_resid_loss minimum: a later val_resid_loss dip whose
    coherence term rose does not count, for old and new members alike."""
    A = _script("accept_retrained_ensemble")
    h = _hist(5)
    h["val_resid_loss"][7] = 0.05                                    # lower val_resid, but the stop metric is worse:
    h["val_coh_loss"][7] = 1.0                                       # 0.05 + 2 x 1.0 > 0.30 + 2 x 0.001
    x = A.at_restore(h, RECIPE)
    assert x["epoch"] == 5 and x["val_resid_loss"] == pytest.approx(0.30) and x["val_loss"] == pytest.approx(0.50)
    assert x["stop_metric"] == pytest.approx(0.302) and min(h["val_resid_loss"]) == 0.05
    assert A.at_restore({"val_resid_loss": []}, RECIPE) is None


def test_the_comparison_must_be_like_for_like(cache, tmp_path):
    import shutil
    import h5py
    A = _script("accept_retrained_ensemble")
    members = [_member(tmp_path / "ens", cache, s) for s in (0, 1)]
    olds = _old_hists(tmp_path / "old", [0.29, 0.31, 0.30])
    assert A.main(_acc_argv(members, cache, olds, tmp_path / "ok.json")) == 0
    r = json.load(open(tmp_path / "ok.json"))
    assert r["checks"]["recipe_equal_to_the_old_ensemble"] and r["rows"]["rows_differing"] == {
        "sim_name": 0, "z_grid": 0, "target_F": 0}
    other = _old_hists(tmp_path / "other", [0.29, 0.31, 0.30])
    h = json.load(open(other[1]))
    h["recipe"]["patience"] = 4                                        # another recipe in one old member
    json.dump(h, open(other[1], "w"))
    assert A.main(_acc_argv(members, cache, other, tmp_path / "recipe.json")) == 1
    assert not json.load(open(tmp_path / "recipe.json"))["checks"]["recipe_equal_to_the_old_ensemble"]
    for key, edit in (("target_F", lambda a: a.__setitem__(3, a[3] * (1 + 1e-15))),
                      ("z_grid", lambda a: a.__setitem__(0, a[0] + 0.2))):
        src = tmp_path / f"old_source_{key}.h5"
        shutil.copy2(cache, src)
        with h5py.File(src, "r+") as f:
            arr = f[key][...]
            edit(arr)
            f[key][...] = arr
        assert A.main(_acc_argv(members, cache, olds, tmp_path / f"{key}.json", old_source=src)) == 1
        rr = json.load(open(tmp_path / f"{key}.json"))
        assert not rr["checks"]["same_rows_and_validation_split_as_the_old_ensemble"]
        assert rr["rows"]["rows_differing"][key] >= 1
    stated = [_member(tmp_path / "stated", cache, s, recipe=dict(RECIPE, n_val=5)) for s in (0, 1)]
    hists = []
    for i in range(3):
        p = tmp_path / f"stated_old{i}.hist.json"
        json.dump(_hist(resid=0.3, recipe=dict(RECIPE, n_val=5)), open(p, "w"))
        hists.append(str(p))
    assert A.main(_acc_argv(stated, cache, hists, tmp_path / "sizes.json")) == 1      # the split has 4 rows
    assert json.load(open(tmp_path / "sizes.json"))["rows"]["n_val"] == 4
    argv = _acc_argv(members, cache, olds, tmp_path / "pin.json")
    argv[argv.index("--old-source-sha256") + 1] = "0" * 64
    with pytest.raises(SystemExit, match="does not hash"):
        A.main(argv)


def test_a_member_left_out_of_a_simulation_is_refused(cache, tmp_path):
    A = _script("accept_retrained_ensemble")
    loso = dict(RECIPE, all_sims=False, holdout_sim="sim1")
    members = [_member(tmp_path / "ens", cache, s, recipe=loso, hist=_hist(recipe=loso)) for s in (0, 1)]
    olds = []
    for i in range(3):
        p = tmp_path / f"old{i}.hist.json"
        json.dump(_hist(resid=0.3, recipe=loso), open(p, "w"))
        olds.append(str(p))
    assert A.main(_acc_argv(members, cache, olds, tmp_path / "r.json")) == 1
    r = json.load(open(tmp_path / "r.json"))
    assert not r["checks"]["trained_on_every_simulation"] and r["checks"]["recipe_equal_to_the_old_ensemble"]
