"""scripts/export_research_bundle.py: research members (cross-validation folds, full-simulation research models) become
digest-anchored research bundles, with the w_c map refitted on the training simulations only; the declared split is
checked against the members and recomputed from the cache; nothing is written on a refusal; every production loader
refuses the result (D5 plan v2 3.5, v3 2.2-2.4, v3.1 E2; the synthetic analogue of T4, T34). Synthetic caches and
small untrained networks only; the values mean nothing, the tests check the rules."""

import importlib.util
import json
import os

import jax
import numpy as np
import pytest

import hcd_analysis.predictor as hp
from hcd_analysis import lineage as L
from hcd_analysis.emulator import eligibility as E
from hcd_analysis.emulator import safe_h5 as H
from hcd_analysis.emulator import train as T
from hcd_analysis.emulator import wc_map as W
from hcd_analysis.emulator.data import load_cache, multi_holdout_row_split, production_row_split
from hcd_analysis.emulator.model import Emulator
from tests import tau0_fixtures as tf
from tests.emulator._fixture import write_synthetic_cache_v34

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCH = {"in_dim": 10, "n_k": 8, "n_basis": 4, "baseline_n_layers": 1, "baseline_width": 16}
PURPOSE = "synthetic fold 0 of a 5-fold cross-validation (tests)"
KNOBS = {"epochs": 12, "patience": 3, "w_coh": 2.0, "val_seed": 12345, "val_frac": 0.1}


@pytest.fixture
def X():
    spec = importlib.util.spec_from_file_location(
        "export_research_bundle", os.path.join(REPO, "scripts", "export_research_bundle.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _hist(n=6, restore=2):
    vr = [0.5 - 0.1 * i if i <= restore else 0.3 + 0.01 * i for i in range(n)]
    return {
        "train_loss": [1.0] * n,
        "val_loss": [v + 0.1 for v in vr],
        "val_resid_loss": vr,
        "val_coh_loss": [0.001] * n,
    }


def _write_members(dirpath, cache, recipe, rows, seeds=(0, 1)):
    d = load_cache(cache)
    os.makedirs(dirpath, exist_ok=True)
    norm = T._fit_norm(d, rows)
    prefixes = []
    for s in seeds:
        pre = os.path.join(dirpath, f"cv_fold0_seed{s}")
        T.save_checkpoint(
            pre,
            Emulator(**ARCH, key=jax.random.PRNGKey(s)),
            {"in_dim": 10, "n_k": 8, "n_basis": 4},
            norm,
            seed=s,
            cache=d,
            cache_path=str(cache),
            recipe=dict(recipe),
            release=True,
            code=dict(tf.CLEAN),
        )
        json.dump(_hist(), open(pre + ".hist.json", "w"))
        prefixes.append(pre)
    sums = os.path.join(dirpath, "SHA256SUMS")
    with open(sums, "w") as fh:
        for p in prefixes:
            for e in (".eqx", ".norm.pkl", ".meta.json"):
                fh.write(f"{L.sha256_file(p + e)}  {os.path.basename(p + e)}\n")
    return prefixes, sums


@pytest.fixture
def world(tmp_path, monkeypatch):
    tf.clean_code(monkeypatch)
    cache = tmp_path / "cache5_v34.h5"
    write_synthetic_cache_v34(cache, n_sims=5, snaps_per_sim=4)
    d = load_cache(cache)
    folds = tmp_path / "folds.json"
    folds.write_text(json.dumps({"folds": {f"fold{k}": [f"sim{k}"] for k in range(5)}}))
    fsha = L.sha256_file(folds)
    tr, va, _ = multi_holdout_row_split(d["sim_name"], ["sim3"])
    recipe = dict(
        KNOBS,
        all_sims=False,
        holdout_sim=None,
        holdout_sims=["sim3"],
        fold=0,
        folds_sha256=fsha,
        n_train=int(tr.size),
        n_val=int(va.size),
    )
    prefixes, sums = _write_members(tmp_path / "fold", cache, recipe, tr)
    return {
        "cache": str(cache),
        "d": d,
        "folds": str(folds),
        "fsha": fsha,
        "prefixes": prefixes,
        "sums": sums,
        "tr": tr,
        "tmp": tmp_path,
        "recipe": recipe,
    }


def _argv(
    w,
    out,
    *,
    prefixes=None,
    sums=None,
    rule=E.RULE_MULTI_HOLDOUT,
    held=("sim3",),
    fold=0,
    folds=True,
    extra=(),
):
    prefixes, sums = prefixes or w["prefixes"], sums or w["sums"]
    a = [
        "--out",
        str(out),
        "--purpose",
        PURPOSE,
        "--split-rule",
        rule,
        "--sha256sums",
        sums,
        "--sha256sums-sha256",
        L.sha256_file(sums),
        "--cache",
        w["cache"],
        "--cache-sha256",
        L.sha256_file(w["cache"]),
    ]
    for s in held:
        a += ["--held-out-sim", s]
    if fold is not None:
        a += ["--fold", str(fold)]
    if folds:
        a += [
            "--folds-digest",
            w["fsha"],
            "--fold-file",
            w["folds"],
            "--fold-file-sha256",
            w["fsha"],
        ]
    for p in prefixes:
        a += ["--member", p]
    return a + list(extra)


def test_a_fold_pair_becomes_a_research_bundle_with_its_own_training_set_map(
    X, world
):  # T4 (synthetic)
    out = world["tmp"] / "out"
    assert X.main(_argv(world, out)) == 0
    s = json.load(open(out / "evidence" / "research_export_summary.json"))
    assert s["verdict"] == "PASS" and all(s["checks"].values()) and s["kind"] == "research"
    assert s["checks"]["production_loaders_refuse"] and s["checks"]["ensemble_P_filt"]
    bundle, sha = str(out / "bundle"), s["bundle_manifest_sha256"]
    rb = H.load_research(
        bundle, expected_manifest_sha256=sha, purpose="evaluate the held-out simulation"
    )
    rs = rb.research
    assert rs["purpose"] == PURPOSE and rs["split_rule"] == E.RULE_MULTI_HOLDOUT
    assert rs["split"] == {
        "held_out_sims": ["sim3"],
        "fold": 0,
        "val_seed": 12345,
        "val_frac": 0.1,
        "n_train": int(world["tr"].size),
        "n_val": world["recipe"]["n_val"],
    }
    assert rs["folds_digest"] == rs["fold_file_sha256"] == world["fsha"]
    assert {c["component"]: c["outcome"] for c in rs["content_checks"]} == {
        "cache": "PASS",
        "split": "PASS",
        "normaliser": "PASS",
        "wc_map": "PASS",
    }
    train = ["sim0", "sim1", "sim2", "sim4"]
    assert rs["wc_map"] == {
        "rule": E.RULE_WCMAP_SUBSET,
        "fitted_on_sims_sha256": E.sims_sha256(train),
        "n_sims": 4,
    }
    assert L.canonical_json(
        {k: v for k, v in rb.wc_map.doc.items() if k != "lineage"}
    ) == L.canonical_json(
        W.refit_subset(world["d"], train)
    )  # the fold map, never the full one
    assert set(rs["restored_epoch"]) == {"cv_fold0_seed0", "cv_fold0_seed1"}
    assert rs["restored_epoch"]["cv_fold0_seed0"]["epoch"] == 2
    for p, m in zip(world["prefixes"], rb.ensemble.members):  # bitwise the checkpoints
        m0, _meta, _n = T.load_checkpoint(p)
        a, b = jax.tree_util.tree_leaves(m0), jax.tree_util.tree_leaves(m)
        assert all(np.asarray(x).tobytes() == np.asarray(y).tobytes() for x, y in zip(a, b))
    for load in (
        lambda: H.load_release(bundle, expected_manifest_sha256=sha),
        lambda: hp.load_bundle(bundle, sha),
    ):
        with pytest.raises(H.SafeFormatError, match="never a production model"):
            load()
    pred = hp.load_research_bundle(bundle, sha, purpose="flags").predict(
        {n: 0.5 * (lo + hi) for n, (lo, hi) in zip(hp.PARAM_NAMES, hp.PARAM_LIMITS)},
        3.0,
        tau_eff=0.4,
    )
    assert pred.flags["research_model"] and pred.flags["research_split"]["held_out_sims"] == [
        "sim3"
    ]


@pytest.mark.parametrize(
    "change, text",
    [
        (dict(held=("sim1",)), "holds out ['sim3'], the bundle's split ['sim1']"),
        (dict(held=("sim1", "sim3")), "holds out"),
        (dict(fold=1), "fold 0 is not the split's 1"),
        (dict(rule=E.RULE_PRODUCTION_SPLIT), "not one of the production_row_split/1 rule"),
        (dict(rule="explicit_rows/1"), "not supported yet"),
        (dict(rule=E.RULE_UNVERIFIABLE), "needs --description"),
        (dict(held=("sim9",)), "holds out"),
    ],
)
def test_a_declared_split_that_is_not_the_members_is_refused_before_writing(
    X, world, change, text, capsys
):
    out = world["tmp"] / "out"
    assert X.main(_argv(world, out, **change)) == 2
    t = capsys.readouterr().out
    assert "refusing" in t and text in t and not out.exists()


def test_the_fold_file_and_the_member_digests_are_pinned(X, world, capsys):
    out = world["tmp"] / "out"
    bad = _argv(world, out)
    bad[bad.index("--fold-file-sha256") + 1] = "0" * 64
    assert X.main(bad) == 2 and "does not hash" in capsys.readouterr().out
    bad = _argv(world, out)
    bad[bad.index("--folds-digest") + 1] = "0" * 64
    assert X.main(bad) == 2 and "not the fold file's sha256" in capsys.readouterr().out
    p = world["prefixes"][1] + ".eqx"
    b = bytearray(open(p, "rb").read())
    b[-1] ^= 1
    open(p, "wb").write(bytes(b))
    assert X.main(_argv(world, out)) == 2 and "nothing unpickled" in capsys.readouterr().out
    assert not out.exists()


def test_a_normaliser_not_fitted_on_the_fold_training_rows_is_refused(X, world, capsys):
    d = world["d"]
    ptr, _, _ = production_row_split(d["sim_name"])  # fitted on every simulation: a leak
    prefixes, sums = _write_members(world["tmp"] / "leak", world["cache"], world["recipe"], ptr)
    out = world["tmp"] / "out"
    assert X.main(_argv(world, out, prefixes=prefixes, sums=sums)) == 2
    t = capsys.readouterr().out
    assert "normaliser MISMATCH" in t and not out.exists()


def test_a_dirty_checkout_is_refused(X, world, monkeypatch, capsys):
    tf.clean_code(monkeypatch, {"commit": "c" * 40, "dirty": True})
    assert (
        X.main(_argv(world, world["tmp"] / "out")) == 2
        and "clean, committed checkout" in capsys.readouterr().out
    )


def test_a_full_simulation_research_model(X, world, capsys):  # T4b analogue
    d = world["d"]
    tr, va, _ = production_row_split(d["sim_name"])
    recipe = dict(
        KNOBS,
        all_sims=True,
        holdout_sim=None,
        n_train=int(tr.size),
        n_val=int(va.size),
        research={"purpose": PURPOSE},
    )
    prefixes, sums = _write_members(world["tmp"] / "full", world["cache"], recipe, tr)
    out = world["tmp"] / "out"
    assert (
        X.main(
            _argv(
                world,
                out,
                prefixes=prefixes,
                sums=sums,
                rule=E.RULE_PRODUCTION_SPLIT,
                held=(),
                fold=None,
                folds=False,
            )
        )
        == 0
    )
    s = json.load(open(out / "evidence" / "research_export_summary.json"))
    rb = H.load_research(
        str(out / "bundle"), expected_manifest_sha256=s["bundle_manifest_sha256"], purpose="x"
    )
    assert rb.research["wc_map"]["rule"] == E.RULE_WCMAP and rb.wc_map.doc["delta_c"]["n_sims"] == 5
    unmarked = {k: v for k, v in recipe.items() if k != "research"}
    prefixes, sums = _write_members(world["tmp"] / "unmarked", world["cache"], unmarked, tr)
    assert (
        X.main(
            _argv(
                world,
                world["tmp"] / "out2",
                prefixes=prefixes,
                sums=sums,
                rule=E.RULE_PRODUCTION_SPLIT,
                held=(),
                fold=None,
                folds=False,
            )
        )
        == 2
    )
    assert "carries no recipe.research" in capsys.readouterr().out


def test_an_unverifiable_split_is_exported_with_its_checks_recorded(X, world):
    d = world["d"]
    tr, va, _ = multi_holdout_row_split(d["sim_name"], ["sim3"])
    recipe = dict(
        KNOBS, all_sims=False, withheld_sims=["sim3"], n_train=int(tr.size), n_val=int(va.size)
    )
    prefixes, sums = _write_members(world["tmp"] / "u", world["cache"], recipe, tr)
    out = world["tmp"] / "out"
    assert (
        X.main(
            _argv(
                world,
                out,
                prefixes=prefixes,
                sums=sums,
                rule=E.RULE_UNVERIFIABLE,
                fold=None,
                folds=False,
                extra=["--description", "simulation sim3 withheld by a hand-written loop"],
            )
        )
        == 0
    )
    s = json.load(open(out / "evidence" / "research_export_summary.json"))
    oc = {c["component"]: c["outcome"] for c in s["research"]["content_checks"]}
    assert oc == {
        "cache": "PASS",
        "split": "UNAVAILABLE",
        "normaliser": "UNAVAILABLE",
        "wc_map": "PASS",
    }


# ------------------------------- D5 review R2 M1: the unverifiable export cannot leak held-out simulations into the map
@pytest.mark.parametrize(
    "held, withheld, text",
    [
        ((), False, "an unverifiable split must declare the held-out simulations"),  # P1
        (("sim1",), False, "holds out ['sim3'], the bundle's split ['sim1']"),  # P2
        ((), True, "an unverifiable split must declare the held-out simulations"),  # P3
    ],
)
def test_an_unverifiable_export_must_declare_the_members_own_held_out_set(
    X, world, capsys, held, withheld, text
):
    prefixes, sums = world["prefixes"], world["sums"]
    if withheld:
        d = world["d"]
        tr, va, _ = multi_holdout_row_split(d["sim_name"], ["sim3"])
        recipe = dict(
            KNOBS, all_sims=False, withheld_sims=["sim3"], n_train=int(tr.size), n_val=int(va.size)
        )
        prefixes, sums = _write_members(world["tmp"] / "u", world["cache"], recipe, tr)
    out = world["tmp"] / "out"
    assert (
        X.main(
            _argv(
                world,
                out,
                prefixes=prefixes,
                sums=sums,
                rule=E.RULE_UNVERIFIABLE,
                held=held,
                fold=None,
                folds=False,
                extra=["--description", "a hand-written loop"],
            )
        )
        == 2
    )
    t = capsys.readouterr().out
    assert "refusing" in t and text in t and not out.exists()


# ------------------------------------ PI ruling 2026-11-04 (PU-0522): the excluded set from every simulation list
def _members_without(world, held, name, **extra):
    tr, va, _ = multi_holdout_row_split(world["d"]["sim_name"], list(held))
    recipe = dict(
        KNOBS, all_sims=False, holdout_sim=None, n_train=int(tr.size), n_val=int(va.size), **extra
    )
    return _write_members(world["tmp"] / name, world["cache"], recipe, tr)


def _unverifiable_argv(world, out, prefixes, sums, held):
    return _argv(
        world,
        out,
        prefixes=prefixes,
        sums=sums,
        rule=E.RULE_UNVERIFIABLE,
        held=held,
        fold=None,
        folds=False,
        extra=["--description", "a hand-written loop"],
    )


@pytest.mark.parametrize("held, status", [(("sim3",), 2), (("sim1", "sim3"), 0)])
def test_two_exclusion_keys_need_their_union_declared_at_export(
    X, world, capsys, held, status
):  # R2 delta n1, D2
    prefixes, sums = _members_without(
        world, ("sim1", "sim3"), "two", holdout_sims=["sim3"], withheld_sims=["sim1"]
    )
    out = world["tmp"] / "out"
    assert X.main(_unverifiable_argv(world, out, prefixes, sums, held)) == status
    t = capsys.readouterr().out
    if status:
        assert "RES_HELD" in t and "holds out ['sim1', 'sim3']" in t and not out.exists()
    else:
        s = json.load(open(out / "evidence" / "research_export_summary.json"))
        assert s["verdict"] == "PASS" and s["research"]["wc_map"][
            "fitted_on_sims_sha256"
        ] == E.sims_sha256(["sim0", "sim2", "sim4"])


@pytest.mark.parametrize(
    "extra, refusal",
    [
        ({"excluded": ["sim3"]}, None),  # an exclusion under an ordinary word
        (
            {"train_sims": ["sim0", "sim1", "sim2", "sim4"]},
            None,
        ),  # a training list: its complement is held out
        (
            {"holdout_sims": ["sim3"], "notes": {"runs": ["sim3"]}},
            "RES_SIM_SET_UNDECLARED",
        ),  # no stated meaning
    ],
)
def test_an_exclusion_in_another_representation_at_export(X, world, capsys, extra, refusal):
    prefixes, sums = _members_without(world, ("sim3",), "rep", **extra)
    out = world["tmp"] / "out3"
    assert X.main(_unverifiable_argv(world, out, prefixes, sums, ("sim3",))) == (
        2 if refusal else 0
    )
    t = capsys.readouterr().out
    assert (
        (refusal in t and not out.exists())
        if refusal
        else json.load(open(out / "evidence" / "research_export_summary.json"))["verdict"] == "PASS"
    )
    out1 = world["tmp"] / "out1"
    assert (
        X.main(_unverifiable_argv(world, out1, prefixes, sums, ("sim1",))) == 2
        and not out1.exists()
    )
    assert "RES_HELD" in capsys.readouterr().out


# ---------------------- D5 R2 pre-merge n2: what the research loader would refuse is refused before anything is written
@pytest.mark.parametrize(
    "extra",
    [
        {"cache_sims": [f"sim{k}" for k in range(5)]},  # PM2 L2: every simulation
        {"sim_weights": {f"sim{k}": 1.0 for k in range(5)}},
    ],
)  # PM2 L6: keyed by all
def test_a_list_the_research_loader_cannot_read_is_refused_before_anything_is_written(
    X, world, capsys, extra
):
    """Names outside the simulation-name form: the export (with the cache's names) sees every simulation, the loader
    (with only the declared held-out names) sees a part under a key of no stated meaning. Refused, nothing written.
    """
    prefixes, sums = _write_members(
        world["tmp"] / "n2", world["cache"], dict(world["recipe"], **extra), world["tr"]
    )
    out = world["tmp"] / "out"
    assert X.main(_argv(world, out, prefixes=prefixes, sums=sums)) == 2
    t = capsys.readouterr().out
    assert (
        "refusing" in t and "research loader would not admit" in t and "RES_SIM_SET_UNDECLARED" in t
    )
    assert not out.exists()  # no manifest, no file at all


def test_a_research_load_failure_after_writing_is_a_fail_verdict_not_a_traceback(
    X, world, capsys, monkeypatch
):
    def refuse(*a, **k):
        raise H.SafeFormatError("a synthetic load refusal (test)")

    monkeypatch.setattr(X.H, "load_research", refuse)
    out = world["tmp"] / "out"
    assert X.main(_argv(world, out)) == 1
    s = json.load(open(out / "evidence" / "research_export_summary.json"))
    assert s["verdict"] == "FAIL" and s["checks"] == {"research_load": False}
    assert "a synthetic load refusal (test)" in capsys.readouterr().out


def test_the_recorded_checks_are_the_read_back_ones(X, world, monkeypatch):
    """D5 review R1 m8: provenance.research.content_checks records the checks run on the files read back."""
    real, calls = X.E.research_content_check, []

    def spy(*a, **k):
        c = real(*a, **k)
        calls.append(c)
        for comp in c["components"]:
            comp["detail"]["call"] = len(calls)
        return c

    monkeypatch.setattr(X.E, "research_content_check", spy)
    out = world["tmp"] / "out"
    assert X.main(_argv(world, out)) == 0
    s = json.load(open(out / "evidence" / "research_export_summary.json"))
    assert len(calls) == 2 and {c["detail"]["call"] for c in s["research"]["content_checks"]} == {2}


def test_the_fold_file_alone_fills_the_folds_digest(X, world):  # R2 m5
    out = world["tmp"] / "out"
    argv = _argv(world, out)
    i = argv.index("--folds-digest")
    del argv[i : i + 2]
    assert X.main(argv) == 0
    s = json.load(open(out / "evidence" / "research_export_summary.json"))
    assert s["research"]["folds_digest"] == s["research"]["fold_file_sha256"] == world["fsha"]
