"""scripts/export_safe_bundle.py: the release export (hcd_priya.h5/2 with the lineage block and the w_c map) refuses
ineligible inputs before writing anything, and its round trip is exact; the earlier conversion stays available as an
explicit historical reproduction (h5/1). Synthetic caches and small untrained networks only."""
import importlib.util
import json
import os

import jax
import numpy as np
import pytest

from hcd_analysis import lineage as L
from hcd_analysis.emulator import safe_h5 as H
from hcd_analysis.emulator import train as T
from hcd_analysis.emulator import wc_map as W
from hcd_analysis.emulator.data import fit_target_norm, load_cache
from hcd_analysis.emulator.model import Emulator
from tests import tau0_fixtures as tf
from tests.emulator._fixture import write_synthetic_cache_v34

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCH = {"in_dim": 10, "n_k": 8, "n_basis": 4, "baseline_n_layers": 1, "baseline_width": 16}


@pytest.fixture
def X(monkeypatch):
    spec = importlib.util.spec_from_file_location("export_safe_bundle", os.path.join(REPO, "scripts",
                                                                                     "export_safe_bundle.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RECIPE = {"synthetic": 1, "all_sims": True, "holdout_sim": None}


def _members(dirpath, cache, *, seeds=(0, 1), name="a4lin_prod", release=True, recipe=RECIPE):
    d = load_cache(cache)
    os.makedirs(dirpath, exist_ok=True)
    prefixes = []
    for s in seeds:
        model = Emulator(**ARCH, key=jax.random.PRNGKey(s))
        pre = os.path.join(dirpath, f"{name}_seed{s}")
        T.save_checkpoint(pre, model, {"in_dim": 10, "n_k": 8, "n_basis": 4}, fit_target_norm(d, np.arange(
            d["P_tier_p"].shape[0])), seed=s, cache=d, cache_path=str(cache), recipe=dict(recipe),
            release=release, code=dict(tf.CLEAN) if release else None)
        prefixes.append(pre)
    sums = os.path.join(dirpath, "SHA256SUMS")
    with open(sums, "w") as fh:
        for p in prefixes:
            for e in (".eqx", ".norm.pkl", ".meta.json"):
                fh.write(f"{L.sha256_file(p + e)}  {os.path.basename(p + e)}\n")
    return prefixes, sums


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    tf.clean_code(monkeypatch)
    cache = tmp_path / "cache_v34.h5"
    write_synthetic_cache_v34(cache, n_sims=3, snaps_per_sim=4)
    wc = tmp_path / "wc_map.json"
    W.build(cache, wc)
    prefixes, sums = _members(tmp_path / "ens", cache)
    return {"cache": str(cache), "wc": str(wc), "prefixes": prefixes, "sums": sums, "tmp": tmp_path}


def _argv(inp, out, **over):
    over = dict(over)
    members = over.pop("members", inp["prefixes"])
    a = {"--out": str(out), "--sha256sums": inp["sums"], "--sha256sums-sha256": L.sha256_file(inp["sums"]),
         "--cache": inp["cache"], "--cache-sha256": L.sha256_file(inp["cache"]), "--wc-map": inp["wc"],
         "--wc-map-sha256": L.sha256_file(inp["wc"])}
    a.update(over)
    argv = [str(x) for k, v in a.items() if v is not None for x in (k, v)]
    for p in members:
        argv += ["--member", str(p)]
    return argv


def test_the_release_export_writes_an_h5_2_bundle_that_the_release_loaders_accept(X, inputs, tmp_path):
    out = tmp_path / "out"
    assert X.main(_argv(inputs, out)) == 0
    s = json.load(open(out / "evidence" / "pilot_summary.json"))
    assert s["verdict"] == "PASS" and s["format"] == H.FORMAT and all(s["checks"].values())
    assert s["checks"]["clean_process_P_filt"] and s["checks"]["predictor_P_class"]
    bundle, sha = str(out / "bundle"), s["bundle_manifest_sha256"]
    rb = H.load_release(bundle, expected_manifest_sha256=sha)
    block = rb.lineage
    cache_rec = load_cache(inputs["cache"])["lineage"]
    assert block["release_eligible"] and block["cache"]["sha256"] == L.sha256_file(inputs["cache"])
    assert block["cache"]["lineage_record_sha256"] == L.record_sha256(cache_rec)
    assert block["code_commit"] == "c" * 40 and block["export_code"] == tf.CLEAN
    assert block["wc_map"]["sha256"] == L.sha256_file(inputs["wc"]) == rb.wc_map.sha256
    assert block["catalogues"]["n"] == 12 and block["finder_ids"] == cache_rec["finder_ids"]
    assert sorted(os.listdir(bundle)) == ["a4lin_prod_seed0.h5", "a4lin_prod_seed0.json", "a4lin_prod_seed1.h5",
                                          "a4lin_prod_seed1.json", "manifest.json", "wc_map.json"]
    man = json.load(open(os.path.join(bundle, "manifest.json")))
    assert "/" not in json.dumps(man["provenance"]["cache_file"]) and man["provenance"]["export_code"] == tf.CLEAN
    import hcd_analysis.predictor as hp
    emu = hp.load_bundle(bundle, sha)
    assert emu.n_members == 2 and emu.provenance["lineage"] == block


@pytest.mark.parametrize("damage", ["dirty", "sums_digest", "member_changed", "member_unlisted", "cache_digest",
                                    "historical_cache", "wc_digest", "wc_other_cache", "members_other_cache",
                                    "members_dev", "member_elsewhere", "holdout_sim", "not_all_sims",
                                    "fold_model", "holdout_sims_only", "fold_only"])
def test_the_release_export_refuses_ineligible_inputs_before_writing(X, inputs, tmp_path, monkeypatch, damage,
                                                                     capsys):
    out = tmp_path / "out"
    over = {}
    if damage == "dirty":
        tf.clean_code(monkeypatch, {"commit": "c" * 40, "dirty": True})
    elif damage == "sums_digest":
        over["--sha256sums-sha256"] = "0" * 64
    elif damage == "member_changed":
        p = inputs["prefixes"][1] + ".eqx"
        b = bytearray(open(p, "rb").read())
        b[-1] ^= 1
        open(p, "wb").write(bytes(b))
    elif damage == "member_unlisted":
        lines = open(inputs["sums"]).read().splitlines()
        open(inputs["sums"], "w").write("\n".join(lines[:-1]) + "\n")
    elif damage == "cache_digest":
        over["--cache-sha256"] = "0" * 64
    elif damage == "historical_cache":
        old = tf.write_v33(tmp_path / "old.h5", [("simA", 16, 3.2, 3, 1)])
        over["--cache"], over["--cache-sha256"] = str(old), L.sha256_file(old)
    elif damage == "wc_digest":
        over["--wc-map-sha256"] = "0" * 64
    elif damage in ("wc_other_cache", "members_other_cache"):
        other = tmp_path / "other.h5"
        write_synthetic_cache_v34(other, n_sims=3, snaps_per_sim=4, seed=7)
        if damage == "wc_other_cache":
            W.build(other, tmp_path / "other_wc.json")
            over["--wc-map"], over["--wc-map-sha256"] = str(tmp_path / "other_wc.json"), L.sha256_file(
                tmp_path / "other_wc.json")
        else:
            prefixes, sums = _members(tmp_path / "ens2", other)
            over.update({"members": prefixes, "--sha256sums": sums, "--sha256sums-sha256": L.sha256_file(sums)})
    elif damage == "members_dev":                         # development checkpoints (no release meta)
        prefixes, sums = _members(tmp_path / "ens3", inputs["cache"], release=False)
        over.update({"members": prefixes, "--sha256sums": sums, "--sha256sums-sha256": L.sha256_file(sums)})
    elif damage in ("holdout_sim", "not_all_sims", "fold_model", "holdout_sims_only", "fold_only"):
        # a leave-one-simulation-out or grouped-fold member (several held-out simulations) is not a release one, even
        # when its recipe claims all_sims
        recipe = {"holdout_sim": dict(RECIPE, all_sims=False, holdout_sim="sim1"),
                  "not_all_sims": dict(RECIPE, all_sims=None),
                  "fold_model": dict(RECIPE, all_sims=False, holdout_sim=None, holdout_sims=["sim0", "sim1"], fold=3),
                  "holdout_sims_only": dict(RECIPE, all_sims=True, holdout_sim=None, holdout_sims=["sim0", "sim1"]),
                  "fold_only": dict(RECIPE, all_sims=True, holdout_sim=None, fold=0)}[damage]
        prefixes, sums = _members(tmp_path / "ens5", inputs["cache"], recipe=recipe)
        over.update({"members": prefixes, "--sha256sums": sums, "--sha256sums-sha256": L.sha256_file(sums)})
    elif damage == "member_elsewhere":
        prefixes, _ = _members(tmp_path / "ens4", inputs["cache"], seeds=(5,))
        over["members"] = inputs["prefixes"] + prefixes
    assert X.main(_argv(inputs, out, **over)) == 2
    assert "refusing" in capsys.readouterr().out and not out.exists()


def test_the_command_line_separates_the_two_modes(X, inputs, tmp_path):
    for argv in ([], ["--out", str(tmp_path / "o")],                                  # release arguments missing
                 _argv(inputs, tmp_path / "o") + ["--historical-reproduction", "r"],  # modes mixed
                 ["--out", str(tmp_path / "o"), "--historical-reproduction", " ", "--manifest-sha256", "0" * 64],
                 ["--out", str(tmp_path / "o"), "--historical-reproduction", "r"],    # no manifest digest
                 _argv(inputs, tmp_path / "o") + ["--container", "safetensors"]):
        with pytest.raises(SystemExit) as e:
            X.main(argv)
        assert e.value.code == 2
    os.makedirs(tmp_path / "exists")
    assert X.main(_argv(inputs, tmp_path / "exists")) == 2


def test_the_historical_reproduction_writes_an_h5_1_bundle(X, inputs, tmp_path, monkeypatch):
    prefixes, _ = _members(tmp_path / "hist", inputs["cache"], release=False)
    man = tmp_path / "production_manifest.json"
    man.write_text("{}")
    monkeypatch.setattr(X.PE, "verify_manifest", lambda manifest_path=None: ({}, prefixes))
    out = tmp_path / "out"
    assert X.main(["--out", str(out), "--historical-reproduction", "a test reproduction", "--manifest", str(man),
                   "--manifest-sha256", L.sha256_file(man)]) == 0
    s = json.load(open(out / "evidence" / "pilot_summary.json"))
    assert s["verdict"] == "PASS" and s["format"] == H.FORMAT_HISTORICAL
    sha = s["bundle_manifest_sha256"]
    with pytest.raises(H.SafeFormatError, match="historical"):
        H.load_ensemble(str(out / "bundle"), expected_manifest_sha256=sha)
    ens, metas, _ = H.load_ensemble_historical(str(out / "bundle"), expected_manifest_sha256=sha, reason="test")
    assert len(ens.members) == 2 and metas.historical_reason == "test"
    assert X.main(["--out", str(tmp_path / "o2"), "--historical-reproduction", "r", "--manifest", str(man),
                   "--manifest-sha256", "0" * 64]) == 2
