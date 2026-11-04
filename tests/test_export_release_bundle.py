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
from hcd_analysis.emulator.data import load_cache, production_row_split
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


RECIPE = {"synthetic": 1, "all_sims": True, "holdout_sim": None, "val_seed": 12345, "val_frac": 0.1}


def _members(dirpath, cache, *, seeds=(0, 1), name="a4lin_prod", release=True, recipe=None, rows=None):
    """Members as the production trainer leaves them: the normaliser fitted on the training rows of the production
    split (``rows``: other training rows) and the recipe stating the split sizes (R3-M2)."""
    d = load_cache(cache)
    tr, va, _ = production_row_split(d["sim_name"], val_seed=12345, val_frac=0.1)
    recipe = dict(RECIPE, n_train=int(tr.size), n_val=int(va.size)) if recipe is None else recipe
    norm = T._fit_norm(d, tr if rows is None else rows)
    os.makedirs(dirpath, exist_ok=True)
    prefixes = []
    for s in seeds:
        model = Emulator(**ARCH, key=jax.random.PRNGKey(s))
        pre = os.path.join(dirpath, f"{name}_seed{s}")
        T.save_checkpoint(pre, model, {"in_dim": 10, "n_k": 8, "n_basis": 4}, norm, seed=s, cache=d,
                          cache_path=str(cache), recipe=dict(recipe), release=release,
                          code=dict(tf.CLEAN) if release else None)
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
    text = capsys.readouterr().out
    assert "refusing" in text and not out.exists()
    named = {"holdout_sim": "RSV_KEY:recipe.holdout_sim", "not_all_sims": "ALL_SIMS_NOT_TRUE",
             "fold_model": "RSV_KEY:recipe.fold", "holdout_sims_only": "RSV_KEY:recipe.holdout_sims",
             "fold_only": "RSV_KEY:recipe.fold"}
    if damage in named:                                                     # the reason is named (D5-R1 F3)
        assert named[damage] in text


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


# -------------------------------------------------------- D5: the content checks, before and after writing
def _refused(X, inputs, out, capsys, **over):
    assert X.main(_argv(inputs, out, **over)) == 2
    text = capsys.readouterr().out
    assert "refusing" in text and not out.exists()
    return text


def _ens(inputs, sub, **kw):
    prefixes, sums = _members(inputs["tmp"] / sub, inputs["cache"], **kw)
    return {"members": prefixes, "--sha256sums": sums, "--sha256sums-sha256": L.sha256_file(sums)}


def test_the_export_records_the_content_checks_in_the_manifest(X, inputs, tmp_path):
    out = tmp_path / "out"
    assert X.main(_argv(inputs, out)) == 0
    man = json.load(open(out / "bundle" / "manifest.json"))
    ec = man["provenance"]["eligibility_checks"]
    assert ec["passed"] and ec["read_back"] and ec["before_writing_passed"] and ec["kind"] == "production"
    assert {c["component"]: c["outcome"] for c in ec["components"]} == {"cache": "PASS", "split": "PASS",
                                                                         "normaliser": "PASS", "wc_map": "PASS"}
    assert ec["environment"]["hcd_priya"] == tf.CLEAN and ec["rules"]["normaliser"] == "norm_fingerprint/1"
    s = json.load(open(out / "evidence" / "pilot_summary.json"))
    assert s["checks"]["content_checks_passed"] and s["checks"]["bundle_lineage_verified"]
    with pytest.raises(H.SafeFormatError, match="not a research bundle"):            # T2: production, not research
        H.load_research(str(out / "bundle"), expected_manifest_sha256=s["bundle_manifest_sha256"], purpose="x")


def test_a_normaliser_not_fitted_on_the_declared_rows_is_refused_before_writing(X, inputs, capsys):      # T16
    d = load_cache(inputs["cache"])
    all_rows = np.arange(d["sim_name"].size)
    text = _refused(X, inputs, inputs["tmp"] / "o1", capsys, **_ens(inputs, "e_all", rows=all_rows))
    assert "normaliser MISMATCH (content" in text
    tr, va, _ = production_row_split(d["sim_name"], val_seed=12345, val_frac=0.1)
    bad = dict(RECIPE, n_train=int(tr.size) + 1, n_val=int(va.size) - 1)
    assert "split MISMATCH" in _refused(X, inputs, inputs["tmp"] / "o2", capsys, **_ens(inputs, "e_n", recipe=bad))


def test_a_simulation_list_under_an_ordinary_key_is_refused_before_writing(X, inputs, capsys):    # PU-0522
    d = load_cache(inputs["cache"])
    tr, va, _ = production_row_split(d["sim_name"], val_seed=12345, val_frac=0.1)
    recipe = dict(RECIPE, n_train=int(tr.size), n_val=int(va.size), excluded=["sim1"])     # by the cache's own names
    assert "SIM_SET:recipe.excluded" in _refused(X, inputs, inputs["tmp"] / "o", capsys,
                                                 **_ens(inputs, "e_x", recipe=recipe))


def test_a_relabelled_fold_member_is_refused_by_its_content(X, inputs, capsys):        # T15
    d = load_cache(inputs["cache"])
    ftr, fva, _ = production_row_split(d["sim_name"], val_seed=12345, val_frac=0.1, holdout_sim="sim1")
    relabelled = dict(RECIPE, n_train=int(ftr.size), n_val=int(fva.size))      # fold keys stripped, all_sims true
    text = _refused(X, inputs, inputs["tmp"] / "o", capsys, **_ens(inputs, "e_f", recipe=relabelled, rows=ftr))
    assert "split MISMATCH" in text
    tr, va, _ = production_row_split(d["sim_name"], val_seed=12345, val_frac=0.1)
    forged = dict(RECIPE, n_train=int(tr.size), n_val=int(va.size))            # the sizes forged too
    assert "normaliser MISMATCH" in _refused(X, inputs, inputs["tmp"] / "o3", capsys,
                                             **_ens(inputs, "e_g", recipe=forged, rows=ftr))


def test_a_w_c_map_fitted_on_a_subset_is_refused(X, inputs, capsys):                    # T17
    d = load_cache(inputs["cache"])
    doc = W.refit_subset(d, ["sim0", "sim2"])
    rec = W.make_record(doc, d["lineage"], {"sha256": d["cache_sha256"], "size": os.path.getsize(inputs["cache"])},
                        code=dict(tf.CLEAN))
    sub = inputs["tmp"] / "wc_subset.json"
    W.write_wc_map(sub, doc, rec)
    text = _refused(X, inputs, inputs["tmp"] / "o", capsys, **{"--wc-map": str(sub),
                                                               "--wc-map-sha256": L.sha256_file(sub)})
    assert "WCMAP_NOT_FULL_CACHE" in text                    # the block rule refuses it before the content check


def test_the_files_read_back_are_checked_and_the_manifest_is_written_last(X, inputs, tmp_path, monkeypatch,
                                                                            capsys):          # T34, F-m8
    real = X.H.export_member

    def corrupting(prefix, model, meta, norm, **kw):                       # a writer that alters what it writes
        nm = dict(norm, dndx={"mean": np.asarray(norm["dndx"]["mean"]) * (1 + 1e-6), "std": norm["dndx"]["std"]})
        return real(prefix, model, meta, nm, **kw)
    monkeypatch.setattr(X.H, "export_member", corrupting)
    out = tmp_path / "out"
    assert X.main(_argv(inputs, out)) == 1
    text = capsys.readouterr().out
    assert "read back" in text and "normaliser MISMATCH" in text
    assert not (out / "bundle" / "manifest.json").exists()                    # nothing loadable was left
    with pytest.raises(H.SafeFormatError):
        H.load_ensemble(str(out / "bundle"), expected_manifest_sha256=None)
