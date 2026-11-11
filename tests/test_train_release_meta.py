"""Release training meta (train.save_checkpoint release mode, scripts/train_production_emulator.py): the strict cache,
a clean known commit, the cache digest hashed at load, its lineage-record digest, a distinct basename and directory,
never overwritten. Synthetic caches only; the training itself is replaced by a stand-in (no optimisation is run).
"""

import importlib.util
import json
import os

import jax
import numpy as np
import pytest

from hcd_analysis import lineage as L
from hcd_analysis.emulator import train as T
from hcd_analysis.emulator.data import fit_target_norm, load_cache, load_cache_historical
from hcd_analysis.emulator.model import Emulator
from tests import tau0_fixtures as tf
from tests.emulator._fixture import write_synthetic_cache_v34

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_K = 8


@pytest.fixture
def cache(tmp_path, monkeypatch):
    tf.clean_code(monkeypatch)
    p = tmp_path / "cache_v34.h5"
    write_synthetic_cache_v34(p, n_sims=3, snaps_per_sim=2, n_k=N_K)
    return p


def _model_norm(d, n_basis=4):
    model = Emulator(
        in_dim=10,
        n_k=N_K,
        n_basis=n_basis,
        baseline_n_layers=1,
        baseline_width=16,
        key=jax.random.PRNGKey(0),
    )
    return model, fit_target_norm(d, np.arange(d["P_tier_p"].shape[0]))


def _files(prefix):
    return [prefix + e for e in (".eqx", ".meta.json", ".norm.pkl") if os.path.exists(prefix + e)]


def test_a_release_checkpoint_records_the_commit_and_the_cache_hashed_at_load(cache, tmp_path):
    d = load_cache(cache)
    model, norm = _model_norm(d)
    prefix = str(tmp_path / "out" / "m_seed0")
    os.makedirs(tmp_path / "out")
    with open(cache, "r+b") as fh:  # the file changes after the load: the meta keeps the
        fh.seek(0, 2)  # digest the model was trained on
        fh.write(b"\0")
    T.save_checkpoint(
        prefix,
        model,
        {"in_dim": 10, "n_k": N_K, "n_basis": 4},
        norm,
        seed=0,
        cache=d,
        cache_path=str(cache),
        recipe={"r": 1},
        release=True,
        code=dict(tf.CLEAN),
    )
    meta = json.load(open(prefix + ".meta.json"))
    assert meta["git_sha"] == "c" * 40 and meta["code_dirty"] is False
    assert meta["cache_sha256"] == d["cache_sha256"] != L.sha256_file(cache)
    assert meta["cache_lineage_sha256"] == L.record_sha256(d["lineage"])
    assert meta["cache_path"] == "cache_v34.h5"
    m2, meta2, n2 = T.load_checkpoint(prefix)
    assert meta2 == meta
    with pytest.raises(FileExistsError, match="never overwritten"):
        T.save_checkpoint(
            prefix, model, {}, norm, seed=0, cache=d, cache_path=str(cache), release=True
        )


def test_a_development_checkpoint_also_keeps_the_digest_hashed_at_load(cache, tmp_path):
    d = load_cache(cache)
    model, norm = _model_norm(d)
    sha = d["cache_sha256"]
    with open(cache, "ab") as fh:
        fh.write(b"\0")
    T.save_checkpoint(
        str(tmp_path / "dev"), model, {}, norm, seed=0, cache=d, cache_path=str(cache)
    )
    assert json.load(open(tmp_path / "dev.meta.json"))["cache_sha256"] == sha


def test_release_refusals_write_nothing(cache, tmp_path, monkeypatch):
    d = load_cache(cache)
    model, norm = _model_norm(d)
    prefix = str(tmp_path / "m")
    hist = load_cache_historical(cache, reason="test")
    with pytest.raises(ValueError, match="strict data.load_cache"):
        T.save_checkpoint(
            prefix, model, {}, norm, seed=0, cache=hist, cache_path=str(cache), release=True
        )
    bad = dict(d, lineage=dict(d["lineage"], release_eligible=False, ineligible_reasons=["x"]))
    with pytest.raises(ValueError, match="not a release-eligible"):
        T.save_checkpoint(
            prefix, model, {}, norm, seed=0, cache=bad, cache_path=str(cache), release=True
        )
    forged = dict(d, lineage=dict(d["lineage"], finder_ids=["hcd_priya.finder/1"]))
    with pytest.raises(ValueError, match="lineage record is refused"):
        T.save_checkpoint(
            prefix, model, {}, norm, seed=0, cache=forged, cache_path=str(cache), release=True
        )
    with pytest.raises(ValueError, match="code changed during training"):
        T.save_checkpoint(
            prefix,
            model,
            {},
            norm,
            seed=0,
            cache=d,
            cache_path=str(cache),
            release=True,
            code={"commit": "d" * 40, "dirty": False},
        )
    for code in ({"commit": "c" * 40, "dirty": True}, {"commit": None, "dirty": None}):
        tf.clean_code(monkeypatch, code)
        with pytest.raises(ValueError, match="clean tree at a known commit"):
            T.save_checkpoint(
                prefix, model, {}, norm, seed=0, cache=d, cache_path=str(cache), release=True
            )
    assert _files(prefix) == []
    tf.clean_code(monkeypatch)
    open(prefix + ".norm.pkl", "wb").write(b"keep")
    with pytest.raises(FileExistsError):
        T.save_checkpoint(
            prefix, model, {}, norm, seed=0, cache=d, cache_path=str(cache), release=True
        )
    assert (
        _files(prefix) == [prefix + ".norm.pkl"]
        and open(prefix + ".norm.pkl", "rb").read() == b"keep"
    )


# ------------------------------------------------------------------------------------------------ the command line
@pytest.fixture
def cmd(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "train_production_emulator", os.path.join(REPO, "scripts", "train_production_emulator.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def stand_in(d, train_idx, val_idx, *, n_basis, **kw):  # no optimisation: a fixed small network
        model = Emulator(
            in_dim=10,
            n_k=kw["n_k"],
            n_basis=n_basis,
            baseline_n_layers=1,
            baseline_width=16,
            key=kw["key"],
        )
        hist = {
            "train_loss": [1.0, 0.5],
            "val_loss": [1.0, 0.6],
            "val_resid_loss": [0.9, 0.4],
            "grad_norm": [1.0, 1.0],
            "lr": [1e-3, 1e-3],
            "stage": [1, 2],
        }
        return model, fit_target_norm(d, train_idx), {k: np.asarray(v) for k, v in hist.items()}

    monkeypatch.setattr(mod.T, "train_fold", stand_in)
    return mod


def test_the_command_writes_a_release_member(cmd, cache, tmp_path):
    out = tmp_path / "ens"
    cmd.main(["--seed", "3", "--cache", str(cache), "--out-dir", str(out), "--name", "a4lin_prod"])
    pre = str(out / "a4lin_prod_seed3")
    meta = json.load(open(pre + ".meta.json"))
    assert meta["git_sha"] == "c" * 40 and meta["code_dirty"] is False and meta["seed"] == 3
    assert meta["cache_sha256"] == L.sha256_file(cache) and meta["cache_path"] == "cache_v34.h5"
    assert meta["cache_lineage_sha256"] == L.record_sha256(load_cache(cache)["lineage"])
    assert meta["recipe"]["val_seed"] == cmd.VAL_SEED and meta["recipe"]["all_sims"] is True
    assert json.load(open(pre + ".hist.json"))["recipe"] == meta["recipe"]
    with pytest.raises(SystemExit, match="never overwritten"):
        cmd.main(
            ["--seed", "3", "--cache", str(cache), "--out-dir", str(out), "--name", "a4lin_prod"]
        )


def test_the_command_refuses_before_reading_the_cache(cmd, cache, tmp_path, monkeypatch, capsys):
    def no_read(*a, **k):
        raise AssertionError("the cache was read")

    monkeypatch.setattr(cmd, "load_cache", no_read)
    base = ["--seed", "0", "--cache", str(cache)]
    for argv in (
        ["--seed", "0", "--out-dir", str(tmp_path), "--name", "n"],  # no default cache
        base + ["--name", "n"],
        base + ["--out-dir", str(tmp_path)],
    ):  # no default output
        with pytest.raises(SystemExit) as e:
            cmd.main(argv)
        assert e.value.code == 2
    for name in ("final_prod", "prod_repaired", "a/b", ".hidden", ""):
        with pytest.raises(SystemExit, match="refusing --name"):
            cmd.main(base + ["--out-dir", str(tmp_path), "--name", name])
    with pytest.raises(SystemExit, match="historical checkpoints directory"):
        cmd.main(base + ["--out-dir", cmd.HISTORICAL_DIR, "--name", "new"])
    (tmp_path / "new_seed0.hist.json").write_text("{}")
    with pytest.raises(SystemExit, match="never overwritten"):
        cmd.main(base + ["--out-dir", str(tmp_path), "--name", "new"])
    for code in ({"commit": "c" * 40, "dirty": True}, {"commit": None, "dirty": None}):
        tf.clean_code(monkeypatch, code)
        with pytest.raises(SystemExit, match="clean tree at a known commit"):
            cmd.main(base + ["--out-dir", str(tmp_path / "x"), "--name", "new"])
    assert not (tmp_path / "x").exists()


def test_the_command_refuses_a_historical_cache(cmd, tmp_path, monkeypatch):
    from hcd_analysis import tau0_cache as TC

    tf.clean_code(monkeypatch)
    old = tf.write_v33(tmp_path / "old.h5", [("simA", 16, 3.2, 3, 1)])
    with pytest.raises(TC.CacheLineageError, match="3.3"):
        cmd.main(
            ["--seed", "0", "--cache", str(old), "--out-dir", str(tmp_path / "o"), "--name", "n"]
        )
