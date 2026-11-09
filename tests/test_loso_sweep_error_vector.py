"""Gate C review blocking test BT-C2 (PU-0056/PU-0057): every 8-fold sweep writes its OWN error vector, next to its
checkpoints, with provenance, and never overwrites another sweep's.

Four gate C sweeps (repaired seeds 0, 1, 2 and the historical-cache seed 0) wrote ``<out parent>/error_vector.npz`` to
the same directory; the surviving file was the repaired seed-1 sweep's and carried no provenance."""
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from hcd_analysis.emulator.error_vector_io import load_error_vector
from hcd_analysis.emulator.train import _git_sha

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWEEP = os.path.join(REPO, "scripts", "run_loso_sweep.py")


def _sweep(cache, out, seed, workdir):
    env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONPATH=REPO, JAX_PLATFORMS="cpu")
    return subprocess.run(
        [sys.executable, SWEEP, "--cache", str(cache), "--n-folds", "2", "--epochs", "1", "--patience", "1",
         "--n-basis", "4", "--z-bands", "1", "--tau0-bands", "1", "--seed", str(seed), "--out", str(out),
         "--figdir", str(workdir / f"figs_{seed}"), "--histdir", str(workdir / f"hist_{seed}")],
        capture_output=True, text=True, env=env, cwd=str(workdir), timeout=900)


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


@pytest.fixture(scope="module")
def two_sweeps(tmp_path_factory):
    from tests.emulator._fixture import write_synthetic_cache
    d = tmp_path_factory.mktemp("sweeps")
    cache = d / "obs.h5"
    write_synthetic_cache(cache, n_sims=4, snaps_per_sim=2, n_alpha=4, n_k=12)
    runs = {s: _sweep(cache, d / f"loso8_seed{s}", s, d) for s in (0, 1)}
    return d, cache, runs


def test_loso_sweep_error_vector_is_run_specific(two_sweeps):
    d, cache, runs = two_sweeps
    for s, r in runs.items():
        assert r.returncode == 0, r.stderr[-3000:]
    assert not (d / "error_vector.npz").exists()                 # nothing at the shared, run-agnostic path
    paths = {s: d / f"loso8_seed{s}.error_vector.npz" for s in (0, 1)}
    assert all(p.exists() for p in paths.values())
    assert _sha(paths[0]) != _sha(paths[1])
    for s, p in paths.items():
        prov = json.loads(str(load_error_vector(p)["provenance"]))
        assert prov["ckpt_prefix"] == str(d / f"loso8_seed{s}")
        assert prov["seed"] == s
        assert prov["cache_sha256"] == _sha(cache)
        assert prov["code_commit"] and prov["code_commit"] == _git_sha()
        assert prov["n_folds"] == 2


def test_loso_sweep_refuses_to_overwrite_an_error_vector_before_training(two_sweeps):
    d, cache, _ = two_sweeps
    p = d / "loso8_seed0.error_vector.npz"
    before = _sha(p)
    r = _sweep(cache, d / "loso8_seed0", 0, d)
    assert r.returncode != 0
    assert "refus" in (r.stderr + r.stdout).lower() and "error_vector" in (r.stderr + r.stdout)
    assert "=== fold 0" not in r.stdout                          # refused before any training
    assert _sha(p) == before


def test_git_sha_falls_back_to_the_launch_export_commit(monkeypatch):
    """A git-archive export has no .git (gate C metas carry git_sha null): the batch exports HCD_CODE_COMMIT."""
    from hcd_analysis.emulator import train as T

    def no_git(*a, **k):
        raise FileNotFoundError("no git here")
    monkeypatch.setattr(T.subprocess, "run", no_git)
    monkeypatch.delenv("HCD_CODE_COMMIT", raising=False)
    assert T._git_sha() is None
    monkeypatch.setenv("HCD_CODE_COMMIT", "01ad2ff")
    assert T._git_sha() == "01ad2ff"
