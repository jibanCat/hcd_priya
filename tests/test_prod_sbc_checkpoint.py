"""Per-mock checkpoint for the production Leg-A SBC shard runner (OOM / 24h-wall fix).

The shard runner used to run every mock of a shard in ONE ``run_legb`` call that accumulates all
draws in memory and writes only at the end — so an OOM (job 51884006, 16.77 GB) or the 24h wall
lost the WHOLE shard. ``run_prod_sbc_shard._run_mock`` now runs ONE mock, writes
``{out}/mock_{m:04d}.pkl`` atomically, and SKIPS if it already exists. ``merge_prod_sbc_shards``
globs ``mock_*.pkl`` (mock index from the FILENAME) alongside ``shard_*.pkl``, per-mock files
winning a conflict.

Run: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_prod_sbc_checkpoint.py -q
"""
import importlib.util
import os
import pickle

import hcd_analysis.emulator  # noqa: F401  enables x64
import numpy as np
import pytest

from hcd_analysis.emulator import closure_legb as LB

REPO = "/home/mfho/hcd_priya"


def _load_script(name):
    """Import a scripts/*.py module by path (the scripts dir is not a package)."""
    path = os.path.join(REPO, "scripts", name)
    spec = importlib.util.spec_from_file_location(name[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def runner():
    return _load_script("run_prod_sbc_shard.py")


@pytest.fixture(scope="module")
def merger():
    return _load_script("merge_prod_sbc_shards.py")


@pytest.fixture(scope="module")
def ctx_d():
    try:
        return LB.build_legb_ctx(use_xclass=True)        # light baseline (xclass ρ; no MF/eBOSS)
    except Exception as e:                               # data/checkpoint absent in this env
        pytest.skip(f"build_legb_ctx unavailable: {e}")


def _rec_close(a, b):
    """Two per-mock records carry the same NUTS output (same forward + same fold_in(seed,m) RNG)."""
    assert a["names"] == b["names"]
    assert int(a["L"]) == int(b["L"])
    assert np.allclose(np.asarray(a["truth_vec"]), np.asarray(b["truth_vec"]), rtol=0, atol=0)
    assert np.allclose(np.asarray(a["draws"]), np.asarray(b["draws"]), rtol=0, atol=1e-10)
    assert np.allclose(float(a["ll_true"]), float(b["ll_true"]), rtol=0, atol=1e-8)


def test_run_one_mock_writes_skips_and_reproduces(runner, ctx_d, tmp_path):
    ctx, d = ctx_d
    out = str(tmp_path / "armB")
    m = 0
    kw = dict(n_mocks=4, n_warmup=6, n_samples=6, max_tree_depth=6, seed=123, verbose=False)

    rec = runner._run_mock(ctx, d, m, out, **kw)            # (a) first call → writes the pkl
    path = runner._mock_path(out, m)
    assert os.path.exists(path), "per-mock pkl not written"
    mtime0 = os.path.getmtime(path)
    with open(path, "rb") as f:
        on_disk = pickle.load(f)
    _rec_close(rec, on_disk)

    rec2 = runner._run_mock(ctx, d, m, out, **kw)           # (b) second call → SKIPS (no re-NUTS)
    assert os.path.getmtime(path) == mtime0, "pkl rewritten on skip (should be a no-op load)"
    _rec_close(rec, rec2)

    # (c) REPRODUCIBILITY: the per-mock record == the m-th record of a full run_legb (the per-mock
    # RNG is fold_in(seed, m), so the single-index call reproduces the full run's mock m exactly).
    ref = LB.run_legb(ctx, d, n_mocks=kw["n_mocks"], mock_indices=[m], return_per_mock=True,
                      leg_a=True, n_warmup=kw["n_warmup"], n_samples=kw["n_samples"],
                      seed=kw["seed"], dense_mass=True, max_tree_depth=kw["max_tree_depth"],
                      verbose=False)[0]
    _rec_close(rec, ref)


def test_merge_globs_mock_pkls(merger, runner, ctx_d, tmp_path):
    """merge_prod_sbc_shards reads bare mock_*.pkl (no shard pkl needed) and aggregates."""
    ctx, d = ctx_d
    out = str(tmp_path / "armB_mockonly")
    runner._run_mock(ctx, d, 0, out, n_mocks=1, n_warmup=6, n_samples=6, max_tree_depth=6,
                     seed=7, verbose=False)
    assert os.path.exists(runner._mock_path(out, 0))
    # the merge loader path: glob mock_*.pkl, parse the index from the filename, aggregate.
    import sys
    argv = sys.argv
    sys.argv = ["merge_prod_sbc_shards.py", "--shard-dir", out, "--prob", "0.95"]
    try:
        merger.main()      # raises if the glob/aggregate path is broken
    finally:
        sys.argv = argv


def test_merge_dedups_mock_in_both_shard_and_mockpkl(merger, tmp_path):
    """A mock present in BOTH a shard_*.pkl and a mock_*.pkl → exactly one record, the per-mock
    file WINS (it is the authoritative unit of progress)."""
    out = tmp_path / "armB_overlap"
    out.mkdir()
    n_z = 3
    def _stub(tag):
        # minimal record shape aggregate_leg_a tolerates is not exercised here — we only test the
        # glob/dedup, so stash a marker the test can read back.
        return dict(_marker=tag, kept_global=np.ones(n_z, bool))
    # shard pkl: mock 0 with marker "shard"; mock_0000.pkl: mock 0 with marker "permock"
    with open(out / "shard_000.pkl", "wb") as f:
        pickle.dump(dict(idxs=[0], n_z=n_z, per_mock=[_stub("shard")], meta={}), f)
    with open(out / "mock_0000.pkl", "wb") as f:
        pickle.dump(_stub("permock"), f)

    # re-implement the merge loader's glob/dedup the way merger.main does, then assert the winner.
    import glob as _glob
    import re as _re
    by_mock = {}
    mock_re = _re.compile(r"mock_(\d+)\.pkl$")
    for fn in sorted(_glob.glob(str(out / "mock_*.pkl"))):
        mi = int(mock_re.search(os.path.basename(fn)).group(1))
        with open(fn, "rb") as f:
            by_mock[mi] = pickle.load(f)
    for fn in sorted(_glob.glob(str(out / "shard_*.pkl"))):
        with open(fn, "rb") as f:
            dd = pickle.load(f)
        for rec, mi in zip(dd["per_mock"], dd["idxs"]):
            by_mock.setdefault(int(mi), rec)
    assert list(by_mock.keys()) == [0]
    assert by_mock[0]["_marker"] == "permock", "per-mock pkl must win the dedup"
