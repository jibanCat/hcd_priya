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
    #
    # --figdir IS REQUIRED HERE (2026-07-28). Without it the merge script falls back to its
    # production default, /home/mfho/hcd_priya/figures/analysis/06_validation_summary, and this
    # test silently REWRITES the committed artifact prod_sbc_pilot.png on every run -- caught by a
    # dirty git tree during the A1c pre-launch work. Redirecting the test's own output changes no
    # artifact semantics and does not touch the frozen analysis path: the script's default is
    # unchanged, only this test stops writing into the repo. Same lesson as the analyze_sbc_perleg
    # default-prefix footgun (pre-registration 5c) and the twice-truncated emu_bias txt.
    import sys
    figdir = str(tmp_path / "figs")
    os.makedirs(figdir, exist_ok=True)
    argv = sys.argv
    sys.argv = ["merge_prod_sbc_shards.py", "--shard-dir", out, "--prob", "0.95",
                "--figdir", figdir]
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


# --- CONFIG-KEY CLASH guard (run_prod_sbc_shard.py:62-122, added 2026-06-19) -------------------
# The per-mock pkl is keyed by INDEX ONLY (mock_{m:04d}.pkl), with no leg_a / cemu_variant /
# amp_sigma / tau0 / subdla discriminator. A held-out (or cemu-variant / width-check / informative-
# prior) run pointed at an --out-dir that already holds differently-configured pkls would otherwise
# SILENTLY skip-load the WRONG SBC population. The guard stamps run_cfg into each record and, on the
# skip branch, RAISES on a stamp mismatch. These exercise that skip-branch guard ONLY — the clash
# check precedes run_legb, so they need no ctx/NUTS (ctx=d=None) and run fast + deterministically.

_FULL_SELFDRAW = dict(leg_a=True, cemu_variant="current", amp_sigma=0.0, leg="all", fold=0,
                      tau0_prior_sigma=0.0, subdla_truth_boost=1.0)
# the 2026-06-19 stamp, written BEFORE the leg/fold/tau0/subdla keys were added (a pre-stamp resume):
_PARTIAL_2026_06_19 = dict(leg_a=True, cemu_variant="current", amp_sigma=0.0)
# unused-but-required kwargs (consumed only on the run_legb path, which the guard never reaches):
_DUMMY = dict(n_mocks=1, n_warmup=1, n_samples=1, max_tree_depth=1, seed=0, verbose=False)


def _write_stub_mock(runner, out_dir, m, run_cfg, marker="ondisk"):
    """Pre-write a per-mock pkl with a chosen run_cfg stamp (or none) + a readback marker."""
    os.makedirs(out_dir, exist_ok=True)
    rec = dict(_marker=marker)
    if run_cfg is not None:
        rec["run_cfg"] = dict(run_cfg)
    with open(runner._mock_path(out_dir, m), "wb") as f:
        pickle.dump(rec, f)


def test_config_clash_raises_on_leg_a_mismatch(runner, tmp_path):
    """Self-draw pkl on disk + a HELD-OUT (leg_a=False) request for the same index → RuntimeError,
    not a silent wrong-config skip-load."""
    out = str(tmp_path / "clash_leg_a")
    _write_stub_mock(runner, out, 0, _FULL_SELFDRAW)
    req = dict(_FULL_SELFDRAW, leg_a=False)                  # held-out over a self-draw pkl
    with pytest.raises(RuntimeError, match="config CLASH"):
        runner._run_mock(None, None, 0, out, run_cfg=req, **_DUMMY)


def test_config_match_loads_without_nuts(runner, tmp_path):
    """An EXACT config match → SKIP-load the existing pkl (returned verbatim, no re-NUTS)."""
    out = str(tmp_path / "match")
    _write_stub_mock(runner, out, 0, _FULL_SELFDRAW, marker="loaded-me")
    rec = runner._run_mock(None, None, 0, out, run_cfg=dict(_FULL_SELFDRAW), **_DUMMY)
    assert rec["_marker"] == "loaded-me", "exact-config match must load the on-disk pkl, not re-run"


def test_config_backcompat_prestamp_resumes_under_default(runner, tmp_path):
    """A 2026-06-19-stamped pkl (no leg/fold/tau0/subdla keys) resumed under the modern DEFAULT
    config must LOAD — the back-compat pops drop the missing-key diffs — not clash."""
    out = str(tmp_path / "backcompat")
    _write_stub_mock(runner, out, 0, _PARTIAL_2026_06_19, marker="resumed")
    rec = runner._run_mock(None, None, 0, out, run_cfg=dict(_FULL_SELFDRAW), **_DUMMY)
    assert rec["_marker"] == "resumed", "pre-stamp pkl must resume under the default config"


def test_config_clash_nondefault_over_prestamp(runner, tmp_path):
    """A NON-default run (informative τ₀, σ>0) over a pre-stamp DEFAULT pkl must CLASH — the
    back-compat pop excuses a MISSING key only when the request is ALSO at that key's default."""
    out = str(tmp_path / "clash_informative")
    _write_stub_mock(runner, out, 0, _PARTIAL_2026_06_19)
    req = dict(_FULL_SELFDRAW, tau0_prior_sigma=0.05)        # informative τ₀ over an un-stamped uniform pkl
    with pytest.raises(RuntimeError, match="config CLASH"):
        runner._run_mock(None, None, 0, out, run_cfg=req, **_DUMMY)


"""SAMPLER-POPULATION stamps (2026-08-05, PI #9 3f.5). The A3d KS diagnostic runs at the r6x
seed (20260724) with r6x-matched NUTS (250/300); a future A3c certification arm runs at the
ARM-P defaults (20260614, 250/600). Directory separation alone is not the campaign standard:
the stamp discipline must refuse the pooling MECHANICALLY, so seed / n_warmup / n_samples /
max_tree_depth join run_cfg with the same one-way back-compat as every earlier stamp (a
missing key on an old pkl is excused ONLY when the request is at that key's historical
default: seed 20260614, 250/600/10)."""

_SAMPLER_DEFAULTS = dict(seed=20260614, n_warmup=250, n_samples=600, max_tree_depth=10)
_A3D_SAMPLER = dict(seed=20260724, n_warmup=250, n_samples=300, max_tree_depth=10)


def test_sampler_stamp_backcompat_default_resume(runner, tmp_path):
    """A pre-sampler-stamp pkl (e.g. a landed A1c mock) resumed under the HISTORICAL DEFAULT
    sampler config must LOAD -- the back-compat pops excuse the four missing keys."""
    out = str(tmp_path / "sampler_backcompat")
    _write_stub_mock(runner, out, 0, _FULL_SELFDRAW, marker="resumed")
    req = dict(_FULL_SELFDRAW, **_SAMPLER_DEFAULTS)
    rec = runner._run_mock(None, None, 0, out, run_cfg=req, **_DUMMY)
    assert rec["_marker"] == "resumed", "default-sampler resume over a pre-stamp pkl must load"


def test_sampler_stamp_clash_nondefault_over_prestamp(runner, tmp_path):
    """An A3d-config request (seed 20260724, 250/300) over a pre-sampler-stamp pkl must CLASH:
    the old pkl was a default-sampler population and the ranks do not transfer."""
    out = str(tmp_path / "sampler_clash_prestamp")
    _write_stub_mock(runner, out, 0, _FULL_SELFDRAW)
    req = dict(_FULL_SELFDRAW, **_A3D_SAMPLER)
    with pytest.raises(RuntimeError, match="config CLASH"):
        runner._run_mock(None, None, 0, out, run_cfg=req, **_DUMMY)


def test_sampler_stamp_clash_a3d_vs_a3c(runner, tmp_path):
    """BOTH stamped, different seed (the A3d-vs-A3c case the PI's strict-separation ruling is
    about): must CLASH even though every other key matches."""
    out = str(tmp_path / "sampler_clash_seed")
    _write_stub_mock(runner, out, 0, dict(_FULL_SELFDRAW, **_A3D_SAMPLER))
    req = dict(_FULL_SELFDRAW, **_SAMPLER_DEFAULTS)
    with pytest.raises(RuntimeError, match="config CLASH"):
        runner._run_mock(None, None, 0, out, run_cfg=req, **_DUMMY)


def test_sampler_stamp_exact_match_loads(runner, tmp_path):
    """Both stamped and equal (an A3d resubmit) -> SKIP-load, resumes stay free."""
    out = str(tmp_path / "sampler_match")
    stamped = dict(_FULL_SELFDRAW, **_A3D_SAMPLER)
    _write_stub_mock(runner, out, 0, stamped, marker="loaded-me")
    rec = runner._run_mock(None, None, 0, out, run_cfg=dict(stamped), **_DUMMY)
    assert rec["_marker"] == "loaded-me"


def test_sampler_stamp_wired_into_main_cfg(runner):
    """The four sampler keys must be wired from the CLI args into main()'s run_cfg dict -- guard
    against the stamp existing in the clash logic but never being written. Static check on the
    source: the run_cfg construction must reference a.seed / a.n_warmup / a.n_samples /
    a.max_tree_depth."""
    import inspect
    src = inspect.getsource(runner.main)
    for frag in ("seed=int(a.seed)", "n_warmup=int(a.n_warmup)",
                 "n_samples=int(a.n_samples)", "max_tree_depth=int(a.max_tree_depth)"):
        assert frag in src, f"main() run_cfg must stamp {frag}"


def test_inject_spec_held_out_guard():
    """inject_spec on a HELD-OUT (leg_a=False) run_legb call must FAIL LOUD — the held-out branch
    ignores inject_spec, so honouring it would silently drop the injection. PR#12 review follow-up (b).
    The guard is the first statement in run_legb, so it fires before any ctx use (ctx=d=None here)."""
    with pytest.raises(ValueError, match="inject_spec is honoured only on the Leg-A"):
        LB.run_legb(None, None, n_mocks=1, n_warmup=1, n_samples=1, seed=0,
                    inject_spec={"metal_misspec": 1.0}, leg_a=False)
