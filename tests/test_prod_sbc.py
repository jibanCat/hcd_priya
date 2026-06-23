"""Phase-C — production-ensemble SBC wiring: ensemble ctx + SLURM-shardable Leg-A driver.

Pins the contract for the production-SBC gate plumbing:
  (a) build_ctx(ensemble_ckpts=[...]) carries an EnsembleEmulator; the single-ckpt path is
      unchanged (no .members);
  (b) the per-mock draw is a PURE function of (key0, m) via fold_in — so a SLURM shard
      (a subset of mock indices) reproduces EXACTLY the mocks a single full run would draw
      (the Talts+2018 validity requirement for sharded SBC);
  (c) aggregate_leg_a merges per-mock records from many shards into one common-L_eff rank
      set (the cross-shard merge), with ranks shape (n_kept, P+1) and L_eff = min over mocks;
  (d) run_leg_a_sbc(return_per_mock=True, mock_indices=…) returns the raw per-mock records
      for the merge, and the selected index draws the fold_in(seed, m) mock.

Env-gated like the other closure tests. Run:
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu \
    /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_prod_sbc.py -q
"""
import glob
import os

import hcd_analysis.emulator  # noqa: F401  enables x64
import jax
import numpy as np
import pytest

from hcd_analysis.emulator import closure_sbc as S
from hcd_analysis.emulator.ensemble import EnsembleEmulator

REPO = "/home/mfho/hcd_priya"
PROD = f"{REPO}/checkpoints/final_prod_seed"
CKPT = f"{REPO}/checkpoints/final_fold0"
EV = f"{REPO}/checkpoints/error_vector.npz"
_have = os.path.exists(CKPT + ".eqx") and os.path.exists(EV)
_have_prod = len(glob.glob(PROD + "*.eqx")) >= 2
gate = pytest.mark.skipif(not _have, reason="final_fold0 / error_vector absent")
prodgate = pytest.mark.skipif(not (_have and _have_prod),
                              reason="production ensemble / error_vector absent")


@pytest.fixture(scope="module")
def ctx1():
    return S.build_ctx(n_z=1, seed=0)


# ---------------------------------------------------------------------------
# (a) ensemble threading into build_ctx
# ---------------------------------------------------------------------------
@prodgate
def test_build_ctx_ensemble_uses_ensemble_model():
    paths = sorted(p[:-4] for p in glob.glob(PROD + "*.eqx"))
    ctx = S.build_ctx(n_z=1, seed=0, ensemble_ckpts=paths)
    assert isinstance(ctx.model, EnsembleEmulator)
    assert len(ctx.model.members) == len(paths) >= 2


@gate
def test_build_ctx_single_model_path_unchanged(ctx1):
    assert getattr(ctx1.model, "members", None) is None


# ---------------------------------------------------------------------------
# (b) per-mock draw is a pure function of (key0, m) — shard reproducibility
# ---------------------------------------------------------------------------
@gate
def test_mock_for_index_deterministic_and_shard_invariant(ctx1):
    key0 = jax.random.PRNGKey(7)
    tv_a = np.asarray(S._mock_for_index(ctx1, key0, 1)[2])
    tv_b = np.asarray(S._mock_for_index(ctx1, key0, 1)[2])
    assert np.array_equal(tv_a, tv_b), "same (key0,m) must draw an identical mock"
    tv_other = np.asarray(S._mock_for_index(ctx1, key0, 2)[2])
    assert not np.allclose(tv_a, tv_other), "different m must draw a different mock"


# ---------------------------------------------------------------------------
# (c) aggregate_leg_a merges per-mock records (the cross-shard merge) — pure, no NUTS
# ---------------------------------------------------------------------------
def _fake_record(rng, n_z, L):
    P = 9 + n_z + 3
    return dict(truth_vec=rng.normal(size=P), draws=rng.normal(size=(L, P)),
                ll_true=float(rng.normal()), ll_draws=rng.normal(size=L))


def test_aggregate_leg_a_merges_shards():
    rng = np.random.default_rng(0)
    n_z = 1
    P = 9 + n_z + 3
    shardA = [_fake_record(rng, n_z, 120) for _ in range(3)]
    shardB = [_fake_record(rng, n_z, 150) for _ in range(2)]
    res = S.aggregate_leg_a(shardA + shardB, n_z, prob=0.95)
    assert res["ranks"].shape == (5, P + 1), res["ranks"].shape   # 5 kept, P params + loglik
    assert res["L"] == 120                                        # common L_eff = min over mocks
    assert len(res["names"]) == P + 1
    assert res["gate_valid"] == (120 >= S.L_FLOOR)
    # ranks live in {0..L_eff}
    assert res["ranks"].min() >= 0 and res["ranks"].max() <= res["L"]


def test_aggregate_leg_a_empty_is_safe():
    res = S.aggregate_leg_a([], 1, prob=0.95)
    assert res["n_kept"] == 0 and res["L"] == 0 and res["gate_valid"] is False


def test_aggregate_leg_a_uses_record_names_with_metals_column():
    """The leg path with sample_metals appends an a_SiIII column to draws+truth_vec and stores
    `names`; aggregate_leg_a must use those names (NOT param_names(n_z)) — else the a_SiIII column
    is an off-by-one that IndexErrors in sbc_ranks_multiparam."""
    rng = np.random.default_rng(2)
    n_z = 3
    P = 9 + n_z + 3 + 1                       # + a_SiIII (the leg-path metals column)
    names = ([f"p{i}" for i in range(9)] + [f"tau0_z{i}" for i in range(n_z)]
             + ["alpha_lls", "alpha_subdla", "alpha_dla", "a_SiIII"])
    recs = [dict(truth_vec=rng.normal(size=P), draws=rng.normal(size=(120, P)),
                 ll_true=float(rng.normal()), ll_draws=rng.normal(size=120),
                 names=names, n_div=0) for _ in range(4)]
    res = S.aggregate_leg_a(recs, n_z, prob=0.95)            # must NOT raise
    assert res["ranks"].shape == (4, P + 1)
    assert "a_SiIII" in res["names"] and res["names"][-1] == "loglik"


# ---------------------------------------------------------------------------
# (d) run_leg_a_sbc return_per_mock + mock_indices (one tiny NUTS)
# ---------------------------------------------------------------------------
@gate
def test_run_leg_a_sbc_return_per_mock_shards(ctx1):
    kept = S.run_leg_a_sbc(ctx1, n_mocks=2, n_warmup=6, n_samples=6, seed=0,
                           mock_indices=[1], return_per_mock=True, verbose=False)
    assert isinstance(kept, list)
    for r in kept:
        for k in ("truth_vec", "draws", "ll_true", "ll_draws"):
            assert k in r
        # the shard for index 1 must carry the fold_in(seed,1) mock truth
        tv_ref = np.asarray(S._mock_for_index(ctx1, jax.random.PRNGKey(0), 1)[2])
        assert np.array_equal(np.asarray(r["truth_vec"]), tv_ref)
