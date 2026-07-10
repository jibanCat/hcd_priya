"""load_shards must not silently double-count a mock into the paired data-nuisance bias gate
(PR#14 panel FIX 1). Today ``.extend()``s clean/inj across sorted(glob) pkls with NO use of the
per-pkl ``idxs`` field and no disjointness check, so a mis-specified FILL shard (e.g. a n_shards=8
shard dropped into an n_shards=4 cell dir) could pool the SAME mock twice. It also does not check
that pooled pkls share a single forward stamp (seed + OOS member/strength), so two different
forwards could be silently averaged into one gate cell.
"""
import pickle

import pytest

from scripts.analyze_dnuis_bias import load_shards


def _write_shard(path, *, arm="resolution_oos", treatment="b", survey="desi", idxs,
                 seed=20260615, member="bstar", strength=1.0):
    """A minimal fake shard pkl carrying only the fields load_shards reads."""
    n = len(idxs)
    rec = {"n_div": 0}
    meta = dict(seed=seed,
               inject_spec={"resolution": {"path": "x.npz", "member": member, "strength": strength}})
    with open(path, "wb") as f:
        pickle.dump(dict(arm=arm, treatment=treatment, survey=survey, idxs=list(idxs),
                         clean_per_mock=[rec] * n, inj_per_mock=[rec] * n, meta=meta), f)


def test_load_shards_pools_disjoint_idxs_fine(tmp_path):
    """The real Phase-2 cell layout: a n_shards=4 shard covering idxs=[0,4] plus a n_shards=8 FILL
    shard covering idx=[1] -- disjoint union, pools cleanly to 3 paired mocks."""
    _write_shard(tmp_path / "resolution_oos_b_desi_shard_000.pkl", idxs=[0, 4])
    _write_shard(tmp_path / "resolution_oos_b_desi_shard_001.pkl", idxs=[1])
    groups = load_shards(str(tmp_path))
    cl, inj, nd, om = groups[("resolution_oos", "b", "desi")]
    assert len(cl) == 3 and len(inj) == 3


def test_load_shards_raises_on_overlapping_idxs(tmp_path):
    """A MIS-SPECIFIED FILL shard (e.g. an n_shards=8 shard dropped into an n_shards=4 cell dir)
    can silently double-count mock idx 4. load_shards must RAISE, naming the duplicated idx."""
    _write_shard(tmp_path / "resolution_oos_b_desi_shard_000.pkl", idxs=[0, 4])
    _write_shard(tmp_path / "resolution_oos_b_desi_shard_001.pkl", idxs=[4])
    with pytest.raises(SystemExit, match="4"):
        load_shards(str(tmp_path))


def test_load_shards_raises_on_mismatched_forward_stamp(tmp_path):
    """Two pkls in the same (arm,treatment,survey) group but with a DIFFERENT OOS member/strength
    cannot be pooled -- that would silently average two different forwards' bias into one cell."""
    _write_shard(tmp_path / "resolution_oos_b_desi_shard_000.pkl", idxs=[0], member="bstar", strength=1.0)
    _write_shard(tmp_path / "resolution_oos_b_desi_shard_001.pkl", idxs=[1], member="bstar", strength=-1.0)
    with pytest.raises(SystemExit):
        load_shards(str(tmp_path))


def test_load_shards_raises_on_mismatched_seed(tmp_path):
    """Two pkls in the same group with a DIFFERENT seed (a different truth/noise RNG stream)
    cannot be pooled either."""
    _write_shard(tmp_path / "resolution_oos_b_desi_shard_000.pkl", idxs=[0], seed=20260615)
    _write_shard(tmp_path / "resolution_oos_b_desi_shard_001.pkl", idxs=[1], seed=1)
    with pytest.raises(SystemExit):
        load_shards(str(tmp_path))
