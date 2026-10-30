"""The historical inference context (closure_legb.build_legb_ctx) reads its training cache through the strict loader
unless a historical reason is given, and refuses an ensemble trained on another cache. Synthetic caches and a small
untrained network; the data legs are never reached (a stand-in stops the build right after the cache checks)."""
import json

import jax
import numpy as np
import pytest

from hcd_analysis import lineage as L
from hcd_analysis import tau0_cache as TC
from hcd_analysis.emulator import closure_legb as CL
from hcd_analysis.emulator import train as T
from hcd_analysis.emulator.data import fit_target_norm, load_cache, load_cache_historical
from hcd_analysis.emulator.model import Emulator
from tests import tau0_fixtures as tf
from tests.emulator._fixture import write_synthetic_cache, write_synthetic_cache_v34


class Reached(Exception):
    """The build got past the cache checks."""


@pytest.fixture
def setup(tmp_path, monkeypatch):
    tf.clean_code(monkeypatch)
    new, old = tmp_path / "v34.h5", tmp_path / "v33.h5"
    write_synthetic_cache_v34(new, n_sims=3, snaps_per_sim=2)
    write_synthetic_cache(old, n_sims=3, snaps_per_sim=2)

    def member(name, d, path):
        model = Emulator(in_dim=10, n_k=8, n_basis=4, baseline_n_layers=1, baseline_width=16,
                         key=jax.random.PRNGKey(0))
        T.save_checkpoint(str(tmp_path / name), model, {}, fit_target_norm(d, np.arange(d["P_tier_p"].shape[0])),
                          seed=0, cache=d, cache_path=str(path))
        return str(tmp_path / name)

    def stop(*a, **k):
        raise Reached()
    monkeypatch.setattr(CL.DL, "load_desi_leg", stop)
    return {"new": new, "old": old, "m_new": member("m_new", load_cache(new), new),
            "m_old": member("m_old", load_cache_historical(old, reason="test"), old)}


def build(ckpts, **kw):
    return CL.build_legb_ctx(ensemble_ckpts=ckpts, ks_kwargs={"k_max": 0.065}, **kw)


def test_the_context_reads_the_cache_strictly_by_default(setup):
    with pytest.raises(Reached):
        build([setup["m_new"]], cache_path=setup["new"])
    with pytest.raises(TC.CacheLineageError, match="v33.h5"):                  # a pre-lineage cache is refused
        build([setup["m_old"]], cache_path=setup["old"])
    with pytest.raises(TC.CacheLineageError):
        build([setup["m_old"]], cache_path=setup["old"], historical_reason=None)


def test_a_historical_reason_reads_the_historical_cache(setup):
    with pytest.raises(Reached):
        build([setup["m_old"]], cache_path=setup["old"], historical_reason="a test of a historical analysis")
    with pytest.raises(ValueError, match="non-empty reason"):
        build([setup["m_old"]], cache_path=setup["old"], historical_reason="")


def test_members_trained_on_another_cache_are_refused(setup):
    with pytest.raises(ValueError, match="trained on caches"):
        build([setup["m_old"]], cache_path=setup["new"])
    with pytest.raises(ValueError, match="trained on caches"):
        build([setup["m_new"]], cache_path=setup["old"], historical_reason="a test")
    with pytest.raises(ValueError, match="trained on caches"):
        build([setup["m_new"], setup["m_old"]], cache_path=setup["new"])
    meta = json.load(open(setup["m_new"] + ".meta.json"))
    assert meta["cache_sha256"] == L.sha256_file(setup["new"])
