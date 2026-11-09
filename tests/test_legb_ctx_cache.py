"""The historical inference context (closure_legb.build_legb_ctx) reads its training cache through the strict loader
unless a historical reason is given, and refuses an ensemble trained on another cache. On the strict (release-cache)
path it needs an explicit delta_c map, never the historical one, and that map reaches every w_c map call of the
context. Synthetic caches, a small untrained network and a synthetic delta_c map; the data legs are never reached (a
stand-in stops the build right after the cache checks)."""
import json
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from hcd_analysis import lineage as L
from hcd_analysis import tau0_cache as TC
from hcd_analysis.emulator import closure_legb as CL
from hcd_analysis.emulator import dndx_wc as DW
from hcd_analysis.emulator import train as T
from hcd_analysis.emulator.data import fit_target_norm, load_cache, load_cache_historical
from hcd_analysis.emulator.model import Emulator
from tests import tau0_fixtures as tf
from tests.emulator._fixture import write_synthetic_cache, write_synthetic_cache_v34


SYNTH_MAP = DW.DeltaCMap(coeffs=np.zeros((4, 3)), z_lo=2.0, z_hi=5.4, label="synthetic test map")


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
        build([setup["m_new"]], cache_path=setup["new"], delta_map=SYNTH_MAP)
    with pytest.raises(TC.CacheLineageError, match="v33.h5"):                  # a pre-lineage cache is refused
        build([setup["m_old"]], cache_path=setup["old"], delta_map=SYNTH_MAP)
    with pytest.raises(TC.CacheLineageError):
        build([setup["m_old"]], cache_path=setup["old"], historical_reason=None, delta_map=SYNTH_MAP)


def test_a_historical_reason_reads_the_historical_cache(setup):
    with pytest.raises(Reached):
        build([setup["m_old"]], cache_path=setup["old"], historical_reason="a test of a historical analysis")
    with pytest.raises(ValueError, match="non-empty reason"):
        build([setup["m_old"]], cache_path=setup["old"], historical_reason="")


def test_members_trained_on_another_cache_are_refused(setup):
    with pytest.raises(ValueError, match="trained on caches"):
        build([setup["m_old"]], cache_path=setup["new"], delta_map=SYNTH_MAP)
    with pytest.raises(ValueError, match="trained on caches"):
        build([setup["m_new"]], cache_path=setup["old"], historical_reason="a test")
    with pytest.raises(ValueError, match="trained on caches"):
        build([setup["m_new"], setup["m_old"]], cache_path=setup["new"], delta_map=SYNTH_MAP)
    meta = json.load(open(setup["m_new"] + ".meta.json"))
    assert meta["cache_sha256"] == L.sha256_file(setup["new"])


# ------------------------------------------------------------------ the delta_c map on the release-cache path
def test_the_release_cache_path_needs_an_explicit_non_historical_delta_map(setup):
    """The strict path pairs a clean cache with the map's delta_c(z); the default (historical, pre-fix) map is
    refused, by identity or by value, as is anything that is not a DeltaCMap."""
    with pytest.raises(ValueError, match="explicit delta_map"):
        build([setup["m_new"]], cache_path=setup["new"])
    with pytest.raises(ValueError, match="explicit delta_map"):
        build([setup["m_new"]], cache_path=setup["new"], delta_map=None)
    with pytest.raises(ValueError, match="HISTORICAL_DELTA_C"):
        build([setup["m_new"]], cache_path=setup["new"], delta_map=DW.HISTORICAL_DELTA_C)
    same_values = DW.DeltaCMap(coeffs=np.array(DW.HISTORICAL_DELTA_C.coeffs), z_lo=2.0, z_hi=5.4, label="renamed")
    with pytest.raises(ValueError, match="HISTORICAL_DELTA_C"):
        build([setup["m_new"]], cache_path=setup["new"], delta_map=same_values)
    with pytest.raises(TypeError, match="DeltaCMap"):
        build([setup["m_new"]], cache_path=setup["new"], delta_map=np.zeros((4, 3)))
    with pytest.raises(Reached):
        build([setup["m_new"]], cache_path=setup["new"], delta_map=SYNTH_MAP)


def test_the_historical_path_takes_no_delta_map(setup):
    with pytest.raises(ValueError, match="historical path"):
        build([setup["m_old"]], cache_path=setup["old"], historical_reason="a test", delta_map=SYNTH_MAP)
    with pytest.raises(Reached):
        build([setup["m_old"]], cache_path=setup["old"], historical_reason="a test", delta_map=None)


def _record_map_calls(monkeypatch):
    calls = []

    def w_c(dndx, Xbar, z, delta_map=None):
        calls.append(("w_c_corrected", delta_map))
        return jax.numpy.full(np.shape(dndx)[:-1] + (4,), 0.25)

    def inverse(alpha, Xbar, z, *, mode="raise", delta_map=None):
        calls.append(("alpha_to_dndx_exact", delta_map))
        return np.full(np.shape(alpha), 0.1)
    monkeypatch.setattr(CL, "w_c_corrected", w_c)
    monkeypatch.setattr(CL, "alpha_to_dndx_exact", inverse)
    return calls


@pytest.mark.parametrize("dm", [None, SYNTH_MAP])
def test_the_ks_reference_passes_its_delta_map_to_every_map_call(monkeypatch, dm):
    calls = _record_map_calls(monkeypatch)
    monkeypatch.setattr(CL, "assert_hcd_pivot_z3", lambda *a, **k: None)
    monkeypatch.setattr(CL, "assert_ks_mapped_pivot", lambda *a, **k: None)
    CL._ks_dndx_reference(np.array([0.1, 0.05, 0.005]), np.array([0.0, 0.0, 0.5]), np.array([2.2, 3.0]),
                          delta_map=dm)
    assert {n for n, _ in calls} == {"w_c_corrected", "alpha_to_dndx_exact"}
    assert all(m is dm for _, m in calls)


def _mapped_ctx(dm):
    return SimpleNamespace(z_global=np.array([2.2, 3.0]), ks_dndx_mapped=True, tau0_pivot_z=3.0,
                           ks_dndx_ref=np.full((2, 3), 0.1), ks_xbar_z=np.array([0.4, 0.6]),
                           ks_dndx_ref_pivot=np.full(3, 0.1), ks_xbar_pivot=0.6, delta_map=dm)


def test_the_mapped_ks_forward_and_its_replay_use_the_contexts_delta_map(monkeypatch):
    import numpyro
    calls = _record_map_calls(monkeypatch)
    with numpyro.handlers.seed(rng_seed=0):
        CL._ks_dndx_sites(_mapped_ctx(SYNTH_MAP))
    assert len(calls) == 2 and all(m is SYNTH_MAP for _, m in calls)
    calls.clear()
    samples = {n: np.zeros(3) for n in ("tau0_amp", "dtau0", "eps_lls", "kappa_lls", "m_sub", "t_sub", "dla_raw",
                                        "t_dla")}
    CL._legb_reconstruct_deterministics(_mapped_ctx(SYNTH_MAP), samples)
    assert len(calls) == 2 and all(m is SYNTH_MAP for _, m in calls)


def test_the_build_passes_its_delta_map_to_the_ks_reference_and_the_context():
    """The two wiring sites the stubbed build above never reaches (they follow the data legs): checked on the source."""
    import ast
    import inspect
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(CL.build_legb_ctx)))
    calls = [c for c in ast.walk(tree) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
             and c.func.id in ("_ks_dndx_reference", "LegBCtx")]
    assert sorted(c.func.id for c in calls) == ["LegBCtx", "_ks_dndx_reference"]
    for c in calls:
        kw = {k.arg: k.value for k in c.keywords}
        assert isinstance(kw.get("delta_map"), ast.Name) and kw["delta_map"].id == "delta_map", c.func.id
