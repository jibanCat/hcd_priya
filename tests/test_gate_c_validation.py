"""Gate C building blocks (spec hcd_priya_notes docs/superpowers/emulator-debug-2026-10/GATE_C_SPEC.md): the production
row split with an optional held-out simulation (upstream-matched 60-fold protocol), the coordinate-agreement check (C1),
interpolation onto a fixed physical grid that refuses extrapolation, and matching of our held-out predictions to the
upstream leave-one-out product (C4)."""
import numpy as np
import pytest

from hcd_analysis.emulator import data as D
from hcd_analysis.emulator import gate_c as G


def _names():
    return np.array([f"sim{i:02d}" for i in range(6) for _ in range(20)])


def test_production_split_without_holdout_is_the_historical_inline_split():
    names = _names()
    tr, va, held = D.production_row_split(names, val_seed=12345, val_frac=0.10)
    perm = np.random.default_rng(12345).permutation(names.size)        # train_production_emulator.py inline split
    n_val = int(0.10 * names.size)
    assert np.array_equal(va, np.sort(perm[:n_val])) and np.array_equal(tr, np.sort(perm[n_val:]))
    assert held.size == 0


def test_production_split_with_a_held_out_simulation():
    names = _names()
    tr, va, held = D.production_row_split(names, val_seed=12345, val_frac=0.10, holdout_sim="sim03")
    assert np.array_equal(held, np.where(names == "sim03")[0])
    assert not np.isin(names[tr], ["sim03"]).any() and not np.isin(names[va], ["sim03"]).any()
    assert np.intersect1d(tr, va).size == 0 and np.union1d(np.union1d(tr, va), held).size == names.size
    assert va.size == int(0.10 * (names.size - held.size))


def test_production_split_refuses_an_unknown_simulation():
    with pytest.raises(ValueError, match="not in the cache"):
        D.production_row_split(_names(), holdout_sim="simXX")


def test_coordinate_agreement_ignores_padding_and_reports_the_worst_mode():
    k_row = np.array([1.0, 2.0, 3.0, np.nan])
    assert G.coordinate_agreement(k_row * (1 + 1e-14), k_row) < 2e-14
    assert abs(G.coordinate_agreement(np.array([1.0, 2.0, 3.0 * (1 + 1e-4), 9.0]), k_row) - 1e-4) < 1e-12


def test_interp_to_grid_matches_linear_and_refuses_extrapolation():
    k = np.array([1.0, 2.0, 3.0])
    P = np.array([10.0, 20.0, 40.0])
    assert np.allclose(G.interp_to_grid(np.array([1.5, 2.5]), k, P), [15.0, 30.0])
    with pytest.raises(ValueError, match="outside"):
        G.interp_to_grid(np.array([0.5]), k, P)


def test_match_upstream_loo_pairs_entries_by_parameters_rung_and_redshift():
    up_params = np.array([[0.7, 0.90, 1.5e-9], [0.8, 0.90, 1.5e-9], [0.7, 0.95, 2.0e-9]])
    up_zout = np.array([3.0, 2.8])
    ours = [dict(params=np.array([0.90, 1.5e-9]), alpha=0.8, z=2.8, key="a"),
            dict(params=np.array([0.95, 2.0e-9]), alpha=0.7, z=3.0, key="b"),
            dict(params=np.array([0.99, 2.0e-9]), alpha=0.7, z=3.0, key="c")]
    got = G.match_upstream_loo(ours, up_params, up_zout)
    assert got == {"a": (1, 1), "b": (2, 0)}
