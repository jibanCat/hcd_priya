"""Gate C review blocking test BT-C1 (PU-0056/PU-0057): the C4 / S7 comparison set is complete and checked, never
shrunk silently.

Both analysis scripts used to drop entries with a non-finite residual before matching and to drop unmatched entries
after it, printing (not asserting) the count 7790; defect F1 was excluded by our simulation name. Now: every upstream
(loo row, z) entry outside F1 is matched exactly once, F1 is identified on the upstream side by a pinned parameter
vector, a non-finite residual in any member of a matched entry raises, and the count is pinned."""
import json
import os

import numpy as np
import pytest

from hcd_analysis.emulator import gate_c as G

EVAL_DIR = "/nfs/turbo/umor-yueyingn/mfho/hcd/emulator_v2/gateC_eval"
UP_LOO = "/home/mfho/lya_emulator_full/kodiaq_2_2_4_6-48-48/loo_fps.hdf5"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = f"{REPO}/hcd_analysis/_emulator_data/observables_tau0_lf.h5"


# ----------------------------------------------------------------------------------------------- synthetic upstream
P0 = np.linspace(0.9, 1.7, 9)
P1 = np.linspace(1.1, 1.9, 9)
RUNGS = (0.7, 1.0)
ZOUT = np.array([3.0, 2.2])


def _upstream():
    sims = (P0, P1, G.F1_PARAMS)
    up = np.array([[a, *p] for p in sims for a in RUNGS])          # sim-major, as upstream's loo_fps
    return up, ZOUT


def _ours(drop=(), dup=(), nan=()):
    """Our held-out entries: every (sim, rung, z) of upstream plus an extra z and an odd rung that upstream lacks."""
    entries, res = [], {}
    for s, p in enumerate((P0, P1, G.F1_PARAMS)):
        for a in (*RUNGS, 0.85):
            for z in (3.0, 2.2, 2.0):
                key = (s, a, z)
                if key in drop:
                    continue
                entries.append(dict(params=p, alpha=a, z=z, key=key))
                res[key] = np.full((2, 3), 0.01)
                if key in nan:
                    res[key][1, 2] = np.nan
                if key in dup:
                    entries.append(dict(params=p, alpha=a, z=z, key=key + ("dup",)))
                    res[key + ("dup",)] = np.full((2, 3), 0.01)
    return entries, res


def test_complete_set_excludes_exactly_the_pinned_f1_entries():
    up, zout = _upstream()
    entries, res = _ours()
    out = G.c4_matched_set(entries, res, up, zout, n_expected=10)
    assert len(out) == 10
    assert {ue for _, ue in out}.isdisjoint({(4, 1), (5, 1)})       # F1 = sim 2 (rows 4, 5) at z 2.2
    assert G.c4_f1_entries(up, zout, n_rungs=2) == {(4, 1), (5, 1)}


def test_an_unmatched_upstream_entry_raises():
    up, zout = _upstream()
    entries, res = _ours(drop={(1, 1.0, 3.0)})
    with pytest.raises(ValueError, match="unmatched"):
        G.c4_matched_set(entries, res, up, zout, n_expected=9)


def test_an_upstream_entry_matched_twice_raises():
    up, zout = _upstream()
    entries, res = _ours(dup={(0, 0.7, 3.0)})
    with pytest.raises(ValueError, match="more than once"):
        G.c4_matched_set(entries, res, up, zout, n_expected=10)


def test_a_non_finite_member_residual_of_a_matched_entry_raises():
    up, zout = _upstream()
    entries, res = _ours(nan={(0, 1.0, 2.2)})
    with pytest.raises(ValueError, match="non-finite"):
        G.c4_matched_set(entries, res, up, zout, n_expected=10)


def test_non_finite_residuals_outside_the_compared_set_are_allowed():
    up, zout = _upstream()
    entries, res = _ours(nan={(0, 1.0, 2.0), (0, 0.85, 3.0), (2, 0.7, 2.2)})   # extra z, odd rung, F1
    assert len(G.c4_matched_set(entries, res, up, zout, n_expected=10)) == 10


def test_f1_is_identified_by_its_pinned_parameters_not_assumed():
    up, zout = _upstream()
    entries, res = _ours()
    up_wrong = up.copy()
    up_wrong[4:6, 1] *= 1.001                                      # upstream no longer holds the pinned vector
    with pytest.raises(ValueError, match="F1"):
        G.c4_matched_set(entries, res, up_wrong, zout, n_expected=10)


def test_the_count_is_pinned():
    up, zout = _upstream()
    entries, res = _ours()
    with pytest.raises(ValueError, match="expected 11"):
        G.c4_matched_set(entries, res, up, zout, n_expected=11)


def test_collect_keeps_every_row_and_refuses_disagreeing_members():
    def ev(rows, res):
        n = len(rows)
        return dict(rows=np.asarray(rows), k_ks=np.array([0.01, 0.02]), res_ks=np.asarray(res, float),
                    sim_name=np.array(["simA"] * n), alpha=np.full(n, 1.0), z=np.array([3.0, 2.2][:n]))
    sim_params = {"simA": P0}
    a = ev([0, 1], [[0.01, np.nan], [0.02, 0.03]])
    b = ev([0, 1], [[0.02, 0.01], [0.01, 0.01]])
    entries, res, k = G.c4_collect([[a, b]], sim_params)
    assert [e["key"] for e in entries] == [(0, 0), (0, 1)]           # the NaN row is kept for the matched-set check
    assert res[(0, 0)].shape == (2, 2) and np.isnan(res[(0, 0)][0, 1])
    assert np.array_equal(k, [0.01, 0.02])
    with pytest.raises(ValueError, match="disagree"):
        G.c4_collect([[a, ev([0, 2], [[0, 0], [0, 0]])]], sim_params)


# ----------------------------------------------------------------------------------------------- real gate C products
def _unavailable(msg):
    if os.environ.get("HCD_GATE_RUN") == "1":
        pytest.fail(msg + " (HCD_GATE_RUN=1)")
    pytest.skip(msg)


def _member_files(n, members):
    files = [f"{EVAL_DIR}/eval_loo60_s{n:02d}.npz"]
    files += [f"{EVAL_DIR}/loo60_ensemble/eval_loo60_s{n:02d}_seed{s}.npz" for s in range(1, members)]
    return files


@pytest.mark.parametrize("members", [1, 5], ids=["C4_single_member", "S7_ensemble"])
def test_c4_matched_set_is_complete(members):
    import h5py
    for p in (UP_LOO, CACHE, *_member_files(59, members)):
        if not os.path.exists(p):
            _unavailable(f"gate C product absent: {p}")
    with h5py.File(CACHE, "r") as f:
        sim_params = dict(zip(f["sim_name"].asstr()[...], f["params"][...]))
    with h5py.File(UP_LOO, "r") as f:
        up_params, up_zout = f["params"][...], f["zout"][...]

    def evals():
        for n in range(60):
            yield [dict(np.load(p, allow_pickle=True)) for p in _member_files(n, members)]
    entries, res, _ = G.c4_collect(evals(), sim_params)
    out = G.c4_matched_set(entries, res, up_params, up_zout)
    assert len(out) == G.C4_N_ENTRIES == 7790
    assert G.c4_f1_entries(up_params, up_zout) == {(r, 12) for r in range(470, 480)}   # upstream sim index 47, z 2.2
    assert all(np.isfinite(res[k]).all() for k, _ in out)
    j = {1: ("gate_c_results.json", lambda d: d["criteria"]["C4"]["n_entries"]),
         5: ("s7_ensemble_loo.json", lambda d: d["n_entries"])}[members]
    assert j[1](json.load(open(f"{EVAL_DIR}/{j[0]}"))) == 7790
