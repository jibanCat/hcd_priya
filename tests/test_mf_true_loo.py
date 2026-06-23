"""True leave-one-out for the HF-LOSO MF correction (plan Task 1.5).

PROBLEM (the bug these tests pin):
    The HF-LOSO MF n_s cert is supposed to leave exactly ONE HR sim out, re-fit the
    MF resolution correction without it, and test recovery on that sim's real HR truth.
    But the held-out set is derived from the LF k-fold GROUP: ``held_out_sims(d, fold)``
    returns the WHOLE LF fold group (``np.array_split`` of the 60 LF sims into 8 groups),
    which is then intersected with the 6 HR sims. Because two HR sims can land in the
    same LF fold group, the cert silently does LEAVE-TWO-OUT for those sims:

        fold 2  ->  {ns0.859, ns0.885}   (both dropped together)
        fold 6  ->  {ns0.972, ns0.979}   (both dropped together)

    So the MF correction for those folds is fit on 4 HR sims instead of 5 -> a worse
    correction -> an inflated apparent bias (the +3.26sigma was ~1.7x inflated by this).

FIX (Task 1.5):
    Thread a ``target_hr_sim`` through so the held HR set is EXACTLY {target_hr_sim}
    (drop only the one sim under test), independent of the LF fold group.

CONTRACT for the CS partner (the implementation these tests require):
    * New module-level PURE helper in ``hcd_analysis/emulator/closure_legb.py``:

          def held_hr_set(d, hr_sim_names, fold=0, target_hr_sim=None) -> set[str]

      - ``hr_sim_names``: the HR cache sim-name array (``hr_cache["sim_name"]``;
        bytes or str entries both accepted; decode internally).
      - ``target_hr_sim is None``  -> BACK-COMPAT: the whole-group held set, i.e.
            {s for s in held_out_sims(d, fold)[0] if s in set(decoded hr_sim_names)}
        (byte-for-byte the set ``build_mf_correction`` computes today).
      - ``target_hr_sim is not None`` -> TRUE LOO: returns exactly {target_hr_sim}
        (i.e. drop ONLY that sim). ``target_hr_sim`` must be one of the HR sims.

    * ``build_mf_correction(..., target_hr_sim=None)`` gains the kwarg and, in its
      ``exclude_held_hr`` block, computes
          held = held_hr_set(d, hr_cache["sim_name"], fold, target_hr_sim)
      replacing the current inline whole-group intersection (lines ~461-470). The
      ``train_rows`` mask (rows whose HR sim is NOT in ``held``) is then exactly the
      complement of ``held`` -- so dropping only the target retains 5 HR sims, not 4.

    * ``build_legb_ctx`` threads ``mf_target_hr_sim`` -> ``build_mf_correction(...,
      target_hr_sim=mf_target_hr_sim)``; ``scripts/run_stepA.py`` (the HFLOSO cert
      driver) passes the per-arm HR sim under test so each arm is genuine LOO.

    The tests below inspect the held set via ``held_hr_set`` (the clean contract
    point); the train-retained HR set is its complement against the 6 HR sims.

These tests are EXPECTED TO FAIL before the fix:
    - ``held_hr_set`` does not exist yet  -> ImportError/AttributeError, and
    - even reconstructing today's behavior (whole-group intersection) yields a
      held set of size 2 for fold 2 (leave-two-out), so the target-only assertion
      cannot hold under current code.
"""
import numpy as np
import pytest

from hcd_analysis.emulator import multifidelity as MF
from hcd_analysis.emulator.data import load_cache
from hcd_analysis.emulator.closure_legb import CACHE_PATH, held_out_sims


# --- The two HR sims that SHARE an LF fold group (the leave-two-out cases). ------
# Resolved at collection time from the real caches so the test stays faithful to
# the actual sim names / split (full names; we match by the ns-prefix for clarity).
def _decode(a):
    return [s.decode() if isinstance(s, bytes) else str(s) for s in a]


@pytest.fixture(scope="module")
def caches():
    d = load_cache(CACHE_PATH)
    hr_cache = MF.load_cache(MF.HR_CACHE)
    hr_sim_names = hr_cache["sim_name"]
    hr_sims = sorted(set(_decode(hr_sim_names)))
    return d, hr_sim_names, hr_sims


def _full_name(hr_sims, ns_prefix):
    """The single full HR sim-name beginning with ``ns_prefix`` (e.g. 'ns0.885')."""
    hits = [s for s in hr_sims if s.startswith(ns_prefix)]
    assert len(hits) == 1, f"expected one HR sim for {ns_prefix!r}, got {hits}"
    return hits[0]


@pytest.fixture(scope="module")
def leave_two_out(caches):
    """Find an LF fold whose held-out group contains >=2 HR sims, and return one
    target sim + its fold-mate (both HR). This is the configuration the bug
    corrupts: today both are dropped; after the fix only the target is dropped."""
    d, _hr_names, hr_sims = caches
    hrset = set(hr_sims)
    n_folds = 8
    for fold in range(n_folds):
        sims, _ = held_out_sims(d, fold=fold)
        held_hr = [s for s in _decode(sims) if s in hrset]
        if len(held_hr) >= 2:
            target, fold_mate = held_hr[0], held_hr[1]
            return dict(fold=fold, target=target, fold_mate=fold_mate,
                        group_held=held_hr)
    pytest.skip("no LF fold groups a pair of HR sims (no leave-two-out case)")


def _train_retained(held, hr_sims):
    """The HR sims RETAINED in the MF head fit = the 6 HR sims minus the held set."""
    return set(hr_sims) - set(held)


# --------------------------------------------------------------------------- #
#  TEST 1: true LOO drops ONLY the target sim (retains its fold-mate -> 5, not 4)
# --------------------------------------------------------------------------- #
def test_true_loo_excludes_only_target(caches, leave_two_out):
    """With ``target_hr_sim`` set to an HR sim that SHARES a fold group with another
    HR sim, the MF train HR set must exclude EXACTLY that one sim and RETAIN its
    fold-mate: held == {target_hr_sim}; 5 HR sims retained, not 4."""
    from hcd_analysis.emulator.closure_legb import held_hr_set  # contract point

    d, hr_sim_names, hr_sims = caches
    fold = leave_two_out["fold"]
    target = leave_two_out["target"]
    fold_mate = leave_two_out["fold_mate"]

    # Sanity on the fixture: target and fold-mate really share this fold group today.
    assert {target, fold_mate} <= set(leave_two_out["group_held"])

    held = set(held_hr_set(d, hr_sim_names, fold=fold, target_hr_sim=target))

    # The held set is EXACTLY the one target sim -- not the whole group.
    assert held == {target}, (
        f"true-LOO must drop only the target; got held={sorted(held)}")

    # The fold-mate is RETAINED (the leave-two-out -> leave-one-out fix).
    retained = _train_retained(held, hr_sims)
    assert fold_mate in retained, (
        f"fold-mate {fold_mate!r} must be retained in the MF train set")

    # 5 HR sims retained for the head fit, not 4.
    assert len(retained) == len(hr_sims) - 1 == 5, (
        f"expected 5 retained HR sims, got {len(retained)}: {sorted(retained)}")


# --------------------------------------------------------------------------- #
#  TEST 2: back-compat -- target_hr_sim=None reproduces today's whole-group held
# --------------------------------------------------------------------------- #
def test_default_unchanged(caches, leave_two_out):
    """With ``target_hr_sim=None`` (default) the held set is the SAME as today: the
    whole LF fold group intersected with the HR sims. For a leave-two-out fold that
    is a 2-element set -- the current (buggy) behavior, preserved for back-compat."""
    from hcd_analysis.emulator.closure_legb import held_hr_set  # contract point

    d, hr_sim_names, hr_sims = caches
    fold = leave_two_out["fold"]
    hrset = set(hr_sims)

    # Today's behavior, computed directly from the public split helper.
    sims, _ = held_out_sims(d, fold=fold)
    expected_whole_group = set(s for s in _decode(sims) if s in hrset)

    held_default = set(held_hr_set(d, hr_sim_names, fold=fold, target_hr_sim=None))

    assert held_default == expected_whole_group, (
        f"default must reproduce the whole-group held set "
        f"{sorted(expected_whole_group)}; got {sorted(held_default)}")

    # And for this leave-two-out fold that whole group is the 2-element set
    # (the very behavior the cert misuses) -- a guard that the back-compat path
    # is genuinely the old, group-based one.
    assert len(held_default) >= 2, (
        "back-compat default on a leave-two-out fold must hold >=2 HR sims")
