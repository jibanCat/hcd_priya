"""Threshold-study registry (PI decision 8, TRIGGERED by the X-battery corner failures).

Design pre-registered in notes `2026-07-26-threshold-study-preregistration.md`; this module
is the SINGLE AUTHORITY for its arms and truth arithmetic, exactly as `ks_xsel_arms` is for
the X battery.

WHY A SEPARATE MODULE (binding constraint, do not "simplify" this away): the X-battery
registry payload is hashed into the `registry_signature` stamped on all 88 X-battery pkls,
so adding arms there would make `analyze_xsel` refuse every one of them. This module has its
own CAMPAIGN tag and its own signature; the two campaigns cannot pool by construction. It
READS the stage-V truth tables through `ks_xsel_arms`' PINNED sha and never re-pins them.

The question this study answers: not "is the pure corner biased" (measured: yes on three
channels) but AT WHAT SELECTION CONTAMINATION does the bias reach the 0.30 sigma_post
budget, per channel and per class, so admissible claims can be stated as "unbiased at the
budget for samples with class-c contamination below f*".

Truth construction, per the pre-registration (both forms have EXACT endpoints against arms
that already ran, which the tests assert):

  * class-template directions (LLS, subDLA) -- alpha is the selected-sightline template
    MIXTURE amplitude (PI amendment #6), so a sample in which a fraction f of sightlines are
    class-c-selected and the remaining (1-f) are drawn from the population has weights

        alpha(f) = (1 - f) * alpha_drawn + f * e_class

    NOT `f * e_class`: that naive form interpolates between a ZERO-HCD sample and the
    corner, confounding the selection fraction with the removal of the baseline HCD
    content. f=0 reproduces the drawn truth (the K0 baseline) and f=1 the X2/X3 corner.

  * the masked-DLA direction (X1) is a DATA swap, not an alpha injection, because that is
    precisely the direction the reduced basis omits. The sample-averaged P1D over a
    two-population mixture is P = (1-f) P_clean + f P_swap, and since P_swap = P_clean * R
    in the deployed swap path this is identically P_clean * [1 + f (R - 1)] -- i.e. the
    EXISTING `x1_per_mock` called with `mixture_ratio(f, R)`. No new runner code is needed
    for it. f=0 leaves the leg byte-identical (R -> 1); f=1 reproduces X1.

STATUS: registry + truth arithmetic + tests only. The runner (`run_thresh_shard.py`) and
readout (`analyze_thresh.py`) are OWED, together with the campaign's 4-lens panel review of
the design, before any production fit runs. See the pre-registration for the build plan.
"""
import hashlib
import json

import numpy as np

from scripts.ks_xsel_arms import (  # the PINNED stage-V artifacts, read-only
    GATE, PAIR_SEED, Z_PIVOT, XSEL_TRUTH_TABLE_PATH, XSEL_TRUTH_TABLE_SHA256,
    f_sel, load_truth_tables, mixture_ratio, truth_admissible,
)

CAMPAIGN = "threshold-study-1"
K0_ARM_ID = "K0_clean"           # same reused baseline and pairing as the X battery
K0_N_MOCKS = 16

__all__ = ["CAMPAIGN", "ARMS", "GATE", "PAIR_SEED", "Z_PIVOT", "K0_ARM_ID", "K0_N_MOCKS",
           "fraction_alpha_rows", "profiled_fraction_f_z", "thresh_ratio_rows",
           "arm_ids", "batch_cells", "shard_pkl_name", "registry_signature",
           "campaign_cost_cpuh"]


# ---------------------------------------------------------------------------------------- #
#  Truth arithmetic
# ---------------------------------------------------------------------------------------- #
def fraction_alpha_rows(alpha_drawn_rows, cls_idx, f):
    """Convex mixture of the mock's OWN drawn composition with the pure class-c corner:
    ``(1 - f) * alpha_drawn + f * e_cls`` at every z. (nZ,3) in class order LLS, subDLA, DLA.

    f=0 returns the drawn rows unchanged (the K0 baseline); f=1 returns the corner."""
    assert cls_idx in (0, 1), \
        f"alpha-space fractions exist for LLS (0) / subDLA (1); the DLA direction is " \
        f"data-side (see thresh_ratio_rows), got {cls_idx}"
    f = float(f)
    assert 0.0 <= f <= 1.0, f"selection fraction must lie in [0, 1], got {f}"
    a = np.array(alpha_drawn_rows, float, copy=True)      # never mutate the caller's truth
    assert a.ndim == 2 and a.shape[1] == 3, f"alpha rows must be (nZ,3), got {a.shape}"
    e = np.zeros(3, float)
    e[cls_idx] = 1.0
    return (1.0 - f) * a + f * e[None, :]


def profiled_fraction_f_z(f_mean, z):
    """The z-profiled selection fraction with the X4 inverted-U SHAPE rescaled to a MEAN of
    ``f_mean`` over the given z grid: ``f(z) = f_mean / mean_z(f_sel) * f_sel(z)``.

    This is the arm that separates the z-PROFILE from the mean fraction (the X4-vs-X3
    linearity diagnostic showed n_s responds to the profile, not to the mean), so it must be
    compared against the FLAT arm at the same f_mean."""
    f_mean = float(f_mean)
    assert 0.0 <= f_mean <= 1.0, f"mean selection fraction must lie in [0, 1], got {f_mean}"
    z_arr = np.asarray(z, float)
    shape = np.atleast_1d(f_sel(z_arr))
    f_z = f_mean * shape / float(shape.mean())
    assert np.all(f_z >= 0.0), "profiled fraction went negative (bug)"
    assert np.all(f_z <= 1.0 + 1e-12), (
        f"profiled fraction exceeds 1 at f_mean={f_mean} (peak {f_z.max():.4f}): the shape "
        f"cannot be rescaled to this mean without leaving the simplex")
    return np.minimum(f_z, 1.0)


def profiled_fraction_alpha_rows(alpha_drawn_rows, cls_idx, f_mean, z):
    """Per-z convex mixture at the profiled fraction: ``(1-f_z) alpha_drawn + f_z e_cls``."""
    assert cls_idx in (0, 1), f"profiled fractions are LLS/subDLA only, got {cls_idx}"
    a = np.array(alpha_drawn_rows, float, copy=True)
    f_z = profiled_fraction_f_z(f_mean, z)
    assert a.shape[0] == f_z.size, f"alpha rows {a.shape} vs z grid {f_z.size}"
    e = np.zeros(3, float)
    e[cls_idx] = 1.0
    return (1.0 - f_z)[:, None] * a + f_z[:, None] * e[None, :]


def thresh_ratio_rows(f, ratio_rows_corner):
    """The masked-DLA (X1-direction) fraction arm: the EXACT P1D mixture identity
    ``1 + f (R - 1)`` on the pinned corrected-fork conditional ratio. Feed the result to the
    existing `run_xsel_shard.x1_per_mock` as its ``ratio_rows``; no new data-side code."""
    f = float(f)
    assert 0.0 <= f <= 1.0, f"selection fraction must lie in [0, 1], got {f}"
    out = mixture_ratio(f, np.asarray(ratio_rows_corner, float))
    assert np.all(np.isfinite(out)) and np.all(out > 0.0), \
        "mixed ratio rows non-finite/non-positive (tripwire)"
    return out


# ---------------------------------------------------------------------------------------- #
#  The arm matrix. kind:
#    data_swap_frac    T1/T2: x1_per_mock with ratio rows mixed to fraction f
#    mixture_frac      T3:    flat alpha fraction f
#    mixture_frac_prof T4:    z-profiled alpha at the SAME mean fraction as T3
# ---------------------------------------------------------------------------------------- #
ARMS = {
    "T1_dla010": dict(
        kind="data_swap_frac", cls=2, f=0.10, n_mocks=16, part1=True,
        table_key="ratio_rows_X1_dla100",
        purpose="10% masked-DLA-selected: brackets the naive n_s threshold (0.30/1.688 = "
                "0.178) from BELOW on the direction the reduced basis omits entirely"),
    "T2_dla025": dict(
        kind="data_swap_frac", cls=2, f=0.25, n_mocks=16, part1=True,
        table_key="ratio_rows_X1_dla100",
        purpose="25% masked-DLA-selected: brackets the same naive threshold from ABOVE"),
    "T3_lls030": dict(
        kind="mixture_frac", cls=0, f=0.30, n_mocks=16, part1=True,
        table_key="ratio_rows_X3_lls100",
        purpose="30% LLS-selected, FLAT in z: at the naive A_p threshold (0.30/1.007 = "
                "0.298); the flat half of the profile contrast"),
    "T4_lls030p": dict(
        kind="mixture_frac_prof", cls=0, f=0.30, n_mocks=16, part1=True,
        table_key="ratio_rows_X3_lls100", profile="f_sel_rescaled",
        purpose="30% LLS-selected with the X4 inverted-U PROFILE at the SAME mean fraction "
                "as T3: the clean z-profile test, since the X4-vs-X3 diagnostic showed n_s "
                "responds to the profile rather than to the mean fraction"),
}
# subDLA arms (f = 0.25, 0.50) are pre-registered as DEFERRED: they run only if the budget
# after ARM-P allows. They are deliberately NOT in this payload, so adding them later moves
# the signature and cannot silently pool with a T1-T4 readout.

COST_CPUH_PER_FIT = 2.28         # MEASURED on the X battery (200.5 CPU-h / 88 fits)


def arm_ids():
    return tuple(ARMS)


def campaign_cost_cpuh():
    return sum(int(e["n_mocks"]) for e in ARMS.values()) * COST_CPUH_PER_FIT


def batch_cells():
    """Array-id -> (arm, mock), registry order. 4 arms x 16 = 64 cells."""
    return [(a, m) for a in ARMS for m in range(int(ARMS[a]["n_mocks"]))]


def shard_pkl_name(arm_id, mock, *, smoke=False):
    """Distinct `ks_thresh_` prefix so no glob can mix the two campaigns' shards."""
    assert arm_id in ARMS, f"unknown threshold arm {arm_id!r}"
    return f"ks_thresh_{arm_id}_shard_{int(mock):03d}{'.smoke' if smoke else ''}.pkl"


# ---------------------------------------------------------------------------------------- #
#  Signature: hashes the payload AND the pinned truth-table sha, exactly as the X registry
#  does, so a drifted arm table or a swapped truth table fails loud at readout.
# ---------------------------------------------------------------------------------------- #
def _registry_payload(table_path=None, expect_sha=None):
    tt = load_truth_tables(table_path, expect_sha)
    return dict(
        campaign=CAMPAIGN, gate=float(GATE), pair_seed=int(PAIR_SEED),
        z_pivot=float(Z_PIVOT), k0_arm=K0_ARM_ID, k0_n_mocks=int(K0_N_MOCKS),
        truth_table_sha256=tt["sha256"],
        arms={a: {k: (float(v) if isinstance(v, float) else v)
                  for k, v in e.items() if k != "purpose"}
              for a, e in ARMS.items()})


def registry_signature(table_path=None, expect_sha=None):
    payload = json.dumps(_registry_payload(table_path, expect_sha),
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()
