"""TDD tests for hcd_analysis/emulator/lit_dndx_kernel.py — the PRIYA-CDDF cache kernel
(spec 2026-07-18 step 5): pinned K1 recipe (class columns + sub-bin power-law floor
integral over [17.2,17.5) — 17.5 is a bin CENTRE, not an edge), per-sim ratio + suite
MEDIAN aggregation, ln(1+z) interpolation, floor factor, cache sha256 provenance,
same-file consistency check, and the 0.88-vs-0.96 adjudication recipes.

Cache-dependent (skipif pattern; CI consumes only the committed JSON).

Env:
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
    /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_lit_dndx_kernel.py -q
"""
import os

import numpy as np
import pytest

import hcd_analysis.emulator  # noqa: F401
from hcd_analysis.emulator import lit_dndx_kernel as LK

_CACHE = "/home/mfho/hcd_priya/hcd_analysis/_emulator_data/observables_tau0_lf.h5"
_HAS_CACHE = os.path.exists(_CACHE)

needs_cache = pytest.mark.skipif(not _HAS_CACHE, reason="LF cache not present")


@pytest.fixture(scope="module")
def inputs():
    if not _HAS_CACHE:
        pytest.skip("LF cache not present")
    return LK.load_kernel_inputs(_CACHE)


# --------------------------------------------------------------------------- #
#  Sub-bin floor integral: the 17.5-is-a-bin-centre trap                       #
# --------------------------------------------------------------------------- #
def test_subbin_powerlaw_integral_exact_on_synthetic_powerlaw():
    """On an exact power-law CDDF f = C N^a the sub-bin integral over [17.2,17.5) must
    match the analytic value to <1% (bin-average f per 0.2-dex bin as in the cache)."""
    edges = np.arange(17.0, 23.01, 0.2)
    a_true, C = -1.7, 1e9
    Nlo, Nhi = 10.0 ** edges[:-1], 10.0 ** edges[1:]
    fbar = C * (Nhi ** (a_true + 1) - Nlo ** (a_true + 1)) / (a_true + 1) / (Nhi - Nlo)
    got = LK.floor_piece_17p2_17p5(fbar[None, :], edges, method="subbin_powerlaw")[0]
    want = C * ((10.0 ** 17.5) ** (a_true + 1) - (10.0 ** 17.2) ** (a_true + 1)) / (a_true + 1)
    assert abs(got / want - 1.0) < 0.01


def test_bin_centre_trap_methods_differ():
    """The trap case (spec step 5): the power-law sub-bin treatment differs from the
    constant-f partial and from naive edge slicing (treating 17.4 or 17.6 as the floor
    boundary) — for a falling CDDF, powerlaw > const_f, and the naive slices bracket."""
    edges = np.arange(17.0, 23.01, 0.2)
    a_true, C = -1.7, 1e9
    Nlo, Nhi = 10.0 ** edges[:-1], 10.0 ** edges[1:]
    fbar = C * (Nhi ** (a_true + 1) - Nlo ** (a_true + 1)) / (a_true + 1) / (Nhi - Nlo)
    f = fbar[None, :]
    pl = LK.floor_piece_17p2_17p5(f, edges, method="subbin_powerlaw")[0]
    cf = LK.floor_piece_17p2_17p5(f, edges, method="const_f")[0]
    lo = LK.floor_piece_17p2_17p5(f, edges, method="naive_edge_17p4")[0]
    hi = LK.floor_piece_17p2_17p5(f, edges, method="naive_edge_17p6")[0]
    assert pl > cf                                    # falling f: more mass below 17.5
    assert lo < pl < hi                               # naive slices bracket the truth
    assert abs(pl / cf - 1.0) > 0.02                  # the trap is material, not cosmetic


@needs_cache
def test_pinned_k1_uses_subbin_powerlaw(inputs):
    tab = LK.k1_table(inputs)
    assert tab["floor_method"] == "subbin_powerlaw"
    assert tab["aggregate"] == "median"


# --------------------------------------------------------------------------- #
#  Pinned K1 recipe on the real cache                                          #
# --------------------------------------------------------------------------- #
@needs_cache
def test_k1_r3_reproducible_and_in_range(inputs):
    """r(3.0) under the pinned recipe: deterministic to 1e-3 across recomputation
    (spec test 5 analog pre-JSON) and inside the plausible (0.80, 1.00) window."""
    r3_a = LK.k1_r_at(inputs, np.array([3.0]))[0]
    r3_b = LK.k1_r_at(LK.load_kernel_inputs(_CACHE), np.array([3.0]))[0]
    assert abs(r3_a - r3_b) < 1e-3
    assert 0.80 < r3_a < 1.00


@needs_cache
def test_k1_monotonic_in_z(inputs):
    """The CS designer's live values rise 0.834@z2.4 -> 0.943@z4.2; the pinned-recipe
    r(z) must be monotonically rising over that range."""
    r = LK.k1_r_at(inputs, np.array([2.4, 3.0, 3.6, 4.2]))
    assert np.all(np.diff(r) > 0)


@needs_cache
def test_floor_factor_range(inputs):
    """F_floor = l([17.2,19.0))/l([17.5,19.0)) expected in PRIYA's ~1.25-1.40 class at
    z~3 (spec sec 3, K2 floor borrow)."""
    F = LK.floor_factor_at(inputs, np.array([3.0]))[0]
    assert 1.15 < F < 1.55


@needs_cache
def test_cache_provenance(inputs):
    assert len(inputs["sha256_f_nhi"]) == 64
    assert inputs["shapes"]["snap_f_nhi"] == (1072, 30)
    assert inputs["path"].endswith("observables_tau0_lf.h5")


@needs_cache
def test_same_file_consistency(inputs):
    """Class-column sum vs CDDF-integrated l(>=17.2) from the same file (the weaker
    same-file substitute; flagged in the JSON)."""
    c = LK.same_file_consistency(inputs)
    assert abs(c["median_ratio"] - 1.0) < 0.05
    assert c["flag"] == "same-file check only (hcd_summary_lf.h5 absent)"


# --------------------------------------------------------------------------- #
#  Adjudication: the 0.88-vs-0.96 same-cache ambiguity (spec step 6)           #
# --------------------------------------------------------------------------- #
@needs_cache
def test_sibling_lx_above_recipe_reproduces_0877(inputs):
    """The sibling's _lx_above-style construction (all-f_nhi integration, const-f partial
    bin at 17.5) must reproduce their 0.877@z3.0 (and 0.834@z2.4) — else the -5% bracket
    end is unreproducible (spec risk 2)."""
    adj = LK.adjudication(inputs)
    sib = adj["recipes"]["sibling_lx_above"]
    assert abs(sib["r"][list(adj["z_eval"]).index(3.0)] - 0.877) < 0.01
    assert abs(sib["r"][list(adj["z_eval"]).index(2.4)] - 0.834) < 0.01


@needs_cache
def test_adjudication_table_attributions(inputs):
    """The adjudication must attribute the pinned-vs-sibling-vs-candidate differences to
    the three named choices (floor sub-bin treatment, mean-vs-median,
    f_nhi-integration-vs-class-columns)."""
    adj = LK.adjudication(inputs)
    assert "pinned_k1" in adj["recipes"]
    for k in ("floor_subbin_treatment", "mean_vs_median", "fnhi_vs_class_columns"):
        assert k in adj["attribution"], k
    # every recipe reports r at the standard z_eval grid
    for name, rec in adj["recipes"].items():
        assert len(rec["r"]) == len(adj["z_eval"]), name


# --------------------------------------------------------------------------- #
#  Real-kernel fit ordering + emitted-JSON reproducibility (steps 7-8)         #
# --------------------------------------------------------------------------- #
@needs_cache
def test_fit_ordering_with_real_kernel(inputs):
    """The full ordering with the real cache kernel: K1a/K1b/K2/K3 all emitted; K3
    telescoping-consistent by construction; width variants sane."""
    from hcd_analysis.emulator import lit_dndx as LD
    k1_tab = LK.k1_table(inputs)
    F_tab = LK.floor_factor_table(inputs)

    def r_k1(z):
        return LK.interp_ln1pz(k1_tab["z_vals"], k1_tab["r"], z)

    def floor_factor(z):
        return LK.interp_ln1pz(F_tab["z_vals"], F_tab["F"], z)

    k1b = LK.k1_smooth_fit(inputs, _tab=k1_tab)
    res = LD.run_fit_ordering(floor_factor, r_k1=r_k1, r_k1_smooth=k1b["r_at"],
                              include_sensitivity=False)
    for name in ("K1a", "K1b", "K2", "K3"):
        assert name in res, name
    # K3 is telescoping-consistent BY CONSTRUCTION (pulls ~0)
    assert np.all(np.abs(res["K3"]["telescoping"]["pull"]) < 0.5)
    # expected alpha/r ordering K1 > K3 > K2 (spec sec 3, computed not assumed)
    assert res["K1a"]["r3"] > res["K3"]["r3"] > res["K2"]["r3"]
    # measurement-only width in the sane band (sibling found 0.17; deployed knob 0.15-0.16)
    w = LD.corrected_width_variants(res["K3"]["law"], s_r3=0.0)
    assert 0.10 < w["meas_only"] < 0.25
    assert w["with_kernel"] >= w["meas_only"]


_JSON = "/home/mfho/hcd_priya/hcd_analysis/emulator/hcd_lit_dndx_corrected.json"


@pytest.mark.skipif(not (_HAS_CACHE and os.path.exists(_JSON)),
                    reason="cache or emitted JSON not present")
def test_emitted_json_kernel_reproducibility(inputs):
    """Spec test-5 analog: r(3.0) recomputed from the h5 under the pinned recipe matches
    the emitted JSON's kernel table to 1e-3; kernel_chosen is the PI-adopted K1a
    (PI 2026-07-18 decision 1) and the adopted (constrained-slope) alpha is recorded."""
    import json
    with open(_JSON) as fh:
        j = json.load(fh)
    assert j["kernel_chosen"] == "K1a"
    z_vals = np.asarray(j["kernel_table"]["z_vals"], float)
    r_json = np.asarray(j["kernel_table"]["K1_r"], float)
    r3_json = LK.interp_ln1pz(z_vals, r_json, np.array([3.0]))[0]
    r3_live = LK.k1_r_at(inputs, np.array([3.0]))[0]
    assert abs(r3_json - r3_live) < 1e-3
    # the emitted laws + alpha blocks exist for every kernel candidate (decision record)
    for n in ("K1a", "K1b", "K2", "K3"):
        assert n in j["laws"]["LLS_per_kernel"], n
        assert n in j["alpha_lls_z3"]["per_kernel"], n
    # the ADOPTED deployed law: K1a-corrected points refit with gamma CONSTRAINED to the
    # deployed z-slope 2.127 (PI decision 1c); the free-gamma K1a fit is the recorded
    # consistency evidence justifying the constraint.
    ad = j["adopted_law"]
    assert ad["kernel"] == "K1a"
    assert ad["gamma"] == 2.127
    assert ad["free_fit_evidence"]["gamma"] == pytest.approx(2.137, abs=0.01)
    assert abs(ad["free_fit_evidence"]["gamma"] - ad["gamma"]) < 0.5 * max(
        ad["free_fit_evidence"]["sigma_gamma"], 1e-9), \
        "constraint only justified if free gamma is consistent with 2.127"
    assert j["alpha_lls_z3"]["adopted"] == pytest.approx(0.172, abs=0.005)
    # adopted corrected points recorded for the cache-free reproduction test
    for k in ("z_bar", "lx", "sig_log", "r"):
        assert len(ad["corrected_points"][k]) == 8, k
