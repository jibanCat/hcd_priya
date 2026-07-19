"""TDD tests for hcd_analysis/emulator/lit_dndx.py — the single-source-of-truth literature
dN/dX data module (corrected-law re-derivation, spec 2026-07-18-corrected-law-spec.md steps 1-2).

Covers: estimand-named arrays + metadata, transcription self-checks (Zafar n sums to 89,
two-route sub-DLA agreement <= 0.025/row), tombstones (old wrong-object arrays greppable,
never fit), the atol=0 CDDF comparison helper, the POW10 l(z)->l(X) conversion recompute,
Gehrels error reproduction, and the unverified-point fallback switches (journal-fix
delta-widening, O'Meara variant, Table 9 exclusion).

Env:
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
    /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_lit_dndx.py -q
"""
import numpy as np
import pytest

import hcd_analysis.emulator  # noqa: F401  x64 before jax (package convention)
from hcd_analysis.emulator import lit_dndx as LD


# --------------------------------------------------------------------------- #
#  Transcription self-checks (import-time; re-run explicitly here)             #
# --------------------------------------------------------------------------- #
def test_zafar_subdla_counts_sum_to_89():
    a = LD.ELL_X_SUBDLA_BINNED_19P0_20P3_ZAFAR13_T3
    assert int(np.sum(a["n"])) == 89


def test_zafar_two_route_subdla_agreement():
    """|n/dX - (l>19.0 - l>20.3)| <= 0.025 per row (printed cumulative column minus the
    row-matched Peroux DLA-block column)."""
    a = LD.ELL_X_SUBDLA_BINNED_19P0_20P3_ZAFAR13_T3
    rate = np.asarray(a["n"], float) / np.asarray(a["dX"], float)
    two_route = np.asarray(a["lx_gt19"], float) - np.asarray(a["lx_gt203"], float)
    assert np.all(np.abs(rate - two_route) <= 0.025)


def test_self_checks_raise_on_transcription_drift():
    """The module's self-check function must raise if a count drifts (anti-drift tripwire)."""
    import copy
    bad = copy.deepcopy(LD.ELL_X_SUBDLA_BINNED_19P0_20P3_ZAFAR13_T3)
    bad["n"] = np.asarray(bad["n"]).copy()
    bad["n"][0] += 1                                   # 89 -> 90
    with pytest.raises(AssertionError):
        LD._check_zafar_subdla(bad)
    bad2 = copy.deepcopy(LD.ELL_X_SUBDLA_BINNED_19P0_20P3_ZAFAR13_T3)
    bad2["lx_gt19"] = np.asarray(bad2["lx_gt19"]).copy()
    bad2["lx_gt19"][2] += 0.05                          # breaks the two-route identity
    with pytest.raises(AssertionError):
        LD._check_zafar_subdla(bad2)


def test_pow10_self_checks_raise_on_drift():
    import copy
    bad = copy.deepcopy(LD.POW10_T4)
    bad["lx"] = np.asarray(bad["lx"]).copy()
    bad["lx"][0] = 0.45                                # breaks m/dX identity
    with pytest.raises(AssertionError):
        LD._check_pow10(bad)


# --------------------------------------------------------------------------- #
#  Estimand naming + metadata                                                  #
# --------------------------------------------------------------------------- #
def test_every_source_array_has_estimand_and_cite():
    for name in ("ELL_X_CUMULATIVE_GE17P5_TAU2",
                 "ELL_X_SUBDLA_BINNED_19P0_20P3_ZAFAR13_T3",
                 "ELL_X_DLA_GE20P3_PW09_T1"):
        a = getattr(LD, name)
        assert isinstance(a.get("estimand"), str) and len(a["estimand"]) > 10, name
        assert isinstance(a.get("cite"), str) and len(a["cite"]) > 5, name


def test_corrected_arxiv_ids():
    """The two commissioning-context IDs were WRONG (0912.0562 graphene / 1306.0333); the
    corrected IDs are pinned here so the provenance chain cannot re-break."""
    assert "0912.0292" in LD.ELL_X_CUMULATIVE_GE17P5_TAU2["cite"]
    assert "1307.0602" in LD.ELL_X_SUBDLA_BINNED_19P0_20P3_ZAFAR13_T3["cite"]
    assert "0811.2003" in LD.ELL_X_DLA_GE20P3_PW09_T1["cite"]
    assert "0912.0562" not in str(LD.ELL_X_CUMULATIVE_GE17P5_TAU2)
    assert "1306.0333" not in str(LD.ELL_X_SUBDLA_BINNED_19P0_20P3_ZAFAR13_T3)


def test_lls_compilation_shape_and_estimand():
    a = LD.ELL_X_CUMULATIVE_GE17P5_TAU2
    for k in ("z_bar", "lx", "err_lo", "err_hi"):
        assert len(np.asarray(a[k])) == 8, k
    assert "cumulative" in a["estimand"]
    assert "17.5" in a["estimand"]
    # POW10 block: hi (Gehrels upper) >= lo everywhere
    assert np.all(np.asarray(a["err_hi"]) >= np.asarray(a["err_lo"]) - 1e-12)


def test_subdla_estimand_is_binned_counts():
    a = LD.ELL_X_SUBDLA_BINNED_19P0_20P3_ZAFAR13_T3
    assert "19.0" in a["estimand"] and "20.3" in a["estimand"]
    assert "binned" in a["estimand"].lower()
    # the pure rate is n/dX, exact Poisson counts (4,11,24,23,19,8)
    assert list(np.asarray(a["n"], int)) == [4, 11, 24, 23, 19, 8]
    np.testing.assert_allclose(np.asarray(a["dX"], float),
                               [87.3, 156.8, 162.8, 124.5, 91.7, 41.0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(a["z_bar"], float),
                               [1.80, 2.26, 2.76, 3.21, 3.72, 4.18], rtol=0, atol=1e-12)


def test_dla_array_counts_representation():
    a = LD.ELL_X_DLA_GE20P3_PW09_T1
    assert list(np.asarray(a["m"], int)) == [79, 132, 169, 227, 86, 46]
    np.testing.assert_allclose(np.asarray(a["dX"], float),
                               [1652.7, 2405.8, 2539.7, 2702.5, 1139.2, 432.8],
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(a["lx"], float),
                               [0.048, 0.055, 0.067, 0.084, 0.075, 0.106], rtol=0, atol=1e-12)
    assert "20.3" in a["estimand"]


# --------------------------------------------------------------------------- #
#  Tombstones (greppable, never fit)                                           #
# --------------------------------------------------------------------------- #
def test_tombstone_lls_old_defects_matches_deployed_wrong_arrays():
    t = LD.LLS_TAU2_OLD_DEFECTS
    np.testing.assert_allclose(np.asarray(t["z"], float),
                               [2.4, 2.8, 3.35, 3.47, 3.58, 3.74, 3.97, 4.23],
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(t["lx"], float),
                               [0.29, 0.33, 0.35, 0.57, 0.41, 0.52, 0.72, 0.78],
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(t["err"], float),
                               [0.05, 0.08, 0.14, 0.12, 0.07, 0.08, 0.15, 0.19],
                               rtol=0, atol=1e-12)
    assert "cumulative" in t["defect"].lower()


def test_tombstone_zafar_dla_block_matches_deployed_wrong_subdla():
    t = LD.ZAFAR13_T3_DLA_BLOCK_WRONG_SUBDLA_LABEL
    np.testing.assert_allclose(np.asarray(t["z"], float),
                               [2.27, 2.73, 3.25, 3.77, 4.20], rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(t["lx"], float),
                               [0.07, 0.06, 0.08, 0.10, 0.10], rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(t["err"], float),
                               [0.01, 0.01, 0.02, 0.02, 0.03], rtol=0, atol=1e-12)
    d = t["defect"].lower()
    assert "dla" in d and ("peroux" in d or "péroux" in d or "wrong" in d)


# --------------------------------------------------------------------------- #
#  atol=0 CDDF comparison helper (footgun rule S8.3)                           #
# --------------------------------------------------------------------------- #
def test_atol0_helper_catches_cddf_scale_mismatch():
    """np.allclose's default atol=1e-8 is vacuously true on f(N)~1e-21; the module helper
    must use rtol with atol=0 and FAIL on a deliberately wrong 1e-21-scale array."""
    good = np.array([1.0e-21, 2.0e-21])
    bad = np.array([2.0e-21, 4.0e-21])                 # 2x wrong — but np.allclose says fine
    assert np.allclose(good, bad)                      # demonstrate the footgun is real
    with pytest.raises(AssertionError):
        LD.assert_close_cddf(good, bad, rtol=1e-3, where="test")
    LD.assert_close_cddf(good, good * (1 + 1e-6), rtol=1e-3, where="test")  # passes


# --------------------------------------------------------------------------- #
#  POW10 l(z)->l(X) conversion recompute (spec step 2)                         #
# --------------------------------------------------------------------------- #
def test_pow10_lx_equals_m_over_dX():
    t = LD.POW10_T4
    lx = np.asarray(t["m"], float) / np.asarray(t["dX"], float)
    assert np.all(np.abs(lx - np.asarray(t["lx"], float)) <= 0.005 + 1e-12)


def test_pow10_lz_to_lx_conversion_recomputed():
    """l(X) = l(z) * (dz/dX) with the survey's own per-bin path ratio dz_i/dX_i; all six
    recomputed, never inherited (fallback policy (d))."""
    t = LD.POW10_T4
    lx_conv = np.asarray(t["lz"], float) * np.asarray(t["dz"], float) / np.asarray(t["dX"], float)
    # 0.008: the printed l(z) is sensitivity-weighted (not exactly m/dz), worst 0.007 @4.23
    assert np.all(np.abs(lx_conv - np.asarray(t["lx"], float)) <= 0.008 + 1e-12)


def test_pow10_path_ratio_consistent_with_cosmology():
    """Per-bin dX/dz ~ (1+z_bar)^2/E(z_bar) with POW10's stated cosmology (Om=0.3, OL=0.7)
    to ~2% (path-weighting inside the bin explains the residual)."""
    t = LD.POW10_T4
    zb = np.asarray(t["z_bar"], float)
    dxdz = np.asarray(t["dX"], float) / np.asarray(t["dz"], float)
    dxdz_cosmo = LD.dX_dz(zb, omega_m=0.3, omega_l=0.7)
    assert np.all(np.abs(dxdz / dxdz_cosmo - 1.0) < 0.02)


def test_pow10_gehrels_errors_reproduce_printed_magnitudes():
    """Gehrels (1986) 1-sigma Poisson errors on the printed m reproduce the printed l(X)
    error magnitudes to the table rounding; upper > lower for all rows (this pins the
    physically-correct orientation of the disputed z=4.23 pair)."""
    t = LD.POW10_T4
    lo_c, hi_c = LD.gehrels_errors(np.asarray(t["m"], float))
    dX = np.asarray(t["dX"], float)
    assert np.all(np.abs(hi_c / dX - np.asarray(t["lx_err_hi"], float)) <= 0.015)
    assert np.all(np.abs(lo_c / dX - np.asarray(t["lx_err_lo"], float)) <= 0.015)
    assert np.all(hi_c >= lo_c)


def test_pw09_gehrels_errors_reproduce_printed():
    t = LD.ELL_X_DLA_GE20P3_PW09_T1
    lo_c, hi_c = LD.gehrels_errors(np.asarray(t["m"], float))
    dX = np.asarray(t["dX"], float)
    assert np.all(np.abs(hi_c / dX - np.asarray(t["err_hi"], float)) <= 0.0015)
    assert np.all(np.abs(lo_c / dX - np.asarray(t["err_lo"], float)) <= 0.0015)


def test_fumagalli_conversion():
    f = LD.FUMAGALLI13
    assert abs(f["lz"] * f["dz_dX"] - f["lx"]) <= 0.005
    assert abs(f["lz_err"] * f["dz_dX"] - f["err"]) <= 0.005


# --------------------------------------------------------------------------- #
#  Fallback switches (unverified points; spec sec 2 fallback policy)           #
# --------------------------------------------------------------------------- #
def test_journal_fix_switch():
    """Default arm = v1 values with the v1-journal delta added in quadrature at z=3.97
    (bracket widening, not silent trust); journal_fix=True arm = the sibling's journal read."""
    v1 = LD.lls_compilation(journal_fix=False)
    jf = LD.lls_compilation(journal_fix=True)
    i397 = int(np.argmin(np.abs(np.asarray(v1["z_bar"]) - 3.97)))
    assert v1["lx"][i397] == pytest.approx(0.72)
    assert jf["lx"][i397] == pytest.approx(0.70)
    # delta-widening: default-arm sigma at 3.97 exceeds the raw v1 sigma
    raw_hi = LD.POW10_T4["lx_err_hi"][4]
    assert v1["err_hi"][i397] > raw_hi
    assert v1["err_hi"][i397] == pytest.approx(np.hypot(raw_hi, 0.02), rel=1e-6)


def test_omeara_variant_switch():
    wm = LD.lls_compilation(omeara_variant="wmean")
    t5 = LD.lls_compilation(omeara_variant="table5")
    i = 0                                              # O'Meara is the lowest-z point
    assert wm["z_bar"][i] == pytest.approx(2.21)
    assert wm["lx"][i] == pytest.approx(0.29) and wm["err_hi"][i] == pytest.approx(0.05)
    assert t5["lx"][i] == pytest.approx(0.28) and t5["err_hi"][i] == pytest.approx(0.06)
    with pytest.raises(ValueError):
        LD.lls_compilation(omeara_variant="table9")    # Table 9 EXCLUDED (unverified)


def test_omeara_z_is_221_not_24():
    """The deployed 0.29@z=2.4 mis-stated the abscissa (2.4 is the f(N) pivot)."""
    a = LD.ELL_X_CUMULATIVE_GE17P5_TAU2
    assert 2.4 not in list(np.round(np.asarray(a["z_bar"], float), 2)[:2])
    assert a["z_bar"][0] == pytest.approx(2.21)


def test_colour_selection_hedge_recorded_not_fit():
    """Fumagalli colour-selected 0.51+/-0.13 is a recorded one-signed hedge, never in the
    fit compilation."""
    assert LD.FUMAGALLI13["colour_selected_lx"] == pytest.approx(0.51)
    a = LD.ELL_X_CUMULATIVE_GE17P5_TAU2
    assert not np.any(np.isclose(np.asarray(a["lx"], float), 0.51))


# --------------------------------------------------------------------------- #
#  Display points for the plot/consumer scripts (single-source rewire)         #
# --------------------------------------------------------------------------- #
def test_lit_points_for_display_corrected_estimands():
    """The plot-script feed: per-class (z, value, err, source) display tuples built from
    the CORRECTED estimands — LLS = the ADOPTED kernel-corrected binned points from the
    committed derivation JSON; subDLA = Zafar counts n/dX (Gehrels errors); DLA = PW09
    binned points. Kills the LIT-dict hard-coding across the six consumer scripts."""
    import json
    pts = LD.lit_points_for_display()
    for cls in ("LLS", "subDLA", "DLA"):
        z, v, e, src = pts[cls]
        assert len(z) == len(v) == len(e) and len(src) > 5, cls
        assert np.all(np.asarray(e) > 0), cls
    with open("/home/mfho/hcd_priya/hcd_analysis/emulator/hcd_lit_dndx_corrected.json") as fh:
        j = json.load(fh)
    np.testing.assert_allclose(pts["LLS"][1], j["adopted_law"]["corrected_points"]["lx"],
                               rtol=0, atol=0)
    a = LD.ELL_X_SUBDLA_BINNED_19P0_20P3_ZAFAR13_T3
    np.testing.assert_allclose(pts["subDLA"][1],
                               np.asarray(a["n"], float) / np.asarray(a["dX"], float),
                               rtol=0, atol=0)
    np.testing.assert_allclose(pts["DLA"][1], LD.ELL_X_DLA_GE20P3_PW09_T1["lx"],
                               rtol=0, atol=0)
    # the estimand labels ride along (the estimand-assert hook for plot scripts)
    assert "binned" in pts["estimand"]["LLS"] and "17.2" in pts["estimand"]["LLS"]
    assert "19.0" in pts["estimand"]["subDLA"]
    # the OLD wrong-object values are NOT what is served (the rewire is real)
    assert not np.allclose(pts["LLS"][1],
                           [0.29, 0.33, 0.35, 0.57, 0.41, 0.52, 0.72, 0.78])
    assert not np.allclose(pts["subDLA"][1][:5], [0.07, 0.06, 0.08, 0.10, 0.10])


# --------------------------------------------------------------------------- #
#  Error symmetrization                                                        #
# --------------------------------------------------------------------------- #
def test_symmetrize_log_errors_mean_and_larger_side():
    v, lo, hi = 0.78, 0.16, 0.20
    s_mean = LD.symmetrize_log_errors(v, lo, hi, mode="mean")
    s_up = 0.5 * (np.log(1 + hi / v) - np.log(1 - lo / v))
    assert s_mean == pytest.approx(s_up, rel=1e-12)
    s_larger = LD.symmetrize_log_errors(v, lo, hi, mode="larger")
    assert s_larger == pytest.approx(max(np.log(1 + hi / v), -np.log(1 - lo / v)), rel=1e-12)
    assert s_larger >= s_mean
    # orientation dependence of the log-space mean is a few-% (third-decimal on the fit);
    # NOTE the spec's parenthetical "both orientations give the same symmetrized sigma" holds
    # only to first order — the Gehrels-resolved orientation (upper=larger) is used throughout.
    s_swap = LD.symmetrize_log_errors(v, hi, lo, mode="mean")
    assert abs(s_swap / s_mean - 1.0) < 0.06
