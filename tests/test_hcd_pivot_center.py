"""HCD prior PIVOT center-construction guard (PI 2026-06-17, the dN/dX low-z overshoot bug).

THE BUG: the LLS/subDLA α-PIVOT center was built from
    w_c_med = nanmedian(w_c_cache[:, 1:], axis=0)        # the MEDIAN over ALL z-groups (z=2.0–5.4)
and consumed as the z=3 PIVOT amplitude. Because w_c rises monotonically with z, the all-z median
(LLS 0.274) equals the z≈3.6 value → the LLS α-center came out ~1.45× too high (0.291 vs the
z=3-consistent ~0.194–0.200), overshooting the lit dN/dX law 2.05× at z=2.4 / 1.63× at z=3 (worst
at low z) = the LLS→n_s leak.

THE FIX: build the pivot from the z=3 STRUCTURAL w_c (closure/SBC, survey=None) or the lit dN/dX law
DIRECTLY (real fit; hcd_lls_realfit_alpha_center / alpha_from_dndx_law). γ_LLS and σ_LLS unchanged.

These tests pin the PRODUCTION-path α-pivot at the z=3-consistent value (NOT the all-z median), and
the runtime pivot guard (a future revert to nanmedian(...all z...) TRIPS it). The CACHE-backed tests
verify the actual numbers; the cache-free tests pin the guard + the lit-law center on toy inputs.

Env (MANDATORY):
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
    /home/mfho/.conda/envs/emu-jax/bin/python3 -m pytest tests/test_hcd_pivot_center.py -q
"""
import json
import os
import numpy as np
import pytest

import hcd_analysis.emulator  # noqa: F401  x64 BEFORE jax
import jax.numpy as jnp

from hcd_analysis.emulator import inference as INF
from hcd_analysis.emulator import closure_legb as CL
from hcd_analysis.emulator.data import load_cache

_CACHE = CL.CACHE_PATH
_HAS_CACHE = os.path.exists(_CACHE)


def _adopted_json():
    """Loader helper: band + adopted-center literals are pinned FROM the committed
    derivation artifact (corrected-law re-derivation, PI adoption 2026-07-18), so the
    test and the deployed constants share one source of numbers."""
    p = "/home/mfho/hcd_priya/hcd_analysis/emulator/hcd_lit_dndx_corrected.json"
    with open(p) as fh:
        return json.load(fh)


_J = _adopted_json()

# z=3-consistent LLS α-pivot (DESI/cosmic-avg boost 1.0): corrected lit-law alt-(b)
# ≈ 0.172 (K1a kernel, constrained slope 2.127; PI 2026-07-18), sim z=3 w_c center
# 0.2004. The BUGGY all-z-median value is ≈0.291 (z≈3.6). Band from the committed JSON
# (= HCD_PIVOT_LLS_ALPHA_Z3_BAND = adopted center × (0.83, 1.21), 2-dp).
_LLS_Z3_LO, _LLS_Z3_HI = (float(b) for b in _J["adopted_band"]["band"])
# ONE human-blessed hard literal kept as the tripwire (adopted center at 3 decimals):
_LLS_CENTER_TRIPWIRE = 0.172
_LLS_ALLZ = 0.2909
_SUB_Z3 = 0.0622           # z=3 structural w_c subDLA × lit/sim 1.00 (slot unchanged)
_SUB_ALLZ = 0.0902         # the all-z-median subDLA (the bug)


# --------------------------------------------------------------------------- #
#  CACHE-BACKED: the PRODUCTION path (build_legb_ctx) α-pivot is z=3-consistent #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAS_CACHE, reason="LF cache not present")
def test_pivot_helper_returns_z3_wc_not_allz_median():
    """closure_legb.hcd_pivot_wc_and_xbar returns the z=3 STRUCTURAL w_c (LLS≈0.189), NOT the
    all-z median (LLS≈0.274 = z≈3.6) — the load-bearing fix the callers consume."""
    d = load_cache(_CACHE)
    w_c_z3, Xbar_z3 = CL.hcd_pivot_wc_and_xbar(d)
    allz = np.nanmedian(np.asarray(d["w_c_cache"])[:, 1:], axis=0)
    assert w_c_z3[0] == pytest.approx(0.1891, abs=2e-3), f"z=3 w_c LLS {w_c_z3[0]} != ~0.189"
    assert w_c_z3[0] < allz[0] - 0.05, "z=3 w_c LLS must be well BELOW the all-z median (z≈3.6)"
    assert allz[0] == pytest.approx(0.2744, abs=2e-3)     # confirm the bug value is what we think
    assert Xbar_z3 == pytest.approx(0.632, abs=0.03)      # Xbar(z=3) for the lit-law center


@pytest.mark.skipif(not _HAS_CACHE, reason="LF cache not present")
def test_production_path_lls_pivot_is_z3_consistent_not_allz_median():
    """The PRODUCTION prior center (built via the z=3 structural w_c, the build_legb_ctx path) gives
    a z=3-consistent α_pivot(LLS) (≈ lit/sim × w_LLS(z3) ≈ 0.188 with the corrected ratio 0.995),
    NOT ≈0.291 (the all-z-median bug). This FAILS on the OLD all-z-median construction (RED) and
    PASSES on the fix. subDLA pinned the same way (its lit/sim slot is unchanged at 1.00)."""
    d = load_cache(_CACHE)
    w_c_z3, _ = CL.hcd_pivot_wc_and_xbar(d)
    mu, _ = map(np.asarray, INF.hcd_incidence_prior(jnp.asarray(w_c_z3), z=3.0, survey="DESI"))
    # LLS pivot z=3-consistent, NOT the all-z median
    assert _LLS_Z3_LO <= mu[0] <= _LLS_Z3_HI, f"LLS α-pivot {mu[0]:.4f} not z=3-consistent"
    assert mu[0] == pytest.approx(float(INF.HCD_LIT_OVER_SIM[0]) * w_c_z3[0], rel=1e-6)
    assert abs(mu[0] - _LLS_ALLZ) > 0.05, f"LLS α-pivot {mu[0]:.4f} ≈ the all-z-median {_LLS_ALLZ}"
    # subDLA pivot z=3-consistent, NOT the all-z median
    assert mu[1] == pytest.approx(_SUB_Z3, abs=0.005), f"subDLA α-pivot {mu[1]:.4f} != z=3 ~0.062"
    assert abs(mu[1] - _SUB_ALLZ) > 0.01, f"subDLA α-pivot {mu[1]:.4f} ≈ the all-z-median {_SUB_ALLZ}"
    # and the OLD all-z-median construction (lit/sim × the all-z-median w_c ≈ 0.273) lands
    # ABOVE the band — the guard band still catches the bug construction
    allz = np.nanmedian(np.asarray(d["w_c_cache"])[:, 1:], axis=0)
    mu_bug, _ = map(np.asarray, INF.hcd_incidence_prior(jnp.asarray(allz), z=3.0, survey="DESI"))
    assert mu_bug[0] == pytest.approx(float(INF.HCD_LIT_OVER_SIM[0]) * allz[0], rel=1e-6)
    assert mu_bug[0] > _LLS_Z3_HI, "the all-z-median construction must land above the band"


@pytest.mark.skipif(not _HAS_CACHE, reason="LF cache not present")
def test_realfit_litlaw_lls_center_is_adopted_and_roundtrips():
    """The REAL-FIT LLS center built from the corrected lit dN/dX law DIRECTLY (alt-(b),
    hcd_lls_realfit_alpha_center) equals the JSON-adopted center (≈0.172; K1a kernel,
    constrained slope, PI 2026-07-18), computed THROUGH the deployed path from
    INF.HCD_LIT_DNDX_LAW — plus the human-blessed 3-dp tripwire literal. Round-trips the
    lit dN/dX_LLS(z=3) to <0.2% (re-measured at the corrected laws). Distinct from the
    all-z-median bug (~0.291)."""
    d = load_cache(_CACHE)
    _, Xbar_z3 = CL.hcd_pivot_wc_and_xbar(d)
    a_lls = INF.hcd_lls_realfit_alpha_center(Xbar_z3, z=3.0, boost=1.0)
    # expected center from the DEPLOYED law dict through the deployed path (not a copy of
    # the constant): the JSON-adopted value at the pinned Xbar 0.632, tolerance covering
    # the small cache-Xbar vs pinned-Xbar difference
    assert a_lls == pytest.approx(_J["alpha_lls_z3"]["adopted"], abs=0.004)
    assert a_lls == pytest.approx(_LLS_CENTER_TRIPWIRE, abs=0.004), \
        f"lit-law LLS center {a_lls:.4f} != the human-blessed tripwire {_LLS_CENTER_TRIPWIRE}"
    assert abs(a_lls - _LLS_ALLZ) > 0.05, "lit-law LLS center must NOT be the all-z-median bug"
    # round-trip: α_LLS → dN/dX vs the lit law A·(1+z)^γ
    from hcd_analysis.emulator.dndx_wc import alpha_to_dndx
    A, g = INF.HCD_LIT_DNDX_LAW["LLS"]
    # build the full (3,) alpha then read the LLS dN/dX back
    A3 = jnp.asarray([INF.HCD_LIT_DNDX_LAW[c][0] for c in ("LLS", "subDLA", "DLA")])
    g3 = jnp.asarray([INF.HCD_LIT_DNDX_LAW[c][1] for c in ("LLS", "subDLA", "DLA")])
    from hcd_analysis.emulator.dndx_wc import alpha_from_dndx_law
    alpha3 = np.asarray(alpha_from_dndx_law(A3, g3, jnp.asarray([Xbar_z3]), jnp.asarray([3.0])))[0]
    dndx_rt = np.asarray(alpha_to_dndx(jnp.asarray(alpha3[None, :]),
                                       jnp.asarray([Xbar_z3]), jnp.asarray([3.0])))[0]
    lit = A * (1.0 + 3.0) ** g
    assert abs(dndx_rt[0] / lit - 1.0) < 0.002, \
        f"lit-law LLS round-trip err {100*(dndx_rt[0]/lit-1):.3f}% must be <0.2% (re-measured)"


@pytest.mark.skipif(not _HAS_CACHE, reason="LF cache not present")
def test_build_legb_ctx_real_fit_alpha_hcd_mu_is_lit_consistent():
    """The FULL production real-fit ctx (build_legb_ctx survey='DESI') carries alpha_hcd_mu[LLS]
    z=3-consistent (~0.172, the corrected lit-law center; PI 2026-07-18) — NOT the all-z-median
    ~0.291. This is the number the real fit's TruncatedNormal(A_hcd) is centered on."""
    ctx, _ = CL.build_legb_ctx(survey="DESI")
    a_lls = float(np.asarray(ctx.alpha_hcd_mu)[0])
    assert _LLS_Z3_LO <= a_lls <= _LLS_Z3_HI, \
        f"production real-fit alpha_hcd_mu[LLS] {a_lls:.4f} not z=3-consistent (~0.172)"
    assert a_lls == pytest.approx(_LLS_CENTER_TRIPWIRE, abs=0.004)
    assert abs(a_lls - _LLS_ALLZ) > 0.05, \
        f"production real-fit alpha_hcd_mu[LLS] {a_lls:.4f} ≈ the all-z-median bug {_LLS_ALLZ}"


@pytest.mark.skipif(not _HAS_CACHE, reason="LF cache not present")
def test_hier_ratios_rederived_at_pivot_z():
    """The hierarchical r_sub/r_dla ratio centers (Option B) are derived at the z=3 PIVOT w_c, NOT
    the all-z-median w_c (which inherited the same bug). r_sub = w_sub(z3)/w_LLS(z3),
    r_dla = 0.10·w_DLA(z3)/w_LLS(z3). build_legb_ctx exposes them via ctx.hcd_ratio_mu."""
    d = load_cache(_CACHE)
    w_c_z3, _ = CL.hcd_pivot_wc_and_xbar(d)
    r_sub = float(w_c_z3[1] / w_c_z3[0])
    r_dla = float(INF.HCD_DLA_RESIDUAL_FRAC * w_c_z3[2] / w_c_z3[0])
    ctx, _ = CL.build_legb_ctx(survey=None, hierarchical_hcd=True)
    got = np.asarray(ctx.hcd_ratio_mu)
    assert got[0] == pytest.approx(r_sub, rel=1e-4), f"r_sub {got[0]} != z=3-pivot {r_sub}"
    assert got[1] == pytest.approx(r_dla, rel=1e-4), f"r_dla {got[1]} != z=3-pivot {r_dla}"
    # The re-derivation is non-trivial: r_dla DIFFERS meaningfully from the all-z-median (LLS rises
    # faster than DLA with z → the all-z-median ratio under-states r_dla by ~12%). (r_sub is nearly
    # invariant — w_subDLA/w_LLS rise ~proportionally — so the LOAD-BEARING re-derivation is r_dla.)
    allz = np.nanmedian(np.asarray(d["w_c_cache"])[:, 1:], axis=0)
    r_dla_allz = float(INF.HCD_DLA_RESIDUAL_FRAC * allz[2] / allz[0])
    assert abs(got[1] - r_dla_allz) > 1e-3, \
        f"r_dla must DIFFER from the all-z-median ratio (re-derived at z=3): z3={got[1]:.5f} allz={r_dla_allz:.5f}"


# --------------------------------------------------------------------------- #
#  CACHE-FREE: the runtime pivot GUARD + the lit-law center math               #
# --------------------------------------------------------------------------- #
def test_pivot_guard_passes_z3_consistent_center():
    """assert_hcd_pivot_z3 PASSES the z=3-consistent LLS α-centers at the z=3 pivot: the
    corrected lit-law center (≈0.172, PI 2026-07-18) AND the sim z=3 w_c center 0.2004
    (the closure-guard reuse the band is REQUIRED to keep containing)."""
    INF.assert_hcd_pivot_z3(_LLS_CENTER_TRIPWIRE, z=3.0, where="test z=3 lit-law corrected")
    INF.assert_hcd_pivot_z3(0.2004, z=3.0, where="test z=3 sim-wc")


def test_pivot_guard_trips_on_allz_median_revert():
    """A future revert to nanmedian(w_c_cache[...all z...]) → α_pivot(LLS)≈0.291 → the guard FIRES
    with the message naming the z=3 / median-over-all-z bug."""
    with pytest.raises(AssertionError, match="z=3 w_c"):
        INF.assert_hcd_pivot_z3(INF.HCD_PIVOT_LLS_ALLZ_MEDIAN, z=3.0, where="test all-z revert")
    # the actual all-z-median production value (0.2909) also trips
    with pytest.raises(AssertionError):
        INF.assert_hcd_pivot_z3(0.2909, z=3.0, where="test all-z 0.291")


def test_pivot_guard_respects_survey_boost():
    """With the KS boost (2.5×) the band + the all-z-median reference scale by the boost: the
    z=3-consistent KS center (≈0.50) PASSES, the boosted all-z-median (≈0.73) TRIPS."""
    INF.assert_hcd_pivot_z3(0.2004 * 2.5, z=3.0, where="test KS z=3", boost=2.5)
    with pytest.raises(AssertionError):
        INF.assert_hcd_pivot_z3(INF.HCD_PIVOT_LLS_ALLZ_MEDIAN * 2.5, z=3.0, where="test KS all-z",
                                boost=2.5)


def test_pivot_guard_is_noop_off_pivot():
    """The guard only checks AT the z=3 pivot (the center-construction point); off-pivot it is a
    no-op (the center legitimately evolves away from the z=3 value)."""
    INF.assert_hcd_pivot_z3(0.50, z=4.2, where="off-pivot")   # would be out-of-band at z3, but z!=3 → OK


def test_lit_law_center_scales_with_boost():
    """hcd_lls_realfit_alpha_center applies the per-survey selection boost linearly."""
    a1 = INF.hcd_lls_realfit_alpha_center(0.632, z=3.0, boost=1.0)
    a25 = INF.hcd_lls_realfit_alpha_center(0.632, z=3.0, boost=2.5)
    assert a25 == pytest.approx(2.5 * a1, rel=1e-6)
    assert a1 == pytest.approx(_LLS_CENTER_TRIPWIRE, abs=5e-4)   # the 3-dp tripwire


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
