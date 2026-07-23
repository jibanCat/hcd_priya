"""Single source of truth: literature HCD dN/dX tables, estimand-named (corrected-law
re-derivation, spec hcd_priya_notes/docs/superpowers/2026-07-18-corrected-law-spec.md sec 2.1).

Every array is ESTIMAND-NAMED and carries ``estimand`` + ``cite`` (arXiv ID) metadata.
Provenance discipline: every number below traces to (a) the 2026-07-18 spec provenance
table, (b) a designer report (workflow wf_fcca71f8-3b2), or (c) an implementation-phase
WebFetch on 2026-07-18 (ar5iv/arXiv, marked "verified: implementer"). Values that could
NOT be verified in-house carry an explicit status and a fallback switch — never silent trust.

THE TWO BUGS this module retires (kept greppable as tombstones, never fit):
  * LLS_TAU2_OLD_DEFECTS — the deployed 8-point "LLS" array is the CUMULATIVE tau912>=2
    l(X) (N>=10^17.5, INCLUDES subDLAs+DLAs) mislabeled as the binned LLS class rate, with
    the O'Meara abscissa mis-stated (2.4 = the f(N) pivot, not the measurement z).
  * ZAFAR13_T3_DLA_BLOCK_WRONG_SUBDLA_LABEL — the deployed 5-point "subDLA" array is
    rows 2-6 of Zafar+2013 Table 3's logN>=20.3 block = the Peroux+2003b DLA rates.

Corrected arXiv IDs (the commissioning context had two WRONG ones): POW10 = 0912.0292
(NOT 0912.0562, a graphene paper); Zafar = 1307.0602 (NOT 1306.0333).

numpy-only (no JAX; repo rule for the derivation code). Import-time self-checks at the
bottom raise on any transcription drift.
"""
from __future__ import annotations

import numpy as np

Z_PIVOT = 3.0

ARXIV = {
    "POW10": "0912.0292",        # Prochaska, O'Meara & Worseck 2010, ApJ 718, 392
    "ZAFAR13": "1307.0602",      # Zafar et al. 2013, A&A 556, A141
    "PW09": "0811.2003",         # Prochaska & Wolfe 2009, ApJ 696, 1543
    "OMEARA13": "1204.3093",     # O'Meara et al. 2013, ApJ 765, 137
    "FUMAGALLI13": "1308.1101",  # Fumagalli et al. 2013, ApJ 775, 78
}


# --------------------------------------------------------------------------- #
#  Small numeric helpers                                                       #
# --------------------------------------------------------------------------- #
def gehrels_errors(m):
    """Gehrels (1986) 1-sigma Poisson confidence errors on counts ``m``.
    Returns (lo, hi) in COUNTS: hi = lambda_u - m = sqrt(m+3/4)+1,
    lo = m - lambda_l with lambda_l = m*(1 - 1/(9m) - 1/(3 sqrt(m)))^3."""
    m = np.asarray(m, float)
    hi = np.sqrt(m + 0.75) + 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        lam_l = m * (1.0 - 1.0 / (9.0 * m) - 1.0 / (3.0 * np.sqrt(m))) ** 3
    lo = np.where(m > 0, m - lam_l, 0.0)
    return lo, hi


def dX_dz(z, omega_m=0.3, omega_l=0.7):
    """Absorption-path measure dX/dz = (1+z)^2 / E(z), E = sqrt(Om(1+z)^3 + OL)."""
    z = np.asarray(z, float)
    return (1.0 + z) ** 2 / np.sqrt(omega_m * (1.0 + z) ** 3 + omega_l)


def assert_close_cddf(a, b, rtol, where):
    """CDDF-scale comparison with atol=0 (footgun rule S8.3: np.allclose's default
    atol=1e-8 is vacuously true on f(N)~1e-21). Raises AssertionError on mismatch."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if not np.allclose(a, b, rtol=rtol, atol=0.0):
        rel = np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300))
        raise AssertionError(
            f"CDDF-scale mismatch [{where}]: max relative deviation {rel:.3e} > rtol={rtol} "
            f"(atol=0 discipline)")


def symmetrize_log_errors(v, lo, hi, mode="mean"):
    """Log-space symmetrization of asymmetric errors (v, -lo, +hi).

    mode="mean": mean of the up/down log widths, 0.5*(ln(1+hi/v) - ln(1-lo/v))
      (both designers' choice). NOTE: only first-order orientation-invariant; the
      Gehrels-resolved orientation (upper = larger magnitude) is used throughout, which
      moots the z=4.23 print-orientation dispute.
    mode="larger": the larger one-sided log width (the sensitivity arm)."""
    v = np.asarray(v, float)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    up = np.log(1.0 + hi / v)
    down = -np.log(1.0 - lo / v)
    if mode == "mean":
        return 0.5 * (up + down)
    if mode == "larger":
        return np.maximum(up, down)
    raise ValueError(f"unknown mode {mode!r}")


# --------------------------------------------------------------------------- #
#  POW10 Table 4 (arXiv:0912.0292 v1; ar5iv fetch, verified: implementer       #
#  2026-07-18 + both designers)                                                #
# --------------------------------------------------------------------------- #
# Columns verbatim from Table 4 (the tau912>=2, S/N=2 sample; six z bins).
# l(X) = m/dX exactly (self-check below); l(z)->l(X) conversion = l(z)*dz/dX (recomputed
# and asserted below, fallback policy (d): conversions never inherited).
# ERROR ORIENTATION: the printed magnitude pairs are stored as (lo, hi) with hi = the
# LARGER member. Gehrels Poisson errors on the printed m reproduce every magnitude
# (self-check below) and Gehrels upper > lower always, which resolves the disputed
# z=4.23 orientation (+0.20 up / -0.16 down physically); the ar5iv render shows the
# smaller member as the superscript uniformly in every row (a rendering artifact).
POW10_T4 = dict(
    z_lo=np.array([3.30, 3.40, 3.50, 3.65, 3.90, 4.10]),
    z_hi=np.array([3.40, 3.50, 3.65, 3.90, 4.10, 4.40]),
    z_bar=np.array([3.35, 3.47, 3.58, 3.74, 3.97, 4.23]),
    dX=np.array([25.7, 49.2, 111.7, 109.7, 41.9, 28.3]),
    dz=np.array([6.8, 12.9, 29.0, 27.9, 10.4, 6.8]),
    m=np.array([9, 28, 46, 57, 30, 22]),
    lz=np.array([1.31, 2.17, 1.59, 2.05, 2.89, 3.22]),
    lz_err_lo=np.array([0.43, 0.40, 0.23, 0.27, 0.52, 0.68]),
    lz_err_hi=np.array([0.60, 0.49, 0.27, 0.31, 0.63, 0.84]),
    lx=np.array([0.35, 0.57, 0.41, 0.52, 0.72, 0.78]),
    lx_err_lo=np.array([0.11, 0.11, 0.06, 0.07, 0.13, 0.16]),
    lx_err_hi=np.array([0.16, 0.13, 0.07, 0.08, 0.16, 0.20]),
    estimand=("cumulative l(X), tau912>=2 == N_HI>=10^17.5, INCLUDES subDLAs (SLLS) and "
              "DLAs (POW10 sec 2 verbatim)"),
    cite=f"arXiv:{ARXIV['POW10']} (POW10) Table 4, v1",
    cosmology=dict(omega_m=0.3, omega_l=0.7, h=0.72),
    verified="implementer ar5iv fetch 2026-07-18 + both designers (v1 level only)",
)

# Journal-only fixes (IOP paywalled; sibling session's journal read, UNVERIFIED in-house).
# Fallback policy (a): default arm keeps v1 values with the v1-journal delta added in
# quadrature to that point's sigma; journal_fix=True adopts the journal read.
POW10_JOURNAL_FIXES = dict(
    lx_397=0.70, m_397=29, dX_397=41.3,        # journal: 0.70 (v1: 0.72) at z_bar=3.97
    err_423_sym=0.18,                          # journal symmetric read at z=4.23 (v1 mean 0.18)
    delta_397=0.02,                            # |0.72 - 0.70| quadrature widening, default arm
    status="UNVERIFIED (IOP paywall; ar5iv serves v1; sibling journal read). PI decision 6.",
)

# O'Meara et al. 2013 (arXiv:1204.3093). Weighted mean over 2.0<z<2.5 (hybrid with
# Ribaudo+2011): VERIFIED verbatim (implementer + both designers). Table 5 combined row
# (self-consistent single row): designer-report-only (Bayesian designer ar5iv fetch;
# implementer fetch truncated before Table 5). Abscissa z_bar=2.21 = Table 5 <z> (same
# designer read); the deployed 2.4 was the f(N) pivot, not the measurement z.
OMEARA13_WMEAN = dict(z_bar=2.21, lx=0.29, err=0.05,
                      verified="implementer+both designers (value); z_bar from Table 5 "
                               "designer read")
OMEARA13_T5_COMBINED = dict(z_bar=2.21, lx=0.28, err=0.06,
                            verified="Bayesian designer ar5iv read ONLY (not in-house)")
OMEARA13_T9_UNVERIFIED_EXCLUDED = dict(
    z_bar=2.23, lx=0.30, err=0.07,
    status="UNVERIFIED by any in-house fetch — EXCLUDED from all fits (fallback policy (b))")

# Fumagalli et al. 2013 (arXiv:1308.1101), non-colour-selected tau>=2 at z~2.8:
# VERIFIED verbatim (implementer + both designers). The colour-selected value is a
# one-signed +55% selection systematic, recorded as a hedge, NEVER folded into fits.
FUMAGALLI13 = dict(
    z_bar=2.80, lz=1.21, lz_err=0.28, dz_dX=0.27, lx=0.33, err=0.08,
    colour_selected_lx=0.51, colour_selected_err=0.13,
    estimand="cumulative l(X), tau>=2 == N_HI>=10^17.5",
    cite=f"arXiv:{ARXIV['FUMAGALLI13']}",
    verified="implementer ar5iv fetch 2026-07-18 + both designers",
)


def lls_compilation(journal_fix=False, omeara_variant="wmean"):
    """The 8-point cumulative tau>=2 l(X) compilation (O'Meara + Fumagalli + 6 POW10),
    with the unverified-point fallback switches.

    journal_fix=False (default): POW10 v1 values, with the v1-journal delta (0.02 at
      z=3.97) added in quadrature to BOTH error sides of that point (bracket widening).
    journal_fix=True: the sibling's journal read (0.70 at z=3.97; symmetric 0.18 at 4.23).
    omeara_variant: "wmean" (0.29+/-0.05, hybrid weighted mean, verified in-house) or
      "table5" (0.28+/-0.06, self-consistent single row, designer-read only). "table9"
      raises (EXCLUDED, unverified). PI decision 3.
    Returns a dict with z_bar, lx, err_lo, err_hi (Gehrels orientation: hi = upper),
    source label per point, estimand + cite."""
    if omeara_variant == "wmean":
        om = OMEARA13_WMEAN
    elif omeara_variant == "table5":
        om = OMEARA13_T5_COMBINED
    elif omeara_variant == "table9":
        raise ValueError("O'Meara Table 9 is UNVERIFIED and EXCLUDED (fallback policy (b))")
    else:
        raise ValueError(f"unknown omeara_variant {omeara_variant!r}")

    p = POW10_T4
    lx = np.asarray(p["lx"], float).copy()
    elo = np.asarray(p["lx_err_lo"], float).copy()
    ehi = np.asarray(p["lx_err_hi"], float).copy()
    if journal_fix:
        lx[4] = POW10_JOURNAL_FIXES["lx_397"]
        elo[5] = ehi[5] = POW10_JOURNAL_FIXES["err_423_sym"]
    else:
        d = POW10_JOURNAL_FIXES["delta_397"]
        elo[4] = np.hypot(elo[4], d)
        ehi[4] = np.hypot(ehi[4], d)

    return dict(
        z_bar=np.concatenate([[om["z_bar"], FUMAGALLI13["z_bar"]], p["z_bar"]]),
        lx=np.concatenate([[om["lx"], FUMAGALLI13["lx"]], lx]),
        err_lo=np.concatenate([[om["err"], FUMAGALLI13["err"]], elo]),
        err_hi=np.concatenate([[om["err"], FUMAGALLI13["err"]], ehi]),
        source=(["OMeara13:" + omeara_variant, "Fumagalli13"] +
                [f"POW10:z{z:.2f}" for z in p["z_bar"]]),
        estimand=("cumulative l(X), tau912>=2 == N_HI>=10^17.5, includes subDLA+DLA "
                  "(NOT the binned [17.2,19.0) LLS class rate — needs the kernel)"),
        cite=(f"arXiv:{ARXIV['OMEARA13']} / arXiv:{ARXIV['FUMAGALLI13']} / "
              f"arXiv:{ARXIV['POW10']}"),
        journal_fix=bool(journal_fix), omeara_variant=omeara_variant,
    )


# The default compilation (v1 + delta-widening, O'Meara weighted mean).
ELL_X_CUMULATIVE_GE17P5_TAU2 = lls_compilation()


# --------------------------------------------------------------------------- #
#  Zafar+2013 Table 3 (arXiv:1307.0602; verified verbatim: implementer         #
#  2026-07-18 + both designers)                                                #
# --------------------------------------------------------------------------- #
# The >=19.0 block. The printed d(n)/dX column is CUMULATIVE (caption verbatim: "The
# d(n)/dz and d(n)/dX for sub-DLAs is measured from the contribution of both DLAs and
# sub-DLAs"); the PURE binned [19.0,20.3) rate is n/dX with exact Poisson counts.
# lx_gt203 = the row-matched printed rate of the logN>=20.3 block (caption: "The DLA
# results are taken from the Peroux et al. (2003b) sample"; that block's own z_bars are
# dla_block_z_bar, slightly offset).
ELL_X_SUBDLA_BINNED_19P0_20P3_ZAFAR13_T3 = dict(
    z_lo=np.array([1.51, 2.00, 2.50, 3.00, 3.50, 4.00]),
    z_hi=np.array([2.00, 2.50, 3.00, 3.50, 4.00, 5.00]),
    z_bar=np.array([1.80, 2.26, 2.76, 3.21, 3.72, 4.18]),
    dz=np.array([29.9, 49.2, 47.1, 33.7, 23.3, 9.9]),
    dX=np.array([87.3, 156.8, 162.8, 124.5, 91.7, 41.0]),
    n=np.array([4, 11, 24, 23, 19, 8]),
    lx_gt19=np.array([0.12, 0.14, 0.21, 0.26, 0.30, 0.30]),        # printed, CUMULATIVE >=19.0
    lx_gt19_err=np.array([0.04, 0.03, 0.04, 0.05, 0.06, 0.09]),
    lx_gt203=np.array([0.08, 0.07, 0.06, 0.08, 0.10, 0.10]),       # printed >=20.3 (Peroux DLA)
    lx_gt203_err=np.array([0.02, 0.01, 0.01, 0.02, 0.02, 0.03]),
    dla_block_z_bar=np.array([1.84, 2.27, 2.73, 3.25, 3.77, 4.20]),
    dla_block_n=np.array([19, 26, 18, 18, 19, 10]),
    estimand=("binned [19.0, 20.3) sub-DLA counts; the pure rate is n/dX with exact "
              "Poisson counts (the printed >=19.0 column is CUMULATIVE, never fit directly)"),
    cite=f"arXiv:{ARXIV['ZAFAR13']} (Zafar+2013) Table 3",
    verified="implementer ar5iv fetch 2026-07-18 + both designers (verbatim, both blocks)",
)


# --------------------------------------------------------------------------- #
#  PW09 Table 1 (arXiv:0811.2003; verified: implementer 2026-07-18 +           #
#  Bayesian designer)                                                          #
# --------------------------------------------------------------------------- #
# logN>=20.3 = the TOP disjoint class: cumulative == binned, no kernel needed.
# l(X) = m/dX exactly; printed asymmetric errors = Gehrels Poisson on m (self-check).
ELL_X_DLA_GE20P3_PW09_T1 = dict(
    z_lo=np.array([2.2, 2.4, 2.7, 3.0, 3.5, 4.0]),
    z_hi=np.array([2.4, 2.7, 3.0, 3.5, 4.0, 5.5]),
    z_bar=np.array([2.31, 2.57, 2.86, 3.22, 3.70, 4.39]),
    m=np.array([79, 132, 169, 227, 86, 46]),
    dz=np.array([514.4, 717.5, 723.7, 732.2, 291.3, 103.6]),
    dX=np.array([1652.7, 2405.8, 2539.7, 2702.5, 1139.2, 432.8]),
    lx=np.array([0.048, 0.055, 0.067, 0.084, 0.075, 0.106]),
    err_lo=np.array([0.005, 0.005, 0.005, 0.006, 0.008, 0.016]),
    err_hi=np.array([0.006, 0.005, 0.006, 0.006, 0.009, 0.018]),
    estimand=("binned l(X), logN>=20.3 (DLA; the top disjoint class — cumulative == "
              "binned by construction)"),
    cite=f"arXiv:{ARXIV['PW09']} (Prochaska & Wolfe 2009) Table 1",
    verified="implementer ar5iv fetch 2026-07-18 (m, dX, lx) + Bayesian designer (errors)",
)


# --------------------------------------------------------------------------- #
#  TOMBSTONES — the retired wrong-object arrays. Greppable, regression-tested, #
#  NEVER fit. Kept so the defects cannot be silently reinstated.               #
# --------------------------------------------------------------------------- #
LLS_TAU2_OLD_DEFECTS = dict(
    z=np.array([2.4, 2.8, 3.35, 3.47, 3.58, 3.74, 3.97, 4.23]),
    lx=np.array([0.29, 0.33, 0.35, 0.57, 0.41, 0.52, 0.72, 0.78]),
    err=np.array([0.05, 0.08, 0.14, 0.12, 0.07, 0.08, 0.15, 0.19]),
    defect=("WRONG-OBJECT: this is the CUMULATIVE tau912>=2 l(X) (N>=10^17.5, includes "
            "subDLAs+DLAs) that was deployed as the binned LLS class rate "
            "(derive_hcd_lls_width.py LIT['LLS'] -> HCD_LIT_DNDX_LAW['LLS'] (0.0201, 2.127)); "
            "the O'Meara abscissa 2.4 is the f(N) pivot, not the measurement z (~2.21); "
            "symmetric errors are ad-hoc roundings of the asymmetric pairs."),
)

ZAFAR13_T3_DLA_BLOCK_WRONG_SUBDLA_LABEL = dict(
    z=np.array([2.27, 2.73, 3.25, 3.77, 4.20]),
    lx=np.array([0.07, 0.06, 0.08, 0.10, 0.10]),
    err=np.array([0.01, 0.01, 0.02, 0.02, 0.03]),
    defect=("WRONG-OBJECT: rows 2-6 of Zafar+2013 Table 3's logN>=20.3 block = the "
            "Peroux et al. 2003b DLA rates, deployed as 'subDLA' "
            "(derive_hcd_lls_width.py LIT['subDLA'] -> HCD_LIT_DNDX_LAW['subDLA'] "
            "(0.0211, 0.937)). The true sub-DLA estimand is n/dX of the >=19.0 block "
            "counts (4,11,24,23,19,8)."),
)


# --------------------------------------------------------------------------- #
#  Import-time transcription self-checks (raise on ANY drift)                  #
# --------------------------------------------------------------------------- #
def _check_zafar_subdla(a):
    n = np.asarray(a["n"], float)
    dX = np.asarray(a["dX"], float)
    assert int(np.sum(n)) == 89, f"Zafar sub-DLA counts sum {np.sum(n)} != 89 (abstract)"
    rate = n / dX
    two_route = np.asarray(a["lx_gt19"], float) - np.asarray(a["lx_gt203"], float)
    bad = np.abs(rate - two_route) > 0.025
    assert not np.any(bad), (
        f"Zafar two-route sub-DLA disagreement > 0.025 at rows {np.where(bad)[0]}: "
        f"n/dX={rate} vs l>19-l>20.3={two_route}")


def _check_pow10(t):
    m = np.asarray(t["m"], float)
    dX = np.asarray(t["dX"], float)
    dz = np.asarray(t["dz"], float)
    lx = np.asarray(t["lx"], float)
    lz = np.asarray(t["lz"], float)
    # count identity l(X) = m/dX (table rounding 0.005)
    assert np.all(np.abs(m / dX - lx) <= 0.005 + 1e-12), (
        f"POW10 l(X) != m/dX: {m / dX} vs {lx}")
    # the l(z)->l(X) conversion recomputed from the survey's own per-bin path ratio
    # (fallback policy (d): all six recomputed, never inherited). Tolerance 0.008: the
    # printed l(z) is the survey-sensitivity-weighted value (not exactly m/dz), so the
    # round-trip through the printed 2-decimal l(z) carries up to ~0.007 (worst z=4.23).
    assert np.all(np.abs(lz * dz / dX - lx) <= 0.008 + 1e-12), (
        f"POW10 l(z)*dz/dX != l(X): {lz * dz / dX} vs {lx}")
    # Gehrels magnitudes reproduce the printed error pairs; upper >= lower
    lo_c, hi_c = gehrels_errors(m)
    assert np.all(np.abs(hi_c / dX - np.asarray(t["lx_err_hi"], float)) <= 0.015)
    assert np.all(np.abs(lo_c / dX - np.asarray(t["lx_err_lo"], float)) <= 0.015)
    assert np.all(hi_c >= lo_c)


def _check_pw09(t):
    m = np.asarray(t["m"], float)
    dX = np.asarray(t["dX"], float)
    assert np.all(np.abs(m / dX - np.asarray(t["lx"], float)) <= 0.0005 + 1e-12)
    lo_c, hi_c = gehrels_errors(m)
    assert np.all(np.abs(hi_c / dX - np.asarray(t["err_hi"], float)) <= 0.0015)
    assert np.all(np.abs(lo_c / dX - np.asarray(t["err_lo"], float)) <= 0.0015)


def _check_fumagalli(f):
    assert abs(f["lz"] * f["dz_dX"] - f["lx"]) <= 0.005
    assert abs(f["lz_err"] * f["dz_dX"] - f["err"]) <= 0.005


def _run_self_checks():
    _check_zafar_subdla(ELL_X_SUBDLA_BINNED_19P0_20P3_ZAFAR13_T3)
    _check_pow10(POW10_T4)
    _check_pw09(ELL_X_DLA_GE20P3_PW09_T1)
    _check_fumagalli(FUMAGALLI13)


_run_self_checks()


# --------------------------------------------------------------------------- #
#  Display points for the plot/consumer scripts (single-source rewire)         #
# --------------------------------------------------------------------------- #
_DERIVATION_JSON = __file__.replace("lit_dndx.py", "hcd_lit_dndx_corrected.json")


def lit_points_for_display():
    """Per-class (z, value, err, source) display tuples for the plot/consumer scripts,
    built from the CORRECTED estimands (kills the wrong-object LIT-dict hard-coding that
    was repeated across six scripts):
      LLS    = the ADOPTED kernel-corrected binned [17.2,19.0) points from the committed
               derivation JSON (K1a r(z) applied to the cumulative compilation; the
               errors are the symmetrized log-space sigmas x the corrected values),
      subDLA = Zafar+2013 Table 3 [19.0,20.3) counts as n/dX with Gehrels errors,
      DLA    = PW09 Table 1 binned points (upper errors, the deployed convention).
    Also returns an 'estimand' sub-dict for the plot scripts' estimand asserts."""
    import json
    with open(_DERIVATION_JSON) as fh:
        j = json.load(fh)
    cp = j["adopted_law"]["corrected_points"]
    z_l = np.asarray(cp["z_bar"], float)
    v_l = np.asarray(cp["lx"], float)
    e_l = np.asarray(cp["sig_log"], float) * v_l
    az = ELL_X_SUBDLA_BINNED_19P0_20P3_ZAFAR13_T3
    rate = np.asarray(az["n"], float) / np.asarray(az["dX"], float)
    lo_c, hi_c = gehrels_errors(az["n"])
    e_s = 0.5 * (lo_c + hi_c) / np.asarray(az["dX"], float)
    ad = ELL_X_DLA_GE20P3_PW09_T1
    return {
        "LLS": (z_l, v_l, e_l,
                f"POW10/O'Meara13/Fumagalli13, kernel-corrected to binned [17.2,19.0) "
                f"({j['kernel_chosen']})"),
        "subDLA": (np.asarray(az["z_bar"], float), rate, e_s,
                   "Zafar+2013 Table 3 counts, binned [19.0,20.3)"),
        "DLA": (np.asarray(ad["z_bar"], float), np.asarray(ad["lx"], float),
                np.asarray(ad["err_hi"], float), "Prochaska & Wolfe 2009 Table 1"),
        "estimand": {"LLS": "binned [17.2,19.0) (kernel-corrected)",
                     "subDLA": "binned [19.0,20.3) counts n/dX",
                     "DLA": "binned >=20.3"},
        "estimand_id": dict(ESTIMAND_ID),
        "estimand_label": dict(ESTIMAND_LABEL),
    }


# =========================================================================== #
#  DISPLAY-PATH ESTIMAND GUARD (paper-agent request 2026-07-20).              #
#  The LAW boundary is guarded by inference.assert_dndx_law_estimand; this    #
#  block guards the DISPLAY boundary: the retired tau_LL>=2 LLS object fails  #
#  loudly if requested by name, and consumer legends can be asserted against  #
#  the deployed estimand wording (the silent-substitution hazard: corrected   #
#  points rendering under old hard-coded tau>=2 labels with no error).       #
# =========================================================================== #
# Machine ids, kept test-equal to inference.HCD_LIT_DNDX_ESTIMAND (the law ids).
ESTIMAND_ID = {"LLS": "binned_17.2_19.0", "subDLA": "binned_19.0_20.3",
               "DLA": "binned_ge20.3"}
# The exact caption/legend wording of record. Paper consumers must use these
# VERBATIM (PI directive 2026-07-20 baseline item 2: no paraphrase).
ESTIMAND_LABEL = {
    "LLS": ("kernel-corrected (K1a) binned LLS incidence, "
            "17.2 <= log10 N_HI < 19.0"),
    "subDLA": ("binned sub-DLA incidence from raw counts n/dX, "
               "19.0 <= log10 N_HI < 20.3"),
    "DLA": "binned DLA incidence, log10 N_HI >= 20.3",
}
# Tokens whose appearance in a display label/source string means the retired
# cumulative tau_LL>=2 object leaked back into a legend.
RETIRED_DISPLAY_TOKENS = ("tau_LL", "\\tau_{\\rm LL}", "τ_LL", "tau912",
                          "tau>=2", "tau >= 2", "τ≥2", "τ ≥ 2")
# Required per-class tokens: a label that lost its binned column-density range
# is no longer stating the estimand.
_REQUIRED_LABEL_TOKENS = {"LLS": ("17.2", "19.0"), "subDLA": ("19.0", "20.3"),
                          "DLA": ("20.3",)}
_RETIRED_ESTIMAND_IDS = {
    "cumulative_tau2_ge17.5": ("the cumulative tau912>=2 l(X) display object was RETIRED "
                               "2026-07-18 (wrong kernel for the binned LLS class)"),
}
# Names under which the retired tau>=2 display object might plausibly be requested.
_RETIRED_NAMES = {
    "LIT_TAU2", "LLS_TAU2_COMPILATION", "LLS_TAU2_DISPLAY", "lit_points_tau2",
    "lls_tau2_points_for_display", "LIT_OLD", "LLS_CUMULATIVE_DISPLAY",
}


def __getattr__(name):
    """Module-level tombstone: requesting the retired tau_LL>=2 display object by any
    of its plausible names fails loudly with the corrected pointer instead of an
    ordinary AttributeError a caller might silently except."""
    if name in _RETIRED_NAMES:
        raise AttributeError(
            f"lit_dndx.{name}: RETIRED. The cumulative tau_LL>=2 LLS display object was "
            f"retired 2026-07-18 (wrong-object/kernel bugs; see LLS_TAU2_OLD_DEFECTS). "
            f"Use lit_points_for_display() (estimand ids {ESTIMAND_ID}) and the "
            f"ESTIMAND_LABEL wording verbatim.")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def assert_display_estimand(cls, estimand_id, where):
    """Fail loudly if a display consumer requests a retired estimand or mislabels the
    deployed one. Mirrors inference.assert_dndx_law_estimand at the display boundary."""
    if estimand_id in _RETIRED_ESTIMAND_IDS:
        raise AssertionError(
            f"[{where}] display estimand {estimand_id!r} for class {cls!r} is RETIRED: "
            f"{_RETIRED_ESTIMAND_IDS[estimand_id]}. Deployed id: {ESTIMAND_ID[cls]!r}.")
    if estimand_id != ESTIMAND_ID[cls]:
        raise AssertionError(
            f"[{where}] display estimand mismatch for class {cls!r}: requested "
            f"{estimand_id!r}, deployed {ESTIMAND_ID[cls]!r}.")


def assert_display_labels(labels_by_class, where):
    """Assert consumer legend/caption strings: no retired token anywhere, and each
    class label still carries its binned column-density range tokens."""
    for c, lab in labels_by_class.items():
        for tok in RETIRED_DISPLAY_TOKENS:
            if tok in lab:
                raise AssertionError(
                    f"[{where}] class {c!r} label contains RETIRED token {tok!r}: {lab!r}")
        for tok in _REQUIRED_LABEL_TOKENS.get(c, ()):
            if tok not in lab:
                raise AssertionError(
                    f"[{where}] class {c!r} label is missing the required estimand token "
                    f"{tok!r}: {lab!r}")


# =========================================================================== #
#  FIT MACHINERY (spec sec 4): shared WLS (the deployed-law regression         #
#  anchor), Poisson GLM (the adopted count-class estimator), log-space GLS     #
#  with the common-mode kernel covariance block, and the fit ordering.         #
# =========================================================================== #
def wls_powerlaw(z, v, e, z_pivot=Z_PIVOT):
    """WLS fit log(dN/dX) = log A + gamma*log(1+z), weighting by the lit errors in log space.
    Returns (A, gamma, cov[2x2], sigma_logA, sigma_gamma, frac_norm_err_at_pivot, frac_scatter).

    VERBATIM port of scripts/derive_hcd_lls_width.py:44-78 (the shared single
    implementation; the meta-reviewer proved this WLS reproduces the deployed laws
    bit-for-bit). ``z_pivot`` generalizes the script's module-level Z_PIVOT=3.0."""
    z = np.asarray(z, float); v = np.asarray(v, float); e = np.asarray(e, float)
    x = np.log(1.0 + z)                       # regressor
    y = np.log(v)                             # log dN/dX
    sig_y = e / v                             # fractional error -> log-space error
    w = 1.0 / sig_y**2                        # WLS weights
    # design matrix [1, x] ; params [logA, gamma]
    X = np.vstack([np.ones_like(x), x]).T
    W = np.diag(w)
    XtWX = X.T @ W @ X
    XtWy = X.T @ W @ y
    beta = np.linalg.solve(XtWX, XtWy)        # [logA, gamma]
    cov = np.linalg.inv(XtWX)                 # parameter covariance (chi2-based, NOT rescaled)
    logA, gamma = beta
    # residuals + reduced chi2
    resid = y - X @ beta
    dof = max(len(z) - 2, 1)
    chi2 = float(np.sum(w * resid**2))
    chi2_red = chi2 / dof
    # NORMALIZATION uncertainty AT THE PIVOT z=3 (the variance of the FIT at x_p = log(1+3)):
    xp = np.array([1.0, np.log(1.0 + z_pivot)])
    var_logfit_pivot = float(xp @ cov @ xp)    # variance of log(dN/dX_fit) at pivot
    frac_norm_err = np.sqrt(var_logfit_pivot)  # ~ fractional error of the fit at the pivot
    # if chi2_red>1, the points scatter MORE than their error bars -> inflate by sqrt(chi2_red)
    frac_norm_err_infl = frac_norm_err * np.sqrt(max(chi2_red, 1.0))
    # per-point SCATTER about the fit (in fractional/log units) - the excess-dispersion hedge
    frac_scatter_rms = float(np.sqrt(np.mean(resid**2)))    # rms of log-residuals = frac scatter
    A = np.exp(logA)
    return dict(A=A, gamma=gamma, cov=cov, logA=logA,
                sigma_logA=np.sqrt(cov[0,0]), sigma_gamma=np.sqrt(cov[1,1]),
                chi2_red=chi2_red,
                frac_norm_err=frac_norm_err, frac_norm_err_infl=frac_norm_err_infl,
                frac_scatter=frac_scatter_rms)


def poisson_glm_powerlaw(z, n, dX, z_pivot=Z_PIVOT, max_iter=200, tol=1e-12):
    """Poisson GLM with log link: n_i ~ Poisson(dX_i * A_p * ((1+z_i)/(1+z_pivot))^gamma).

    The ADOPTED estimator for count classes (spec sec 4a: exact at n as low as 4; the
    log-WLS E[ln n] < ln E[n] bias is the likely source of the sibling transplant's
    inconsistency). Plain numpy Newton/IRLS; offset = ln dX; pivot parametrization
    (lnA_p, gamma) so the amplitude error decorrelates from gamma.

    Returns dict(A_pivot, A (=A_p/(1+z_p)^gamma, the deployed dict convention), gamma,
    cov (2x2 in (lnA_p, gamma)), sigma_lnAp, sigma_gamma, deviance, dof, deviance_dof,
    mu_hat, converged, n_iter)."""
    z = np.asarray(z, float); n = np.asarray(n, float); dX = np.asarray(dX, float)
    x = np.log((1.0 + z) / (1.0 + z_pivot))
    X = np.vstack([np.ones_like(x), x]).T
    off = np.log(dX)
    # init from a crude log rate regression (n floored at 0.5 for the init only)
    y0 = np.log(np.maximum(n, 0.5)) - off
    beta = np.linalg.lstsq(X, y0, rcond=None)[0]
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        mu = np.exp(off + X @ beta)
        W = mu                                       # Poisson IRLS weights
        XtWX = X.T @ (W[:, None] * X)
        score = X.T @ (n - mu)
        step = np.linalg.solve(XtWX, score)
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            converged = True
            break
    mu = np.exp(off + X @ beta)
    cov = np.linalg.inv(X.T @ (mu[:, None] * X))
    with np.errstate(divide="ignore", invalid="ignore"):
        dev_terms = np.where(n > 0, n * np.log(n / mu), 0.0) - (n - mu)
    deviance = float(2.0 * np.sum(dev_terms))
    dof = max(len(n) - 2, 1)
    A_p = float(np.exp(beta[0]))
    gamma = float(beta[1])
    return dict(A_pivot=A_p, A=A_p / (1.0 + z_pivot) ** gamma, gamma=gamma, cov=cov,
                sigma_lnAp=float(np.sqrt(cov[0, 0])), sigma_gamma=float(np.sqrt(cov[1, 1])),
                deviance=deviance, dof=dof, deviance_dof=deviance / dof, mu_hat=mu,
                converged=converged, n_iter=it, estimator="poisson_glm", z_pivot=z_pivot)


def poisson_vs_logwls(z, n, dX, z_pivot=Z_PIVOT):
    """The documented Poisson-vs-logWLS comparison (spec 4a: the implement phase must not
    silently fall back to WLS). Runs BOTH estimators on the same count points; the WLS
    uses the naive sqrt(n)/dX error bars on the rate n/dX."""
    z = np.asarray(z, float); n = np.asarray(n, float); dX = np.asarray(dX, float)
    glm = poisson_glm_powerlaw(z, n, dX, z_pivot=z_pivot)
    rate = n / dX
    err = np.sqrt(n) / dX
    wls = wls_powerlaw(z, rate, err, z_pivot=z_pivot)
    wls_Ap = float(wls["A"] * (1.0 + z_pivot) ** wls["gamma"])
    return dict(glm=glm, wls=wls, wls_A_pivot=wls_Ap,
                delta_A_frac=wls_Ap / glm["A_pivot"] - 1.0,
                delta_gamma=float(wls["gamma"]) - glm["gamma"])


def gls_powerlaw_log(z, v, sig_log, s_common=0.0, sigma_eta=0.0, z_pivot=Z_PIVOT,
                     gamma_fixed=None):
    """Log-space GLS power-law fit with the kernel covariance blocks (spec sec 4b):
    Sigma = diag(sig_log^2) + s_common^2 * J (fully-correlated common-mode r(3) block)
    + sigma_eta^2 * x x^T (the eta tilt, marginalized analytically). Pivot-parametrized
    (lnA_p, gamma) with x = ln((1+z)/(1+z_pivot)).

    Both extra blocks lie in the column space of the design, so the POINT ESTIMATE is
    invariant; they inflate sigma_lnAp / sigma_gamma respectively (the r(3) decomposition
    rationale: common mode hits the normalization, the tilt hits the slope).

    ``gamma_fixed``: CONSTRAINED variant (PI 2026-07-18 decision 1c — the deployed LLS law
    keeps the deployed z-slope): the slope is pinned, only lnA_p is fit, under the SAME
    Sigma. Closed form lnA_p = (1' Si (y - gamma*x)) / (1' Si 1); sigma_gamma = 0 and the
    returned dict carries gamma_fixed=True. The free fit remains the consistency evidence."""
    z = np.asarray(z, float); v = np.asarray(v, float)
    sig = np.asarray(sig_log, float) * np.ones_like(z)
    x = np.log((1.0 + z) / (1.0 + z_pivot))
    y = np.log(v)
    S = np.diag(sig ** 2) + (s_common ** 2) * np.ones((len(z), len(z))) \
        + (sigma_eta ** 2) * np.outer(x, x)
    Si = np.linalg.inv(S)
    if gamma_fixed is None:
        X = np.vstack([np.ones_like(x), x]).T
        XtSiX = X.T @ Si @ X
        cov = np.linalg.inv(XtSiX)
        beta = cov @ (X.T @ Si @ y)
        resid = y - X @ beta
        gamma = float(beta[1])
        lnAp = float(beta[0])
        sigma_lnAp = float(np.sqrt(cov[0, 0]))
        sigma_gamma = float(np.sqrt(cov[1, 1]))
        dof = max(len(z) - 2, 1)
        fixed = False
    else:
        gamma = float(gamma_fixed)
        one = np.ones_like(x)
        denom = float(one @ Si @ one)
        lnAp = float(one @ Si @ (y - gamma * x)) / denom
        cov = np.array([[1.0 / denom, 0.0], [0.0, 0.0]])
        resid = y - (lnAp + gamma * x)
        sigma_lnAp = float(np.sqrt(1.0 / denom))
        sigma_gamma = 0.0
        dof = max(len(z) - 1, 1)                     # one fitted parameter
        fixed = True
    chi2 = float(resid @ Si @ resid)
    A_p = float(np.exp(lnAp))
    return dict(A_pivot=A_p, A=A_p / (1.0 + z_pivot) ** gamma, gamma=gamma, cov=cov,
                sigma_lnAp=sigma_lnAp, sigma_gamma=sigma_gamma,
                chi2=chi2, dof=dof, chi2_red=chi2 / dof,
                frac_norm_err=sigma_lnAp,
                frac_norm_err_infl=float(sigma_lnAp * np.sqrt(max(chi2 / dof, 1.0))),
                frac_scatter=float(np.sqrt(np.mean(resid ** 2))),
                estimator="gls_log" if not fixed else "gls_log_gamma_fixed",
                z_pivot=z_pivot, gamma_fixed=fixed,
                s_common=float(s_common), sigma_eta=float(sigma_eta))


# --------------------------------------------------------------------------- #
#  Consistency statistics + gates                                              #
# --------------------------------------------------------------------------- #
def law_vs_points_stats(A, gamma, z, v, sig_abs):
    """Reporting stats of a law dN/dX = A(1+z)^gamma against points (v +- sig_abs):
    weighted-mean and weighted-rms fractional residuals (w = 1/sig_frac^2) and chi2.
    NOTE: for count-scale data the weighted rms is ~the per-point Poisson noise (~10-20%)
    for ANY law, and signed residuals can cancel in the weighted mean — so the ADJUDICATING
    internal-consistency gate is law_consistency_vs_refit, not these (both are reported)."""
    z = np.asarray(z, float); v = np.asarray(v, float); sig = np.asarray(sig_abs, float)
    fit = A * (1.0 + z) ** gamma
    r = fit / v - 1.0
    w = (v / sig) ** 2
    return dict(wmean_frac_resid=float(np.sum(w * r) / np.sum(w)),
                wrms_frac_resid=float(np.sqrt(np.sum(w * r ** 2) / np.sum(w))),
                chi2=float(np.sum(((fit - v) / sig) ** 2)), n=len(z))


def law_consistency_vs_refit(A, gamma, refit, z_eval):
    """INTERNAL-CONSISTENCY GATE (the transplant-killer): a deployed law must equal what
    the recorded estimator produces on its own input points. ``refit`` = the estimator
    result dict (poisson_glm_powerlaw / wls_powerlaw / gls_powerlaw_log). Returns the max
    fractional deviation of the law from the refit law over ``z_eval``."""
    z = np.asarray(z_eval, float)
    law = A * (1.0 + z) ** gamma
    if "A_pivot" in refit:
        ref = refit["A"] * (1.0 + z) ** refit["gamma"]
    else:
        ref = refit["A"] * (1.0 + z) ** refit["gamma"]
    dev = law / ref - 1.0
    return dict(max_frac_dev=float(np.max(np.abs(dev))), frac_dev=dev)


# --------------------------------------------------------------------------- #
#  Kernel constructions that need only laws (K2 cap, K3 cap)                   #
# --------------------------------------------------------------------------- #
POW10_L_SLLS_LAW = dict(A=0.066, gamma=1.70)   # POW10's OWN adopted l_SLLS(z) (per unit z!)


def lx_slls_pow10(z):
    """POW10's own l_SLLS law converted to X units: 0.066(1+z)^1.70 * dz/dX(z) with
    POW10's stated cosmology (Om=0.3, OL=0.7). K2 input (verified an ASSUMPTION of
    theirs, not a flat 0.20)."""
    z = np.asarray(z, float)
    lz = POW10_L_SLLS_LAW["A"] * (1.0 + z) ** POW10_L_SLLS_LAW["gamma"]
    return lz / dX_dz(z, omega_m=0.3, omega_l=0.7)


def cap_fraction_from_laws(z, sub_law, dla_law, cum_law, slls_own_words=False):
    """Cap-removal fraction 1 - [l_sub(z)+l_DLA(z)]/l_cum(z). With slls_own_words=True the
    sub-DLA share uses POW10's own l_SLLS law (K2); else the fitted sub-DLA law (K3)."""
    z = np.asarray(z, float)
    l_sub = lx_slls_pow10(z) if slls_own_words else sub_law["A"] * (1.0 + z) ** sub_law["gamma"]
    l_dla = dla_law["A"] * (1.0 + z) ** dla_law["gamma"]
    l_cum = cum_law["A"] * (1.0 + z) ** cum_law["gamma"]
    return 1.0 - (l_sub + l_dla) / l_cum


# --------------------------------------------------------------------------- #
#  The fit ordering (spec sec 4: subDLA GLM -> DLA GLM+WLS anchor ->           #
#  uncorrected cumulative GLS -> K3 -> corrected LLS GLS)                      #
# --------------------------------------------------------------------------- #
TELESCOPE_Z = (2.5, 3.0, 3.5, 4.0)


def corrected_width_variants(law_block, s_r3):
    """sigma_LLS width variants on the corrected points (PI decision 2).
    MEASUREMENT-ONLY = the deployed one_x convention max(norm_err_infl, scatter) from a
    DIAGONAL refit (no kernel covariance blocks); MEAS+KERNEL-COMMON-MODE = quadrature
    with s_r3."""
    pts = law_block["corrected_points"]
    diag = gls_powerlaw_log(pts["z_bar"], pts["lx"], pts["sig_log"])
    meas = max(diag["frac_norm_err_infl"], diag["frac_scatter"])
    return dict(meas_only=float(meas), norm_infl=float(diag["frac_norm_err_infl"]),
                scatter=float(diag["frac_scatter"]),
                with_kernel=float(np.hypot(meas, s_r3)))


def _fit_corrected_lls(comp, r_at, kernel_budget, err_mode="mean"):
    """Corrected-points GLS: y_i = ln(r(z_i) * l_i), Sigma = diag + common-mode blocks."""
    zb = np.asarray(comp["z_bar"], float)
    r = np.asarray(r_at(zb), float)
    v = np.asarray(comp["lx"], float) * r
    sig = symmetrize_log_errors(comp["lx"], comp["err_lo"], comp["err_hi"], mode=err_mode)
    kb = kernel_budget or {}
    fit = gls_powerlaw_log(zb, v, sig, s_common=kb.get("s_r3", 0.0),
                           sigma_eta=kb.get("sigma_eta", 0.0))
    stats = law_vs_points_stats(fit["A"], fit["gamma"], zb, v, sig * v)
    # kernel-CANDIDATE laws have no deployed referent at fit time; the adopted constrained
    # law is gated deployed-vs-refit in derive_hcd_dndx_corrected.py + test case 1
    return dict(fit=fit, corrected_points=dict(z_bar=zb, lx=v, sig_log=sig, r=r),
                stats=stats, consistency_vs_refit=None,
                A=fit["A"], gamma=fit["gamma"])


def _telescoping_check(lls_fit, floor_factor, sub_law, dla_law, cum_fit, z_grid=TELESCOPE_Z):
    """binned-LLS-law(z)/floor_factor(z) + subDLA law + DLA law vs the cumulative
    compilation law, in units of the compilation's log-error at z (pull)."""
    zg = np.asarray(z_grid, float)
    lls = lls_fit["A"] * (1.0 + zg) ** lls_fit["gamma"]
    F = np.asarray(floor_factor(zg), float)
    sub = sub_law["A"] * (1.0 + zg) ** sub_law["gamma"]
    dla = dla_law["A"] * (1.0 + zg) ** dla_law["gamma"]
    cum = cum_fit["A"] * (1.0 + zg) ** cum_fit["gamma"]
    total = lls / F + sub + dla
    # compilation error of the cumulative law at z (log space, chi2-inflated)
    x = np.log((1.0 + zg) / (1.0 + cum_fit.get("z_pivot", Z_PIVOT)))
    cov = np.asarray(cum_fit["cov"])
    var = np.array([np.array([1.0, xi]) @ cov @ np.array([1.0, xi]) for xi in x])
    sig_log = np.sqrt(var) * np.sqrt(max(cum_fit.get("chi2_red", 1.0), 1.0))
    pull = np.log(total / cum) / sig_log
    chi2 = float(np.sum((np.log(total / cum) / sig_log) ** 2))
    return dict(z=zg, total=total, cum=cum, ratio=total / cum, sig_log=sig_log,
                pull=pull, chi2=chi2)


def run_fit_ordering(floor_factor, r_k1=None, r_k1_smooth=None, kernel_budget=None,
                     journal_fix=False, omeara_variant="wmean", larger_side=False,
                     drop_z180=False, include_sensitivity=True, reference_laws=None):
    """The pinned fit ORDERING (spec sec 4): (1) subDLA + DLA count laws (Poisson GLM,
    independent); (2) UNCORRECTED cumulative LLS compilation (diagonal GLS = WLS); (3) K3
    cap from (1)+(2) x the PRIYA floor factor -> corrected LLS points -> GLS with the
    common-mode kernel covariance. K1a/K1b/K2 evaluated through the same steps as bracket
    arms when their r(z) inputs are supplied.

    floor_factor: callable z -> l([17.2,19.0))/l([17.5,19.0)) (PRIYA cache; a pinned
      constant only in kernel-free test paths).
    r_k1 / r_k1_smooth: callables z -> b(z) (per-point K1a / smooth K1b), optional.
    kernel_budget: dict(s_r3, sigma_eta) — the common-mode kernel covariance blocks.
    reference_laws: optional {class: (A, gamma)} of DEPLOYED laws; when given, each class's
      ``consistency_vs_refit`` compares that REFERENCE against the fresh refit (the real
      transplant-killer). Without a referent the field is None — never self-vs-self
      (2026-07-18 review meta finding 6: the old self-compare was vacuous).
    """
    err_mode = "larger" if larger_side else "mean"

    def _cons(cls, refit, z_eval):
        if reference_laws and cls in reference_laws:
            a_ref, g_ref = reference_laws[cls]
            return law_consistency_vs_refit(a_ref, g_ref, refit, z_eval)
        return None

    # (1a) subDLA: Poisson GLM on the verified Zafar counts
    az = ELL_X_SUBDLA_BINNED_19P0_20P3_ZAFAR13_T3
    keep = slice(1, None) if drop_z180 else slice(None)
    z_s, n_s, dX_s = az["z_bar"][keep], az["n"][keep], az["dX"][keep]
    sub_glm = poisson_glm_powerlaw(z_s, n_s, dX_s)
    sub_cmp = poisson_vs_logwls(z_s, n_s, dX_s)
    rate_s = np.asarray(n_s, float) / np.asarray(dX_s, float)
    err_s = np.sqrt(np.asarray(n_s, float)) / np.asarray(dX_s, float)
    sub = dict(sub_glm, stats=law_vs_points_stats(sub_glm["A"], sub_glm["gamma"], z_s,
                                                  rate_s, err_s),
               consistency_vs_refit=_cons("subDLA", sub_glm, z_s),
               glm_vs_wls=dict(delta_A_frac=sub_cmp["delta_A_frac"],
                               delta_gamma=sub_cmp["delta_gamma"]),
               n_points=len(np.asarray(z_s)))

    # (1b) DLA: Poisson GLM on PW09 counts + the shared-WLS regression anchor
    ad = ELL_X_DLA_GE20P3_PW09_T1
    dla_glm = poisson_glm_powerlaw(ad["z_bar"], ad["m"], ad["dX"])
    dla_wls = wls_powerlaw(ad["z_bar"], ad["lx"], ad["err_hi"])   # deployed convention
    dla_cmp = poisson_vs_logwls(ad["z_bar"], ad["m"], ad["dX"])
    dla = dict(dla_glm,
               stats=law_vs_points_stats(dla_glm["A"], dla_glm["gamma"], ad["z_bar"],
                                         ad["lx"], np.asarray(ad["err_hi"], float)),
               consistency_vs_refit=_cons("DLA", dla_glm, ad["z_bar"]),
               wls_anchor=dict(A=float(dla_wls["A"]), gamma=float(dla_wls["gamma"]),
                               chi2_red=float(dla_wls["chi2_red"])),
               glm_vs_wls=dict(delta_A_frac=dla_cmp["delta_A_frac"],
                               delta_gamma=dla_cmp["delta_gamma"]))

    # (2) UNCORRECTED cumulative compilation (diagonal GLS == the shared WLS)
    comp = lls_compilation(journal_fix=journal_fix, omeara_variant=omeara_variant)
    sig_cum = symmetrize_log_errors(comp["lx"], comp["err_lo"], comp["err_hi"], mode=err_mode)
    cum_fit = gls_powerlaw_log(comp["z_bar"], comp["lx"], sig_cum)
    cum = dict(cum_fit,
               stats=law_vs_points_stats(cum_fit["A"], cum_fit["gamma"], comp["z_bar"],
                                         comp["lx"], sig_cum * np.asarray(comp["lx"], float)),
               consistency_vs_refit=None)     # no deployed cumulative referent exists

    # (3) K3: fit-based cap (cannot go negative by gate) x PRIYA floor factor
    def r_k3(zz):
        cap = cap_fraction_from_laws(zz, sub_glm, dla_glm, cum_fit, slls_own_words=False)
        assert np.all(cap > 0), f"K3 cap non-positive at z={np.asarray(zz)[cap <= 0]}"
        return cap * np.asarray(floor_factor(zz), float)

    k3 = dict(law=_fit_corrected_lls(comp, r_k3, kernel_budget, err_mode),
              r_at_lit=r_k3(np.asarray(comp["z_bar"], float)),
              r3=float(r_k3(np.array([3.0]))[0]))
    k3["telescoping"] = _telescoping_check(k3["law"]["fit"], floor_factor, sub_glm,
                                           dla_glm, cum_fit)

    out = dict(subDLA=sub, DLA=dla, cum_uncorrected=cum, K3=k3,
               inputs=dict(journal_fix=journal_fix, omeara_variant=omeara_variant,
                           larger_side=larger_side, drop_z180=drop_z180,
                           kernel_budget=kernel_budget or {}),
               compilation=comp)

    # K3b diagnostic: per-point data-side subtraction (cross-check ONLY, never adopted)
    zb = np.asarray(comp["z_bar"], float)
    sub_at = sub_glm["A"] * (1.0 + zb) ** sub_glm["gamma"]
    dla_at = dla_glm["A"] * (1.0 + zb) ** dla_glm["gamma"]
    F_at = np.asarray(floor_factor(zb), float)
    pts_k3b = (np.asarray(comp["lx"], float) - sub_at - dla_at) * F_at
    out["K3b_diagnostic"] = dict(z_bar=zb, corrected_points=pts_k3b,
                                 n_nonpositive=int(np.sum(pts_k3b <= 0)),
                                 note="per-point subtraction diagnostic; can go negative; "
                                      "NEVER a deployment candidate")

    # K2: POW10-own-words cap (their l_SLLS law + the PW09 law) x PRIYA floor factor
    def r_k2(zz):
        cap = cap_fraction_from_laws(zz, None, dla_glm, cum_fit, slls_own_words=True)
        assert np.all(cap > 0), f"K2 cap non-positive at z={np.asarray(zz)[cap <= 0]}"
        return cap * np.asarray(floor_factor(zz), float)

    k2 = dict(law=_fit_corrected_lls(comp, r_k2, kernel_budget, err_mode),
              r_at_lit=r_k2(zb), r3=float(r_k2(np.array([3.0]))[0]),
              note="not literature-pure: only the cap is POW10-own-words; the floor "
                   "factor borrows the PRIYA CDDF shape. Law-based at all z (the "
                   "below-z=3.4 extrapolation policy is PI sub-decision).")
    k2["telescoping"] = _telescoping_check(k2["law"]["fit"], floor_factor, sub_glm,
                                           dla_glm, cum_fit)
    out["K2"] = k2

    # K1 arms (cache kernel), when supplied
    if r_k1 is not None:
        k1a = dict(law=_fit_corrected_lls(comp, r_k1, kernel_budget, err_mode),
                   r_at_lit=np.asarray(r_k1(zb), float),
                   r3=float(np.asarray(r_k1(np.array([3.0])))[0]))
        k1a["telescoping"] = _telescoping_check(k1a["law"]["fit"], floor_factor, sub_glm,
                                                dla_glm, cum_fit)
        out["K1a"] = k1a
    if r_k1_smooth is not None:
        k1b = dict(law=_fit_corrected_lls(comp, r_k1_smooth, kernel_budget, err_mode),
                   r_at_lit=np.asarray(r_k1_smooth(zb), float),
                   r3=float(np.asarray(r_k1_smooth(np.array([3.0])))[0]))
        k1b["telescoping"] = _telescoping_check(k1b["law"]["fit"], floor_factor, sub_glm,
                                                dla_glm, cum_fit)
        out["K1b"] = k1b

    # sensitivity arms (spec step 7): z=1.80 drop, larger-side errors, journal-fix toggle
    if include_sensitivity:
        out["sensitivity"] = dict(
            drop_z180=run_fit_ordering(floor_factor, kernel_budget=kernel_budget,
                                       journal_fix=journal_fix,
                                       omeara_variant=omeara_variant,
                                       larger_side=larger_side, drop_z180=True,
                                       include_sensitivity=False),
            larger_side=run_fit_ordering(floor_factor, kernel_budget=kernel_budget,
                                         journal_fix=journal_fix,
                                         omeara_variant=omeara_variant, larger_side=True,
                                         include_sensitivity=False),
            journal_fix=run_fit_ordering(floor_factor, kernel_budget=kernel_budget,
                                         journal_fix=not journal_fix,
                                         omeara_variant=omeara_variant,
                                         larger_side=larger_side,
                                         include_sensitivity=False),
        )
    return out
