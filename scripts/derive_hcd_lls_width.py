"""[SUPERSEDED 2026-07-18 — kept as the historical WIDTH-RECORD script] Derive the 1x/2x
HCD fractional dN/dX measurement widths ON THE OLD (WRONG-OBJECT) literature arrays.

SUPERSESSION: the corrected-law re-derivation (PI adoption 2026-07-18, spec
hcd_priya_notes/docs/superpowers/2026-07-18-corrected-law-spec.md) retired the arrays this
script fits: its 'LLS' points are the CUMULATIVE tau>=2 compilation (not the binned class)
and its 'subDLA' points are Zafar Table 3's Peroux DLA column (wrong object). They are now
served from the TOMBSTONE constants in hcd_analysis/emulator/lit_dndx.py (greppable, never
fit for deployment), so this script's printed numbers stay byte-identical as the
historical record of where the old 0.15/0.30 knobs came from. THE DEPLOYED WIDTH OF RECORD
is now derived by scripts/derive_hcd_dndx_corrected.py on the corrected K1a points:
sigma_LLS 1x = 0.287 (measurement 0.174 + kernel common-mode s_r3=0.227 in quadrature,
PI decision 4), 2x hedge = 0.574; see inference.HCD_LLS_SURVEY_FRAC_SIGMA(+HEDGE2X).

The PI WIDTH RULE: set sigma_LLS to 1-2x the LITERATURE dN/dX MEASUREMENT error.
1x = the fractional lit dN/dX measurement uncertainty for LLS (derived here), 2x = double.

We combine THREE contributions of the lit measurement error into ONE fractional sigma/mu:
  (a) per-point measurement error bars (the lit error bars themselves), as a fractional
      weighted-rms over the points;
  (b) the WLS-fit NORMALIZATION uncertainty at the pivot z=3 (the A error from the power-law
      WLS), i.e. how well the lit fixes dN/dX at the prior pivot;
  (c) the per-point SCATTER about the WLS fit (excess scatter beyond the error bars, the
      heterogeneous-compilation hedge).
The 1x sigma/mu = max(b_chi2-inflated, c) reported alongside the raw per-point (a).

RESULT (HISTORICAL record, superseded 2026-07-18; re-run verified identical): LLS 1x
sigma/mu = 0.160 (WLS norm err 0.086 chi2-inflated vs per-point scatter 0.160 -> max
0.160) -> the old clean 1x knob 0.15; 2x = 0.30. WLS gamma_LLS = +2.127, A = 0.0201 (the
cumulative-estimand fit; the DEPLOYED corrected law is now (0.0184, 2.127) constrained).
subDLA 1x = 0.102, DLA 1x = 0.094 (wrong-object subDLA; the repo keeps subDLA
sigma/mu=0.40, DLA=0.50 broad). Deployed knobs of record: 0.287 / 0.574 (see above).

Env: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/derive_hcd_lls_width.py
"""
import numpy as np

# The WLS implementation was refactored VERBATIM into the shared module (corrected-law
# re-derivation, spec 2026-07-18 step 3) so old and new scripts share ONE implementation;
# behavior here is byte-identical (regression-tested in tests/test_lit_dndx_fits.py).
from hcd_analysis.emulator.lit_dndx import (  # noqa: E402
    wls_powerlaw, LLS_TAU2_OLD_DEFECTS, ZAFAR13_T3_DLA_BLOCK_WRONG_SUBDLA_LABEL,
    ELL_X_DLA_GE20P3_PW09_T1)

# The OLD literature arrays, served from the estimand-named TOMBSTONES (LIT dict deleted;
# the DLA points are the verified PW09 Table 1 values with upper errors, the deployed
# convention — numerically identical to the old hard-code).
_T_LLS = LLS_TAU2_OLD_DEFECTS
_T_SUB = ZAFAR13_T3_DLA_BLOCK_WRONG_SUBDLA_LABEL
_T_DLA = ELL_X_DLA_GE20P3_PW09_T1
LIT = {
    "LLS":    (list(_T_LLS["z"]), list(_T_LLS["lx"]), list(_T_LLS["err"]),
               "O'Meara13 / Fumagalli13 / Prochaska10 (tau>=2) [TOMBSTONE: cumulative, "
               "wrong estimand]"),
    "subDLA": (list(_T_SUB["z"]), list(_T_SUB["lx"]), list(_T_SUB["err"]),
               "Zafar+2013 (Table 3) [TOMBSTONE: Peroux DLA column, wrong object]"),
    "DLA":    (list(_T_DLA["z_bar"]), list(_T_DLA["lx"]), list(_T_DLA["err_hi"]),
               "Prochaska & Wolfe 2009 (Table 1)"),
}
Z_PIVOT = 3.0


def per_point_frac(z, v, e):
    """Weighted-rms fractional per-point measurement error (contribution a)."""
    v = np.asarray(v, float); e = np.asarray(e, float)
    frac = e / v
    # weighted (by 1/frac^2) and plain mean both, plus the simple median
    return dict(mean=float(np.mean(frac)), median=float(np.median(frac)),
                wmean=float(np.sqrt(np.average(frac**2))))   # rms fractional


print("=" * 92)
print("HCD dN/dX literature MEASUREMENT-error -> prior sigma/mu  (PI WIDTH RULE: 1x lit error)")
print("=" * 92)
out = {}
for cls in ("LLS", "subDLA", "DLA"):
    z, v, e, src = LIT[cls]
    pp = per_point_frac(z, v, e)
    fit = wls_powerlaw(z, v, e)
    print(f"\n[{cls}]  ({src})")
    print(f"  N points = {len(z)},  z range {min(z):.2f}-{max(z):.2f}")
    print(f"  WLS power-law: A={fit['A']:.4f}  gamma={fit['gamma']:+.3f} +- {fit['sigma_gamma']:.3f}"
          f"   chi2/dof={fit['chi2_red']:.2f}")
    print(f"  (a) per-point frac err   : mean={pp['mean']:.3f}  median={pp['median']:.3f}  rms={pp['wmean']:.3f}")
    print(f"  (b) WLS norm err @z=3    : {fit['frac_norm_err']:.3f}  (chi2-inflated {fit['frac_norm_err_infl']:.3f})")
    print(f"  (c) per-point scatter rms: {fit['frac_scatter']:.3f}")
    # The 1x = the lit fractional MEASUREMENT uncertainty: combine norm-error (chi2-inflated, so it
    # captures excess scatter) AND the per-point scatter in quadrature, then take the dominant.
    # The norm-err-infl ALREADY includes scatter (via chi2), so use max(norm_infl, scatter) as the 1x,
    # not a double-count. Round to 2 decimals for a clean prior knob.
    one_x = max(fit['frac_norm_err_infl'], fit['frac_scatter'])
    out[cls] = dict(one_x=one_x, two_x=2.0*one_x, gamma=fit['gamma'], A=fit['A'])
    print(f"  => 1x sigma/mu = max(norm_infl, scatter) = {one_x:.3f}   2x = {2.0*one_x:.3f}")

print("\n" + "=" * 92)
print("SUMMARY (the prior knobs):")
print("=" * 92)
for cls in ("LLS", "subDLA", "DLA"):
    o = out[cls]
    print(f"  {cls:7s}  1x sigma/mu = {o['one_x']:.3f}   2x sigma/mu = {o['two_x']:.3f}"
          f"   (WLS gamma={o['gamma']:+.3f}, A={o['A']:.4f})")
print()
print(f"  LLS gamma (the real-fit forward z-slope) = {out['LLS']['gamma']:+.4f}  "
      f"(spec target 2.127; guard floor 1.5)")
print(f"  LLS 1x rounded to a clean prior knob: {round(out['LLS']['one_x'],2)} ; 2x: {round(out['LLS']['two_x'],2)}")
