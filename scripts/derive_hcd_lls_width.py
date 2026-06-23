"""Derive the 1x and 2x HCD LLS (and subDLA/DLA) fractional dN/dX measurement uncertainty.

The PI WIDTH RULE: set sigma_LLS to 1-2x the LITERATURE dN/dX MEASUREMENT error.
1x = the fractional lit dN/dX measurement uncertainty for LLS (derived here), 2x = double.

We combine THREE contributions of the lit measurement error into ONE fractional sigma/mu:
  (a) per-point measurement error bars (the lit error bars themselves), as a fractional
      weighted-rms over the points;
  (b) the WLS-fit NORMALIZATION uncertainty at the pivot z=3 (the A error from the power-law
      WLS), i.e. how well the lit fixes dN/dX at the prior pivot;
  (c) the per-point SCATTER about the WLS fit (excess scatter beyond the error bars, the
      heterogeneous-compilation hedge).
The 1x sigma/mu = max(b_chi2-inflated, c) reported alongside the raw per-point (a); we report all
three so the PI can see the breakdown, and take the LLS 1x as the load-bearing number.

RESULT (the derived prior knobs, PI re-determination 2026-06-17): LLS 1x sigma/mu = 0.16 (the WLS
norm err 0.086 chi2-inflated, vs the per-point scatter 0.160 -> max 0.160) -> the clean 1x knob 0.15;
2x = 0.30 (cosmic-variance hedge). WLS gamma_LLS = +2.127 (the real-fit forward z-slope), A = 0.0201.
subDLA 1x = 0.10, DLA 1x = 0.09 (sub-dominant; the repo keeps subDLA sigma/mu=0.40, DLA=0.50 broad).

Env: PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
     /home/mfho/.conda/envs/emu-jax/bin/python3 scripts/derive_hcd_lls_width.py
"""
import numpy as np

# Literature dN/dX (z, value, +-err) per class - verbatim from scripts/plot_dndx_vs_literature.py
LIT = {
    "LLS":    ([2.4, 2.8, 3.35, 3.47, 3.58, 3.74, 3.97, 4.23],
               [0.29, 0.33, 0.35, 0.57, 0.41, 0.52, 0.72, 0.78],
               [0.05, 0.08, 0.14, 0.12, 0.07, 0.08, 0.15, 0.19],
               "O'Meara13 / Fumagalli13 / Prochaska10 (tau>=2)"),
    "subDLA": ([2.27, 2.73, 3.25, 3.77, 4.20],
               [0.07, 0.06, 0.08, 0.10, 0.10],
               [0.01, 0.01, 0.02, 0.02, 0.03],
               "Zafar+2013 (Table 3)"),
    "DLA":    ([2.31, 2.57, 2.86, 3.22, 3.70, 4.39],
               [0.048, 0.055, 0.067, 0.084, 0.075, 0.106],
               [0.006, 0.005, 0.006, 0.006, 0.009, 0.018],
               "Prochaska & Wolfe 2009 (Table 1)"),
}
Z_PIVOT = 3.0


def wls_powerlaw(z, v, e):
    """WLS fit log(dN/dX) = log A + gamma*log(1+z), weighting by the lit errors in log space.
    Returns (A, gamma, cov[2x2], sigma_logA, sigma_gamma, frac_norm_err_at_pivot, frac_scatter)."""
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
    xp = np.array([1.0, np.log(1.0 + Z_PIVOT)])
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
