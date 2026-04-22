"""
Why fast mode works on DLAs when observational EW methods don't.

Demonstrates three things:
  1. Sum rule:  ∫ τ(v) dv = N_HI · _SIGMA_PREFACTOR  (exact for any Voigt).
  2. Truncation bias: what fraction of ∫ τ dv lies in pixels with τ > 100,
     as a function of (N_HI, b)?  This is the ONLY approximation in fast mode.
  3. Cross-check against a fitter that sees the full profile (no forest bias):
     fast mode vs a Voigt fit on clean synthetic input — do they agree?

Also makes one plot summarising all of this for the audit record.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hcd_analysis.voigt_utils import _SIGMA_PREFACTOR, fit_nhi_from_tau
from tests.test_nhi_recovery import tau_voigt_physical  # physical normalisation

DV = 1.0       # km/s (fine pixel grid for sum-rule test)
V_HALF = 10000 # km/s half-box (wide enough to contain the full profile)
TAU_THR = 100.0
OUT = ROOT / "figures" / "diagnostics"
OUT.mkdir(parents=True, exist_ok=True)


def profile(logN, b_kms, dv=DV, v_half=V_HALF):
    v = np.arange(-v_half, v_half + dv, dv)
    tau = tau_voigt_physical(v, 10.0**logN, b_kms, v0_kms=0.0)
    return v, tau


def fast_mode_nhi(tau, dv):
    """Replicate fast_mode: tau_int over τ>100 / _SIGMA_PREFACTOR."""
    mask = tau > TAU_THR
    tau_int = float((tau[mask] * dv).sum())
    return tau_int / _SIGMA_PREFACTOR


def voigt_fit_nhi(v, tau, b_init=30.0):
    """Fit N_HI using the codebase's fit_nhi_from_tau on the FULL v-grid.
    (Bypasses catalog.py's window expansion, so this is wing-contamination-free
    when tau has no forest added.)"""
    NHI, _, _ = fit_nhi_from_tau(
        tau, v, b_init=b_init, b_bounds=(1.0, 300.0),
        tau_cap=1.0e12, max_iter=500,
    )
    return NHI


def main():
    print("=" * 66)
    print("1.  Sum-rule check (should be exact to machine precision)")
    print("=" * 66)
    print(f"{'logN':>5}  {'b':>5}  {'∫τdv':>12}  {'NHI·SIG_PREF':>14}  ratio")
    for logN in (17.5, 19.0, 20.3, 21.0, 22.0):
        for b in (15.0, 30.0, 60.0):
            _, tau = profile(logN, b)
            tau_int_full = float((tau * DV).sum())
            expected = 10.0**logN * _SIGMA_PREFACTOR
            print(f"  {logN:>4.2f}  {b:>4.0f}  {tau_int_full:>12.4e}  "
                  f"{expected:>14.4e}  {tau_int_full/expected:6.4f}")

    print()
    print("=" * 66)
    print("2.  Truncation: what fraction of ∫τdv is above τ=100?")
    print("    (This IS the fast-mode error budget.)")
    print("=" * 66)
    print(f"{'logN':>5}  {'b=15':>8}  {'b=30':>8}  {'b=60':>8}  {'b=100':>8}")
    logN_grid = np.arange(17.5, 22.5, 0.25)
    b_grid = (15.0, 30.0, 60.0, 100.0)
    frac_table = np.zeros((len(logN_grid), len(b_grid)))
    for i, logN in enumerate(logN_grid):
        row = f"  {logN:>4.2f}"
        for j, b in enumerate(b_grid):
            _, tau = profile(logN, b)
            full = (tau * DV).sum()
            above = (tau[tau > TAU_THR] * DV).sum()
            frac_table[i, j] = above / full if full > 0 else 0.0
            row += f"  {frac_table[i, j]:>8.4f}"
        print(row)

    print()
    print("=" * 66)
    print("3.  Fast mode vs Voigt-fit on clean synthetic input")
    print("=" * 66)
    print(f"{'logN_true':>10}  {'b':>4}  {'fast':>8}  {'voigt':>8}  "
          f"{'fast-truth':>11}  {'voigt-truth':>12}")
    fast_results, voigt_results = {}, {}
    for b in b_grid:
        fr, vr = [], []
        for logN in logN_grid:
            v, tau = profile(logN, b)
            NHI_fast = fast_mode_nhi(tau, DV)
            lf = np.log10(max(NHI_fast, 1.0))
            try:
                NHI_fit = voigt_fit_nhi(v, tau, b_init=30.0)
                lv = np.log10(max(NHI_fit, 1.0))
            except Exception:
                lv = np.nan
            fr.append(lf); vr.append(lv)
            print(f"  {logN:>9.2f}  {b:>4.0f}  {lf:>8.3f}  {lv:>8.3f}  "
                  f"{lf-logN:>+11.3f}  {lv-logN:>+12.3f}")
        fast_results[b] = np.array(fr)
        voigt_results[b] = np.array(vr)

    # -- Figure -----------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: captured fraction vs logN for each b
    for j, b in enumerate(b_grid):
        ax[0].plot(logN_grid, frac_table[:, j], "o-", label=f"b={b:.0f} km/s")
    ax[0].axhline(1.0, color="k", ls="--", alpha=0.4)
    ax[0].axhline(0.90, color="gray", ls=":", alpha=0.4)
    for thr, name in [(17.2, "LLS"), (19.0, "subDLA"), (20.3, "DLA")]:
        ax[0].axvline(thr, color="gray", lw=0.5, ls=":")
    ax[0].set_xlabel("log10(N_HI)")
    ax[0].set_ylabel("fraction of ∫τdv captured by τ>100 cut")
    ax[0].set_title("Truncation: the only error in fast mode")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[0].set_ylim(0, 1.05)

    # Panel 2: implied log-NHI bias vs logN at each b
    for j, b in enumerate(b_grid):
        bias = np.log10(np.clip(frac_table[:, j], 1e-30, 1.0))
        ax[1].plot(logN_grid, bias, "o-", label=f"b={b:.0f} km/s")
    for thr, name in [(17.2, "LLS"), (19.0, "subDLA"), (20.3, "DLA")]:
        ax[1].axvline(thr, color="gray", lw=0.5, ls=":")
    ax[1].axhline(0, color="k", ls="--", alpha=0.4)
    ax[1].axhline(-0.1, color="gray", ls=":", alpha=0.4)
    ax[1].set_xlabel("log10(N_HI)")
    ax[1].set_ylabel("log10(fast N_HI / true N_HI) = log(fraction captured)")
    ax[1].set_title("Truncation bias (log-N)")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    ax[1].set_ylim(-3.0, 0.3)

    # Panel 3: fast vs voigt-fit
    for j, b in enumerate(b_grid):
        ax[2].plot(logN_grid, fast_results[b] - logN_grid, "o-",
                   label=f"fast  b={b:.0f}", alpha=0.9)
    for j, b in enumerate(b_grid):
        ax[2].plot(logN_grid, voigt_results[b] - logN_grid, "x--",
                   label=f"voigt b={b:.0f}", alpha=0.7)
    for thr in (17.2, 19.0, 20.3):
        ax[2].axvline(thr, color="gray", lw=0.5, ls=":")
    ax[2].axhline(0, color="k", ls="--", alpha=0.4)
    ax[2].set_xlabel("log10(N_HI)")
    ax[2].set_ylabel("recovered − true (dex)")
    ax[2].set_title("fast vs Voigt-fit on clean input")
    ax[2].legend(ncol=2, fontsize=8)
    ax[2].grid(alpha=0.3)
    ax[2].set_ylim(-2.0, 0.5)

    fig.suptitle("Fast-mode NHI estimator: theory, truncation error, and cross-check")
    fig.tight_layout()
    outpath = OUT / "fast_mode_theory.png"
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    print(f"\nFigure: {outpath}")


if __name__ == "__main__":
    main()
