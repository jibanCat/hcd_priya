"""
NHI recovery injection test — Phase A1 of the HCD-identification fix plan.

Purpose
-------
Confirm whether the log-N pile-up at ≈19.5 seen in the production catalogs
(see figures/intermediate/nhi_distributions.png) is caused by the Voigt
fitter itself, or by something upstream.

Method
------
1. Synthesise tau(v) for a grid of (NHI, b) using the SAME voigt_utils.tau_voigt
   that the pipeline uses to generate its model during fitting.
2. Run the pipeline's fit via the full catalog path — find_systems_in_skewer,
   then measure_nhi_for_system — so we exercise exactly the code that runs
   on real sightlines (window expansion, tau_cap, log-space residual, etc).
3. Do this both for pure Voigt injections and with a forest background τ.
4. Plot recovered vs true log NHI.

If the pile-up reproduces on clean synthetic input, the fitter is the cause
and we have a reproducible benchmark to iterate on.

Usage
-----
    python3 tests/test_nhi_recovery.py            # plots + prints summary
    python3 tests/test_nhi_recovery.py --quick    # small grid, fast

Produces
--------
    figures/diagnostics/nhi_recovery.png
    figures/diagnostics/nhi_recovery_forest.png
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hcd_analysis.catalog import (
    find_systems_in_skewer,
    measure_nhi_for_system,
)
from hcd_analysis.voigt_utils import tau_voigt, fit_nhi_from_tau, _SIGMA_PREFACTOR


# ---------------------------------------------------------------------------
# Independent physical tau(v) — does NOT share the codebase's _SIGMA_PREFACTOR.
# Used to verify the fitter recovers true NHI from *externally* generated tau,
# which is the situation with real fake_spectra output.
# ---------------------------------------------------------------------------

def tau_voigt_physical(v_kms, NHI, b_kms, v0_kms=0.0):
    """
    Independent-of-voigt_utils Voigt optical-depth profile.

    tau(v) = sqrt(pi) * NHI * e^2 f lambda / (m_e c b_cm_per_s) * H(a, u)
    where H is the Voigt-Hjerting function and (a, u) are the standard Voigt
    coordinates (a = Gamma*lambda/(4*pi*b_cm_per_s), u = (v-v0)/b_kms).
    This matches the normalisation used by fake_spectra.
    """
    from scipy.special import wofz
    e = 4.80320427e-10          # esu
    m_e = 9.10938e-28           # g
    c = 2.99792458e10           # cm/s
    f_lu = 0.4164
    lam = 1.21567e-5            # cm
    Gamma = 6.265e8             # s^-1

    b_cms = b_kms * 1e5
    u = (np.asarray(v_kms) - v0_kms) / b_kms   # dimensionless
    a = Gamma * lam / (4.0 * np.pi * b_cms)
    H = wofz(u + 1j * a).real
    sigma0 = np.sqrt(np.pi) * e * e * f_lu * lam / (m_e * c * b_cms)
    return NHI * sigma0 * H

# Production values from config/default.yaml
DV_KMS = 10.0
NBINS = 1556
TAU_THRESHOLD = 100.0
MERGE_DV_KMS = 100.0
MIN_PIXELS = 2
B_INIT = 30.0
B_BOUNDS = (1.0, 300.0)
TAU_FIT_CAP = 1.0e6
VOIGT_MAX_ITER = 200


# ---------------------------------------------------------------------------
# Test 1 — direct fitter, clean Voigt input (no pipeline wrapping)
# ---------------------------------------------------------------------------

def run_fitter_direct(
    log_nhi_true: np.ndarray,
    b_true: float = 30.0,
    wing_kms: float = 2000.0,
    dv_kms: float = DV_KMS,
    use_physical_tau: bool = False,
) -> np.ndarray:
    """
    Fit each clean Voigt profile with the pipeline's fit_nhi_from_tau.

    If use_physical_tau=True, the input tau is built with the standalone
    tau_voigt_physical() (matching fake_spectra's normalisation) — this is
    the critical cross-normalisation test.  Under the pre-fix bug, the fit
    recovered log N ~5 dex too low.  Under the fix, both modes agree.
    """
    n_pix = int(2 * wing_kms / dv_kms) + 1
    v = (np.arange(n_pix) - n_pix // 2) * dv_kms
    synth = tau_voigt_physical if use_physical_tau else tau_voigt

    recovered = np.empty_like(log_nhi_true)
    for i, logN in enumerate(log_nhi_true):
        tau = synth(v, 10.0**logN, b_true, v0_kms=0.0)
        NHI_fit, _, _ = fit_nhi_from_tau(
            tau, v, b_init=B_INIT, b_bounds=B_BOUNDS,
            tau_cap=TAU_FIT_CAP, max_iter=VOIGT_MAX_ITER,
        )
        recovered[i] = np.log10(max(NHI_fit, 1.0))
    return recovered


# ---------------------------------------------------------------------------
# Test 2 — full pipeline path, synthetic sightline with/without forest
# ---------------------------------------------------------------------------

def synthesise_sightline(
    log_nhi_true: float,
    b_true: float,
    v_center_kms: float,
    nbins: int = NBINS,
    dv_kms: float = DV_KMS,
    forest_tau: float = 0.0,
    seed: int = 0,
    kinematic_sigma_kms: float = 0.0,
    n_components: int = 1,
) -> np.ndarray:
    """
    Build a synthetic tau skewer.

    Base: single Voigt at (NHI_true, b_true, v_center).

    If kinematic_sigma_kms > 0 and n_components > 1: simulate halo gas kinematics
    by splitting NHI_true into `n_components` Voigt sub-components whose
    v-offsets are drawn from N(0, kinematic_sigma_kms).  Each sub-component has
    NHI_true/n_components and the same b_true.  This mimics fake_spectra's
    velocity-broadened DLA cores.

    Forest background: optional lognormal floor at mean forest_tau.
    """
    v = np.arange(nbins) * dv_kms
    rng = np.random.default_rng(seed)

    if n_components > 1 and kinematic_sigma_kms > 0.0:
        nhi_each = 10.0**log_nhi_true / n_components
        offsets = rng.normal(0.0, kinematic_sigma_kms, size=n_components)
        tau = np.zeros_like(v)
        for dv in offsets:
            tau = tau + tau_voigt(v, nhi_each, b_true, v0_kms=v_center_kms + dv)
    else:
        tau = tau_voigt(v, 10.0**log_nhi_true, b_true, v0_kms=v_center_kms)

    if forest_tau > 0.0:
        tau_bg = rng.lognormal(mean=np.log(forest_tau), sigma=1.4, size=nbins)
        tau = tau + tau_bg.astype(tau.dtype)
    return tau


def run_pipeline_path(
    log_nhi_true: np.ndarray,
    b_true: float = 30.0,
    forest_tau: float = 0.0,
    seed: int = 0,
    kinematic_sigma_kms: float = 0.0,
    n_components: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each injected (NHI, b_true), run the full catalog measurement path:
    find_systems_in_skewer → measure_nhi_for_system.

    Returns (recovered_log_nhi, n_systems_found).
    """
    recovered = np.full_like(log_nhi_true, np.nan)
    n_found = np.zeros_like(log_nhi_true, dtype=int)
    merge_gap_pixels = max(1, int(MERGE_DV_KMS / DV_KMS))

    v_center = NBINS // 2 * DV_KMS

    for i, logN in enumerate(log_nhi_true):
        tau = synthesise_sightline(
            logN, b_true, v_center,
            nbins=NBINS, dv_kms=DV_KMS,
            forest_tau=forest_tau, seed=seed + i,
            kinematic_sigma_kms=kinematic_sigma_kms,
            n_components=n_components,
        ).astype(np.float64)

        systems = find_systems_in_skewer(
            tau, TAU_THRESHOLD, merge_gap_pixels, MIN_PIXELS,
        )
        n_found[i] = len(systems)
        if not systems:
            continue

        # Pick the system that covers the injected v_center (the "true" one).
        # If none does, take the first.
        best = systems[0]
        pix_center = int(v_center / DV_KMS)
        for s, e in systems:
            if s <= pix_center <= e:
                best = (s, e)
                break

        pix_start, pix_end = best
        NHI, _, _ = measure_nhi_for_system(
            tau, pix_start, pix_end, DV_KMS,
            b_init=B_INIT, b_bounds=B_BOUNDS,
            tau_fit_cap=TAU_FIT_CAP, max_iter=VOIGT_MAX_ITER,
            fast_mode=False,
        )
        recovered[i] = np.log10(max(NHI, 1.0))

    return recovered, n_found


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_recovery(
    log_nhi_true: np.ndarray,
    results: dict[str, np.ndarray],
    title: str,
    outpath: Path,
) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: recovered vs true
    for label, rec in results.items():
        ok = np.isfinite(rec)
        ax[0].plot(log_nhi_true[ok], rec[ok], "o-", label=label, alpha=0.8)
    lo, hi = 17.0, 22.5
    ax[0].plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="perfect")
    for thr, name in [(17.2, "LLS"), (19.0, "subDLA"), (20.3, "DLA")]:
        ax[0].axvline(thr, color="gray", lw=0.5, ls=":")
        ax[0].axhline(thr, color="gray", lw=0.5, ls=":")
    ax[0].set_xlabel("true log10(N_HI)")
    ax[0].set_ylabel("recovered log10(N_HI)")
    ax[0].set_xlim(lo, hi)
    ax[0].set_ylim(lo, hi)
    ax[0].legend(loc="upper left")
    ax[0].grid(alpha=0.3)
    ax[0].set_title("recovered vs true")

    # Panel 2: residual log N_rec - log N_true
    for label, rec in results.items():
        ok = np.isfinite(rec)
        ax[1].plot(log_nhi_true[ok], rec[ok] - log_nhi_true[ok], "o-", label=label, alpha=0.8)
    ax[1].axhline(0, color="k", ls="--", alpha=0.5)
    for thr, name in [(17.2, "LLS"), (19.0, "subDLA"), (20.3, "DLA")]:
        ax[1].axvline(thr, color="gray", lw=0.5, ls=":")
    ax[1].set_xlabel("true log10(N_HI)")
    ax[1].set_ylabel("log10(N_HI_rec) - log10(N_HI_true)")
    ax[1].set_xlim(lo, hi)
    ax[1].legend(loc="lower left")
    ax[1].grid(alpha=0.3)
    ax[1].set_title("residual")

    fig.suptitle(title)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    print(f"  saved {outpath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="small grid, fast")
    parser.add_argument("--b", type=float, default=30.0, help="Doppler b (km/s)")
    args = parser.parse_args()

    n = 15 if args.quick else 40
    log_nhi_grid = np.linspace(17.2, 21.8, n)

    out_dir = Path(__file__).resolve().parent.parent / "figures" / "diagnostics"

    # -- Test 1: direct fitter on clean Voigt -------------------------------
    print("Test 1: direct fit_nhi_from_tau on clean Voigt...")
    t0 = time.time()
    rec_direct_2k = run_fitter_direct(log_nhi_grid, b_true=args.b, wing_kms=2000.0)
    rec_direct_8k = run_fitter_direct(log_nhi_grid, b_true=args.b, wing_kms=8000.0)
    # CROSS-NORMALISATION: synthesise tau with the independent physical
    # formula (= fake_spectra convention), then fit with the codebase.
    # Must round-trip under the fix.
    rec_direct_phys = run_fitter_direct(
        log_nhi_grid, b_true=args.b, wing_kms=2000.0, use_physical_tau=True,
    )
    print(f"  done in {time.time()-t0:.1f}s")

    # Report agreement
    disagreement = np.abs(rec_direct_phys - log_nhi_grid)
    print(f"  cross-norm max |log N_rec - log N_true| over grid = {np.nanmax(disagreement):.2f} dex")
    print(f"  (pre-fix: ~5 dex; post-fix: < 0.3 dex expected)")

    plot_recovery(
        log_nhi_grid,
        {"codebase synth (self-consistent)": rec_direct_2k,
         "wing ±8000 km/s": rec_direct_8k,
         "physical synth → codebase fit (cross-norm)": rec_direct_phys},
        title=f"Direct Voigt fit on synthetic tau (b={args.b} km/s)",
        outpath=out_dir / "nhi_recovery_clean.png",
    )

    # -- Test 2: pipeline path, no forest -----------------------------------
    print("Test 2: pipeline path (find + measure), no forest...")
    t0 = time.time()
    rec_pipe_clean, nfound_clean = run_pipeline_path(
        log_nhi_grid, b_true=args.b, forest_tau=0.0,
    )
    print(f"  done in {time.time()-t0:.1f}s, systems found: {nfound_clean}")

    # -- Test 3: pipeline path, realistic forest background -----------------
    # At z=3, mean forest tau ≈ -ln(0.67) ≈ 0.4
    print("Test 3: pipeline path, forest_tau=0.4 (z≈3)...")
    t0 = time.time()
    rec_pipe_forest, nfound_forest = run_pipeline_path(
        log_nhi_grid, b_true=args.b, forest_tau=0.4, seed=42,
    )
    print(f"  done in {time.time()-t0:.1f}s")

    plot_recovery(
        log_nhi_grid,
        {"pipeline, clean": rec_pipe_clean,
         "pipeline, forest τ=0.4 (z≈3)": rec_pipe_forest},
        title=f"Full pipeline NHI recovery (b_true={args.b} km/s, wing=±2000 km/s)",
        outpath=out_dir / "nhi_recovery_pipeline.png",
    )

    # -- Test 4: pipeline path with velocity-broadened DLA cores ------------
    # Real fake_spectra DLAs have cores broadened by halo gas kinematics:
    # the column-density budget is spread over ~50-200 km/s at different
    # velocities.  We simulate this with n_components Voigts whose offsets
    # are drawn from N(0, sigma_kms).
    print("Test 4: kinematic-broadened cores (n=8 components)...")
    for sigma_kms in (50.0, 100.0, 200.0):
        t0 = time.time()
        rec_kin, _ = run_pipeline_path(
            log_nhi_grid, b_true=args.b, forest_tau=0.4, seed=7,
            kinematic_sigma_kms=sigma_kms, n_components=8,
        )
        print(f"  sigma={sigma_kms:3.0f} km/s: {time.time()-t0:.1f}s")
        # collect for combined plot
        if sigma_kms == 50.0:
            rec_kin_50 = rec_kin
        elif sigma_kms == 100.0:
            rec_kin_100 = rec_kin
        else:
            rec_kin_200 = rec_kin

    plot_recovery(
        log_nhi_grid,
        {"σ_v=50 km/s, 8 components": rec_kin_50,
         "σ_v=100 km/s, 8 components": rec_kin_100,
         "σ_v=200 km/s, 8 components": rec_kin_200,
         "clean single Voigt (ref)": rec_pipe_clean},
        title=f"NHI recovery with velocity-broadened cores (b_true={args.b}, forest τ=0.4)",
        outpath=out_dir / "nhi_recovery_kinematic.png",
    )

    # -- Summary ------------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"{'true log N':>12}  {'direct 2k':>10}  {'direct 8k':>10}  "
          f"{'pipe clean':>11}  {'pipe forest':>12}")
    print("-" * 72)
    for i, ln in enumerate(log_nhi_grid):
        print(f"{ln:>12.2f}  {rec_direct_2k[i]:>10.2f}  {rec_direct_8k[i]:>10.2f}  "
              f"{rec_pipe_clean[i]:>11.2f}  {rec_pipe_forest[i]:>12.2f}")
    print("=" * 72)

    # Quick verdicts
    def pile_up_bias(true_lN, rec_lN, dla_thresh=20.3):
        is_dla = true_lN >= dla_thresh
        if not is_dla.any():
            return None
        return float(np.nanmean(rec_lN[is_dla] - true_lN[is_dla]))

    print("\nMean bias on injected DLAs (log N_rec - log N_true):")
    for label, rec in [("direct 2k-wing", rec_direct_2k),
                        ("direct 8k-wing", rec_direct_8k),
                        ("pipeline clean", rec_pipe_clean),
                        ("pipeline forest", rec_pipe_forest),
                        ("pipeline kinematic σ=50", rec_kin_50),
                        ("pipeline kinematic σ=100", rec_kin_100),
                        ("pipeline kinematic σ=200", rec_kin_200)]:
        b = pile_up_bias(log_nhi_grid, rec)
        if b is not None:
            print(f"  {label:>24s}: {b:+.2f} dex")

    # Specifically highlight the production pile-up signature: fraction of
    # true DLAs (log N >= 20.3) whose recovered value falls in [19, 20).
    print("\nProduction pile-up signature: frac(true DLA) that recovers as subDLA (19 ≤ rec < 20):")
    is_true_dla = log_nhi_grid >= 20.3
    if is_true_dla.any():
        for label, rec in [("pipeline clean", rec_pipe_clean),
                            ("pipeline forest", rec_pipe_forest),
                            ("pipeline kinematic σ=50", rec_kin_50),
                            ("pipeline kinematic σ=100", rec_kin_100),
                            ("pipeline kinematic σ=200", rec_kin_200)]:
            r = rec[is_true_dla]
            mis = np.isfinite(r) & (r >= 19.0) & (r < 20.0)
            print(f"  {label:>24s}: {int(mis.sum())}/{int(np.isfinite(r).sum())}")


if __name__ == "__main__":
    main()
