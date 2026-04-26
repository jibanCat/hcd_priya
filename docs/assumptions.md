# Explicit Assumptions

## Data

1. **tau is raw simulation optical depth** — not rescaled to any mean flux.
   The UV background is the simulation's native UVB; no post-processing
   mean-flux rescaling is applied before P1D or CDDF computation.

2. **colden is empty** — the `colden` HDF5 group key exists but contains no
   datasets. All column density estimates are derived from `tau/H/1/1215`.

3. **Only Lyman-alpha tau** — only `tau/H/1/1215` (HI 1216 Å) is used.
   No higher Lyman series, no metal lines.

4. **Primary file is grid_480** — 691200 sightlines on a 480×480×3 grid.
   When absent, falls back to `lya_forest_spectra.hdf5` (32000 sightlines).

5. **Snapshots.txt is authoritative** for (snap, a) mapping. If a SPECTRA
   folder exists without a matching Snapshots.txt entry, the redshift is
   read from the HDF5 Header.

## Absorber identification

6. **System = connected tau region** — an absorption system is defined as a
   contiguous set of pixels where `tau > tau_threshold`. The post-audit
   production default is **`tau_threshold = 100`** (`config/default.yaml`),
   which together with `b = 30 km/s` corresponds to log10(NHI) ≳ 16.85 — well
   below the LLS edge. The downstream `min_log_nhi = 17.2` filter then
   discards sub-LLS detections.

7. **Merging criterion** — two adjacent blobs are merged into one system if
   the gap between them is < `merge_dv_kms` (default: 100 km/s) in velocity
   space. This prevents artificial splitting of DLAs with narrow sub-structure.

8. **NHI via the τ sum rule (fast_mode, production default)** —
   `absorber.fast_mode = true` in `config/default.yaml`. NHI is recovered from
   the τ sum rule: `NHI = (Σ τ · dv) / σ_integrated` over the above-threshold
   core. The pre-audit Voigt-fit estimator (single 2-parameter Voigt
   component) is retained as a fallback when `fast_mode = false` but is no
   longer the production path; see `docs/fast_mode_physics.md` for the
   derivation, truncation analysis, and Prochaska comparison.

9. **Classification thresholds** (standard literature values):
   - LLS    : 10^17.2 ≤ NHI < 10^19.0 cm^-2
   - subDLA : 10^19.0 ≤ NHI < 10^20.3 cm^-2
   - DLA    : NHI ≥ 10^20.3 cm^-2

   `min_log_nhi = 17.2` removes sub-LLS detections from the saved catalog.

## P1D

10. **Mean flux normalisation** — delta_F = F/<F> - 1 where <F> is computed
    globally over all skewers in the file (not per-skewer).

11. **No mean flux rescaling** — we do not rescale tau to match observed mean
    flux. If rescaling is needed, apply it externally before running this pipeline.

12. **k units** — k is in s/km (inverse velocity). Conversion to h/Mpc:
    k [h/Mpc] = k [s/km] × H(z)/h × 1000.

13. **Box boundary** — sightlines are periodic (from MP-Gadget). No windowing
    is applied to the FFT.

## CDDF

14. **Absorption path** — uses the canonical Bahcall-Peebles (1969) form:
    `dX = (1+z)² · L_comoving_Mpc · H_0/c`.
    Six unit tests in `tests/test_absorption_path.py` lock this to the
    derivation, an inline port of `fake_spectra.unitsystem.absorption_distance`,
    and a numerical-integration cross-check. (Pre-audit code used a
    `(1+z)·L_com·H_0/c` form which combined with a hardcoded `H_0=100` to give
    a `dX_buggy = dX_correct/[(1+z)·h]` error — bug #7 in `bugs_found.md`.)

15. **NHI bins** — log10(NHI) from 17 to 23 in 30 bins by default (bin width
    0.2 dex). The NHI-distribution figure uses bin width 0.1 with
    `np.linspace` so the 17.2 / 19.0 / 20.3 class boundaries fall on exact
    edges (bug #4).

16. **Masking** — production uses the PRIYA recipe only: `max(τ) > 10⁶`
    threshold, contiguous mask around `argmax(τ)`, expand outward while
    `τ > 0.25 + τ_eff`, fill with `τ_eff`. LLS / subDLA are **not** spatially
    masked; their residual P1D contribution is recovered via the Rogers+2018
    α template in post-processing (`hcd_analysis/hcd_template.py`).
    The pre-audit τ-space per-class mask is deprecated (over-masked the
    forest at k > 0.03 — bug #3). See `docs/masking_strategy.md`.

## General

17. **Cosmology is per-sim** — omega_m, omega_l, h are read from the HDF5 Header
    for each file. There is no global assumed cosmology.

18. **Box size is fixed** — all sims use a 120 Mpc/h comoving box.

19. **No peculiar velocity correction** — the tau array from fake_spectra
    already includes peculiar velocity contributions (it is computed in
    redshift space). No correction is applied.
