# Bugs found during the audit

Summary of the defects discovered while auditing the HCD identification and masking pipeline in April 2026, and how each was detected, quantified, and fixed. Read this alongside `docs/fast_mode_physics.md`, which lays out the physics the fixes restore.

## 1. `_SIGMA_PREFACTOR` wrong by a factor of √π · 10⁵ = 1.77 × 10⁵

**Location**: `hcd_analysis/voigt_utils.py` lines 46–48 (pre-audit).

**What was wrong**. The prefactor in front of the Voigt profile was computed as

```python
_SIGMA_PREFACTOR = sqrt(pi) * e^2 * f * lambda / (m_e * c)
```

with `c` in cm/s but the Voigt profile φ_v returned in (km/s)⁻¹. Two independent errors:

1. **Unit mismatch (×10⁵)**. The formula missed a cm → km conversion. `_SIGMA_PREFACTOR` should have been divided by 10⁵.
2. **Normalisation (×√π)**. The canonical velocity-space integrated absorption cross section is

   ∫σ(v) dv = **π** · e² · f · λ / (m_e · c)

   (Ladenburg–Reiche sum rule; Draine 2011 *Physics of the ISM* eq. 6.15). The codebase used √π instead of π.

Combined, `_SIGMA_PREFACTOR` was a factor of √π · 10⁵ = 1.77 × 10⁵ too large relative to the correct velocity-space sum rule.

**How this manifested in production data**. The codebase's `tau_voigt` function produced τ values 1.77 × 10⁵ times larger than the canonical Voigt profile at the same (N_HI, b). Fake_spectra output uses the canonical normalisation, so when `fit_nhi_from_tau` inverted real τ using the broken forward model, it recovered N_HI values 1.77 × 10⁵ ≈ 10⁵·² too small. The pipeline's 17.2 min-log-N filter then discarded most true DLAs entirely.

**Concrete example**. Sightline 207 in `ns0.803.../SPECTRA_017/lya_forest_spectra_grid_480.hdf5` has max(τ) = 1.16 × 10⁸ (clearly a strong DLA, true log N ≈ 21.8). Pre-fix fit returned log N = 16.87 → below 17.2 → dropped from catalog. Post-fix fit on the same row returned log N = 21.87 ✓.

**How it was detected**. Two steps:

1. The NHI injection test (`tests/test_nhi_recovery.py`) showed the fitter is self-consistent on clean synthetic input (synthesise and fit with the same broken Voigt profile → round-trip works). This was initially reassuring but deceptive.
2. A **cross-normalisation** test was added that synthesised τ using an independent physical formula (`tau_voigt_physical` in the same test file, which uses the canonical oscillator-strength formula verbatim from Draine §6.1) and then ran the codebase's fit. That test showed a 0.25 dex residual bias after the first (×10⁵) fix — which is exactly log₁₀(√π). That led to the second fix.
3. A direct sum-rule check — `∫τ(v)dv = N_HI · _SIGMA_PREFACTOR` on a fine velocity grid — gave ratio = 1.0000 only after both corrections.

**The fix**. `hcd_analysis/voigt_utils.py`:

```python
_SIGMA_PREFACTOR = (
    np.pi * _E_CGS**2 * _F_LU * _LAMBDA_LYA_CM / (_M_E_CGS * _C_CGS)
) / 1.0e5
# Units: cm^2 * (km/s).  Numeric value: 1.3435e-12.
```

and the compensating `* 1e5` factors in `fit_nhi_from_tau`'s initial guess (line 206) and `nhi_from_tau_fast`'s thick-mode inversion (line 278) were removed. Matching `_tau_min` calculation in `catalog.py:382` was also updated.

**Blast radius**. All 925 production `catalog.npz` files were built with the broken prefactor and have N_HI values biased too low by ≈5 dex for any saturated absorber; most true DLAs are missing entirely (filtered by min_log_nhi=17.2). Downstream `cddf.npz`, `p1d_ratios.npz` with class-based masks, `p1d_excl.npz`, `cddf_stacked.npz`, and the perturbed-P1D products are affected. Unaffected: `p1d_all`, `p1d_no_DLA_priya` (tau-based, no NHI dependence), headers/metadata. Full rerun needed.

**Verification**. After the fix:

- `tests/test_tau_sum_rule.py`: sum-rule ratio = 1.0000 for all (log N, b) on the test grid.
- For NHI = 10²⁰·³, b = 30 km/s, the codebase's `tau_voigt` peak = 5.04 × 10⁶, matching Draine 2011 eq. 6.28 to all displayed digits.
- Real-data spot check on 10 sightlines with max(τ) > 10⁷: post-fix fits give log N ∈ [20.99, 21.87] (all DLAs), pre-fix fits gave log N ∈ [15.99, 16.87] (all discarded).
- Snap 017 + snap 010 full rebuilds track Prochaska+2014 CDDF within 0.3–0.8 dex — see `docs/fast_mode_physics.md` §Validation.

**Lesson**. Internal consistency (synth + fit with the same broken model) does not catch normalisation bugs. Every forward-model function should have at least one cross-check against an independent reference — either an external package (fake_spectra, which uses the canonical form), a textbook numerical example, or an analytical sum-rule identity.

## 2. Class-based mask covers only the τ > 100 core, not the damping wings

**Location**: `hcd_analysis/masking.py:apply_mask_to_skewer` and `build_skewer_mask`.

**What is wrong**. `AbsorberCatalog.pix_ranges()` returns `(pix_start, pix_end)` for each absorber, where `pix_start..pix_end` is the τ > τ_threshold = 100 detection region only. `build_skewer_mask` marks exactly those pixels as masked. The class-based masks (`no_LLS`, `no_subDLA`, `no_DLA`, `no_HCD`) therefore replace only the saturated core and leave the damping wings (τ ∈ [0.25, 100]) entirely in the spectrum.

**Why this matters scientifically**. For a DLA at log N = 20.3, b = 30 km/s the damping wings at τ > 0.25 extend several hundred km/s past the boundary of the τ > 100 core. That residual wing carries most of the correlated flux deficit that Rogers et al. 2018 identify as the dominant HCD contribution to the 1D flux power spectrum. Masking the core alone removes almost none of the large-scale P1D bias.

**How it manifested**. I measured P1D ratios directly from the production `p1d.npz` at snap 017 (`ns0.803`). Across the emulator k range (10⁻³ – 10⁻² s/km), `p1d_no_DLA / p1d_all = 1.000 ± 0.001`, and `p1d_no_HCD / p1d_all` is similarly pinned at 1. The stacked diagnostic plot `figures/intermediate/p1d_masking.png` (across all 953 completed snaps from the pre-fix pipeline) shows ratios indistinguishable from 1 across the emulator range, with deviations only near the Nyquist frequency where a handful of pixels is enough to matter. Independently of bug #1, this would already have made the class masks useless for measuring the Rogers-style HCD template.

**Status**. Not yet fixed. This is Phase B of the fix plan (see TODO). Proposed replacement: for each system, walk outward from the τ peak until τ < `wing_threshold + τ_eff`, producing a contiguous mask whose width depends on the local τ profile rather than a fixed pixel range. `wing_threshold` becomes a tunable per class (`0.25` for DLA per Rogers / PRIYA, larger for subDLA/LLS). Generalises the existing PRIYA DLA mask (`masking.priya_dla_mask_row`) to all three HCD classes.

## 3. PRIYA DLA mask was possibly over-restricted by an uncommitted contiguity patch

**Location**: `hcd_analysis/masking.py:priya_dla_mask_row` — uncommitted working-tree change relative to commit `c3b48d2`.

**What changed**. Previously the PRIYA mask was `mask = tau_row > (tau_mask_scale + tau_eff)` — all pixels above threshold, anywhere on the sightline. The uncommitted change restricts the mask to a **single contiguous region around the argmax of τ**, expanded outward until τ drops below threshold. Scattered IGM pixels elsewhere on the sightline that happen to exceed `0.25 + τ_eff` are no longer masked.

**Impact**. For a single-DLA sightline the two behaviours are identical (the damping wing is already contiguous around the peak). For sightlines with two DLAs, or a DLA plus an unrelated saturated LLS, the old behaviour masked both; the new one masks only the brightest. For sightlines with bright forest structure, the old behaviour erroneously masked forest pixels; the new one does not.

**Measured effect on P1D**. At snap 017, `p1d_no_DLA_priya / p1d_all` ≈ 0.999 across the emulator k range — too close to 1 to diagnose the difference between the two variants without a dedicated A/B test. The original contiguity argument in the PRIYA paper (arXiv:2306.05471 §3.3) is physically reasonable (the mask should correspond to the wings of a single identified DLA), so the patch is probably correct in direction; the question is whether stopping at the first τ dip loses legitimate wing pixels when the profile has brief under-threshold notches.

**Status**. Not committed. Re-run an A/B comparison once the Phase B per-class mask is in place and decide whether to keep the contiguity restriction.

## 4. `fake_spectra` integration is documented but not wired

**Location**: `docs/fake_spectra_integration.md` + `hcd_analysis/voigt_utils.py:59–73`.

**What's documented**. README and the integration doc describe fake_spectra as a live dependency whose line parameters, Voigt profile, and absorber-finder are used by the pipeline.

**What actually runs**. `import fake_spectra` raises `ModuleNotFoundError` in the current environment (`python3 -c "import fake_spectra"`). The `_HAVE_FAKE_SPECTRA` / `_FS_VOIGT_OK` flags in `voigt_utils.py` are always False. The pipeline has always been running the pure-scipy fallback.

**Impact**. Minor: the scipy path is correct after bugs #1 is fixed, and fake_spectra would provide the same line parameters. No scientific wrong-answer. But documentation is misleading — it suggests a rigour the code does not have. At minimum the doc should be rewritten to say "we re-implement the canonical formulas, independently of fake_spectra."

**Status**. Cosmetic. Fix in the docs-cleanup phase.

## 5. `docs/assumptions.md` contains stale parameter values

**Location**: `docs/assumptions.md:26` says "tau_threshold default: 1.0". The actual default in `config/default.yaml` is 100.0. Also item 9 in that document lists thresholds (LLS/subDLA/DLA) correctly.

**Status**. Cosmetic; fix in docs cleanup.

## 6. The Phase B "τ-space per-class mask" over-masks forest pixels and adds high-k artefacts

**Location**: `hcd_analysis/masking.py:build_skewer_mask_from_tau`, `apply_tauspace_mask_to_batch`, and `iter_tauspace_masked_batches`; wired into `p1d.compute_p1d_single` via `mask_scheme="tauspace"`.

**What was wrong**. I implemented a generalisation of the PRIYA DLA mask to all HCD classes, with class-dependent wing thresholds (0.25 for DLA, 0.5 for subDLA, 1.0 for LLS) and a walk-outward-from-each-system-peak construction. The idea was that "damping wings" extended beyond the τ > 100 detection region for every HCD class and should be masked per class.

In a full-sample P1D test at snap 017 this mask produced a 3-4% deficit at low k and a 10-40% excess above k = 0.02 s/km (cyclic) versus the unmasked P1D. The PRIYA paper claims ≤1% across the emulator range, and the literal PRIYA recipe matches that claim on the same data. So my mask was clearly wrong.

**Why it's wrong**. Two compounding errors:

1. **LLS and subDLA don't have meaningful damping wings.** Their NHI is too low. Their τ profile drops through the `wing_threshold + τ_eff` boundary already inside the τ > 100 detection region — in other words, the "wing extension" of the mask walked into forest-level pixels that are not part of the absorber at all. I was masking real forest power.
2. **My wing thresholds were set by guessing at LLS/subDLA importance, not by physics.** LLS and subDLA P1D contamination is real but lives at low k (see Rogers+2018 α template), not in damping-wing extent. Spatial masking is the wrong instrument for that problem — the Rogers parametric correction in spectral domain is.

**How detected**. Via a full-sample A/B test of four mask schemes (no mask, PRIYA, my τ-space, my pixrange; `tests/validate_priya_mask.py`). The plot `figures/diagnostics/priya_mask_comparison.png` shows PRIYA stays within ±1 % of unmasked across the whole k range, while my τ-space mask diverges up to 40 %. Also shown per-class: the LLS-only template in `figures/diagnostics/template_per_class.png` is ≈1.0 across the whole k range — LLS is not a spatial-mask target at all.

**Fix**. Deprecate `build_skewer_mask_from_tau`, `apply_tauspace_mask_to_batch`, and `iter_tauspace_masked_batches` as production tools. Remove the `no_LLS`, `no_subDLA`, `no_DLA`, `no_HCD` variants from the default `ALL_VARIANTS` list in `p1d.py`. The production P1D variants reduce to `all` and `no_DLA_priya`. LLS/subDLA residuals are handled downstream via the Rogers+2018 α template (4 parameters).

**Status**. Code cleanup in progress (this audit commit). See `docs/masking_strategy.md` for the full evidence and the final recommendation.

**Related k-convention note.** PRIYA stores k in *angular* units (`2π · rfftfreq(nbins) · nbins / vmax`), while our `P1DAccumulator._k_native()` uses cyclic `rfftfreq`. A quoted `k_max = 0.1 rad·s/km` in PRIYA notation is `k_max ≈ 0.016 s/km` in ours. Both are below Nyquist (no zero-padding in either pipeline); this is just a labelling choice. See §4 of `docs/masking_strategy.md`.

## Things checked and ruled OUT

The audit considered but found to be fine:

- `io.py`, `snapshot_map.py` — HDF5 reading, folder discovery, Snapshots.txt parsing all work correctly; confirmed against directory listings.
- `p1d.py:P1DAccumulator`, `compute_p1d_single` — the streaming two-pass mean-flux + rfft P1D computation is consistent with Palanque-Delabrouille 2013 / Chabanier 2019 convention and documented correctly in `docs/p1d_definition.md`.
- `find_systems_in_skewer` (catalog.py) — threshold + merge + min-pixels detection is internally correct; wrap-around handling is right.
- The `_WING_PIXELS = 200` (±2000 km/s) expansion window in `measure_nhi_for_system` is the cause of the Voigt-fit over-counting observed post-fix (forest pixels in the window bias the single-Voigt fit upward); but this is a design issue, not a bug in the strict sense. The resolution is to default to `fast_mode=True`, which bypasses this window entirely.
