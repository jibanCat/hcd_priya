# Student project ideas

This is a curated list of self-contained projects that build on the
existing pipeline.  The catalog and the per-class P1D / CDDF / dN/dX
products are already produced for all 60 LF + 4 HiRes simulations across
~20 snapshots each, so most projects here are *analysis* problems on
top of cached data — no new simulations or new pipeline infrastructure
needed.

Each project lists:

- **Difficulty** (entry / intermediate / advanced)
- **Time estimate** (in weeks of part-time work)
- **Prereqs** (which tutorial notebooks to do first; sometimes a paper)
- **Concrete deliverable** (the figure / table / number you should
  produce)
- **Pointers** (the files and docs to start from)

If you finish one quickly and want a follow-up, every project has an
"Extensions" line at the end that suggests a natural next step.

---

## Recommended starter path  (4–7 weeks, A1 → B1 → B2)

The science focus for a new student should be the **CDDF / dN/dX
studies in section B** — that is where the project most needs new
work, and the deliverables there are paper-figure-quality.

But before going into B you should do one short project from
section A.  The reason: B's deliverables interpret per-class CDDF /
dN/dX values, and to interpret them honestly you need to know the
*error budget* of the underlying NHI catalog.  Skipping A and going
straight into B is feasible but you risk over-claiming on features
that turn out to be inside the catalog's measurement error.

The recommended sequence is:

| Step | Project | Why this one | Time |
|------|---------|--------------|------|
| 1 | **A1.** Voigt vs fast-mode NHI residuals on 100 DLAs | Builds intuition for what "log NHI = 20.5" means at the per-absorber level — bias, scatter, and where each estimator goes wrong.  The figure you produce will be cited from your B-project notes whenever a feature lives within ~0.3 dex of a class boundary. | 1–2 weeks |
| 2 | **B1.** dN/dX redshift evolution | Natural follow-up.  You already trust the NHI values from step 1; now use them to make the canonical "sim vs observation" figure across redshift.  Single-sim is fine for the first pass. | 1–2 weeks |
| 3 | **B2.** CDDF parameter sensitivity across the 60-LF LHS | The big-finish project.  Uses the full cached dataset (`hcd_summary_lf.h5` + the 60 `cddf_corrected.npz` files) to identify which of the 9 parameters drive HCD abundance.  This output directly informs the HCD emulator's parameter priors. | 2–3 weeks |

Optional add-on: if you finish step 1 quickly and have time before
step 2, **A2** (τ-threshold sensitivity) is a one-week addition that
deepens the same catalog-level understanding without distracting from
the B arc.

If steps 1–3 leave time at the end of the term, **B3** (multi-source
CDDF observational comparison) is the natural fourth project: it lets
you finish the term with a single high-quality figure that compares
your B1/B2 simulation results to the broader observational landscape.

The full project list below is for reference and follow-up choices —
the recommended path is the three rows above.

---

## How to start

1. Read the four tutorial notebooks in order: 00 → 04.  Make sure you
   can run them on your machine, not just look at the committed
   outputs — the moment you change `n_skewers` or pick a different
   sim, you're already doing science.
2. Read `docs/SESSION_HANDOVER_2026_04_28.md` (the most recent state of
   the project) and skim `docs/SESSION_HANDOVER.md` (the previous
   audit handover) for context.
3. Start on **A1** (the recommended starter; see the table above).
   Open a feature branch named for the project (e.g.
   `student/voigt-vs-fast-residuals`) and put any new code under
   `scripts/` and any new figures under `figures/analysis/`.
4. Use the existing test pattern in `tests/test_*.py`: write a small
   regression test for any general claim you make, *before* you write
   the prose summarising the result.  See
   `tests/test_absorption_path.py` for the three-independent-
   verifications template.
5. Check in with one of the project leads early (after ~3–5 days of
   work) so we can sanity-check the direction.

---

## A. Catalog & per-spectrum diagnostics  (entry level)

These build directly on tutorial notebook 04 and require no new
infrastructure.  They are good first projects to verify your
understanding before tackling something larger.

### A1. Voigt vs fast-mode NHI residuals on 100 DLAs

**Difficulty.** Entry.  **Time.** 1–2 weeks.
**Prereqs.** NB04.

The production LF catalog uses the fast NHI estimator
(`nhi_from_tau_fast`) for speed, which has nominal 0.1–0.3 dex error.
Quantify the bias for DLAs specifically by re-fitting 100 randomly
selected DLAs with `fit_nhi_from_tau` (Voigt) and comparing.

**Deliverable.**

- A figure showing `log(NHI_voigt) − log(NHI_fast)` as a function of
  `log(NHI_fast)`, with median and 16/84-percentile bands.
- A one-paragraph summary in `docs/` answering: is the fast estimator
  biased high, low, or unbiased at log NHI ≥ 20.3?  Does the bias
  grow with NHI?

**Pointers.**

- `hcd_analysis/voigt_utils.py` — both estimators
- `hcd_analysis/catalog.py:measure_nhi_for_system` — how the production
  pipeline uses each
- `docs/fast_mode_physics.md` — sum-rule derivation that motivates fast
  mode
- `docs/dla_truth_validation.md` — independent particle-based truth
  comparison; your residuals can be cross-referenced against the
  ±0.05 dex bias measured there.

**Extensions.** Repeat for subDLA and LLS — the fast estimator is
expected to do *worse* in the optically-thin regime where damping
wings carry less information.

---

### A2. tau-threshold sensitivity study

**Difficulty.** Entry.  **Time.** 1 week.
**Prereqs.** NB04.

The current pipeline detects HCDs by `tau > tau_threshold` with
`tau_threshold = 100`.  How sensitive is the resulting catalog
(per-class counts, dN/dX) to this choice?

**Deliverable.**

- A plot of dN/dX(LLS), dN/dX(subDLA), dN/dX(DLA) as a function of
  `tau_threshold ∈ {10, 30, 100, 300, 1000}` for one sim at one snap.
- A short note in `docs/` on whether 100 is in a flat plateau or near
  a sensitivity edge.

**Pointers.**

- `hcd_analysis/catalog.py:build_catalog` — the entry point.  You can
  call `build_catalog` directly with a different `tau_threshold` on a
  small subsample (say `n_skewers=5000`) and check sensitivity in a
  few minutes per threshold value.
- `meta.json["n_absorbers"]` — the production count is at threshold 100.

**Extensions.** Sweep `min_pixels` and `merge_dv_kms` jointly — there
are three knobs and the catalog's stability under all three is the
right thing to know.

---

### A3. Multi-component absorber detection

**Difficulty.** Intermediate.  **Time.** 2–4 weeks.
**Prereqs.** NB04, plus reading Cooke et al. 2018 §3 (or any paper on
DLA Voigt fitting).

The current Voigt fitter assumes a single component per absorber.
Real DLAs often have multiple velocity components (visible as multiple
saturated dips inside one detection window).  Build a 2-component
Voigt fitter and compare on the 100 strongest DLAs in one sim.

**Deliverable.**

- A `fit_nhi_from_tau_2comp` function in `hcd_analysis/voigt_utils.py`
  with tests.
- A figure showing 4–6 example DLAs where the 2-component fit
  visibly improves the wing match compared to the 1-component fit.
- A scalar summary: how often does 2-component reduce the fit χ² by
  > 30%?

**Pointers.**

- `scipy.optimize.minimize` is already used; just add 3 more
  parameters (NHI₂, b₂, v₀₂).
- Be careful with degeneracy: when the components are too close, the
  two-component fit becomes equivalent to one component with a wider
  b.  A reasonable initialisation: peak-finding inside the detection
  window to seed the two centres.

**Extensions.** Replace the catalog's single-component NHI per absorber
with the sum of fitted components, and re-measure the CDDF.  Does the
DLA-end excess vs Prochaska+05 shift?

---

## B. CDDF and dN/dX studies  (entry / intermediate)

These build on NB02.  They are a good fit for someone who likes
plotting and statistical comparisons against observations.

### B1. dN/dX redshift evolution

**Difficulty.** Entry.  **Time.** 1–2 weeks.
**Prereqs.** NB02.

For one sim, plot dN/dX(DLA), dN/dX(subDLA), dN/dX(LLS) as functions
of redshift over all available snapshots.  Compare against PRIYA
papers (Bird+2017, Khaire+2024) and BOSS/DESI compilations.

**Deliverable.**

- A 3-panel figure showing the three dN/dX(z) curves with literature
  comparison bands.
- A short note describing where the sim agrees / disagrees with
  observations and at what redshift.

**Pointers.**

- `cddf_corrected.npz` per (sim, snap) already has dN/dX bin counts;
  see notebook 02 §3 for the integration-over-NHI recipe.
- `docs/cddf_model.md` — the analytic CDDF model in the package, which
  has its own dN/dX(z) prediction you can overplot.
- `/home/mfho/DLA_data/` for observational compilations.

**Extensions.** Repeat for all 60 LF sims and show the within-LHS
scatter at fixed redshift.  Decide whether the scatter grows or shrinks
with z.

---

### B2. CDDF parameter sensitivity

**Difficulty.** Intermediate.  **Time.** 2–3 weeks.
**Prereqs.** NB02 and basic linear regression.

Across the 60 LF sims at fixed snapshot, which of the 9 parameters
(`ns`, `Ap`, `herei`, `heref`, `alphaq`, `hub`, `omegamh2`,
`hireionz`, `bhfeedback`) drives the variation in CDDF amplitude at
the DLA end?

**Deliverable.**

- A heatmap or correlation table showing the partial correlation
  between each of the 9 parameters and `f(N_HI = 10²⁰·⁵)` at z = 3.
- A short note arguing which 1–2 parameters dominate.

**Pointers.**

- `hcd_summary_lf.h5` already has `params/*` and `dndx/*` per
  (sim, snap).  You may need to extend it with `f(N_HI=10²⁰·⁵)`,
  which is just an interpolation on `cddf_corrected.npz`.
- A Spearman partial correlation works: it captures monotonic
  dependence without assuming linearity.

**Extensions.** Repeat for `f(N_HI = 10¹⁹)` (subDLA range) and
`f(N_HI = 10¹⁷·⁵)` (LLS range).  Does the dominant parameter change
with NHI?

---

### B3. CDDF vs multiple observations

**Difficulty.** Entry.  **Time.** 1 week.
**Prereqs.** NB02.

The repo currently compares against Ho+21 and Prochaska+05.  Add at
least two more compilations (Crighton+15, O'Meara+13, DESI EDR DLA
catalog) and produce a 4-panel figure (one per redshift bin) with all
sources.

**Deliverable.**

- A 4-panel CDDF figure at z = {2.2, 2.5, 3.4, 4.5} with at least 4
  observational sources overplotted.

**Pointers.**

- `scripts/plot_cddf_vs_ho21.py` — the existing 1-source script;
  extend rather than rewrite.
- `/home/mfho/DLA_data/` — already has many of the data files.

---

## C. P1D systematics  (intermediate)

Build on NB03.  These projects probe how mask choice and threshold
choices affect the per-class P1D.

### C1. Masking systematic on P1D

**Difficulty.** Intermediate.  **Time.** 2–3 weeks.
**Prereqs.** NB03, NB04.

For one sim at one snap, compute the P1D under all three masks:
`pixrange`, `tauspace`, and `priya`.  Quantify the spread between them
as a function of k.

**Deliverable.**

- A figure showing `P_masked(k) / P_priya(k) − 1` for each mask,
  alongside the per-bin Poisson error band.
- A statement: at which k do the masks differ by more than the
  Poisson error?

**Pointers.**

- `hcd_analysis/p1d.py:compute_p1d_single` accepts `mask_scheme`
  (`pixrange` or `tauspace`) and `compute_p1d_priya_masked` is the
  PRIYA path.
- `figures/diagnostics/p1d_masking/priya_mask_comparison.png` —
  there's already a 4-way comparison done in the audit; your job is
  to refresh and quantify it on a different sim.

**Extensions.** Repeat across the 60 LF sims at one z.  The
within-LHS spread of the masking systematic gives you an estimate of
the parameter-dependence of the systematic itself.

---

### C2. Rogers α-template fits across the LHS

**Difficulty.** Intermediate.  **Time.** 3–4 weeks.
**Prereqs.** NB03, plus reading Rogers+2018.

For each (sim, snap), fit the Rogers α-template to the per-class
P1D ratio `P_<class>_only / P_clean`.  Then study how each `α_class`
varies with the 9-D parameter vector.

**Deliverable.**

- A table of `(α_LLS, α_subDLA, α_DLA)` per (sim, snap).
- 3 partial-correlation tables (one per class) showing which
  parameters drive `α`.
- A note: is `α` more sensitive to cosmology or to astrophysics?

**Pointers.**

- `hcd_analysis/hcd_template.py:fit_alpha` — the Rogers fitter is
  already implemented.
- `figures/diagnostics/clustering/template_per_class.png` — what the
  template looks like for one (sim, snap).

**Extensions.** Train a Gaussian process emulator for `α(params, z)`
and fold it into the upstream Lyα-emulator pipeline as a parameter-
dependent template.

---

## D. Clustering  (advanced)

These come straight out of `docs/SESSION_HANDOVER_2026_04_28.md` §4.
Read that file first.  These are open research problems, not exercises.

### D1. Tighten β_DLA via LF+HR + z-stacking

**Difficulty.** Advanced.  **Time.** 1–2 months.
**Prereqs.** All four notebooks, plus
`docs/multipole_jacobian_explained.md` and `docs/clustering_test10_results.md`.

Current PR-#8 result: `β_DLA = −0.17 ± 0.27` on 11 655 DLAs at z = 2.2
— consistent with 0 because the cross-quadrupole signal is too weak.
Tighten by combining the 4 HiRes sims' DLAs with the LF sample, and
by stacking across redshift bins where `β_DLA` is expected to be
similar.

**Deliverable.**

- A combined-sample joint fit reporting `(b_DLA, β_DLA)` with
  Gaussian errors, plus a covariance ellipse plot.
- A statement on whether the new σ(β_DLA) is competitive with
  Pérez-Ràfols+2018 (BOSS DR12, ~30k DLAs, σ_β = 0.1).

**Pointers.**

- `hcd_analysis/lya_bias.py:fit_b_beta_from_xi_cross_multipoles`
- `scripts/run_test10.py --mode rmu` — the production driver.
- The cached test-10 grids in `figures/analysis/06_clustering/` are
  reusable for re-fitting without re-running pair counts.

---

### D2. ξ_FF FFT estimator

**Difficulty.** Advanced.  **Time.** 1–2 months.
**Prereqs.** All four notebooks, plus FFT-based 3D correlation function
literature (Sefusatti, Crocce).

Direct pair counting is O(N²) and infeasible for ξ_FF on 691k
sightlines × 1250 pixels per sim.  Implement a FFT-based estimator
that folds the box, FFTs, multiplies, IFFTs.  This is the long-term
unblock for the production-sweep clustering work.

**Deliverable.**

- A `xi_FF_fft` function with unit tests showing it agrees with
  `pair_count_rmu` on a small synthetic dataset.
- A wall-time comparison on one production sim (target: ≤ 1 minute,
  vs ~hours for direct pair counting).

**Pointers.**

- `hcd_analysis/clustering.py:pair_count_rmu` — the direct estimator
  to validate against.
- The `numpy.fft.rfftn` / `irfftn` interface is enough; you do not
  need a specialised library.

---

## E. HiRes vs LF convergence  (intermediate)

### E1. Per-class P1D HR/LF ratio on the 4 matched pairs

**Difficulty.** Intermediate.  **Time.** 2–3 weeks.
**Prereqs.** NB03.

The 4 HR sims under `hcd_outputs/hires/` are matched to specific LF
sims.  Compute `P_<class>_only_HR / P_<class>_only_LF` for each pair,
across all 4 classes and all available snapshots.  Identify the k
range where convergence is < 5%.

**Deliverable.**

- A 4×4 grid of plots (4 sims × 4 classes) with the HR/LF ratio
  vs k, with a horizontal band at ±5%.
- A table of "minimum k of < 5% convergence" per (sim, class) — i.e.
  the smallest scale on which the LF run is trustworthy for emulator
  training.

**Pointers.**

- `scripts/matched_pair_hr_vs_lf.py` already does this for global
  P1D; extend to per-class.
- `figures/diagnostics/p1d_masking/hires_vs_lf_subset.png` is the
  existing HR/LF comparison plot for one pair.

**Extensions.** Use the convergence cut as a `k_max` for emulator
training and re-measure how much information you lose vs k_Nyq.

---

## F. Emulator scaffolding  (intermediate, links to active work)

These overlap with the current emulator-scaffolding effort on
`joint-emulator-scaffold`.  Best done in coordination with whoever is
on that branch.

### F1. Phase-1 emulator data cache

**Difficulty.** Intermediate.  **Time.** 1–2 weeks.

Walk all 1076 (sim, snap) outputs and stack the four observables
(per-class P1D, CDDF, dN/dX, parameters) into a single in-repo HDF5,
interpolated onto a shared k-grid and NHI-grid.  This is the literal
phase-1 deliverable for the HCD emulator.

**Deliverable.**

- A `scripts/build_emulator_cache.py` that produces
  `hcd_analysis/_emulator_data/observables.h5` (gitignored).
- The cache is structured so that `params (1076, 9)` is one
  matrix-column-friendly array and the four per-class P1Ds are each
  `(1076, n_k_shared)`.

**Pointers.**

- The shared k-grid lives in `hcd_analysis/p1d.py:_DEFAULT_K_BINS`
  (50 bins to k_Nyq).
- The shared NHI-grid is the 30-bin grid already used in
  `cddf_corrected.npz`.
- `hcd_summary_lf.h5` is a precedent — same-shape file with scalar
  summaries; this project adds the per-bin arrays.

---

### F2. Mean-flux rescaling pipeline

**Difficulty.** Advanced.  **Time.** 1–2 months.

The cached per-class P1D is at the sim's natural mean flux.  An HCD
emulator with a mean-flux dimension needs the same per-class P1D at
multiple `<F>` values per sim.  Build the rescaling pipeline:
`tau → α · tau`, recompute per-class P1D, sweep α to cover the
desired τ_eff range.

**Deliverable.**

- A `scripts/build_meanflux_grid.py` that takes a (sim, snap), a
  list of α values, and produces a 3D array
  `(α, k, class) → P1D`.  Validate that α = 1 reproduces the cached
  per-class P1D to floating-point precision.
- A short note on the chosen α grid (how many points, how spaced) and
  why.

**Pointers.**

- `hcd_analysis/p1d.py:compute_p1d_per_class` — the function to
  re-call on rescaled tau.  The rescaling itself is one line:
  `tau_rescaled = alpha * tau`.
- The lya_emulator_full repo's `MeanFluxFactor` machinery is the
  reference parametrisation.

**Extensions.** Build a 2D emulator (params, τ_eff) → per-class P1D
ratio.

---

## What's already done (don't redo)

To avoid duplicating work, here's what's already in the repo:

- 60 LF + 4 HR simulations fully processed: catalog, CDDF, P1D, per-
  class P1D, P1D-with-PRIYA-mask, P1D ratios.  See
  `docs/SESSION_HANDOVER.md` §1 for the production state.
- Catalog validation against particle-based truth on the 4 HR sims.
  See `docs/dla_truth_validation.md` and
  `docs/dla_truth_unmatched_analysis.md`.
- 7 absorption-path / NHI-recovery bugs found and fixed.  See
  `docs/bugs_found.md`.
- HCD clustering: ξ_DD, ξ_FF, ξ_×, b_F monopole, b_DLA monopole,
  joint (b_DLA, β_DLA) on (r, |μ|) Hamilton-multipole grid (PR #7
  and PR #8).  See `docs/clustering_test10_results.md`.
- Rogers α-template fitting infrastructure: `hcd_analysis/hcd_template.py`.
- Per-class P1D infrastructure: `hcd_analysis/p1d.py:compute_p1d_per_class`.
- Five tutorial notebooks, this folder.

If you find yourself wanting to do something not on this list, that's
great — talk to one of the project leads and we'll add it.

---

## Hand-off etiquette

- Use a feature branch named `student/<short-project-tag>`.
- Commit small and often; aim for one clear commit message per
  conceptual change.  No huge dump commits.
- Open a draft PR early (after the first one or two figures) so we
  can give feedback before you go too deep.
- If you produce a new diagnostic figure, add it to `figures/analysis/`
  and link it from the relevant doc in `docs/`.
- If you change behaviour or add a new feature, write a unit test for
  it under `tests/test_<topic>.py`.  All `tests/test_*.py` should
  pass; that's the regression contract.
