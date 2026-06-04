# Phase 2 — Per-class HCD P1D emulator with a mean-flux dimension — design spec

**Date:** 2026-05-17
**Branch:** `phase2-emulator-jax`
**Status:** approved design — **partially superseded 2026-05-20 (see banner)**
**Supersedes the open questions in:** `docs/SESSION_HANDOVER_2026_05_15.md` §9,
`docs/superpowers/2026-05-15-phase2-design-memo.md`

> ## ⚠️ PARTIALLY SUPERSEDED — 2026-05-20
>
> A PRIYA bit-identity consistency check
> (`docs/superpowers/2026-05-20-priya-p1d-consistency-check.md`) showed the
> P1D pipeline must be built on `fake_spectra`'s actual machinery to match
> PRIYA's training data (verified to floating-point precision, 120 points).
> The following sections of THIS spec are revised by that work and the
> refactor plan `docs/superpowers/plans/2026-05-20-phase2a-refactor-fake-spectra.md`:
>
> - **§3 (τ₀ "freeze-core" recipe) → REVISED.** Freeze-core is NOT the
>   production recipe. The cache now has two tiers (the user's 2026-05-20
>   framing):
>   - **Tier P (baseline, PRIYA-compatible):** `fake_spectra`'s
>     `_filter_single_tau_complex` (`tau_thresh=1e6`, destructive DLA
>     trough-fill) → survey-equivalent total P1D. This is what the main
>     forest emulator trains on.
>   - **Tier C (HCD adds-on):** per-class P1D (clean/LLS/subDLA/DLA) on the
>     UNFILTERED τ, sharing Tier-P's mean-flux normalisation. Used to build
>     the HCD adds-on module that re-adds DLA contribution at inference.
>   - **Freeze-core is now a *future* Tier-C masking variant**, not the
>     production recipe — alongside future NHI≥20.3-cut and Rahmati
>     partial-self-shielding options.
> - **Mean-flux convention → CLARIFIED.** α is PRIYA's **Kim 2013
>   slope-multiplier** on `obs_mean_tau(z)=2.3e-3(1+z)^3.65`, NOT a direct
>   τ-multiplier. The τ-rescale is solved by `_rescale_mean_flux`. Redshift
>   is snapped to PRIYA's `zout` grid (multiples of 0.2) before computing
>   the mean-flux target.
> - **§4 (cache schema) → REVISED to v2.0.** Per-row datasets are now
>   `P_tier_p` + `P_clean`/`P_LLS`/`P_subDLA`/`P_DLA` (stored SEPARATELY, not
>   as ratios) + `n_clean`/`n_LLS`/`n_subDLA`/`n_DLA` + `target_F`/`scale`/
>   `z_meta`/`z_grid`/`alpha_slope`. Top-level attrs add `cache_version="2.0"`,
>   `priya_convention`, `tau_thresh`. See the refactor plan §"Task 4 / write_cache_tau0".
> - **§9 (validation) → AUGMENTED.** The headline validation is now the PRIYA
>   bit-identity multi-point test (`tests/test_priya_p1d.py`), not the
>   freeze-core physics unit tests (those move to the legacy
>   `hcd_analysis/tau0_rescale.py` path).
>
> §5 (architecture: encoder + Head A/B), §6 (loss), §7 (likelihood), §8
> (splits), §10-§14 are UNCHANGED by the 2026-05-20 work. Read the sections
> below as historical design rationale; where they conflict with the banner,
> the banner wins.

This spec was produced through a brainstorming session backed by eight research
agents (PRIYA/`fake_spectra` code dives, self-shielding astrophysics, a
Lyα-P1D-simulation survey, and a faithfulness brainstorm). It records the
decisions and the design; the implementation plan lives separately in
`docs/superpowers/plans/2026-05-17-phase2-hcd-emulator.md`.

---

## 1. Goal and scope

Build a JAX neural-network emulator of the Lyα-forest 1D flux power spectrum
(P1D), **split into four sightline classes** by the strongest absorber present
(clean forest / LLS / sub-DLA / DLA), with a **mean-flux (τ₀) dimension** so the
downstream Bayesian inference (HMC) can marginalise over UV-background
uncertainty. The headline requirement: the emulator-driven likelihood must be
**faithful**, especially in the τ₀-dependence of the HCD classes.

**In scope (Phase 2):**
- A τ₀-extended observable cache built by rescaling the Lyα optical depth.
- The JAX emulator (shared encoder + two heads) and its training loop.
- The likelihood-consumption contract (how 4 per-class P1Ds combine into one
  observable P1D).
- A validation battery, including a small set of ground-truth spot-checks.

**Out of scope (deferred to Phase 3):**
- Smooth column-density-dependent rescaling ("Tier B′", Rahmati Eq. 14).
- Multi-fidelity (LF/HF) emulation.
- A heteroscedastic uncertainty head (output slots reserved, training deferred).
- Freeze-boundary sensitivity sweep (boundary fixed at τ = 1e6 for Phase 2).
- CDDF as an independent likelihood data vector (interface wired, off by default).

---

## 2. Established background (inputs to this design)

- **PRIYA** (Bird et al. 2023) is the *only* published effort that keeps HCD
  sightlines and mean-flux-rescales them. Its `_filter_tau` with the production
  `tau_thresh = 1e6` masks the saturated cores of **all DLAs and ~56% of
  sub-DLAs** (verified: τ = 1e6 ↔ log N_HI ≈ 19.1–19.6), then applies a single
  uniform mean-flux factor to everything else. PRIYA never trains a per-class
  P1D; HCD contamination is handled at likelihood time by the Rogers+2018
  `DLA4corr` analytic multiplier with per-class nuisance amplitudes.
- **Self-shielding physics** (Rahmati et al. 2013, arXiv:1210.7808; Altay+2011;
  Bird+2014): the H I column response to the UVB rate Γ runs smoothly from
  `N_HI ∝ Γ⁻¹` (optically thin forest) to `N_HI ∝ Γ⁰` (self-shielded DLA core).
  A DLA's damping wings track its *total* N_HI and are Γ-insensitive. A uniform
  `τ → α·τ` rescale is rigorous **only** for optically-thin forest pixels.
- **The field treats the HCD P1D contribution as mean-flux-independent**
  (McDonald+2005, Rogers+2018, DESI DR1). Letting the HCD classes genuinely
  respond to τ₀ is new — it must therefore be *validated*, not assumed.
- The raw mock-spectra files (`/nfs/turbo/umor-yueyingn/mfho/emu_full/<sim>/
  output/SPECTRA_<NNN>/lya_forest_spectra_grid_480.hdf5`, dataset
  `tau/H/1/1215`) hold **only τ** — the density/temperature groups are empty.
  Re-deriving the neutral fraction at a new Γ requires re-running `fake_spectra`
  (v2.2.3, conda env `emu-3.9`).
- Phase 1 cache `observables.h5`: 1076 (sim, snap) rows, 9 sim params, shared
  50-bin angular k-grid, per-class P1D + per-class ⟨F⟩ + CDDF `f_nhi[30]` +
  per-class dN/dX. Built by `scripts/build_emulator_cache.py`.

---

## 3. The τ₀ rescaling recipe — "freeze-core"

The mean-flux dimension is generated by post-processing the optical depth, the
PRIYA approach, but with a physically-correct treatment of self-shielded gas.

**Recipe (production, "Tier B"):** for each (sim, snap), read the τ grid once;
for each α in the grid:

1. **Freeze** every pixel with native τ > `TAU_FREEZE = 1e6` — keep its native
   value, do not rescale. These are self-shielded saturated cores (DLAs and the
   strongest sub-DLAs); their optical depth physically does not respond to the
   UVB.
2. **Rescale** every other pixel: `τ → α·τ` (forest, LLS, weak sub-DLA, and the
   far damping wings — all optically thin, all UVB-responsive).
3. From this rescaled field, **recompute** per-class ⟨F⟩ and the four per-class
   P1Ds. The per-subset mean flux (Rogers convention) is *not* linearly
   rescalable, so it must be re-derived at each α.

**α grid:** 20 points, uniform in α over PRIYA's range ≈ [0.66, 1.36] with the
endpoint-shift (`coarse_grid.py:540-545`). `τ₀ = -ln⟨F⟩_clean` of the rescaled
state is the physical quantity the emulator sees; α is the internal generator.

**Freeze boundary:** fixed at τ = 1e6 (PRIYA's value) for Phase 2. Recorded as a
cache attribute. A sensitivity sweep of this boundary is explicit Phase 3 work.

**CDDF / dN/dX are τ₀-invariant.** Absorbers are classified on **native** N_HI
(catalogued from the native τ field). Because the cores are frozen, their N_HI
does not move with α — so the CDDF and per-class dN/dX are identical across all
α of a given (sim, snap). This is physically correct (self-shielded columns do
not respond to the UVB) and is **pinned by a unit test**. The per-class **P1D**
does carry real τ₀-dependence (its forest pixels rescale; ⟨F⟩_class moves).

**Tier-A twin (systematic).** Build a second cache under a *uniform* rescale
(`τ → α·τ` for all pixels, no freeze). The per-row, per-class, k-resolved
difference `P_TierB − P_TierA` is a measured systematic for the rescaling recipe
— cheap to produce and it converts an unquantified approximation into a number
the likelihood covariance can carry.

**Tier-C spot-checks (ground truth).** At 2–3 (sim, snap, α) points (one low α,
one high α), re-extract the per-class P1D with `fake_spectra` at a genuinely
scaled Γ. This is the only true test of the rescaling approximation; it
calibrates whether freeze-core is faithful enough or whether Phase 3 must adopt
Tier B′. Not the full grid — a calibration sample only.

---

## 4. Caches and schema

Two on-disk caches, both gitignored, alongside the immutable Phase 1
`observables.h5`:

- `hcd_analysis/_emulator_data/observables_tau0.h5` — Tier B (production).
- `hcd_analysis/_emulator_data/observables_tau0_uniform.h5` — Tier A (twin).

**Schema** (both files identical structure), keyed by (sim, snap, α_idx):

- Per-row scalars: `sim_name`, `snap`, `z`, `dv_kms`, `nbins_native`,
  `alpha`, `tau0` (= -ln⟨F⟩_clean), `alpha_idx`,
  per-class `mean_F_<class>`, per-class `n_sightlines_<class>`,
  per-class `dNdX_<class>`, `total_path_dX`.
- Per-row vectors: `params[9]`, four P1D arrays `P_<class>[50]`.
- Per (sim, snap), **not** replicated across α (stored once): `f_nhi[30]`,
  `n_absorbers[30]`, `log_nhi_centres`, `log_nhi_edges` — the τ₀-invariant CDDF.
- Shared: `k_target[50]`, `param_names`, attributes
  `tau_freeze=1e6`, `rescale_tier`, `alpha_range`, `source_commit`.

The τ₀-cache builder (`scripts/build_emulator_cache_tau0.py`) reuses the Phase 1
helpers `discover_sim_snap_pairs`, `interp_p1d_loglog` and the
`hcd_analysis/p1d.py` / `cddf.py` machinery. It reads the raw τ grid (not the
precomputed `p1d_per_class.h5`), applies the freeze-core rescale, and recomputes.

---

## 5. Emulator architecture

JAX (Flax or Equinox). Shared encoder + two heads — the architecture locked in
the 2026-05-05 handover, with τ₀ confined to Head B.

```
params (B,9), z (B,1)            tau0 (B,1)
        |                             |
        v                             |
   encoder MLP  [10 -> 256 -> 128 -> 64]
        |  latent (B,64)               |
        +-----> Head A [64 -> 128 -> 33]  -> f_nhi[30] + dNdX[3]
        |
        +-----> Head B [65 -> 256 -> 200] -> 4 x P1D[50]
                       ^ concat(latent, tau0)
```

- **Head A** takes the latent only (no τ₀ — CDDF/dN/dX are τ₀-invariant, §3).
- **Head B** takes `concat(latent, τ₀)` — P1D depends on both z (via latent)
  and mean flux.
- **Output transforms:** log-space targets for `f_nhi`, `dN/dX`, and all four
  P1Ds (dynamic range > 10⁴).
- Output slots are reserved for a future per-bin predictive variance
  (heteroscedastic head); Phase 2 trains point estimates only.

---

## 6. Loss design

A single joint scalar loss, NaN-safe:

- **Log-space MSE** per head/target (weights fractional error uniformly across
  k — the right metric for a fractional-accuracy likelihood).
- **α-multiplicity reweighting.** Head A's targets (CDDF, dN/dX) are identical
  across the 20 α of each (sim, snap). Its loss contribution is divided by the
  α-multiplicity so each physical (sim, snap) contributes once to the Head-A
  gradient, while Head B sees all 20 α rows. (Equivalently: the CDDF stored once
  per (sim,snap), §4, makes this natural.)
- **NaN / Nyquist mask.** ~2% of P1D bins are NaN above each snap's native
  Nyquist; the loss uses masked reductions (no imputation). The mask is
  **per-row** — native Nyquist varies by snap. The freeze-core rescale leaves
  the velocity grid unchanged, so the mask carries over verbatim.
- **CDDF–P1D consistency (auxiliary).** An auxiliary term on the *reconstructed
  contaminated P1D* (§7): the combination `Σ_c w_c·P_c` built from the model's
  own predictions is penalised against the cached contaminated P1D. This trains
  exactly what the likelihood evaluates and is the operational meaning of
  "tightly coupled to CDDF statistics" — soft consistency, not a hard
  architectural constraint.
- **Mean-flux consistency (auxiliary, clean class only).** Predicted
  `mean_F_clean` must equal `exp(-τ₀)` to tolerance. This forces a smooth,
  monotone τ₀-response. It is **not** applied to HCD classes — under freeze-core
  their ⟨F⟩ is not `exp(-τ₀)`; the cache-recorded `mean_F_<class>` is their
  target.
- Term weights: start equal among the likelihood-relevant terms (P1D, dN/dX);
  sweep later.

---

## 7. Likelihood-consumption contract

**Decision: the emulator *replaces* the Rogers+2018 `DLA4corr` analytic
multiplier.** The per-class emulated P1D *is* the HCD model.

The observable contaminated P1D is reconstructed as a class-fraction-weighted
combination:

```
P_obs(k, z) = Σ_c  w_c · A_c · P_c(k, z)
```

- `c ∈ {clean, LLS, subDLA, DLA}`.
- `P_c` — emulated per-class P1D (Head B).
- `w_c` — sightline-class fractions. Deterministic function of the emulated
  per-class dN/dX (Head A) and the mean sightline absorption path length. The
  cache stores `n_sightlines_<class>`, so `w_c = n_sightlines_c / n_total` is
  directly measurable; a unit test pins the dN/dX→w_c map against it. **The
  exact algebra (including path-length normalisation) is the first deliverable
  of the implementation plan** — it gates the loss's auxiliary term.
- `A_c` — one residual-amplitude nuisance per HCD class (`A_clean ≡ 1`), tight
  prior `N(1, small)`. A safety valve for emulator/recipe error, not a free
  fudge — if the emulator is faithful these stay ≈ 1.

**Likelihood nuisance parameters:**
- `τ₀` — mean flux / UVB. The headline nuisance; an emulator input.
- `A_LLS, A_subDLA, A_DLA` — residual HCD amplitudes, tight priors.
- Emulator error — a static per-(class, k, z) error vector measured on the
  held-out test set, added in quadrature to the data covariance. The likelihood
  interface accepts an emulator-covariance contribution from day one.
- Thermal state — the 9 sim params include the heat-injection parameters
  (`herei`, `heref`, `hireionz`); the IGM thermal state is implicit. **No
  separate T₀/γ nuisance is planned**; revisit only if validation shows the
  heat params are insufficient.

CDDF (`f_nhi`) is emulated and the likelihood interface is wired to accept it as
an optional second data vector with its own covariance — off by default in
Phase 2.

---

## 8. Training data and splits

- ~20 α × 1076 (sim, snap) ≈ **21,500 training rows**.
- **Hold out 6–8 whole sims** (all snaps, all α) for test — no τ₀-sibling
  leakage. Within the training sims, validation is on whole (sim, snap) blocks
  (all 20 α together) — even val must not split α-siblings.
- Additionally hold out the **extreme α** (lowest, highest) of a few training
  sims as an α-extrapolation probe (the HMC prior will query the α edges).
- Single-fidelity (LF) for Phase 2. The LF→HF P1D bias is measured on the
  matched HF/LF pairs and carried as a resolution systematic; multi-fidelity is
  Phase 3.

---

## 9. Validation battery

- **Leave-one-sim-out (LOSO)** — the headline generalisation test. Error
  reported **stratified by class, k-band, z, and α-band** (a single scalar test
  loss hides the failures that matter).
- **`P_clean` vs PRIYA forest P1D** — our clean-class P1D at native α must
  reproduce PRIYA's forest P1D (k-convention, normalisation, δF). A strong
  independent check of the whole clean pipeline.
- **Tier-A − Tier-B systematic** — produced from the twin caches, k- and
  class-resolved, carried in the likelihood covariance.
- **Tier-C spot-checks** — 2–3 `fake_spectra` re-extractions at scaled Γ;
  ground truth for the rescaling approximation.
- **α-interpolation test** — train on every-other α, predict the held-out α.
- **Physics unit tests** (the repo's TDD style, e.g. `tests/test_*.py`):
  - CDDF and dN/dX identical across all α of a (sim, snap) — τ₀-invariance.
  - Frozen pixels (native τ > 1e6) identical in rescaled vs native rows;
    non-frozen pixels scale by exactly α.
  - `mean_F_clean` of a rescaled row equals `exp(-τ₀)` to tolerance;
    `mean_F_DLA` does not (cores frozen).
  - NaN/Nyquist mask is per-row and unchanged after rescale.
  - dN/dX derived from catalog `n_absorbers`, never CDDF-bin-sums.
  - Angular k-grid convention (2π factor) preserved end to end.
- **Likelihood closure test** — feed a pipeline-generated (not emulator-
  generated) mock P1D at known (θ, τ₀) into the HMC likelihood; the posterior
  must recover the input within the emulator + systematic error budget. This
  gates "Phase 2 done."

---

## 10. Module breakdown

- `scripts/build_emulator_cache_tau0.py` — the freeze-core τ₀-cache builder
  (Tier A and Tier B via a flag). Blocked on nothing; reuses Phase 1 helpers.
- `hcd_analysis/emulator/data.py` — loader: ingests Phase 1 + τ₀ caches, yields
  batched `(params, z, τ₀, targets, masks)`. Can be built against a synthetic
  τ₀-cache before the real builder lands, given the §4 schema is fixed.
- `hcd_analysis/emulator/model.py` — encoder + Head A + Head B + the joint loss.
- `hcd_analysis/emulator/train.py` — training loop, checkpointing, the static
  emulator-error-vector computation on the held-out set.
- `hcd_analysis/emulator/likelihood.py` — the §7 contract: combination algebra,
  nuisance parameters, emulator-covariance interface.
- `scripts/spot_check_tau0_fake_spectra.py` — the Tier-C re-extraction harness.
- `tests/test_emulator_tau0_cache.py`, `tests/test_emulator_model.py`,
  `tests/test_emulator_likelihood.py` — the §9 unit tests.

---

## 11. Phasing

**Essential for a first faithful Phase-2 emulator:**
1. τ₀-extended cache — Tier B production + Tier A twin, per-α recomputed
   ⟨F⟩_class, native-τ CDDF classification.
2. Loader + JAX model + training loop; log-space, NaN-safe joint loss with
   Head-A α-multiplicity reweighting.
3. The written-down combination algebra (§7) and the likelihood interface,
   consuming 4 P1Ds + dN/dX with tight residual amplitudes.
4. Static per-(class, k, z) emulator-error vector into the covariance.
5. Validation: LOSO + PRIYA forest cross-check + physics unit tests + a few
   Tier-C spot-checks + the likelihood closure test.

**Deferred (Phase 3), in payoff order:**
Tier B′ smooth Rahmati attenuation (only if the Tier-A−B systematic is large at
likelihood-relevant k) → heteroscedastic uncertainty head → multi-fidelity
LF/HF → CDDF as an independent likelihood data vector → freeze-boundary
sensitivity sweep.

---

## 12. Decisions settled (2026-05-17, with the user)

1. **Replace** Rogers `DLA4corr` with the emulated per-class P1D — agreed.
2. **Freeze boundary τ = 1e6** — agreed; sensitivity test deferred to Phase 3.
3. **20 α-grid points** uniform in α over [0.66, 1.36] — agreed (first try).
4. **Tier-C spot-checks** (2–3 `fake_spectra` re-extractions) scoped into
   Phase 2 as ground-truth calibration — agreed.

Earlier locked decisions still in force: JAX framework; 3 separate HCD classes
(not summed); in-repo gitignored small caches; per-subset ⟨F⟩ (Rogers
convention); the encoder/Head sizes above.

---

## 13. Riskiest unknowns (carried into the plan)

- **The combination algebra** (§7) — exact `w_c` from dN/dX, path-length
  normalisation. First plan deliverable; everything downstream depends on it.
- **Is freeze-core faithful enough?** — answered only by the Tier-A−B
  systematic + Tier-C spot-checks. The plan must fund the spot-checks even
  though they look like a side quest.
- **Tier-C cost** — re-running `fake_spectra` for even 2–3 points needs the
  particle snapshots and a working `fake_spectra` extraction; confirm
  feasibility early.

## 14. References

- `docs/SESSION_HANDOVER_2026_05_15.md` §4, §9
- `docs/superpowers/2026-05-15-phase2-design-memo.md`
- `docs/superpowers/2026-05-15-priya-coarse-grid-dive.md` (§6 corrected
  2026-05-17 — `tau_thresh=1e6` is a real DLA cut, not "off")
- Rahmati et al. 2013 (arXiv:1210.7808); Altay et al. 2011 (arXiv:1012.4014);
  Bird et al. 2014 (arXiv:1405.3994); Rogers et al. 2018 (arXiv:1706.08532);
  Bird et al. 2023 PRIYA (arXiv:2306.05471); DESI DR1 P1D cosmology
  (arXiv:2601.21432).
