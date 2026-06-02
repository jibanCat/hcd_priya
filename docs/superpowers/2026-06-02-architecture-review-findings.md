# Phase-2b architecture review — consolidated findings (CS/ML + cosmology)

**Date:** 2026-06-02. Two parallel expert reviews of the spec + plan + implemented code
(`hcd_analysis/emulator/*`). Both read-only. This consolidates and prioritizes by **when it
bites**: (A) pre-training engineering, (B) the user's requested tasks 2 & 3, (C) pre-cosmology
physics gates. Items flagged by BOTH reviews are marked ★ (higher confidence).

---

## A. Pre-training engineering (must fix before/with training) — from CS review

- **A1 (Critical) — target transform/normalization is UNWIRED.** `safe_log`/`signed_log`/
  `apply_norm`/`fit_norm` are defined in `data.py` but never connected; `joint_loss` compares raw
  predictions to raw *linear* cache values. The spec's "log-space P1D / standardized outputs" is
  **false in code today**, and a linear-space MSE is dominated by the largest-amplitude bins.
  **Fix:** add `make_batch(d, idx, norm_stats)` applying `safe_log`(P_filt)/`signed_log`(Δ) +
  train-split `apply_norm`; the loss already assumes this — it has no producer. Highest-leverage gap.
- **A2 (Critical) — the "exact structural identity" is a *cache* property, not the *trained*
  reconstruction**; the invert→exp→`structural_tier_p` composition is never tested end-to-end.
  **Fix:** integration test (Emulator → invert_norm → exp → `structural_tier_p` vs cache `P_tier_p`);
  soften spec language to "structural (not free), accurate to emulator error in P_c^filt."
- **A3 (Important) — `meanF` loss term is DEAD** (`joint_loss` `mean((mean_F_clean−exp(−τ0))²)`
  depends only on `batch`, not `model` → zero gradient). **Fix:** connect to a model output or delete.
- **A4 (Important) — `term_w` is hand-set** (`{...,meanF:0.1}`), contradicting spec §4 "no hand-sweep";
  only defensible once A1 standardizes every channel. **Fix:** after A1, drop magic weights / use
  running-loss or gradient-norm balancing.
- Minor: LOSO×τ₀-holdout composition can leak HMC-extrapolation rows into training (M4); `assemble_
  covariance` diagonal-only (M2, see C-side too); Head-A→Xbar→w_c chain only tested with literal Xbar (M3).

## B. The user's tasks 2 & 3 — refined by both reviews

- **Task 2 — learned low-rank output bottleneck (★ CS-endorsed; the lit §8 preferred refinement).**
  Dense Head B is ~370k params on ~1200 rows — the dominant overfitting surface. **Concrete guidance
  (CS I2):** apply to `P_filt` only (head-specific toggle `n_basis≈8–16`); **init the basis from an
  SVD of the cache `P_filt`** (warm start — avoids rank collapse); keep `Δ_c` dense/arcsinh (its
  low-k sign flip can't go through a positive/log basis); the structural sum through a shared basis
  is fine (`Σ_c w_c·(B·code_c)=B·(Σ_c w_c code_c)`). Add a rank-collapse guard test.
- **Task 3 — `α_c(z)` (2-node/PW14) + analytic `M₀`-inverse `alpha_to_dndx`.** Both reviews confirm
  the α_c→effective-dN/dX readout is sound (Rogers&Bird §6 framing). **Add (CS I3):** clamp `delta_c(z)`
  to the fit range z∈[2,5.4] (the deg-2 poly extrapolates unbounded; the τ₀-edge holdout probes
  extremes); load the per-class δ_c residual-std as the `α_c` prior width.

## C. Pre-cosmology physics gates (before trusting cosmology) — from cosmology review

- **C1 (Critical) — the deferred damping-wing systematic is exactly the channel `α_c` is fit
  against.** Uniform rescale scales the Γ-insensitive wings as if optically-thin → `Δ_c(τ₀)` biased
  in the shape that separates classes + marginalizes mean flux. **Fix:** run `diag_tierc_tau0_ratio.py`
  across the full τ₀ grid, bound the wing's spurious τ₀-response at low/intermediate k, and carry it
  as a **τ₀-correlated model-error term in the covariance** — not a Phase-3 prose deferral.
- **C2 (Critical) ★ — baseline-masking mismatch can bias COSMOLOGY, not just nuisances.** `P_tier_p`
  uses the sim filter (`tau_thresh=1e6`, near-total self-shielded clip) + sim `w_c`, but the data's
  DLA finder is ~70% complete for N_HI>2e20 and misses ~all LLS/sub-DLA. So the sim baseline removes
  *more* than the data → `α_c·Δ_c` must add back an **O(1)** residual (not a small nuisance), exactly
  where LLS↔cosmology degeneracy is worst. **Fix (resolve NOW, not "when wiring"):** measure the sim
  filter's per-class removal fraction vs the survey finder's completeness/purity; either rebuild the
  filtered tier at the survey masking, or prove `α_c` stays in the small-residual regime the one-sided
  prior assumes. (Needs the target survey's DLA-completeness spec — a real-data input.)
- **C3 (Important) — emulated `Δ_c(cosmology,τ₀,z)` is more flexible than the field-standard FIXED
  shape**; extra shape freedom can absorb cosmological signal. **Fix:** make "Δ_c reduces to the
  Rogers kernel at fiducial cosmology" a **hard closure gate** (inject Rogers-template mock → recover
  unbiased Ωm/ns/As); report Δ_c's cosmology sensitivity. (`fit_rogers_alpha.py` machinery exists.)
- **C4 (Important) ★ — δ_c residual carried as `α_c` prior width, but `w_c` also sits in the
  structural `P_tier_p` baseline with NO marginalization.** A ~1% `w_c` error on the dominant
  clean+LLS baseline is degenerate with the cosmological amplitude (large vs DESI-era errors).
  **Fix:** marginalize `w_c` (let the δ_c residual move `P_tier_p`), or closure-test it's sub-statistical.
- **C5 (Important) — single-fidelity LF + HR-as-validation carries an UNBOUNDED resolution
  systematic** on the absolute small-scale P1D (degenerate with T₀/γ and the HCD high-k tail).
  **Fix:** measure the LF→HF P1D ratio (`hf_lf_template_convergence.py`) and correct or inflate the
  covariance — feed the error budget, not just a plot.
- Minor: M1 no metals/continuum nuisance in the likelihood (mask/down-weight affected k); M2-cosmo
  `α_c` prior centered on sim `w_c` partly fights the field rationale (verify data-dominated; prefer
  an **external LLS-incidence prior** — DESI's explicit recommendation for the worst degeneracy);
  M3-cosmo `assemble_covariance` propagates emulator error through `w_c` but the HCD term uses `α_c`
  — the two error channels (P_filt-error→w_c, Δ_c-error→α_c) need their correct multipliers.

## Cross-cutting recommendation (both reviews)
**Join `f_nhi` (CDDF) — already emitted by Head A — and ideally an external LLS-incidence prior (or
the b_DLA clustering thread) to the data vector.** Using Head A's outputs only as training targets
leaves the HCD/LLS↔cosmology degeneracy (DESI's explicit warning) unbroken; the emulator is
architecturally one step from closing it. Currently Phase-3-deferred — both reviews suggest pulling it forward.

## What is SOLID (don't redo)
NaN-safe masked loss + adversarial gradient tests; the JAX-pure dN/dX→w_c→P_obs path; Head-A exact
τ₀-invariance; x64 + static-field handling; split hygiene (LOSO by sim, fit_norm on train only); the
single-α_c likelihood form (field-standard); PRIYA bit-identity at Tier-P.
