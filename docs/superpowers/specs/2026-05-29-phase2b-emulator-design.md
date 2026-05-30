# Phase-2b — JAX/Equinox HCD P1D + CDDF emulator — design spec

**Date:** 2026-05-29
**Branch:** `phase2-emulator-jax`
**Status:** approved design, **revised post-review 2026-05-29** (CS/ML review +
Lyα-physics review + meta-review; all four open decisions resolved with the user).
**Builds on:** `docs/superpowers/specs/2026-05-17-phase2-hcd-emulator-design.md`
(architecture rationale; its §5–§14 predate the v3.1 cache and are re-specced
here), `docs/superpowers/2026-05-29-wc-dndx-poisson-coupling.md` (w_c↔dN/dX),
`docs/SESSION_HANDOVER_2026_05_22.md` (v3.1 cache schema).

> **Review outcome:** all three reviews returned APPROVE-WITH-CHANGES. The
> skeleton (max-class partition, Poisson w_c, total-P1D single covariance, PRIYA
> bit-anchor) was endorsed. The changes below are folded in. The single
> timing-critical item is the **freeze-core cache fix (§2a)**, which must land
> **before** the production cache build because that build is the expensive,
> gating compute.

**Resolved decisions (2026-05-29):**
1. **Head B re-parametrized** — drop the free `P_tier_p` and the separate clean
   filtered output; **emit the HCD delta `Δ_c` directly**; keep dense per-k
   outputs (**no PCA basis**).
2. **Freeze-core restored in the unfiltered Tier-C cache path NOW**, with a
   lower (calibrated) self-shielding threshold, and **store the uniform-rescale
   twin** alongside for the freeze-vs-uniform systematic.
3. **dN/dX prior centered on observed incidence** (PW14/DESI), width brackets the
   sim; the "keep w_c near sim default" intent moves to the `A_c` amplitude prior.
4. **τ₀-response validation reframed as reconstruction-internal** (controls for
   the shared mean-flux normalization), not a physical-ordering law.

---

## 1. Goal and scope

Build a JAX (**Equinox**) emulator of the Lyα-forest P1D split by HCD class, with
a mean-flux (τ₀) dimension, so downstream HMC can marginalise UV-background
uncertainty. The emulator-driven likelihood must be faithful — especially in the
τ₀-dependence of the HCD classes, which is the headline deliverable and the
focus of the physics review.

**In scope (Phase-2b):** the Equinox model (shared encoder + Head A + Head B);
loader; NaN-safe masked loss; the likelihood contract (total-P1D reconstruction,
single cosmic-variance covariance, nuisances); validation incl. the τ₀-response,
the freeze-vs-uniform systematic, and a likelihood-closure gate. **Plus the
cache-builder changes in §2a** (freeze-core + twin), which gate the build.

**Out of scope (Phase 3):** multi-fidelity LF/HF (HR = systematic + validation
only); 15-bin fine emulation (loader flag, untrained); heteroscedastic
uncertainty head (deferred entirely — no reserved output slots); PART-particle
per-class DLA UV study.

---

## 2. Inputs — the v3.1 cache

`hcd_analysis/_emulator_data/observables_tau0_lf.h5` (+ `_hr.h5`), schema v3.1.
Single-fidelity **LF**, `n_k = 172`.

Per-row (sim × snap × α): `params[9]`, `alpha_slope`, `target_F`, `scale`,
`z_meta`, `z_grid`, `dv_kms`, `nbins_native`, `snap_group_idx`, `sim_name`,
`alpha_idx`; `kfkms[172]`, `P_tier_p[172]`; `P_tier_c[15,172]` (unfiltered),
`P_tier_c_filtered[15,172]`, `tier_c_counts[15]`.
Per snap-block (τ₀-invariant): `f_nhi[30]`, `n_absorbers[30]`, `total_path_dX`,
`dNdX_{LLS,subDLA,DLA}`.

**Class collapse 15→4** in the loader (`merge_fine_to_classes`): clean=0; LLS=1–7;
subDLA=8–12; DLA=13–14. **τ₀ coordinate:** `τ₀ = −ln(target_F)` per row.

### 2a. Cache-builder changes REQUIRED before the production build (decision #2)

The current unfiltered Tier-C path (`_per_class_p1d_at_scale` in
`hcd_analysis/priya_p1d.py`) computes `exp(−scale·τ)` over the **whole** τ array
with **no freeze**, so it unphysically rescales partially-self-shielded gas
(τ ~ 10³–10⁵; subDLA bodies + inner damping wings) that should be Γ-insensitive.
This corrupts the per-class τ₀ response — the headline physics. Fixes:

1. **Freeze-core in the unfiltered Tier-C P1D.** Replace the uniform `exp(−scale·τ)`
   with `exp(−τ_eff)`, `τ_eff = where(τ > τ_freeze, τ, scale·τ)` — freeze
   self-shielded pixels, rescale the optically-thin rest. Reuse the existing
   `freeze_core_rescale` machinery (`hcd_analysis/tau0_rescale.py`), but adapted
   to the fake_spectra `scale` (not the legacy `alpha`).
   **Tier P is unchanged** (it is the *filtered* total and keeps PRIYA
   bit-identity — bit-identity only constrains Tier P, not the per-class tier).
2. **`τ_freeze` is NOT 1e6.** The legacy default (`TAU_FREEZE_DEFAULT=1e6` ↔
   log N_HI ≳19.3) only protects fully-saturated cores. The transition gas that
   must be frozen sits lower. **Set `τ_freeze` at the Rahmati+2013 self-shielding
   onset**, bracketed `τ_freeze ∈ [10³, 10⁶]`, and **finalise it by a pre-build
   mini-calibration** (one sim, a few snaps, 3–4 `τ_freeze` values) against the
   fake_spectra ground-truth spot-checks (§8). Provenance for the starting value:
   Rahmati+2013 self-shielding density `n_{H,SSh}` (their fitting formula, Γ- and
   z-dependent); map to per-pixel Lyα τ via the pipeline's τ↔local-N_HI relation.
   Record the chosen value + calibration in the cache attrs. (Do not hard-code an
   unsourced number; the calibration pins it.)
3. **Store the uniform-rescale twin.** Compute the unfiltered Tier-C P1D **both**
   ways (frozen + uniform) and store both blocks. The τ read + FFT are already
   paid, so the only cost is one extra `[15,172]` block/row — this gives the
   freeze-vs-uniform systematic (§8) without a re-run.
4. **Store per-class mean-F** so the shared-vs-per-class `target_F` disentangle-
   ment (the τ₀-response control, §8) is possible later without a re-run.

These are cache-builder edits (`priya_p1d.py`, `build_emulator_cache_tau0.py`),
gating the production build; the emulator consumes the result.

---

## 3. Architecture (Equinox, LF, n_k=172)

```
params(9) ⊕ z  →  encoder MLP [10 → 256 → 128 → 64]  →  latent (64)
                                              ├── Head A branch [64 → 64 → 33]   (τ₀-invariant)
                                              │       → f_nhi[30] + dN/dX[3]
                                              └── Head B [65 → 256 → 7×172]       (τ₀-dependent)
                          concat(latent, τ₀) ↗      → 4 filtered class P1D + 3 HCD deltas Δ_c
```

- **Head A** (latent only, own short branch — decision: separate branch so Head
  B's ~20× gradient volume does not pull the shared encoder off τ₀-invariance):
  `f_nhi[30]` + per-class `dN/dX[3]`. `w_c` is **derived** (§5), not emitted.
- **Head B** (`concat(latent, τ₀)`): **4 filtered class P1Ds** + **3 HCD deltas**
  `Δ_c = P_c^unfilt − P_c^filt` (LLS/subDLA/DLA; clean Δ≈0, fixed to 0 — its core
  is untouched). Dense per-k outputs (**no PCA** — it would smear the low-k vs
  high-k structure the freeze-core fix protects).
  - **`P_tier_p` is reconstructed structurally** as `Σ_c w_c·P_c^filt` (exact
    cache identity to 8e-15) — not a free output. This turns the old "total-
    reconstruction anchor" from a soft penalty into an exact identity.
  - **`P_c^unfilt` reconstructed** as `P_c^filt + Δ_c` only if needed downstream.
  - **`Δ_c` uses a sign-safe transform** (signed-log / `arcsinh`) — Δ flips sign
    at low k (the unfiltered DLA can sit below filtered there), so plain log is
    undefined.
- **Output transforms:** log-space for `f_nhi`, `dN/dX`, the filtered P1Ds;
  sign-safe for `Δ_c`. Standardise each channel by **training-split** mean/std in
  the transformed space; store stats in the checkpoint.
- **No reserved heteroscedastic slots** (Phase 3); predictive uncertainty is the
  static per-(class,k,z) error vector from k-fold LOSO (§8).

---

## 4. Targets, masks, loss

Single joint scalar loss, log/sign-space, NaN-safe.

- **NaN-safe masked loss (mandatory, pinned by a test).** ~2% of P1D bins are NaN
  above each row's native Nyquist. The `jnp.where` double-NaN-gradient trap:
  **sanitise targets to finite (`nan_to_num`) BEFORE masking**, mask after, and
  guard every masked-mean denominator with `max(n,1)`. A unit test runs one
  `jax.grad` step on a batch with NaN targets and asserts all gradients finite.
- **Head-A α-multiplicity reweighting.** `f_nhi`/`dN/dX` identical across the ~20
  α of a (sim,snap); divide their per-row loss by `n_α(sim,snap)`. (The dedicated
  Head-A branch, §3, additionally protects the encoder.)
- **Per-class sample-variance weighting = analytic `1/n_c`** (per-class P1D is
  normalised by its own count, so variance ∝ 1/n_c; the Δ_c variance is likewise
  ~1/n_c). Do **not** estimate per-bin scatter on the trained sample. Down-weights
  the noisy DLA class (~3% of sightlines) without leaking.
- **Mean-flux consistency (clean only):** `mean_F_clean ≈ exp(−τ₀)`.
- **Term-weight balancing.** log-P1D[172] vs log-dN/dX[3] have very different
  summed magnitudes (P1D would dominate ~57×). Use **per-element** (mean, not sum)
  term weights, or gradient-normalised weighting; do not hand-sweep coupled raw
  weights.

---

## 5. w_c / dN/dX  (full detail: the 2026-05-29 note)

- Head A emits **dN/dX_c(θ,z)** + `f_nhi`. **`w_c` derived** via the diagonal
  telescoping-Poisson `M₀` + calibrated `δ_c(z)` (≤2.5% vs counted w_c,
  z=2.0–5.4; `scripts/diag_crossclass_coupling.py`).
- **`δ_c(z)` must be JAX-pure** — a frozen fixed-coefficient polynomial in z
  (compile-time coeffs), so the whole `dN/dX → w_c → P_obs` path is one
  differentiable `jit`. Pin a finite-grad test on that combined path.
- **dN/dX likelihood prior = class-specific incidence forms**, centered on
  **observed incidence** (PW14 power law for DLA; **broken power law / turnover
  for LLS**; Crighton+2015-style for subDLA), width bracketing the sim default.
  Pull the exact forms + fiducial/widths from `sbird/dla_data` (no invented
  numbers). The "keep w_c near sim default" intent lives in the `A_c` P1D
  amplitude prior, **not** the incidence prior center.
- Boundary consistency: dN/dX edges == Tier-C edges (17.2/19.0/20.3). Note the
  LLS 17.2 edge has sub-percent residual Γ-dependence (not strictly τ₀-invariant).

---

## 6. Likelihood-consumption contract

**Data-space object = the TOTAL P1D**, summed over all classes (filtered DLA/subDLA
included), with a **single cosmic-variance covariance**. Per-class P1Ds are model
internals.

    P_obs(k,z) = P_tier_p  +  Σ_{c∈HCD} w_c · A_c · Δ_c          (Δ_c = P_c^unfilt − P_c^filt)
    P_tier_p   = Σ_c w_c · P_c^filt                              (structural, §3)

- **Difference form is the DEFAULT** (preserves the real forest↔HCD cross-term;
  endorsed by both reviewers). Ratio kept as an alternative toggle only.
- **The total now back-propagates through `w_c` (Head A + δ_c).** This couples
  Head A and Head B at the loss level (desirable — ties the total to incidence);
  the finite-grad test (§5) must cover this combined path.
- **Covariance:** the total's cosmic variance, with the **per-class emulator-error
  vector propagated into the total through the weights** (in quadrature). The
  error vector must flag k-bins where the DLA class is **shot-limited** (near-zero
  effective sightlines at high k) — otherwise it understates DLA uncertainty.
- **Nuisances:** `τ₀` (Head B input); `A_LLS,A_subDLA,A_DLA` (tight `N(1,·)`;
  carry the "near sim default" intent); class-specific incidence `(A_c, γ_c…)` →
  `w_c`; static emulator-error covariance from day one. No separate T₀/γ thermal
  nuisance (heat params are in the 9 sim params).
- **`A_subDLA` priors must be re-derived against the NEW (freeze-core) cache** —
  the τ_freeze change alters the subDLA filtered-tier masking fraction, shifting
  what `A_subDLA` means. Do not carry over from the uniform cache.

---

## 7. Training data and splits

- ~20 α × ~1072 LF (sim,snap) ≈ 21k rows.
- **k-fold LOSO (~8 folds)** — rotate the held-out 6–8 sims so **every sim is held
  out once**; aggregate the per-cell emulator error over folds, then smooth.
  (A single 6–8-sim holdout under-populates the class×k×z×α error covariance.)
- Within training sims, validation on whole (sim,snap) blocks (all α; no
  α-sibling leakage).
- **α-extrapolation probe in τ₀-space, not α-space.** Because
  `target_F(α,z)=exp(−α·τ_obs(z))`, the same α maps to very different τ₀ across z;
  hold out the **τ₀ edges** (not raw α edges) so the probe actually exercises the
  τ₀ range HMC queries.
- Normalisation stats from the training split only.

---

## 8. Validation battery

- **k-fold LOSO**, error stratified by class × k-band × z × α/τ₀-band.
- **Tier-P (& clean) vs PRIYA** bit-anchor cross-check.
- **fake_spectra Tier-C ground-truth spot-checks (GATING).** Re-extract per-class
  P1D at a genuinely scaled Γ at a handful of (sim,z,Γ); confirm the freeze-core
  proxy matches. Also drives the §2a `τ_freeze` mini-calibration. **Restored from
  the 2026-05-17 spec** (the 2026-05-29 draft had dropped it).
- **Freeze-vs-uniform systematic** — from the §2a twin; k/class-resolved; carried
  in the covariance.
- **τ₀-response test, REFRAMED** (decision #4): reconstruction-internal
  consistency, run with **per-class `target_F` vs shared** and **frozen vs uniform**
  controls — not a "physical class-ordering" rubber stamp.
- **Cross-class additivity** — `Σ_c w_c·Δ_c` vs directly-computed
  `(P_total^unfilt − P_tier_p)`; pin as a unit test.
- **DLA-shape clustering check** — `P_DLA` from DLA-only vs all-max-class-DLA
  sightlines across z; close Physics-M3 if within emulator error (no mixing
  matrix).
- **w_c↔dN/dX round-trip** ≤2.5%; **α/τ₀-interpolation**; physics unit tests
  (τ₀-invariance of CDDF/dN/dX as a *gradient* test; mask per-row; angular-k
  convention; Poisson sum-to-1); **likelihood closure test** — gates "Phase-2b done."

---

## 9. Module breakdown

- `hcd_analysis/emulator/data.py` — loader (v3.1 → batched
  `(params,z,τ₀,targets,masks,1/n_c weights)`; 15→4 collapse with 15-bin flag;
  k-fold LOSO splits; τ₀-space holdout; normalisation).
- `hcd_analysis/emulator/model.py` — Equinox encoder + Head A branch + Head B +
  the joint NaN-safe loss; structural `P_tier_p` reconstruction; sign-safe Δ.
- `hcd_analysis/emulator/dndx_wc.py` — `M₀` + JAX-pure `δ_c(z)` polynomial +
  class-specific incidence forms.
- `hcd_analysis/emulator/train.py` — optax loop, LR schedule, early stopping,
  seeding, checkpointing (bundle leaves + arch config + norm stats + seed), the
  static k-fold emulator-error vector.
- `hcd_analysis/emulator/likelihood.py` — §6 contract.
- `tests/test_emulator_{data,model,dndx_wc,likelihood}.py` (+ the NaN-grad,
  finite-grad, additivity, sum-to-1 tests).

---

## 10. Phasing — prioritised action list

Tags: `[cache | model | loss | likelihood | validation | infra]`, effort S/M/L,
impact H/M/L.

### A. BEFORE the production cache build (timing-critical)
1. `[cache]` Freeze-core in unfiltered Tier-C (scale-based) + **store the uniform
   twin**. *M / H.*
2. `[cache]` Pin `τ_freeze` via the pre-build mini-calibration (3–4 values, one
   sim) against fake_spectra ground-truth; record in attrs. *M / H.*
3. `[validation]` fake_spectra Tier-C ground-truth spot-check harness. *M / H.*
4. `[cache]` Store per-class mean-F (for the τ₀-response control). *S / M.*

### B. BEFORE training
5. `[loss]` NaN-safe masked loss + finite-gradient test. *S / H.*
6. `[model]` Head B re-parametrization (structural `P_tier_p`; emit sign-safe
   `Δ_c`; drop clean-filtered; dense per-k). *M / H.*
7. `[loss]` Analytic `1/n_c` weights; per-element term weights. *S / M.*
8. `[model]` Dedicated Head-A branch + τ₀-invariance gradient test. *S–M / M.*
9. `[likelihood]` JAX-pure `δ_c(z)` polynomial + class-specific dN/dX forms;
   finite-grad on dN/dX→w_c→P_obs. *M / H.*
10. `[likelihood]` Observed-incidence-centered dN/dX prior; sim intent → `A_c`. *S / H.*
11. `[infra]` Synthetic-cache loader fixture (TDD before the real cache lands);
    checkpoint bundling; LR/seed/early-stop. *S–M / M.*

### C. BEFORE inference / error budget
12. `[validation]` k-fold LOSO error vector (+ DLA high-k shot-noise flags). *M / H.*
13. `[validation]` Reframed τ₀-response gate; cross-class additivity test;
    DLA-shape clustering check; freeze-vs-uniform systematic. *S–M / M.*
14. `[likelihood]` Difference form as default; ratio as toggle; re-derive
    `A_subDLA` priors against the freeze-core cache. *S / M.*

**Deferred (Phase 3):** 15-bin fine emulation → heteroscedastic head →
multi-fidelity LF/HF → PART-particle DLA UV response → CDDF as independent vector.

---

## 11. Prerequisites (gating)

- **§2a cache-builder changes** (freeze-core + twin + per-class mean-F) — then the
  **production v3.1 τ₀ cache** build (`--fidelity lf`, ~1072×20 LF; HR ~103) →
  merge → `observables_tau0_lf.h5`. The loader + `dndx_wc` can be built/tested
  against the schema (synthetic fixture) before the cache lands; training waits.
- The cddf.npz flip (handover §5.1) is **done** (commit 858bc8d).

---

## 12. Risks / items the reviewers (incl. meta) surfaced

- **`τ_freeze` choice** — physically uncertain; mitigated by the twin + the
  ground-truth calibration. The most important number to get approximately right.
- **Δ_c sign change at low k** — handled by the sign-safe transform; verify the
  transform is smooth through zero.
- **Head A↔B loss coupling via the structural total** — desirable but must be in
  the finite-grad test.
- **DLA P1D shot-noise at high k** even at full skewers — loss `1/n_c` + the
  error-vector flagging.
- **subDLA filtered tier is threshold-defined, half-masked** — `A_subDLA` priors
  re-derived against the new cache.
- **Missing faithfulness (Tier-P-level, note only):** metals, continuum-fitting
  low-k power (also a reason to distrust raw low-k response), LF/HF resolution
  systematic (scoped as carried). RSD is handled (km/s τ field).

---

## 13. References

- `docs/superpowers/specs/2026-05-17-phase2-hcd-emulator-design.md` (rationale,
  freeze-core history, Tier-C spot-checks).
- `docs/superpowers/2026-05-29-wc-dndx-poisson-coupling.md` (w_c↔dN/dX).
- `docs/superpowers/2026-05-20-priya-p1d-consistency-check.md` (PRIYA conventions).
- `docs/SESSION_HANDOVER_2026_05_22.md` (v3.1 schema + production build).
- Diagnostics: `scripts/diag_tierc_tau0_{response,ratio}.py`,
  `diag_wc_from_dndx.py`, `diag_crossclass_coupling.py`.
- Rahmati+2013 (self-shielding `n_{H,SSh}`), Rogers+2018, Bird+2023 (PRIYA),
  Prochaska & Wolfe / Crighton+2015 (dN/dX), DESI DR1 P1D — verify numbers
  against sources before quoting.
```
