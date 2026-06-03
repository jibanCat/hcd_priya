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
2. **Production uses the plain UNIFORM rescale (UPDATED 2026-05-30).** A
   per-pixel `tau_freeze` was implemented and then shown to be a **numerical
   no-op** (`scripts/diag_tau_freeze_sensitivity.py`: self-shielded pixels have
   τ≫1 → flux already saturated → frozen ≡ scaled, bit-identical per-class P1D at
   any threshold ≥~10). So the cache stores only the uniform unfiltered Tier-C
   (no frozen twin). The `tau_freeze` knob remains in `compute_tier_c_p1d` (a
   no-op default) for a future absorber-level (Voigt-profile) **wing** treatment.
   The residual approximation — the Γ-insensitive **damping wings** are still
   scaled (a per-pixel threshold can't isolate them) — is a documented systematic
   deferred to Phase 3. PART re-extraction ground truth is infeasible now;
   threshold/recipe anchored to literature (Rahmati+2013 / Rogers+2018).
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
single cosmic-variance covariance, nuisances); validation incl. the τ₀-response
and a likelihood-closure gate. The cache builder is **uniform-rescale** (§2a;
the per-pixel freeze was proven a no-op).

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

### 2a. Tier-C recipe: uniform rescale (the freeze-core investigation, resolved)

**Motivation (the original concern).** The unfiltered Tier-C path
(`_per_class_p1d_at_scale` in `hcd_analysis/priya_p1d.py`) rescales τ by
`exp(−scale·τ)`. A uniform rescale is only rigorous for optically-thin forest
gas (`n_HI ∝ Γ⁻¹`); for self-shielded gas (`n_HI → Γ⁰`) it is unphysical. The
review (C1) worried this corrupts the per-class τ₀ response.

**Investigation.** A scale-based `tau_freeze` knob was added (freeze pixels with
native τ > τ_freeze, scale the rest) — `_per_class_p1d_at_scale(..., tau_freeze)`
and `compute_tier_c_p1d(..., tau_freeze)`, default `inf` = uniform — and a
one-snapshot sensitivity scan run (`scripts/diag_tau_freeze_sensitivity.py`,
ns0.803 z=3, τ_freeze ∈ {∞,1e5,1e4,1e3,3e2}).

**Result (2026-05-30): the per-pixel freeze is a numerical no-op.** The per-class
P1D and the τ₀-response are **bit-identical** across all thresholds. Reason: a
self-shielded pixel has `τ_Lyα ≫ 1`, so its flux `exp(−τ) ≈ 0` (saturated) and
its contrast `δF = −1` **whether frozen or scaled** — `exp(−400)` vs `exp(−312)`
both vanish. The freeze could only bite at moderate τ ~ O(1), which is the
optically-thin **forest** that *should* be scaled. So C1, while physically valid,
does **not** bite numerically; the cores are saturated-invariant.

**Decision: production uses the plain UNIFORM rescale.** The cache stores the
uniform unfiltered Tier-C only (cache **v3.3**; no frozen twin — it would be
identical). The `tau_freeze` knob stays in the code (no-op default). `Tier P` and
the filtered tier are unchanged (PRIYA bit-identity intact). Per-class ⟨F⟩
(`mean_F_by_bin`, computed under uniform) is kept for the shared-vs-per-class
normalization check.

**Residual systematic (deferred to Phase 3).** The one real piece a per-pixel
threshold *cannot* address is the **damping wings**: Γ-insensitive (set by total
N_HI) but with *low* per-pixel τ, so indistinguishable from forest pixels — they
get scaled. Capturing this needs an **absorber-level (Voigt-profile) freeze** à
la Rogers+2018, not a τ threshold. Documented; its magnitude is bounded by the
low-k part of the per-class response (the C2 ratio diagnostic) and deferred. PART
re-extraction (the self-consistent-Γ ground truth) is infeasible now, so the
treatment is anchored to literature (Rahmati+2013 self-shielding; Rogers+2018).

---

## 3. Architecture (Equinox, LF, n_k=172) — REVISED 2026-06-02 (baseline+residual P_filt)

```
params(9) ⊕ z  →  encoder MLP [10 → 256 → 128 → 64]  →  latent (64, encodes θ)
                                   ├── Head A branch [64 → 64 → 33]   (τ₀-invariant)
                                   │       → f_nhi[30] + dN/dX[3]
                                   └── Head B (P1D), two paths:
  (z, τ₀) ────────────────────────────→ BASELINE head [2 → … → 4·n_basis→4×172]  (θ-BLIND)
                                   │        → m̂(z,τ₀): the (z,τ₀)-conditional mean log-P1D (the dominant ~99.5%)
  concat(latent, τ₀) ─────────────────→ RESIDUAL head [65 → 256 → 4·n_basis + 3×172]
                                            → r̂(θ,z,τ₀): cosmology residual (σ_cosmo units) + 3 HCD Δ_c
   logP_filt = (m̂·σ_marg + μ_marg) + σ_cosmo·r̂   ⇒   exp   (θ-response = σ_cosmo·∂r̂/∂θ)
```

**The normalization redesign (why two P_filt heads).** The cosmology (θ) signal is only ~0.4% of the
per-k log-variance of P_filt; ~99.5% is the (z,τ₀) variation (which the network gets as INPUTS). A
single P1D head standardized by the *marginal* σ_k buries θ, so an MSE under-resolves it (held-out
within-cell θ-tracking corr ~0.80). Fix (Kennedy–O'Hagan structured mean / Δ-learning, **in-network so
no stored reference / no interpolation**; see `2026-06-02-normalization-fix-research.md`): a **θ-blind
baseline head** `m̂(z,τ₀)` carries the dominant (z,τ₀)-conditional mean; a **residual head** `r̂(θ,z,τ₀)`
carries the cosmology signal, trained in **conditional σ_cosmo units** (the within-(z,τ₀)-cell std) so
the loss budget is cosmology. **VALIDATED:** fold-0 within-cell θ-tracking **0.80 → 0.96**, exact
reconstruction round-trip (1e-16). Trained **joint (non-staged)** — the 3-stage schedule had a bug
(stage-1 did not train the baseline head). Absolute per-class RMS now ~6–18% (clean-limited) → a tuning
step (epochs/capacity) to reach ~1%, not structural.

- **Head A** (latent only, own short branch — decision: separate branch so Head
  B's ~20× gradient volume does not pull the shared encoder off τ₀-invariance):
  `f_nhi[30]` + per-class `dN/dX[3]`. `w_c` is **derived** (§5), not emitted.
- **Head B** (`concat(latent, τ₀)`): **4 filtered class P1Ds** + **3 HCD deltas**
  `Δ_c = P_c^unfilt − P_c^filt` (LLS/subDLA/DLA; clean Δ≈0, fixed to 0 — its core
  is untouched). Dense per-k outputs (**no PCA** — it would smear the low-k vs
  high-k structure the freeze-core fix protects).
  - **`P_tier_p` is reconstructed structurally** as `Σ_c w_c·P_c^filt` —
    structural (not a free output); exact on the cache (bit-identity to 8e-15),
    and on the trained reconstruction accurate to the emulator error in
    `P_c^filt`. This turns the old "total-reconstruction anchor" from a soft
    penalty into a structural relation.
  - **`P_c^unfilt` reconstructed** as `P_c^filt + Δ_c` only if needed downstream.
  - **`Δ_c` uses a sign-safe transform** (signed-log / `arcsinh`) — Δ flips sign
    at low k (the unfiltered DLA can sit below filtered there), so plain log is
    undefined.
- **Output transforms:** log-space for `f_nhi`, `dN/dX`, the filtered P1Ds;
  sign-safe (`arcsinh`) for `Δ_c`. Standardise by **training-split** stats in the
  transformed space, stored in the checkpoint. **P_filt uses the (z,τ₀)-CONDITIONAL
  normalization above** (baseline `μ_marg/σ_marg` + residual `σ_cosmo`), NOT a single
  marginal σ_k — that conditional scaling is what un-buries the cosmology signal.
  (f_nhi/dN/dX likely need the same per-z conditional treatment — TODO, see §3 note.)
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

## 6. Likelihood-consumption contract  (REVISED 2026-06-01 — single per-class α_c)

**Data-space object = the TOTAL P1D**, with a **single cosmic-variance covariance**.
Per-class P1Ds are model internals. The HCD sector follows the field standard
(Rogers & Bird 2018; DESI DR1; PRIYA 2025) — see
`docs/superpowers/2026-06-01-hcd-marginalization-literature.md`.

    P_obs(k,z) = P_tier_p(k,z)  +  Σ_{c∈HCD} α_c · Δ_c(k,z)      (Δ_c = P_c^unfilt − P_c^filt)
    P_tier_p   = Σ_c w_c · P_c^filt                              (structural baseline, §3)

- **One free amplitude per HCD class, `α_c`** — NOT the old redundant `w_c·A_c`
  product. `α_c` IS the **effective, residual (post-masking) incidence** of class c
  (∝ a CDDF integral over that class's N_HI range for the *survey's* unmasked
  population). **P1D constrains it directly** (the distinct per-class damping-wing
  shapes of `Δ_c` separate the classes), and **its posterior is the rough per-class
  effective dN/dX** — the headline HCD deliverable. (The earlier "P1D can't give
  dN/dX / w_c–A_c degeneracy" framing was wrong; corrected here.)
- **Prior on `α_c`:** centered on Head A's **sim-intrinsic** `w_c(dN/dX)` (cosmology-
  driven, via the M₀+δ_c map), but **wide / one-sided-positive** (PRIYA-style) so it
  can float to the data's residual value, which **legitimately differs from the
  simulation** because the data's DLA-finder completeness/purity reshape the residual
  CDDF (Rogers&Bird: "clipping changes the survey CDDF"). Head A's dN/dX is the prior
  CENTER, not a second free amplitude.
- **`α_c` is z-dependent, parametrized LOW-ORDER** (matched to DESI DR1, arXiv:2601.21432
  §5.2): NOT free-per-z-bin, NOT a single constant — either a **2-redshift-node interpolation**
  (DESI: nodes z≈2.2, 4.2, "equivalent to a single power-law in z") or a **PW14 `A_c(1+z)^{γ_c}`**
  law (~2 params/class). The *shape* z-evolution is carried by the emulated `Δ_c(k,z)` (more
  flexible than DESI's fixed Rogers `a(z),b(z)` kernel); the *abundance* z-evolution lives in
  `α_c(z)`. DESI uses broad **flat-log** priors on `f_i^HCD∈[−11,−0.03]`; PRIYA uses one-sided
  positive; our default centers on the sim `w_c(dN/dX)` — all viable (DESI flag: the **LLS
  amplitude is strongly degenerate with cosmology**, so an external LLS-incidence prior is
  advisable). The `M₀`-inverse maps the `α_c(z)` posterior → per-class effective dN/dX(z).
- **Difference form is the DEFAULT** (preserves the forest↔HCD cross-term). Ratio is
  an alternative toggle.
- **Head A↔Head B coupling** is now via the `α_c` PRIOR (centered on `w_c(dN/dX)`),
  plus the structural `P_tier_p = Σ_c w_c·P_c^filt` baseline which still uses the
  sim-intrinsic `w_c`. The finite-grad test (§5) covers `dN/dX → w_c → P_tier_p`.
- **Covariance:** total cosmic variance + the **per-class emulator-error vector
  propagated through the per-class weights** (in quadrature), with DLA high-k
  **shot-limited** bins flagged so DLA uncertainty isn't understated. **REVISED
  2026-06-02 — `C_emu` is τ₀-aware** (folds the τ₀×cosmology findings, memory
  `phase2-tau0-cosmology-interaction`): the error vector is stratified `(4,K,Zb,Tb)`,
  the fractional σ is multiplied by the predicted `P_c²` (units fix — the prior
  `Σ w_c²·σ²` dropped the `P_c²` factor), `C_emu(θ,τ₀)` is smoothly τ₀-indexed, and
  the now-state-dependent `−½logdet C` term enters the logL (a real NUTS gradient on
  τ₀). Full design + closure/SBC + DLA σ_cosmo + mean-flux prior:
  `docs/superpowers/2026-06-02-tau0-error-model-and-closure-design.md`.
- **Nuisances:** `τ₀` (Head B input); the per-class `α_c` (above); static emulator-
  error covariance from day one. No separate `A_c` amplitude (subsumed into `α_c`);
  no separate T₀/γ thermal nuisance (heat params are in the 9 sim params).
- **Half-masking is subsumed:** `α_subDLA` directly IS the effective post-masking
  sub-DLA incidence, so the old "`A_subDLA` absorbs the ~56%-masked hybrid" note is no
  longer a separate parameter — it lives in `α_subDLA`'s value/prior.
- **Baseline subtlety (resolve when wiring):** `P_tier_p` uses the *sim's* filter
  (`tau_thresh=1e6`) and *sim's* `w_c`; the data's masking differs. Default framing:
  keep `P_tier_p` as the sim-filtered baseline and let `α_c·Δ_c` add back the residual
  relative to it (matches Rogers adding residual contamination to a clipped baseline).
  Alternative: rebuild the filtered tier at the survey's masking. See the lit-review §6.
- **`Δ_c` shape caveat:** Rogers/DESI fix the per-class shape (cosmology-independent,
  z-evolving); ours **emulates** it (cosmology+τ₀-dependent). Validate that the emulated
  `Δ_c` reduces to the Rogers `1/(a e^{bk}−1)^2` kernels at fiducial cosmology (§8).

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
- **Tier-C recipe sensitivity (DONE).** `scripts/diag_tau_freeze_sensitivity.py`
  showed the per-pixel freeze is a numerical no-op → uniform rescale adopted
  (§2a). The self-consistent-Γ ground truth (PART re-extraction) is infeasible;
  the **damping-wing** systematic is deferred to Phase 3 and bounded by the
  low-k per-class response.
- **τ₀-response test, REFRAMED** (decision #4): reconstruction-internal
  consistency, run with the **per-class `target_F` vs shared** control — not a
  "physical class-ordering" rubber stamp. (No frozen-vs-uniform control — proven
  identical.)
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

### A. Cache builder (DONE — uniform-rescale v3.3)
1. `[cache]` ✅ `tau_freeze` knob added (no-op default) + one-snapshot sensitivity
   scan → per-pixel freeze proven a no-op → **uniform rescale** adopted; cache
   v3.3 stores uniform unfiltered Tier-C + filtered + per-class ⟨F⟩ (no frozen
   twin). `[model]` HEAD f7c2151 + the v3.3 simplification.
   (Damping-wing absorber-level treatment deferred to Phase 3.)

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
13. `[validation]` Reframed τ₀-response gate (per-class vs shared `target_F`
    control); cross-class additivity test; DLA-shape clustering check. *S–M / M.*
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

- **Damping-wing systematic (was the `τ_freeze` concern)** — RESOLVED for the
  cores (per-pixel freeze is a no-op; uniform adopted). The wings remain
  Γ-insensitive-but-scaled; absorber-level Voigt freeze deferred to Phase 3, its
  magnitude bounded by the low-k per-class response (C2 diagnostic).
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
