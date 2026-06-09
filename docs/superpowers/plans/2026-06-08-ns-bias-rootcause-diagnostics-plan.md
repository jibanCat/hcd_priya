# n_s-bias root-cause: diagnostic ladder + the fix/marginalize fork — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to execute this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.
> This is a **diagnostic/investigation plan**, not a feature build: each task is a *measurement with a
> decision gate*, ordered cheap-and-decisive first. Do NOT skip a gate to start a fix.

**Goal:** Pin the root cause of the emulator's coherent **−0.65σ n_s under-prediction**
(`figures/analysis/04_emulator/emu_bias_allfolds.{png,txt}`, t=−5.84, 50/60 held-out sims negative)
with **forward-only** diagnostics (no retrain), then choose between **fix-at-source** vs
**marginalize-as-C_emu** on evidence — and separately install the real-fit coverage guard for n_s>0.995.

**Architecture / approach:** The emulator is a θ-blind `BaselineHead(z,τ₀)→(4,n_k)` conditional mean +
a τ₀-dependent `HeadB(concat(latent,τ₀))` σ_cosmo-whitened residual `r̂`, so **all** cosmology flows
through `r̂` (`hcd_analysis/emulator/{model.py,predict.py}`). A_p is a low-k amplitude boost; **n_s is a
k-tilt** (response energy peaks at k≈0.029, high-k). The all-folds bias is A_p≈0, n_s≈−0.65σ — i.e. the
emulator under-resolves the **tilt direction** specifically.

**KEY FACT confirmed 2026-06-08 (PI dialogue):** the emulator's 172-point k-grid is **LINEARLY spaced**
(0.0004→0.069 s/km, constant Δk=0.0004; linear-diff CV=0.000, verified from
`hcd_analysis/_emulator_data/observables_tau0_lf.h5::kfkms`). A power law is linear in log-log, so the
tilt's natural coordinate is the slope in **log k**; on a linear-k per-bin SVD target the tilt has no
protected DOF (low-k k≈0.005 pivot gets ~12 bins; the slope is smeared across ~160 high-k bins). We
already take log P (y-axis) — which is why A_p is fine — but NOT a smooth-log-k-coefficient target
(the missing LaCE move). This is concrete evidence for #3 and sharpens the D7 fix.

**`coherent_debias_term` blind spot (confirmed in code, `model.py:342`):** it penalizes the per-cell
θ-MEAN `⟨r̂−t_p_resid⟩_θ`, but `⟨t_p_resid⟩_θ=0` by construction, so a response ATTENUATION `r̂≈β·t`
with β<1 has `⟨r̂−t⟩_θ=0` and is invisible to it. The n_s bias IS such an attenuation (D1 measures β).
This is why production `w_coh=80` ran yet left −0.65σ. The #4 fix must constrain the response SLOPE
(cov of r̂ with θ), not the θ-mean.

**Tech stack:** JAX/Equinox emulator + numpyro likelihood. Env for every command:
`PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" /home/mfho/.conda/envs/emu-jax/bin/python3`

---

## 0. The convergent verdict (4-lens review, 2026-06-08)

Reference findings: `docs/superpowers/2026-06-08-ref-{priya-orig,priya-extended,lace-norm,code-chains}.md`.
Specialist reports: `docs/superpowers/onboarding/2026-06-08-rootcause-{bayesian,cs,cosmology,lya}.md`.

**Priority ranking of the user's 4 feedback points as the ROOT CAUSE of the −0.65σ:**

| feedback point | Bayesian | CS | Cosmology | Lyα | consensus |
|---|---|---|---|---|---|
| **#3 normalization / tilt** | #1 (med) | #2 (med) | **#1 (high)** | **#1 (high)** | **#1 — the proximate cause** |
| **#4 conditional-mean loss** | #2 (med) | **#1 (high)** | #4 (low) | #2 (med) | **co-#1 — same gap, loss-side** |
| **#1 τ₀–n_s heads** | #3 (med) | #3 (med) | #2 (med) | #3 (low) | **CHANNEL/amplifier, not cause** |
| **#2 extended prior n_s>1.0** | #4 (low) | #4 (low) | #3 (low) | #4 (low) | **real-fit hazard, NOT this bias** |

**#3 and #4 are one mechanism seen from two sides** (target/basis vs loss): the σ_cosmo-whitened per-k
MSE gives the **coherent k-tilt no protected degree of freedom**. Specifics all four lenses converged on:
- `t_p_resid = (logP − cell_mean)/σ_cosmo` divides each k by its own within-cell std, and **σ_cosmo RISES
  with k** (clean class ≈0.067→0.110 across the band) → the per-k whitening **down-weights the residual
  MSE exactly in the high-k band where the n_s tilt lives**, relative to the low-k A_p band. → **A_p
  protected, tilt diluted** = exactly the observed A_p≈0 / n_s≈−0.65σ split.
- `coherent_debias_term` (run in production at `w_coh=80`) penalizes the per-cell **θ-mean** of the
  residual, which is `0` by construction for a slope error and is **flat-in-k** → it is **blind to a
  coherent k-tilt** (a tilt pivots in-band, ~zero k-mean). This is *why w_coh=80 did not remove the bias.*
- The shared 24-mode SVD residual basis is **non-orthonormal** (‖BBᵀ−I‖ reported 2.7–7.9, cond up to 163)
  → high-k mis-fit **leaks into the low-k/tilt direction**. n_s *is* the k-slope.
- **LaCE (2305.19064)**: the *target basis* is the dominant accuracy lever (linear-k→log-k coefficient
  target: 10%→1.4%) and gives the slope a **named, separately-weighted DOF**. Our redesign adopted
  log-space + inverse-variance + θ-blind baseline + no-stored-grid (3 of 4 LaCE moves) but **NOT the
  smooth-log-k-coefficient target** — the one piece that protects the tilt.
- **Lyα localization:** the under-tilt is **z≈2.0–2.4-localized** (slope in ln k = −0.00138 there vs
  ~−0.0005 at z>4), carried by the **residual head** not the baseline — and z≈2.0–2.4 is the
  DESI-leg resolution-*converged* regime, so **MF will not fix it** (it reshapes high-k/KS). This is an
  LF-emulator tilt problem.

**Two cross-cutting dissents the review insists on (act on both):**

1. **The "C1 resolved" claim is WRONG — the user (feedback #2) is right.** Only **2 of 60** PRIYA design
   points have n_s>1.0 (1.0188, 1.0396), gap-separated (0.9979→1.0188→1.0396) and **pinned to the
   high-α_q corner** — a uniform LHS would place ~12. The original PRIYA LHS capped n_P at **0.995**
   (Bird+2023 Table 2); the 0.995→1.05 extension (Fernandez+2024) added 12 LF sims jointly with α_q→3.0
   to reach the 2σ posterior. `data.PARAM_LIMITS == emulator_params.json` proves only the *box* matches,
   **not** that (0.995, 1.05] is *populated*. **Downgrade C1 to "box matches; high-n_s edge is a sparse
   HeII-coupled extrapolation."** This matters for the real fit because eBOSS n_P=1.009 lands in that
   sparse band. *(Minor correction to the user's mechanism: Bayesian-opt was used in the ORIGINAL 48, not
   the 12-sim extension — but the coverage-sparsity core stands.)*

2. **The prior "finite-60-sim LOSO floor → just marginalize as zero-mean C_emu" framing is challenged by
   all four.** The bias is **structured and at-least-partly fixable**, not an irreducible floor:
   - Bayesian: the LOSO folds are **contiguous sorted-n_s blocks** (per-fold mean n_s_unit
     0.06,0.18,…,0.80); per-fold bias sign tracks (fold_mean − global_mean) almost perfectly (fold 1,
     below the mean, is the *only* positive fold). That is **shrink-to-fold-train-mean — a partition
     artifact**, not (only) a data floor.
   - Cosmology: the bias **zero-crosses at n_s≈0.776, below the entire design range** (min 0.803) — a
     directional tilt under-prediction, not generic regression-to-mean (which would zero-cross at the
     design mean 0.904).
   - **A structured/fixable bias marginalized as zero-mean noise UNDER-COVERS n_s.** → **Measure the
     slope-attenuation β and re-fold first; do NOT build a zero-mean correlated C_emu for this mode until
     the diagnostics below establish it is ensemble-symmetric rather than a structured tilt bias.**

**What the review already ran (in-flight; treat as starting points, re-establish as committed artifacts):**
- `scripts/diag_lya_ns_tau0_split.py` (on disk) — τ₀-split of the n_s-equivalent mis-tilt across 8 folds.
  Discriminator ⟨c⟩_low_τ₀ − ⟨c⟩_high_τ₀ = **+0.0335 (wrong sign for #1, z-driven)** → **#1 refuted as the
  driver.** Keep & commit.
- `/tmp/resp_shapes.npz` (transient) — empirical clean-class whitened A_p vs n_s response shapes; the basis
  for the projection tests. **Re-generate to a committed path in D1.**
- Cosmology forward prior-cut test on `emu_bias_allfolds.txt`: n_s≤{1.0, 0.995, 0.965} → bias stays
  −0.646/−0.633/−0.634σ; only n_s≤0.90 (dropping 33/60 sims) drops it to −0.24σ. → **#2 not the cause.**
- `corr(n_s, τ₀)` from 7 PRIYA cobaya chains at
  `/home/mfho/student_projects/InferenceLyaData/Chains/{fps-only,fps-meant}/*.1.txt` = **−0.56…−0.78
  (median −0.66)** → the degeneracy #1 needs is real and strong, but it is a *channel*.

---

## ✅ RESOLVED (2026-06-08) — the −0.65σ is low-z KS data + an LF-resolution systematic

The diagnostic ladder ran to a conclusion. Headline: the −0.65σ is **NOT** a normalization/loss (#3/#4),
coverage (#2), τ₀-channel (#1), tilt-attenuation (D1: β≈0.98), or C_emu-sizing problem. The z-attribution
(`scripts/diag_nsbias_z_attribution.py`) localized **86% to the KS leg at z=2.0+2.2; DESI is +0.04σ (clean)**.
Two converging causes, both low-z/small-scale:
1. **Closure bias = low-z KS data quality.** KS z=2.0/2.2 are real Karaçaylı-2021 rows but are exactly the
   low-z P1D the PI's KODIAQ-SQUAD paper excludes (z<2.8, DLA-finder incompleteness). Dropping KS z<2.8
   (`load_ks_leg z_lo=2.8`) collapses the bias **−0.78σ → +0.05σ** (`scripts/diag_nsbias_kscut_scan.py`).
   Within KS it is broadband over k=0.02–0.04 (NOT k>0.04-specific).
2. **Real-fit systematic = LF resolution (the PI's MF hypothesis, CONFIRMED).**
   `scripts/diag_lf_vs_hr_highk.py` (vs `observables_tau0_hr.h5`, 6 overlap sims, exact τ₀-rung match): LF is
   **−6% power-deficient at low-z high-k** (z=2.0 −7.6%), tilt-shaped (+4% low-k → −6% high-k), coherent
   6/6 sims, matching PRIYA ~7% convergence. The LF-only emulator carries this vs reality(HR) → biases real
   n_s low. **Distinct from the closure bias** (LF-vs-LF, fixed by the z-cut); fixed by MF wiring.

**ACTIONS:** (a) make `load_ks_leg z_lo=2.8` the default (the PI's published cut) → closure validates clean;
(b) proceed with the planned MF wiring for the real-fit high-k systematic, then re-run
`diag_emu_bias_allfolds.py` through the MF forward to quantify residual at z=2.8–3.4 high-k. DESI primary
cosmology was never biased. The diagnostic-ladder tasks below (D2–D7) are now SUPERSEDED by this resolution;
kept for provenance.

---

## ⚠️ D1 RESULT (2026-06-08) — overturned the tilt-attenuation hypothesis (step on the way to the resolution above)

`scripts/diag_ns_slope_attenuation.py` (run; `figures/analysis/04_emulator/ns_slope_attenuation.{npz,png,txt}`,
14,820 held-out rows, 8 folds, 260 (z,τ₀) cells, sanity checks all passed — emu-vs-truth n_s-slope shape
corr **+0.997**, in-sample P_filt reproduced to 0.5%):

> **β ≈ 0.98** (interior 0.978; divide-free regression slope 0.980), **FLAT in τ₀** (low-τ₀ 1.006 vs
> high-τ₀ 0.959 — if anything the *opposite* of #1), **NOT high-k concentrated** (high-k tilt band is the
> *least* attenuated, β≈0.985), **no edge collapse** (n_s_unit>0.9 → 0.936, not a cliff). The
> (1−β)≈0.02 attenuation predicts only **≲0.06σ** of n_s bias — **10–30× too weak** to be the −0.65σ.

**Interpretation (the redirect):** the emulator tracks the n_s *response* (the tilt slope) **faithfully** —
so the −0.65σ is **NOT a response attenuation**, and therefore NOT the tilt-budget / linear-k-target
mechanism #3/#4 were built around (as response-attenuation). β cleanly **refutes that specific mechanism.**
What D1 does NOT measure: the bias is the Fisher projection of a **coherent held-out prediction RESIDUAL**
`ΔP = P_truth − P_emu` (a misprediction at the true cosmology), which lands on n_s because of its k/z shape.
D1's note: the largest β deviations are at the **low-k pivot**, not the tilt — consistent with a low-k/
structural coherent residual, not a response defect. **So #3 may still matter as the cause of the residual
SHAPE, and #4 likely reasserts as a GENERALIZATION GAP** (`coherent_debias_term` zeroes the coherent error
*in-sample* but it is never enforced on held-out cosmologies). The gate's **"β≈1 ⇒ re-weight to D5/D6"**
branch fires.

**Re-prioritized next steps:** **D2 first** (characterize the residual: in-sample vs held-out coherent
error → is it a generalization gap? baseline vs residual head? its k/z shape → tilt-like?), then **D6**
(the contiguous-block-partition artifact test — now the sharpest remaining structural lead). D7's
"fix-at-source via tilt-aware target" is **demoted** (the response is already faithful); D7-BranchB
(marginalize the coherent residual as correlated C_emu) is **promoted IF** D6 shows it is a genuine
generalization residual rather than a block artifact.

---

## Diagnostic ladder (forward-only, no retrain) — ordered most-informative-and-cheapest first

### Task D1: Slope-attenuation β diagnostic — the single script that scores all four hypotheses

**Why first:** CS/Cosmology/Bayesian independently proposed a variant of this; it is ~minutes on the
existing per-fold emulators and answers #1–#4 *directionally* in one run.

**Files:**
- Create: `scripts/diag_ns_slope_attenuation.py`
- Reuse: the per-fold emulator load + `jax.jacrev`/`jacfwd` loop in `scripts/diag_emu_bias_allfolds.py`
  (it already loads `final_fold{0..7}` and the 60 honestly-held-out sims and has the θ-Jacobian wired).
- Cache truth: same source `diag_emu_bias_allfolds.py` uses for the held-out truth P_filt.
- Output: `figures/analysis/04_emulator/ns_slope_attenuation.{npz,png,txt}`

- [ ] **Step 1: Compute the per-sim n_s attenuation factor.**
  For each held-out sim and z, form
  `β_ns(z,k) = ⟨∂ logP̂/∂n_s⟩ / ⟨∂ logP_true/∂n_s⟩`, where the *truth* slope `∂ logP_true/∂n_s` is a
  **within-(z,τ₀)-cell regression** of `logP_filt` on `n_s_unit` (the cell already fixes z,τ₀), and `∂
  logP̂/∂n_s` is the emulator's autodiff Jacobian (reuse the jacrev from `diag_emu_bias_allfolds.py`).
  β<1 coherently ⇒ **tilt shrinkage** (the #3/#4 signature).
- [ ] **Step 2: Stratify the SAME β three ways** (one script, three group-bys):
  - β vs `n_s_unit` (tests **#2**: does β collapse *only* above 0.9?),
  - β vs τ₀-rung (tests **#1**: is β materially smaller at low τ₀?),
  - β's k-shape / projection onto the whitened n_s-response vector (tests **#3**: is the deficit in the
    high-k tilt band?).
- [ ] **Step 3: Run it.**
  `… python3 scripts/diag_ns_slope_attenuation.py`
  **Expected output:** pooled β, β-vs-{n_s_unit, τ₀-rung, k}, and a one-line bias prediction
  `bias ≈ (1−β)·projection` to check it reproduces ≈−0.65σ.
- [ ] **Step 4: Record the gate verdict** in the output `.txt` and in this plan:
  - **β<1 coherently across the well-sampled interior** ⇒ structural tilt shrinkage ⇒ **#3/#4 own it**
    → proceed to D2/D3; this is *bias-to-fix*, not noise-to-marginalize.
  - **β≈1** (no attenuation) ⇒ the bias is not response-attenuation → demote #3/#4, re-weight toward
    D5/D6 (coverage / block-artifact).
  - **β collapses only at n_s_unit>0.9** ⇒ #2 (extrapolation) is the driver after all → jump to D5.
  - **β materially smaller at low τ₀** ⇒ #1 is a live multiplier → keep D4.
- [ ] **Step 5: Commit.** `git add scripts/diag_ns_slope_attenuation.py figures/analysis/04_emulator/ns_slope_attenuation.* && git commit -m "diag(phase-c): n_s slope-attenuation beta — scores feedback #1-#4 forward-only"`

### Task D2: Log-k coefficient decomposition + the coherent-debias blind-spot check (#3 + #4)

**Files:**
- Create: `scripts/diag_ns_logk_decomp.py`
- Output: `figures/analysis/04_emulator/ns_logk_decomp.{npz,png,txt}`

- [ ] **Step 1: Project held-out residual onto a low-order log-k basis.**
  For each held-out sim and (class, z), regress the prediction residual `(logP̂ − logP_true)` onto
  `{1, ln k, (ln k)²}`; report the **coherent mean of the `ln k` (slope) coefficient** over all 60 sims
  vs the constant (amplitude) coefficient.
  **Expected (per the review):** coherently **negative slope coefficient**, ~zero constant ⇒ the bias is
  a **tilt mode**, amplitude-neutral — matching A_p≈0/n_s<0.
- [ ] **Step 2: Locate it in baseline vs residual head (#4).**
  Recompute the **held-out** cell-mean (including the held-out sim) and split the slope error into
  `BaselineHead` (`m̂ − m_cell_heldout`) vs `HeadB` (`r̂`) contributions. (Generalizes the becker-row
  decomposition in `scripts/embias_arch_localize.py`/`embias_arch_tau0regime.py`, which already found
  base_err≈flat, resid_err carries the tilt.)
  **Expected:** baseline slope-error ≈0, slope bias in `HeadB`'s residual ⇒ a *loss/target* problem, not
  a broken conditional mean.
- [ ] **Step 3: Show the production `coherent_debias_term` is blind to it.**
  Compute the per-(z,τ₀)-cell mean `⟨r̂ − t_p_resid⟩_θ` (the exact quantity `coherent_debias_term`
  squares) and fit **its slope vs ln k per z-band**. Then re-evaluate that term under a *slope-weighted*
  (L1-Legendre) k-weight on the same residuals.
  **Expected:** non-zero **slope** with ~zero **k-mean** at z≈2.0–2.4 ⇒ confirms the flat `w_coh=80` term
  cannot remove the tilt; the fix is a **tilt-aware (not flat-in-k) coherent term**.
- [ ] **Step 4: Run + commit.**
  `… python3 scripts/diag_ns_logk_decomp.py` ; then
  `git add scripts/diag_ns_logk_decomp.py figures/analysis/04_emulator/ns_logk_decomp.* && git commit -m "diag(phase-c): log-k tilt decomposition — baseline-vs-residual + coherent-term blind spot (#3,#4)"`

### Task D3: σ_cosmo whitening counterfactual (#3) — the "is the normalization the channel?" test

**Files:**
- Create: `scripts/diag_ns_whitening_counterfactual.py`
- Reuse: the all-folds Fisher MAP-shift in `scripts/diag_emu_bias_allfolds.py`;
  `fit_baseline_residual_norm` in `hcd_analysis/emulator/data.py` (the σ_cosmo definition, z-collapsed).
- Output: `figures/analysis/04_emulator/ns_whitening_counterfactual.{npz,txt}`

- [ ] **Step 1: Flatten-σ_cosmo counterfactual.** Re-weight the *existing* per-sim held-out residual
  errors by `σ_cosmo(k)/median(σ_cosmo)` (i.e. undo the rising-with-k down-weighting) and re-run the
  all-folds Fisher MAP-shift. **If the n_s bias shrinks ⇒ #3 confirmed** as the loss-budget channel.
- [ ] **Step 2: z-resolved σ_cosmo counterfactual (Lyα's most-informative test).** Recompute σ_cosmo in
  **3 z-bins** instead of one z-collapsed constant, re-whiten the held-out residuals, re-fit the per-z
  tilt slope. **If the z≈2.0–2.4 tilt deficit shrinks ⇒ the z-collapsed whitening is mis-budgeting the
  low-z high-k tilt** ⇒ a z-resolved σ_cosmo is a candidate source fix.
- [ ] **Step 3: Gate.** Record in `.txt`: how much of −0.65σ each counterfactual removes. >~50% removal
  from either ⇒ **bias-to-fix at the normalization/loss layer** (informs the D7 fork).
- [ ] **Step 4: Commit.** `git add scripts/diag_ns_whitening_counterfactual.py figures/analysis/04_emulator/ns_whitening_counterfactual.* && git commit -m "diag(phase-c): sigma_cosmo flatten + z-resolved whitening counterfactuals (#3)"`

### Task D4: τ₀-resolved tilt split (#1) — confirm channel-not-cause, consolidate the Lyα result

**Files:**
- Modify/keep: `scripts/diag_lya_ns_tau0_split.py` (already run: discriminator **+0.0335**, wrong sign,
  z-driven). Commit it.
- Optionally extend: `scripts/embias_arch_tau0regime.py` to bin the **high-k (k>0.02) tilt-band** coherent
  error by τ₀-rung (the existing version is low-k/amplitude-flavored).

- [ ] **Step 1: Commit the existing split test** and record its verdict (#1 refuted as driver:
  the mis-tilt tracks z, not τ₀; Kim τ rises with z so the low-τ₀ bin *is* the z≈2.4 bin).
- [ ] **Step 2: (Only if D1 found β smaller at low τ₀)** add the high-k τ₀-binned tilt projection onto
  the n_s-response; if τ₀-flat ⇒ #1 is a channel, not the cause ⇒ no τ₀-resolved C_emu needed.
- [ ] **Step 3: Commit.** `git add scripts/diag_lya_ns_tau0_split.py [scripts/embias_arch_tau0regime.py] && git commit -m "diag(phase-c): tau0-resolved tilt split — #1 is a channel not the n_s-bias cause"`

### Task D5: n_s>0.995 coverage audit + real-fit edge guard (#2) — independent of the interior bias

**Files:**
- Create: `scripts/audit_ns_extension_coverage.py`
- Read: `/home/mfho/lya_emulator_full/kodiaq_2_2_4_6-48-48/emulator_params.json` (the 60 `sample_params`);
  `hcd_analysis/emulator/data.py` (`PARAM_LIMITS`).
- Output: `figures/analysis/04_emulator/ns_extension_coverage.{png,txt}`

- [ ] **Step 1: Quantify the sparsity.** For the design points with n_s>0.99, compute nearest-neighbour
  spacing / convex-hull volume over the **other 8 params** vs a random 12-subset, and confirm they pin to
  the high-α_q corner. **Expected:** confirms a sparse, HeII-coupled wedge (2/60 above 1.0).
- [ ] **Step 2: Confirm the prior-cut null** (re-establish the cosmology agent's forward test as a
  committed artifact): re-pool `emu_bias_allfolds.txt` for n_s≤0.995 and n_s<0.97 → bias stays ≈−0.63σ ⇒
  #2 is **not** the interior-bias driver.
- [ ] **Step 3: Install the real-fit guard.** **PI DECISION (2026-06-08): keep the domain [0.8, 1.05]
  (no hard cap)** and carry an explicit **step-inflated C_emu for n_s>0.995** (the sparse-extension
  uncertainty), so the posterior is not railed against a wall if the true n_s is high (eBOSS n_P=1.009).
  Implement as a multiplicative C_emu inflation gated on the sampled n_s crossing 0.995; calibrate the
  step from the n_s>0.995 LOO error in Step 1.
- [ ] **Step 4: Fix the documentation.** Downgrade the "C1 resolved" wording in
  `docs/superpowers/onboarding/2026-06-06-onboarding-meta.md` (and the spec ledger) to
  "box matches design JSON; **the (0.995,1.05] n_s edge is a sparse HeII-coupled extrapolation**, ~2/60
  training points — treat as untrusted in the real fit." Cite Bird+2023 Table 2 + Fernandez+2024 §2.
- [ ] **Step 5: Commit.** `git add scripts/audit_ns_extension_coverage.py figures/analysis/04_emulator/ns_extension_coverage.* docs/superpowers/onboarding/2026-06-06-onboarding-meta.md && git commit -m "diag(phase-c): n_s>0.995 sparse-extension audit + C1 correction + real-fit edge guard (#2)"`

### Task D6: Random/stratified re-fold — the decisive bias-vs-block-artifact arbiter (compute-gated)

**Why last among diagnostics:** Bayesian's #1-information test, but it requires **retraining 8 LF
emulators** with non-contiguous fold membership (the current `final_fold{0..7}` are contiguous sorted-n_s
blocks). Not forward-only → **profile and get PI sign-off on compute first** (budget ~4000 CPU-h,
memory `cavestru0-compute-budget`).

**Files:**
- Modify: the fold-assignment in `scripts/run_loso_sweep.py` (add a `--fold-scheme {block,random,stratified}`
  switch; stratified = round-robin by sorted n_s so every fold spans the full n_s range).
- Reuse: `scripts/diag_emu_bias_allfolds.py` to re-measure the bias on the new emulators.
- Output: `figures/analysis/04_emulator/emu_bias_refold_{random,stratified}.{png,txt}`

- [ ] **Step 1: Profile one fold retrain** and report wall-clock × 8 vs budget before launching.
- [ ] **Step 2: Retrain 8 stratified-fold emulators**, then re-run `diag_emu_bias_allfolds.py` against them.
- [ ] **Step 3: Gate (the arbiter).**
  - Bias drops to **≤0.2σ** ⇒ the −0.65σ was largely the **contiguous-block confound** → the genuine
    residual is small; marginalize the remainder (small correlated C_emu) and move on.
  - Bias persists at **~0.4σ+** ⇒ a **genuine structural tilt bias** independent of the partition ⇒
    commit to the source fix (D7).
- [ ] **Step 4: Commit** the re-fold scheme + results.

---

## Task D7: The fork — fix-at-source vs marginalize (decide AFTER D1–D3, gated by D6)

**Do not start D7 until D1–D3 are in and D6's gate is read.** Choose on the evidence:

- [ ] **Branch A — BIAS-TO-FIX (expected, if β<1 + whitening counterfactual removes the bulk + re-fold
  persists).** A *targeted retrain*, validated on ONE fold before any full re-run:
  1. **z-resolved σ_cosmo** whitening (D3 Step 2 result) — replace the z-collapsed per-(c,k) constant.
  2. **Tilt-aware coherent term** — replace the flat-in-k `coherent_debias_term` with a slope-weighted
     (Legendre-L1) penalty so the per-cell *tilt* is constrained, not just the k-mean.
  3. **(Strongest lever, LaCE)** switch the residual target to a **smooth-log-k coefficient**
     parameterization so the slope is a named, separately-weighted DOF (App. A of 2305.19064).
  4. **(Near-retrain-free pre-check, do first)** re-orthonormalize the trained `p_filt_basis` (re-SVD /
     Gram-Schmidt), re-fit ONLY the `HeadB` output/coeff layer (minutes), re-run
     `diag_emu_bias_allfolds.py` — quantifies how much of −0.65σ is the band-coupling channel alone.
  - **Re-measure the bias post-fix AND post-MF before any cert.** Invalidates prior certs → re-run the
    golden guard (`tests/test_legb_golden.py`) and the closure.
- [ ] **Branch B — NOISE-TO-MARGINALIZE (only if β≈1 and D6 collapses the bias to a small ensemble-symmetric
  residual).** Build the planned **correlated-in-k/z C_emu** (`plan §0b-2`), **z-resolved/edge-weighted**
  (the bias is z-localized at z≈2.0–2.3, NOT a global rank-1 mode), with the **σ-inflation measured
  OUT-of-sample** (not the in-sample whitening 1.28→0.93 number, which is mild-circular). Gate the
  inflation at ~1.15× via ρ = alignment of the bias mode with the n_s response.
  **Do NOT marginalize a structured tilt bias as zero-mean C_emu — it under-covers n_s.**

---

## PI decision (RESOLVED 2026-06-08)

**The real-fit n_s domain:** **KEEP [0.8, 1.05]** — do NOT hard-cap at 0.995. Carry the sparse-extension
uncertainty as a **step-inflated C_emu above n_s=0.995** (D5 Step 3), so the posterior can reach a
genuinely high n_s (eBOSS n_P=1.009 lands there) without railing against a wall, while the inflated C_emu
honestly down-weights the extrapolated upper tail. (Independent of the interior −0.65σ root cause.)

---

## Self-review

- **Spec coverage:** all four feedback points have a dedicated forward-only test with a decision gate
  (#3→D1/D2/D3, #4→D2, #1→D1/D4, #2→D5); the prior "marginalize" framing is gated on D1+D6; the C1
  correction (#2) is an explicit doc task (D5 Step 4); the fix/marginalize choice is the explicit D7 fork.
- **No placeholders:** each task names exact create/modify files, the existing script to copy the
  cache/jacrev wiring from, the run command (with the env line), the expected output, and the gate.
- **Decisiveness ordering:** D1 (minutes, scores all four) → D2/D3 (forward-only, localize #3/#4) → D4
  (#1 channel) → D5 (#2 coverage, doc + guard) → D6 (compute-gated arbiter) → D7 (fork). No fix begins
  before its gate.
- **Consistency:** β (D1), the log-k slope coefficient (D2), and the whitening counterfactual (D3) are
  three independent views of the *same* tilt-attenuation claim — if they disagree, that disagreement is
  itself the signal and must be resolved before D7.
