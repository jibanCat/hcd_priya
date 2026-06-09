# Root-cause review (CS / ML / JAX architecture lens) — 2026-06-08

**Object under review:** the honest all-folds LOSO measurement that the emulator coherently
UNDER-predicts n_s by −0.65σ (t=−5.84, 50/60 held-out sims negative; A_p ~0). Charge: assess the
user's 4 candidate root causes through the ML-architecture + training lens, with one cheap (forward-only,
no-retrain) test each, then rank, dissent, and order next actions.

I reproduced the bias table structure, the design-point coverage, the tau0-regime diagnostic, the norm
stats, and the trained basis conditioning (numbers below). My lens-specific bottom line up front:

> The strongest ML-mechanism finding this session is **structural, not data-side**: the production loss
> has **NO term that constrains the n_s-SLOPE of the residual head**. The two "anti-bias" terms
> (`coherent_debias_term`, the U-shaped `edge_emphasis_k_weight`) both operate on the WRONG quantity —
> the per-cell θ-MEAN and the per-k MSE respectively — and a shrink-to-centre tilt error is *invisible*
> to both **by construction**. So feedback #4 (no loss on the right conditional object) is the lever I
> rank highest, **and it generalizes the user's #4 from "the (z,τ₀) mean" to "the θ-RESPONSE conditioned
> on (z,τ₀)".** Feedbacks #1 and #2 are real and contribute, but the data/code show they are not the
> primary driver of the −0.65σ.

---

## The numbers I reproduced (so the assessments below are grounded)

1. **Fold = contiguous n_s slice.** Folds 0..7 are sorted by n_s (fold 0 = lowest, fold 7 = highest;
   each n=7–8). The per-fold mean n_s bias is `[-0.47, +0.47, -0.52, -0.84, -1.04, -1.55, -0.75, -0.63]`.
   It is **NOT monotone in n_s** and **worst in the MIDDLE** (fold 5, mid-n_s, the DENSEST design region).
2. **`corr(bNs, ns_u) = -0.349`** (mild shrink-to-centre) BUT **`corr(bNs, |ns_u-0.5|) = +0.424`** — the
   bias is worst at the *extremes only weakly*; mid-n_s (ns_u 0.3–0.6) has the most negative mean (-0.975)
   vs ns_u>0.6 (-0.76) and ns_u<0.3 (-0.19). A pure high-n_s-sparsity story (#2) predicts worst at fold 7;
   fold 7 is only -0.63. A pure shrink-to-centre predicts symmetric pull-to-mean; the actual minimum sits
   at the mean, not the edges. **Both #2 and naive shrink-to-centre are under-determined by this shape.**
3. **`corr(bNs, ap_u) = +0.214`; mean bNs at low-Ap (ap_u<0.4) = -0.894 vs high-Ap (ap_u>0.6) = -0.391.**
   The n_s bias is ~2× worse in the low-amplitude regime. This is the *one* cross-parameter coupling in
   the table, and it is the partial fingerprint feedback #1 predicts (amplitude regime ↔ n_s).
4. **n_s design coverage:** only **2 of 60** sims have n_s>1.0 (vs ~12 under uniform LHS over [0.8,1.05]);
   top decile (ns_u>0.9) has **1** sim, with a visible gap (Δns jumps to 0.0208 at the top two points
   0.9979→1.0188→1.0396). **Confirms #2's structural claim** that n_s>1.0 is a sparse BO-style extension,
   not space-filling — but see #2 assessment for why this is not where the −0.65σ lives.
5. **tau0-regime (embias_arch_tau0regime.npz):** clean low-k coherent error is monotone WORSE at LOW τ₀
   (`bandmean` -0.275% at α=0.69 rung → -0.076% at α=1.30). The carrier is `resid_err`, not `base_err`
   (baseline near-exact). **Confirms #1's asymmetry claim — at the amplitude level, on the residual head.**
6. **sig_cosmo is k-dependent** (clean class 0.063→0.123, rising at high-k), median sig_cosmo/sig_marg ≈
   **0.076** (cosmology is ~7.6% of the marginal log-variance). The residual target whitens **each k
   independently** → the loss has no coherent-slope-across-k DOF (the LaCE critique, #3).
7. **Trained residual basis `p_filt_basis` (24,172): ‖BBᵀ−I‖_F = 7.9, cond(BBᵀ) = 163** on final_fold0
   (worse than the 2.7 cited for another fold). The global 24-mode basis is a genuine high-k↔low-k/tilt
   coupling channel.

---

## Feedback #4 — NO loss on the conditional mean (and, generalized, on the conditional θ-RESPONSE)

**Likelihood as root cause: HIGH.** This is the lens-specific headline.

The user's literal worry ("no loss on the mean prediction for fixed (τ₀,z)") is **REFUTED for the mean
itself**: `joint_loss` (model.py L437) has an explicit `p_base` term = masked MSE of `BaselineHead` vs
`t_p_base` = the σ_marg-standardized (z,τ₀) cell-mean, AND there is a dedicated 8000-epoch
`_prefit_baseline` (train.py L165) that trains the baseline alone on a ≈306-cell (z,τ₀) table, AND the
production recipe FREEZES the baseline (train.py L370, `freeze_baseline=did_prefit`) so the joint loop
cannot drift it. So "the mean is unconstrained / drifts off" is not the bug in the default recipe.

**But the user's instinct points at the real gap once you generalize it.** The quantity that must be
correct for an unbiased n_s is not the (z,τ₀)-conditional MEAN — it is the (z,τ₀)-conditional **θ-RESPONSE**,
i.e. `∂r̂/∂θ_ns`. And **nothing in the loss constrains the slope of r̂ in the n_s direction.** Two pieces
of evidence:

- **`coherent_debias_term` (model.py L342–384) penalizes the per-cell θ-MEAN of the residual error,
  `⟨r̂ − t_p_resid⟩_θ`.** But `t_p_resid = (logP − cell_mean)/sig_cosmo` has `⟨t_p_resid⟩_θ = 0` per cell
  BY CONSTRUCTION (the cell-mean is subtracted). So this term drives `⟨r̂⟩_θ → 0` — a coherent OFFSET. A
  **shrink-to-centre** error (r̂ tracks the sign of t but with too-small magnitude, i.e. `∂r̂/∂θ <
  ∂t/∂θ`) has `⟨r̂ − t⟩_θ ≈ 0` and is **completely invisible to this regularizer.** This is *exactly*
  why w_coh=80 (which the recipe runs) drove the gate failures down but left a −0.65σ n_s residual: it
  was never the right penalty for a slope deficiency.
- **`p_resid` is a per-(row,k) MSE in σ_cosmo-whitened units.** A per-element MSE is minimized by
  regressing each prediction toward the conditional mean of the target when the inputs are noisy /
  the feature is hard to extract — the textbook attenuation/shrinkage. With the n_s signal at ~7.6% of
  the marginal log-variance and 9 params competing through one shared trunk, the optimizer's
  cheapest-loss solution is to UNDER-respond to n_s (predict r̂ flatter in n_s than truth). MSE does not
  penalize systematic under-response; it rewards low per-row variance. **The −0.65σ coherent
  under-prediction is the signature of MSE shrinkage on the weakest signal direction.**

So feedback #4, read as "you do not have a loss that guarantees the conditioned prediction is correct,"
is RIGHT — the conditioned object that is wrong is the θ-slope, and the loss is structurally blind to it.

**Cheap test (forward-only, no retrain):** With the *already-trained* fold emulators, compute the
**n_s-response attenuation factor** directly: for each held-out sim, finite-difference the emulator's
predicted log-P1D w.r.t. n_s (`∂logP̂/∂ns` via the existing `jax.jacrev` in `diag_emu_bias_allfolds.py`)
and compare to the *empirical* truth slope from the design — i.e. regress true `logP_filt(sim)` on `ns_u`
within each (z,τ₀) cell (the cell already groups fixed (z,τ₀)) to get `∂logP_true/∂ns`, then form
`β = ⟨∂logP̂/∂ns⟩ / ⟨∂logP_true/∂ns⟩` per (z,k). If `β < 1` coherently (predicted slope is attenuated),
that is the shrinkage and it *quantitatively predicts* the bias sign and rough magnitude
(bias ≈ (1−β)·(truth−mean projection)). This is ~30 lines reusing the all-folds machinery and needs no
training. It also directly tests #1 (is β worse at low τ₀?) and #2 (is β worse only for ns_u>0.9?).

---

## Feedback #1 — τ₀ / n_s degeneracy through the two heads (good high-τ₀, poor low-τ₀)

**Likelihood as root cause: MEDIUM (a real contributing channel; not shown to be primary).**

The architecture *is* the two-stage design the user describes: `BaselineHead(z,τ₀)` is θ-blind
(model.py L125–135, takes only `[z,τ₀]`, never the latent) → the (z,τ₀)-conditional mean; `HeadB(concat
(latent,τ₀))` (L198–208) → the σ_cosmo-whitened cosmology residual, so **all** of n_s flows through r̂,
and the n_s prediction is literally "cosmology around the (z,τ₀) mean." The user's mechanism — if the
forward is accurate at high τ₀ but not low τ₀, a coherent n_s bias follows via the strong n_s↔τ₀
anti-correlation (reference: corr ≈ −0.66 in the cobaya chains) — is a coherent, real channel.

Two pieces of code/data evidence make this MEDIUM, not HIGH:
- **The asymmetry is confirmed but small and amplitude-flavored.** `embias_arch_tau0regime.npz` shows the
  low-k clean coherent error worsens monotonically toward low τ₀ (-0.275% vs -0.076%), and the table's
  `corr(bNs, ap_u)=+0.21` / low-Ap-quartile mean bNs -0.97 corroborate that the low-amplitude regime
  carries the n_s bias. But the existing diagnostic measures an **amplitude/low-k** error, not the
  n_s-TILT error split by τ₀. The degeneracy converts an amplitude error into ~2/3 of an n_s shift, but
  that is the route by which a TILT error and an AMPLITUDE error both end up partly in n_s — it does not
  prove the *primary* source is a τ₀-regime mean error.
- **The frozen θ-blind baseline cannot be the τ₀-asymmetry carrier under the default recipe.** Since the
  baseline is frozen and θ-blind, the n_s response is entirely `sig_cosmo·∂r̂/∂θ`. Any τ₀-dependent n_s
  bias is HeadB's residual being a worse function of (latent, τ₀) at low τ₀ — consistent with #4's
  shrinkage being *deeper* where the signal is weaker (low τ₀ = lower mean flux = lower P1D = noisier
  cosmology extraction). So #1 is best read as **a τ₀-localized amplification of the #4 shrinkage**, not
  an independent cause.

**Cheap test (forward-only):** extend the #4 attenuation test to split β by τ₀-rung (the
`embias_arch_tau0regime` binning already exists): compute `β_ns(τ₀-rung)` = predicted-slope / truth-slope
in n_s, per τ₀ band. If β is materially smaller at low τ₀ (e.g. 0.4 at α=0.69 vs 0.8 at α=1.3), #1 is a
genuine multiplier on the bias and a τ₀-resolved C_emu (or a τ₀-stratified residual loss weight) is
warranted. ~15 lines on top of the #4 test. (No retrain.)

---

## Feedback #3 — log-P1D + per-k standardization under-resolves the TILT (spectral bias)

**Likelihood as root cause: MEDIUM-HIGH (mechanistically the same gap as #4, seen from the target/basis
side; the cheapest *fix* lives here).**

The redesign (2026-06-02) already adopted log-space + a θ-blind baseline + σ_cosmo re-whitening +
no-stored-grid — three of the four LaCE-aligned moves. The **missing fourth piece is exactly the tilt
protection** the user flags, and I can point to two concrete places it is absent:

1. **Per-k independent whitening flattens the tilt.** `t_p_resid = (logP − cell_mean)/sig_cosmo` divides
   each k by its OWN within-cell std (I measured sig_cosmo rising 0.063→0.123 across k). The n_s tilt is
   a COHERENT slope across k; dividing each k by a different number, then summing a per-k MSE, gives the
   *coherent slope direction* no special status — it is diluted across 172 nearly-independent scalar
   regressions. LaCE (2305.19064) deliberately avoids per-k marginal-std division: it emulates the
   COEFFICIENTS of a smooth log-k polynomial of `log(P/median)`, so the slope (`α_1 ≈ dlogP/dlogk`,
   ~the tilt) is a NAMED, separately-weighted output. Their own ablation shows the basis/target choice is
   the dominant accuracy lever (10%→1.4%). This is the same finding as #4 from the target side: the loss
   gives the tilt no protected DOF.
2. **The shared 24-mode SVD basis is non-orthonormal and couples bands.** I measured `‖BBᵀ−I‖_F = 7.9`,
   `cond(BBᵀ) = 163` on final_fold0. A high-k mis-fit (where #1/PRIYA convergence is worst) leaks into the
   low-k/tilt direction because the modes are not orthogonal — the channel finding from §3 of the
   walkthrough, now confirmed numerically worse than reported. n_s IS the k-slope, so a basis that smears
   high-k error into a slope-like mode directly biases n_s.

The `edge_emphasis_k_weight` (data.py L380) is a U-shaped per-k REWEIGHT — but it reweights the per-k MSE,
which still does not create a coherent-slope DOF; it just up-weights the two endpoints. It addresses the
A_p low-k amplitude bias it was built for, not a tilt.

**Cheap test #1 (forward-only, diagnostic):** project each held-out sim's residual error
`(r̂ − t_p_resid)` onto a low-order log-k Legendre/polynomial basis (constant, slope, curvature) per
(class, z). If the **slope coefficient** of the error is coherently negative across sims (the emulator's
tilt is too shallow), #3 is confirmed as the *target-misspecification* face of the bias. ~25 lines, reuses
the all-folds predictions. **Cheap test #2 (the cheapest FIX to validate, near-retrain-free):** re-fit
the residual head's `p_filt_basis` to a forced-orthonormal version (Gram-Schmidt / re-SVD of the current
basis) and *keep the coefficient layer*, then re-measure the bias — this tests whether de-coupling the
bands alone removes a chunk of the −0.65σ without touching the trunk. If yes, a 1-layer re-fit (not a full
retrain) is the cheap remedy.

---

## Feedback #2 — SVD basis / encoder extrapolates badly for n_s>1.0 (not on a Latin hypercube)

**Likelihood as root cause: LOW for the −0.65σ headline; the C1 "resolved" claim is WRONG.**

The reference agents and my coverage count CONFIRM the structural claim: n_s>1.0 is a 2-of-60 BO-style
extension (≈12 expected under LHS), gap-separated, jointly tied to the high-αq corner. **So C1 ("ns_hi
1.05 is backed by training points by construction") is FALSE as stated** — the box edge is defined, but it
is NOT space-filled, and `data.PARAM_LIMITS == emulator_params.json` proves only that the box is the same,
not that it is populated. I dissent from the onboarding C1 claim (see Dissents).

But the user's own hypothesis predicts the bias should live at the HIGH-n_s edge, and the data say
otherwise:
- The worst fold is **fold 5 (mid-n_s, the DENSEST region)**, not fold 7 (high-n_s). Fold 7 (the only
  fold reaching n_s>1.0) is only -0.63σ — *better* than the mid folds.
- `corr(bNs, |ns_u-0.5|)=+0.42` but the minimum is AT the centre, and the high tail (1 sim) does not
  dominate; dropping the largest outlier barely moves the pooled mean (-0.65→-0.61).
- The bias is coherent across the *whole* n_s range including the well-sampled interior — an extrapolation
  artifact would be confined to the sparse n_s>0.995 sliver.

So n_s>1.0 sparsity is a **real validity caveat for the real-data fit** (if the posterior wanders to
n_s>1.0 the emulator is unsupported there), but it is **not the mechanism of the interior −0.65σ**. It is
a second, separate problem: the production fit should either (a) refuse n_s>0.995 (the user's preference),
or (b) carry a sharply inflated C_emu for n_s>0.995.

**Cheap test (forward-only):** the #4 attenuation β-test, restricted to ns_u>0.9 vs ns_u<0.9. If β
collapses *only* above 0.9 → #2 is the driver (extrapolation). If β is uniformly <1 across the interior →
#2 is ruled out as the primary cause and #4/#3 own it. Same script, a one-line stratification. (My
prediction from the table: β<1 across the interior, so #2 will be ruled out as primary.)

---

## Priority ranking (this lens), most → least promising as the actual root cause

1. **#4 (conditional θ-RESPONSE loss gap / MSE shrinkage)** — the only mechanism that explains a coherent
   under-prediction the existing anti-bias terms cannot catch, and the table's coherence across the whole
   interior. The right object (slope, not mean) is provably unconstrained.
2. **#3 (per-k whitening + non-orthonormal basis → no protected tilt DOF)** — the same gap seen from the
   target/basis side; carries the cheapest *fix* (basis re-orthonormalization, LaCE-style coefficient
   target). #3 and #4 are two faces of one root cause.
3. **#1 (τ₀-regime asymmetry amplifying the shrinkage via n_s↔τ₀)** — confirmed as a real multiplier
   (worse at low amplitude/τ₀) but a modifier of #4, not independent.
4. **#2 (n_s>1.0 extrapolation)** — real validity caveat + the C1 claim is wrong, but NOT the interior
   −0.65σ; a separate production-fit guard.

**Cheapest to TEST:** #2 and #1 (both are one-line stratifications of the single #4 β-attenuation script).
**Cheapest to FIX without a full retrain:** #3 — re-orthonormalize the existing 24-mode basis and re-fit
ONLY the coefficient/output layer (the trunk is reused), then re-measure. #4's true fix (a θ-response /
finite-difference-slope loss term) needs a retrain; #2's fix is a prior bound (n_s≤0.995) and costs
nothing.

---

## Dissents

- **Against the "C1 resolved" claim (onboarding):** I dissent. `data.PARAM_LIMITS ==
  emulator_params.json` proves the box is identical, NOT that n_s∈[0.995,1.05] is "backed by training
  points by construction." Only 2/60 points exceed 1.0, gap-separated, BO-style — the edge is *defined*
  but *unpopulated*. C1 should be downgraded to "the box matches; the high-n_s edge is sparse and the
  emulator is an extrapolant there." (Agrees with the user's #2 and the PRIYA reference agents.)
- **Against the "finite-60-sim LOSO floor to MARGINALIZE" prior framing:** partial dissent. The floor is
  real, but the leading-hypothesis framing treats the −0.65σ as irreducible scatter to be absorbed by a
  correlated C_emu. The CS evidence says a large fraction is **a fixable loss/target/basis
  misspecification** (no protected tilt DOF; non-orthonormal basis; MSE shrinkage), i.e. a coherent,
  *removable* mode, not a generalization floor. Marginalizing a removable coherent bias with C_emu inflates
  σ(n_s) unnecessarily and risks under-coverage if the mode is mis-estimated in-sample (the whitening
  1.28→0.93 number is already flagged as in-sample / circular). **Measure β (the attenuation) FIRST**; if
  β<1 coherently, the bias is structural and at least partly removable — do not jump to C_emu.
- **Against treating w_coh=80 as "the de-bias is handled":** the coherent_debias_term penalizes the wrong
  moment (per-cell θ-MEAN, which is 0 by construction for a slope error). It cannot have removed the n_s
  tilt bias and should not be cited as evidence the coherent mode is controlled.

---

## Ordered, cheap-first next actions

1. **Build the one β-attenuation diagnostic** (forward-only, reuses `diag_emu_bias_allfolds.py`'s jacrev):
   per held-out sim, `β_ns(z,k) = ⟨∂logP̂/∂ns⟩ / ⟨∂logP_true/∂ns⟩` (truth slope = within-cell regression
   of logP_filt on ns_u). Output: pooled β, β vs ns_u (tests #2), β vs τ₀-rung (tests #1), β's k-shape
   (tests #3). ONE script answers #1, #2, #3, #4 directionally. **Do this first.**
2. **Project the residual error onto a low-order log-k polynomial** (constant/slope/curvature) per
   (class,z) and check the slope coefficient's coherence — confirms #3 as target-misspecification.
3. **Re-orthonormalize the trained `p_filt_basis` (re-SVD), re-fit ONLY the HeadB output/coeff layer**
   (trunk + baseline frozen, ~minutes), re-run `diag_emu_bias_allfolds.py`. Quantifies how much of the
   −0.65σ is the band-coupling channel (#3) vs the trunk's θ-response (#4). Near-retrain-free.
4. **Add an n_s≤0.995 production-fit bound (or a step-inflated C_emu above 0.995)** — addresses #2 as a
   validity guard at zero training cost; independent of the interior fix.
5. **Only if β is ≈1 (no attenuation) and 1–3 leave the bias intact**, fall back to the
   z/τ₀-resolved correlated C_emu marginalization — but with σ-inflation measured out-of-sample, not on
   the in-sample whitening number.
6. **If a retrain is approved (last resort):** add a finite-difference n_s-slope-matching term to
   `joint_loss` (penalize `(∂r̂/∂ns − ∂t/∂ns)²` estimated from cell-paired sims) — the direct fix for #4 —
   and switch the residual target to a smooth-log-k-coefficient (LaCE) parameterization to protect the
   tilt (#3). Re-certify everything.
