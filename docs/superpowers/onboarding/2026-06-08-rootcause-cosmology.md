# Root-cause adjudication — the −0.65σ coherent n_s under-prediction (COSMOLOGY/EMULATOR lens)

Date 2026-06-08. Reviewer: cosmology / P1D-emulator specialist (per-checkpoint adversarial review).
All numbers below are reproduced this session from disk (cache, norm.pkl, design json, the all-folds txt),
not quoted. Env:
`PYTHONNOUSERSITE=1 PYTHONPATH=/home/mfho/hcd_priya JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" /home/mfho/.conda/envs/emu-jax/bin/python3`

---

## 0. The headline adjudication (feedback #2 / the "C1" claim)

**The meta-onboarding "C1" claim — that the ns box to 1.05 is "backed by training points by
construction" — is WRONG as stated, and the user is RIGHT that ns>1.0 is a sparse Bayes-opt
extension, NOT a Latin hypercube.** I confirm from the design file directly
(`/home/mfho/lya_emulator_full/kodiaq_2_2_4_6-48-48/emulator_params.json`, `sample_params` col 0):

- 60 design points, ns∈[0.8032, 1.0396]. **Only 2 of 60 exceed ns=1.0** (1.0188, 1.0396); only 3 exceed
  0.995. There is a visible **gap** between 0.9979 and 1.0188 (unit 0.792→0.875). A uniform LHS over
  [0.8,1.05] would put ~12 of 60 above 1.0. **C1's "by construction" is false** — `data.PARAM_LIMITS`
  matching `emulator_params.json` proves the BOX is the design box, not that the box is uniformly POPULATED.
  This matches the extended-PRIYA reference (12 LF sims opened ns 0.995→1.05 jointly with the
  alphaq=2.5–3.0 corner; sparse coupled wedge, not a re-LHS). **So: C1 is NOT resolved; it conflated the
  parameter box with parameter coverage.**

**HOWEVER — and this is the decisive measurement — the ns>1.0 hole is NOT the root cause of the −0.65σ
bias.** I computed the per-sim ns bias vs ns value from `emu_bias_allfolds.txt` (60 honestly-held-out
sims). The bias does **not** concentrate at the high-ns extension; it concentrates in the **interior
Planck band**:

| ns band | n | mean ns_bias | frac neg |
|---|---|---|---|
| ns<0.85 (interior-low) | 14 | **−0.06σ** | 0.57 |
| 0.85–0.90 | 13 | −0.43σ | 0.77 |
| **0.90–0.95** | 18 | **−1.16σ** | 1.00 |
| 0.95–0.965 | 4 | −0.92σ | 1.00 |
| 0.965–0.995 (Planck→old edge) | 8 | −0.63σ | 0.88 |
| ≥0.995 (sparse extension) | 3 | −0.89σ | 1.00 |
| ≥1.0 (only 2 design pts) | 2 | −0.66σ | 1.00 |

**Decisive cut test (forward-only, on the existing data):** cutting the prior to ns≤1.0, ≤0.995, or even
≤0.965 leaves the pooled bias essentially unchanged (−0.646 → −0.646 → −0.633 → −0.634σ). Only cutting
to **ns≤0.90 — discarding the entire upper half of the design (33 of 60 sims)** — drops it to −0.24σ
(t=−1.26, no longer significant). **An ns<1.0 prior cut would NOT collapse the bias.** The worst-biased
sims (0.90–0.95) sit squarely INSIDE the original [0.8,0.995] LHS, far from any edge.

The structure is a coherent **downward tilt-pull whose magnitude grows with ns** (regression slope
d(ns_bias)/d(ns) = −5.05 per unit ns ≈ −0.05σ per Δns=0.01; corr(ns_bias, ns_unit) = −0.35). Its
zero-crossing is at **ns≈0.776 — BELOW the entire design range (min 0.803)**. So this is NOT clean
regression-to-the-design-mean (which would zero-cross at 0.904); it is a near-uniform negative tilt
under-prediction that worsens at high ns. That is the signature of a **tilt-direction emulator error**,
not a box-edge coverage hole.

**Verdict on the user's mechanism:** the coverage HOLE is real and the C1 claim is wrong, but the hole is
not where the bias lives. The user's stronger intuition — "ns>1.0 might work near Planck fiducial but not
at edge cases" — would predict the WORST bias at high ns AND at edge (extreme other-param) cases; instead
the worst bias is at interior, near-Planck ns. **Cost of an ns<1.0 cut:** it would (a) not fix the bias,
(b) re-introduce a hard prior boundary right where the published eBOSS result lands (n_P=1.009, JUST above
0.995), biasing the real-data ns posterior LOW by railing it against 1.0 — which would *amplify*, not
cure, a low-ns pull. **Recommend AGAINST the ns<1.0 cut as a bias fix; recommend FOR fixing the C1 wording
and adding an honest "ns>0.995 is sparsely-trained, treat the upper tail as extrapolation" caveat** (the
real-data result, if it lands above ~0.99, should be flagged as resting on the sparse extension).

---

## 1. Feedback #1 — τ₀/n_s degeneracy through the heads — MEDIUM

The user's worry: τ₀ and ns are strongly anti-correlated (chains: corr≈−0.66); if the two-stage head is
accurate at high τ₀ but not low τ₀, that asymmetry re-expresses as an ns bias.

**What I verified (the physics is RIGHT, the asymmetry is REAL):**
- I built a toy Fisher from the *empirically measured* clean-class log-P response shapes in the cache
  (within-cell regression for ns, A_p; fixed-cosmology τ₀-ladder regression for τ₀), whitened by σ_cosmo,
  over the data-range k. The induced posterior correlations reproduce the **correct sign and ordering** of
  the field-standard degeneracies: corr(ns,τ₀)=−0.91, corr(Ap,τ₀)=−0.79, corr(ns,Ap)=+0.60 (single-z,
  diagonal — magnitudes are inflated vs the z-marginalized chains' −0.66/−0.7..−0.81/+0.18..0.45, exactly
  as expected because z-borrowing breaks the degeneracy). **So the emulator's response GEOMETRY is
  physically correct — the τ₀↔ns degeneracy is not broken or mis-signed.** The response shape cos(τ₀,ns)
  in whitened output space is **0.845** — τ₀ and ns are *more* aligned than τ₀ and A_p (0.727); a small
  high-k tilt error therefore leaks ~2/3 into ns by this coupling.
- The τ₀-regime asymmetry IS present and in the worrying direction. The on-disk diagnostic
  `embias_arch_tau0regime.npz` (`bandmean`) shows the clean low-k coherent error worsening **monotonically
  toward low τ₀**: −0.275% at the lowest rung (α=0.69) vs −0.076% at the highest (α=1.30). The forward is
  least accurate at low τ₀.

**Why only MEDIUM, not high:** (a) the existing τ₀-regime diagnostic is a LOW-k / amplitude-flavored error,
carried by the residual head — it is the A_p-direction signature, and the all-folds measurement shows A_p
bias ≈ 0. NO existing diagnostic isolates the ns-TILT bias split by τ₀ regime. (b) The −0.65σ ns bias is
coherent across ALL τ₀ (the all-folds Fisher already marginalizes τ₀), so it cannot be purely a
low-τ₀-only artifact — if it were, marginalizing τ₀ would absorb most of it. The degeneracy is the
*amplifier/channel* by which a tilt error becomes an ns number, not the *source* of the tilt error.

**Cheap test:** extend `embias_arch_tau0regime.py` to bin the **HIGH-k (k>0.02) coherent fractional error**
(the ns-tilt band) by τ₀ rung, separately per fold, and project it onto the whitened ns-response vector
(`/tmp/resp_shapes.npz` `ns_resp`). If the high-k tilt error is τ₀-flat, feedback #1 is the *channel* not
the cause (drop to low); if it is strongly low-τ₀-skewed AND aligned with `ns_resp`, #1 is promoted to high.

---

## 2. Feedback #2 — extended prior / ns>1.0 not on a Latin hypercube — LOW (as the root cause) / but C1 is wrong

Adjudicated in §0. **As a root cause of the −0.65σ: LOW.** The C1 "backed by construction" claim is
factually wrong (2/60 points >1.0, gapped — confirmed against the design file), and the user's coverage
concern is legitimate for the real-data UPPER tail. But the bias is interior-ns, an ns<1.0 cut does not
collapse it (forward-only cut test: −0.646→−0.633σ), and cutting ns<1.0 would *bias the real ns posterior
low* by railing it against a boundary right at the published 1.009 result. The extended-prior reference's
own mechanism (sparse high-ns wedge → GP reverts to mean → high-ns pulled down) is *consistent in sign*
with the mild shrink-to-centre and plausibly explains the *extra* −0.2σ between the 0.90–0.95 and ≥0.965
bands, but it cannot explain the −1.16σ core at interior near-Planck ns. The dominant ns-bias correlate is
ns itself (−0.35); the OTHER extended parameter, alphaq, correlates +0.26 (the "wrong" way for an
edge-hole story — high-alphaq sims are LESS biased).

**Cheap test (kills or confirms cleanly):** the cut test is already done (above). To directly test the
"sparse-region GP-revert" sub-claim, audit whether the 12 ns>0.995 design points space-fill the other 8
params or pin to the alphaq corner — load `sample_params`, restrict to ns>0.99, and check the convex-hull
volume / nearest-neighbour spacing of those rows vs a random 12-subset. If they pin to the corner, flag the
real-data upper-ns tail as extrapolation (a *reporting* fix, not a bias fix).

---

## 3. Feedback #3 — log-P normalization friendly for A_p, maybe not for the tilt — MEDIUM-HIGH (the leading lens-specific cause)

This is the candidate I rank HIGHEST from the cosmology/emulator lens, and the LaCE reference supports it.

**The physics, measured this session (`/tmp/resp_shapes.npz`):** in log-P (clean class), the two cosmology
directions live in *different* k-bands and the σ_cosmo whitening treats them asymmetrically.
- **A_p is an amplitude:** d logP/d A_p_unit is a low-k boost (+0.21 at k≈0.005) declining to ~0 by
  k=0.06. Its whitened signal energy peaks at low-k (25/50/75% energy at k = 0.004/0.008/0.013).
- **n_s is a tilt:** d logP/d ns_unit is a **pivot** — negative at low-k (−0.27 at k=5e-4), zero-crossing
  near k≈5e-3, positive at high-k (+0.16 at k=0.04). Its whitened energy is a **HIGH-k** phenomenon
  (25/50/75% energy at k = 0.019/0.029/0.042) — exactly the band the walkthrough's PART-B/PART-D found is
  the mis-fit, SVD-basis-coupled channel.
- **σ_cosmo RISES with k** for the clean class: median 0.067 at k<0.005 vs 0.095 at k>0.03 (≈1.6×; 0.110 at
  k=0.06). So the per-(c,k) whitening **down-weights the residual loss exactly in the high-k band where the
  n_s tilt signal concentrates**, relative to the low-k A_p band. The MSE-on-whitened-residual loss
  therefore allocates relatively *less precision per natural unit of tilt signal* than per unit of
  amplitude signal — a tilt-disadvantaging normalization, which is precisely the user's worry.

**Why MEDIUM-HIGH not HIGH:** the net whitened SIGNAL ENERGY is actually *larger* for ns than A_p
(Σ_k(resp/σ_cosmo)² = 564 for ns vs 325 for A_p, ratio 1.73), so the loss is not *globally* starved of ns
information. The disadvantage is RELATIVE and k-localized, and it compounds with two things to produce a
coherent bias rather than just scatter: (i) the rising-σ_cosmo down-weight sits on top of the high-k band
where the finite-60 GP/NN interpolation of the tilt SHAPE is hardest (fewest constraints, SVD basis
non-orthonormal ‖BBᵀ−I‖≈2.7); (ii) LaCE's documented lesson (their ablation: linear-k→log-k coefficient
target moved accuracy 10%→1.4%) is that the TARGET BASIS — not under-training — governs whether the tilt
direction is resolvable. We adopted log-space + inverse-variance + baseline-removal, but NOT LaCE's
smooth-log-k POLYNOMIAL-COEFFICIENT target that makes the slope a *named, separately-weighted DOF* (their
alpha_1 ≈ dlogP/dlogk). With our per-(c,k) whitening, the tilt is one diffuse high-k mode the per-sim MSE
(dominated by within-cell scatter) does not directly penalize the per-cell θ-MEAN offset of — `joint_loss`
adds `coherent_debias_term` (w_coh=80) precisely because the plain MSE leaves the coherent k-tilt as an
unconstrained d.o.f. (the de-bias term is FLAT in k by design; the team found edge-weighting it
over-corrects low-k). That this regularizer exists and is still leaving a −0.65σ coherent ns pull is
itself evidence the *tilt direction* specifically is under-constrained by the loss geometry.

**Cheap test (decisive, no retrain):** compute the **per-cell θ-mean residual** `⟨r̂ − t_p_resid⟩_θ` of the
deployed model (the exact quantity `coherent_debias_term` minimizes) and project it onto the whitened
ns-response `ns_resp/σ_cosmo` vs the A_p-response. If the deployed coherent residual is significantly
aligned with the ns-tilt direction (and not A_p), the normalization/loss geometry is the proximate cause.
Then a forward-only counterfactual: re-whiten the residual target by a k-FLAT σ_cosmo (use the median, not
the per-k value), re-derive the *target*, and re-measure the all-folds ns bias on a SINGLE quick-retrained
fold (or, cheaper, just re-weight the existing per-sim residual errors by σ_cosmo(k)/median and re-run the
Fisher MAP-shift) — if a flat whitening shrinks the ns bias, #3 is confirmed HIGH.

---

## 4. Feedback #4 — no loss on the conditional MEAN at fixed (τ₀,z) — LOW / largely refuted for the default recipe

The user worries there is no loss guaranteeing the (z,τ₀) conditional mean is correct, so joint fine-tune
could pull the baseline off and contaminate the cosmology.

**What the code shows (`model.py`, `train.py`):** there IS an explicit, separate conditional-mean loss
(`joint_loss` term `p_base` = MSE of `BaselineHead` vs `t_p_base` = the σ_marg-standardized (z,τ₀) cell
mean), AND a dedicated `_prefit_baseline` (8000-epoch baseline-ONLY fit on the ~306-cell table), AND the
production default FREEZES the baseline (`freeze_baseline=did_prefit`; eqx-partitioned out of the joint
loop). So the joint fine-tune CANNOT pull the baseline off its prefit floor in the default path. The user's
"you might not have a loss on the conditional mean" is **factually answered — the loss and the prefit both
exist.**

**Why not fully ruled_out:** the design PINS the conditional mean at the prefit *train*-cell-mean, it does
not GUARANTEE it equals the held-out TRUE conditional mean (under LOSO the held-out cell mean can differ —
the finite-60 gap the freeze does not close). And a non-default `freeze_baseline=False` path can drift it.
BUT for the −0.65σ ns bias specifically this is a near-dead-end: the baseline is θ-BLIND
(∂m̂/∂θ≡0, verified jacfwd=0), so by construction it carries **zero ns response** — ALL ns/tilt response
flows through r̂ (the residual head). A baseline mean error is an amplitude/offset error per (z,τ₀) cell,
which marginalizes onto τ₀/A_p, not onto the ns tilt. The reference's decomposition agrees: the low-k
coherent error is carried by HeadB's residual (−0.136%), the baseline is near-exact (+0.024%). So feedback
#4 cannot be the source of an ns-TILT bias.

**Cheap test:** measure the deployed baseline's held-out per-cell mean error vs the LOSO true cell mean and
project onto `ns_resp` — confirm (as I expect) it projects onto the amplitude/τ₀ direction, not the tilt.
If the baseline error has ~0 ns-tilt projection, #4 is ruled_out as the ns-bias cause.

---

## 5. Priority ranking (this lens), dissents, next actions

**Priority (most→least likely as the actual root cause of −0.65σ ns):**
1. **#3 normalization/loss tilt-disadvantage** — the tilt lives at high-k, σ_cosmo rises at high-k
   down-weighting it, LaCE says the target basis is the lever, and the existing coherent-debias regularizer
   still leaves a coherent ns pull. Most direct cosmology-lens mechanism.
2. **#1 τ₀/ns degeneracy** — the channel/amplifier (cos(τ₀,ns)=0.845, correct-sign degeneracy, real
   low-τ₀ accuracy degradation), but the existing diagnostic is amplitude-flavored and the bias survives
   τ₀ marginalization, so likely the conduit, not the source.
3. **#2 extended ns>1.0 prior** — C1 is genuinely wrong (real coverage hole, real reporting fix), but
   forward-only cut tests show it is NOT the bias source.
4. **#4 conditional-mean loss** — the loss + prefit + freeze already exist; θ-blind baseline carries no
   tilt response by construction; cannot source an ns-tilt bias.

**DISSENTS (where I disagree with the prior framing):**
- **I disagree that "C1 is resolved."** It conflated the design BOX with parameter COVERAGE. Only 2/60
  design points exceed ns=1.0 (gapped). The wording must change and the real-data upper-ns tail must be
  flagged as extrapolation. (This is the user's point #2 and they are right; the prior onboarding's "backed
  by training points by construction" is the error.)
- **I partly disagree with the "finite-sim LOSO floor — just marginalize it with C_emu" leading
  hypothesis.** The bias is NOT a generic regression-to-mean (its zero-crossing at ns≈0.776 is below the
  entire design range; a pure shrink would zero-cross at the design mean 0.904). It is a *directional,
  near-uniform tilt under-prediction* — i.e. the loss geometry systematically under-resolves the tilt
  DIRECTION, which is a fixable target/loss-design issue (feedback #3), not an irreducible data floor.
  Marginalizing a coherent, response-aligned ns mode with a diagonal/cross-class-but-k-diagonal C_emu only
  *resizes* it, it does not whiten a mode aligned with the ns-response — so "just C_emu it" risks an
  uncalibrated ns posterior. If it IS partly a finite-sim floor, the fix is a tilt-aware loss/target, not
  C_emu.
- The walkthrough's "high-k-sourced, SVD-basis-coupled" channel finding (PART B/D) is, in my reading, the
  SAME mechanism as feedback #3 viewed in basis space: the tilt = the high-k shape, and the global
  non-orthonormal basis + high-k σ_cosmo down-weight is exactly what under-resolves it.

**Ordered cheap-first next actions (all forward-only / no full retrain):**
1. **Project the deployed coherent per-cell residual `⟨r̂−t_p_resid⟩_θ` onto the whitened ns-response vs
   A_p-response** (`/tmp/resp_shapes.npz`). Decides #3 vs #1 vs #4 in one measurement. (~30 min.)
2. **σ_cosmo-flatten counterfactual:** re-weight the existing per-sim residual errors by σ_cosmo(k)/median
   and re-run the all-folds Fisher MAP-shift; if the ns bias shrinks, #3 is confirmed and a flat (or
   LaCE-coefficient) whitening is the fix. (~1 hr.)
3. **High-k (k>0.02) coherent error binned by τ₀ rung, per fold, projected onto ns_resp** — promotes or
   demotes #1. (extend `embias_arch_tau0regime.py`; ~1 hr.)
4. **Fix the C1 wording** in the meta-onboarding + add the "ns>0.995 sparsely trained → upper tail is
   extrapolation" caveat; audit the 12 ns>0.99 design points' space-filling. (reporting; ~30 min.)
5. Only if 1–3 implicate the basis/target: prototype a LaCE-style smooth-log-k coefficient target for r̂
   (the slope as a named, up-weighted DOF) on ONE fold and re-measure the ns bias before any full retrain.

**Why NOT an ns<1.0 prior cut:** forward-only cut test shows it does not collapse the bias (−0.65→−0.63σ)
and it would bias the real-data ns posterior LOW by railing against a boundary at the published 1.009.
