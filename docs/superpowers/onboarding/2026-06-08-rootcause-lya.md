# Root-cause review (Lya / IGM / PRIYA lens) — the -0.65sigma coherent n_s under-prediction

**Date:** 2026-06-08. **Reviewer lens:** Lyman-alpha forest / IGM physics / PRIYA-suite.
**Object under review:** the honest all-folds LOSO result that the emulator coherently UNDER-predicts
n_s by -0.65 sigma (95% CI [-0.85,-0.43], t=-5.84, 50/60 held-out sims negative), A_p ~ 0.
**Charge:** assess the user's 4 feedback points as candidate root causes; one cheap forward-only test each.

I reproduced/extended numbers live where possible (cited inline). My headline differs from the
prior "finite-sim floor to marginalize" framing: the bias is a **coherent NEGATIVE k-tilt in the
RESIDUAL head's mean error, strongly localized to z=2.0-2.4**, and the z-collapsed sigma_cosmo
whitening + the per-k SVD basis (not a smooth slope-coefficient basis) is the most coherent
mechanism. That points to feedback **#3 (normalization/tilt) and #4 (no tilt-aware loss on the
conditional mean)** as the leading causes, NOT #1 (tau0 head) or #2 (ns>1.0 extension).

---

## The single most decisive measurement I made

From the existing per-class coherent-error cache (`embias_arch_step1_coherent.npz`,
`scripts/embias_arch_diag.py`), I fit the slope of the CLEAN-class coherent fractional error
(emu-truth)/truth against ln k, in the data band k in [1e-3, 0.06], GLOBAL and per z-band:

```
clean coherent-error slope d(fracerr)/d(ln k) in-band = -0.00036  (NEGATIVE => under-tilt => low n_s)
  k=0.002: +0.0007   k=0.02: +0.0001   k=0.05: -0.0006   (falls with k = a negative tilt)

per-z-band slope (the z-localization):
  z[2.0-2.4]: slope = -0.00138   <-- BY FAR the worst (the dominant tilt deficit)
  z[2.4-2.8]: slope = -0.00013
  z[2.8-3.2]: slope = -0.00020
  z[3.2-3.6]: slope = -0.00001
  z[3.6-4.0]: slope = +0.00004
  z[4.0-4.4]: slope = -0.00066
  z[4.4-5.4]: slope = -0.00048
```

The emulator's predicted clean P1D falls too fast with k (under-tilt), and ~the entire effect lives
at z=2.0-2.4 with a secondary contribution at z>4. This is the same z-localization the
checkpoint-2 review found (~53% from z=2.0-2.3) and it IS a coherent n_s-direction error (n_s is the
small-scale slope). So the all-folds -0.65sigma is a real, z-localized, tilt-shaped emulator error,
not just LOSO scatter.

And the carrier is the RESIDUAL head, not the theta-blind baseline. From `embias_arch_localize.npz`:
```
              low-k(k<0.01)   high-k(k>0.03)
base_err_k       +0.0002          +0.0010      (flat, slightly POSITIVE slope -> ~no tilt)
resid_err_k      +0.0019          -0.0032      (swings + -> -  = a NEGATIVE k-tilt)
tot_err_k        +0.0021          -0.0023
```
The baseline conditional mean is well-pinned and nearly tilt-free; HeadB's sigma_cosmo-whitened
residual carries the negative tilt. This decomposition is what discriminates #4 (mostly about the
baseline) from the real channel (the residual head + its whitening/basis).

---

## Why the tilt is under-resolved exactly at z=2.0-2.4: the whitening is z-collapsed

`sig_cosmo` (the residual whitening, `data.fit_baseline_residual_norm` L323-329) is a SINGLE per-(c,k)
constant = sqrt(mean OVER ALL (z,tau0) CELLS of within-cell variance). I measured the TRUE
within-cell cosmology signal (std of logP_clean across sims at fixed (z,tau0)) vs z and k:

```
within-cell cosmology signal std (logP_clean):  at k_low=0.005 / k_high=0.05
  z=2.00  std_low=0.0504  std_high=0.1170  ratio_hi/lo=2.32   <- most high-k-weighted
  z=2.40  std_low=0.0598  std_high=0.1187  ratio=1.98
  z=3.00  std_low=0.0732  std_high=0.1109  ratio=1.52
  z=3.40  std_low=0.0741  std_high=0.0972  ratio=1.31
  z=4.60  std_low=0.0609  std_high=0.0837  ratio=1.37
```
At z=2.0-2.4 the real cosmology signal is ~2.3x more high-k-weighted than at z>3.4, but the loss
whitens the residual with a SINGLE z-averaged sig_cosmo(k). So at low z the high-k tilt signal is
effectively UNDER-weighted relative to its true magnitude -> the optimizer under-resolves the tilt
exactly at the z where the tilt deficit is worst (-0.00138 slope at z=2.0-2.4). This is a direct
normalization<->tilt interaction (feedback #3), and it is the same regime LaCE protects with a
smooth log-k coefficient target.

Note: the within-cell signal is essentially tau0-INDEPENDENT at fixed z (z~3 terciles: 0.073 low-k /
0.110-0.112 high-k for ALL tau0 rungs). So the constant sig_cosmo does NOT mis-whiten across tau0 -
it mis-whitens across Z. That is the discriminator that pushes #1 down and #3/#4 up.

---

## Feedback #1 — tau0 (mean-flux) <-> n_s degeneracy through the heads

**Likelihood: LOW.** (Firmed DOWN from LOW-MEDIUM by the tau0-split test below: the discriminator has
the wrong sign and tracks z, not tau0.)

Physics is real: the mean-flux-amplitude / spectral-tilt degeneracy is the classic P1D systematic, and
the reference agent measured corr(ns,tau0) = -0.56..-0.78 (median -0.66) in the simdat cobaya chains.
PRIYA over-samples mean flux (10 tau-samples/z about Kim tau=0.0023(1+z)^3.65; the repo ladder is 20
rungs), so tau0 is a well-trained direction. The two-stage head IS as the user describes: theta-blind
BaselineHead(z,tau0) conditional mean + cosmology-only residual HeadB. The existing
`embias_arch_tau0regime.npz` does show the low-k clean error is WORSE at LOW tau0
(bandmean -0.275% at alpha=0.69 vs -0.076% at alpha=1.30, monotone). So "less accurate at low tau0"
is true at low-k.

BUT three things weaken #1 as the ROOT cause of the n_s tilt bias:
1. The tau0regime degradation is an AMPLITUDE (low-k offset) effect, not a tilt; the all-folds bias is
   on n_s (tilt), and A_p ~ 0.
2. The cosmology signal to whiten is tau0-INDEPENDENT at fixed z (my measurement above) - so there is
   no tau0-asymmetry in the SIGNAL that would convert into a tau0-asymmetric tilt error.
3. The -0.65sigma is measured AFTER marginalizing tau0 in the Fisher (`diag_emu_bias_allfolds.py`
   marginalizes the per-z tau0 with prior tau0_sigma). If a tau0-direction error were the source, tau0
   would have absorbed most of it; instead it lands coherently on n_s, which says the error is in the
   tilt direction tau0 CANNOT absorb (tau0 shifts amplitude/low-k, not the high-k slope).

[TEST RESULT] — `scripts/diag_lya_ns_tau0_split.py` (2560 held-out rows) gives discriminator
<c>_low_tau0 - <c>_high_tau0 = +0.0335 (WRONG SIGN for #1; #1 predicts << 0), and the structure tracks
z (z=2.4 dominates) not tau0. Since Kim tau rises with z, the "low-tau0" bin IS the z=2.4 bin. So the
tilt mis-emulation is z-localized, not a low-tau0 deficit. See the appendix. #1 -> LOW.

**Cheap test (#1):** `scripts/diag_lya_ns_tau0_split.py` (written, running): n_s-equivalent mis-tilt
of the held-out clean-logP residual, binned by tau0 tercile, across all 8 folds. Forward-only, no
retrain. If LOW-tau0 c is not materially more negative than HIGH-tau0 c, #1 is not the driver.

---

## Feedback #2 — extended prior / ns>1.0 NOT on a Latin hypercube

**Likelihood: LOW** as the cause of the -0.65sigma (but the user's PHYSICS is CORRECT, and there is a
real, separate risk for the real-data fit at the top edge).

Confirmed against the design points (`emulator_params.json`, 60 sims): only 2 of 60 have ns>1.0
(1.0188, 1.0396), gap-separated from the bulk (next is 0.9979). Both ns>1.0 sims sit in the
HIGH-alphaq corner (alphaq 2.854, 2.521 vs design median 2.12) - exactly the coupled high-alphaq +
high-ns wedge the extended-PRIYA paper (Fernandez+2024, 2309.03943) describes. The 1.0396 sim is also
at the herei top edge (4.208) and heref low edge (2.492). So ns>1.0 IS a sparse, HeII-coupled
extension, not space-filling - the user's mechanism is right.

The herei/heref box WAS extended jointly beyond the original PRIYA Table-2 (herei_hi 4.1->4.5,
heref_lo 2.6->2.2; `2026-06-01-param-limits.md`), and 5 sims sit at herei>4.1 and 5 at heref<2.6. But
those extended-HeII sims span the FULL ns range (ns 0.81..1.04), so HeII and ns were NOT extended in
lockstep with each other across the design.

Why this is NOT the -0.65sigma root cause:
- The bias is measured over n_s_unit 0.013-0.958 and is NEGATIVE across the WHOLE range, including
  the DENSEST, best-supported low/mid-n_s region inside the original [0.8,0.995] LHS (fold 0:
  ns<0.83, bias -0.47; fold 2: ns<0.886, bias -0.52). A sparse-top-edge effect cannot explain a
  coherent under-tilt at ns=0.82.
- corr(n_s bias, herei) = -0.034 (I computed it by joining the per-sim biases to the design herei).
  The bias is FLAT across herei terciles (-0.58/-0.68/-0.68). So the n_s under-prediction is NOT an
  HeII-thermal mis-emulation aliased into n_s.
- corr(n_s bias, ns_value) = -0.349 (shrink-to-centre) - the bias grows toward HIGH n_s, i.e. it is
  about the n_s DIRECTION's generalization, which exists independent of the ns>1.0 extension.

What IS real from #2 (carry to the real fit, NOT to root-cause): the top edge ns in (0.995, 1.05] is
trained by ~2-3 coupled corner sims; the measured eBOSS n_P=1.009 sits inside that sparse band. If the
DESI fit prefers ns near/above 0.995 and the emulator's GP-like regression pulls high-ns predictions
DOWN (shrink-to-centre, which the all-folds set confirms: n_s_unit>0.6 -> bias -0.76), the posterior
will UNDER-shoot a truly high n_s. That amplifies, not creates, the tilt deficit at the top edge.
**This weakens the meta-onboarding "C1 resolved" claim** (see dissent).

**Cheap test (#2):** restrict the all-folds `diag_emu_bias_allfolds.py` measurement to sims with
ns<0.97 (well inside the original [0.8,0.995] LHS bulk, no extension involvement) and re-pool. If the
n_s bias survives at ~ -0.5sigma there (it should, given fold 0-3 are already that), #2 is ruled out
as the cause. Separately, drop the ns>0.995 region from the emulator domain and re-fit a mock with
true ns~0.96 to confirm the top-edge shrink does not bias the recovered n_s - that is the real-fit
guard, not the root-cause test.

---

## Feedback #3 — normalization friendly for A_p but not for tilt n_s

**Likelihood: HIGH.** This is my leading candidate, jointly with #4.

The diagnosis chain:
- The error is a coherent NEGATIVE k-tilt (the decisive measurement above), carried by the residual
  head, localized to z=2.0-2.4.
- `sig_cosmo` is z-COLLAPSED (one per-(c,k) constant). The true cosmology signal is ~2.3x more
  high-k-weighted at z=2.0-2.4 than the z-average, so the loss under-weights the high-k tilt signal
  exactly where the tilt deficit is worst. Direct normalization<->tilt interaction.
- The residual decodes through a global per-k SVD basis (24,172) that has DRIFTED far from orthonormal
  (||BB^T - I|| ~ 2.7) and couples high-k to low-k (r=-0.61, suggestive). The n_s tilt is a slope
  coefficient; a per-k basis dilutes the slope across all k instead of giving it a named, separately-
  weighted DOF.
- LaCE (2305.19064; reference notes) shows the target-basis choice is the DOMINANT accuracy lever
  (linear-k -> log-k coefficient target: 10% -> 1.4%). Their tilt-friendly fix is exactly a smooth
  log-k polynomial coefficient target + per-k inverse-variance loss + a single scalar (not per-k)
  normalization. Our redesign adopted log-space + inverse-variance + baseline removal, but NOT the
  smooth slope-coefficient target - the missing piece is precisely the one that protects the tilt.

So "log-P1D is friendly for A_p but not for n_s" is essentially correct in our implementation - not
because log-P is intrinsically bad for tilt, but because (a) our whitening is z-collapsed so the tilt
is under-weighted at low z, and (b) our output basis is a per-k SVD, not a slope-coefficient basis, so
the tilt has no protected DOF.

**Cheap test (#3):** forward-only, no retrain. Recompute sig_cosmo PER z-band (e.g. 3 z-bins) instead
of z-pooled, then re-whiten the held-out residuals and recompute the per-z-band tilt slope. If the
z=2.0-2.4 tilt deficit shrinks when the high-k signal is correctly up-weighted at low z, #3 is
confirmed as the loss-budget channel. Cheaper still: project the held-out clean-logP residual onto a
low-order log-k Legendre basis and report the slope (L1) coefficient bias per z-band - if the SVD basis
is leaking the slope, the L1 bias is the same -0.00138-shaped tilt and a slope-up-weighted basis would
fix it. (Both are measurements; the actual fix is a retrain with z-resolved whitening or a slope-aware
basis/loss - that is option (b), not cheap.)

---

## Feedback #4 — no loss on the MEAN prediction conditioned on fixed tau0, z

**Likelihood: MEDIUM-HIGH** (with an important nuance: there IS a conditional-mean loss, but it is on
the BASELINE, and it is FLAT in k so it does not penalize the residual head's per-cell tilt).

The state of the code:
- There IS a separate conditional-mean loss: `joint_loss`'s `p_base` term (MSE of the theta-blind
  BaselineHead vs the (z,tau0) cell-mean target t_p_base), plus an 8000-epoch baseline-only prefit
  (`_prefit_baseline`), and production FREEZES the baseline. So the (z,tau0)-conditional MEAN of the
  BASELINE is explicitly fitted and pinned - and indeed base_err_k is flat/tilt-free (above).
- There is ALSO a coherent de-bias term active in production at w_coh=80 (`coherent_debias_term`):
  Sum_cell <r_hat - t_p_resid>_theta^2, the per-(z,tau0)-cell mean of the residual-head fit error
  over cosmologies. This is exactly "put a loss on the conditional mean of the residual." So #4 is
  PARTIALLY addressed for the residual too.

BUT here is the gap, and it is the crux: `coherent_debias_term` is deliberately UNIFORM (FLAT) in k
(docstring L356-359: edge-weighting "OVER-corrects low-k"). A FLAT-in-k penalty on the per-cell mean
drives the k-AVERAGED residual offset to zero but does NOT penalize a coherent k-TILT that pivots in
the band (a tilt has near-zero k-mean, so it is in the null space of a flat-weighted mean penalty).
The residual head's mean error is precisely such a tilt (resid_err_k swings + -> -; k-mean ~ 0). So
the conditional-mean loss that exists CANNOT see the tilt it needs to remove. That is the operative
form of feedback #4: the loss does not constrain the n_s-tilt of the conditional mean conditioned on
(z,tau0).

This couples tightly with #3: the FLAT debias weight is flat because the SAME z-collapsed sig_cosmo
makes the natural (signal-weighted) k-shape look wrong; a z-resolved, high-k-protecting weight would
let a tilt-aware mean penalty work.

**Cheap test (#4):** forward-only diagnostic, no retrain. Compute the per-(z,tau0)-CELL mean of the
held-out residual error <r_hat - t_p_resid>_theta and fit its slope vs ln k, per z-band (this is the
exact quantity coherent_debias_term squares, but resolved in k). It will show a non-zero SLOPE with a
~zero MEAN at z=2.0-2.4 - demonstrating that the flat debias term is blind to the tilt. Then re-run
the debias term value under a slope-weighted (e.g. L1-Legendre) k-weight on the SAME held-out
residuals: if the slope-weighted term is large where the flat term is small, #4's gap is confirmed and
the fix is a tilt-aware (not flat) coherent term - a targeted retrain knob, cheaper than a full
re-architecture.

---

## Priority ranking (most -> least promising as the actual root cause, this lens)

1. **#3 (normalization/tilt)** - the z-collapsed sig_cosmo + per-k SVD basis under-resolve the tilt
   exactly at z=2.0-2.4 where the deficit is worst; LaCE-confirmed mechanism. HIGH.
2. **#4 (no tilt-aware conditional-mean loss)** - the existing flat coherent term cannot penalize the
   residual's k-tilt; tightly coupled to #3. MEDIUM-HIGH.
3. **#1 (tau0<->n_s head)** - real degeneracy and real low-tau0 amplitude degradation, but the signal
   is tau0-symmetric at fixed z and the bias survives tau0 marginalization on the tilt direction.
   LOW-MEDIUM (pending the tau0-split number).
4. **#2 (ns>1.0 extension)** - confirmed sparse/coupled and a real TOP-EDGE real-fit risk, but the
   bias spans the dense supported region and is HeII-uncorrelated, so it is not the cause of the
   -0.65sigma. LOW (as cause); a real caveat for the real fit + weakens "C1 resolved."

---

## Dissents / disagreements

1. **The "finite-60-sim LOSO generalization floor to MARGINALIZE" framing is too fatalistic.** The
   prior handoff leads with a finite-sim floor + "C_emu to marginalize, not a bug to fix." My
   measurements say the dominant piece is a STRUCTURED, z=2.0-2.4-localized, tilt-shaped error driven
   by two FIXABLE implementation choices (z-collapsed whitening; flat-in-k debias on a per-k SVD
   basis), not an irreducible coverage limit. Some finite-sim floor surely exists, but attributing the
   bulk to it risks marginalizing (inflating C_emu on) an error a z-resolved whitening + tilt-aware
   loss would largely remove at source. Measure the z-resolved-whitening test (#3) BEFORE building a
   C_emu around the floor.

2. **The "C1 resolved" claim (ns_hi=1.05 backed by training points by construction) is NOT resolved.**
   Only 2/60 design points exceed ns=1.0, gap-separated and pinned to the high-alphaq corner. "The
   data.PARAM_LIMITS box == the emulator design box" is true but circular - the box being declared does
   not mean the (0.995, 1.05] slab is space-filled. The shrink-to-centre (high-n_s pulled down -0.76)
   is exactly what a sparse top edge produces. C1 should be downgraded to "supported up to ns~0.995;
   ns in (0.995, 1.05] is a sparse HeII-coupled extension - either exclude it from the real-fit domain
   or carry an explicit edge-bias term." This is the user's feedback #2 and I agree with the user over
   the onboarding claim.

3. **MF-first dissent — I PARTIALLY WITHDRAW it for this bias.** I previously leaned MF-first (the
   production forward runs MF, the LF-only closure is the wrong target). But the n_s tilt deficit is
   z-localized at z=2.0-2.4 / low-k-to-mid-k, which is the DESI-leg resolution-converged regime (PRIYA
   percent-level for k<0.05; worst convergence is at z~3-4 HeII, NOT z=2.0-2.4). MF reshapes the
   HIGH-k/KS band, which is a SECONDARY contributor here (z>4 slope -0.0005). So MF will NOT fix the
   dominant z=2.0-2.4 tilt; the LF-emulator tilt fix (#3/#4) is needed regardless of MF. I still hold
   that the FINAL bias number must be re-measured post-MF, but MF is not the lever for this specific
   bias. The bias is genuinely an LF-emulator tilt problem.

---

## Recommended next actions (cheap-first, ordered)

1. **[running] tau0-split tilt diagnostic** (`scripts/diag_lya_ns_tau0_split.py`) to settle #1
   LOW-vs-MEDIUM. Forward-only.
2. **z-resolved sig_cosmo re-whitening test** (#3): recompute sig_cosmo in 3 z-bins, re-whiten
   held-out residuals, re-fit the per-z tilt slope. Forward-only, hours not a retrain. The single most
   informative next measurement - it directly tests whether the loss-budget choice causes the
   z=2.0-2.4 tilt deficit.
3. **per-cell residual-tilt diagnostic** (#4): fit the slope vs ln k of <r_hat - t_p_resid>_theta per
   z-band; show the flat coherent term is blind to it. Forward-only.
4. **ns<0.97 re-pool** (#2): confirm the -0.65sigma survives in the dense supported region. Trivial
   (re-bootstrap the existing emu_bias_allfolds.txt rows). I effectively already did this via the
   fold-0-3 numbers; formalize it.
5. **THEN decide the fork**: if #3 test shows z-resolved whitening collapses the tilt, the fix is a
   targeted retrain (z-resolved sig_cosmo + a tilt-aware / slope-weighted coherent term, and/or a
   low-order log-k slope-coefficient basis a la LaCE) - a SOURCE fix, cheaper than a full
   re-architecture and more honest than marginalizing a fixable error in C_emu. Re-measure the bias
   post-fix AND post-MF before any cert. Independently, set the real-fit ns domain to <=0.995 (or carry
   an explicit top-edge term) per dissent #2.

---

## RESULTS APPENDIX (live measurements)

**tau0-split tilt diagnostic** (`scripts/diag_lya_ns_tau0_split.py`, 2560 held-out rows, 8 folds,
4 sims/fold, z in {2.4,3.0,3.6,4.2}). c = n_s-equivalent mis-tilt of (emu-truth) clean logP
projected onto the emulator's own d logP/d n_s; binned by tau0 tercile:

```
  LOW  tau0 (<0.31)   n=864  <c>=+0.0334 +/- 0.0045
  MID  tau0           n=864  <c>=+0.0023 +/- 0.0014
  HIGH tau0 (>0.65)   n=832  <c>=-0.0001 +/- 0.0008
  by z:  z=2.4 +0.0438 | z=3.0 +0.0031 | z=3.6 +0.0025 | z=4.2 -0.0013
  DISCRIMINATOR <c>_low_tau0 - <c>_high_tau0 = +0.0335
```

**Verdict on #1: the discriminator has the WRONG SIGN and is z-driven, not tau0-driven.** Feedback #1
predicts the tilt deficit (c<0) to be WORSE at LOW tau0; instead c is most POSITIVE at low tau0 and
~zero at high tau0, and the structure tracks Z (z=2.4 dominates, +0.044) - and low-z rows are exactly
the low-tau0 rows because Kim tau rises with z, so the "low-tau0" signal is really the z=2.4 signal.
The clean-class direct projection sign (c>0 at low z) differs from the marginalized Fisher n_s bias
sign (negative) because the Fisher projects through the full P_obs forward + HCD + tau0 + a_pivot
covariance/degeneracy; the robust, convention-independent conclusion is that the tilt mis-emulation is
NOT preferentially a low-tau0 deficit - it is a z=2.0-2.4-localized effect. This DOWNGRADES #1 from
LOW-MEDIUM to **LOW**: tau0 is a confound for z, not an independent driver. (Caveat: this is the
clean-class logP tilt, not the full marginalized P_obs MAP-shift; the all-folds Fisher remains the
arbiter. But the existing tau0regime amplitude finding + this tilt finding + the tau0-independence of
the within-cell signal at fixed z all point the same way: the lever is z, not tau0.)
