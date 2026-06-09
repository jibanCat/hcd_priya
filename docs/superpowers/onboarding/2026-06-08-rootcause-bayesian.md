# Root-cause review (Bayesian / PPL / identifiability lens) — 2026-06-08

Charge: assess the four user feedback points as candidate root causes for the **−0.65σ coherent
n_s under-prediction** (`emu_bias_allfolds.txt`: mean −0.646σ, 95% CI [−0.85,−0.43], t=−5.84, 50/60
sims negative, A_p +0.03σ ~ 0). Lens = inference geometry, calibration/coverage, loss/likelihood
mis-specification, and whether this is a **bias-to-fix** or **noise-to-marginalize**.

All numbers below I reproduced from `figures/analysis/04_emulator/emu_bias_allfolds.txt`,
`emu_bias_allfolds`/`embias_*` npz, `emulator_params.json`, and the code on `phase2c-likelihood`.

---

## 0. The single most important thing I found that the prior framing missed

**The LOSO folds are contiguous sorted-n_s blocks.** Verified directly:

| fold | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| mean n_s_unit | 0.060 | 0.176 | 0.292 | 0.403 | 0.462 | 0.566 | 0.666 | 0.797 |
| mean n_s bias [σ] | −0.47 | **+0.47** | −0.52 | −0.84 | −1.04 | −1.56 | −0.75 | −0.63 |

Global mean n_s_unit = 0.415. The sign of the per-fold n_s bias tracks `(fold_mean_ns − global_mean_ns)`
almost perfectly: fold 1 (below the global mean → the held-out sims sit *below* their own training set)
is the **only positive** fold; every fold above the global mean is negative. This is the **textbook
regression-to-the-training-mean signature of a finite-sample interpolator under a covariate-blocked
split** — and `scripts/embias_stats_perfold.py` was written to test exactly this hypothesis
("shrinkage pulls each held-out fold's prediction toward the TRAINING mean … because the folds are
contiguous ns-blocks").

The consequence for the four feedback points is large and is my central dissent:

- The measured −0.65σ is **partly an artifact of the n_s-blocked LOSO partition**, not a property the
  real-data fit will see. Holding out a whole contiguous n_s slice removes the emulator's ability to
  interpolate that slice's tilt from straddling neighbours; on the real fit the truth is a single point
  with all 60 sims surrounding it. A **random** (or n_s-stratified) re-fold would shrink the coherent
  pull substantially if the dominant mechanism is shrink-to-block-mean. **This is the cheapest, most
  decisive test in the whole investigation and it has not been run.** It also re-grades feedback #2:
  the n_s>1.0 coverage hole cannot be the driver of a −0.65σ pull that is *strongest in the box centre*.

- The bias is **not monotone in n_s_unit** (which a pure shrink-to-*box*-centre or a high-edge coverage
  hole would require). By tercile: low [0,0.34) **−0.18σ**, mid [0.34,0.62) **−1.13σ**, high [0.62,1.0]
  **−0.69σ**. `corr(bias, |n_s_unit−0.5|) = +0.42` — the bias is **largest near the centre of the box**,
  the *opposite* of an edge/coverage-hole signature. This is what shrink-to-*fold*-train-mean predicts
  (the centre folds have the most training mass on both sides → the most regression pull when their
  slice is removed), not what an unsupported high-n_s extrapolation predicts.

- The bias survives dropping the worst folds: drop fold 5 → −0.53σ (t=−4.7); drop folds 4+5 → −0.45σ
  (t=−3.65). So there IS a real coherent residual under-prediction beyond the block artifact — but its
  honest magnitude for a representative fold is more like **−0.4 to −0.5σ**, and the −0.65σ/−1.56σ
  headline numbers are inflated by the block structure + fold-5.

---

## 1. Feedback #1 — τ₀/n_s degeneracy through the two-stage head — **MEDIUM**

The architecture is exactly as the user describes (`model.py`): `BaselineHead(z,τ₀)` is θ-blind and
emits the (z,τ₀)-conditional mean m̂; `HeadB(latent,τ₀)` emits the σ_cosmo-whitened residual r̂; all
cosmology flows through r̂ (`predict.py:38`, `∂logP/∂θ = sig_cosmo·∂r̂/∂θ`). The user's worry — "good
prediction at high τ₀ but not low τ₀ → biased n_s" via the strong τ₀–n_s degeneracy — has three
verifiable pieces, and they line up:

1. **The degeneracy is real and strong.** The reference agent's weighted-Pearson over 7 PRIYA cobaya
   chains gives `corr(n_s,τ₀) ∈ [−0.56,−0.78]`, median −0.66. So any coherent τ₀-direction emulator
   error gets re-expressed ~2/3 into n_s.
2. **Accuracy IS worse at low τ₀.** `embias_arch_tau0regime.npz` `bandmean` is monotone in the α-rung:
   α=0.69 → **−27.5%**, α=1.30 → **−7.6%** (clean low-k coherent frac). The large magnitudes are
   fold-0-only + include out-of-range bins, so they are inflated — but the **monotone direction (worse
   at low τ₀)** is the load-bearing claim and it holds. The trustworthy in-range becker-row decomposition
   is base-head +0.024% / **residual-head −0.136%**.
3. **The carrier is the residual head, not the baseline.** Because production freezes the baseline
   (`run_loso_sweep.py`: `freeze_baseline` via prefit), ∂P/∂θ comes only from r̂, so any θ-direction
   (incl. tilt) bias MUST live in HeadB. The decomposition confirms it.

**Why MEDIUM, not HIGH.** This is a *channel*, not clearly the *cause*. The crucial gap: **no
diagnostic isolates the n_s-TILT bias by τ₀ regime.** `tau0regime` measures the low-k *amplitude*
(A_p-flavoured) error vs τ₀; it does not measure whether the *k-slope* (n_s) error is τ₀-dependent. And
the Fisher MAP-shift in `diag_emu_bias_allfolds.py` already includes τ₀ in the joint information matrix
`(F+P)^{-1}JᵀC⁻¹ΔP` (Jacobian wrt theta9, τ₀, a_pivot) — so the −0.65σ is the n_s shift *after*
marginalizing τ₀. If the mechanism were purely "τ₀-error leaks into n_s," the τ₀ marginalization in that
Fisher would have re-absorbed a large chunk of it, yet a coherent n_s residual remains. So feedback #1
is a contributing channel whose magnitude is unquantified in the tilt direction.

**Cheap test (no retrain):** add a τ₀-resolved *tilt* decomposition to the existing `tau0regime` script —
for each held-out sim, regress the per-z log-P residual `(pred−true)` on log-k to get a per-(sim,z,τ₀-rung)
**slope error**, then bin that slope error by τ₀-rung. If the slope (tilt) error is flat in τ₀ but the
amplitude error is τ₀-sloped, feedback #1 is largely ruled out as the *n_s* driver (it would only bias
A_p, which is ~0). If the slope error itself worsens at low τ₀, #1 is confirmed as a genuine n_s channel.
Forward-only, ~minutes, reuses `embias_arch_tau0regime.py` machinery.

---

## 2. Feedback #2 — extended prior / n_s>1.0 not on a Latin hypercube — **LOW** (as the driver of THIS bias) / the coverage concern is **real but separate**

The user is **factually right** and the reference agents confirm it: PRIYA's original LH design caps
n_P at 0.995 (Bird+2023 Table 2); the 0.995→1.05 extension (Fernandez+2024) added only 12 LF sims
placed jointly in the high-α_q corner. I verified the production grid directly: of 60 design points,
**only 2 have n_s>1.0** (n_s_unit 0.875, 0.958, both in fold 7) and 3 have n_s>0.995, with a clear gap
(0.9979 → 1.0188 → 1.0396). A uniform LH over [0.8,1.05] would place **~12** above 1.0. So the n_s>1.0
slab is sparse, BO-/corner-placed, **not space-filling** — and the meta "C1 resolved" claim ("ns_hi 1.05
backed by training points by construction") is **wrong as a coverage statement**: being inside the box's
unit-cube hull is not the same as being supported by space-filling training density.

**But it is LOW as the root cause of the −0.65σ pull**, for three independent reasons:

1. **Direction.** The data/closure pulls n_s **down**, toward the dense interior [0.8,0.995], *away* from
   the sparse edge. A coverage hole at n_s>1.0 hurts predictions queried *in* (1.0,1.05); it does not
   manufacture a coherent downward pull on sims sitting at n_s ≈ 0.9.
2. **Shape.** The bias peaks in the **box centre** (tercile mid −1.13σ vs high −0.69σ; `corr(bias,
   |n_s_unit−0.5|)=+0.42`). A high-edge coverage hole would make the bias **worst at high n_s** —
   the opposite of what is observed.
3. **The two n_s>1.0 sims behave normally.** In `emu_bias_allfolds.txt` the n_s_unit=0.875 sim has
   n_s bias −0.39σ and the 0.958 sim −0.93σ — within the fold-7 scatter, not catastrophic outliers as
   a true extrapolation hole would produce.

So #2 is a **legitimate coverage caveat for the real-data fit** (if the true n_s lands ≥0.995 the
emulator is extrapolating off a sparse ridge and the posterior there is untrustworthy — this should
gate the blinded fit) but it is **not the mechanism behind the LOSO −0.65σ**.

**Cheap test (no retrain):** two parts. (a) Re-run `diag_emu_bias_allfolds.py` with the pooled mean
computed *excluding the 3 sims with n_s>0.995* — the −0.65σ will be essentially unchanged, confirming
the edge is not the driver. (b) For the coverage caveat itself: re-run the bias measurement restricted
to held-out sims with n_s_unit>0.85 (the sparse slab) vs <0.85, and compute the emulator LOO error
restricted to the n_s>1.0 design points — quantifies the *separate* edge-trust problem the real fit
must respect. Both forward-only.

---

## 3. Feedback #3 — log-P normalization friendly for A_p but maybe not for tilt — **MEDIUM-HIGH**

This is, through the inference lens, the most *structurally plausible mechanism* for a **tilt-specific,
amplitude-neutral** bias — which is exactly the observed signature (n_s biased, A_p ~ 0). The redesign
(`fit_baseline_residual_norm`, `data.py:270`) does the right things (in-network θ-blind baseline +
σ_cosmo-whitened residual = Kennedy–O'Hagan structured mean), but two normalization choices specifically
under-protect the **tilt direction**:

1. **σ_cosmo is a per-(c,k) scalar pooled over cells**, and the residual target is
   `t_p_resid = (logP − m_cell)/σ_cosmo`. The cosmology signal that distinguishes n_s from A_p is a
   *correlated-across-k tilt* of ~0.4% of the per-k variance. Whitening per-k independently (one σ_cosmo
   per k) treats each k as an independent unit-variance target — it does **not** give the *coherent
   k-slope mode* (the n_s direction) any protected weight. The MSE then spends its budget on the
   per-k amplitude (A_p-aligned, which dominates the residual variance) and under-resolves the slope.
   This is precisely the LaCE finding the reference agent surfaced: their ablation shows the
   **target basis** (per-k → smooth log-k polynomial *coefficients*) is the dominant accuracy lever
   (10%→1.4%), because it makes the slope a *named, separately-weighted DOF* (α₁ ≈ dlogP/dlogk) instead
   of diluting it across independently-whitened k bins. We adopted log-space + inverse-variance +
   baseline-removal but **not** the smooth-coefficient target — and that missing piece is the one that
   makes tilt first-class.

2. **The `coherent_debias_term` (the production w_coh=80 regularizer) flattens the wrong thing.** It
   drives `⟨r̂ − t_p_resid⟩_θ → 0` *per (z,τ₀) cell, UNIFORM in k* (`model.py:342`). This pins the
   per-cell **mean over cosmologies** (which is 0 by construction) — i.e. it removes a θ-*independent*
   coherent offset. It does **not** constrain the θ-*dependent* tilt response (the n_s direction is a
   *cosmology-varying* slope, which averages out of the per-cell θ-mean). So the one explicit
   coherent-flattening term in the loss is, by construction, **blind to the n_s bias**. The comment even
   says edge-weighting it "over-corrects low-k" — i.e. the team already found this term cannot be aimed
   at the slope without breaking. That is the loss-side fingerprint of a tilt the normalization buries.

**Why MEDIUM-HIGH not HIGH:** I cannot, from disk artifacts alone, prove the per-k whitening (vs a
log-k coefficient target) is *the* cause without an ablation. But it is the only one of the four with a
*mechanism that is intrinsically tilt-specific and amplitude-neutral*, matching the A_p≈0 / n_s≠0
signature better than #1 (which would also bias A_p) or #2 (wrong direction/shape).

**Cheap test (no retrain):** project the held-out residual `(logP_pred − logP_true)` onto a low-order
log-k basis {1, log k, (log k)²} per (sim,z), and report the **mean coefficient on the `log k` term**
across the 60 honestly-held-out sims. If the linear-in-log-k (slope) coefficient is coherently nonzero
while the constant (amplitude) coefficient is ~0, the bias **is** a tilt-mode the per-k normalization
failed to resolve → #3 confirmed as the channel. This is a pure post-processing of the existing forward
predictions, ~minutes, and it directly tests the LaCE hypothesis. (Bonus: compute the *whitening
variance of the slope-coefficient* under the current per-k σ_cosmo — if it is ≪1 you've shown the loss
under-weights the slope.)

---

## 4. Feedback #4 — no explicit loss on the conditional mean for fixed (τ₀,z) — **MEDIUM** (partially refuted as stated, but a real residual concern)

As literally stated ("you don't put a loss on the mean prediction conditioned on τ₀,z") this is
**refuted for the production recipe**: there IS a separate named `p_base` loss (`joint_loss`,
`model.py:437`) that MSE-fits `BaselineHead` to the (z,τ₀) cell-mean target `t_p_base`, AND a dedicated
8000-epoch `_prefit_baseline` (`train.py:165`) on the ~306-cell table, AND the production default
**freezes** the baseline so the joint loop cannot pull it off (`train.py:370`). So the conditional mean
is explicitly fit and pinned.

**But the deeper Bayesian point the user is gesturing at survives, and it interacts with #1/#3:**

1. **The design PINS the conditional mean, it does not GUARANTEE it is correct.** Under LOSO the
   baseline is fit to the **train-cell-mean** (`_baseline_cell_table`, `train.py:137`: the held-out sim
   is excluded). The frozen baseline therefore carries `m_cell^train`, and HeadB's residual must supply
   the *entire* deviation of the held-out cosmology from the train-cell-mean. With the folds being
   contiguous n_s blocks (§0), the train-cell-mean is **systematically offset in the tilt direction**
   from the held-out truth — so the residual head is being asked to predict a deviation whose *mean over
   the held-out slice is nonzero*, and the per-sim MSE (dominated by within-cell CV scatter) does not
   penalize a coherent slope offset. **This is the loss-mis-specification that actually bites:** there
   is no term that says "the θ-*conditional* tilt response must be unbiased on held-out cosmologies."
   The `coherent_debias_term` is the closest, but (§3.2) it is blind to the tilt by construction.

2. **The non-default `freeze_baseline=False` path** *can* let the joint fine-tune drift the baseline
   (the docstring warns it "DRIFTS off its floor"), which would inject an uncontrolled (z,τ₀)-mean error
   straight into the A_p/n_s degeneracy — but production does not use it, so this is a latent rather than
   active risk.

**Why MEDIUM:** the explicit conditional-mean loss exists (refutes the literal claim), but the *coupling
between the LOSO block structure and the absence of a θ-conditional-tilt-unbiasedness term* is a genuine
loss-side contributor, and it is the same root as #3 viewed from the loss rather than the target.

**Cheap test (no retrain):** for each held-out sim, compute the baseline-head error
`m̂(z,τ₀) − m_cell^heldout` (the *held-out* cell-mean, recomputed including the held-out sim) and the
residual-head error separately, decomposed into amplitude vs slope (log-k regression, as in #3). If the
baseline-head's *slope* error is coherently nonzero on held-out cells (because it was fit to the
block-offset train mean), #4 is confirmed as the channel; if the baseline slope error is ~0 and the
slope bias is all in HeadB, the issue is purely the residual head's loss (still #3/#4-flavoured but
locates it in HeadB). This generalizes the existing `embias_arch_tau0regime.py` becker-row decomposition
to all folds and to the *slope* (not just low-k amplitude).

---

## 5. Priority ranking (most → least likely as the ROOT cause, this lens)

1. **#3 (normalization buries the tilt)** — the only mechanism intrinsically tilt-specific +
   amplitude-neutral, matching A_p≈0 / n_s≠0; loss-side fingerprint (the w_coh term is tilt-blind);
   LaCE corroborates the basis/target lever.
2. **#4 (no θ-conditional-tilt-unbiasedness term, × the n_s-blocked LOSO)** — the loss-side twin of #3;
   the block structure makes the residual head's coherent-slope d.o.f. systematically offset and
   un-penalized. Strongly coupled to #3 and to the §0 fold artifact.
3. **#1 (τ₀/n_s degeneracy channel)** — real and strong degeneracy, accuracy worse at low τ₀, residual
   head is the carrier; but unquantified in the *tilt* direction and the Fisher already marginalizes τ₀.
4. **#2 (n_s>1.0 coverage hole)** — factually confirmed and a real *real-data-fit* caveat, but wrong
   direction and wrong box-shape to drive THIS LOSO bias.

**Overarching:** all four are downstream of, or confounded by, the **n_s-blocked LOSO partition (§0)**,
which I rank as the single highest-value thing to test before trusting any of the four as "the cause."

---

## 6. Dissents

- **I dissent from the "finite-60-sim LOSO generalization floor → marginalize via correlated C_emu"
  framing as currently justified.** The fold-structure + "shrink-to-centre" that the handoff reads as a
  *floor* is, on inspection, **shrink-to-FOLD-TRAIN-MEAN under an n_s-blocked split** — a measurement
  artifact of the contiguous-block LOSO, not (only) an irreducible data limit. A floor you marginalize as
  zero-mean correlated C_emu must be **genuinely random/ensemble-symmetric**; a deterministic
  shrink-toward-the-train-mean that *flips sign with the fold's position relative to the global mean* is
  **not zero-mean** — it is a structured bias. Marginalizing it as a zero-mean covariance term will
  **under-cover** in the n_s direction on the real fit (the real fit is not n_s-blocked; its bias, if any,
  is whatever the *single* true-point interpolation gives, which the block-LOSO over-states). The
  bias-vs-noise question is **not yet answerable** because the block confound has not been removed. This
  is the same warning I raised in my onboarding R1 (option C is only sound if the amplitude is set to the
  true uncertainty range, not the in-sample residual) — here it applies to whether C_emu should carry
  this mode *at all*.

- **I dissent from "C1 resolved."** The claim that n_s_hi=1.05 is "backed by training points by
  construction" conflates *hull membership* with *training density*. Only 2/60 points exceed 1.0 (vs ~12
  under uniform LH), gap-separated, corner-placed. C1 is **not resolved**; the n_s>0.995 region is a
  sparse extrapolation ridge and the blinded real fit must treat any posterior mass there as untrusted
  (gate, or a prior cut at 0.995, pending an explicit edge-LOO accuracy measurement). This *partially
  agrees* with the user's feedback #2 — but I separate the (valid) coverage caveat from the (invalid)
  claim that #2 explains the −0.65σ.

- **Mild dissent from the walkthrough's "shrink-to-centre (corr −0.35)".** The −0.35 corr with n_s_unit
  is real but the bias is **non-monotone** (worst in the centre, `corr(bias,|n_s−0.5|)=+0.42`), so
  "shrink toward the box centre" is the wrong picture; "shrink toward each fold's train mean" fits all
  the signs.

---

## 7. Recommended next actions (cheap-first, forward-only — no retrain)

1. **Re-fold the LOSO with a RANDOM (or n_s-stratified) split and re-run `diag_emu_bias_allfolds.py`.**
   This is the decisive bias-vs-artifact test (§0). If the coherent n_s pull drops to ≲0.2σ, the −0.65σ
   was largely the block confound and the "floor to marginalize" framing collapses. If it persists at
   ~0.4σ, it is a genuine residual to address (then #3/#4). *(Needs 8 re-trained fold emulators on a new
   split — this is the one "retrain" I recommend, but it is a re-fold not a new architecture, and it is
   the highest-information single action. If a full re-fold is too costly, the cheaper surrogate is the
   slope-decomposition tests below, which can run on the EXISTING fold emulators.)*
2. **Log-k slope decomposition of the held-out residual (test for #3), on the existing fold emulators.**
   Project `(logP_pred−logP_true)` onto {1, log k, (log k)²} per (sim,z); report the coherent mean
   `log k` coefficient over all 60 sims. Confirms/refutes "the bias is a tilt mode the per-k
   normalization buried." Pure post-processing.
3. **Baseline-vs-residual slope split (test for #4), all folds.** Recompute the *held-out* cell-mean and
   decompose the slope error into baseline-head vs HeadB contributions. Locates the tilt bias in the
   structured mean (fit to block-offset train mean) vs the residual head.
4. **τ₀-resolved TILT decomposition (test for #1).** Bin the per-(sim,z) slope error by τ₀-rung. Tells
   whether the τ₀–n_s degeneracy is an *active* n_s channel or only an amplitude (A_p) one.
5. **Edge-trust quantification (test for #2's real concern).** Restrict the bias measurement to
   n_s_unit>0.85 and compute LOO error on the n_s>1.0 design points; feed the result into a blinded-fit
   gate (or a prior cut at n_s=0.995). This addresses the legitimate coverage caveat independently of the
   −0.65σ.
6. **Only after 1–4:** decide bias-to-fix (tilt-aware target/loss — the LaCE log-k-coefficient target, or
   a slope-direction up-weight in σ_cosmo) vs noise-to-marginalize (correlated C_emu). Do **not** build a
   zero-mean correlated C_emu for this mode until §0/#3/#4 establish it is ensemble-symmetric rather than
   a structured tilt bias — a structured bias marginalized as zero-mean noise under-covers n_s.

The right diagnostic to distinguish a coverage hole (#2) from a generalization floor: a **random-fold
re-measurement** (action 1) collapses a block artifact but leaves a true coverage hole intact; a
**per-region LOO error vs training density** map separates "sparse-edge extrapolation error" (rises with
distance from training mass, localized at n_s>0.995) from "uniform interior interpolation floor" (flat in
n_s). The current evidence (centre-peaked, sign-flipping-with-fold) reads as **neither a clean coverage
hole nor a clean floor — it is the block-split shrink artifact plus a genuine tilt-resolution residual
(#3/#4)**.
