# Bayesian / PPL / inference checkpoint — low-k A_p bias investigation — 2026-06-06

Lens = the inference statistics: the Fisher MAP-shift bias as a coverage proxy, the FULL−EMU
differential, the z-accumulation reconciliation, the leakage correlation, and what object actually
marginalizes a z-structured emulator mode. All numbers re-run in the project env (CPU, x64) against
the two committed scripts. I tried to break the conclusions; some held, the central reconciliation
did not survive scrutiny.

## Verdict (one line)

**GO-WITH-CHANGES.** The slope-prior settlement (TL;DR #1) is statistically sound and I reproduced it
exactly; the relocation of risk onto a correlated C_emu is the right *direction* — but the headline
"z-COHERENT, ×18.6 z-accumulation" mechanism (§3) is **mis-derived** (it is a force-share artifact, the
mode is measurably z-NOISY not z-coherent), the +0.62σ and r=−0.61 are **n=8 estimates not
significantly different from zero**, and the load-bearing reconciliation is **not reproduced by any
committed script**. Fix the mechanism story and the significance caveats before this drives the C_emu design.

---

## Claims I VERIFIED (re-ran or re-derived) vs claims I could not check

**VERIFIED — held up exactly:**

1. **FULL−EMU slope/anchor bias** (doc §1 table, spec §8). Transcribed the 8 per-sim FULL/EMU values
   from `legb_slope_prior_rerun.txt:8-15` and differenced: A_p mean −0.0794, max 0.143, RMS 0.0857;
   n_s mean −0.0271, max 0.168, RMS 0.1116. Matches the doc to 3 digits. The **differential logic is
   correct**: `bias_FULL − bias_EMU = Cpost·Jᵀ C⁻¹ (P_exact − P0)` (same J, same Cinv, same Cpost in
   both `gvec`/`gvec_emu`, rerun :226-227, :247-248) ⇒ the shared emulator-error force cancels, leaving
   exactly the (sim-shape − lit-`g_fixed`) + slope/anchor-prior response. This is a *valid* isolation of
   the slope/anchor's own contribution. **TL;DR #1 is sound.** Width-insensitivity (lit/edge/2× rows
   18-20 identical to 3 digits) confirms the slope is not the driver.

2. **The ×18.6 ratio reproduces** — but see the §"Errors" below for what it actually is. My independent
   re-implementation of PART B with the force masked to z=3 (keeping the full-z F/σ): FULL-z A_p
   = +0.617σ (matches PART B), z=3-only = +0.033σ, **ratio = 18.6**. So the *arithmetic* in the doc is
   real and reproducible. The *interpretation* is wrong.

3. **The Phase-2 0.067σ is a z=3-RESTRICTED, diagonal-cov Fisher** (`diag_ap_fisher_bias.py:322,334`:
   "fiducial=cube centre, **z=3**", "DESI-DR1-like **diagonal** per-mode covariance"). Confirmed it is a
   different normalization from the reconciliation's "single-z" (see Errors #1).

4. **C_emu is block-diagonal in z** (`data_likelihood.py:321-322, 360-367`): per-z 4×4 cross-class
   blocks placed on the flat diagonal; legs block-diagonal (DESI⊥KS, :407). **No cross-z term exists.**
   This is the structural fact the fork turns on, and the doc states it correctly (§3 line 155).

5. **Basis non-orthonormality** ‖BBᵀ−I‖≈2.68 (`emu_lowk_investigation.txt:1`) — reproduced from the
   header; the oblique projector handling (`diag_emu_lowk_investigation.py:60-62`) is correct.

**COULD NOT fully check / not reproducible:**

- The **z-accumulation reconciliation numbers (0.033, 18.6, 1.26, the "ruler" table §3) are produced by
  NO committed script.** `grep` for `0.033|18.6|1.26|single.z` across `scripts/` returns nothing in the
  two diagnostic scripts. They live only in prose (doc §3, spec §8). I had to *re-derive* the 18.6
  myself to check it. A load-bearing headline with no reproducible artifact is a governance gap.
- NUTS coverage (the actual arbiter) — not run; all numbers are Fisher MAP-shift. The doc states this
  (§5) — fair, but it bounds how much weight the "0.6σ on C_emu" conclusion can carry.

---

## Errors / overclaims / things the doc gets wrong (specific)

### E1 (MAJOR) — the z-accumulation "reconciliation" compares two non-comparable quantities; ×18.6 is a force-share artifact, not coherent accumulation.

The doc's sharpest claim (§3, spec §8 lines 206-210): "+0.033σ at single z=3 → +0.62σ full-z ⇒ ×18.6,
because the per-z error is z-COHERENT and sums instead of averaging down." Three problems, two fatal:

**(a) The "single-z +0.033σ" is not a z=3 fit — it uses the FULL-z posterior width in the denominator.**
My reproduction (matching the doc's 18.6 exactly) computes single-z by masking the *force* `g` to z=3
while keeping the full-z `F`/σ_Ap. So 0.033σ = "what fraction of the full-fit A_p force comes from the
z=3 residual," with σ_Ap set by all 25 z-slices. The Phase-2 0.067σ
(`diag_ap_fisher_bias.py:322,334`) is a **z=3-restricted, diagonal-cov** Fisher where σ_Ap is set by
z=3 information alone. **Different denominators ⇒ different objects.** "0.033 ∼ consistent with 0.067"
(doc §3 line 150) is a magnitude coincidence, not a derived equivalence. The reconciliation's logical
chain — "Phase-2 saw the single-z number, we see the summed number, same physics" — does not hold,
because the two single-z numbers are normalized differently.

**(b) The mode is NOT z-coherent. Measured |mean_z|/std_z = 0.45 (mean over 8 sims), mostly < 1.** I
measured the per-z low-k clean emulator error directly (std-log). Per-sim ratios: 0.06, 0.38, 0.64,
1.56, 0.17, 0.12, 0.39, 0.27. A z-coherent mode (the doc's claim "biased the same way at every z",
§3 line 150) requires ratio ≫ 1; the signs flip across z (sim 0: − − − − + + + − − + + + −). **The
data contradict the stated mechanism.** A z-noisy error averages *down* over z, it does not sum up.

**(c) The arithmetic that produces 18.6 is the trivial force-share ratio.** z=3 appears in both legs
(2 of 25 z-slices); its share of a force spread over ~19 effective bins is ~1/18.6 = 5.4% (naive 2/25 =
8%). So ×18.6 ≈ N_eff, the number of z-bins contributing to the force sum — **regardless of
coherence**. A genuinely z-coherent mode would scale the *per-σ* bias by √N≈4 (numerator sums ∝N,
σ_Ap shrinks ∝1/√N, F grows ∝N → net √N), NOT ×N. The fact that the observed ratio matches ~N rather
than √N is itself evidence the "single-z" number is a force-share fraction (which scales as N by
construction), not a coherent-accumulation factor. **The doc reads a force-decomposition identity as a
physical accumulation mechanism.**

Net: the +0.62σ full-fit number is real (I reproduced it), but the *story for why it is large* —
"z-coherent summation, reconciled with Phase-2" — is wrong. The honest statement is: "the full-fit A_p
force receives contributions from all ~19 effective z-bins; the per-z residual is z-noisy; Phase-2's
z=3 number simply never summed the other 18 bins." That is still a real effect, but it is **not**
coherent accumulation and it is **not** reconciled-by-equivalence with 0.067σ.

### E2 (MAJOR) — the headline +0.62σ is an n=8 mean not significantly different from 0.

The EMU-only A_p bias (rerun EMU column): mean +0.528σ, sd 0.942, SE 0.333, **t=1.58, p=0.16, 95% CI
[−0.26, +1.32]σ**. The doc states "a coherent +0.6σ low-k A_p bias" as established fact (TL;DR #2,
abstract). With n=8 and ~0.9σ per-sim scatter, the mean's CI nearly reaches zero. The conclusion that
"the load-bearing risk moved onto C_emu (~0.6σ)" rests on a point estimate that is ~1.6σ from null.
Additionally, the "+0.62" is quoted from two different configs interchangeably: PART B (sim-shape
forward) gives +0.617; rerun EMU (lit-pivot amp) gives +0.527. Minor, but they are not the same number.

### E3 (MODERATE) — the leakage correlation r=−0.61 (n=8) is not statistically significant.

Doc §2 PART D treats r=−0.61 as evidence "the leakage channel is real" (line 110). For n=8: t=−1.90,
**two-sided p=0.107, Fisher 95% CI on r = [−0.92, +0.16]** — crosses zero. The hypothesis (global basis
couples bands) may well be true and the ‖BBᵀ−I‖≈2.7 + emu≫floor evidence is independent support, but
**r=−0.61 alone does not establish it at 5%**, and a single influential sim (the ns0.813 outlier with
its −2.7σ n_s) could move it. The doc should down-weight the correlation to "suggestive" and lean on
the basis-orthonormality + rank-floor evidence instead, which is stronger.

### E4 (MINOR) — n_s gate tested only at the box edge.

All 8 fold-0 held-out sims have n_s_unit ∈ [0.013, 0.117] — clustered at the LOW edge of the [0,1] box.
The n_s "max 0.168 < 0.2 gate" (TL;DR, spec §8) is therefore certified only at an edge region where the
Fisher MAP-shift is least reliable as a posterior-MEAN proxy (the doc itself flags a "known n_s
box-edge effect", spec :194). The interior spot-checks (n_s_u 0.44–0.53) are reassuring but
emulator-IN-SAMPLE, so they cannot probe the emulator-error × n_s interaction. The gate is plausible
but not cleanly demonstrated for interior n_s on held-out sims.

### E5 (NOTE, not an error) — the MAP-shift = posterior-mean equivalence is assumed, not stated as a limit.

`bias = (F+P)⁻¹ Jᵀ C⁻¹ ΔP` is the correct MAP shift for a Gaussian-likelihood × Gaussian-prior model,
EXACT in ΔP (only J linearized). It equals the posterior-**mean** shift (the coverage quantity) only in
the Gaussian-interior limit. Here θ9 is a uniform box (auto-bijector), DLA is softplus (pinned here so
moot), and n_s sits at a box edge (E4). So every "0.xσ bias" is a coverage *surrogate*. The doc's §5
caveat covers this for "Fisher not NUTS," which is adequate — flagging it so the C_emu design doesn't
treat 0.6σ as a coverage number.

---

## My lens's take on the fork: correlated-in-k/z C_emu vs targeted emulator retrain

**As a marginalization problem, a correlated-in-k/z C_emu is the right *kind* of object — but ONLY if
it is low-rank and z-structured, and ONLY if the emulator mode is NOT parallel to the A_p response. I
verified both halves of this, and the second is an unaddressed risk.**

I ran the toy explicitly:
- **A per-z-DIAGONAL C_emu FAILS** (the whole point). To suppress a force-summed coherent bias from
  +0.75σ → ~0.07σ you must inflate the per-z variance to c≈10×σ_data at *every* z — which throws away
  essentially all low-k information (σ_Ap balloons). A diagonal inflation cannot tell a coherent mode
  from noise; it just degrades everything. So the current block-diagonal-in-z C_emu
  (`data_likelihood.py:321`) is genuinely the wrong tool, confirming the doc's §3 line 155 — even though
  the mechanism story (E1) is wrong, this conclusion is right for a *different* reason (force-summed, not
  coherent).
- **A RANK-1 C_emu aligned with the z-structure of the mode PROJECTS IT OUT** at bounded cost — *iff*
  the mode direction u differs from the A_p response direction J. In my toy where I forced J ∝ u (A_p
  response z-coherent too), projecting out u also kills the A_p signal: σ_Ap blows up (0.027 → 2.67 as
  τ→∞). **This is the critical caveat the doc misses entirely:** if the coherent emulator residual lies
  along the A_p response direction in (k,z) space, no covariance term can separate them — marginalizing
  the mode marginalizes A_p. Whether the C_emu route even works depends on the *angle* between the
  emulator mode and ∂P/∂A_p, which nobody has measured.

**Recommendation: C_emu route (a), with a mandatory prerequisite measurement.** Reasons:
1. It is the robust, no-retrain path and the marginalization is principled IF structured correctly.
2. The retrain (route b) is uncertain payoff (the doc concedes this) and the rank-floor evidence
   (emu 6–18× floor) shows headroom but not that a retrain *would* remove the *coherent* part — the
   coherent mean is small per-z (E1b) and may be irreducible sampling, not trainable.
3. BUT: **before building the C_emu, measure the overlap between the emulator coherent mode and the A_p
   Fisher direction.** If overlap is high, the C_emu will inflate σ_Ap without removing bias-per-σ (it
   marginalizes the thing you want to measure). The required structure is a **low-rank (rank 1–3)
   term whose (k,z) eigenvectors are the measured emulator residual modes**, NOT a diagonal inflation
   and NOT a per-z-independent block. A nuisance-template / mode-projection marginalization (add a
   nuisance amplitude × the measured residual template to the forward, with a wide prior) is
   *mathematically equivalent* to a rank-1 C_emu and is more transparent about exactly which mode is
   being marginalized — I'd prefer it for auditability.

In short: the fork answer is "structured low-rank C_emu (or its equivalent template-marginalization),"
the doc is directionally right, but the doc has not established that this mode is separable from A_p,
which is the precondition for the route to work at all.

---

## Additional risks or missing checks

- **R1 — σ_anchor > σ_α inversion (carried over from checkpoint-1 F2, still unresolved and now
  load-bearing).** The rerun uses σ_anchor KS=0.27, DESI=0.12 (`diag_legb_slope_prior_rerun.py:66`) vs
  the global LLS incidence prior σ_α=0.15 (the tightest, most cosmology-relevant prior). The doc credits
  "the per-leg amplitude anchor is the real absorber" (TL;DR #1) — but an anchor *wider* than the global
  prior can absorb anything, by partially un-doing the global incidence prior's identifiability hold on
  A_p. The "slope-prior is SAFE because the anchor absorbs it" conclusion is only benign if the anchor
  is not itself eroding A_p. This needs the σ_anchor < σ_α check (matched units) the prior checkpoint
  demanded; it was NOT done in this rerun.
- **R2 — no reproducible artifact for §3.** The entire reconciliation (0.033/18.6/1.26 table) is prose
  only. Even setting aside E1, a checkpoint headline must regenerate from a committed script. Add one.
- **R3 — the "1.26× correlated-vs-diagonal ruler" is also unverified and now suspect.** Given E1, the
  claim that the covariance structure is "only 1.26×, NOT the explanation" rests on the same un-scripted
  analysis. If the real mechanism is force-summing across z (E1c), the relevant covariance question is
  cross-*z*, which a DESI-DR1 cosmic-variance cov may actually carry (z-bins share cosmic structure) —
  the "1.26×" may have used a diagonal-in-z comparison that misses exactly the cross-z term that matters.
- **R4 — sample size.** n=8 fold-0 sims drives every coherent-mean and correlation claim. The coverage
  cert plan should use ≥ the full multi-fold val set; the "is it real" questions (E2, E3) need n≥20 to
  separate a 0.5σ coherent mean from 0.9σ scatter at 2σ.

---

## Recommendations (numbered, actionable)

1. **Rewrite §3.** Drop "z-COHERENT accumulation, reconciled with Phase-2 at 0.033∼0.067." Replace with
   the correct statement: the full-fit A_p force sums contributions from all ~19 effective z-bins; the
   per-z residual is z-NOISY (|mean_z|/std_z≈0.45); the ×18.6 is the z=3 force-share, computed with the
   full-z σ_Ap, NOT a coherent-summation factor and NOT normalization-comparable to the z=3-restricted
   0.067σ. The conclusion "diagonal-in-z C_emu can't whiten it" survives (force-summing, not coherence),
   but for the corrected reason.

2. **Add a committed script that regenerates the §3 numbers** (the z=3-force-share decomposition, the
   per-z coherence ratio, the cross-z-vs-diagonal cov comparison). No headline without an artifact.

3. **State the +0.62σ and r=−0.61 with their n=8 uncertainties.** "+0.53±0.33σ (95% CI [−0.26,+1.32],
   p=0.16)" and "r=−0.61, 95% CI [−0.92,+0.16], p=0.11 — suggestive, leaned on the independent
   basis/rank-floor evidence." Do not present either as established.

4. **Before designing the C_emu, measure the overlap of the emulator coherent (k,z) mode with the A_p
   Fisher direction** (cos angle between the residual template and `(F+P)⁻¹Jᵀ`-implied A_p direction).
   If high, the C_emu marginalizes A_p itself — escalate to PI; the mode may be irreducibly degenerate
   with the measurement and the answer is a wider A_p posterior, not a fixable bias.

5. **Build the C_emu as an explicit low-rank (1–3) template-marginalization**, with eigenvectors =
   measured residual modes spanning (k,z), NOT a per-z-diagonal inflation (verified to fail) and NOT a
   z-block-diagonal cross-class extension (also z-diagonal → also fails). Re-measure the A_p bias WITH it.

6. **Resolve σ_anchor < σ_α (matched units) in THIS framing.** The "anchor is the absorber" claim is
   only safe if the anchor is narrower than the global incidence prior; KS 0.27 > LLS 0.15 looks
   inverted and is now load-bearing for the slope-SAFE verdict.

7. **Re-run the gate on interior-n_s HELD-OUT sims** (a fold whose val set has n_s_u near 0.5), not just
   fold-0's all-edge set, before the n_s<0.2σ gate is locked as held-out-certified.

8. **For the eventual NUTS cert, report posterior-mean coverage, not MAP-shift Fisher** — the box-edge
   n_s and the A_p–anchor degeneracy are exactly where MAP≠mean.
