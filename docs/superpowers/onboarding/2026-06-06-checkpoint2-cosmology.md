# Cosmology/P1D checkpoint — low-k A_p bias investigation — 2026-06-06

Lens: the physics of the A_p/n_s bias — is it real, is it alarming for the science goal, and
which fork preserves cosmological precision. I re-ran the closure machinery (not just read it).

## Verdict (one line)

**GO-WITH-CHANGES** — the slope-prior decision is sound and reproduces exactly; the emulator-bias
*direction* is real and I reproduced it independently; but two headline framings (z-"coherent"
×18.6 accumulation; "high-k = few-bin shot-noise" sourcing) are misleading and the load-bearing
reconciliation number is not backed by any committed script. Fix the framing + add the missing
script before this becomes the basis for a fork decision.

---

## Claims I VERIFIED (re-ran or re-derived) vs claims I could not check

VERIFIED by independent re-run (my own code against `build_legb_ctx` + the production model):

1. **Slope/anchor's own bias (FULL−EMU) is < 0.17σ on every sim, width-insensitive.** Re-derived
   from `legb_slope_prior_rerun.txt` per-sim columns: FULL−EMU A_p mean −0.079σ, RMS 0.086σ,
   max 0.143σ; n_s mean −0.027σ, max 0.168σ. Matches doc Table §1 to the digit. The width sweep
   (lit/edge/2×) moves FULL bias by <0.01σ → the slope genuinely is not the driver. **(A) HOLDS.**

2. **The EMU-only A_p mean ≈ +0.62σ is real and reproducible.** My independent forward
   (sim-exact w_c shape, no anchor, full-z) gives **A_p EMU mean = +0.6173σ** — bit-matching the
   `emu_lowk_investigation.txt` PART B FULL (+0.617). Per-sim: [−0.65, 1.48, 1.98, −0.03, 1.03,
   −0.24, 1.32, 0.04]. **The number is not a coding artifact.**

3. **Band sourcing: high-k (k>0.02) dominates.** Reproduced PART B qualitatively — the A_p force
   is overwhelmingly in k>0.02. The leakage correlation r=−0.61 and emu/floor 6–18× I accept from
   the output (PART C/D); the oblique projector Bᵀ(BBᵀ)⁻¹B is the correct construction. **(B)
   mechanism HOLDS.**

4. **z=3-single-slice is tiny.** My z∈[2.9,3.1) restriction gives A_p EMU mean **+0.019σ** (doc:
   +0.033σ; same conclusion). The Phase-2 `diag_ap_fisher_bias.py` is confirmed single-z
   (`z_fid`, `z=3`, `zsel = isclose(z_grid, z_fid)` at lines 110-118). So the *premise* of
   reconciliation (C) — Phase-2 measured one z — is **correct**.

5. **Truth-side high-k is NOT shot-noise-noisy** (the alternative hypothesis I was asked to test).
   Cache `P_filt` log-P 2nd-difference roughness: clean low-k 2.96e-2, mid 1.10e-2, **high-k
   4.47e-3** (LLS: low 2.61e-2, high 7.26e-3). The high-k truth is the SMOOTHEST band — the bias
   is a genuine emulator systematic, not ΔP_emu contaminated by noisy truth. **This REFUTES the
   "truth shot noise leaking in" escape hatch and STRENGTHENS the doc's "emulator error" reading.**

COULD NOT fully check (accepted from outputs / flagged):
- The ×18.6 and +0.033σ exact values: **no committed script produces them** (see Errors §below).
  I reproduced the *qualitative* claim (ratio ≈ 33 with my z-window, +0.019σ at z=3) but the
  specific numbers in the doc/spec are unsourced.
- σ_anchor magnitudes (KS 0.27/DESI 0.12) — un-measured, as the doc admits.
- Everything is Fisher, not NUTS — the residual-after-marginalization is a forecast, not coverage.

---

## Errors / overclaims / things the doc gets wrong (specific, with numbers)

**E1 (MEDIUM, reproducibility) — the reconciliation numbers (0.033σ, ×18.6) are in NO committed
script.** `grep` for `0.033|18.6|single-z|z=3 alone` across `scripts/` returns nothing; both
`diag_emu_lowk_investigation.py` and `diag_legb_slope_prior_rerun.py` contain no z=3-vs-full-z
comparison. Reconciliation (C) is called "the sharpest finding" (doc:153) yet is unreproducible
from the artifacts under review. I had to re-derive it myself. **Add a committed script that
emits the z=3↔full-z ratio**, or the central claim has no audit trail.

**E2 (MEDIUM, physics framing) — "z-COHERENT, sums over ~13 z-bins (×18.6)" is misleading.** I
decomposed the +0.617σ into per-z-window contributions (summing exactly to +0.6173):
```
 z[2.0,2.3) +0.326   <- 53% of the total, KS-only band
 z[3.5,3.7) +0.157   <- 25%, the second peak
 z[2.9,3.1) +0.033 ; all others |·|<0.08, MIXED SIGN (z[3.1,3.3) is -0.040)
```
The bias is NOT a uniform coherent sum over 13 bins — it is **concentrated at the z≈2.0-2.3 and
z≈3.5-3.7 edges/peaks**, with sign flips in between. The clean-class low-k coherent emu error
confirms this: +0.94% at z=2.2, −0.3..−0.8% at z=2.6-3.2, **+1.07% at z=3.6** — it sign-flips.
The "×18.6" is an artifact of dividing by a z=3 slice that sits near a zero-crossing. This is not
a cosmetic point: the **actionable** structure (two z-peaks) is very different from the doc's
"spread coherently, needs a z-coherent C_emu mode" — it argues for **z-LOCALIZED** treatment
(edge down-weighting / z-resolved C_emu), which is cheaper. The spec §8 inherits the same framing.

**E3 (LOW-MEDIUM, physics framing) — "high-k = few-bin, shot-noise-noisy regime the PI flagged"
(doc:108) is wrong for the FIT band.** In the actual legs, k>0.02 is **39% of DESI's kept bins
(263/681) and 83% of KS's (70/84)** — the majority, not "few." And k_max is 0.041 (DESI) / 0.063
(KS), well INSIDE the validated band (memory: analysis capped k<0.06). These are
well-measured bins, not the shot-noise extreme of the sim P1D. The leakage mechanism is real
(global basis + bin-count weighting), but the "noisy few high-k bins" story is the wrong physical
picture — and it matters because it wrongly implies the fix is "trust high-k less / it's noise."
It is signal-level emulator error at trustworthy k.

**E4 (LOW, headline inconsistency) — the doc quotes "+0.62σ" but the gating rerun is +0.53σ.** The
TL;DR (doc:21,84) and figure use +0.62, which is `emu_lowk` PART B (sim-exact shape, no anchor, no
DLA mask). The production-faithful `legb_slope_prior_rerun.txt` (lit g_fixed, DLA-masked, per-leg
anchor) gives **EMU A_p mean +0.527σ** (line 18). These differ by the anchor + DLA mask + lit-vs-sim
shape. The doc should state which is the production-relevant number (+0.53 is closer to production;
+0.62 is the cleaner emulator-only diagnostic) rather than using them interchangeably.

**E5 (LOW, statistics) — the "+0.62σ MEAN" is a 1.6σ detection on 8 sims, not established.** EMU
A_p mean +0.528, per-sim std 0.94, **SE 0.33 → t = 1.58** (the spec §8 even says "SE 0.31 → ~1.7σ").
A t=1.6 mean is suggestive, not significant. Calling it a settled "+0.6σ coherent bias the C_emu
must marginalize" overstates an 8-sample result. The per-z decomposition (E2) is actually stronger
evidence it's real-and-localized than the noisy pooled mean.

---

## My lens's take on the fork: correlated-in-k/z C_emu vs targeted emulator retrain

**Reframe the fork using E2.** The doc poses it as "marginalize a z-coherent ~0.6σ mode via a
correlated-in-k/z C_emu (a) vs retrain (b)." My decomposition shows the bias is **z-localized at
the leg edges (z≈2.0-2.3 dominant, z≈3.5-3.7 secondary)**, not a uniform z-coherent mode. That
changes the calculus:

**Cost of option (a) as specced (a single rank-1 z-coherent direction).** Marginalizing a nuisance
direction aligned with A_p inflates σ(A_p) by 1/√(1−ρ²) where ρ is the alignment with the A_p
response. The A_p signal IS a low-k boost and the bias mode IS low-k-projected → ρ is plausibly
0.7-0.9, i.e. **σ(A_p) inflation ×1.4 to ×1.9** if the mode is broad in k. That is FAR more than
the ×1.04 the slope prior costs, and it would dominate the cosmology error budget. A naive "add a
0.6σ-amplitude low-k nuisance" risks trading a 0.6σ bias for a ~2× looser A_p. **Quantify the
alignment before committing** — this is the load-bearing number the fork needs, and it is not in
any artifact.

**Why z-LOCALIZED is the better (a).** Because 78% of the bias lives in two narrow z-windows, a
**z-resolved C_emu that inflates/correlates only the z≈2.0-2.3 and z≈3.5-3.7 bins** removes most
of it while touching far fewer modes → much smaller σ(A_p) hit than a global z-coherent mode. This
also aligns with onboarding R7 (C_emu z-banding is only 3 bins; HeII reion z≈3-4 worst) — the
z≈3.6 peak sits exactly at the He II reionization convergence trough. **My recommendation: build
the correlated C_emu z-RESOLVED, weighted to the edge/HeII windows, NOT a single global
z-coherent rank-1 mode.**

**On retrain (b).** PART C says emu err is 6-18× the rank-24 truth floor → reducible in principle.
But: (i) the floor comparison uses an in-range truth SVD, an optimistic target; (ii) the basis is
well-conditioned (I checked: cond 12.8, min singular value 0.22, full rank) — the "||BBᵀ−I||≈2.7
drifted far from orthonormal" claim is **mostly diagonal row-norm (1.05-1.92), max off-diagonal
only 0.87**; a non-unit-norm but well-conditioned decoder is normal and does NOT by itself create
band coupling (any low-rank basis couples bands). So "fix the non-orthonormality" is a weak lead.
A retrain that genuinely reduces the **z=2.2 and z=3.6 low-k coherent error** is the clean fix, but
it is uncertain payoff and re-opens the validated emulator. **Defer (b); do (a)-z-resolved now.**

**Net fork stance:** option (a), but **z-resolved/edge-weighted**, not a global z-coherent mode —
and gate it on a measured σ(A_p) inflation < ~1.1× (if the alignment forces ×1.5+, that is itself
a NO-GO signal pushing back toward retrain or a tighter k-cut).

---

## Additional risks or missing checks

R-a (HIGH) — **No σ(A_p)-inflation number for the fork.** The entire fork hinges on "can C_emu
marginalize the mode cheaply," and that cost (the alignment ρ → 1/√(1−ρ²)) is unmeasured. This is
the single most important missing number. A 0.6σ bias removed at ×1.9 σ(A_p) is a bad trade; at
×1.05 it is free. Without it the checkpoint cannot actually choose.

R-b (MEDIUM) — **The bias is measured against a single fold-0 emulator.** All 8 "held-out" sims
share ONE trained model. The z=2.2/z=3.6 coherent error could be a quirk of `final_fold0`'s
training, not a property of the architecture. An 8-fold spread of the per-z coherent error (one
model per fold) would distinguish "this model's residual" from "the method's residual" — relevant
because C_emu is supposed to already pool LOSO fold scatter (per memory). If the edge-coherent
error is fold-specific, the pooled C_emu may already partly cover it.

R-c (MEDIUM) — **z≈2.0-2.3 is KS-only and below DESI's range.** KS is HR and capped k<0.06; its
z=2.0-2.2 lowest bins carry 53% of the bias. Is that band actually trustworthy for KS, or is it a
selection/continuum-fitting edge that should be cut? A simple test: **drop z<2.3 and re-measure** —
if the A_p bias falls to ~0.3σ, the cheapest fix is a z-floor cut, not a C_emu rebuild.

R-d (LOW) — **n_s is worse than A_p and under-discussed.** The rerun shows per-sim n_s EMU biases
up to −2.6σ (sim 2), FULL n_s mean −0.27σ with max |2.72σ|. The doc foregrounds A_p; n_s has
larger tails (partly the known box-edge effect, but not entirely — interior fold-4 n_s is +0.16).
The cosmology result is (A_p, n_s) jointly; n_s deserves equal scrutiny in the C_emu design.

R-e (LOW) — **MF re-measurement is correctly flagged but the sign is unknown.** Since MF acts
multiplicatively at high-k (the source band) and the bias is partly z=3.6 (HeII, where MF
res_corr is largest), MF could *increase* the bias. The doc says "re-measure"; it should say
"re-measure and treat the LF number as possibly optimistic."

---

## Recommendations (numbered, actionable)

1. **Measure the σ(A_p) inflation of the proposed C_emu mode BEFORE the fork is decided.** Compute
   the alignment ρ between the A_p response and the bias mode in the C-metric; report
   1/√(1−ρ²). Gate: inflation > ~1.15× → reconsider (retrain or k/z-cut). This is the missing
   load-bearing number. (forward-only, cheap.)

2. **Re-frame the reconciliation as z-LOCALIZED, not z-coherent.** Add the per-z-window
   decomposition (I get z[2.0,2.3)=+0.326, z[3.5,3.7)=+0.157, sum +0.617) to the doc and spec §8,
   replacing "sums coherently over 13 bins ×18.6." Design the correlated C_emu **z-resolved and
   edge/HeII-weighted**, not a single global rank-1 mode.

3. **Commit the z=3↔full-z reconciliation script.** The 0.033σ / ×18.6 numbers must be
   reproducible from a checked-in script, not asserted in prose.

4. **Run the cheap z-floor test:** drop z<2.3 (KS lowest) and re-measure the A_p EMU bias. If it
   falls to ≲0.3σ, a z-cut is the cheapest fix and avoids a C_emu rebuild + retrain entirely.

5. **Fix the framing errors:** (a) the doc:108 "few-bin shot-noise high-k" → "well-measured bins,
   39%/83% of the fit band, inside the validated k<0.06"; (b) state +0.53σ (production-faithful) vs
   +0.62σ (emulator-only diagnostic) explicitly; (c) downgrade the "||BBᵀ−I||≈2.7 → coupling"
   lead — it's mostly row-norm, basis is well-conditioned (cond 12.8).

6. **Report the +0.6σ with its uncertainty (t≈1.6 on 8 sims).** State it as "suggestive coherent
   mean, established by the per-z localization" not "a confirmed +0.6σ bias."

7. **Give n_s equal billing.** Re-measure the n_s EMU bias on interior-n_s sims (away from the box
   edge) and include it in the C_emu/fork decision; the −2.6σ tails need a separate read.

8. **The eventual gate is NUTS coverage, not Fisher.** Keep all of the above as forward-only
   priors to the coverage run; the residual-after-marginalization claim is a forecast until the
   N≥99 closure (with the z-resolved C_emu wired) confirms |bias|<0.2σ AND coverage≥nominal on
   BOTH A_p and n_s.
