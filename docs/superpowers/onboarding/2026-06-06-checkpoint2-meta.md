# Checkpoint meta — low-k A_p emulator-bias investigation — 2026-06-06

Adjudicator over four referees (Bayesian, CS/JAX, Cosmology/P1D, Lyα/IGM-fidelity). All four
re-ran the two committed scripts (`diag_emu_lowk_investigation.py`, `diag_legb_slope_prior_rerun.py`)
in the project env. I read all four reports in full and spot-verified the load-bearing process
claims (reconciliation-script existence, KS k-cap) directly. Below I separate **verified** (a lens
re-ran/re-derived it and it held) from **disputed/corrected** (a lens broke or qualified it) from
**PI must decide** (no lens can settle it from the current artifacts).

---

## Overall verdict: GO-WITH-CHANGES

**Unanimous across all four lenses: GO-WITH-CHANGES.** No lens said NO-GO; none said clean GO.

Rationale. Two things are solid and four are not:
- **Solid (proceed):** (1) The slope-prior settlement — the FULL−EMU differential that gates the
  per-leg α refactor — reproduces bit-for-bit on all four lenses, the differential logic is
  statistically valid, the slope/anchor's own bias is <0.17σ and width-insensitive. (2) The code is
  correct: Jacobian shape is right (the old flatten bug is gone), band-attribution is an exact
  linear decomposition (sums to FULL at 1e-9), the golden guard is live and tight (1e-6 trips it),
  and `||BBᵀ−I||≈2.7` is an expected trained-leaf drift, not a bug.
- **Not solid (must fix before this drives a fork):** (1) the §3 reconciliation has **no committed
  script** (verified: only the docstring mentions "ruler"); (2) the "z-COHERENT, same sign at every
  z" mechanism is wrong/imprecise; (3) the headline +0.6σ and r=−0.61 are **n=8, ~1.6–1.9σ point
  estimates stated as settled fact**; (4) the fork's load-bearing cost — σ(A_p) inflation from
  marginalizing the mode — is **unmeasured**.

The changes required are documentation-honesty + one missing artifact + one missing measurement.
None invalidate the decision the checkpoint actually needs to make now (the slope-prior refactor).
The C_emu-vs-retrain fork should NOT be decided at this checkpoint — it is not ready.

---

## Did the headline numbers survive scrutiny?

| Headline | Status | Adjudication |
|---|---|---|
| **×1.04** (slope-prior σ(A_p) cost) | **VERIFIED** | Not independently re-derived by lenses but uncontested; consistent with the width-insensitivity all four confirmed. Cheap and benign. |
| **slope/anchor own-bias < 0.17σ** (A_p max 0.143, n_s max 0.168) | **VERIFIED (4/4)** | All four re-derived FULL−EMU by hand from the per-sim columns: A_p mean −0.079, max 0.143; n_s mean −0.027, max 0.168. Differential logic valid (shared emu-error force cancels). **The most robust result in the checkpoint.** |
| **+0.62σ EMU A_p bias** | **VERIFIED as a point estimate, DISPUTED as "settled"** | Magnitude reproduces (CS bit-for-bit +0.617; Cosmology +0.6173; Bayesian/Lyα confirm). BUT it is an **n=8 mean, SE≈0.33, t≈1.6–1.9, 95% CI ≈ [−0.26,+1.32], 4 pos / 4 neg signs.** All four flag it is stated too confidently. Also quoted as +0.62 (emu-only diagnostic) vs +0.53 (production-faithful rerun) interchangeably — two different configs (anchor + DLA-mask + lit-vs-sim shape). |
| **high-k source +0.72σ** | **VERIFIED arithmetic, DISPUTED framing** | +0.723 reproduces. But "high-k = few-bin shot-noise" is **wrong**: Cosmology shows k>0.02 is 39%/83% of DESI/KS fit bins (well-measured, inside validated band); Lyα shows **+0.31σ is sourced from the IN-SCOPE 0.02–0.06 band**, not the out-of-scope tail. Cosmology also refuted the "noisy truth leaking in" hypothesis (high-k truth is the *smoothest* band → genuine emulator error). |
| **r=−0.61 leakage** | **VERIFIED value, DISPUTED significance** | −0.612 reproduces. But n=8: t≈−1.9, p≈0.11, Fisher 95% CI [−0.92,+0.16] **crosses zero**, one leverage outlier. "Suggestive," not established. The independent basis-orthonormality + rank-floor evidence is the stronger support. |
| **6–18× rank floor** (training-limited not rank-limited) | **VERIFIED, with two caveats** | Reproduces; CS checked the subtle apples-to-apples point (model trains in `sig_cosmo` space, floor computed in `sig_marg` space) and the **ratio is space-invariant** (15.6× vs 17.9× etc.) → verdict survives. Caveats: (a) "6–18× in *every* band" is false — mid band is 2–4× (Lyα); (b) the floor is a *pooled-ensemble* RMS, not per-sim (CS). |
| **z-accumulation ×18.6** | **VERIFIED arithmetic, CORRECTED mechanism** | The ratio reproduces EXACTLY on all four (single-z3 +0.033 → full-z +0.617 = ×18.6). BUT the *interpretation* is wrong. See the next section — this is the single biggest correction. |

---

## Errors / overclaims the lenses found (and whether they change conclusions)

**MAJOR — the ×18.6 "z-coherent accumulation" mechanism is mis-derived.** This is the strongest
cross-lens finding (all four independently, by different routes):
- **Bayesian:** measured per-z coherence directly, |mean_z|/std_z ≈ 0.45 (<1) — the mode is
  z-NOISY; ×18.6 is the trivial z=3 force-share (~1 of ~19 effective z-bins), and a genuinely
  coherent mode would scale ×√N≈4, not ×N≈19. Also: the "+0.033σ single-z" uses the **full-z**
  posterior width in its denominator, so it is NOT normalization-comparable to Phase-2's
  z=3-restricted 0.067σ — the claimed "0.033 ∼ 0.067 consistency" is a magnitude coincidence.
- **Cosmology:** per-z decomposition (sums exactly to +0.617) shows the bias is **z-LOCALIZED**:
  53% from z≈2.0–2.3, 25% from z≈3.5–3.7, mixed-sign between; the clean low-k error sign-flips
  (+0.94% z=2.2, −0.8% z=3.0, +1.07% z=3.6).
- **CS:** raw per-z high-k signed error sign-flips (−0.009 at z<2.5, +0.004 at z~3); coherence
  exists only in the **C_inv-whitened, A_p-Jacobian-projected** direction, not the raw spectrum.
- **Lyα:** P_filt error flips sign in z (frac z>0 = 0.69, mean ≈ 0); ×18.6 is projection
  accumulation, not physical coherence.

**Does it change conclusions?** The *number* +0.62σ survives (it is real, reproduced). The
conclusion "a diagonal-in-z/k C_emu cannot whiten this mode → the correlated C_emu is the right
*kind* of tool" **also survives** — but for the corrected reason (the force sums over z-bins;
a diagonal cov cannot tell summed-structure from noise), NOT because the mode is coherent. What
changes is the **C_emu design**: a single global z-coherent rank-1 mode is the wrong shape; the
bias is z-localized at the leg edges (z≈2.0–2.3, z≈3.5–3.7/HeII), arguing for a z-resolved /
edge-weighted treatment (Cosmology, Lyα). The §3 prose must be rewritten.

**MAJOR — no committed script regenerates §3.** Verified directly: `grep` for the reconciliation
numbers hits only the docstring (line 3) of `diag_emu_lowk_investigation.py`. All four had to
re-derive ×18.6 by hand. The z-accumulation leg checks out; the **1.26× correlated-vs-diagonal
"ruler" and the 3%-diagonal −0.60σ legs were verified by NO lens** (no artifact, no inline
derivation). Process gap: a load-bearing headline with no reproducible artifact. Does not change
the verified numbers but blocks trusting the unverified ruler legs.

**MODERATE — significance overstated throughout.** +0.6σ (t≈1.6), r=−0.61 (p≈0.11), high-k +0.72σ
(t≈2.9, the one that is actually ~3σ). All n=8, all 4-pos/4-neg or leverage-driven. The n_s EMU
mean (−0.24σ) is **a single-outlier artifact** (CS: drop the −2.6σ ns0.813 sim → +0.10σ); there is
no coherent n_s EMU bias. Does not change the slope gate (uses FULL−EMU). Does change how much
weight the +0.6σ can bear as the "relocated load-bearing risk."

**MODERATE — scope-cap violation in the rerun (Lyα).** KS keep extends to k=0.0627 (14 bins ≥0.06),
violating the locked KS k<0.06 cap. Inflates the "high-k"/OUT-band number relative to the
production config. DESI is clean (k_max 0.0409). Must re-run KS-capped before quoting as production.

**MINOR — attribution/citation errors.** The "n_s box-edge effect" attribution is unsupported by
the data (CS: r=+0.00 between edge-proximity and |bias|; the most-edge sim has the *smallest* n_s
bias). The doc cites `model.py:135` (θ-blind BaselineHead) but the A_p gradient flows through
HeadB's `p_filt_basis` (Lyα; the script reads the right object, only the cite is wrong). The
"||BBᵀ−I||≈2.7 → band coupling" retrain lead is weak — basis is well-conditioned (cond 12.8), the
2.7 is mostly row-norm (Cosmology). None change conclusions; all are doc fixes.

---

## THE FORK — correlated-in-k/z C_emu vs targeted retrain vs MF-wiring-first

**This is where the lenses genuinely diverge. The fork is NOT ready to be decided at this checkpoint.**

**Cross-lens convergence (3 of 4):** Bayesian, CS, and Cosmology all favor **option (a), the
correlated-in-k/z C_emu, over a retrain (b)**:
- CS (implementability): the likelihood already dense-Choleskys a full C; the change is replacing
  one `jnp.diag(emu_var_flat)` (`data_likelihood.py:377`) with a dense block, reusing the existing
  `rho_zb` cross-class plumbing — **zero downstream certs invalidated**. A retrain mutates the
  trained basis → regenerates the golden, re-opens all 8 LOSO folds, invalidates every bias cert,
  against a noisy (~1.6σ) target. Strong, concrete, hard to argue with.
- Bayesian and Cosmology agree (a) is the right *kind* of object, but **both impose a precondition
  the doc entirely omits**: the cost of marginalizing the mode is σ(A_p) inflation = 1/√(1−ρ²),
  where ρ is the alignment between the bias mode and the A_p response direction. If ρ is high (both
  are low-k boosts → plausibly 0.7–0.9), the C_emu **marginalizes A_p itself**: Bayesian's toy
  shows σ_Ap blowing up (0.027→2.67) when J∥u; Cosmology estimates ×1.4–1.9 σ(A_p) inflation. That
  would be a far worse trade than the ×1.04 the slope prior costs. **This number is in no artifact
  and is the load-bearing input the fork actually needs.**
- Both also refine the *shape*: not a single global rank-1 z-coherent mode (wrong, per the
  corrected mechanism), but a **z-resolved / edge-weighted, rank-1–3 template marginalization**
  targeting z≈2.0–2.3 and z≈3.6/HeII. Bayesian prefers an explicit nuisance-template
  marginalization (mathematically equivalent to rank-1 C_emu, more auditable).

**The dissent (Lyα):** do NOT treat this as a 2-way fork at all — it is **3-way and gated on MF**.
The +0.6σ is measured for a forward model production will not run (no MF res_corr, no log_rho).
res_corr already reshapes the exact bias band by 2–4% in-scope / 6–9% out-of-scope, and MF carries
its own ~1.4σ A_p budget. So the honest sequence is **MF-wire → re-measure (KS<0.06, all folds) →
correlated C_emu on the POST-MF residual → retrain only if a monotone-z physical mode survives**
(current evidence says it does not — the residual is clean-forest-dominated and training-limited,
not HCD-template; per-class, DLA/subDLA are closest to the floor). Cosmology's R-e and the MF
memory partly support this (MF could increase the bias, not absorb it).

**What the PI must decide.** Three positions, reconcilable into a sequence:
1. Is the LF-only +0.6σ even the right target, or must MF be wired first (Lyα's dissent)? This is
   the **gating question** — if MF must come first, the C_emu design is premature regardless.
2. Conditional on the target being settled: build the C_emu **z-resolved/edge-weighted**, NOT a
   global z-coherent mode (3-of-4 refinement), and **measure ρ / the σ(A_p) inflation first**
   (Bayesian + Cosmology precondition). Gate: inflation > ~1.15× → push back toward retrain or a
   z/k-cut.
3. A cheap alternative no one has tested: **drop z<2.3** (KS-only, carries 53% of the bias) and
   re-measure. If A_p falls to ≲0.3σ, a z-floor cut is cheaper than any C_emu rebuild (Cosmology
   R-c).

My adjudication: the lenses do NOT actually conflict on the *tool* — all four accept the correlated
C_emu is structurally right and a retrain is premature. They conflict on **sequencing and
prerequisites**. The synthesis that honors all four: **do not pick the fork at this checkpoint.**
First (i) wire MF or explicitly justify the LF-only number as adequate, (ii) measure the
ρ/σ(A_p)-inflation cost, (iii) run the cheap z<2.3 cut test, (iv) re-run KS-capped over all folds
with a bootstrap CI. Then the fork decides itself.

---

## What the next implementation step should be

**The slope-prior refactor PROCEEDS NOW. It is NOT gated by the C_emu/retrain decision.**

All four lenses verified the FULL−EMU differential (the quantity that gates the per-leg α
refactor) reproduces exactly and the slope/anchor own-bias is <0.17σ, width-insensitive. CS
confirmed the golden guard is live and tight and explicitly recommends it as the hard gate on the
refactor. The slope-prior decision rests on the differential (which cancels the shared emulator
error), so it is **independent of the unresolved +0.6σ EMU-bias / fork questions**. Greenlight it,
guarded by the golden test.

The C_emu/retrain fork is a **separate, later** workstream that is explicitly NOT ready: it needs
(a) MF wired or justified-absent, (b) the σ(A_p)-inflation measurement, (c) the §3 rewrite +
committed script, (d) all-folds re-measurement with KS<0.06. None of these block the refactor.

One precondition the Bayesian lens raises that touches the refactor: **σ_anchor (KS 0.27,
DESI 0.12) > σ_α (0.15)** — an anchor wider than the global incidence prior could itself erode A_p
identifiability ("the anchor absorbs the slope" is only benign if the anchor isn't absorbing A_p).
This was carried over from checkpoint-1 and is still unresolved. It does not block the refactor but
should be checked (matched units) before the slope-SAFE verdict is locked.

---

## Open items for the PI

1. **DECIDE the gating question:** is the LF-only +0.6σ an acceptable target, or must MF be wired
   first before any fork decision? (Lyα says MF-first is non-negotiable; 3 lenses are silent on
   sequencing.) This determines whether the C_emu work starts now or after MF.
2. **COMMISSION the missing load-bearing measurement:** ρ (alignment of the bias mode with the A_p
   response) → σ(A_p) inflation 1/√(1−ρ²). The fork cannot be chosen without it. Forward-only,
   cheap. Gate at ~1.15×.
3. **APPROVE the cheap z<2.3 cut test** — if it kills 53% of the bias, the whole C_emu rebuild may
   be unnecessary.
4. **REQUIRE a committed script for §3** before the reconciliation drives any design (the 1.26×
   ruler and 3%-diagonal legs were verified by no lens).
5. **ACCEPT the doc-honesty edits:** state +0.6σ / r=−0.61 with n=8 uncertainties; rewrite the
   z-coherence prose to "does not average down / coherent in the A_p-projected direction" (not
   "same sign at every z"); state +0.53 (production) vs +0.62 (diagnostic) explicitly; drop the
   n_s box-edge attribution (outlier-driven); fix the `model.py:135` cite.
6. **DIRECT scope/scale-up:** re-run KS-capped at k<0.06; run the EMU bias over ALL folds'
   held-out sims + at least one interior-n_s held-out fold (current 8 are all low-n_s box-edge,
   interior spot-checks are in-sample) + bootstrap CI, before the fork is decided.
7. **RESOLVE σ_anchor < σ_α (matched units)** — load-bearing for the slope-SAFE verdict; carried
   over unaddressed from checkpoint-1.
8. **GIVE n_s equal billing** in the eventual C_emu design — its tails (−2.6σ) are larger than A_p's
   and under-discussed.
9. **NOTE the eventual arbiter is NUTS coverage, not Fisher** — every number here is a MAP-shift
   surrogate; the box-edge n_s and the A_p–anchor degeneracy are exactly where MAP≠mean.
