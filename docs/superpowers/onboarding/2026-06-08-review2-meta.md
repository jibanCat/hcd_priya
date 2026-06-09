# Meta-review — n_s-bias resolution (review-2, 4-lens) — 2026-06-08

**Reviewer:** meta-synthesizer (Opus 4.8 1M)
**Inputs:** Bayesian/PPL, CS/ML/JAX, Cosmology/P1D, Lyα/IGM referee reports + their plots
**Object under review:** the resolution of the coherent −0.65σ n_s EMU bias (root-caused 2026-06-08):
z-attribution → low-z KS leg → `load_ks_leg` z_lo cut + the planned MF (LF→HR) wiring with a z+τ₀-resolved
resolution correction.

---

## OVERALL VERDICT: **GO_WITH_CHANGES**

All four lenses returned GO_WITH_CHANGES. **None returned NO_GO.** The resolution's *science core* is
unanimously endorsed and I independently reproduced the load-bearing facts from the committed artifacts. The
changes required are not re-derivations; they are (a) one cheap missing certification run, (b) one
data-provenance default flip with three docstring/plan reconciliations, and (c) a set of MF-wiring design
constraints that must be respected *when the wiring is built* (not before). The MF wiring itself is correctly
deferred — this review gates the *pre-MF-wiring to-do*, and that to-do is small and concrete.

I separate this into **what is settled** (do not relitigate), **CRITICAL blockers** (must close before the
production likelihood is trusted / before unblinding), and **IMPORTANT design constraints for the MF wiring**
(must be respected during the next task, verifiable then).

---

## 1. WHAT IS SETTLED — unanimous, reproduced, do not relitigate

These have 4/4 agreement AND I re-verified them against the npz/txt artifacts:

- **The z-attribution is an EXACT linear Fisher decomposition**, not a per-z re-fit. Σ_z bias_z = total to
  2.1e-13 (verified: `nsbias_z_attribution.npz` `bias_z_mean` sums to −0.6461 = `cum[-1]`). Leg additivity
  (DESI+KS) to 1.1e-15. C_post is the full information matrix, so the τ₀–n_s degeneracy is propagated, not
  swept. **This is the cleanest, most decisive part of the resolution.**

- **The bias is the low-z KS leg.** z=2.0 = −0.473σ, z=2.2 = −0.204σ (`bias_z_ks_mean`), summing to −0.676σ
  = ~105% of the −0.646σ total; DESI is +0.039σ (clean, k≤0.02 driver). z=2.4 KS is **independently clean**
  (+0.0094σ). Reproduced exactly across all four lenses and by me.

- **It is NOT a C_emu-sizing / β-attenuation / normalization-loss artifact.** C_emu toggle 0→×2 moves the
  bias only ±0.01σ (verified: `toggle_prod`=−0.6461, `data_only`=−0.6597, `inflate2`=−0.6344). β≈0.98, the
  (1−β) contribution is ~0.06σ (10–30× too weak). The n_s *response* is faithful.

- **The LF→HR −6% deficit is a genuine resolution effect**, not a mean-flux match artifact (Lyα verified
  achieved ⟨F⟩ matched LF/HR to <0.15%; a 0.1% flux offset cannot make a −7.6% deficit). Coherent 6/6 sims,
  smooth-in-k, tilt-shaped. The τ₀-dependence (~4.3pp rung0→rung19) is physically expected (mean-flux
  modulation of which structures are resolution-sensitive).

- **The closure object (−0.65σ, LF-emu-vs-LF-truth) and the real-fit object (−6%, LF-vs-HR/reality) are
  genuinely DISTINCT**, sharing only the low-z/small-scale regime. The z-cut fixes the first; the MF wiring
  fixes the second. 4/4 agree this distinction is correctly drawn.

- **The z+τ₀-resolved MF correction is the right design** (vs the current α-pooled gbar). 4/4 endorse the
  *direction*.

- **keep ns∈[0.8,1.05] + step-inflated C_emu above 0.995 is correct over a hard ns<1.0 cap** (eBOSS
  n_P=1.009 would rail against a wall). 4/4 agree.

- **The MF LOSO is honestly weak** (6 clustered sims, ns∈[0.859,0.979], edges untested,
  necessary-not-sufficient) and the worst-per-sim coherent residual +0.91% (mid-cluster, ns=0.914, not an
  edge artifact) is correctly the C_emu floor, not certifiable to zero. 4/4 agree.

---

## 2. THE ONE REAL DISAGREEMENT: z_lo=2.4 vs z_lo=2.8

This is the only place the lenses split, and it is **the single most important thing this meta-review must
resolve**, because it is the *shipped production default* and the diff is already committed.

**The split:**
- **Bayesian** and **CS** endorse z_lo=2.4 *as the Bayes-optimal closure cut* (keep the most clean,
  informative low-z constraining power; 2.4 itself is demonstrably clean in closure). Both say it is
  evidence-led, not p-hacking, because the cut is externally pre-registered (DLA-finder incompleteness) and
  the localization is post-hoc confirmation.
- **Cosmology** and **Lyα** say z_lo=2.4 optimizes the WRONG objective and recommend **defaulting to z_lo=2.8**
  (the published KODIAQ-SQUAD cut), exposing 2.4 as opt-in.

**Resolution (I side with Cosmology/Lyα on the DEFAULT, with Bayesian/CS on the SCIENCE):**

The two camps are not actually contradicting each other on facts — they are answering two different
questions, and both answers are correct:

1. *"Is z_lo=2.4 sufficient to kill the closure bias?"* — **YES, unanimously.** The closure is statistically
   blind to 2.4-vs-2.8: subset numbers are 2.4=+0.041σ, 2.6=+0.046σ, 2.8=+0.049σ — all within noise of each
   other (I verified `comb_ns` in `nsbias_kscut_scan.npz`). The bias is gone the instant z=2.0+2.2 drop.

2. *"Which cut should SHIP in the REAL likelihood?"* — This is a **data-provenance** decision the
   sim-truth closure is structurally blind to. The closure replaces the data with the emulator's own
   prediction, so a systematic common to emulator-and-data (DLA-finder incompleteness in the real KS data)
   is invisible to it. The published analysis excludes z<2.8 for exactly this reason, and
   DLA/sub-DLA completeness degrades *continuously* toward low z — so the real z=2.4 and z=2.6 KS bins are
   plausibly compromised by the same systematic even though closure cannot see it.

The deciding factor: **the upside of keeping z=2.4/2.6 is essentially zero.** Cosmology computed it directly
— the two bins where 2.4 and 2.8 differ contribute +0.008σ of *closure* constraint, and the marginal n_s
constraining power is negligible (DESI carries the low-z constraint). So the "keep clean informative data"
argument (Bayesian/CS) is *correct in principle* but applies to a quantity of constraining power that rounds
to zero here, against a real (if closure-invisible) data-quality risk. **When the upside is ~0 and the
downside is a paper-deemed-unreliable-data risk, you match the published cut.**

There is also a hard, non-negotiable defect independent of the science: **the code is internally
inconsistent.** The `load_ks_leg` docstring (verified, lines 159–161) cites *"the KODIAQ-SQUAD analysis
excludes z<2.8"* as its rationale and then sets the default to **2.4**. Code, its own stated rationale, the
plan RESOLVED text, and MEMORY all disagree on 2.4 vs 2.8. This must be reconciled regardless of which value
wins. Defaulting to 2.8 reconciles all four at once; keeping 2.4 requires rewriting the rationale to *not*
cite the 2.8 paper cut and to justify 2.4 on its own grounds (which the closure cannot supply).

**My ruling: default to z_lo=2.8, expose z_lo=2.4 as an opt-in sensitivity variant.** This honors the
Bayesian/CS finding (2.4 is scientifically safe for closure → this is GO_WITH_CHANGES, not NO_GO) while
adopting the more defensible production default and ending the inconsistency. It is also the *conservative*
choice, which is the right default posture for a blinded real fit.

---

## 3. CRITICAL BLOCKERS (close before the production likelihood is trusted)

### C1. The SHIPPED z_lo default has no all-folds (60-sim) certification.
Raised by **Bayesian** [MINOR→I escalate], **CS** [IMPORTANT], implicit in **Cosmology/Lyα**. I verified:
the only ALL-FOLDS closing number is **z_lo=2.6 = +0.038σ** (`allfold26_ns`). z_lo=2.4 exists ONLY on the
30-sim subset {0,2,5,7} (+0.041σ). The subset is a **biased estimator**: its z_lo=2.0 baseline is −0.775σ vs
the all-folds anchor −0.646σ. **You cannot ship a production default certified only on a biased 30-sim
subset.** I escalate this above Bayesian's "MINOR" because it is the *shipped* config, not a scan point.

→ **Resolution:** This blocker is *dissolved* if you adopt z_lo=2.8 (C2 below), because z_lo=2.8's nearest
all-folds anchor is z_lo=2.6=+0.038σ and the attribution shows z≥2.6 KS is clean — BUT the cleanest path is
to run **one** all-folds (60-sim) config at **whatever z_lo ships** and record it next to the z_lo=2.6
all-folds row. Cost: one `run_config({k_max:0.06, z_lo:<shipped>}, folds=range(8))`, forward-only, no
retrain. **Do not ship the production default on a subset number.**

### C2. Reconcile the z_lo default + its docstring/plan/memory rationale (see §2).
Raised by **Cosmology** [IMPORTANT] + **Lyα** [IMPORTANT×2]. The docstring/plan/code disagreement is a
factual defect, not a matter of taste. Adopt z_lo=2.8 as default and align docstring + plan RESOLVED text +
MEMORY; OR, if 2.4 is kept for a stated reason, rewrite the rationale so it no longer cites the z<2.8 paper
cut. Either way the inconsistency must end.

---

## 4. IMPORTANT — design constraints for the MF wiring (verifiable when built)

These do not block the current commit; they are binding constraints on the *next* task (the MF wiring) and
on the C_emu sizing. They have strong cross-lens agreement and must be in the MF-wiring plan before it
starts.

### I1. Build the τ₀-resolved correction ADDITIVELY/SEPARABLY, then re-run the n_s Fisher check THROUGH the MF forward.
**CS** [IMPORTANT, strongest statement] + **Bayesian** + **Lyα** endorse the additive form; **Cosmology**
dissents partially (see §5). The danger: a free 2-D (z,τ₀,k) table fit on 6 clustered sims with a documented
(z,τ₀) sign-flip will **alias the n_s tilt into the τ₀ axis** and recreate the exact over-fit that got
`delta_mode='mlp'` retired (4σ n_s Fisher bias — the reason `delta_mode='none'` is the validated default).
CS's prescription: `log_rho(k) + gbar_z(z,k) + gbar_tau(τ₀,k)` so τ₀ is a 1-D marginal that cannot absorb the
(z,k) tilt. **Then re-run `diag_emu_bias_allfolds` THROUGH `MultiFidelity.logP_mf`** (not the raw cache) before
adopting — this is the single most important MF-wiring verification gate.

### I2. The C_emu small-scale floor must be sized OUT-OF-SAMPLE, through the production forward, Z-RESOLVED, with finite-sample inflation.
Four-way agreement on the components:
- **Size to the WORST-per-sim coherent residual (~0.9%), never the pooled signed mean (+0.01%)** — pooling
  cancels opposite-sign z-bins (Cosmology: the deficit sign-flips, −9.5% z=2.0 → +0.9% z=4.0). All 4 agree.
- **Z-RESOLVED, never pooled** (Cosmology [IMPORTANT], Bayesian, Lyα). Pooling hides the per-z residual.
- **Add finite-sample inflation (~1.2–1.5×)** since n=6 gives a noisy variance estimate (Bayesian
  [IMPORTANT]).
- **Measure it through `MultiFidelity.logP_mf` (incl. tail extrapolation above the LF Nyquist 0.069 s/km),
  not the raw cache**, and report per-band: LF-resolvable (k<0.069) vs extrapolated KODIAQ band (k>0.07) (CS
  [IMPORTANT]).
- **The floor covers LF→HR GENERALIZATION only, NOT HR→truth non-convergence** (Lyα [IMPORTANT]); the k<0.06
  analysis cap is the actual HR→truth protection. State this explicitly; do not imply MF "resolves" the
  low-z physics to truth.

### I3. Budget the MF-correction EDGE/EXTRAPOLATION uncertainty SEPARATELY from the LF-emulator sparse-edge uncertainty in the ns>0.995 step-C_emu.
Raised by **all four**, strongest from **Cosmology** and **Lyα**. The 6 HR sims span only ns∈[0.859,0.979];
the real fit lands near ns~1.0 (eBOSS 1.009), i.e. AT/ABOVE the upper edge where the MF correction was ever
measured — extrapolating *in the tilt direction* (the one we cannot afford to get wrong). The step-inflated
C_emu above 0.995 was sized for the LF-emulator's 2/60-point sparsity, a **different object**. The real fit
hits BOTH the MF coverage gap AND the sparse LF extension simultaneously — the highest-risk corner. The MF
correction must NOT be certified globally from the 6-sim LOSO; tie an MF-extrapolation inflation to ns
outside [0.86,0.98]. (Optional but ideal: acquire ≥1–2 HR sims near ns~0.82 and ns≥1.0; otherwise document
the MF-edge as extrapolation in the blinding gate.)

### I4. Resolve the gbar-vs-res_corr double-counting question before the cert.
Raised by **Cosmology** [MINOR]. Verified in `multifidelity.py`: the forward is
`P_MF = (rho(k,z)·f_LF + delta)·res_corr`, applying BOTH the LF→HR box-resolution ratio (gbar/rho, the object
all these diagnostics measure) AND the L15 384/512 particle-convergence `res_corr` table. The docstrings call
these distinct (box-resolution vs particle-load), but it is not stated whether the LF→HR ratio already
subsumes part of the particle convergence. **Confirm they correct DIFFERENT physical effects and are not both
applied to the same convergence gap; if they overlap, drop/rescale one.** Document in `multifidelity.py` and
the real-fit plan. Cheap to confirm, expensive if wrong (double-corrected power → biased n_s).

---

## 5. DISSENTS (surfaced, unresolved)

- **Separable vs 2-D (z,τ₀) table for the MF correction.** **CS** (and Bayesian, Lyα) insist on
  *separable* (`gbar_z + gbar_tau`) to prevent the n_s-tilt→τ₀-axis aliasing / mlp-head over-fit.
  **Cosmology** explicitly calls for a *2-D (z,τ₀) table or a low-order (z×τ₀) interaction term, NOT a
  separable product*, because the deficit has a real **(z,τ₀) sign-flip interaction** (z=5.0 shows ρ<1 at
  high τ₀) that a separable product *cannot represent*. **This is a genuine, unresolved physics-vs-overfit
  tension.** My read: CS's anti-aliasing concern is about *degrees of freedom vs 6 sims*, and Cosmology's is
  about *representational adequacy for a real interaction*. The synthesis is a **constrained low-order
  interaction**: `log_rho(k) + gbar_z(z,k) + gbar_tau(τ₀,k) + λ·b(z)·b(τ₀)` with a SINGLE low-order
  interaction coefficient (or a rank-1 term), NOT a free (z,τ₀,k) table — enough to capture the sign-flip,
  too few DOF to alias the tilt. **This must be decided explicitly in the MF-wiring plan, with the n_s
  Fisher check through the MF forward (I1) as the arbiter.** Whichever form is chosen, it does not pass
  until that check is clean.

- **Severity of the z_lo default mismatch.** Bayesian (MINOR — "implemented default not directly measured")
  and CS (IMPORTANT) vs Cosmology/Lyα (IMPORTANT, recommend flipping to 2.8). I resolved this in §2/§3 by
  escalating to a CRITICAL blocker on the *certification gap* (C1) and treating the *default value* as a
  required change (C2) — the substantive recommendation (default 2.8) follows Cosmology/Lyα.

- **Whether the low-z deficit z-trend is externally corroborated.** Lyα [MINOR] notes the
  worse-at-low-z trend disagrees with PRIYA (peaks at HeII z~3–4) and Khan+2023 (peaks at high z), so the
  package's "matching PRIYA worst at z~3–4" framing conflates two drivers. Cosmology agrees the framing
  conflates them. This is a *prose/framing* dissent, not a numbers dissent (the measurement is internally
  self-consistent and the τ₀/⟨F⟩ split confirms it). → Reframe as "transmissivity-driven, measured cleanly,
  internally self-consistent," not "expected from the literature." Not blocking.

---

## 6. ORDERED PRE-MF-WIRING TO-DO

1. **[C1, cheap]** Run ONE all-folds (60-sim) closure config at the z_lo that will ship
   (`run_config({k_max:0.06, z_lo:<2.8 if adopting §2>}, folds=range(8))`, forward-only, no retrain); record
   it next to the committed z_lo=2.6 all-folds row (+0.038σ). The production default must have a 60-sim
   number, not a 30-sim subset one.
2. **[C2]** Set `load_ks_leg` default to **z_lo=2.8** (the published KODIAQ-SQUAD cut); expose z_lo=2.4 as an
   opt-in sensitivity. Reconcile the docstring, the plan RESOLVED text, and MEMORY so they no longer cite
   the z<2.8 paper cut to justify a non-2.8 default. (Verified inconsistency: `data_likelihood.py` lines
   159–161.)
3. **[I4, cheap]** Confirm gbar/rho (LF→HR box-resolution) and res_corr (L15 particle-convergence) correct
   DIFFERENT effects and are not double-applied to the same gap; document the decision in `multifidelity.py`
   and the real-fit plan. Drop/rescale if they overlap.
4. **[prose, cheap]** Fix the handoff/plan prose to the verified numbers: "KS z2.0+z2.2 = −0.68σ ≈ the entire
   −0.65σ; DESI is +0.04σ (clean)" (drop the imprecise "86%"); reword MF "θ-stable (0.6%)" → "θ-insensitive
   WITHIN the 6-sim cluster; globally untested (necessary-not-sufficient)"; reframe the low-z deficit as
   "transmissivity-driven, internally self-consistent" not "expected from PRIYA." Add the one-line note to
   `make_legb_golden.py` that the leg-mean dla_core shifts retained-row P slightly when the z-range changes
   (benign; DESI stays byte-identical, the invariant that matters).
5. **[MF-wiring plan, I1]** Specify the MF correction form BEFORE coding: `log_rho(k) + gbar_z(z,k) +
   gbar_tau(τ₀,k)` **plus at most a single low-order (z×τ₀) interaction term** (resolving the §5 dissent),
   never a free (z,τ₀,k) table. Make the n_s Fisher-bias check THROUGH `MultiFidelity.logP_mf` (not the raw
   cache) the adoption gate.
6. **[MF-wiring plan, I2]** Specify the C_emu small-scale floor as: worst-per-sim coherent residual,
   Z-RESOLVED (never pooled), through the production forward (incl. tail extrapolation), reported per-band
   (k<0.069 vs k>0.07), with ~1.2–1.5× finite-sample inflation, stated to cover LF→HR generalization ONLY
   (k<0.06 cap covers HR→truth).
7. **[MF-wiring plan, I3]** Budget the MF-correction extrapolation uncertainty for ns outside [0.86,0.98]
   SEPARATELY from the LF-emulator sparse-edge step-C_emu above 0.995. Flag in the real-fit/blinding report
   that any posterior mass above ns~0.98 rests on MF extrapolation.

Items 1–4 are the actual pre-MF-wiring gate (cheap, days). Items 5–7 are binding inputs to the MF-wiring
plan that must be written into it before that task starts.

---

## 7. BOTTOM LINE

The science is sound and was independently reproduced on every load-bearing claim. The −0.65σ bias is
correctly root-caused, exactly decomposed, and the closure-vs-real-fit distinction is right. The resolution
is a **GO_WITH_CHANGES**: ship after closing two cheap CRITICAL items (the missing all-folds certification of
the shipped z_lo, and the z_lo default/docstring reconciliation toward the published 2.8 cut), and carry the
four MF-wiring design constraints (separable+low-order correction with a through-forward Fisher gate;
out-of-sample z-resolved per-band inflated C_emu floor; separate MF-extrapolation budget at the ns edge; and
the gbar/res_corr double-counting confirmation) into the MF-wiring plan as binding requirements. No NO_GO
findings; the one substantive disagreement (z_lo=2.4 vs 2.8) resolves cleanly to 2.8 because the closure is
blind to the difference and the marginal constraining power is ~0.
