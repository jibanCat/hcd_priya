# Cosmology / P1D checkpoint review — HCD slope-prior spec — 2026-06-06

Lens: does the HCD-incidence z-slope nuisance design protect the recovered cosmology
(A_p, n_s)? Reviewed: the spec `docs/superpowers/specs/2026-06-06-hcd-slope-prior-spec.md`,
the sweep `scripts/diag_legb_slope_prior_tradeoff.py` + its results
`figures/analysis/05_likelihood/legb_slope_prior_tradeoff.txt`, the α(z) site
`closure_legb.py:432-456`, the incidence prior `inference.py:54-106`, the forward
`data_likelihood.py:294-378`, and the dN/dX fit `scripts/plot_dndx_vs_literature.py`.

## Verdict summary

**GO-WITH-CHANGES.** The factorized α(z) = amplitude × known-shape × sampled-slope design
is the right structure and the ×1.04 σ(A_p) cost is credible. BUT the spec mis-frames its own
safety case: (a) it headlines the A_p bias (+0.167σ) while the SAME sweep shows the **n_s bias
is −0.223σ at lit width — already OVER the 0.2σ gate**, unmentioned in the spec; (b) the bias is
**width-INSENSITIVE** (it is +0.185σ at the narrowest prior and only +0.167σ at lit — the slope
WIDTH is a weak lever on bias, so "narrow the prior to be safe" does the opposite); (c) the whole
bias table is computed at a **single fiducial whose n_s sits ON the box edge** (n_s_unit=0.013 →
n_s=0.803, the lower boundary), exactly the COS-R5 edge-pileup caveat, so the n_s bias number is
the least trustworthy of all. The production-faithful g_fixed (Q1) is REQUIRED, not optional, for
the coverage to mean anything for the real A_p. These are fixable in the spec + one cheap rerun.

---

## Q1 — Should the closure forward use the PRODUCTION g_fixed(z) (lit-derived), with held-out sim truth as the thing coverage must survive? (PARTLY MY LENS)

**YES, REQUIRED — not merely "recommended" as the spec hedges (line 97). For the A_p/n_s coverage
number to transfer to the real fit, the closure forward MUST use the production-faithful
g_fixed.** This is my onboarding R1 verbatim (`onboarding-cosmology.md:221-241`): the closure can
be made to "pass" by matching the SIM's own w_c(z), while production runs the literature dN/dX
shape (`lit_over_sim_at_z`, `inference.py:77-83`) — certifying a shape production never executes.

The cosmology argument is sharp and specific:
- The low-k excess that the HCD incidence injects is the **A_p/n_s regime** (A_p is a z-flat low-k
  boost; n_s pivots at k≈0.005 s/km). A z-shape error in α(z) tilts that low-k excess **in z**,
  which the diagonal-in-k (and even cross-class-k-diagonal) C_emu cannot whiten — it leaks onto
  A_p amplitude AND n_s tilt through the α↔cosmo degeneracy (the buggy pilot: A_p ≈10σ low).
- If the closure g_fixed = sim w_c(z) (the spec's "closure-only" framing, lines 83-87), then
  s_c=0 is the +0.03σ EXACT operating point **by construction** — the slope nuisance has nothing
  to absorb, the marginalization is a no-op robustness buffer, and the coverage certifies a model
  with the sim's z-shape baked in. The real fit's g_fixed is the lit shape, which differs from the
  real data's true (unknown) z-shape by the real-lit gap — UNCERTIFIED.
- The honest construction (spec lines 88-93): g_fixed = lit-derived shape (via `dndx_wc.w_c_from_mu`,
  the SAME map the amplitude prior uses), and the sim-truth differs from g_fixed by the
  **sim-vs-lit shape gap** — which the slope nuisance + per-leg anchor must then absorb. Then the
  closure exercises EXACTLY the production forward, and s_c=0 is NOT free — the sweep's bias
  (+0.167σ A_p / −0.223σ n_s) is precisely the residual the marginalization must cover. That is
  the number that means something for the real A_p.

**Caveat the spec must state:** even production-faithful, the closure's TRUTH is one held-out
SIM, whose w_c(z) is the sim's (≈2.6 abs slope), not the real data's. So the closure tests
"can the sampler recover A_p when g_fixed(lit) is wrong by the sim-vs-lit gap?" — a proxy for
"...wrong by the real-vs-lit gap." This is the best available test (no real HCD truth exists), but
the coverage number is conditional on sim-vs-lit gap ≈ real-vs-lit gap. Report it as such.

**Verdict: Q1 = production-faithful g_fixed is REQUIRED.** The sweep itself already adopted this
framing (`diag_legb_slope_prior_tradeoff.py:5-8`: "g_fixed = w_c,sim(z)" — WAIT, the sweep uses
the SIM shape, line 90 `g_fixed = wc_perz[sel_sim]/wc_pivot`). **So the sweep numbers were
computed in the closure-only (sim-shape) framing, NOT the production framing.** This is a real
inconsistency: the locked widths were measured against the sim w_c(z), but the spec's preferred
framing (and the only one that transfers) is lit-derived g_fixed. The sweep must be re-run with
the production g_fixed before the bias/variance numbers can be quoted as production-relevant
(see Q3 — same rerun).

## Q2 — Where does s_c CENTER (WLS lit 0.76/0.05 vs code 0.95/0.15), fixed or hyper-prior? (DEFER primary to Bayesian/Lyα; cosmology position below)

The provenance: the code constants `HCD_LIT_OVER_SIM_SLOPE = (0.95, 0.15, 0.40)`
(`inference.py:74`) come from an **unweighted** `np.polyfit` over the lit dN/dX
(`plot_dndx_vs_literature.py:102-108`, `gl[0]-gs[0]`, no error bars). The spec's WLS center
(0.76/0.05/1.16) is a weighted re-fit propagating the quoted dN/dX errors. The WLS value is the
better estimate (it down-weights the noisy high-z points the code-comment itself flags as
unreliable, `inference.py:71-73`).

**Cosmology position:** the center choice (0.76 vs 0.95 on LLS) is a 0.19 shift on the slope, well
inside the 0.52 lit width — so the CENTER barely moves σ(A_p) (the sweep shows σ(A_p) flat to
0.4% across the entire width range). The center matters for the BIAS, not the variance: the bias
is `Cpost[A_p,s_c]·Pp[s_c]·(s_center − s_true)`. The sweep tested the offset = the **sim-vs-lit
gap** (0.76 LLS, `SIM_LIT_GAP`, line 50), independent of where the prior centers. So the center
choice does not change the headline bias as long as the truth-offset is measured consistently.

**Fixed vs hyper-prior:** a tighter hyper-prior on the center is over-engineering here. The slope
is the *correction* to a known shape; the sweep shows it barely competes with A_p (×1.04). Adding
a hyper-layer adds a funnel (BAY-R6) for ≈zero cosmology benefit. **Fixed constant at the WLS
center.** Defer the WLS-vs-polyfit numerical adjudication to the Lyα lens (it owns the dN/dX fit);
from cosmology, either center is fine for σ(A_p) — use the WLS one for correctness, keep it fixed.

## Q3 — Is the Fisher bias (0.167σ) trustworthy to SET the prior, or compute the exact construction bias first? (MY LENS — sharpest finding)

**NO, not as currently reported — recompute it, AND the 0.167σ framing is doubly misleading.**
Three cosmology-specific problems, in order of severity:

**(1) The bias is width-INSENSITIVE — the spec's safety logic is inverted.** I re-read the sweep
table (`legb_slope_prior_tradeoff.txt`): the A_p bias is **+0.185σ at the NARROWEST prior
(m=0.10, σ_s=0.05)** and DROPS to +0.167σ at lit (m=1), then to +0.056σ only at 5× lit. The total
bias swing across the entire scanned width range is just 0.129σ. This is textbook Fisher behavior:
tightening the slope prior pins s_c harder → the slope absorbs LESS of the truth-offset → MORE
leaks to A_p. So "the lit width keeps bias < gate" (spec line 48-49) is fragile: the bias is set
by the GAP, not the width, and you cannot narrow your way to safety — you must WIDEN. The spec's
own "free headroom to go wider if desired" (line 49) is actually the only bias-reducing lever, and
should be reframed as the recommended direction, not optional.

**(2) The n_s bias is −0.223σ at lit width — OVER the 0.2σ gate — and the spec never mentions it.**
Same table, column 6: bias(n_s)/σ = **−0.235σ at narrow, −0.223σ at lit**, only crossing under the
gate at ≈2× lit width. The spec §2 reports ONLY the A_p bias and asserts "σ(n_s) ≈ flat" (line 45)
— true for the VARIANCE, but silent on the n_s BIAS, which is LARGER in magnitude than A_p's and
violates the same 0.2σ gate the spec invokes. n_s is a cosmology headline. This omission must be
fixed: the gate is failed on n_s at the locked width.

**(3) The single fiducial sits on the n_s box edge.** The fiducial is n_s_unit=0.013 → n_s=0.803,
**hard on the lower box boundary** (box [0.8,1.05]); A_p_unit=0.717 is interior. The Fisher has no
boundary, so for n_s the bias/σ ratio is computed against a σ(n_s) the boundary would shrink — the
n_s number is the least reliable in the whole table (COS-R5). The A_p number (interior) is the
trustworthier of the two, but is still one fiducial. The onboarding already flagged the σ=2%
multi-z mis-centering bias sits AT 0.2σ (`onboarding-cosmology.md:283-289`); this sweep is a
SECOND, independent path to the same gate, and it crosses it on n_s.

**(4) The linearization is 13% (DESI)/15% (KS) off** (sweep lines 6-8), over a large slope step;
`(1+z)^s` is nonlinear over the gap. 15% on a 0.167σ A_p bias is ±0.025σ — not enough to flip A_p
across the gate, but the n_s bias is already over the gate, and the linearization could be a
different sign/magnitude there.

**Recommendation (cheap, forward-only — the spec already says it is):** before setting the prior,
re-run the sweep (i) with the **production lit-derived g_fixed** (Q1 — fixes the framing
inconsistency), (ii) computing the **EXACT non-linearized construction bias** ΔP(s=gap)−ΔP(0)
projected through (F+P)⁻¹ rather than J_s·gap, (iii) reporting **n_s bias alongside A_p**, and
(iv) at **≥1 interior-n_s fiducial** (a held-out sim with n_s_unit ∈ [0.3,0.7]). If the exact
interior-n_s n_s-bias is also > 0.2σ, the locked width must be WIDENED (the bias-reducing
direction), not the "free headroom" footnote. This is hours of forward-only compute and gates the
whole inference; do it before NUTS.

## Q4 — Per-leg anchor float: amplitude (as specced) or also slope? (MY LENS)

**AMPLITUDE only, as specced — confirmed correct from cosmology.** The cosmology-relevant logic:
- The sweep is unambiguous that the SLOPE barely competes with A_p (σ(A_p) ×1.04 fully
  marginalized; the fixed steep g_fixed carries the z-leverage). A per-leg slope float would add
  per-leg degeneracy with A_p for ≈zero systematic-absorption benefit — the slope is already a
  global nuisance absorbing the lit-vs-truth shape gap.
- The PI's physical concern (spec §3, KS biased-high LLS/subDLA from HR selection) is an
  **incidence-AMPLITUDE** effect: a survey selects more/fewer absorbers per sightline → shifts
  α_pivot, not the z-evolution of the ratio. A z-SLOPE difference between KS and DESI would require
  the two surveys to disagree on dN/dX(z) EVOLUTION, which is not the stated concern and is not
  supported (both anchor to the same lit dN/dX z-fit).
- Cosmology cost of getting it wrong: a per-leg slope float opens a second per-leg low-k-tilt
  direction that competes with n_s (the tilt parameter) on each leg independently — directly
  eroding n_s, which is already the weaker-protected of the two (Q3). Amplitude float competes
  mainly with A_p (the boost), which the sweep shows is cheap (×1.04). So amplitude-only is also
  the n_s-protective choice.

**One Q4 sub-point the spec should add:** the per-leg amplitude floats (KS σ_anchor≈0.25-0.30,
DESI≈0.10-0.15, spec §3) are NOT literature-measured (spec line 75 admits this). Since the bias is
LLS-amplitude-driven and the binding leg is DESI low-k (R2), the DESI σ_anchor=0.10-0.15 is the
one that matters for A_p — confirm in the rerun that floating it does not itself inflate σ(A_p)
materially (a wide DESI amplitude float would re-open the α↔A_p degeneracy the slope nuisance was
meant to be cheap about).

---

## Errors or risks in the LOCKED parts

**ERROR 1 (must fix the spec, not just the prior): the n_s bias violates the gate and is
undocumented.** The spec §2 (lines 43-49) locks the width on the A_p bias (0.167σ < 0.2σ) and the
A_p variance (×1.04), and explicitly states "σ(n_s) ≈ flat" — but the same sweep
(`legb_slope_prior_tradeoff.txt` col 6) shows bias(n_s)/σ = −0.223σ at the lit width, OVER the
0.2σ gate, and −0.235σ at narrow. n_s is one of the two cosmology headlines and is blinded equally
(`onboarding-cosmology.md:144-152`). The locked claim "keeps bias < gate" is FALSE for n_s. This
is a real error in the LOCKED §2.

**ERROR 2 (framing, in the LOCKED part): the width is the wrong lever for the bias.** §2 presents
the lit width as the bias-safe choice; the sweep shows the bias DECREASES with WIDER priors. The
locked "width = lit 1σ with z-edge inflation" sits at +0.167σ/+0.223σ; the edge-inflated width
(σ_s^LLS≈0.65) gives +0.158σ A_p (sweep line 38) — marginally better, still −0.223σ-ish on n_s.
The lock is defensible for VARIANCE but the bias rationale is backwards.

**NOT an error but a framing inconsistency (see Q1/Q3):** the sweep that produced the LOCKED widths
used the SIM-shape g_fixed (`diag_legb_slope_prior_tradeoff.py:90`), while the spec's preferred and
only-transferable framing is the lit-derived g_fixed (lines 88-93). The locked widths were measured
in a framing the spec itself says is "not the production config." Re-measure in the production
framing before locking.

## Additional findings from this lens

1. **The interaction with the unbuilt correlated-in-k C_emu (spec §5) is the real cosmology
   exposure, and it is under-stated.** The slope nuisance and the correlated-in-k C_emu are TWO
   tools for the SAME coherent low-k mode (COS-R1/R2; meta §5 option C). The spec defers the
   correlated C_emu to "out of scope" (line 111) — but the bias the slope nuisance leaves on
   A_p/n_s (+0.167/−0.223σ) is precisely the residual that the correlated C_emu is supposed to
   absorb. With the slope nuisance ALONE (this spec) and a diagonal C_emu, the residual coherent
   low-k tilt is marginalized only through the 3 slope params — which the sweep shows compete
   weakly. **The two must be co-designed, or the slope nuisance + diagonal C_emu leaves the n_s
   bias over the gate.** Recommend the spec state that the locked width is provisional until the
   correlated-in-k C_emu is built and the n_s bias re-measured WITH it (the C_emu, by whitening
   the coherent mode, should pull the n_s bias under the gate — that is the real fix, not the
   slope width).

2. **The low-k DESI z~3.2 regime (C_emu≈0.5·C_data, COS-R2) is exactly where the slope nuisance
   bites.** The slope nuisance tilts the low-k excess in z; the DESI low-k bins (k<5e-3, z~3.2)
   are both where A_p/n_s constraining power concentrates AND where C_emu is a half-sized budget
   contribution carried as DIAGONAL variance against a COHERENT residual. So the slope nuisance is
   operating in the one regime where the covariance model is weakest. The sweep's Fisher used the
   x-class (still k-diagonal) C_emu (`diag_legb_slope_prior_tradeoff.py:124-127`, `rho_zb`), so the
   ×1.04 and the bias are computed against a C that cannot whiten the coherent mode — the TRUE
   σ(A_p) with a correlated C_emu could be larger and the bias smaller. Another reason the locked
   numbers are provisional.

3. **The Fisher uses `cemu_inflate=ctx.cemu_inflate` (default 1.0, uncalibrated, meta §3-ledger).**
   If the eventual cemu_inflate > 1, σ(A_p) grows and the bias/σ ratio shrinks — so the locked
   bias/σ is, if anything, a conservative (high) estimate on that axis. Worth noting but not
   blocking.

4. **n_s edge fiducial → the n_s bias number needs an interior re-check more urgently than A_p.**
   Because the fiducial n_s is on the boundary, the n_s σ in the denominator is the unbounded
   Fisher σ; the realized (boundary-aware) σ is smaller, so bias/σ on n_s is likely WORSE than
   −0.223σ for an interior truth, OR the edge artifact is inflating it. Either way the number is
   uninterpretable at this fiducial — the interior-n_s rerun (Q3) is required specifically to pin
   the n_s bias.

## Concrete recommendations (numbered, actionable)

1. **FIX the spec §2: add the n_s bias.** Report bias(n_s)/σ = −0.223σ at lit (−0.235σ at narrow)
   and state plainly that it EXCEEDS the 0.2σ gate. Do not lock the width on the A_p bias alone.

2. **Re-run `diag_legb_slope_prior_tradeoff.py` with the PRODUCTION lit-derived g_fixed**
   (replace `g_fixed = wc_perz[sel_sim]/wc_pivot`, line 90, with `dndx_wc.w_c_from_mu` of the lit
   dN/dX — the same map `inference.lit_over_sim_at_z`/`hcd_incidence_prior` use). The currently
   locked widths were measured in the sim-shape framing the spec itself rejects (Q1).

3. **In that rerun, compute the EXACT (non-linearized) construction bias** at the lit width — the
   spec says it is cheap and forward-only (lines 100-103); the linearization is 13-15% off
   (sweep lines 6-8). Project the exact ΔP through (F+P)⁻¹ rather than J_s·gap.

4. **Add ≥1 interior-n_s fiducial** (held-out sim with n_s_unit ∈ [0.3,0.7]) to the rerun. The
   current single fiducial has n_s ON the box edge (n_s=0.803), making the n_s bias/σ ratio the
   least trustworthy number in the table. The n_s gate must be checked at an interior truth.

5. **Reframe the bias-reducing lever:** the bias DECREASES with WIDER slope priors (the spec's
   "free headroom," line 49). If the exact interior-n_s n_s-bias is > 0.2σ, WIDEN the width (the
   variance cost is ×1.04 → modest), do not narrow. State this in §2.

6. **State that the locked width is PROVISIONAL on the correlated-in-k C_emu (§5).** The slope
   nuisance + diagonal C_emu leaves a coherent low-k residual on n_s; the correlated C_emu is the
   real whitener for it. Re-measure the n_s bias WITH the correlated C_emu before the final NUTS
   cert — that is the co-design, not two separate steps.

7. **Q1 = production-faithful g_fixed REQUIRED** (not "recommended"). Q2 = WLS center (0.76/0.05),
   fixed constant, no hyper-prior. Q4 = amplitude-only float, confirmed; verify the DESI
   σ_anchor=0.10-0.15 does not itself inflate σ(A_p) in the rerun.

8. **Carry the n_s bias into the Leg-B coverage gate.** The handoff makes coverage≥nominal + bias≈0
   the ONLY valid Leg-B headlines (meta §1-7); the NUTS coverage run must report the n_s bias and
   pass it through the 0.2σ gate, since the forward Fisher already fails it. Do not certify on A_p
   coverage alone.
