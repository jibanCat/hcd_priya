# Bayesian / PPL / inference checkpoint review — HCD slope-prior spec — 2026-06-06

Lens = the numpyro generative model, prior identifiability, posterior geometry, the SBC/coverage
methodology. All file:line refs to the `phase2c-likelihood` branch state on 2026-06-06. I cite the
sweep numbers from `figures/analysis/05_likelihood/legb_slope_prior_tradeoff.txt` directly.

## Verdict summary

**GO-WITH-CHANGES.** The (B+slope) marginalization is the statistically correct fix and the LOCKED
width is sound. But three things must change before this can certify the real fit: (1) Q1 — the sweep
*as written* uses the SIM's own `g_fixed` (closure-only framing), so the headline numbers DO NOT yet
demonstrate the production-faithful closure the spec recommends; this must be re-measured. (2) The
factorized `α_pivot·exp(δ_leg)·g_fixed·exp(lnz·s)` is a **product of four sampled terms with a
near-flat amplitude direction (α_pivot·exp(δ_leg))** — this needs an explicit identifiability anchor
and a non-centered parametrization before the NUTS cert, or the ~28-dim posterior will funnel and the
coverage verdict will be uninterpretable. (3) Q3 — a Fisher bias is NOT adequate to SET the prior at
the 0.167σ level given the 13–15% linearization error AND the ignored prior boundary/non-Gaussianity;
compute the exact forward bias first (cheap).

---

## Q1 — production-faithful `g_fixed` vs the sim's own shape (THIS IS MY QUESTION)

**Position: YES, certify on a production-faithful `g_fixed` (lit-derived shape), with the held-out
sim-truth as the survival target. The spec's recommendation is correct, and it is the only framing
that makes the Leg-B coverage number mean what we need it to mean. But note the sweep does not yet
implement it.**

The decisive observation, and the one thing the checkpoint must not miss: **the sweep that produced
the LOCKED numbers uses the SIM's own shape, not the production shape.**
`scripts/diag_legb_slope_prior_tradeoff.py:90` sets `g_fixed = wc_perz[sel_sim] / wc_pivot[None,:]`
— i.e. `w_c,sim(z)/w_c,sim(z_p)`, the held-out sim's own incidence — and the docstring
(:4-8) says so explicitly: "g_fixed_c(z) = w_c,sim(z)/w_c,sim(z_p) is the KNOWN steep abundance shape
threaded from the truth." So the sweep IS the closure-only framing of §4, the one the spec's §4-Q1
recommends AGAINST. In that framing `s=0` is the +0.03σ exact operating point by construction, exactly
as the onboarding R1 warned ("self-fulfilling construction": the forward and the mock truth share the
z-shape, so coverage on s is a buffer around an already-correct mean).

Why this is the statistically wrong certification (connecting to onboarding R1): a closure that pins
the forward shape to the truth's shape validates the SAMPLER given a correct mean model. It is
**necessary but not sufficient**. The real fit will NOT have `w_c,sim(z)` — it will have a
literature-derived `g_fixed` whose slope (the lit ratio-slope 0.15–0.95, plus whatever the WLS center
is) differs from the sim's absolute slope ~2.6. The quantity that has to survive coverage is the gap
between THAT production `g_fixed` and the sim truth, marginalized over `s`. If we certify on the sim's
own `g_fixed`, we certify a configuration that will never run.

The correct construction, which the spec §4-Q1 names and I endorse:
- forward uses `g_fixed,c(z)` = lit dN/dX shape via `dndx_wc.w_c_from_mu` (the SAME map as the
  amplitude prior — this is the consistency the spec wants);
- the mock truth is the held-out sim (steep slope ~2.6), DLA-masked per §0c;
- `s_c` is sampled with the lit-width prior; `δ_leg` floats the amplitude;
- the verdict = coverage ≥ nominal + bias ≈ 0 **on this gap**, NOT on a zero gap.

**One consequence the spec must absorb (this is the load-bearing caveat):** if the forward uses the
lit `g_fixed` while the truth carries slope ~2.6, then **`s=0` is no longer the exact case**. The
residual slope mismatch is roughly (sim 2.6) − (lit center) ≈ 1.8–2.5 in `s` units, which is FAR
outside the lit width σ_s≈0.52–0.65. The slope nuisance with the lit width **cannot absorb a 3–5σ_s
slope offset.** That is precisely R1's deepest point: the lit width is the right width for the
*nuisance uncertainty* but is NOT wide enough to bracket the full sim-vs-lit shape gap. So one of two
things must be true for the production-faithful closure to pass:
  (a) the construction residual after the lit `g_fixed` (NOT the sim `g_fixed`) is already small at the
      truth θ — i.e. the lit shape is "close enough" to the sim shape on the DESI low-k bins that the
      A_p leakage is < gate even at s=0; OR
  (b) it is NOT small, and then the correlated-in-k C_emu (spec §5, plan §0b-2) is REQUIRED to whiten
      the residual coherent mode — the slope nuisance alone will not do it.

The forward-only check that settles this is cheap and MUST be run before the cert:
re-run `diag_legb_zresolved_alpha_check.py` (the −0.78→−0.24→+0.03σ ladder) with `g_fixed` set to the
**lit** shape instead of the sim shape, and read the whitened low-k construction Δ at the truth. The
existing ladder's −0.24σ ("LIT-shape α(z)") is the closest proxy we have and it is OVER the practical
budget — which already signals (b). The spec's Verified-vs-Assumed §6 lists "variance ×1.04; bias
0.167σ" as VERIFIED, but those were measured in the closure-only (sim-`g_fixed`) framing; **they are
NOT yet verified for the production-faithful framing the spec recommends.** That mismatch between what
was measured and what is recommended is the single most important thing for the meta-reviewer.

Defer the physics of WHICH lit shape (Rogers plateau, dndx_wc map) to [LYA]/[COS]; the Bayesian
verdict is: certify on the production `g_fixed`, treat the sim truth as the survival target, and
re-measure the bias/variance in that framing — the locked numbers do not yet describe it.

## Q2 — where does `s_c` center, and fixed-constant vs hyper-prior?

**Position: center `s_c` at 0 in the production-faithful framing (the prior is a correction on top of
the lit `g_fixed`, so its natural center is "no correction"); do NOT center at the WLS 0.76 nor the
code 0.95. Keep it a FIXED constant, NOT a hyper-prior, for the cosmology cert.**

First, the WLS-vs-code discrepancy is real and I verified its provenance. The code constants
`HCD_LIT_OVER_SIM_SLOPE=(0.95,0.15,0.40)` (inference.py:74) come from an **unweighted** `np.polyfit`
in `plot_dndx_vs_literature.py:102,104` (`gl[0]-gs[0]`, OLS in log-log, no error bars). The spec's
`S_CENTER=(0.76,0.05,1.16)` (sweep :48) is a WLS re-fit "with the quoted error bars." These differ
because OLS and WLS weight the z-points differently; the LLS 0.95→0.76 shift is ~0.2 in `s`, i.e.
~0.4σ_s — non-negligible. The code constant is the less-defensible of the two (it ignores the
measurement errors that the width is derived from), so if a literal lit-slope is ever used as a fixed
shape, prefer the WLS value.

But the deeper point is that **the center question is conflated between two different objects in the
spec**, and the answer depends on which `g_fixed` framing wins Q1:
- In the **closure-only framing** (sweep as written), `s_c` is the full ratio-slope and its center is
  the lit slope (WLS 0.76). The sweep centers the *prior* at `s=0` but tests a truth offset by the
  full gap `SIM_LIT_GAP=(0.76,0.05,1.16)` (:50, identical to S_CENTER) — i.e. it bakes the lit slope
  into the bias test, not the prior center. That is internally consistent for measuring robustness.
- In the **production-faithful framing** (recommended), `g_fixed` ALREADY encodes the lit shape (it is
  built from `w_c_from_mu` of the lit dN/dX), so `s_c` is the *residual correction* on top of it. Its
  honest center is **0**: "the lit shape is my best estimate; the correction is symmetric." This is the
  spec's own observation at §4 line 92-93 ("the slope prior CENTER then sits at the
  lit-vs-(forward-g_fixed) differential (likely ~0 if g_fixed already encodes the lit shape)"). I agree
  with that line and it should be the locked answer.

So: **center = 0 in the production framing.** Centering at WLS 0.76 in the production framing would be
double-counting the lit slope (once in `g_fixed`, once in the center). Centering at the code 0.95 is
strictly worse (unweighted fit). This is not a free parameter — it follows from the factorization.

**Fixed constant vs hyper-prior.** Make it a fixed constant. A hyper-prior on the center (a
hierarchical `s_c ~ Normal(μ_s, τ_s)` with `μ_s` itself sampled) buys nothing here and costs
identifiability: with only ~12 DESI z-bins per leg and the slope being nearly degenerate with A_p at
low-k (the sweep shows the slope barely competes — variance ×1.04 only), the data cannot
inform a hierarchical center; the hyper-prior would just shift weight between `μ_s` and `s_c` along a
flat direction, adding a second funnel-prone level (the classic "centered hierarchical" funnel) to an
already-funnel-prone product. The marginal posterior on A_p would be essentially unchanged but the
geometry would be worse. A fixed center + a fixed (lit) width is the right amount of structure: it is a
weakly-informative prior, fully specified, no extra latent. If the PI wants robustness to the center
itself, the cheaper and more honest move is to WIDEN the fixed width (the z-edge inflation already does
this), not to add a level.

## Q3 — is a Fisher bias adequate to SET the prior?

**Position: NO, not at the 0.167σ level. The locked width is fine as a width, but the bias number that
justifies "under the 0.2σ gate" must be recomputed exactly (forward-only, cheap) before it is used to
certify. The Fisher bias is a ranking tool, not a gate-deciding number, for three independent reasons.**

Reason 1 — the linearization is measured to be 13–15% off, ON THE QUANTITY THAT SETS THE BIAS. The
sweep's own cross-check (`legb_slope_prior_tradeoff.txt:6-8`) reports
`||J_s·gap − (P(gap)−P(0))|| / ||P(gap)−P(0)|| = 0.129 (DESI), 0.152 (KS)`. The Fisher bias is
`(F+P)⁻¹ P (s_center − s_true)` linearized through exactly that `J_s·gap` (sweep :135, :17). A 13–15%
error in the response over the gap propagates ~linearly into the bias estimate, so the true bias could
plausibly be 0.167σ × (1 ± 0.15) ≈ 0.14–0.19σ from this effect ALONE. That already brushes the 0.2σ
gate. You cannot SET a gate-passing prior on a number whose own validity check says it is 15% wrong in
the regime being tested.

Reason 2 — the Fisher bias ignores the prior BOUNDARY, and the binding prior here is the tightest one.
The LLS amplitude prior is σ/μ=0.15 (inference.py:58, "LLS TIGHT") — the most informative,
most-cosmology-relevant prior in the model. The Fisher bias formula `(F+P)⁻¹ P Δ` is the exact MAP
shift ONLY for a Gaussian likelihood × Gaussian prior with no boundary. But θ is `Uniform^9` with an
auto-bijector (closure_legb.py:438), the α's are Gaussian but the DLA is softplus-one-sided
(closure_legb.py:447-449), and n_s sits near a box edge (onboarding C1/R7). Near a tight prior + a box
edge the posterior is asymmetric, and the *mean* (which coverage is about) is not the *MAP shift* the
Fisher computes. The 0.167σ is a MAP-shift surrogate for a coverage quantity; they coincide only in the
Gaussian-interior limit, which is exactly where we are NOT (A_p sits on the (A_p, α_LLS) degeneracy
ridge, onboarding §2.5).

Reason 3 — coverage is about the WHOLE interval, not the point bias. Even if the bias were exactly
0.167σ, a +0.167σ mean shift with a slightly-too-narrow interval can UNDER-cover. The Fisher gives σ(A_p)
and bias separately but does not give the coverage probability, which is the actual Leg-B verdict
(onboarding §2.8). 0.167σ "under the 0.2σ gate" is reassuring as a ranking but is not a coverage
statement.

What to do (cheap, the spec already names it as Q3): compute the **exact forward construction bias**.
Build P at the truth θ with `s=0` and with `s=gap`, whiten by C_total, project onto the A_p Fisher
direction `(F+P)⁻¹ J_Ap^T Cinv`, and read the exact shift. This is forward-only, ~minutes (the same
machinery the sweep already has at :136-138 — it computes `P_s` and `P_0` for the cross-check; just
project them onto the A_p direction instead of taking the norm). If the exact bias is < 0.15σ, lock the
width. If it is 0.18–0.25σ, the width is NOT free and you need either a wider prior or the correlated
C_emu. **Do not lock the width on the linearized 0.167σ.**

## Q4 — anchor float on amplitude, slope, or both?

**Position: AMPLITUDE only, as specced. This is the identifiable, non-degenerate lever, and adding a
per-leg slope float would create an unidentifiable second amplitude-like direction. Confirmed.**

The sweep is unambiguous that the bias is amplitude-driven and the slope barely matters: at the lit
width the A_p bias is +0.167σ, decomposed +0.106σ LLS / +0.004σ subDLA
(`legb_slope_prior_tradeoff.txt:37`); subDLA's gap is ~0 (0.05) so its slope is essentially data-set.
And the variance cost of marginalizing the slope is only ×1.04 (σ(A_p) 0.390→0.405, :10) — the slope
"barely competes with A_p" (spec §2). So the corrective lever the closure needs is the amplitude, and
the per-leg `δ_leg` float on the amplitude is the right and sufficient knob.

Adding a per-leg SLOPE float would be actively harmful to identifiability. The factorization already
has `α_pivot,c · exp(δ_{c,leg}) · g_fixed · exp(lnz·s_c)`. A per-leg slope `s_{c,leg}` would mean each
leg gets its own (amplitude-offset, slope-offset) pair against a shared `α_pivot,c`. With ~12 DESI
z-bins and the slope-A_p near-degeneracy, the per-leg slope is informed almost entirely by its prior
and trades off against both `δ_leg` and the global `s_c` — a flat ridge. It adds 2 latents (LLS+subDLA
× 2 legs = up to 4 with the global slopes) that the data cannot separate, inflating ESS cost and funnel
risk for no bias reduction. Keep the slope GLOBAL (shared across legs, the lit z-evolution is a physical
shape that should not differ by instrument) and float only the amplitude per leg. This matches the
spec.

One refinement: the per-leg σ_anchor values (KS 0.25–0.30, DESI 0.10–0.15, §3) are "defensible
expectations, NOT literature-measured" (spec §3, §6). Because `δ_leg` is the lever that absorbs the A_p
bias, its width directly trades against σ(A_p) and against the bias. **The per-leg anchor needs its own
coverage line in the Leg-B verdict** (does the recovered per-leg amplitude cover the truth?), and a
mini-sweep of σ_anchor like the slope-width sweep would be worth it before locking those numbers — they
are doing real work and are currently un-measured.

---

## Errors or risks in the LOCKED parts

**One real methodology error in the LOCKED measurement (not in the math, in what it certifies):**

The LOCKED variance (×1.04) and bias (0.167σ) numbers were measured with `g_fixed` = the SIM's own
shape (`diag_legb_slope_prior_tradeoff.py:90`, docstring :4-8), i.e. the closure-only framing. The spec
§4-Q1 RECOMMENDS the production-faithful framing (lit `g_fixed`). **The locked numbers therefore do not
describe the recommended configuration.** In the production framing, `s=0` is no longer the +0.03σ exact
point (the sim slope ~2.6 vs the lit center is a 3–5σ_s offset that the lit-width prior cannot absorb),
so both the variance cost and especially the bias will change — likely the bias grows, because the
residual coherent low-k mode reappears (the −0.24σ "LIT-shape" rung of the existing ladder is the
proxy, and it is already over budget). This is not an arithmetic error; the Fisher math, the prior
precision injection (sweep :148-158), and the SPD handling are all correct. It is a "the number proves
the wrong thing" error. **Fix: re-run the sweep with the lit `g_fixed` before locking.** Until then, the
"×1.04, 0.167σ, free" headline should be flagged as closure-only.

Everything else in the locked parts checks out: the width = lit dN/dX 1σ is a defensible
weakly-informative choice; the z-edge inflation (β=1, mean ×1.25) is a reasonable, monotone, smooth
widening (`edge(z)=1+β(clip(2.5−z,0)+clip(z−3.5,0))`, sweep :93) and correctly reuses the existing DLA
inflation slope (inference.py:104); LLS-as-binding is correct (tightest amp prior + biggest gap); the
DLA-MOOT claim is consistent with §0c masking α_DLA→0.

## Additional findings from this lens

**F1 (HIGH) — funnel risk on the product `α_pivot · exp(δ_leg) · exp(lnz·s)`.** This is a product of a
Gaussian amplitude and two exponentials of Gaussians. `α_pivot · exp(δ_leg)` is two amplitude-like
terms multiplying — a textbook funnel/non-identifiability pair (the data constrains the product, not
each factor). And `exp(lnz·s)` × `α_pivot` is the same product structure that the onboarding R3 already
flagged ("α_pivot · g(z) is a product → a classic funnel"). On a ~28-dim posterior with dense_mass and
ESS≈0.18/sample (onboarding R3), a funnel will silently degrade coverage — divergences cluster in the
neck, the retry escalation (closure_legb.py:542-553) re-runs the whole NUTS, and the kept set biases
toward the easy region. **Mitigation: non-centered parametrization.** Sample `δ_leg` and `s` as
standard normals and reconstruct `log α_c(z) = log α_pivot,c + δ_{c,leg} + lnz·s_c` (work in log-space
so the product becomes a SUM, which HMC handles far better — there is no funnel in a sum of Gaussians).
The current code already builds `alpha_hcd` multiplicatively (closure_legb.py:453-454); switching to a
log-space accumulation is a small, differentiable change and is the single highest-leverage geometry
fix. This should be in the implementation plan, not discovered at NUTS time.

**F2 (HIGH) — `α_pivot` and `δ_leg` are jointly unidentifiable WITHOUT the `α_pivot` prior anchoring
them.** The product `α_pivot,c · exp(δ_{c,leg})` is the per-leg amplitude; only the product enters the
likelihood. If `α_pivot` had a flat prior, `(α_pivot, δ_leg)` would slide freely along
`α_pivot → α_pivot·k, δ_leg → δ_leg − log k`. The model is saved ONLY because `α_pivot` has the
informative incidence prior (Normal(μ,σ), σ/μ=0.15 LLS, inference.py:58, closure_legb.py:443-444) and
`δ_leg` has its own Normal(0, σ_anchor). So the two amplitudes ARE separately identified — but the
identification is entirely prior-driven, and the strength of it depends on the relative widths
(σ_LLS=0.15·μ vs σ_anchor=0.10–0.30). If σ_anchor ≳ σ_α the global `α_pivot` loses its meaning and the
per-leg amplitudes float almost freely (re-introducing the very A_p degeneracy this is meant to fix).
**Check before locking σ_anchor: ensure σ_anchor < σ_α per class** so the global pivot stays the
anchor and `δ_leg` is a bounded offset, not a free amplitude. This is a concrete inequality the spec's
§3 numbers should be tested against (LLS: σ_α≈0.15·μ; is σ_anchor=0.25–0.30 in the same units? if
σ_anchor is a *log*-offset and σ_α is a *fractional* width they are comparable, and 0.25–0.30 > 0.15 —
a potential problem). The spec is ambiguous on whether σ_anchor is a log-offset or fractional; pin it.

**F3 (MEDIUM) — the coverage verdict definition must explicitly include the new latents.** Onboarding
§2.8 locks the Leg-B headline as "coverage ≥ nominal + bias ≈ 0," computed by `_aggregate_legb` on the
cosmo+α block (closure_legb.py:621-653). Adding `s_c` (global) and `δ_{c,leg}` (per-leg) means the
packed draws matrix `_draws_matrix` (closure_legb.py:497-505) must be extended, AND the verdict should
report coverage on `s_c` and `δ_leg` too — not just to gate, but to confirm the nuisances are
behaving (a slope that rails to its prior edge, or a δ_leg that rails, is a red flag the amplitude
degeneracy is unbroken, per F2). The PRIMARY gate stays A_p/n_s coverage (the slope is a means, not an
end), but the nuisance coverage is the diagnostic that the marginalization worked rather than just
inflated the interval. **The implementation must add s/δ to the coverage report.**

**F4 (MEDIUM) — adding ~3 LLS+subDLA slopes + up to 4 δ_leg pushes the dim from ~25 to ~30, into the
budget wall.** Onboarding R3: ESS≈0.18/sample, N=600 ≈ 4800 CPU-h > 4000 budget, dense_mass on the
bigger posterior. Each new amplitude-degenerate direction lowers ESS. The non-centered reparam (F1)
partly offsets this (better geometry → higher ESS), but the cert plan should re-profile ESS/sample on a
single mock WITH the new latents before committing N. Do not assume the LF cert's ESS transfers.

**F5 (LOW) — the bias-test offset uses the FULL gap, which is correct for the closure framing but
overstates the production bias.** The sweep tests truth offset by `SIM_LIT_GAP=(0.76,0.05,1.16)` =
S_CENTER (sweep :50). In the closure-only framing where `g_fixed`=sim and the prior centers at s=0,
that is the right offset (the truth's slope-correction relative to the prior center). In the
production framing it would be the sim-vs-lit DIFFERENTIAL, which is the part `g_fixed` does NOT already
absorb — a different (larger, see Q1) number. This is the same Q1 point from the bias-test side: the
offset must be recomputed in the production framing.

## Concrete recommendations (numbered, actionable)

1. **Re-run `diag_legb_slope_prior_tradeoff.py` with `g_fixed` = the LITERATURE shape (via
   `dndx_wc.w_c_from_mu`), not `w_c,sim` (currently line 90), before locking ANY number.** The
   variance ×1.04 and bias 0.167σ are closure-only and do not describe the recommended
   production-faithful config. This is the gating action for the whole spec. (Q1, error in locked part.)

2. **Compute the EXACT (non-linearized) construction bias on A_p at the literature width, in the
   production framing, forward-only.** Reuse the sweep's `P_s`/`P_0` (lines 136-138); project onto the
   A_p Fisher direction instead of taking the norm. Lock the width only if the exact bias < ~0.15σ;
   otherwise widen or require the correlated-in-k C_emu. Do NOT lock on the linearized 0.167σ (15% off,
   ignores boundary + non-Gaussianity). (Q3.)

3. **Center `s_c` at 0 in the production framing (the correction-on-top-of-`g_fixed` interpretation);
   keep it a FIXED constant, NOT a hyper-prior.** If a literal lit-slope is ever needed as a fixed
   value, use the WLS 0.76/0.05/1.16, not the code's unweighted-OLS 0.95/0.15/0.40 (provenance:
   `plot_dndx_vs_literature.py:102-104` is `np.polyfit`, no weights). (Q2.)

4. **Float the per-leg anchor on the AMPLITUDE only, slope GLOBAL.** Do not add a per-leg slope (it is
   an unidentifiable second amplitude-like direction; bias is amplitude-driven; slope cost ×1.04). (Q4.)

5. **Reparametrize NON-CENTERED in log-space:** sample `δ_leg ~ N(0,1)`, `s ~ N(0,1)`, reconstruct
   `log α_c(z) = log α_pivot,c + σ_anchor·δ̃_leg + lnz·(s_center + σ_s·s̃)`. Turns the product into a
   sum of Gaussians, kills the funnel on the ~28-dim posterior. (F1.)

6. **Enforce σ_anchor < σ_α per class** (in matched units — pin whether σ_anchor is a log-offset or a
   fractional width) so the global `α_pivot` prior stays the identifiability anchor for the per-leg
   amplitude product. LLS σ_anchor=0.25–0.30 vs σ_α=0.15 looks inverted; check it. Run a mini σ_anchor
   sweep like the slope sweep — these numbers are doing real work and are currently un-measured. (F2.)

7. **Extend the coverage verdict** (`_draws_matrix` :497, `_aggregate_legb` :605) to report coverage on
   `s_c` and `δ_leg`, as a diagnostic that the marginalization worked (nuisances not railing) — primary
   gate stays A_p/n_s. (F3.)

8. **Re-profile ESS/sample on one mock WITH the new latents before choosing N** (budget wall, R3); do
   not assume the LF cert's ESS≈0.18 transfers to the ~30-dim posterior. (F4.)

9. **Sequence after the golden guard** (onboarding R2/[CS-R1]) — the α refactor touches the live
   `_legb_model:451-454` and must not land before `tests/golden/` exists.
