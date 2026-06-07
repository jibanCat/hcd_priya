# HCD incidence z-slope nuisance — prior spec

**Status:** reviewed by the 4-lens checkpoint (2026-06-06) → **GO-WITH-CHANGES**. Structure LOCKED;
the slope **width's bias justification is PROVISIONAL pending one forward-only rerun** (see §7).
**Branch:** `phase2c-likelihood`. **Decision owner:** M.-F. Ho (PI).
**Feeds:** the closing-step implementation (golden guard → forward rerun → §0c → factorized α(z) → NUTS closure).

> **READ §7 FIRST** — the checkpoint resolved all four open questions and found two real errors in the
> originally-"locked" numbers (§2/§4 below are the pre-review draft, kept for the audit trail; §7 supersedes).

---

## 0. Context (why this exists)

The Leg-B closure exposed a coherent low-k over-prediction that the diagonal-in-k C_emu
cannot whiten → it leaks onto A_p/n_s (buggy pilot: A_p ≈10σ low). Root cause: the forward
applied a z-shape for the HCD incidence α(z) that was too shallow (the literature *ratio*
slope 0.15–0.95) vs the sim's *absolute* incidence slope ~2.6. The 4-lens onboarding review
(`docs/superpowers/onboarding/2026-06-06-onboarding-*.md`) concluded the documented "thread
the exact sim w_c(z)" fix is **necessary-but-not-sufficient** (it certifies the sampler, not
the production mean model). Cross-lens recommendation: **(B+slope)** — factorize the z-shape
into a KNOWN fixed part × a SAMPLED literature-correction slope, and certify coverage with the
slope marginalized.

## 1. The model (factorized α(z))

Per HCD class c ∈ {LLS, subDLA, DLA}, per leg, replace the current
`α_c(z) = α_pivot,c · ((1+z)/(1+z_p))^{HCD_LIT_OVER_SIM_SLOPE_c}` (`closure_legb.py:453`, a
fixed lit-ratio slope) with:

```
α_c(z) = α_pivot,c · exp(δ_{c,leg}) · g_fixed,c(z) · ((1+z)/(1+z_p))^{s_c}
         └ amplitude ┘ └ per-leg ┘ └ KNOWN shape ┘ └ sampled lit-correction slope ┘
```

- `α_pivot,c` — incidence amplitude at z_p=3 (existing `alpha_lls/subdla`, softplus DLA → see §0c).
- `δ_{c,leg}` — **per-leg anchor float** (log-offset on the amplitude), §3.
- `g_fixed,c(z)` — the KNOWN steep abundance shape (slope ~2.6). **What this IS in production
  vs closure is the OPEN question, §4.**
- `s_c` — the SAMPLED literature-correction ratio-slope nuisance, §2.

## 2. LOCKED — slope-prior width (the tradeoff is measured)

Forward-only Fisher sweep `scripts/diag_legb_slope_prior_tradeoff.py` →
`figures/analysis/05_likelihood/legb_slope_prior_tradeoff.{png,txt}`:

- **Marginalizing s_c is nearly FREE:** slope-pinned → slope-free σ(A_p) = 0.390 → 0.405
  (**×1.04**); at the literature width ×1.003. σ(n_s) ≈ flat. The fixed steep g(z) carries the
  z-leverage; the small correction slope barely competes with A_p.
- **Bias** (truth offset by the sim-vs-lit gap 0.76 LLS): at the literature width **+0.167σ**
  (under the 0.2σ gate), **dominated by LLS** (+0.106σ); subDLA negligible (+0.004σ, gap 0.05).
- ⇒ **Width = the literature dN/dX-slope 1σ, with z-edge inflation.** It costs ~nothing in
  σ(A_p) and keeps bias < gate; there is free headroom to go wider if desired.

**Locked numbers** (Lyα consult, WLS re-fit of the repo's dN/dX-vs-lit data with the quoted
error bars — the code constants had NO uncertainty):

| class  | base width σ_s^lit | binding? |
|--------|--------------------|----------|
| LLS    | **0.52**           | **YES** (tightest amp prior 0.15, biggest gap) — drives σ(A_p) |
| subDLA | **0.53**           | no (gap ~0, broad amp prior) — essentially data-set |
| DLA    | 0.33               | MOOT (§0c masks α_DLA→0) |

**z-edge inflation** (z=2.5–3.5 trusted; purity-limited below, completeness-limited above →
quoted errors overconfident outside): per-z `edge(z) = 1 + β·(clip(2.5−z,0) + clip(z−3.5,0))`,
β=1 (reproduces the existing DLA `dla_inflate` on the high side), applied as the leg-z-mean
scalar to σ_s (and, for consistency, to the amplitude σ — extending the current DLA-only
inflation to LLS/subDLA). Mean factor ≈1.25 over the DESI+KS z-range → σ_s^LLS ≈ 0.65.

## 3. LOCKED — per-leg anchor float (PI: KS may be biased-high LLS/subDLA)

A hard literature center on a leg whose true incidence differs re-injects an A_p bias. So the
amplitude center floats per leg:
```
α_pivot,c,leg = α_pivot,c · exp(δ_{c,leg}),   δ_{c,leg} ~ Normal(0, σ_anchor,c,leg)
```
- KS (LLS, subDLA): σ_anchor ≈ **0.25–0.30** (absorbs a high-resolution-selection incidence excess).
- DESI (LLS, subDLA): σ_anchor ≈ **0.10–0.15** (large-area, closer to published dN/dX).
- DLA does NOT float (§0c). σ_anchor numbers are defensible expectations, NOT literature-measured.

This is NEW plumbing: `DataLeg` (`data_likelihood.py:70-99`) has no per-leg incidence field today.

## 4. OPEN for this checkpoint — `g_fixed(z)` and the slope-prior CENTER

The subtlety the lenses must adjudicate (it sets whether the closure is production-faithful):

- **Closure-only framing (what the sweep used):** `g_fixed,c(z) = w_c,sim(z)/w_c,sim(z_p)`
  (the held-out sim's own incidence). Then `s_c=0` is the +0.03σ EXACT operating point, and the
  slope marginalization is a robustness buffer. BUT production has no sim `w_c(z)` → this is not
  the production config, so coverage may not transfer (the "certifies a shape production never
  runs" trap the onboarding flagged).
- **Production-faithful framing:** `g_fixed,c(z)` = a LITERATURE-derived incidence shape
  (observed dN/dX → w_c via `dndx_wc.w_c_from_mu`, the same map used for the amplitude prior).
  Then the closure forward uses EXACTLY what the real fit uses; the sim-truth differs from
  `g_fixed` by the sim-vs-lit shape gap, and the slope nuisance + per-leg anchor must absorb it.
  The slope prior CENTER then sits at the lit-vs-(forward-g_fixed) differential (likely ~0 if
  g_fixed already encodes the lit shape), NOT at the sim's absolute slope.

**Questions for the lenses:**
1. Should the closure forward use the production `g_fixed` (lit-derived), with the sim-truth as
   the thing coverage must survive — i.e. the honest production-faithful closure (recommended)?
2. Where does `s_c` center: WLS lit value (0.76 LLS / 0.05 subDLA) or the code's (0.95/0.15)?
   And is the center a fixed constant or itself given a (tighter) prior?
3. The nonlinear cross-check showed `J_s·gap` vs exact `ΔP` differ **13% (DESI) / 15% (KS)** —
   the gap is a large slope step and `(1+z)^s` is nonlinear over it. Is the Fisher bias (0.16σ)
   trustworthy to set the prior, or do we need the exact (non-linearized) construction bias at
   the literature width before committing? (Cheap, forward-only.)
4. Does the per-leg anchor float belong on the amplitude (as specced) or also on the slope?
   (The bias is LLS-amplitude-driven; the sweep says the slope barely matters — so amplitude
   float is likely the right and sufficient lever. Confirm.)

## 5. Out of scope here (handled by the sequenced plan, not this spec)

- The **golden-regression guard** (must land FIRST, before any α refactor — `tests/golden/` absent).
- §0c **DLA mask** (α_DLA Gaussian-at-0 + DLA-masked mock truth) — coupled, but its own step.
- The **low-rank correlated-in-k C_emu** (belt-and-braces for any residual coherent mode).
- MF wiring (cert is LF-only until then).

## 6. Verified-vs-assumed for this spec

- VERIFIED (this session, forward-only): variance ×1.04; bias 0.167σ at lit width, LLS-dominated;
  σ(A_p) curve + per-class decomposition (`legb_slope_prior_tradeoff.{png,txt}`); sim w_c slope
  ~2.6, lit ratio-slope widths 0.52/0.53/0.33 (WLS).
- ASSUMED: the Fisher linearization (13–15% off on the full gap — Q3); the per-leg σ_anchor
  magnitudes (defensible, not measured — Q4); that the LF-only cert transfers post-MF (it does not — §5).

---

## 7. CHECKPOINT RESOLUTION (4-lens review, 2026-06-06 — SUPERSEDES §2/§4)

Verdict: **GO-WITH-CHANGES** (4/4 lenses). Report:
`docs/superpowers/onboarding/2026-06-06-checkpoint-{bayesian,cs,cosmology,lya,meta}.md`.

**Two errors found in the originally-"locked" numbers (verified against code/output by the main agent):**
- **E-A:** the sweep `diag_legb_slope_prior_tradeoff.py:90` used `g_fixed = wc_perz[sel_sim]/wc_pivot`
  = the **SIM's own shape** — the closure-only framing this spec argued AGAINST. So the +0.167σ bias
  describes a config production never runs. (The **×1.04 variance is framing-robust and STANDS.**)
- **E-B:** the **n_s bias is −0.22σ at the literature width** (−0.235σ at narrowest) — OVER the 0.2σ
  gate, and §2 reported only A_p. The locked "bias < gate" claim is FALSE for n_s.

**Resolved verdicts (no PI call needed — unanimous technical resolutions):**
- **Q1 → YES, production-faithful `g_fixed` is REQUIRED.** Build it once host-side as
  `g_fixed,c(z) = w_c_from_mu(dN/dX_lit · X̄(z))` normalized to z=3, a frozen `(n_z_leg,3)` constant on
  `LegBCtx` per leg. **Never from dN/dX alone** — the incidence-only shape IS the original bug in
  disguise (Lyα verified live: lit LLS w_c slope +3.20 = +2.02 path-geometry + 1.18 incidence; vs sim
  +2.60 → bounded shape gap −0.60 ≈ the lit dN/dX ratio-slope). X̄(z) from sim `snap_total_path_dX/n_skewers`.
- **Q2 → `s_c ~ Normal(0, σ_s)`, FIXED center at 0, NO hyper-prior.** Under production `g_fixed` the lit
  z-shape is already in `g_fixed`, so `s_c` is the residual correction → center 0 (centering at WLS 0.76
  would double-count). Width = WLS lit (0.52/0.53/0.33) + z-edge inflation. (If the sim-`g_fixed` framing
  were ever kept, use WLS 0.76/0.05, not the code's unweighted 0.95/0.15.) Report the in-window
  (z=2.5–3.5) WLS center as a check.
- **Q3 → compute the EXACT (non-linearized) bias FIRST.** `J_s·gap` is 13–15% off (the sweep's own
  cross-check) and `(1+z)^s` is convex over the gap so it UNDER-estimates; 0.167σ is already 83% of the
  gate. Extend the sweep (~15 lines: `P_s,P_0,J,Cinv,Cpost` in scope) to propagate exact `ΔP` for **A_p
  AND n_s** on the production `g_fixed`, §0c-masked, at ≥1 interior-n_s fiducial.
- **Q4 → AMPLITUDE-only per-leg float, slope GLOBAL.** A per-leg slope is a degenerate second
  amplitude-like direction (funnel/banana, erodes n_s) modeling a mechanism that doesn't exist (KS's
  selection shifts incidence amplitude, not z-evolution). Keep the LLS τ≥1-vs-τ≥2 definitional offset in
  the amplitude CENTER (`HCD_LIT_OVER_SIM[0]`), not `δ_leg`.

**Additional locked-in changes:**
- **Non-centered log-space reparam** (Bayesian, HIGH): sample `δ_leg, s` as standard normals,
  reconstruct `log α_c(z) = log α_pivot,c + σ_anchor·δ̃ + lnz·σ_s·s̃` → the amplitude×slope product
  becomes a SUM of Gaussians, killing the funnel on the ~28–30-dim posterior.
- **Implementation is a real contract change, NOT a 1-line multiply** (CS E-C): `_legb_model` builds one
  global-z α; the per-leg `δ_leg` forces moving α(z) assembly into `_data_loglik_legcore`'s per-leg loop
  (~40–60 lines) + truth-packing updates. ⇒ the **golden guard is a hard gate** (`tests/golden/` absent).
- **σ_anchor units:** specify as a log-offset and enforce `σ_anchor < σ_α` per class (LLS σ_α≈0.15 vs
  σ_anchor 0.25–0.30 looks inverted — resolve units).
- **s/δ coverage is NOT a valid null** (sim has no lit-correction slope) → report A_p/n_s coverage as the
  gate; s/δ railing is a diagnostic only.
- **Width is PROVISIONAL on the correlated-in-k C_emu** (§5): the slope nuisance + diagonal C_emu leaves a
  coherent low-k residual on n_s; the correlated C_emu is the real whitener. Co-design, re-measure the n_s
  bias WITH it before the final cert.

**Endorsed sequence:** (1) golden guard FIRST → (2) forward-only rerun [Q1+Q3+E-A+E-B+§0c, the gating
measurement; runs parallel to (1)] → (3) §0c DLA mask → (4) per-leg α refactor + non-centered reparam →
(5) edge-inflation prior-builder + σ_anchor mini-sweep + in-window center check → (6) re-profile ESS on
the bigger posterior → (7) NUTS Leg-B closure over MULTIPLE held-out sims, gate = coverage≥nominal AND
|bias|<0.2σ on BOTH A_p and n_s.

## 8. RERUN RESULT (2026-06-06 — the gating measurement, checkpoint-corrected)

`scripts/diag_legb_slope_prior_rerun.py` → `figures/analysis/05_likelihood/legb_slope_prior_rerun.{png,txt}`.
Production lit-derived `g_fixed` = `w_c_from_mu(dN/dX_lit·X̄(z))` (NEVER dN/dX alone); DLA masked on BOTH
sides; per-leg amplitude anchor (KS σ=0.27, DESI σ=0.12); EXACT ΔP bias for A_p AND n_s; 8 fold-0
held-out sims + 3 interior-n_s spot-checks. Decomposed into FULL (shape+amp+emulator) vs EMU-only
baseline (forward at the sim's EXACT per-z w_c → pure emulator error).

**THE GATING ANSWER — the slope/anchor's OWN contribution (FULL − EMU; normalization + shared
emulator error cancel, so this is the trustworthy quantity):**
- **A_p:** mean −0.079σ, RMS 0.086σ, **max 0.143σ** — under the 0.2σ gate on all 8 sims.
- **n_s:** mean −0.027σ, RMS 0.112σ, **max 0.168σ** — under gate (interior-n_s spot-checks even smaller,
  <0.06σ; the larger fold-0 values are partly the known n_s box-edge effect).
- **Width-INSENSITIVE** (lit / edge / 2× lit give identical FULL bias) → confirms the slope is NOT the
  driver; the per-leg amplitude anchor does the absorbing. ⇒ **the locked width (lit + z-edge inflation)
  is SAFE.** The earlier −0.22σ n_s "gate fail" was an artifact of the broken framing (lit-DLA in the
  forward but masked in the truth, no anchor); corrected, it is −0.03σ.

**THE SEPARATE, LARGER ISSUE — the EMU-only baseline (pure emulator error, NOT the slope's job):**
- A_p mean **+0.53σ**, per-sim scatter **0.88σ**, max 1.9σ; n_s mean −0.24σ (edge-contaminated), scatter
  0.97σ. The ~1σ per-sim SCATTER is coverage-consistent (random per-mock offset within a correctly-sized
  C_total); the MEAN +0.53σ A_p (SE 0.31 over 8 sims → ~1.7σ) is a *possible* coherent low-k emulator
  residual — exactly the onboarding-R1 mode the **correlated-in-k C_emu (§0b-2, NOT yet built)** is meant
  to whiten. It is out of scope for the slope prior and is neither helped nor hurt by it.
- ✅ **RECONCILED (2026-06-06, `scripts/diag_emu_lowk_investigation.py` + doc
  `2026-06-06-slope-prior-and-emu-lowk-investigation.md`):** the +0.6σ-vs-0.067σ gap is
  **z-ACCUMULATION**. At a single z=3 slice (the Phase-2 ruler) the A_p bias is **+0.033σ** (∼0.067σ);
  across the full ~13-z range it sums to +0.62σ (×18.6), because the per-z emulator error is small but
  **z-COHERENT**. The covariance ruler (correlated vs diagonal) is only 1.26× — NOT the explanation.
  The bias is **sourced from high-k** (k>0.02: +0.72σ; low-k −0.06σ) and routed to A_p through the
  **GLOBAL SVD output basis** (low/high-k residuals correlate r=−0.61); it is **training-limited, not
  rank-limited** (emu error 6–18× the rank-24 truth floor). MF is structurally not the cause (LF-vs-LF
  closure) but acts on the same high-k channel → re-measure after MF. ⇒ this is the R1 z-coherent mode
  for the **correlated-in-k/z C_emu (§0b-2, not built)**, NOT the slope prior's job. The slope-gating
  conclusion is independent of it (rests on the FULL−EMU differential).

## 8b. ALL-FOLDS RE-MEASUREMENT (2026-06-06 — reverses the parameter, not the existence)

`scripts/diag_emu_bias_allfolds.py` → `figures/analysis/04_emulator/emu_bias_allfolds.{png,txt}`.
EMU bias over ALL 8 folds, **each fold's OWN held-out emulator** (`final_fold{0..7}`), KS capped
k<0.06, 60 honestly-held-out sims spanning n_s_unit 0.013–0.958, bootstrap CI:
- **A_p: mean +0.03σ, 95% CI [−0.22, +0.29], t=0.23, 31+/29−.** ⇒ the +0.62σ from §8 was a
  **low-n_s box-edge fluctuation** (fold-0 sims are all n_s_unit<0.12); on the honest set the
  **A_p EMU bias is consistent with ZERO.** (Large per-mock scatter, max 3.4σ, is coverage-consistent.)
- **n_s: mean −0.65σ, 95% CI [−0.85, −0.43], t=−5.84, 10+/50−.** ⇒ the REAL coherent emulator bias is
  on **n_s, NEGATIVE** (emulator under-predicts the tilt), highly significant, NOT a box-edge artifact
  (corr with edge-proximity −0.04), survives dropping the largest outlier (−0.61σ).
- ⇒ **The investigation was right in KIND (a real coherent emulator-sourced bias a diagonal C_emu
  can't whiten) but WRONG in PARAMETER** — it is n_s at −0.65σ, not A_p at +0.6σ. The fold-0-only
  sample (all low-n_s) mis-attributed it to A_p. Cleaner root-cause target: why the emulator coherently
  UNDER-predicts the n_s tilt — a k-shape/SVD-basis question, consistent with the high-k coupling.
- The slope-prior settlement is UNAFFECTED (rests on FULL−EMU, which cancels this shared term).

## 9. PI calls that remain (3):
1. **The gate-vs-widen-vs-C_emu fork, AFTER the step-2 rerun.** If the exact production-framing bias
   (A_p or n_s) lands in 0.18–0.25σ: (a) widen the slope prior (cheap ×1.04, but width-INSENSITIVE so
   limited leverage), or (b) block the cert on building the correlated-in-k C_emu (the real whitener,
   more work). **Load-bearing; needs the number first.**
2. **Accept the conditional coverage claim:** certified coverage is "conditional on sim-vs-lit gap ≈
   real-vs-lit gap" — the best available test, but a stated assumption, not a proof.
3. **The σ_anchor magnitudes** (KS 0.25–0.30, DESI 0.10–0.15) — un-measured, doing real work; informed by
   the mini-sweep, but the physical prior on "how high can KS selection bias incidence" is the PI's.
