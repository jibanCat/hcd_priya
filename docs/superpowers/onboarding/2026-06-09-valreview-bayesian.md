# Validation-plan review — Bayesian / PPL / inference lens (2026-06-09)

**Object:** adjudicate the **convergence-first ladder** the main agent proposed (STEP A:
2–3 fiducial mocks, long multi-chain NUTS, R-hat + per-mock bias; STEP B: SBC coverage
ensemble N≈99–150 only if A is clean AND a calibration cert is needed) against the
original-convention Leg-B SBC/coverage run (N≥99 different mocks, moderate ESS each), in
light of the PI's pushback ("a few fiducials with longer chains, not 300 mocks with few
samples each — cosmology chains are very hard to converge").

**Verdict: ENDORSE_WITH_CHANGES.** The convergence-first ORDER is correct and the PI's
instinct is statistically sound. But STEP A and STEP B certify **different, non-substitutable**
things, and for the (A_p, n_s) error bars the coverage ensemble is **conditionally NECESSARY**
— it is the only object that certifies the WIDTH of the bars across the prior; a few fiducials
cannot. The minimal defensible bar is a TWO-STAGE one, not a one-or-the-other choice.

---

## 1. Convergence vs coverage: what each certifies, and why they are not substitutes

These answer two orthogonal questions, and conflating them is the central error to avoid.

**(a) Convergence (STEP A) certifies the SAMPLER, conditional on a truth.**
A long multi-chain NUTS fit of fiducial mock `i` with split-R̂≈1, high ESS, 0 divergences,
and energy/E-BFMI healthy tells you: *the chain has reached and explored the true stationary
posterior `p(θ | y_i)` for THAT data realization.* It validates that the 25-dim dense-mass
geometry, the slope×amplitude funnel (`α_pivot · g(z)`), and the τ₀-ladder are tractable. It
gives a PRECISE point-bias on that one truth `(truth − mean)/σ` because the σ and mean are now
trustworthy (not low-ESS artifacts like the MF-SMOKE-01 n=1 "−4.5σ" which the log itself flags
as an ESS≈20 artifact). **What it CANNOT do:** it says nothing about whether the reported
posterior WIDTH is correctly calibrated across the prior — because a converged chain reproduces
the *model's* posterior exactly, and if the model is misspecified (the C_emu mis-sized, the MF
floor too tight/wide, the diagonal-in-k covariance unable to whiten a coherent mode — onboarding
R1/R4), the converged posterior is confidently wrong. Convergence certifies you sampled the
posterior you wrote down; it does NOT certify that posterior's credible intervals contain the
truth at the nominal rate.

**(b) Coverage / SBC (STEP B) certifies the ERROR-BAR CALIBRATION across the prior.**
Running N different truths drawn from (or spanning) the prior and asking "does the q-credible
interval contain the truth a fraction ≥ q of the time?" is the ONLY direct test that the
reported `σ(A_p), σ(n_s)` are honest. It is the object that catches model misspecification that
a converged single fit cannot see by construction. For Leg-B specifically the null is NOT
rank-uniform (truth is a held-out SIM, not an emulator draw — onboarding §2.8), so the headline
is **empirical coverage ≥ nominal + |bias| < 0.2σ**, NOT the rank-ECDF (which is diagnostic-only
here, correctly enforced at closure_legb.py `ll_ecdf_diagnostic_only=True`). **What it CANNOT
do:** if the per-mock chains are NOT converged, every coverage number is garbage — a chain
stuck in one mode of a funnel reports a too-narrow interval, faking under-coverage that is a
sampler artifact, not a model defect. So coverage REQUIRES convergence as a precondition.

**The implication for the order:** STEP A is a strict PRE-REQUISITE for STEP B, not an
alternative to it. You cannot read coverage from chains you have not shown converge. This is
exactly why the convergence-first ORDER is right — and exactly why "a few fiducials INSTEAD OF
the ensemble" is the wrong framing. The fiducials are the convergence gate that LICENSES the
ensemble; they do not replace what the ensemble measures.

---

## 2. Is the SBC coverage ensemble NECESSARY to certify the (A_p, n_s) error bars?

**Conditional-YES.** The honest answer turns on what claim the real-fit report needs to make.

- If the deliverable is **"here is the posterior on (A_p, n_s) and here are its credible
  intervals, which we certify are correctly calibrated"** — then YES, a coverage ensemble is
  necessary. No number of converged single-fiducial fits can certify interval calibration: each
  fiducial gives a point bias and confirms the sampler, but the *width* claim is uncertified.
  Three fiducials at three n_s values probe three points; they cannot establish that 95% of
  truths from the prior land in their 95% intervals. This is precisely the gap the onboarding
  R1/R4 risks live in (the coherent low-k mode that the diagonal-in-k C_emu cannot whiten will
  show up as under-coverage in A_p — and ONLY a coverage run reveals it).

- If the deliverable is **"the sampler is healthy and unbiased at representative truths, and the
  error-bar calibration is forecast (Fisher) + argued, not measured"** — then NO, the ensemble
  is not strictly necessary, and the fiducials + Fisher tightening note (the rotation-review
  −6% σ(n_s) tightening) suffice for a CAVEATED claim. But this is a weaker claim than the
  project convention ("coverage ≥ nominal") and the blinding protocol (memory `blinding-strategy`)
  imply, and it leaves the credible interval uncertified at exactly the place the analysis is
  most exposed: the low-k A_p mode and the n_s>0.98 MF-extrapolation edge.

**My ruling:** for a DESI+KS P1D cosmology result that will be published with error bars, the
coverage ensemble is necessary, but it can be RIGHT-SIZED down hard (see §3, §5). The PI's
instinct is correct that 300 short chains is NOT obviously better than a few long ones — but the
resolution is not "drop the ensemble," it is "converge first on fiducials, then run a SMALLER,
properly-converged ensemble." The two are complementary, and the budget supports both.

---

## 3. Per-mock ESS: does ESS≈400 (0.44/sample) bias the coverage/rank statistic?

I tested this directly with a conjugate-Gaussian SBC (correct posterior, independent draws):

| L (≈ESS) | PIT mean | PIT std | empirical 68% cov | empirical 95% cov |
|---|---|---|---|---|
| 25  | 0.502 | 0.299 | **0.631** | **0.888** |
| 100 | 0.497 | 0.293 | 0.663 | 0.932 |
| 400 | 0.503 | 0.289 | **0.677** | **0.947** |

Two clean conclusions:

1. **The RANK statistic is UNBIASED at every L.** PIT mean 0.50, std → 0.289 (uniform) as L
   grows. With *independent* (thinned-to-ESS) draws from a *converged* chain, the SBC rank of
   the truth is Uniform{0..L} regardless of L (Talts+2018 §4). So ESS≈400 does NOT bias the
   ranks — the rank-uniformity / ECDF machinery is fine at moderate ESS. **The two preconditions
   the code already enforces are the right ones:** thin to near-independence (`thin_to_ess`,
   thinning by the WORST-ESS param) and converged chains.

2. **The empirical-COVERAGE estimator is biased LOW at finite L, from quantile-estimator
   noise.** This is the real finding and it bears directly on the design. The central interval
   `np.quantile(draws, [(1-q)/2, (1+q)/2])` (closure_diagnostics.central_interval) from L draws
   is systematically too NARROW for small L, so the measured coverage undershoots nominal even
   for a perfectly-calibrated posterior: **95% reads as 88.8% at L=25, 93.2% at L=100, and only
   94.7% at L=400.** At the design ESS≈400 the 95% coverage is biased low by ≈0.5pp; at 68% it
   is essentially exact. This is small but NON-ZERO at L=400, and it is a one-sided bias that
   pushes toward a FALSE under-coverage verdict. Three consequences:

   - **ESS≈400 is ADEQUATE for the 68% coverage gate and the rank ECDF, but marginal for a
     strict 95% gate.** The MF-COVERAGE-01 plan targets ESS≥400 on n_s (902 samples). I endorse
     that as the FLOOR, and recommend reading the verdict primarily off the **68% interval**
     (exact at L=400) and the **per-param bias** (unaffected by quantile noise — it uses mean/std,
     not tail quantiles), treating the 95% coverage as a softer, finite-L-corrected check.
   - **The cleanest fix is to compare measured coverage to the finite-L EXPECTATION under perfect
     calibration, not to the nominal q.** I.e. simulate the L-draw quantile-estimator coverage of
     a correctly-calibrated Gaussian (the table above) and gate on `measured ≥ E[coverage | L,
     calibrated] − Wilson`. This removes the spurious low bias entirely and is ~10 lines. This is
     the standard correction and I recommend it be added to `_aggregate_legb` before reading any
     95% verdict. **(CHANGE, IMPORTANT.)**
   - Equivalently, push ESS to ~800–1000 where even the 95% estimator bias is <0.3pp. Cheaper to
     correct the estimator than to double the chain.

**Bottom line on (c):** ESS≈400 does NOT bias the ranks and is adequate for the coverage
ENSEMBLE provided (i) chains are converged (STEP A precondition) and (ii) the 95% coverage gate
is finite-L-corrected (or the 68% gate + bias is the primary readout). The handoff's earlier
"ESS≈0.18/sample → thinned L≈28–42 < L_FLOOR" regime would have been genuinely under-powered
(L=25 row: 95%→88.8%); the re-profile to ESS≈0.44/sample (L≈400) lifts it out of that danger
zone, which is the right call.

---

## 4. How many mocks N for a meaningful coverage estimate?

The coverage fraction is k/N ~ Binomial; the Wilson half-width and the smallest detectable
under-coverage gap (one-sided ~2σ) are:

| N | Wilson ± on 68% cov | Wilson ± on 95% cov | ~2σ-detectable under-cov gap @95% |
|---|---|---|---|
| 8   | 0.272 | 0.192 | — |
| 20  | 0.189 | 0.114 | 0.060 |
| 50  | 0.125 | 0.066 | 0.060 |
| 99  | 0.090 | 0.045 | 0.043 |
| 150 | 0.074 | 0.036 | 0.035 |
| 300 | 0.053 | 0.025 | 0.025 |
| 600 | 0.037 | 0.018 | 0.017 |

And the s.e. on the **mean per-param bias-z** (the ±0.2σ gate), assuming per-mock bias-z spread
≈1:

| N | s.e. of mean bias-z |
|---|---|
| 3   | 0.577 |
| 8   | 0.354 |
| 20  | 0.224 |
| 99  | 0.101 |
| 300 | 0.058 |

**Reading these tables:**

- **The L_FLOOR=99 is the right ENSEMBLE floor for two independent reasons.** (i) The ECDF
  simultaneous-band gate is only correctly sized at L≥99 thinned draws (per-chain ESS, the
  documented type-I 0.20@L49 → 0.05@L99). (ii) At N=99 mocks the Wilson half-width on 95%
  coverage is ±0.045 and the bias s.e. is 0.101 — enough to resolve the 0.2σ bias gate at ~2σ
  and to detect a 0.043 under-coverage shortfall. Note L_FLOOR (per-chain ESS) and N (#mocks)
  are DIFFERENT axes; both ≈99–100 by coincidence of the design choices. The L+1=100 highly-
  composite requirement (closure_diagnostics.rank_histogram_bins, L=100→101 prime → useless) is
  correctly handled by L=99→L+1=100.

- **N≈99–150 is the sweet spot.** At N=99 (≈600 CPU-h) you can certify |bias|<0.2σ at ~2σ and
  coverage shortfalls ≳0.043. N=150 (≈900 CPU-h) tightens the bias s.e. to 0.074 and the
  coverage detection to 0.035 — a worthwhile, affordable bump. N=300 (≈1.6–2.0k) buys 0.058
  bias s.e. and 0.025 detection but the marginal value drops (you are now resolving sub-0.1σ
  biases that are below the closure's own systematic floor and below the 6-HR MF noise ~0.05σ).
  **N=600 (3.65–4.0k, AT the ceiling) is NOT worth it** — it resolves 0.017 coverage gaps that
  are smaller than the C_emu-floor's own 1.35× uncertainty and would consume the entire budget,
  leaving nothing for STEP A, retries, or the real fit. The PI is RIGHT to resist N=600.

- **A few fiducials (N=2–3) cannot estimate coverage at all** — Wilson ±0.30 on 95%, bias s.e.
  0.58. They are a convergence + point-bias instrument ONLY. This is the quantitative statement
  of §1: 3 mocks ≠ a coverage cert.

---

## 5. The convergence-first ORDER and the right convergence diagnostics

**The order is CORRECT and I endorse it strongly.** Running 2–3 long multi-chain fits BEFORE
fanning out an ensemble is exactly how you avoid burning 600–4000 CPU-h on chains that turn out
not to converge — which, given the PI's (correct) warning that cosmology chains are hard to
converge AND the onboarding R3 funnel risk from the `α_pivot · g(z)` product, is a live hazard.
STEP A is the cheap insurance (≈70 CPU-h) that de-risks the expensive STEP B. The codebase
already has the machinery: `scripts/diag_legb_sampling.py` runs 4-chain R-hat + cross-chain ESS
via `numpyro.diagnostics.gelman_rubin` on the exact `_legb_model` — STEP A is largely wiring
that script up at production mtd=10 on 2–3 chosen fiducials.

**The minimal convergence diagnostic battery for STEP A (all required, all standard):**

1. **Split-R̂ < 1.01 per param** (not the looser 1.1) — multi-chain (≥4 chains, different inits;
   the code currently runs `init_to_median` which gives identical inits → MUST use dispersed
   inits for R-hat to be meaningful, e.g. `init_to_sample` or jittered medians). **R̂ on a single
   init_to_median start is NOT a valid convergence diagnostic.** (CHANGE, CRITICAL — see concerns.)
2. **Bulk-ESS AND tail-ESS ≥ 400 per param** (tail-ESS specifically, because the 95% credible
   interval lives in the tails and the τ₀–n_s degeneracy + the softplus α_DLA can have heavy/
   skewed tails the bulk-ESS hides). The current `thin_to_ess` uses a single Geyer ESS; add the
   arviz bulk/tail split for STEP A.
3. **0 divergences after the target_accept escalation** (0.9→0.95→0.99, already wired). Divergences
   are the funnel canary — watch them specifically on the high-z α-slope where `α_pivot·g(z)` is
   most product-like. If they persist, reparametrize (non-centered slope) BEFORE the ensemble.
4. **E-BFMI > 0.3** (energy diagnostic) — cheap, catches the dense-mass-on-25-dim failure where
   the momentum resampling can't traverse the τ₀-ladder. Not currently computed; add it (it is in
   `mcmc.get_extra_fields()` energy).
5. **The slope×amplitude funnel specifically:** plot `log α_pivot` vs the per-z `α_c(z)` draws
   and inspect for the funnel neck; if present, the non-centered parametrization of the
   slope×amplitude product is the fix (onboarding R3). This is the single most likely pathology
   given the model structure and the PI's convergence warning — STEP A's PRIMARY job.

**Fiducial choice:** the proposed low/mid/high n_s is right but INSUFFICIENT on its own — add a
fiducial at the **n_s>0.98 MF-extrapolation edge** (where the sigma_edge term and the rotation-
review's guarded-not-nominal flag bite) and one at the **low-k A_p-sensitive corner** (high
C_emu/C_data ratio, the 0.52 spike). The hard geometry lives at the edges, not the center;
convergence at the median does not certify convergence at the corners the real fit may visit.

---

## 6. The minimal DEFENSIBLE validation bar (Bayesian lens)

A two-stage bar, both stages required:

**STAGE A — convergence + point-bias (gate to even launch the ensemble), ~70–120 CPU-h:**
- 4 fiducials (low/mid/high n_s + one at the n_s>0.98 edge), each 4 dispersed-init chains at
  production mtd=10, ESS≥800/param.
- PASS: split-R̂<1.01 all params, bulk+tail-ESS≥400, 0 divergences post-escalation, E-BFMI>0.3,
  no visible funnel neck, AND per-fiducial |bias| < 0.2σ on (A_p, n_s).
- If the funnel bites → reparametrize → re-run STAGE A. Do NOT proceed to the ensemble on a
  pathological geometry.

**STAGE B — coverage ensemble (the calibration cert), N=99–150, ≈600–900 CPU-h:**
- N=99 minimum (L_FLOOR + bias-resolution), N=150 preferred (tightens bias s.e. to 0.074).
- Per mock: ESS≥400 (the re-profiled 0.44/sample → ~902 samples), thinned to near-independence,
  single chain acceptable HERE (R-hat was certified in STAGE A; coverage uses one chain/mock by
  design — the cross-mock ensemble IS the variance source).
- PASS: empirical **68% coverage ≥ nominal − Wilson** on (A_p, n_s, τ₀-block) AND |mean bias-z|
  < 0.2σ on (A_p, n_s); **95% coverage gated against the finite-L-corrected expectation** (§3),
  not raw nominal; rank-ECDF reported DIAGNOSTIC-ONLY (never gates Leg-B).
- Report τ₀ coverage per-z with its n (onboarding R5), and split the whitening var by (k-band,z)
  (R4) so a low-k under-coverage cannot hide in a pooled scalar.

**Total ≈ 700–1000 CPU-h** — comfortably inside the 4000 budget, leaving headroom for retries,
the Leg-A SBC pre-req (R6), and the real fit. This is strictly more defensible than EITHER the
N=600 ensemble (over-spends, under-converges per mock) OR fiducials-only (no calibration cert),
and it directly implements the PI's convergence-first instinct without discarding the coverage
guarantee the published error bars need.

---

## 7. Subfield standard — what Lyα-forest P1D cosmology analyses actually do to validate

The Lyα-P1D field is a useful cross-check because its validation practice is well-documented and
the present analysis is closely modeled on it. What the field actually runs:

- **eBOSS DR14 P1D (Chabanier+2019, 1812.03554; Palanque-Delabrouille+2020, 1911.09073).** The
  cosmology is inferred with a Gaussian likelihood + a simulation-calibrated covariance; the
  emulator/sim grid is validated by **mock data fits at the fiducial cosmology + a small set of
  alternative cosmologies** and by checking the MCMC chain convergence (Gelman-Rubin R−1 < 0.01
  / 0.03 thresholds are standard in their CosmoMC/Montepython runs). They do NOT run a full SBC
  coverage ensemble of ~100 different truths; the calibration is argued via the covariance
  construction + a handful of mock recoveries + R-hat-converged chains. This is the
  "STAGE A + Fisher/argued calibration" tier.

- **DESI Lyα emulator papers — LaCE / the Gaussian-process & NN emulators
  (Pedersen+2021 2011.15127; Pedersen+2023 2209.09895; Cabayol-Garcia+2023 2305.19064).** The
  emulator is validated by **leave-one-out (LOO) over the sim suite** (the analog of this
  project's 8-fold LOSO) — recovering each held-out sim's cosmology and checking the recovered
  P1D residual + parameter bias is within the emulator error. This is precisely the
  closure/point-bias tier (the project's "+0.035σ all-folds"), NOT a coverage ensemble. The LOO
  is over the sims, so it spans the prior, but they report it as a per-sim bias distribution, not
  a calibrated coverage fraction.

- **DESI DR1 P1D (Karaçaylı+2025, the data this analysis uses).** The pipeline validation is
  dominated by mock-spectra end-to-end recoveries (synthetic skies → the full estimator →
  recovered P1D vs input) and covariance validation (regularized + diagonal-inflation), plus
  chain convergence. The cosmology-inference coverage is again argued through the mock recoveries
  + the published covariance, not a 100-mock SBC.

- **Where SBC IS used:** SBC / rank-based calibration is the rising standard in the broader
  cosmology-inference and SBI literature (Talts+2018 1804.06788; Säilynoja+2022 2103.10522 —
  both already implemented in `closure_diagnostics`; Lemos+2023 "calibration of SBI" for the
  TARP/coverage variant; and the DES/LSST SBI pipelines now report coverage). The Lyα-P1D
  subfield has NOT historically run full SBC, but the methodological direction is toward it, and
  a coverage ensemble would put THIS analysis AHEAD of the eBOSS/DESI-P1D standard, not behind it.

**The honest synthesis for the PI:** the Lyα-P1D field's de-facto standard is **STAGE A
(R-hat-converged chains + a handful of mock/LOO recoveries with per-sim bias) + a
covariance-construction argument for the error-bar calibration** — i.e. essentially the
convergence-first ladder WITHOUT a formal coverage ensemble. So the proposed STAGE A alone would
already MEET the published-field bar. The coverage ensemble (STAGE B) is a STRENGTHENING beyond
field standard, justified here specifically because (i) the diagonal-in-k C_emu cannot whiten the
known coherent low-k mode (R1/R4) — a documented misspecification that ONLY coverage can bound,
and (ii) the analysis is blinded and will publish calibrated intervals. Given the modest cost
(N=99–150 ≈ 600–900 CPU-h), running the right-sized STAGE B is the prudent, defensible call —
but the PI is correct that it should be SMALL and convergence-gated, not N=600.

---

## 8. Concerns (severity-ranked)

See the structured object. The load-bearing ones: the dispersed-init requirement for R-hat
(CRITICAL — a single init_to_median makes R-hat meaningless), the finite-L 95%-coverage
estimator bias (IMPORTANT — gate against the finite-L expectation, not raw nominal), and the
fiducial set must include the edge corners (IMPORTANT — convergence at the median does not
certify the n_s>0.98 / low-k-A_p corners the real fit visits).
