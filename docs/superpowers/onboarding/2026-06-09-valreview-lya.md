# Validation-plan review (Lyα / IGM / P1D-cosmology lens) — convergence-first vs SBC coverage ensemble

**Date:** 2026-06-09. **Reviewer lens:** Lyman-alpha forest / IGM physics / P1D-cosmology subfield standard.
**Object:** the proposed validation of the Phase-C MF Leg-B likelihood before the real DESI+KODIAQ-SQUAD fit
— the PI's pushback ("a few fiducial mocks with LONG chains beats 300 mocks with few samples each") and the
main agent's revised **convergence-first ladder** (STEP A = 2–3 fiducial mocks × 4 long multi-chain NUTS →
R-hat + per-mock bias; STEP B = SBC coverage ensemble only if A is clean AND a cert is needed).

**VERDICT: ENDORSE_WITH_CHANGES.** The convergence-first ladder is the right shape AND it is exactly what
the subfield does — including the paper this analysis reproduces. The SBC coverage ensemble is **NOT
necessary** to certify the (A_p, n_s) error bars and is, as scoped (on `final_fold0`), partly an illusion:
it cannot test coverage where the real fit lands. My changes: (1) STEP A's fiducials MUST include a
**high-n_s** mock near the real-data tilt (n_s≈0.98–1.0), not low/mid drawn from fold 0; (2) the binding
real-fit certificate is **mock recovery across different sims/resolutions + R-hat + the C_emu-floor coverage
note**, matching the DESI-DR1 standard — and STEP A's "long chains" need a real convergence criterion
(R−1<0.01, the PRIYA-cosmology bar), not just ESS; (3) reframe STEP B honestly: a *Leg-B empirical-coverage*
ensemble, not SBC, and only ever as a *calibration cross-check*, never the gate.

---

## 1. THE SUBFIELD STANDARD — what Lyα P1D cosmology analyses ACTUALLY do (my primary charge)

I read the validation sections of the relevant papers directly (PDF text-extracted with PyMuPDF where the
WebFetch summary was too lossy). The picture is strikingly consistent: **convergence + a small number of
fiducial/seed-varied mock recoveries + leave-one-out emulator accuracy + (modern) blinding. NO published
Lyα-P1D cosmology analysis runs an SBC rank-uniformity ensemble or a large empirical-coverage ensemble to
certify its cosmology error bars.**

### 1.1 The paper THIS analysis reproduces — Fernandez, Bird & Ho 2024 (JCAP 07 029 / 2309.03943)

This is the decisive datum: the validation the convergence-first ladder must at minimum match is the one in
the very paper being reproduced. I extracted §3.5 "Inference Using Simulation Data" + §3.4 verbatim. They ran:

- **A single MAP recovery on one HF sim.** *"We first used the flux power spectrum from one of the three
  high fidelity simulations, and confirmed that the maximum likelihood was indeed at the input parameter
  values for all parameters. All input parameters were recovered to better than one sigma."*
- **A handful of MCMC chains on ONE seed-varied mock** (their Fig. 3: `Seed FPS`, `Seed FPS+GPERR`,
  `Seed FPS+T0`). *"We next ran chains using data from a low-fidelity simulation with a different random seed
  from the main PRIYA suite … we consistently recover the true values of the cosmological parameters nP and
  AP to better than 1 −σ … we deliberately constructed our test data, with a different structure seed, to be
  different from the training data."* The seed-varied mock is precisely the "honest held-out truth" idea —
  but it is **ONE mock**, run with the **three error-model variants**, not an ensemble.
- **Leave-one-out emulator cross-validation** (deferred to the sims paper, ref [47]): *"on average 0.2%
  accuracy for the low-fidelity simulations and 1% for the high fidelity simulations."* This is the LOSO
  this repo already reproduces (8-fold, 60 sims).
- **MCMC convergence by Gelman-Rubin:** *"Convergence is determined using the Gelman-Rubin statistic, R …
  The chains presented here were run until a convergence of R −1 < 0.01."* Sampler = Cobaya/Metropolis.
- **Zero** occurrences of "coverage", "simulation-based calibration", or "blind" in the entire 41-page
  paper (I counted: mock×3, recover×4, leave×17, Gelman×1, Rubin×1, coverage×0, SBC×0, blind×0).

**So the analysis being reproduced validated with: R−1<0.01 convergence + one MAP recovery + one
seed-varied mock (×3 error models) + LOSO emulator errors. That IS the convergence-first ladder's STEP A —
and they shipped a Planck-tension cosmology result on it.** The convergence-first plan is therefore not a
shortcut below the field; it matches (and STEP A's *4 long chains* exceeds) the bar of the paper it reproduces.

### 1.2 The contemporary DESI standard — DESI DR1 P1D cosmology (2601.21432, 2026, LaCE)

The most current, highest-bar sibling (same DESI DR1 data, neural emulator). Their validation:

- **30 leave-one-out emulator tests** ("train a new emulator using all simulations except one … evaluate
  against the excluded simulation"; "better than 1% across the parameter space").
- **Mock recovery on hydro mocks from THREE different suites** — MP-Gadget, Lyssa (Nyx), Sherwood
  (P-Gadget3) — i.e. recovery tested against **different codes, resolutions, and cosmologies**, the
  cross-suite robustness test. This is mock recovery on *several distinct truths*, but it is a *recovery /
  robustness* battery, **not a rank-statistic SBC or a large empirical-coverage ensemble**.
- **Blinding:** *"the analysis [was] validated … with cosmological parameters kept blinded throughout the
  validation process … We optimized the model and validated the analysis under blinded conditions to avoid
  experimenter bias."*
- **No formal SBC / posterior-rank coverage** ("the paper does not explicitly mention formal SBC or
  posterior rank statistics").

So the *modern* upgrade over Fernandez+2024 is **more LOSO + cross-suite mock recovery + blinding** — NOT a
coverage ensemble. The blinding is already in this repo's plan (memory `blinding-strategy`: parameter-blind
A_p,n_s, frozen-on-mocks). The cross-suite recovery is the part worth importing (see §3).

### 1.3 The other anchors (consistent)

- **Lyssa / eBOSS NN-emulator cosmology (2412.05372, 2025):** **single fiducial mock recovery** ("recovers
  very well the cosmological parameters of the fiducial simulation … no significant bias"; explicitly *one*
  fiducial, "not multiple independent realizations") + LOSO (≤1–2%); MontePython sampler; **no SBC/coverage,
  no blinding**.
- **LaCE line (Pedersen+2021 2011.15127, Cabayol-Garcia+2023 2305.19064):** emulator LOSO to sub-percent +
  fiducial mock recovery; the emulator papers establish accuracy, the cosmology applications add fiducial
  recovery; **no SBC/coverage ensemble**.
- **Walther+2019 (1808.04367, thermal P1D):** forward-modeled mock-spectra recovery against a ~70-sim grid
  emulator; recovery/coverage-by-eye on the thermal parameters; **no formal SBC**.
- **Karaçaylı III (2306.06316, the KODIAQ/DESI-EDR P1D measurement, this repo's KS leg):** mock validation
  is at the **P1D-estimator** level (quickquasars synthetic spectra → estimator is unbiased + covariance
  correctly captured), NOT a cosmology-posterior coverage test — it is a *measurement* paper, not an
  inference paper, so its mocks certify the data vector, which is the C_data this repo ingests.

### 1.4 Where SBC/coverage IS the standard (and why this analysis is NOT that regime)

SBC (Talts+2018, Säilynoja+2022 ECDF bands — the very references this repo's `closure_sbc` cites) and large
empirical-coverage ensembles are the accepted bar for **amortized simulation-based inference (SBI)** —
likelihood-free neural posteriors (NPE/NLE/NRE) where (a) there is **no explicit likelihood** to trust, and
(b) the amortized posterior makes running N≫100 mocks essentially free. In that regime SBC is *necessary*
because nothing else certifies the learned posterior is calibrated. **This analysis is the opposite regime:
an explicit, differentiable Gaussian likelihood (`data_loglik`) sampled by exact NUTS.** The posterior is
not "learned" — it is the exact Bayesian posterior of a stated model, up to (i) sampler convergence and (ii)
model misspecification. Convergence is checked by R-hat; misspecification (sim≠emu, LF→HR, HCD) is checked
by mock recovery + the C_emu construction. The SBI/SBC literature itself flags that for "posteriors that
conduct inference with MCMC and hence are slow," SBC is the expensive path — which is exactly why no
explicit-likelihood Lyα MCMC analysis runs it.

**Subfield-standard bottom line:** the defensible, citable bar for an emulator + HCD-marginalized + 2-leg
explicit-likelihood NUTS analysis is **R−1<0.01 convergence (PRIYA-cosmology) + mock recovery on several
distinct held-out sims spanning the relevant cosmology, incl. a high-n_s one (DESI-DR1 cross-suite spirit) +
the LOSO emulator-error + C_emu construction this repo already has + blinding (already planned).** An SBC
rank ensemble or a large coverage ensemble would EXCEED every published Lyα-P1D cosmology analysis. It is
not wrong to run one, but it is not the field bar and is not what certifies the error bars.

---

## 2. CONVERGENCE vs COVERAGE — what each certifies here (the PI's question, answered)

The PI is methodologically right, and the convergence-first ordering is correct, for three lens-specific
reasons:

1. **Cosmology P1D posteriors ARE hard to converge, and that is the dominant risk for THIS sampler.** The
   geometry is a 25-dim dense-mass NUTS over θ9 + a 13-rung τ₀ ladder + 3 HCD α — with the documented
   A_p–τ₀ amplitude/mean-flux degeneracy (the classic forest funnel), an n_s–τ₀ ~0.13 rotation under MF,
   and prior-edge effects near n_s=0.98. These are *convergence* pathologies (funnels, ridges,
   multimodality), and they bite hardest exactly in (A_p, n_s, τ₀). A coverage ensemble of *short*
   chains (ESS≈400 is fine; the smoke ran ESS≈18 and was visibly multimodal, `mf_closure_smoke_posterior`)
   would AVERAGE OVER these pathologies and could report "nominal coverage" while every individual chain is
   under-converged — coverage is not sensitive to per-chain convergence failure if the failures are
   symmetric. **Long multi-chain R-hat on a few mocks is the ONLY thing that detects a funnel.** STEP A is
   the right first gate.

2. **Coverage on `final_fold0` cannot certify the error bars where the real fit lands (the load-bearing
   structural fact).** I verified the fold structure live: the 8 LOSO folds are **n_s-ORDERED**. The
   production model is `final_fold0`, whose held-out validation set is **only the lowest-n_s octant: 8 sims,
   n_s∈[0.803, 0.829]**. The real fit is at n_s≈1.009 (eBOSS) / DESI-DR1 (≈Planck). So a Leg-B coverage
   ensemble built on `final_fold0` (as `closure_legb`/`profile_legb_mtd10`/`profile_mf_closure_smoke` all
   are) draws its truth from `sims[m % len(sims)]` over **8 distinct low-n_s truths**, reused with fresh
   cosmic noise. **N=99 or N=300 mocks ≠ 99/300 different truths from the prior** — it is 8 truths each
   repeated ~12–37×. It is an empirical-coverage estimate over *cosmic-noise realizations at 8 fixed low-n_s
   points*, ~30 n_s-units (in σ) away from the real-data tilt. It says almost nothing about coverage at
   n_s≈1.0, which is precisely the regime the MF correction extrapolates into (HR ceiling 0.979; T4 §4 edge
   budget) and the one the analysis most needs to defend. **A big-N coverage run on fold 0 buys precision on
   a number that does not certify the real result.** This is the single most important point in this review.

3. **What coverage genuinely WOULD add is already provided more cheaply.** Coverage's job is to certify that
   C = C_data + C_emu is the right size so the posterior width is calibrated. But the repo already certifies
   C-sizing by (i) the whitened-residual-at-truth χ²/dof ≈ 1.24 (DESI) / 0.83 (KS) at the MF mock truth
   (`mf_closure_whitened_resid` — a direct, per-mock coverage-of-the-data-vector check), and (ii) the
   T3/T4/rotation Fisher certs that the MF correction is in-gate (+0.026σ) and the floor re-widens the
   under-tested LF→HR + n_s-edge directions. The remaining uncertainty is whether the FULL non-Gaussian
   posterior (not the Fisher scout) covers — and that is answered by **per-mock bias from STEP A's converged
   chains** (bias in σ-units, with honest s.e.), which is what Fernandez+2024 reported and shipped on.

**Net:** convergence on a few well-chosen mocks certifies the sampler (the dominant risk); per-mock bias
from those converged chains certifies the centering; the existing χ²/dof + C_emu-floor certify the width.
The large coverage ensemble certifies width-calibration at 8 low-n_s points the real fit never visits — low
marginal value at high cost.

---

## 3. THE MINIMAL DEFENSIBLE BAR (this lens) + my required changes

The convergence-first ladder is endorsed with these changes, which bring it to the DESI-DR1-spirit bar:

**STEP A (the binding cert) — required composition, not just count:**
- **R−1 < 0.01 Gelman-Rubin across ≥4 chains per mock** (the PRIYA-cosmology criterion, NOT just ESS≥400).
  Report R-hat per (A_p, n_s, τ₀-rungs, α). ESS≥400 is a floor, R-hat is the gate. This is what "make sure
  results converge" means in the subfield and it is the PI's actual concern.
- **The fiducials MUST span the cosmology the real fit visits — especially a HIGH-n_s mock.** "low/mid/high
  n_s" must be in TRUTH terms, and "high" must be n_s≈0.96–1.0. CRITICAL: a high-n_s mock cannot be drawn
  from `final_fold0` (max 0.829). Use a held-out HIGH-n_s sim and the matching fold's model (fold 6/7 reach
  0.956–1.040), OR explicitly note that the production `final_fold0` is NOT validated at high n_s and add a
  fold-7-model recovery as the high-n_s anchor. Otherwise STEP A inherits the same blind spot as the
  coverage ensemble. This is the change that makes the recovery *relevant to the real fit*.
- **Recover (A_p, n_s) to < 0.2σ per mock with converged chains** (matches the project gate AND
  Fernandez+2024's "<1σ"; 0.2σ is stricter, fine). Report the bias in the mock's OWN posterior σ (so a
  tightening can't hide a bias, per the rotation note).
- **Include at least one mock built at a DIFFERENT τ₀ anchor / a different sim resolution** if feasible —
  the DESI-DR1 cross-suite-recovery spirit. Within this repo the cheap version is: one LF-truth mock + one
  MF/HR-resolution mock (the gate invariant already supports `mf=` on the truth), to show recovery survives
  the resolution correction it depends on.

**STEP B (the SBC/coverage ensemble) — NOT necessary; conditional and reframed:**
- Run it ONLY if STEP A surfaces a per-mock bias trend or a borderline funnel that a single calibration
  number would resolve, OR if a referee explicitly demands coverage. It is a *cross-check*, never the gate.
- If run, it is a **Leg-B EMPIRICAL-COVERAGE** ensemble, NOT SBC. The code is honest about this
  (`closure_legb` §810: "Leg-B's null is NOT rank-uniform … this ECDF must NEVER gate the Leg-B verdict";
  the L_FLOOR=99 ECDF band is a Leg-A/`closure_sbc` object). Do not let the L_FLOOR=99 / N≥99 framing import
  a true-SBC justification that does not apply to held-out-sim truths. Size it to budget (N≈99, ~600 CPU-h),
  span MULTIPLE FOLDS so the truths actually vary in n_s (a single-fold ensemble is the trap of §2.2), and
  report it as a calibration cross-check with the explicit n_s-coverage caveat.

**The standing physics caveats that the validation report MUST carry regardless (from my prior reviews,
unchanged):** (i) the C_emu floor covers LF→HR generalization, NOT HR→truth — the k<0.06 cap does that;
(ii) any posterior mass above n_s≈0.98 rests on MF-correction extrapolation (T4 §4 sigma_edge), guarded not
nominal — this is *exactly* why STEP A needs a high-n_s recovery; (iii) the closing-step incidence z-slope
must be MARGINALIZED in the real-fit config, not fixed to the truth's w_c(z) (onboarding R1) — STEP A should
run at least one mock with the z-slope sampled (real-fit config), not fixed, or the recovery is
necessary-but-not-sufficient.

---

## 4. CONCERNS (severity + fix) — see structured object for the canonical list

- **[CRITICAL] Fold-0 blind spot.** Both STEP A "fiducials" and any STEP B ensemble, as currently scoped on
  `final_fold0` (n_s∈[0.803,0.829]), cannot certify (A_p, n_s) coverage/recovery at the real-data n_s≈1.0.
  *Fix:* add a high-n_s recovery using a high-fold model/sim (fold 6/7); state that `final_fold0` is the
  low-n_s-octant model and the real fit's tilt regime is certified by the high-n_s mock + the T4 edge budget.
- **[IMPORTANT] "Long chains" needs R-hat, not just ESS.** The PI's concern is convergence; the cert must
  report Gelman-Rubin R−1<0.01 (PRIYA-cosmology bar) across ≥4 chains, not only ESS≥400. *Fix:* add R-hat to
  the STEP A gate.
- **[IMPORTANT] SBC label misuse.** Calling the Leg-B ensemble "SBC" / leaning on L_FLOOR=99 imports a
  rank-uniformity justification that is false-by-construction for held-out-sim truths (the code says so).
  *Fix:* call STEP B a "Leg-B empirical-coverage cross-check"; if run, span multiple folds; never gate on
  the ECDF.
- **[IMPORTANT] Recovery must use the real-fit incidence config.** If STEP A fixes the truth's w_c(z)
  z-slope in the forward, it is circular (onboarding R1). *Fix:* at least one STEP A mock fits with the
  per-class z-slope MARGINALIZED.
- **[MINOR] Import the DESI-DR1 cross-suite-recovery spirit cheaply.** Add an MF/HR-resolution-truth mock
  recovery alongside the LF-truth one (the `mf=` truth path already exists) to show recovery survives the
  resolution correction. Blinding is already planned (memory) — keep it.

---

## 5. Files / evidence
- PRIYA cosmology paper text: `/tmp/priya_cosmo.txt` (41pp, 2309.03943; §3.5 + §3.4 quoted above);
  context windows: `tool-results/br3ri4f89.txt`.
- Fold structure verified live: `final_fold0` held-out = 8 sims, n_s∈[0.803,0.829]; folds are n_s-ordered,
  fold 7 reaches 1.040 (`held_out_sims`, `closure_legb.py:283`).
- Leg-B cycles `sims[m % len(sims)]` (`closure_legb.py:688`) → N>8 reuses 8 truths with fresh noise.
- Leg-B null is NOT rank-uniform; ECDF is diagnostic-only (`closure_legb.py:810-820`).
- Existing C-sizing cert: `mf_closure_whitened_resid` χ²/dof 1.24/0.83.
- Subfield: 2309.03943 (R−1<0.01 + 1 MAP + 1 seed mock, no SBC); 2601.21432 (30 LOSO + 3-suite mock
  recovery + blinding, no SBC); 2412.05372 (1 fiducial mock + LOSO, no SBC); SBC = amortized-SBI standard
  (Talts+2018, Säilynoja+2022), not explicit-likelihood NUTS.
