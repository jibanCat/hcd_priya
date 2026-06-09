# Meta-review — Phase-C MF closure VALIDATION plan (convergence-first ladder vs the SBC coverage ensemble)

**Date:** 2026-06-09
**Branch:** `phase2c-likelihood`
**Reviewers synthesized:** Bayesian/PPL (ENDORSE_WITH_CHANGES, ensemble: *conditional*) · CS/ML/JAX (ENDORSE_WITH_CHANGES, ensemble: *conditional*) · Lyα-P1D subfield (ENDORSE_WITH_CHANGES, ensemble: *no*)
**Question:** Is the convergence-first ladder right? Is the SBC coverage ensemble NECESSARY to certify the real-fit (A_p,n_s) error bars, or is convergence + bias on a few fiducials enough? What is the minimal DEFENSIBLE validation bar before the real DESI+KS fit?

---

## VERDICT (the decision, not a recap)

**The convergence-first ladder is RIGHT and is ADOPTED.** All three lenses endorse the *ordering* (converge on fiducials first, then — maybe — an ensemble). The PI's instinct is sound: do not burn budget on chains that may not converge, and a handful of long well-converged fiducials carry more information per CPU-h about the things that can actually go wrong (sampler pathology + point bias) than a fan-out of short chains.

**The SBC coverage ensemble is CONDITIONAL — and, as literally scoped on `final_fold0`, it is NOT-NEEDED in its current form.** I rule it **CONDITIONAL** rather than necessary, with a strong steer to **not run the big N as designed**, for one structural reason all three lenses converge on and that I verified live in the code:

> `closure_legb.py:688` does `sim = sims[m % len(sims)]`, cycling **only the ~8 held-out sims of `final_fold0`**, and the folds are **n_s-ordered** so fold-0's held-out octant is **n_s ∈ [0.803, 0.829]**. The real fit lands at **n_s ≈ 1.009** (rotation review line 139: "above the HR ceiling 0.979"). So N=99/300/600 mocks are **8 fixed low-n_s truths re-drawn with fresh cosmic noise**, NOT "N different truths from the prior." The handoff's premise for the ensemble is **false-by-construction**.

That single fact collapses two of the three claimed benefits of the big ensemble:
1. It is **not SBC** — the code itself says so (`closure_legb.py:7-8`: "judged on one-sided empirical coverage (≥ nominal) + bias, NOT rank uniformity (uniformity is false-by-construction here)"). So the `L_FLOOR=99`/rank-ECDF machinery (a Leg-A `closure_sbc` object) does **not** license the N≥99 floor for Leg-B. The ECDF is correctly tagged DIAGNOSTIC-ONLY and **must never gate**.
2. As a **noise-coverage** check over 8 fixed truths, it **saturates by N~30-40** (CS lens). Re-drawing noise on 8 truths 37-75× (N=300-600) is **pure waste** of the scarce ~4000 CPU-h, AND it measures width-calibration at 8 points that are **tens of σ from where the real fit lands** (Lya lens), i.e. precision on a number irrelevant to the real result.

So the original N=99-600 single-fold ensemble is **over-validation aimed at the wrong place.** The defensible bar does NOT require it. What the analysis *does* need — and what the ensemble was a clumsy proxy for — is (a) sampler convergence at the geometry the real fit will visit, (b) point-bias <0.2σ where it matters (the **edge**, not the median), and (c) a width-calibration argument for `C = C_data + C_emu` that holds at **n_s ~ 1.0**, not at n_s ~ 0.82.

---

## Reconciling the three lenses

The lenses **agree far more than the conditional/conditional/no split suggests.** All three say: ladder ordering correct; R-hat and bias certify orthogonal things and **both** must be gated; the big single-fold ensemble is not the binding check. They split only on **whether a *modest* ensemble is worth running at all** once STEP A passes:

| Question | Bayesian | CS | Lyα | Reconciled meta-call |
|---|---|---|---|---|
| Ladder ordering correct? | Yes (convergence is a strict *pre-req* for coverage) | Yes (better CPU-h use) | Yes (matches Fernandez+2024) | **Yes — adopted** |
| R-hat alone sufficient? | No — coverage is the only direct error-bar cal test | No — R-hat is blind to shared bias | No — but ESS≠convergence; gate on R-1 | **No. Gate BOTH R-hat AND per-mock bias z** |
| Big ensemble (N=300-600)? | Resist; N=99-150 if any | Resist; N≤48-99 if any | Exceeds every published Lyα analysis | **Do not run big-N. Wasteful (8-truth saturation)** |
| Modest ensemble necessary? | *Conditional* — required IF claiming calibrated intervals (the blinded fit does) | *Conditional* — garnish, only after STEP A + only if edge bias appears | *No* — convergence + bias + the existing C_emu certs suffice | **CONDITIONAL — see decision rule below** |
| Where is the real risk? | n_s>0.98 edge + low-k coherent C_emu mode | tau0 funnel + n_s-edge + fast_postprocess golden | **Fold-0 blind spot at n_s~1.009** | **The fold-0/edge blind spot is the dominant risk** |

**The reconciliation:** the Bayesian "conditional-because-blinded-intervals" and the Lya "no-because-not-subfield-standard" are not actually in conflict once the fold-0 fact is admitted. The Bayesian lens is right that *a calibrated-interval claim wants a coverage certificate*; the Lya lens is right that *the single-fold ensemble as coded does not provide one at the real cosmology*. Both are satisfied by the same move: **replace the single-fold noise-coverage ensemble with a small, MULTI-FOLD, edge-spanning coverage cross-check** — i.e. pool held-out sims across folds so truths actually vary in n_s up toward 1.0 (CS lens's "pool across folds" + Lya lens's high-n_s mock). That is the form in which a coverage object is *worth* the CPU-h; the single-fold form is not.

**The CS/Bayesian "conditional" is therefore the operative verdict, with the Lya lens's caveat folded into its DESIGN** (multi-fold span, never gate on ECDF, rename off "SBC").

---

## THE BAR — anchored to the Lyα-forest P1D subfield standard

All three lenses independently read the primary literature and converged on the same subfield fact, which is the anchor for the bar:

> **Lyα-forest P1D cosmology analyses do NOT run an SBC rank ensemble or a large empirical-coverage ensemble.** Their de-facto validation is **convergence (Gelman-Rubin R-1 < 0.01) + a few fiducial/seed-varied mock recoveries to <1σ + leave-one-out emulator accuracy + (modern) blinding.**

Concretely, from the PDFs the lenses cite:
- **Fernandez, Bird & Ho 2024 (2309.03943) — the paper THIS analysis reproduces:** R-1<0.01 (Cobaya/Metropolis) + **one** MAP recovery on an HF sim (<1σ) + **one** seed-varied LF mock with 3 error-model chains (<1σ) + LOSO emulator errors (0.2% LF / 1% HF). "coverage", "SBC", "blind" appear **zero times in 41 pages.**
- **DESI DR1 P1D cosmology (2601.21432, 2026, LaCE) — the contemporary highest bar:** 30 LOSO emulator tests + **mock recovery across THREE hydro suites** (different codes/resolutions/cosmologies) + **blinding** — and *still* no formal SBC/posterior-rank ensemble. Its robustness evidence is in the **systematic-variation** runs, not a coverage fraction.
- **Lyssa/eBOSS NN emulator (2412.05372), LaCE (Pedersen+2021/Cabayol-Garcia+2023), Walther+2019:** single-fiducial recovery + LOSO; no SBC.
- **SBC (Talts+2018, Sailynoja+2022)** — the refs `closure_diagnostics` implements — is the standard for **amortized SBI** (likelihood-free neural posteriors), the *opposite* regime from this explicit-differentiable-Gaussian-likelihood NUTS pipeline. The SBI literature itself flags SBC as the **expensive** path for MCMC posteriors.

**THE DEFENSIBLE BAR (the gate before the real DESI+KS fit):**

> **R-1 < 0.01 Gelman-Rubin across ≥4 DISPERSED-INIT chains per fiducial** (the PRIYA-cosmology gate; ESS≥400 bulk *and* tail is a secondary floor, plus the standard NUTS health battery: 0 divergences after the 0.9→0.99 target_accept ladder, max-tree-depth saturation <~2%, E-BFMI>0.3), **AND per-mock recovery bias |z|=|(truth−postmean)/postsd| < 0.2σ on (A_p, n_s)** at every fiducial — **across fiducials that SPAN the real-fit cosmology, INCLUDING a high-n_s mock at n_s ≈ 0.96-1.0** (which cannot come from `final_fold0` whose max is 0.829 — use a high-fold model/sim and explicitly state `final_fold0` is the low-octant production model), **with the HCD incidence z-slope MARGINALIZED** in at least one fiducial (real-fit config, anti-circularity), **plus** the existing LOSO emulator errors, the cross-class C_emu construction (+ the KS small-scale floor / sigma_edge), and blinding.

This **matches and slightly exceeds Fernandez+2024** (it adds explicit rank-R-hat/ESS/divergence/funnel reporting the published papers do not even show) and **imports the DESI-DR1 robustness spirit** (cross-cosmology, ideally cross-resolution, recovery). A coverage ensemble — single- or multi-fold — would **exceed every published Lyα-P1D cosmology analysis**; it is a strengthening, not the gate.

**Why a strengthening is nonetheless defensible to *offer*:** this analysis has one feature the published ones lack — a **diagonal-in-k C_emu that provably cannot whiten the documented coherent low-k mode** (C_emu/C_data spike ~0.52 at low-k, the A_p/n_s-sensitive regime). A single converged fit reproduces the model's posterior *exactly*, so if the model is misspecified there, the converged interval is "confidently wrong with no signal" (Bayesian lens). Coverage over varied truths is the one direct test that bounds that. **But** the Lya lens's counter is decisive on cost-effectiveness: the misspecification is *already* bounded by the whitened-residual-at-truth χ²/dof = 1.24 (DESI) / 0.83 (KS) + the T3/T4/rotation Fisher certs + the C_emu floor. So a coverage object adds marginal assurance, not a missing certificate — which is exactly why it is **CONDITIONAL, run only in the cheap multi-fold form, and only after STEP A passes.**

---

## THE COVERAGE-ENSEMBLE DECISION RULE (CONDITIONAL — the operative ruling)

Run STEP A first. Then:

- **If STEP A passes clean** (R-1<0.01 everywhere incl. the n_s-edge and tau0-extreme fiducials; |bias|<0.2σ at the high-n_s mock; marginalized-slope recovery holds) **AND no truth-dependent edge bias appears** → the **defensible bar is already met. The big single-fold ensemble is NOT-NEEDED.** Proceed to the real fit. *(This is the Lya-lens outcome, and the most likely one given the +0.035σ all-folds bias + rotation σ-ratios ≈ 1.00.)*
- **If STEP A reveals truth-dependent bias near the edge, OR the report will headline a *calibrated* (not caveated) credible-interval claim** → run a **SMALL, MULTI-FOLD, edge-spanning coverage cross-check (N≈48-99)**, pooling held-out sims across folds so truths span up toward n_s~1.0; per-mock single-chain ESS≥400 (R-hat already certified in STEP A); **never gate on the rank-ECDF**; report as a calibration cross-check with the n_s-coverage caveat. **Do NOT run N=300-600** (8-truth saturation if single-fold; and even multi-fold, N~48-99 resolves the 0.2σ gate at ~2σ — see Bayesian arithmetic: bias s.e. 0.101 at N=99).

In neither branch is the original single-fold N≥99-600 SBC ensemble the right object.

---

## ORDERED NEXT STEPS

**0. CS pre-flight (mandatory, before any fan-out — ~minutes).** Golden-assert `fast_postprocess=True` reconstructs the 3 deterministic sites byte-identically: run one mock both `fast_postprocess` True/False, `assert allclose(rtol=0)` on `tau0_vec / alpha_dla / alpha_hcd_z`; freeze a golden `(P_model, C_total)` on the LF and MF paths. A silent reconstruction error would corrupt the entire ensemble as a *coverage anomaly, not a crash* (CS R1: no golden guard exists on disk).

**1. Fix the convergence harness so R-hat is valid (CRITICAL — Bayesian).** Replace `init_to_median` (`closure_legb.py:612`) with **dispersed inits** (`init_to_sample` / jittered over-dispersed prior draws) for the STEP-A chains. Identical starts make split-R-hat meaningless. Seed STEP-A chains on a **separate axis** `fold_in(k_nuts, chain_id)`, distinct from the `base_seed + attempt` retry stream (`:702`), to avoid seed-collision/correlated chains. Shard over `(mock, chain)` on SLURM, num_chains=1/task (not numpyro num_chains=4 — serializes on CPU). Use **warmup ≥ 150 (prefer 250-300)**; warmup=40 under-conditions the 25-dim dense mass and already killed one mtd10 run (experiment log §1.5).

**2. STEP A — convergence + point-bias gate (~130 CPU-h).** Fit a SPANNING set of fiducials, each ≥4 dispersed-init chains at production mtd=10, dense_mass, ESS≥400:
   - low / mid / high n_s × (low/high A_p), **PLUS**
   - **a high-n_s mock at n_s ≈ 0.96-1.0 from a high fold (6/7 reach 0.956-1.040)** — the real-fit regime `final_fold0` cannot reach (CRITICAL, Lya);
   - **the n_s-edge mock with sigma_edge active** (rotation review: ns>0.98 is "GUARDED, not nominal");
   - **a tau0-ladder extreme** (the centered Normal at high-z weak-data bins is a funnel source, CS);
   - **at least one mock with the HCD incidence z-slope MARGINALIZED** (anti-circularity, Lya);
   - **ideally one MF/HR-resolution-truth mock** (DESI-DR1 cross-suite spirit; the `mf=` truth path exists).

   **GATE:** rank-normalized split-R-1 < 0.01 (Gelman-Rubin, the PRIYA bar) on all 25 params; bulk AND tail-ESS ≥ 400 over **all** params (`thin_to_ess` thins by the worst param); 0 divergences post-ladder; max-tree-depth saturation <~2%; E-BFMI>0.3; no funnel neck on the `alpha_pivot·g(z)` product; **AND |bias z| < 0.2σ on (A_p,n_s) at every fiducial incl. the high-n_s edge** (MC error ~0.05σ at ESS=400 — 2× better than the whole N=99 ensemble resolves the average).

**3. If the funnel bites → reparametrize and re-run STEP A.** Test a **non-centered tau0-ladder** (`alpha_raw~N(0,1); ladder = mu/kim + (sigma/kim)*alpha_raw` — Jacobian-free, near-zero JAX risk) as a parallel arm; adopt it if it cuts leapfrogs without moving the golden-pinned posterior (CS — also makes any STEP B cheaper).

**4. Apply the coverage-ensemble decision rule (above).** Default expectation: STEP A passes → **skip the ensemble**, the bar is met. Only if edge bias appears or a calibrated-interval headline is wanted → run the **small multi-fold edge-spanning cross-check (N≈48-99)**, never the single-fold big-N.

**5. Tighten the 95% coverage estimator IF any ensemble runs (Bayesian, IMPORTANT).** `central_interval`'s `np.quantile` on L draws is biased LOW at finite L (94.7% at L=400, 88.8% at L=25), giving a false under-coverage verdict on the 95% gate. Gate against the **finite-L expectation under perfect calibration** (~10 lines in `_aggregate_legb`), or read the verdict off the 68% interval (exact at L=400) + bias-z, or push ESS to ~800-1000. The 68% and bias gates are essentially unaffected.

**6. Proceed to the real blinded DESI+KS fit** once STEP A (and any conditional cross-check) pass, with the standard NUTS health battery reported, parameter-blind on (A_p,n_s), frozen-on-mocks.

**Total cost:** STEP A ≈ 130 CPU-h; optional cross-check ≈ 260-540 CPU-h; **≤ ~670 CPU-h (≤17% of the ~4000 ceiling)** — leaving ample headroom for the Leg-A SBC pre-req, retries, and the real fit. The big ensemble (N=600 ≈ 3.65-4.0k) would have consumed essentially the entire budget for precision on the wrong number.

---

## DISSENTS / OPEN TENSIONS

- **Lya lens dissents from running ANY coverage ensemble** (verdict "no"): convergence + multi-fold bias recovery + the existing C_emu/χ² certs already exceed every published Lyα-P1D analysis, and the budget is better spent on **DESI-DR1-style systematic-variation runs** (estimator, covariance, emulator-variant, contaminant model) — the robustness evidence the modern bar actually treats as decisive. The meta-ruling **partially upholds** this: the ensemble is demoted to conditional/optional and the single-fold big-N is rejected; but a *small multi-fold* cross-check is kept available because the diagonal-in-k C_emu misspecification (which Fernandez+2024 did not have) makes a coverage object marginally more defensible here than in the published precedents.
- **Bayesian vs Lya on the binding certificate:** Bayesian holds coverage is the *only* direct error-bar calibration test (a converged fit cannot self-diagnose misspecification); Lya holds the whitened-residual χ²/dof + Fisher certs + C_emu floor already discharge that. Reconciled: both are right about *what* coverage uniquely tests; they differ on whether the *existing* certs are sufficient substitutes. Resolved pragmatically by the decision rule (cheap multi-fold cross-check only if a calibrated headline is claimed).
- **The "few long fiducials INSTEAD OF the ensemble" framing (PI's literal phrasing) is a category error** (all three lenses): fiducials certify the *sampler + point bias*; only varied truths certify *interval calibration*. They are complements, not substitutes. The ladder is right precisely because STEP A **licenses** (or obviates) STEP B — it does not replace its function. At N=2-3, coverage cannot be estimated at all (Wilson ±0.30). This is a framing fix, not a plan change.
- **`L_FLOOR=99` is a Leg-A object misapplied to Leg-B** (CS + Lya): it imports a rank-uniformity justification false-by-construction for held-out-sim truths. Any Leg-B coverage object must be renamed off "SBC", span multiple folds, and never gate on the ECDF. The code already tags the ECDF DIAGNOSTIC-ONLY — keep it that way.
