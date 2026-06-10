# STEP-A closure: HCD + τ₀ prior investigation (2026-06-10)

PI redirect after the Tier-1/Tier-2 bias results (walkthrough `2026-06-09-stepA-closure-walkthrough.md` §6):
the closure converges cleanly (R-hat≈1.007, 0 div) but recovers (A_p, n_s) biased ~0.8–1.8σ in the interior
and worse at the box edges. PI's three suspicions, to be checked by a Lyα-forest agent + a Bayesian/PPL agent:

1. **The HCD nuisance prior correlates with (A_p, n_s)** — does the α-prior pull cosmology?
2. **The τ₀ prior is the prime suspect.** The closure samples **13 INDEPENDENT per-z rungs** on the α-ladder
   coordinate (τ₀/Kim(z)), each `Normal(Becker+2013 center, σ=5%·μ)`. **PRIYA instead marginalizes τ₀ with a
   smooth 2-parameter model: an AMPLITUDE × a z-SLOPE power-law in (1+z), with UNIFORM priors.** τ₀ is very
   constraining and degenerate with A_p, so a structurally-wrong / over-flexible / mis-centered τ₀ prior can
   bias A_p and n_s.
3. **The HCD marginalization itself** — check the recovered HCD posteriors against the mock truth and whether
   the additive-excess + per-class-incidence assumptions hold.

## Empirical findings from the completed STEP-A chains (the data to review)

Pooled over 4 chains/fiducial. `bias_z = (truth − post-mean)/post-sd`; **+ = recovered LOW.** Cosmology in
box-normalized units. Source: `checkpoints/stepA/{L1a_fold0,L1a_fold4,L1a_fold7,M1,M2}_c{0..3}.npz`.

### (A) HCD α posterior recovery vs mock truth + correlation with cosmology

| fiducial | α_LLS bias_z | α_subDLA bias_z | α_DLA bias_z | corr(α_LLS,A_p) | corr(α_subDLA,n_s) |
|---|---|---|---|---|---|
| L1a-lo (n_s .810, LF)  | −0.22 | **+1.90** | −0.80 | −0.36 | +0.27 |
| L1a-mid (.920, LF)     | +0.01 | **+3.29** | −0.38 | −0.37 | +0.39 |
| L1a-hi (.998, LF)      | +0.16 | **+2.06** | −0.63 | −0.35 | +0.36 |
| M1 (1.019, MF)         | +1.01 | **+4.09** | −0.34 | −0.48 | +0.39 |
| M2 (1.040, τ₀-extreme, MF) | −1.36 | **+3.67** | −0.82 | −0.57 | +0.14 |

- **`α_subDLA` is recovered 2–4σ LOW in EVERY fiducial** (truth ~0.07–0.10 → posterior ~0.02–0.03, ≈⅓ of
  truth). Systematic, not scatter. The subDLA incidence the mock carries is NOT being reproduced.
- `α_LLS` recovers near truth (except M2's funnel) but **anti-correlates with A_p at −0.35 → −0.57**.
- `α_subDLA` **correlates with n_s at +0.27 → +0.39**.
- `α_DLA` recovers near truth (slightly high), small couplings.

### (B) τ₀ ↔ cosmology + τ₀ posterior structure

- **`max|corr(A_p, τ₀-rung)| ≈ 0.46–0.58`** every fiducial — A_p is the most τ₀-degenerate parameter.
  `max|corr(n_s, τ₀)| ≈ 0.15–0.31`.
- **τ₀ rung displacement from truth is JAGGED**, |2nd-difference| RMS ≈ 1–2 (not ≪1) → the 13 independent
  per-z rungs wiggle individually to absorb the emulator residual, producing a non-smooth τ₀(z). A smooth
  amplitude+slope model would forbid this. Per-fiducial displacement vectors (mean−truth)/sd over the 13
  rungs, e.g. L1a-hi: `-2.3 -1.4 -1.6 -1.2 +0.3 +0.1 +0.2 -0.4 -0.8 -0.9 -1.5 +0.6 -0.4` (low-z rungs pulled
  hardest). M2: ALL rungs −7 to −10σ (the τ₀-extreme anchor sits far from the Becker-centered prior → funnel,
  R-hat 1.019, A_p recovered +53% high).

### (C) The downstream cosmology bias (context, from §6)

L1a (LF, 4-chain): n_s bias fold0 −1.40σ, fold4 +0.01σ, fold7 −0.84σ; A_p +0.10/+0.61/+0.76σ. MF: M1
n_s +1.80σ/A_p +1.58σ (n_s pulled to the 0.995 dense-edge), M2 n_s −2.82σ/A_p −3.85σ (extrapolation ridge +
funnel). L1b corners (single chains) up to +6σ on the n_s-min sims.

## Code pointers

- τ₀ prior: `hcd_analysis/emulator/closure_legb.py:211` `meanflux_tau0_prior(z_global, center="becker13")` →
  `tau0_mu/tau0_sigma` per-z; `meanflux_logprior` in `inference.py:42` (per-rung Normal). Truth τ₀ selection:
  `make_truth_from_sim(..., tau0_anchor="becker13")` picks the ladder rung nearest Becker per z
  (`closure_legb.py:311,328-370`). The α-ladder coordinate + `becker13_tau0` are in `meanflux_prior.py`.
- HCD α prior: `inference.py` `hcd_incidence_prior` (~line 93-113), per-class Normal centered on (lit/sim)·w_c;
  DLA softplus on `HCD_DLA_RESIDUAL_FRAC=0.10`, σ/μ=0.50. Per-leg amplitude anchors KS σ=0.27, DESI σ=0.12.
- Forward (additive excess): `P_obs = P_clean + Σ_c α_c·(P_c − P_clean)`, `α_c(z)=a_pivot·exp(δ_leg)·
  g_fixed,c(z)·((1+z)/(1+z_p))^{s_c}` — `data_likelihood.predict_P_obs_on_leg` / `closure_legb`.
- PRIYA mean-flux model reference: Bird+2023 arXiv:2306.05471 (the τ₀ amplitude+slope marginalization).

## Questions for the agents (ground every claim in the code/chains/refs; end with a prioritized plan)

**Bayesian/PPL lens:**
- B1. Is the **13-independent-per-z-Gaussian** τ₀ prior the structural problem vs PRIYA's **2-param
  amplitude+slope uniform**? The jagged τ₀ displacement (B above) suggests the rungs absorb emulator residual.
  Would a smooth slope model (a) reduce the A_p/n_s bias by denying that absorption channel, or (b) just
  relocate the misfit? Is the 5% Gaussian over-informative or mis-centered relative to the mock's true τ₀?
- B2. Why does **α_subDLA collapse to ⅓ of truth (2–4σ low) everywhere**? Degeneracy with α_LLS / cosmology
  amplitude / τ₀? A prior-vs-likelihood tension? Is it an identifiability problem (subDLA template ≈ a linear
  combination of LLS + clean)? Check the template overlaps and the α_LLS↔α_subDLA posterior correlation.
- B3. Are the HCD↔cosmology correlations (α_LLS↔A_p −0.5, α_subDLA↔n_s +0.35) benign marginalization or are
  they biasing? Does the per-leg amplitude anchor (KS σ=0.27 / DESI σ=0.12) interact with this?
- B4. Concrete, prioritized plan: what to change (τ₀ model? subDLA prior/template? anchors?) and what cheap
  test confirms each (e.g. a forward-only Fisher/profile, a re-fit with the slope τ₀ model, a template-overlap
  check) before any expensive re-run.

**Lyα-forest lens:**
- L1. Is the **additive-excess + per-class-incidence** HCD model physically right, and is the **subDLA
  template/prior** reasonable? Does recovering subDLA at ⅓ of truth indicate a template normalization or a
  prior-center misspecification (cf. the Rogers normalization / dN/dX-vs-literature work)?
- L2. What does **PRIYA / the Lyα-P1D subfield actually use for the τ₀ mean-flux marginalization** (amplitude
  + z-slope, uniform; Becker τ_eff)? Is the closure's per-z-independent treatment defensible, or is the smooth
  slope model the field standard we should match? What σ on τ₀(z) is realistic (Becker reports ~3–8% z-dep)?
- L3. HCD posterior-vs-mock physical consistency: are the recovered α (LLS near truth, subDLA low, DLA ok)
  physically sensible given the templates' shapes and where each class lives in (k,z)? Any sign the mock's HCD
  content is being mis-attributed across classes?
- L4. Prioritized plan from the Lyα side (template fixes, prior re-centering, z-slope treatment).

---

## RESOLUTION (2026-06-10) — the literature is unanimous, and the PI's design

**Literature read (2 parallel agents, quote-grounded):** ALL the P1D measurements we fit mask **only DLAs
(log N_HI ≥ 20.3)** and **keep the full subDLA population (10¹⁹–2×10²⁰) in the forest**:
- DESI DR1 QMLE (Karaçaylı 2025, arXiv:2505.07974 = our `/home/mfho/data/desi_dr1_p1d/`): *"Sub-DLA detections
  contain many false positives, so we do not mask them."* subDLAs template-marginalized, not masked.
- eBOSS (Chabanier 2019), DESI EDR FFT/QMLE (Ravoux 2023 / Karaçaylı 2023), DESI DR1 FFT (Ravoux 2025): all
  20.3-only, Voigt core>20% + wing correction; subDLAs kept. The "19.5" in the QMLE paper is a *mock*
  diagnostic, not the data mask.
- KODIAQ-SQUAD + XQ-100 (Karaçaylı 2022 arXiv:2108.10870; Iršič 2017): mask only ≥20.3 via Sánchez-Ramírez
  2016 / Murphy 2019 catalogs, which *explicitly exclude* subDLAs. Karaçaylı flags un-removed subDLA/LLS as a
  known residual contaminant (Rogers 2018).

**Implication:** our forward uses the τ=1e6-**filtered** subDLA template (~37% of subDLA power removed; cache
surviving fraction 0.63, IQR[0.46,0.87]) → it **under-represents subDLAs vs the data**, which keeps them all.

**PI DESIGN (2026-06-10), the clean three-way separation:**
- **Emulator unchanged** — keeps emulating PRIYA's τ=1e6-filtered tier_p (still "matches PRIYA"; no retrain).
- **Forward adds back the missing subDLA** as an HCD-template term: the cache's `delta_subDLA = P_unfiltered −
  P_filtered` (`data.py:128`), so the subDLA excess becomes the **full (unfiltered)** population. Scaled by α
  whose **prior is calibrated to the filtered→unfiltered subDLA ratio** (center restores the full population;
  width = the 0.63 IQR scatter folded with the literature subDLA dN/dX uncertainty). subDLA amplitude is now
  anchored to **literature-physical full subDLAs**, not the sim's filtered remnant.
- **Mock keeps the full subDLA population** (filtered tier_p + full subDLA add-back) → the closure is
  **realistic to the data**, not the over-masked sim convention.
- DLA stays masked + 10% residual (data masks DLAs); LLS stays (below the filter floor).

Net: emulator↔PRIYA(filtered), subDLA nuisance↔literature(full), mock↔data(full). This attacks the A_p
deficit at root (the forward stops missing 37% of the subDLA power the data carries). The residual subDLA↔
LLS↔DLA identifiability degeneracy (KS 0 modes <k=0.005) is then *marginalized* (prior correctly centered)
rather than biasing; merging the high-N_HI nuisances remains the robust belt-and-braces.
