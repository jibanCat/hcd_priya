# Reference: EXTENDED PRIYA constraints paper (Fernandez, Bird, Ho 2024 JCAP 07 029)

- **ADS:** 2024JCAP...07..029F
- **arXiv:** 2309.03943 (v1 2023-09-07; v2 2024-07-17, the published JCAP version)
- **Title:** "Cosmological constraints from the eBOSS Lyman-α forest using the PRIYA simulations"
- **Companion / Ref [47]:** original PRIYA suite paper = Bird, Fernandez, Ho et al. 2023JCAP...10..037B = arXiv 2306.05471.
- Read via WebFetch of arxiv.org/html/2309.03943(v2) and arxiv.org/html/2306.05471v2 on 2026-06-08.

Bears on **USER FEEDBACK #2**: "the n_s>1.0 region of the extended PRIYA suite is poorly generalized — high-n_s sims are pulled down hardest (shrink-to-centre), consistent with poor coverage at high n_s."

---

## (a) Did the suite go 48 -> 60, and were the EXTRA sims chosen by Bayesian optimization?

**48 LF -> 60 LF (+ 3 HF unchanged).** Verbatim (Sec 2):
> "For this work we have added 12 new LF simulations to those of Ref. [47], which extend the
> simulated parameter range to better cover the posterior range allowed by SDSS DR14."

**The 12 NEW sims (this paper): the paper attributes their placement to a space-filling rule, NOT to Bayesian optimization.** The only design sentence it gives is (Sec 2):
> "Sampled parameters are chosen to maximise spread in parameter space, as described in Ref. [47]."
No "Bayesian optimisation"/"active learning"/"acquisition" language appears anywhere near the 12-sim extension. So as far as the 2024 paper *states*, the 12 extension points are a space-filling addition (maximise spread), described by reference to the Ref [47] method.

**BUT the ORIGINAL 48-sim design (Ref [47], 2306.05471) WAS a hybrid that DID use Bayesian optimization.** Verbatim (2306.05471, Sec 2.8 "Experimental design"):
> "We ran 30 simulations in our initial Latin Hypercube design. We then ran 2 more low fidelity
> at parameters chosen by Bayesian optimisation. However, we found that in our initial space
> Bayesian optimisation frequently chose points on the extreme boundaries of the parameter space.
> We thus ran an extra 8 simulation Latin Hypercube, before 3 more low fidelity simulations whose
> simulation parameters were chosen with Bayesian Optimisation."
> "...we ran 6 simulations uniformly spaced between 1.325 <= alpha_q <= 1.575" (a refinement to
> expand the low-alpha_q limit after comparing to observed mean IGM temperatures).
30 LH + 2 BO + 8 LH + 3 BO + ~5-6 refinement ≈ 48. The HF set was chosen by an active-learning
criterion (single-fidelity emulator trained on 2-of-N LF to predict the rest).

**Net answer to (a):** The user's "extra sims chosen by Bayesian optimization not a Latin hypercube"
is HALF-RIGHT but pointed at the wrong batch. The original 48 mixed LH + BO (and BO is explicitly
reported to have over-sampled boundaries). The *new 12* that opened up n_P>1.0 are described as
space-filling ("maximise spread"), referencing the same Ref [47] machinery — the paper is not fully
explicit on whether each of the 12 was LH or BO, but it does NOT call them optimization points.

---

## (b) Which ranges were EXTENDED relative to 2306.05471?

Confirmed extension of exactly TWO parameters (this paper, Sec 2.1, Table 2):
> "We have expanded the limits from Ref. [47] with 12 additional LF simulations covering the
> parameter ranges alpha_q = 2.5-3.0 and n_P = 1.0-1.05."
> "This was done so that the emulator covers the 2 sigma posterior range for n_P."

| Parameter | Original (2306.05471, Table 2) | Extended (2309.03943, Table 2) |
|-----------|-------------------------------|--------------------------------|
| n_P (scalar index @ k0=0.78 Mpc^-1) | 0.8 - 0.995 | **0.8 - 1.05** (extended up) |
| alpha_q (QSO spectral index, HeII) | 1.3 - 2.5 | **1.3 - 3.0** (extended up) |
| z_HeII_i (HeII reion start) | 3.5 - 4.1 | 3.5 - 4.1 (unchanged in paper text) |
| z_HeII_f (HeII reion end)   | 2.6 - 3.2 | 2.6 - 3.2 (unchanged in paper text) |

- **n_s claim CONFIRMED:** n_P upper limit extended from 0.995 to **1.05**, explicitly to reach the
  2σ posterior. The measured value n_P = 1.009 (+0.027/-0.018) sits *just above* the old 0.995 edge —
  i.e. the central value lives entirely inside the NEW extension band.
- **HeII claim PARTIALLY confirmed:** the HeII-related parameter that was extended is **alpha_q**
  (the quasar spectral index that drives HeII heating), to 3.0. The HeII *redshift* limits
  (z_HeII_i, z_HeII_f) were NOT extended per the 2024 text/Table 2.
  (NOTE: the repo's own box — see below — records herei_hi=4.5 and heref_lo=2.2, WIDER than the
   paper's Table-2 herei 4.1 / heref 2.6. So the *production grid box* used downstream is wider on
   the HeII-z axes than the paper's stated extension. Treat the JSON box as authoritative for our
   normalization; flag the discrepancy vs the paper.)

---

## (c) Is n_P in (1.0, 1.05) space-FILLED, or only a thin/corner extension?  [the crux for the bias]

This is the load-bearing point and the wording strongly favors the user's hypothesis.

The single sentence that opens n_P>1.0 is:
> "...12 additional LF simulations covering the parameter ranges **alpha_q = 2.5-3.0 AND n_P = 1.0-1.05**."

Interpretation: only **12** LF points cover the *joint* high-corner where alpha_q is high AND n_P is
high. This is a thin, coupled wedge bolted onto a 9-dimensional box, NOT a fresh space-filling
Latin hypercube that re-samples all 9 parameters across the full n_P in (1.0,1.05) slab. 12 points
in 9D is sparse even for a single slab; if they are also pinned to the high-alpha_q corner, then the
n_P in (1.0,1.05) region is effectively covered only along a narrow ridge and is NOT space-filled
across hub, omegamh2, A_p, bhfeedback, hireionz, z_HeII at high n_P.

Consequence: a GP/emulator queried at high n_P with *generic* (non-corner) values of the other
parameters is extrapolating off the training ridge -> reverts toward the GP mean (which is
calibrated by the dense n_P<=0.995 bulk) -> predictions are **pulled toward lower-n_P behaviour**.
That is exactly the "shrink-to-centre / high-n_s pulled down hardest" signature the user reports.

The paper does NOT show a corner/scatter plot proving the 12 points fill the slab, and does NOT
report leave-one-out accuracy *restricted to* n_P>1.0. So we cannot positively confirm uniform
coverage; the textual evidence points to a sparse coupled extension.

---

## (d) Emulator validation / accuracy near n_P>1.0 and box edges; do the authors caution?

- **Global accuracy:** "average interpolation error of 0.2% at low fidelity and 1% at high fidelity"
  (Sec 2.3) — but this is the suite-wide / LOO average, NOT a per-region or near-edge number.
- **General edge caution (Sec 3.5):**
  > "the GP expects the emulator error to be larger near the boundary of the space, which penalises
  > the fit in this region when the constraints from the data are weak."
- **Explicit 'too-few-sims-near-a-boundary' caution (Sec 3.5), made about the alpha_q LOWER edge:**
  > "...for which alpha_q is at the lower boundary of the emulator parameter range...with few
  > simulations in this region, there is not enough information present for the GP to learn this."
  This is the authors themselves articulating the failure mode the user suspects — just for the
  alpha_q lower edge. The SAME logic applies a fortiori to the n_P UPPER edge, which is covered by
  only the 12 coupled extension points.
- **No dedicated n_P>1.0 validation:** there is NO leave-one-out, no per-bin error, and no edge
  accuracy figure specific to the n_P in (1.0,1.05) extension. The authors added sims to *cover* the
  2σ range but did not separately *validate* the emulator there. So the extension is asserted-fit,
  not demonstrated-fit, at high n_P.

---

## Verdict on USER hypothesis ("n_s>1.0 not generalized")

**PARTIALLY CONFIRMS, leaning CONFIRM.**
- Confirmed: n_P range was extended to 1.05 (vs 0.995) using only **12** LF sims, added *jointly*
  with the high-alpha_q corner ("alpha_q=2.5-3.0 AND n_P=1.0-1.05") to reach the 2σ posterior.
- Confirmed: the central result n_P=1.009 sits inside the sparse new band, just above the old edge.
- Confirmed (by the authors' own words): the GP penalises/under-informs near sparse boundaries;
  they say this explicitly for the alpha_q low edge, and the same mechanism is expected at the
  n_P high edge.
- NOT confirmed (no positive evidence either way): no per-region LOO/accuracy restricted to n_P>1.0,
  and no proof the 12 points space-FILL the slab vs sit on a high-alpha_q ridge. The text reads as a
  thin coupled extension, which is consistent with — and a plausible cause of — a shrink-to-centre /
  high-n_s-pulled-down emulator bias.
- The user's specific phrasing "extra sims by Bayesian optimization not Latin hypercube" is aimed at
  the wrong batch: BO was used in the ORIGINAL 48 (and over-sampled boundaries), whereas the n_P>1.0
  opening 12 are described as space-filling. The *coverage-sparsity* core of the hypothesis stands;
  the *mechanism attribution* (BO) should be corrected.

## Cross-reference to repo
- /home/mfho/hcd_priya/docs/superpowers/2026-06-01-param-limits.md confirms the production grid box
  (emulator_params.json for kodiaq_2_2_4_6-48-48) uses ns 0.8-1.05, alphaq 1.3-3.0 — i.e. WE are
  using the 60-sim EXTENDED box. "All 60 PRIYA design sample_params map into [0,1]." Our real-data
  fit therefore queries n_s right up to 1.05, into the sparse extension. The repo box also widens
  herei to 4.5 and heref to 2.2 beyond the paper's Table-2 values (open discrepancy to flag).
- Relevant memory: reference-priya-sims.md (LF 48->note "60" reconcile), phase2-hr-phase1-and-bins.md.

## Open questions
- Were each of the 12 new LF points LH or BO? The 2024 paper doesn't say explicitly; need the
  released design/`sample_params` to check whether n_P>1.0 points fill the 9D slab or sit on the
  high-alpha_q ridge. (We HAVE the 60 sample_params locally — a direct coverage audit is possible.)
- Why does our production box (herei 4.5 / heref 2.2) exceed the paper's stated HeII-z limits?
- Is there any unpublished/appendix LOO error map for n_P>1.0?
