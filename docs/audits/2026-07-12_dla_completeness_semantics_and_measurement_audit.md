# DLA-completeness semantics and measurement audit (authoritative synthesis)

Date: 2026-07-12. Repo: /home/mfho/hcd_priya @ f89bea7 (branch gate-b-dla-desi-rerun).
Commissioned by the PI to separate the SIX distinct objects all currently called "DLA
completeness" and to adjudicate the core doubt: does the 10 percent residual assumed in the
alpha_dla prior (HCD_DLA_RESIDUAL_FRAC = 0.10) double-count the DLA-completeness systematic
already contained in the DESI DR1 data covariance?

Synthesis of six evidence roles (C0 definitions ledger, C1/C4 measurement archaeology, C2
emulator assumptions, C3 propagation model, C5 double-counting analysis, C6 statistical
questions) plus an adversarial referee who independently re-derived every load-bearing number.
The synthesis auditor additionally re-verified the two numeric referee adjudications (D1
latent offsets, D2 exact cov_syst recipe) from the shipped npz this session. Companion:
the settled implementation audit docs/audits/
2026-07-12_dla_alpha_prior_and_completeness_injection_audit.md (taken as RECORD, not redone).
Full evidence reports: scratchpad dla_semantics_audit/ (six role reports + referee.md +
scripts and npz artifacts); machine-readable summary: the .json next to this file.

Evidence tags: MEASURED (computed in this audit, referee-reproduced), RECORD (verbatim
code/paper/git/prior-audit), INFERRED, ASSUMED. Blind-safe throughout: templates, priors,
covariances, whitened projections, mock differentials only; no real-data posteriors. This
report FRAMES decisions; it recommends no prior change, no new nuisance, no ledger
finalization, no re-SBC, no eBOSS campaign (PI stop conditions honored).

---

## 1. Executive summary

1. The PI's doubt is semantically well-posed and the answer is nuanced: **the Gate-B FAIL
   magnitude is NOT an artifact of double counting, but a partial variance-side double-count
   does exist and a ledger-denomination double-report is possible if worded carelessly.**
2. Six distinct objects share the name "DLA completeness". They are NOT interchangeable. One
   physical unknown (the residual unmasked-DLA population, 1 - catalog completeness, times its
   P1D response) is re-represented six ways: DESI's catalog completeness estimate (~0.85), our
   mock-truth residual fraction (0.10), the alpha_dla prior (center 0.0044097), DESI's shipped
   envelope column (0.15 x r_DLA x P_smooth), the per-z rank-1 DLA mode inside cov_syst
   (inside the deployed C_data), and the Gate-B +-1 injection of that column. Section 2 is the
   full ledger.
3. MEAN side: NOT double-counted. DESI leaves the residual-DLA contamination IN the data
   ("We applied corrections only for continuum fitting biases", main.tex:459 RECORD) and
   budgets only a zero-mean variance. Our alpha_dla mean term is the only mean model, and it
   is largely shape-ineffective against DESI's stated contamination shape (5 percent whitened
   mismatch reduction; row-wise sign-opposed on 665/681 rows, MEASURED). It must not be
   described as "compensating DESI's unsubtracted bias".
4. VARIANCE side: PARTIAL double-count, conservative in direction. The alpha_dla prior and the
   cov_syst DLA mode both price the same named unknown; they overlap 10.4 percent
   directionally (cos_wh = +0.323), with the prior supplying 0.80x the cov term's variance
   along the envelope direction; each device separately adds ~4-6 percent of the n_s posterior
   variance. Tolerable, documentable, and conditional on an OPEN shape identity (the two
   response shapes may not share one parent error). Disposition: document, do not repair.
5. The +-1 z-coherent campaign is an envelope STRESS TEST, neither a 1-sigma realization nor a
   covariance worst case: exactly ~1 sigma per z block, jointly chi2 = 10.71 (~3.3 sigma)
   under DESI's own z-block model, and 74 percent of the DERIVED same-radius worst case (the
   maximizer is a sign-changing z profile peaked at z 3.4-3.6, now derived, satisfying the
   stop condition). Three honest denominations exist (Section 7); the choice among them is the
   PI's, entangled with the unknowable z-coherence length of DESI's catalog error.
6. The deployed likelihood contains an exact in-repo precedent for the "float + surgery"
   alternative: the resolution mode is removed from C_data (rank-1 per z) while b_res floats;
   the DLA mode is deliberately kept in C_data while alpha_dla floats. DESI's own cosmology
   paper (arXiv:2601.21432, verbatim RECORD) does the removal for BOTH resolution and HCD.
   Both our treatments are sanctioned by DR1 recommendation #3; ours is the more conservative;
   the asymmetry is now a ledger RECORD fact, not an error.
7. The structurally rigid axes of our compression are the real modeling exposure, not the
   0.10-vs-0.15 amplitude: the z-shape (a completeness z-gradient larger than ~3 percentage
   points across the band exceeds the s_dla prior at 1 sigma) and the k-shape (80.3 percent of
   the whitened DESI envelope is orthogonal to ANY z-reparametrization of alpha_dla). The
   amplitude question is sub-1-sigma (0.15 sits at +0.41 latent sigma of the deployed prior).

Confidence: HIGH on every MEASURED number (all referee-reproduced from independent
implementations; the two adjudicated disagreements re-verified a third time by this
synthesis); HIGH on paper quotes (verbatim hits at cited main.tex lines); MEDIUM on
INFERRED items (pro-rata direction, cross-z-zero as convention); the five genuinely
unknowable items are listed in Section 7.4 and the JSON.

---

## 2. The completeness-definition LEDGER

### 2.1 The six PI objects, resolved

| PI's object | Ledger rows | One-line disambiguation |
|---|---|---|
| 1. finder/catalog completeness probability | O1-O4 (DESI side), O21-O23 (sim side) | THREE unrelated finders: DESI Wang+2022 CNN/GP concordance (~0.85), our assumed DESI-effective (0.90 implied by O5), our cache-build tau-peak finder (0.971, sim-internal QA only) |
| 2. residual DLA fraction in mock/emulator construction | O6 (TRUTH_DLA_FRAC) + O20 | truth side: 10 percent of the full sim DLA excess added to the DESI closure target; 0 percent KS; the EMULATOR itself contains NO completeness object (Section 4) |
| 3. alpha_dla prior on post-masking incidence | O5 -> O11 -> O12 | forward side: marginalized sightline WEIGHT centered at 0.10 x (lit/sim) x w_DLA = 0.0044097 |
| 4. DESI syst_e_dla_completeness envelope | O10 | 0.15 x (fitted no-masking fractional excess r_DLA) x P_smooth; unsigned, per (z,k) |
| 5. DLA component of cov_syst | O14 (inside O15) | per-z rank-1 outer(e_dla, e_dla), exactly z-block-diagonal, inside the deployed C_data |
| 6. Gate-B additive injection | O13 | +-1 x O10 mapped exactly onto the 681 leg rows, z-coherent by hand (joint chi2 10.71 under O14) |

### 2.2 Full object table (C0, referee-corrected)

Class order (clean, LLS, subDLA, DLA); k angular s/km; P1D km/s.

| # | Object | Code/symbol | Definition | Fiducial | Source (file:line or paper) | Tag |
|---|---|---|---|---|---|---|
| O1 | True DESI catalog completeness | C_true | P(catalog flags a DLA given a real DLA on a DESI sightline) | unknown latent | concept; DR1 Sec 4.4.1 | RECORD |
| O2 | Estimated DESI completeness | C_hat | Wang+2022 CNN validation on synthetic DESI mock spectra; DR1: "completeness and purity ... around 85%" | ~0.85, no quoted sigma | main.tex:206,219,340 | RECORD |
| O3 | DESI missed-DLA probability | 1 - C_hat | "estimated 15% incompleteness ratio in our baseline DLA catalog" | 0.15 | main.tex:235 | RECORD |
| O4 | DESI catalog purity | ~0.85 | false positives ~15 percent; high-confidence rerun "entirely consistent" so NOT budgeted; no impurity envelope exists | ~0.85 | main.tex:221,227 | RECORD |
| O5 | OUR residual DLA fraction | HCD_DLA_RESIDUAL_FRAC | scalar multiplying (lit/sim)-corrected w_DLA in the prior center; "finder misses ~10%, PI-confirmed 2026-06-09" | 0.10 (ASSUMED, PI design) | inference.py:70, :225, :246 | RECORD |
| O6 | Closure mock-truth fraction | TRUTH_DLA_FRAC | fraction of full sim DLA excess in the closure target | {DESI 0.10, KS 0.0} | closure_legb.py:181, :1323 | RECORD |
| O7 | Sim DLA sightline weight | w_DLA(z) | telescoping-Poisson 1 - exp(-dN/dX x Xbar) | 0.03291 at z=3 | dndx_wc.py:12-19; closure_legb.py:141 | RECORD |
| O8 | lit/sim incidence ratio | HCD_LIT_OVER_SIM[2] | 1.34 x ((1+z)/4)^0.40 | 1.34 at z=3 | inference.py:83,93,208-214 | RECORD |
| O9 | Per-leg forward gate | dla_forward_frac | BINARY structural gate, NOT a completeness fraction | DESI 1.0, KS 0.0, eBOSS 0.0 | data_likelihood.py:158-166,199-203 | RECORD |
| O10 | DESI envelope column | syst_e_dla_completeness | 0.15 x r_DLA(k,z) x P_smooth; r_DLA = Eq 4.1 fit (gamma -0.443, a 1.82, b 690, c 7.49e-3) of the (ratio - 1)-shaped no-masking excess; reproduces the shipped column to median ratio 0.9999 | median e/P 0.118 percent; up to 18.5 percent at lowest k | paper Eq 4.1 + MEASURED | MEASURED |
| O11 | alpha_dla prior center | alpha_hcd_mu[2] | 0.10 x 1.34 x 0.03291 | 0.0044097 | inference.py:246; closure_legb.py:695-711 | RECORD |
| O12 | alpha_dla (sampled) | softplus(raw), raw ~ N(-5.4218, 1.0) | z-resolved forward alpha(z) = pivot x ((1+z)/4)^s_dla, s_dla ~ N(2.366, 0.33); prior median 0.00441, mean 0.00722; named HCD_PRIOR_FRAC_SIGMA[2] = 0.50 NOT deployed | eff sigma/mu ~1.28 | closure_legb.py:1993-2000, :86, :118 | RECORD |
| O13 | Gate-B injection | e_comp, --strength | P_data += strength x e_dla, exact 681/681 row map, z-coherent by hand | +-1 of O10 | run_dla_completeness_shard.py:24-57 | RECORD |
| O14 | DLA mode of cov_syst | (inside COVARIANCE_SYST) | per z block EXACTLY outer(e_dla, e_dla); cov_syst = 5 outer products + diag(e_continuum_add^2 + e_noise_add^2), exact to 6.2e-18; cross-z EXACTLY 0 | rank-1 per block | MEASURED (this synthesis re-verified) | MEASURED |
| O15 | Deployed DESI C_data | leg.C_data | cov(= stat + syst exactly) + diag inflation, MINUS per-z rank-1 resolution surgery (sample_res=True deployed); DLA mode KEPT | 681 kept rows | data_likelihood.py:242-301, :511-515; closure_legb.py:508,638 | MEASURED |
| O16 | Forward DLA response | R_DLA, t_dla | (P_filt[3] + dla_core) - P_clean; negative on ~96 percent of DESI rows | cos_wh(e, t_dla) = +0.323 | predict.py:82-86 | RECORD |
| O17-O20 | P_filt[3], P_DLA^unf, dla_core, dla_excess_true | cache/template internals | see C0 report | | priya_p1d.py, data.py:164, closure_legb.py:1093 | RECORD |
| O21 | Sim-side finder completeness | dla_truth summary | tau-peak finder vs particle colden truth on HiRes sims | 0.971 | dla_truth.py:343,387; docs/dla_truth_validation.md:186 | RECORD |
| O22 | Sim-side finder purity | | | 0.947 | dla_truth.py:346,388 | RECORD |
| O23 | Sim masking "completeness" | (does not exist) | tau > 1e6 deterministic detection; no probabilistic completeness anywhere in masking.py | ~100 percent by construction | masking.py:415-457 | MEASURED |
| O24 | BAL completeness envelope | syst_e_bal_completeness | same Eq-4.1 family, different contaminant, ~10x smaller | | paper Sec 4.4.2 | RECORD |

Referee corrections adopted: the 0.15-implied center sits at +0.41 latent sigma (NOT +0.58;
re-verified: +0.4066); C0's "unexplained z=2.2 block structure" gap is CLOSED (it is exactly
the two diagonal-only add-terms, not structure; re-verified 6.2e-18).

### 2.3 Pairwise relation classification

MI = mathematically identical; AM = approximately mapped; CE = calibrated empirically;
CR = conceptually related only; UN = unrelated despite naming; UK = unknown.

| # | Pair | Relation | Key fact |
|---|---|---|---|
| P1 | 0.10 (ours) vs 0.15 (DESI) | AM | same intended object, numerically inconsistent 1.5x; 0.15-implied center at +0.41 latent sigma, inside the broad prior; 0.10 must never be presented as "the DESI number" |
| P2 | HCD_DLA_RESIDUAL_FRAC vs TRUTH_DLA_FRAC["DESI"] | MI by design | closure self-consistency; distinct roles and code sites, must be mirrored on change |
| P3 | alpha_hcd_mu[2] vs 0.10 | MI mapping | deterministic formula; different objects (fraction vs absolute weight) |
| P4 | alpha_dla vs 1 - C | AM | exact map is 1 - exp(-f mu); deployed uses f x w: 1.5 percent understatement, negligible vs 128 percent width |
| P5 | envelope vs 0.10 x R_DLA | AM generatively | same parent (missed fraction x DLA response) but different fraction (0.15 vs 0.10) AND different response SHAPE: DESI r_DLA positive-definite, PRIYA R_DLA bipolar; cos_wh = +0.323 (unwhitened cos = -0.048); shape identity is ASSERTED, not measured |
| P6 | cov_syst DLA mode vs envelope | MI content | exactly outer(e_dla) per block, nothing more |
| P7 | injection vs envelope | MI per row | byte-equal 681/681; z-coherence convention is the only addition (joint chi2 10.71 vs ~1 per block) |
| P8 | injection vs cov worst case | AM | one hand-signed realization; the derived maximizer differs (Section 7) |
| P9 | DESI C_hat (~0.85) vs sim finder 0.971 | UN | different finder, data, purpose; 0.971 must never appear in a DESI budget row |
| P10 | sim finder complement (0.029) vs 0.10 | UN | no code path connects them (grep-verified) |
| P11 | sim tau>1e6 mask vs DESI catalog masking | CR | sim mask deterministic; the only sim analogue of incompleteness is the deliberate TRUTH_DLA_FRAC add-back |
| P12 | dla_forward_frac vs any completeness fraction | UN | binary structural gate |
| P13 | dla_core vs R_DLA | MI component | omitting the core manufactured the retired anti-absorption artifact (RECORD H8) |
| P14 | dla_excess_true vs alpha_dla x R_DLA | MI at truth | gate-exact; AM on the real fit (z-mean core MVP) |
| P15 | DLA vs BAL completeness columns | UN for DLA | same construction family only |
| P16 | named 0.50 width vs deployed width | UN in deployment | deployed latent sigma 1.0; any text quoting 0.50 is wrong ~2.5x |
| P17 | injection vs floated e_DLA nuisance | UK | design candidate only; no code object at f89bea7 |

---

## 3. How completeness was measured (C1) and what syst_e_dla_completeness is (C4)

All quotes verbatim from the downloaded LaTeX source of arXiv:2505.07974 (Karacayli et al.
2025, DESI DR1 Lya P1D optimal estimator); every cited line referee-verified at 100 percent
hit rate.

**How the 15 percent was measured.** CNN finder performance validated on synthetic DESI mock
spectra (Wang, Zou, Cai, Prochaska et al. 2022, ApJS 259), summarized by DR1 as "our DLA
catalog's completeness and purity are around 85% [Wang+2022]" (main.tex:206), "depending on
the confidence level and SNR cuts". No z/NHI/SNR-resolved curve is propagated anywhere in the
DR1 systematic, although Wang+2022 measured SNR-resolved performance. NO uncertainty on the
15 percent is stated. The high-confidence catalog re-run was a purity CONSISTENCY CHECK
("entirely consistent with our baseline", main.tex:221), not the error estimator.

**What the column is (MEASURED).** The shipped E_DLA_COMPLETENESS column is EXACTLY

    e_DLA(z,k) = 0.15 x [ c + ((1+z)/3)^gamma / (a exp(b k) - 1)^2 ] x P_smooth(z,k)

with gamma = -0.443, a = 1.82, b = 690 km/s, c = 7.49e-3 (median ratio to shipped 0.9999;
alternative readings fail by 64 percent and factor ~130). The bracket is the fitted
FRACTIONAL EXCESS (ratio minus 1 shaped) of a two-run differential on REAL DR1 data: the
whole pipeline with no DLA masking vs the baseline masked run ("The systematic error budget
is based on a fit to the no-masking case", main.tex:227). So the column is:

    (assumed scalar residual fraction 0.15, from finder validation on mocks)
    x (smooth fit to the MEASURED total unmasked-DLA relative excess, real data)
    x (P_smooth).

It is a phenomenological template x assumed residual fraction. It equals a first-order
|dP/dC| x DeltaC ONLY under the PRO-RATA assumption that the missed 15 percent of DLAs
contribute 15 percent of the total unmasked-DLA excess power. INFERRED, one-directional:
missed DLAs are preferentially low-SNR (downweighted by the QMLE, the paper's own argument
for impurities at main.tex:221) and low-NHI (weaker wings), so true residual contamination is
plausibly below pro-rata. It is NOT a signed derivative, NOT a two-catalog difference, NOT a
covariance sqrt, NOT a max excursion.

**Signs.** The parent effect is one-signed positive (damping wings add low-k power,
main.tex:219); the data vector carries an expected positive contamination bias that DESI
deliberately does NOT subtract ("We applied corrections only for continuum fitting biases",
main.tex:459). The sign is discarded when the column enters cov_syst as an outer product.

**cov_syst assembly (MEASURED, exact).** Per z block: sum of 5 outer products (DLA, BAL,
resolution, continuum, noise_scale) + diag(e_continuum_add^2 + e_noise_add^2), exact to
6.2e-18; cross-z EXACTLY zero; cov = cov_stat + cov_syst exactly. Each systematic is one
fully k-correlated latent per z block; e_dla is exactly a 1.000-sigma direction of its own
block at every z. The cross-z zero is CONVENTION (INFERRED; the paper is silent; cov_stat by
contrast does carry small cross-z structure, proving the zero is syst-construction-specific).
Physically one catalog with one incompleteness is as z-coherent as a systematic gets.

**Intended semantics (RECORD, main.tex:450,461).** Primary: covariance-level zero-mean
marginalization "without extra parameters". Alternative: remove a mode from C and float it
with a prior (spelled out for resolution; the per-mode columns shipped precisely to enable
this). GoF inflation is a separate shipped diagonal. Recommendation #3 sanctions additional
HCD templates on top of the full covariance "for conservative analyses".

**Producer asymmetry (RECORD, upgraded to HIGH by the referee's own fetch of arXiv
2601.21432v2).** The DESI DR1 COSMOLOGY paper: "we omit the contributions from residual HCD
contamination and uncertainties in the spectrograph resolution to the systematic covariance
matrix, as both effects are explicitly marginalized over in our analysis". The data producer's
own cosmology fit therefore implements remove-from-cov + float for BOTH resolution and HCD;
our deployed DESI leg does it for resolution only and keeps the DLA mode in C while floating
alpha_dla. Both treatments are sanctioned by DR1 recommendation #3; ours is the more
conservative; the overlap cost is the measured 10 percent of Section 6.

**Our sim side (C4).** dla_truth.py validates the CACHE-BUILD tau-peak finder against SPH
particle colden truth: completeness 0.971, purity 0.947 (RECORD). The NUMBERS feed nothing in
the deployed inference (grep-verified: diagnostics and tests only). The FINDER itself is
load-bearing indirectly: its catalog assigns the per-sightline class labels that define the
class templates (hence R_DLA and dla_core) and the structural weights w_c behind the prior
center. So dla_truth underwrites the response SHAPE and the prior-center DENOMINATOR, while
DESI-side completeness governs the residual-population SIZE. Distinct; never conflate (P9/P10).
Minor doc flag: docs/dla_truth_validation.md Sec 6 still quotes the stale pre-update 0.86
match fraction alongside the current 0.971.

---

## 4. Emulator and prior assumptions (C2): what 0.10 means and what the emulator contains

**Meaning.** 0.10 is the assumed fraction of DLA-class sightline incidence that survives
DESI's masking, i.e. 1 minus an assumed effective finder completeness of ~0.90, applied to
the observed (lit-corrected) DLA class weight. Semantics: "10 percent of true-DLA sightlines
remain unmasked in the DESI forest sample". It is NOT a per-system catalog probability, NOT a
CDDF renormalization; class-conditioned on DLA only (LLS/subDLA modeled fully unmasked).
Missed systems are modeled as FULL average DLA systems coupling to the unfiltered
class-average template.

**Provenance (git archaeology, RECORD).** 0.30 at introduction (86eda00, "masking ~70%
complete") -> 0.10 (1e1b278, PI-confirmed 2026-06-09, "DLA-finder ~90% complete -> 10%
residual full systems"); an intermediate "~0 / 0.05" reading exists only in a retrospective
comment, never deployed. All three candidate numbers trace to the 2026-06-01 literature
review: 0.30 = 1 - (DESI cosmology paper's ~70 percent at SNR > 2); 0.10 = 1 - (the "> 90
percent complete at logNHI > 20.3, good SNR" finder-validation figure); DESI's own P1D budget
uses 0.15. The value was never fit, never derived from a measured DESI completeness curve:
it is a literature-informed PI design assumption.

**Amplitude coverage (MEASURED).** In residual-fraction units the deployed prior places 0.05
at -0.69 latent sigma, 0.10 at 0, 0.15 at +0.41, 0.30 at +1.10; the +1 sigma quantile is
fraction 0.271. The deliberately broad width (deployed latent sigma 1.0, NOT the named 0.50)
covers the entire documented completeness range in amplitude. Center choice is a sub-1-sigma
question, not a coverage hole.

**What is rigid.** The 0.10 is constant in z, NHI, SNR, and mask width. The z evolution is
carried entirely by the sim incidence slope s_dla ~ N(2.366, 0.33); the lit/sim ratio slope
enters only the z=3 pivot. Illustrative rigidity (ASSUMED scenario): a completeness decline
0.90 -> 0.70 across z 2.2-4.2 would require Delta s = +2.26 = 6.9 prior sigma. The slope
window +-0.33 corresponds to only ~+-2.6 percentage points of completeness drift across the
band (MEASURED, C3). NHI dependence: none; the residual couples to the class-average
template although real finder misses skew low-NHI/low-SNR (weaker wings), a shape error with
no dial, consistent with the measured 58 percent stranded fraction of the envelope.

**The emulator itself contains NO completeness object (MEASURED).** Cache class assignment is
sim TRUTH (highest-logNHI absorber, priya_p1d.py:110-122); no finder runs in cache
construction; populations are class-pure and 100 percent complete by construction. The tau=1e6
trough fill is a masking-procedure analog, not a completeness model. Completeness enters at
exactly three downstream sites: the alpha_dla prior center (inference.py:246), the closure
truth (TRUTH_DLA_FRAC), and the per-leg binary gate (data_likelihood.py:195-203).

**Mock families differ on exactly the PI-relevant point.** Sim-truth TARGET mocks carry the
residual at fixed 0.10 with no uncertainty; Leg-A prior SELF-DRAW mocks (the Gate-B DLA
campaign and production SBC) draw alpha_dla AND s_dla from the deployed prior while the mock
noise comes from C_total built on the full DESI covariance. Self-draw closure therefore
carries BOTH accountings simultaneously by construction; the adjudication of that coexistence
is Section 6.

---

## 5. The generative mapping: completeness -> alpha_dla (C3)

**The chain the code compresses.**

    dN/dX_resid(z) = L(z) x INT_{20.3}^{inf} f_sim(NHI, z) [1 - Cbar(NHI, z)] dNHI
    alpha_dla(z)   = 1 - exp(-Xbar(z) x dN/dX_resid(z))

with L(z) = lit/sim = 1.34 x ((1+z)/4)^0.40. The deployed code compresses this at the z=3
pivot with a flat NHI-independent residual fraction: mu = 0.10 x 1.34 x w_DLA(3) = 0.0044097.
Compression validity at the center (MEASURED): the full integral with flat C = 0.90 gives
0.00457 vs deployed 0.00441 (3.6 percent gap: CDDF-vs-class-count 2.1 percent + Poisson
curvature 1.6 percent). The AMPLITUDE compression is exact to a few percent.
d alpha / d Cbar = -0.046 at the pivot: 1 percentage point of completeness = 10.4 percent of
the deployed center.

**Implied vs deployed prior, per attribute (MEASURED).**
- CENTER: DESI's own 15 percent implies 0.0066-0.0069, at +0.41 latent sigma (the softplus
  right-skew puts the prior MEAN 0.00722 within 5-9 percent of it). Mildly miscentered
  against DESI's own number, comfortably in-support; second-order for cosmology (weak lever,
  cos^2 = 10 percent, contraction ~0).
- WIDTH: latent sigma 1.0 maps to a 1-sigma completeness band C in [0.73, 0.96], covering the
  full literature bracket including the ~0.71 DR1-catalog evaluation. The named-but-undeployed
  0.50 would NOT reach the bracket low end; the hardcoded 1.0 is the generatively defensible
  choice. Width valid, generous.
- Z-SLOPE: the weakest attribute. Implied slope +2.9 (flat C, sim z-shape) to +3.6 (lit-law
  z-shape) vs deployed N(2.366, 0.33), i.e. +1.6 to +3.8 sigma; completeness z-gradient
  scenarios reach -5.3 to +7.9 sigma. Any real z-gradient above ~3 percentage points across
  the band leaves the representable family.
- ONE-SIDED SUPPORT: appropriate for this object (residual incidence >= 0). What it cannot
  represent is the PURITY limb: false positives (~15-28 percent) mask clean forest, an
  opposite-sign estimator distortion that neither alpha_dla >= 0 nor DESI's
  incompleteness-only envelope covers. Uncovered on BOTH sides of the ledger; unsized.

**What scalar alpha_dla structurally cannot represent (capture ladder, MEASURED, full
C_total whitening, referee-reproduced).**

| DLA-sector freedom | share of whitened envelope captured |
|---|---|
| L1 deployed pivot (1 dof) | 10.4 percent |
| L2 pivot + z-slope (2 dof) | 19.1 percent (optimum at ds ~ -5 to -6: double-digit prior sigma, prior-forbidden) |
| L3 oracle per-z alpha(z) (11 dof) | 19.7 percent |
| k-shape floor (orthogonal to ANY z-reparam) | 80.3 percent |

The empirical falling, sign-flipping per-z coefficient profile is EXPLAINED by the generative
model as the ratio of two mismatched z-scalings (DESI's fractional excess shape falls with z
while R_DLA deepens and alpha(z) rises). Inverted into completeness units it "wants"
(1 - C) = 0.40 at z 2.2 falling to 0.06 at z 4.2; direction-plausible (blue-end SNR) but
mostly an artifact of forcing DESI's r_DLA(k) through PRIYA's R_DLA(k) lever; where the block
cosines decohere (z >= 3, |cos| < 0.3) the "implied completeness" reading stops being
meaningful. The NHI-reweighting result says the true residual response is plausibly a THIRD,
weak-wing shape (residual mean logNHI 20.56 vs class-average 20.82 under a CNN-like ramp).

---

## 6. Double-counting adjudication (C5 + referee): the PI's core question

### 6.0 A load-bearing deployed-covariance fact (MEASURED, new in this audit)

The deployed DESI C_data is NOT the shipped (cov + diag inflation): because the production
forward has sample_res=True, the option-b rank-1 resolution surgery runs at leg build
(data_likelihood.py:511-515): C_data = (cov + diag inflation) - sum_z outer(e_resolution|z).
Verified by reconstruction (PSD residual; the no-surgery hypothesis goes negative to -0.67
relative; the SNR3 hypothesis directly falsified). **The deployed likelihood therefore
already contains an exact in-repo precedent for "float the nuisance AND remove its quadrature
term from cov", and deliberately does NOT do this for DLA completeness.** That asymmetry is
the precise code-level form of the PI's doubt.

### 6.1 The graph

    p_det (DESI catalog completeness ~0.85)
      |  1 - p_det = residual unmasked-DLA population (the ONE physical unknown)
      |-> DESI pipeline: residual damping wings stay IN P_data (no subtraction)
      |      |-> O10 envelope 0.15 x r_DLA x P_smooth  (their effect estimate)
      |      |-> O14 per-z rank-1 outer(e_dla) in cov_syst, inside deployed C_data
      |            [z-INDEPENDENT unit latent per block, by shipping convention]
      |-> our forward: alpha_dla(z) x R_DLA  (models the same residual effect)
             |-> O12 prior: softplus(N(-5.42, 1)), center 0.10 x 1.34 x w_DLA;
             |            slope N(2.366, 0.33)  [z-COHERENT power law]
             |-> O6 truth draw: same distribution (self-draw mocks)
    Gate-B: O13 injects s x e_dla (s = +-1, z-coherent) into the mock mean
      -> deployed fit (C keeps O14; alpha_dla floats): 42 percent of the whitened mode
         absorbed, 58 percent stranded -> tau0_amp sink -> n_s/A_p relocation (RECORD)

Caveat carried on every "same physical unknown" sentence (referee sharpening 1): the identity
of the two response shapes is ASSERTED at the generative level, not measured. The shapes are
sign-opposed on 665/681 rows, unwhitened cosine -0.048, and the plausible true residual
response is a third weak-wing shape. The 10 percent directional overlap is a statement IN THE
DEPLOYED METRIC between two shapes whose common parent is assumed.

### 6.2 The five candidate overlaps, adjudicated with numbers

**(1) alpha_dla prior floated WHILE cov_syst contains the envelope: the core doubt.
VERDICT: PARTIAL DOUBLE-COUNT on the variance side only; deliberate hierarchy in intent;
conservative in direction; document, do not repair.** All MEASURED (2e5-draw MC through the
exact linear-in-alpha forward, deployed C_total metric; referee-reproduced with an
independent MC):
- Whitened-trace totals: cov DLA term 2.730 vs prior-induced term 2.434. Same size overall.
- Along the ENVELOPE direction: cov contributes 0.318, the prior adds another 0.253-0.256
  on top (0.80x): along the shared component the same named uncertainty is genuinely priced
  twice.
- Along the PRIOR's own response direction: prior 2.43 vs cov 0.049 (~50x): each device
  overwhelmingly prices a direction the other barely covers (mutual overlap cos^2 = 10.4
  percent).
- Posterior-variance shares (linear GLS, deployed priors): the alpha_dla float adds 4.3
  percent of the n_s posterior variance; the cov DLA term adds 5.9-6.1 percent. Each device
  is a ~5 percent variance-level widening.
- Both are variance devices: the duplication CANNOT create the Gate-B mean bias; it only
  over-widens, i.e. errs conservative.

**(2) The MEAN side. VERDICT: NOT double-counted, and our mean term is largely ineffective
rather than compensating.** DESI applies no subtraction (main.tex:459); a mean model exists
only on our side. Referee sharpening 2 (MEASURED): the deployed mean term's whitened size is
0.746 (43 percent of the envelope's 1.723) but, because of the 10 percent shape overlap, it
reduces the whitened mismatch to a pro-rata contamination by only 5 percent (1.723 -> 1.641)
and moves the prediction in the OPPOSITE direction to the expected contamination on 665/681
rows. Honest ledger sentence: the mean is modeled ONCE (ours) and, if DESI's r_DLA shape is
the truth, mostly INEFFECTIVELY; it is not a meaningful subtraction of DESI's residual bias.
(The paired gate design cancels the fiducial mean term exactly; no gate number changes.)

**(3) Gate-B injecting a full envelope on top of a fit whose C already contains it. VERDICT:
mechanically clean, NOT a double-count; the FAIL magnitude is denomination, not dosing.**
The in-C DLA term discounts the injection's whitened chi2 by ~33 percent (|e|_wh 1.72 with
vs 2.10 without) yet damps the fitted GLS n_s bias by only ~7-10 percent (b_ns -0.02808 vs
-0.03021): covariance prices variance, not the mean. The ll identity diff(+1) + diff(-1) =
-e^T C^-1 e confirms exactly one dose. The paired estimator using the same C in both arms is
correct accounting.

**(4) Marginalized-in-likelihood AND budgeted-in-ledger. VERDICT: PARTIAL DOUBLE-COUNT in
ledger DENOMINATION only; wording, not code.** The quoted sigma-denominated ratios are ~3
percent conservative (the denominator contains the tested systematic's variance). But under
DESI's own z-block model the covariance already prices a 1-sigma-typical completeness-draw
n_s shift of 0.229 sd_post = 40 percent of the coherent-arm linear bias (1/2.47). Booking the
full coherent-arm number as a new budget row without that note re-reports that share.
Denomination note only; it must NOT be read as "40 percent of the bias is removed by the
covariance" (only ~10 percent damping of the mean).

**(5) The 0.10 population in truth vs the injection. VERDICT: legitimate hierarchy; the
paired self-draw design neutralizes it by construction.** The residual population is present
identically in both arms (prior self-draw truth, bit-identical clean truths); the paired
delta responds to e_dla alone. Magnitude context (MEASURED): the envelope is comparable to
the response of the ENTIRE modeled 0.10 residual population (|a_fid t_dla|/e median 1.60).

### 6.3 Bottom line for the PI

The Gate-B FAIL is not an artifact of double counting. The PI's instinct is half right in two
specific places: (i) variance side, two live pricings of the same named unknown, overlapping
10 percent, prior supplying 0.80x the cov term along the envelope direction, each ~5 percent
of n_s posterior variance: partial, conservative, tolerable, conditional on the open shape
identity; (ii) ledger side, 40 percent of the coherent-arm linear n_s bias equals the shift
the covariance already prices as 1-sigma-typical: a wording hazard when booking the row.
Whether to REMOVE the overlap (surgery + float) is a physics/design decision (a z-coherence
assertion), expressly not adjudicated here.

---

## 7. The three statistical questions (C6): what the campaign measured

Structural fact settling the covariance reading (MEASURED): DESI's shipped cov_syst encodes
each systematic as ONE fully k-correlated latent per z block (rank-1 per block, cross-z
exactly zero). Reading R1 = DESI's stated model; R2 = diag(e^2) (a bracket, NOT DESI's
model); R0 = one global z-coherent latent (the campaign-implicit reading). DLA dominates
cov_syst at z >= 3 (>= 64 percent of block trace).

**Question A, expected RMS shift (zero mean by construction; spread not bias), sigma_clean
units, linear GLS (campaign-calibrated in parentheses; calibration factors ns 0.63 / Ap 0.82
/ tau0 0.98 measured at the +-e direction, ASSUMED transferable):**

| reading | ns | Ap | tau0_amp |
|---|---|---|---|
| R1 per-z-block (DESI's model) | 0.157 (~0.10) | 0.116 (~0.10) | 0.118 (~0.12) |
| R0 fully z-coherent | 0.388 (~0.24) | 0.291 (~0.24) | 0.334 (~0.33) |
| exp(-|dz|/l) family, l = 0.4 | 0.273 | 0.201 | 0.212 |

The decision-relevant lever is the z-correlation length of the catalog error, which DESI's
bookkeeping does not settle; the family spans the non-exhaustive middle ground (stop-rule
compliance).

**Question B, covariance-constrained worst case, DERIVED (satisfying the PI's stop
condition).** For a single linear functional the chi2 = 1 constrained max equals the Question
A RMS (identity). The n_s-maximizing latent profile under R1 is SIGN-CHANGING (positive
z <= 2.4, negative above, peak at z 3.4-3.6), Mahalanobis cosine to the shipped all-positive
envelope -0.744. At the coherent arm's own radius (chi2 = 10.71) the derived worst case is
0.521 sigma_clean (linear) vs the arm's 0.388: **the coherent arm reaches 74 percent (Ap 76,
tau0 85) of the derived same-radius maximum.** The joint (ns, Ap) bias operator is
effectively rank-1 (singular values 0.194 vs 0.025): one cosmology mode.

**Question C, the campaign as-run.** A hand-signed, z-coherent, all-positive stress test:
exactly ~1 sigma per z block, jointly chi2 = 10.71 (~3.27 sigma) under DESI's stated model,
|e|_wh = 1.723 inside the deployed likelihood, 74-85 percent worst-case-aligned. Measured
pulls: ns -0.210 +- 0.066 / +0.280 +- 0.081 sigma_clean (Ap -0.240/+0.237, tau0 +0.388/
-0.270). (Referee note: an independent recompute gives -0.224/+0.297 under a different
pooling convention; immaterial.) Radius-rescaled to chi2 = 1 and efficiency-corrected, the
arm implies ~0.085-0.113 sigma per 1 sigma of DESI's stated block model, converging with the
A/B routes on ~0.10 sigma_post.

**The campaign answered Question C ONLY.** It overstates the 1-sigma RMS of DESI's stated
model by ~2.5x and understates the same-radius worst case by ~26 percent. Honest candidate
denominations for the ledger row (per-parameter, sigma_clean):
- typical (R1, 1 sigma of DESI's stated model): ns ~0.10 calibrated;
- as-run coherent stress test: ns 0.21-0.28 at joint ~3.3 sigma, 74 percent efficient;
- derived same-radius worst case: 1.36x the arm;
- fully z-coherent 1-sigma reading (R0, a physics choice): ns ~0.24 calibrated.
Presenting only "3.3-sigma worst case vs renormalized 1 sigma" is the forbidden false
dichotomy; the correlated-latent family and the derived maximizer span the middle.

**Denominator reconciliation (referee D6, adopted):** the settled audit's "GLS overstates by
factor 2-2.7" mixes denominators (Fisher sd 0.0496 vs sigma_clean 0.0724); in one denominator
the true GLS-to-NUTS gap at +-e is ~1.6x for n_s (= 1/0.63). Use the explicit calibration
factors, drop the bare "2-2.7".

**Genuinely unknowable from available sources (referee Section 8):** (1) the z/NHI/SNR-resolved
DESI completeness curve for the P1D forest sample (decides 0.10-vs-0.15 centering, the
z-slope risk, and pro-rata; needs Wang+2022 figures / Brodzeller+2025 / DESI-internal);
(2) the z-correlation length of the catalog error (spans ns RMS 0.157-0.388; needs a DESI
statement or arXiv:2509.13593); (3) which excess shape is faithful for the residual
population (r_DLA vs R_DLA vs a third weak-wing shape; a mask-85-percent sim experiment would
decide, out of compute scope); (4) whether DESI's 15 percent is a point estimate or itself a
1-sigma; (5) the purity/over-masking limb, sized by nobody; (6) the eBOSS analog (no lever,
completeness only in eBOSS C_data; inherited open).

---

## 8. Mitigation-model reassessment (C7): what each candidate actually models

No option below is recommended or built here (stop conditions); this section states exactly
which uncertainty each would represent, so the PI's decision is about the right object.

1. **Existing alpha_dla (deployed).** Models: the MEAN residual-DLA incidence (amplitude,
   correct to a few percent as a compression) plus its AMPLITUDE uncertainty (broad width
   covering completeness 0.73-0.96 at 1 sigma), through the PRIYA in-situ R_DLA shape with a
   rigid sim-anchored z power law. Does NOT model: completeness z-gradients beyond ~3
   percentage points, the NHI-mix shape change of the residual population, the k-shape of
   DESI's envelope (80.3 percent orthogonal), or purity. Its variance overlaps the in-C DLA
   mode 10 percent (the documented conservatism).
2. **Widened or recentered prior (e.g. center 0.15).** Models: the same object with a
   different amplitude center/width. Buys: removal of the 1.5x center inconsistency with
   DESI's own number (+0.41 latent sigma, already in-support). Does not touch z-shape,
   k-shape, or NHI-mix. A wording-level fix at most; a PI option, not a defect repair.
3. **z-resolved alpha_dla (free per-z or looser slope).** Models: arbitrary z profiles of the
   residual incidence. Fixes ONLY the z mismatch, NOT the k mismatch: the oracle per-z ladder
   captures 19.7 percent of the whitened envelope vs 10.4 deployed; the remaining 80.3
   percent is a k-shape floor unreachable by ANY z-reparametrization of an R_DLA-shaped
   lever. Also enlarges the HCD-cosmology degeneracy surface.
4. **Floated e_DLA amplitude (envelope-shaped nuisance).** Models: an error with EXACTLY
   DESI's stated shape. The only candidate that spans the 80 percent floor. REQUIRED cov_syst
   modification to avoid a genuine double-count: C_data <- C_data - sum_z outer(e_dla|z)
   (the exact mirror of the deployed option-b resolution surgery, data_likelihood.py:511-515).
   SPD-feasibility MEASURED on the actual matrices: min eig 0.048 after surgery (0.0071 in
   the worst no-inflation variant; the z = 4.0/4.2 blocks become exactly singular because the
   DLA mode saturates them there: maximal but valid). Two variants: eleven per-z amplitudes
   s_z ~ N(0,1) = mathematically identical to the removed cov term (pure re-representation,
   useful only for posterior inspection); ONE z-coherent amplitude = a STRICTLY more
   informative correlation structure than DESI shipped, i.e. a physics ASSERTION about
   z-coherence, not a neutral repair. Floating either WITHOUT the surgery = full variance
   double-count (conservative but wrong).
5. **Cov-eigenmode amplitudes (float top modes of cov_syst blocks).** Models: the dominant
   shipped variance directions as mean freedom. Same surgery requirement per floated mode;
   blurs the per-systematic attribution (modes mix DLA with resolution/continuum at low z);
   no advantage over option 4 for a DLA-specific question.
6. **Pure covariance treatment (remove alpha_dla, rely on C_data alone).** Models: zero-mean
   variance only, in DESI's shape and z-block convention. Loses the only MEAN model of a
   contamination DESI states is real and unsubtracted, and loses the R_DLA-direction variance
   (which the cov barely covers, 50x less along t_dla). Strictly less faithful than deployed.
7. **External budget row (no model change).** Models: nothing inside the likelihood; books a
   number outside it. Requires the Section 7 denomination discipline: state the excursion
   convention, the 40-percent-already-priced share, and the denominator, or it double-reports
   the covariance-priced share at the ledger level.

---

## 9. PI decision table and the twelve answers

| Scientific quantity to protect against | Best-supported uncertainty model | Existing implementation that represents it | Missing mapping or evidence | Appropriate test | Appropriate treatment |
|---|---|---|---|---|---|
| Mean residual-DLA contamination bias in the DESI data vector (unsubtracted, positive) | positive-mean term with a FAITHFUL residual-population response shape | alpha_dla mean term (only mean model anywhere; largely shape-ineffective vs DESI's stated shape: 5 percent whitened-mismatch reduction, sign-opposed 665/681 rows) | which excess shape is faithful for the residual population (r_DLA vs R_DLA vs third weak-wing shape) | sim experiment: mask ~85 percent of sim DLAs, re-measure P1D (exceeds current compute envelope); or Wang+2022 missed-population NHI histogram + reweighted template | keep as deployed; ledger the ineffectiveness honestly; no repair without the shape evidence |
| Amplitude uncertainty of the residual fraction (documented range 0.05-0.30) | broad one-sided prior on incidence | deployed softplus width (latent 1.0): covers the full range within ~1.1 sigma; 0.15 at +0.41 sigma | whether DESI's 15 percent is a point estimate or a 1-sigma | DESI-internal query (Karacayli/Wang); DR1 states no uncertainty | existing prior adequate in amplitude; recentering 0.10 -> 0.15 is a PI wording option |
| z-shape of the completeness error | z-resolved or z-correlated incidence freedom | NOT represented: s_dla prior window = ~3 percentage points of completeness drift across the band | the z-resolved DESI completeness curve C(z) for the P1D sample | read Wang+2022 SNR/z figures + Brodzeller+2025 DR2 catalog paper | decision pending evidence; note a z-resolved alpha_dla fixes z ONLY (capture ceiling 19.7 percent) |
| k-shape of the envelope (the 80 percent stranded mode) | envelope-shaped freedom: either zero-mean in-C (current) or floated e_DLA amplitude | cov_syst DLA mode inside deployed C_data (variance only; prices spread, damps the mean by only ~10 percent) | none mechanical; the float variant requires the specified surgery (SPD-verified) | already characterized (GLS + derived worst case); NUTS-level check only if the float is ever built | current covariance treatment is DESI-sanctioned and conservative; the float+surgery alternative is fully specified in Section 8.4 |
| z-coherence of the catalog error (the denomination lever) | correlated-latent family exp(-dz/l) between R1 and R0 | shipped cov_syst asserts l = 0 (convention, INFERRED); the campaign asserts l = inf | DESI's intent; the validation paper arXiv:2509.13593 unread | one WebFetch pass over 2509.13593 + Zenodo record docs | quote the RMS as a range (ns 0.157-0.388 linear, ~0.10-0.24 calibrated) until settled |
| Purity / over-masking (opposite-sign estimator distortion) | none exists on either side | NOT representable by alpha_dla >= 0 nor by DESI's incompleteness-only envelope | any size estimate | DESI-side estimate (mask random clean windows at the false-positive rate) | external ledger note, explicitly unsized; bounded small at DESI's sensitivity by their high-confidence rerun |
| Ledger denomination of the Gate-B DLA row | the Section 7 triple (typical / as-run / derived worst case) | campaign cells = the as-run stress test only | PI's choice of denomination + the z-coherence length | none further; all three routes converge on ~0.10 sigma_post per stated-model sigma | book the row with the triple + the 40-percent-already-priced note + the excursion convention; never the bare dichotomy |

### The PI's twelve questions, answered

1. **What does "completeness" mean here?** Six different things (Section 2). The primary
   referent is DESI's catalog completeness: the probability that the DR1 concordance
   CNN/GP finder flags a real DLA on a DESI sightline (~0.85). Every other use (our 0.10, the
   envelope, the cov mode, the injection, the sim finder's 0.971) is a distinct derived or
   unrelated object; the sim-side 0.971 must never appear in a DESI budget context.
2. **How was it measured?** By validating the Wang+2022 CNN finder on synthetic DESI mock
   spectra with inserted DLAs, summarized by DR1 as "around 85%" completeness and purity,
   "depending on the confidence level and SNR cuts", with no quoted uncertainty and no
   z/NHI/SNR-resolved curve propagated; the high-confidence catalog rerun was a purity
   consistency check, not the estimator (all verbatim-verified, main.tex:206,219,221,227,235).
3. **What is the latent quantity?** The residual unmasked-DLA population, (1 - C) times the
   DLA incidence, times its P1D response. One physical unknown; all six objects are
   re-representations of it, with the caveat that the two response SHAPES in play (DESI's
   fitted r_DLA and PRIYA's in-situ R_DLA) are asserted, not shown, to share that one parent.
4. **Scalar, z-function, NHI-function, or data-vector covariance?** DESI collapses it to a
   scalar (0.15) times a fitted (k,z) template entering a per-z rank-1 covariance mode; our
   pipeline collapses it to a scalar (0.10) at the z = 3 pivot times a rigid sim-anchored z
   power law through R_DLA. Neither side carries an NHI axis; both are scalar compressions of
   what is physically a C(NHI, z, SNR) function that nobody has propagated.
5. **Does alpha_dla represent the implied mean?** In amplitude, yes: the deployed center is a
   valid compression of the generative integral to 3.6 percent, sitting 0.41 latent sigma
   below the DESI-15-percent-implied center (prior mean within 5-9 percent of it). In shape,
   mostly no: at most 19.7 percent of the whitened DESI envelope is reachable by any
   z-reparametrization of the lever, and the deployed 1-dof center captures 10.4 percent.
6. **Is the prior empirically calibrated?** No. The center is a PI design assumption (commit
   1e1b278, 2026-06-09) compressing the literature "> 90 percent complete at logNHI > 20.3,
   good SNR" figure; it was never fit and never derived from a measured DESI curve. The width
   is a hardcoded latent 1.0 (the named 0.50 is not deployed) that happens to cover the whole
   documented completeness range; that coverage is a check performed here, not a calibration.
7. **Does 0.10 duplicate, approximate, or differ from the DESI uncertainty?** It approximates
   the same intended object (the missed-DLA fraction of the same survey's masking) with a
   numerically inconsistent value (1.5x below DESI's 0.15; +0.41 latent sigma, in-support).
   As an UNCERTAINTY it only partially duplicates the in-C envelope: 10 percent directional
   overlap in the deployed metric, 0.80x duplicate pricing along the envelope direction, ~5
   percent of n_s posterior variance per device; the mean role is not duplicated at all.
8. **What is the column, statistically?** A phenomenological 1-sigma envelope: (assumed
   scalar incompleteness 0.15) x (smooth Eq-4.1 fit to the MEASURED mask-none-vs-mask-all
   fractional excess on real DR1 data) x P_smooth. Equivalent to |dP/dC| DeltaC only under a
   pro-rata assumption that plausibly overstates the weighted residual (missed DLAs are
   low-SNR, downweighted, low-NHI). Not a derivative, not a two-catalog difference, not a
   covariance sqrt, not a max excursion; its parent measurement is signed positive, the sign
   discarded at the outer product.
9. **Is cov_syst a probability model or an envelope?** Both, layered: in FORM it is an exact
   probability model (zero-mean Gaussian, one fully k-correlated unit latent per z block,
   cross-z exactly zero, verified to 6.2e-18), and DESI's stated intent is marginalization
   "without extra parameters"; but its input SCALE is an envelope-grade number (the 15
   percent, no stated uncertainty) and its cross-z zero is an assembly convention the paper
   never defends: a probability model built from envelope inputs.
10. **Is the campaign a 1-sigma realization, a constrained worst case, or an envelope stress
    test?** An envelope STRESS TEST: exactly ~1 sigma per z block, jointly chi2 = 10.71
    (~3.27 sigma) under DESI's own z-block model, 1.72 whitened sigma inside the deployed
    likelihood, and 74 percent of the DERIVED same-radius worst case (whose maximizer is a
    sign-changing z profile peaked at z 3.4-3.6). It is a 1-sigma realization only under the
    fully z-coherent reading R0, which is a physics choice DESI's bookkeeping does not settle.
11. **Would a floated e_DLA double count?** Yes if floated on top of the shipped C_data (the
    mode priced twice: conservative but wrong); no if paired with the rank-1 surgery
    C_data <- C_data - sum_z outer(e_dla|z), which is SPD-verified feasible and has an exact
    in-repo precedent (the deployed option-b resolution surgery) plus the data producer's own
    cosmology-fit precedent. The per-z eleven-amplitude variant is mathematically identical
    to the removed cov term; the single z-coherent amplitude additionally asserts z-coherence
    DESI did not ship, a physics claim, not a neutral repair.
12. **Which quantity should enter the ledger?** Not one number: the Section 7 triple,
    sigma_clean-denominated, with its conventions stated: (a) covariance-consistent typical
    shift under DESI's stated model, ns ~0.10 calibrated (0.157 linear; rising to ~0.24
    calibrated if the error is fully z-coherent); (b) the as-run z-coherent stress test, ns
    0.21-0.28 at joint ~3.3 sigma, 74 percent worst-case-efficient; (c) the derived
    same-radius worst case, 1.36x the arm. Every row must carry the note that 0.23 sd_post of
    the arm's shift is already priced by the in-C covariance (the 40 percent denomination
    share) and that the z-coherence length is the open decision variable. The bare
    "3.3-sigma vs renormalized 1-sigma" dichotomy must not appear.

---

## Provenance

Evidence reports and scripts: /tmp/claude-114399728/-home-mfho-hcd-priya/
b1bb98ec-5b1e-4e34-aead-7ffef9ccbc69/scratchpad/dla_semantics_audit/ (definitions-ledger-
builder.md, completeness-measurement-archaeologist.md, emulator-assumptions-auditor.md,
propagation-modeler.md, double-counting-analyst.md, statistical-questions-analyst.md,
referee.md; compute_overlap.py, c6_statistical_questions.py, c3_propagation.py,
c3_capture_ladder.py, referee_checks/, plus npz artifacts and the downloaded arXiv:2505.07974
LaTeX source). Settled prior audit: docs/audits/
2026-07-12_dla_alpha_prior_and_completeness_injection_audit.md and
/home/mfho/hcd_priya_notes/docs/superpowers/dla-deep-audit-artifacts/. Data:
/home/mfho/data/desi_dr1_p1d/. Campaign pkls: /scratch/cavestru_root/cavestru0/mfho/
dla_deployed_{m1,p1}. Synthesis re-verification commands (this session, verbatim):

    /home/mfho/.conda/envs/emu-jax/bin/python3 <heredoc>  # D1 latent offsets: -0.694/+0.407/+1.103
    /home/mfho/.conda/envs/emu-jax/bin/python3 <heredoc>  # D2 recipe: 6.18e-18; off-block 0.0; cov=stat+syst 3e-17

Stop-condition compliance: no floated nuisance added, no prior changed, no ledger finalized,
no re-SBC, no eBOSS campaign, no campaigns/sbatch, zero forward-build chains spent; the
worst-case direction is DERIVED wherever a worst case is named; the three fraction objects
and six PI objects are kept distinct throughout; this report frames decisions and recommends
no action that violates any stop condition.
