# HCD marginalization in the Lyα-forest P1D — literature review + architecture correction

**Date:** 2026-06-01. Three parallel literature agents (web-verified, full-text equations
pulled). This corrects a mis-framing in our Phase-2b likelihood and aligns it with the
field standard (McDonald 2005 → Rogers & Bird 2018 → eBOSS/DESI → PRIYA 2025).

---

## 1. The field-standard model

HCD (LLS / sub-DLA / small-DLA / large-DLA) contamination of the forest P1D is a
**multiplicative, scale-dependent bias** with a **fixed per-class shape** and a **single
free amplitude per class**:

    P1D_obs(k,z) / P1D_forest(k,z) = α_0(z) + Σ_c α_c(z) · S_c(k,z)
    S_c(k,z) = ((1+z)/(1+z0))^(-3.55) · 1/(a_c(z)·e^{b_c(z)·k} − 1)^2 + c_c(z)   [Rogers&Bird18 Eq.6]

- The damping-wing **shape** `S_c` (the `1/(a e^{bk}−1)^2` kernel) is **calibrated/fixed
  from simulations** (Rogers Table 2); its k-turnover scales with N_HI, so the four classes
  have **distinct shapes** → P1D can separate them.
- The **amplitude `α_c` is the single free, fit-and-marginalized parameter per class.**
- Legacy eBOSS (McDonald05/PD2015/Chabanier19) used **one** lumped `α_DLA` on a single
  combined template: `1 − 0.2·α_DLA·[1/(15000k − 8.9) + 0.018]`. DESI DR1 uses the **4
  per-class** Rogers form with `a,b` fixed and the four `f_c^HCD` free.

## 2. THE KEY RESULT: α_c IS the (effective, residual) incidence — P1D measures it

Rogers & Bird 2018 §6 (verbatim): the `α_c` *"are each related to integrals of the HI CDDF
for a particular survey over the appropriate column density ranges (and absorption distance
per sightline)."* Eq. 4 text: *"α_i(z) are the fraction of spectra in each absorber category
… In a real survey, α_i(z) may change from their raw values."*

DESI DR1 cosmology (arXiv:2601.21432 §4.2): the `f_c^HCD` *"correspond to the fraction of
line-of-sights containing at least a particular type of HCD"* and *"will primarily reflect
the efficiency of the masking algorithm."*

⇒ **The per-class HCD amplitude is the effective residual (post-masking) incidence — an
effective dN/dX integral. Fitting it from P1D yields a (rough, per-class) effective dN/dX.**
The distinct per-class damping-wing shapes provide the leverage to separate the classes.
"Rough" because the HCD term is subdominant to the forest (eBOSS's single `α_DLA` fits to
**null** with large error; DESI floats 4 with weak priors).

## 3. THE MASKING POINT: effective dN/dX ≠ simulation dN/dX, by construction

DLA-finder completeness/purity reshape the residual CDDF away from intrinsic:
- DLA finders (Parks+2018 CNN, Ho-Bird-Garnett GP, Wang+2022) are **>90% complete/pure for
  log N_HI > 20.3 at good S/N**, but completeness **collapses for sub-DLA (19.5–20.3) and LLS
  (17.2–19.0)**, which are largely **not masked**. DESI DR1 concordance catalog: **~85%
  completeness & purity** (Karaçaylı+2025); cosmology paper quotes **~70% complete** for
  N_HI>2e20 at SNR>2, "drops rapidly" below.
- Rogers & Bird 2018 (verbatim): *"the clipping of the survey spectra changes the survey
  CDDF. The bias will have a different scale-dependence (not just amplitude), since this is
  driven by the distribution of the widths of damping wings remaining."*
- Karaçaylı+2025: residual modeled as a systematic `σ_DLA = 0.15·r_DLA(k,z)·P_smooth` (15%
  assumed incompleteness). They note even **survey-to-survey** masking differs ("eBOSS masks
  more sub-DLAs as DLAs than DESI").

⇒ The residual unmasked population contaminating the **data** P1D has an effective dN/dX set
by the **data's** completeness/purity — **necessarily different from the simulation's
intrinsic dN/dX.** This is exactly why `α_c` is floated, not fixed.

## 4. PRIYA's own analysis already does this

PRIYA XQ100/KODIAQ-SQUAD (arXiv:2509.18271): *"we include the full four-parameter HCD
template in the likelihood: α_LLS, α_subDLA, α_small-DLA, and α_large-DLA,"* template
"based on simulated absorber populations in Illustris", with **one-sided positive priors**
on the DLA amplitudes "accounting for potential residual contamination in the data" (present
in data, ~absent in the masked sim). This is the template to match.

## 5. Per-survey summary

| Analysis | arXiv | # HCD params | form | prior | dN/dX? |
|---|---|---|---|---|---|
| Rogers&Bird (template) | 1706.08532 | per-class α_c | `(a e^{bk}−1)^{-2}` fixed shape | calibrated on sims | α_c = CDDF integral per class |
| Chabanier eBOSS | 1812.03554 | 1 (α_DLA) | single lumped | weak → **nulls to 0** | not converted |
| PD eBOSS cosmo | 1911.09073 | 1 | same | weak (Δχ²≈0) | — |
| eBOSS emulator | 2412.05372 | 1 (A_DLA) | same | flat | — |
| DESI DR1 cosmo | 2601.21432 | 4 (f_c^HCD) | Rogers, a,b fixed | free/flexible-z | "reflects finder efficiency" |
| **PRIYA 2025** | **2509.18271** | **4 (α_c)** | **Rogers template (Illustris)** | **one-sided positive** | **effective residual incidence** |

**Open niche:** nobody yet publishes a P1D-measured residual HCD dN/dX *with* uncertainty —
our per-class, cosmology+τ₀-aware emulator is positioned to.

---

## 6. Correction to our Phase-2b likelihood (what changes)

**Before (wrong):** `P_obs = P_tier_p + Σ_c w_c·A_c·Δ_c` — TWO free amplitudes (`w_c` from
dN/dX + nuisance `A_c`) → artificial degeneracy; I claimed "P1D can't give dN/dX."

**After (field-standard, PRIYA-consistent):**

    P_obs(k) = P_tier_p(k) + Σ_{c∈HCD} α_c · Δ_c(k)

- **`α_c`** = the **single** free per-class amplitude = **effective residual (post-masking)
  incidence** (∝ effective dN/dX integral over class c). **P1D measures it; its posterior is
  the rough per-class effective dN/dX deliverable.**
- **Prior on `α_c`:** centered on Head A's sim-intrinsic `w_c(dN/dX)` (cosmology-driven), but
  **wide / one-sided-positive** (PRIYA-style) to absorb the data-vs-sim masking difference.
  Head A's dN/dX is the **prior center**, NOT a second free amplitude.
- **`Δ_c(k)`** = the emulated per-class HCD power template (the damping-wing-shaped
  contribution). NOTE: Rogers/DESI *fix* the shape (z-evolving, cosmology-independent); ours
  **emulates** it (cosmology+τ₀-dependent) — a refinement to validate (does the emulated shape
  stay consistent with the fixed Rogers kernels at fiducial cosmology?).
- The old "A_subDLA absorbs the half-masking" intent is **subsumed**: `α_subDLA` directly IS
  the effective post-masking sub-DLA incidence.

**Baseline subtlety to resolve when wiring:** `P_tier_p = Σ_c w_c·P_c^filt` is built with the
**sim's** filter (`tau_thresh=1e6`) and the **sim's** `w_c`. The data's masking differs, so
the data baseline is not exactly `P_tier_p(sim filter)`. Two consistent framings: (i) keep
`P_tier_p` as the sim-filtered baseline and let `α_c·Δ_c` add back the residual relative to
it; or (ii) make the filtered tier use the survey's actual masking. Flag for the likelihood
build; (i) is the lighter touch and matches how Rogers adds residual contamination to a
clipped baseline.

## Sources
McDonald+2005 (astro-ph/0407378); Rogers&Bird 2018 (1706.08532) + 3D (1711.06275);
Chabanier+2019 (1812.03554); PD+2019/2020 (1911.09073); eBOSS emulator (2412.05372);
Ravoux+2023 (2306.06311); Karaçaylı+2025 (2505.07974); DESI DR1 cosmo (2601.21432);
PRIYA 2025 (2509.18271); Parks+2018 (1709.04962); Ho-Bird-Garnett (2103.10964);
Wang+2022 (2201.00827).
