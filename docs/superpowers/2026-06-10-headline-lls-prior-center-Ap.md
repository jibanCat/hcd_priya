# Headline — a tight LLS prior carries ~1σ of the DESI A_p (the width is the lever)

*2026-06-10, updated 2026-06-11 with the clean same-mock isolation. A focused finding doc. The closure
machinery is unbiased when the prior is right, but a **tight LLS-incidence prior whose center is offset
from the data's true LLS abundance** moves the DESI cosmology amplitude A_p by **~1σ**. The controllable
lever is the prior **width**; the source is the **center offset**. This is the dominant real-fit A_p risk,
and it is a prior/data problem, not a likelihood bug.*

> **Note on a mid-investigation correction:** an earlier version of this doc headlined "the prior
> *center* carries ~1σ," based on a truth-center-vs-offset-center comparison that used *different* mock
> noise. The clean same-mock arms (below) show the *width* is the demonstrated lever and that two offset
> centers (sim-mean vs literature) give the *same* A_p — so the accurate statement is "a *tight* prior
> *enforcing* an offset center." Kept here as the record.

---

## The claim, in one line

In the separate-inference closure, at the current LLS-prior width (σ/μ=0.15, fairly **tight**), **DESI A_p
is biased ~1σ** whenever the prior center is offset from the sim's true LLS abundance — and across
fiducials it scatters ±~1σ (zero-mean). **Loosening the prior (σ/μ→0.40) cuts the A_p bias to +0.29σ** —
but trades it into n_s (+0.54σ) and widens A_p. KS is much less sensitive. So the LLS prior is a ~1σ
DESI-A_p lever, controlled by its **width** and sourced by its **center offset**.

![LLS-prior → A_p budget](../../figures/analysis/05_likelihood/lls_center_ap_budget.png)

- **(A)** At a non-circular (sim-population-mean) LLS center, **DESI A_p scatters ±~1σ, zero-mean**
  (D_f3 −0.92, D_f4 −0.62, D_f6 +1.02, D_f7 +0.01σ; RMS 0.75σ); **KS stays within ±0.4σ** (survey-specific
  — DESI biases A_p, KS biases n_s).
- **(B)** The clean cut — **same mock** (same noise), only the prior knob changes:
  - **center sim-mean vs literature, both σ=0.15: A_p +1.02σ vs +0.99σ** — the center (within this offset
    range) **barely matters**.
  - **width σ=0.15 → 0.40 (same center): A_p +1.02σ → +0.29σ** — the **width is the strong lever**.
  - the truth-center reference (+0.20σ, dotted) is much smaller but used different noise, so it only
    *suggests* the offset is the source — it isn't a clean cut.

## Why this happens (the mechanism)

LLS absorption adds low-k power **interchangeable with the clustering amplitude**, so A_p ↔ α_LLS are
degenerate, and the data **cannot pin α_LLS on its own** (the LLS/subDLA templates are collinear over the
measured k-range) — so α_LLS is effectively **set by its prior**. Then:
- A **tight** prior **enforces** the prior center. If that center is offset from the sim's true LLS
  abundance (both sim-mean and literature are ~+11–17% high for D_f6's sim), α_LLS is pinned high and
  **A_p compensates ~1σ**.
- A **looser** prior lets the **data pull α_LLS back toward truth**, so the center offset bites less →
  A_p bias drops. The cost: A_p widens, and the residual offset **redistributes into n_s**.

DESI is more exposed than KS because its low-k modes are where the LLS damping wings and the clustering
amplitude overlap; KS's bias surfaces in n_s instead.

## Why it matters + the fix

- The machinery is **unbiased** (truth-centered → in-gate), so this is **not a likelihood bug**.
- In the real fit we don't know the truth; the prior center comes from literature and the sample's true
  LLS abundance can differ (selection bias — the PRIYA-KS published α_LLS≈2 case). So the LLS prior is a
  **~1σ A_p error budget**.
- **The fix needs BOTH knobs:** (1) a **correct center** — an external, per-survey LLS-abundance
  measurement of the specific sample (not an assumed/shared center); (2) a **moderate width** — not tight
  (σ=0.15 enforces a wrong center) and not flat (which only reshuffles A_p↔n_s and widens A_p). Loosening
  alone is *not* a fix.

## The flatter-prior question — ANSWERED (your future-check)

The σ_LLS=0.40 arm (`D_f6_sig40`, same mock as D_f6) directly tested it:

| same mock, sim-mean center | bias A_p | bias n_s | A_p width | ESS-tail |
|---|---|---|---|---|
| σ_LLS=0.15 (tight) | +1.02σ | −0.03σ | 0.072 | 155 |
| σ_LLS=0.40 (flatter) | **+0.29σ** | **+0.54σ** | 0.090 | 102 |

**Verdict:** a flatter prior **does** reduce the A_p sensitivity (your hypothesis holds for A_p), **but it
is not a clean fix** — the cosmology bias **redistributes from A_p into n_s** (−0.03→+0.54σ), A_p widens
(+25%), and the geometry gets harder (ESS 155→102). So *flattening alone reshuffles the bias rather than
removing it.* The clean removal requires the **center** to be right (external pin); the width should be
**moderate**, chosen to keep the center-mismatch bias acceptable on *both* A_p and n_s without over-widening.

**→ Now done (Phase-4b below):** the fuller σ_LLS scan (0.08→0.80) *and* a subDLA scan are run and mapped
in [the Phase-4b width scans](#phase-4b--the-full-hcd-prior-width-scans). Still open: settling the
**external LLS-abundance pin per survey** so the center is right in the first place.

## Phase-4 full result (36/36 complete)

| mock | survey | n_s | center | σ_LLS | τ₀-ext | R-hat | div | bias A_p | bias n_s |
|---|---|---|---|---|---|---|---|---|---|
| D_f3 | DESI | 0.901 | sim-mean | 0.15 | | 1.006 | 0 | −0.92 | +0.10 |
| D_f4 | DESI | 0.920 | sim-mean | 0.15 | | 1.008 | 0 | −0.62 | +0.36 |
| D_f6 | DESI | 0.966 | sim-mean | 0.15 | | 1.010 | 0 | +1.02 | −0.03 |
| D_f7 | DESI | 0.998 | sim-mean | 0.15 | | 1.009 | 0 | +0.01 | −0.44 |
| K_f4 | KS | 0.920 | sim-mean | 0.15 | | 1.022 | 0 | −0.43 | −0.35 |
| K_f6 | KS | 0.966 | sim-mean | 0.15 | | 1.011 | 0 | +0.02 | +0.39 |
| **D_f6_lit** | DESI | 0.966 | **lit** | 0.15 | | 1.006 | 0 | **+0.99** | −0.12 |
| **D_f6_sig40** | DESI | 0.966 | sim-mean | **0.40** | | 1.004 | 0 | **+0.29** | **+0.54** |
| **D_f6_tau0x** | DESI | 0.966 | sim-mean | 0.15 | **Y** | 1.009 | 0 | +1.16 | +0.87 |

**Convergence: clean** — every fiducial R-hat ≤ 1.022 (K_f4 the only one >1.01), **0 divergences everywhere**.
**The τ₀-funnel is RESOLVED:** the τ₀-extreme arm (`D_f6_tau0x`, truth at the upper PRIYA corner) has 0
divergences and R-hat 1.009 — the smooth 2-param τ₀ handles the extreme with no funnel (the old 13-rung M2
had R-hat 1.019 + a funnel). It *does* still bias cosmology (A_p +1.16, n_s +0.87σ) — but via the same
tight-LLS-prior×offset interaction, not a sampling pathology.

## Caveats

- Several arms are ESS-short (102–413 tail vs the 400 target); the cert proper wants longer chains — the
  bias *directions* are robust, the exact σ values less so.
- The cross-fiducial ±1σ scatter (panel A) is center-offset + per-sim emulator-LOSO + noise combined; the
  same-mock arms (panel B) isolate the prior knobs cleanly, the cross-fiducial scatter does not.

---

# Phase-4b — the full HCD prior-width scans

*2026-06-11. 42 new chains (PROD config, NW250/NS400/mtd10, 3 chains per width point, 4 per IGM fiducial),
0 divergences everywhere, R-hat ≤ 1.015. Two scans on the **same mock D_f6** (n_s=0.966, sim-mean center,
same noise) varying **one** HCD prior width at a time, plus 6 IGM-parameter stress fiducials.*

![HCD prior-width scans](../../figures/analysis/05_likelihood/p4b_hcd_width_scans.png)

## σ_LLS scan (subDLA width fixed at the 0.40 default)

| σ_LLS/μ | bias A_p | bias n_s | joint √(A_p²+n_s²) |
|---|---|---|---|
| 0.08 (very tight) | +1.13σ | −0.18σ | 1.15 |
| 0.15 (default) | +1.02σ | −0.03σ | 1.02 |
| **0.25** | **+0.55σ** | **+0.30σ** | **0.63** |
| **0.40** | **+0.29σ** | **+0.54σ** | **0.61** |
| 0.80 (very flat) | −0.21σ | +0.92σ | 0.94 |

**The LLS width is a monotonic A_p↔n_s lever.** Tightening pins α_LLS → A_p compensates high (up to +1.13σ);
loosening lets the data pull α_LLS back → A_p relaxes and even crosses zero (−0.21σ at 0.80), but the bias
**redistributes into n_s** (−0.18→+0.92σ). The **joint** cosmology bias is minimized in a **moderate band
σ_LLS≈0.25–0.40 (~0.6σ)** and rises at *both* ends. So neither tight nor flat is right — the recommended
width is moderate, and it must sit on a *correct* center (the external pin).

## σ_subDLA scan (LLS width fixed at the 0.15 default)

| σ_subDLA/μ | bias A_p | bias n_s | joint |
|---|---|---|---|
| 0.20 | +0.71σ | −0.51σ | 0.87 |
| 0.40 (default) | +1.02σ | −0.03σ | 1.02 |
| 0.80 | +1.43σ | +0.45σ | 1.50 |
| 1.50 | +1.50σ | +0.62σ | 1.63 |

**The subDLA width behaves oppositely — loosening it *worsens* A_p monotonically** (+0.71→+1.50σ) and also
pushes n_s up. The joint bias is smallest at the **tight** end. So the fix is *not* "loosen everything":
**keep the subDLA prior tight.** (Mechanism: subDLA is the least data-constrained HCD class — see the
+1–3σ subDLA mis-recovery in the IGM table below — so a loose subDLA prior just hands the emulator/τ₀
misfit a free low-k knob that A_p then has to fight.)

**Recommendation for the real fit:** moderate σ_LLS (≈0.25–0.40) on an externally-pinned center; **tight**
σ_subDLA (≤0.20–0.40). One ESS caveat: `D_f6_sigS020` had ESS-tail 82 (short) so its −0.51σ n_s is the
least certain point; the directions are robust.

---

# Phase-4b — IGM-stress fiducials: the bias is emulator-LOSO-driven, not HCD-driven

*The PI's question: under IGM-parameter stress, is the cosmology biased, and **is it sensitive to the HCD
nuisances**? Six fiducials at the extremes of HeII-reionization (start/end), QSO spectral slope, and
BH-feedback, each run with the **default** HCD prior, 4 chains.*

![IGM HCD-sensitivity](../../figures/analysis/05_likelihood/p4b_igm_hcd_sensitivity.png)

| fiducial | fold | bias A_p | bias n_s | corr(A_p,α_LLS) | corr(n_s,α_subDLA) | α_subDLA rec | **emu-LOSO Fisher A_p** |
|---|---|---|---|---|---|---|---|
| baseline D_f6 | 6 | +1.02 | −0.03 | −0.17 | +0.24 | +1.82σ low | (n/a) |
| αq slope LOW | 2 | +0.07 | +0.06 | −0.23 | +0.37 | +2.74σ low | −0.22 |
| BH-feedback LOW | 1 | −0.65 | +0.28 | −0.26 | +0.16 | +1.03σ low | +1.29 |
| HeII-end LOW | 2 | +0.10 | −0.79 | −0.15 | +0.34 | +1.31σ low | −1.09 |
| HeII-end HIGH | 4 | +0.07 | **+1.59** | −0.10 | +0.37 | +1.29σ low | −1.11 |
| **HeII-start HIGH** | 3 | **+2.03** | **+1.13** | −0.17 | +0.26 | +2.87σ low | +0.30 |
| **QSO slope HIGH** | 1 | **+3.74** | **+2.63** | −0.18 | +0.10 | +3.09σ low | **+3.45** |

**Three findings:**

1. **The HCD↔cosmology coupling is FLAT.** corr(A_p, α_LLS) sits at −0.10…−0.26 across *every* fiducial
   and **does not grow with IGM stress** — the worst-biased mock (QSO-slope-HIGH, A_p +3.74σ) has the same
   −0.18 coupling as baseline. The default HCD priors pin the nuisances, so any degeneracy is spent on the
   prior and the bias is a *mean shift*, not a widened/correlated posterior. **HCD marginalization is not
   the channel through which IGM stress biases cosmology** — the PI's prime worry is de-risked.

2. **The worst case is pure emulator-LOSO error.** QSO-slope-HIGH's closure A_p +3.74σ is reproduced almost
   exactly by the *standalone* emulator-LOSO Fisher bias **+3.45σ** (Δ=+0.29) — the likelihood adds
   essentially nothing. This catastrophic corner is the emulator being inaccurate at a held-out
   parameter-space *edge* (the known LOSO partition artifact), faithfully propagated. HeII-start-HIGH is the
   one case where the closure (+2.03σ) exceeds its emulator bias (+0.30σ) by ~1.7σ — there the tight-prior
   mechanism adds on top (and its α_subDLA is the most mis-recovered, +2.87σ). And HCD/τ₀ freedom can even
   *absorb* emulator error (BH-feedback-LOW: closure −0.65σ vs emulator +1.29σ).

3. **subDLA is the weak nuisance.** α_subDLA is recovered 1–3σ *below* truth in every fiducial (worst where
   cosmology bias is worst), confirming it is poorly constrained by the data — which is exactly why the
   σ_subDLA scan says to keep its prior tight.

**Implication for the real fit:** the residual cosmology-bias risk at IGM extremes is an **emulator
accuracy** problem at parameter-space edges, *not* an HCD-marginalization problem. Real DESI/KS data sit in
the IGM interior (not at these held-out edges), and the analysis already carries an emulator-bias term in
C_emu; the HCD-specific lever remains the LLS/subDLA prior **width** (moderate LLS, tight subDLA) on a
**correct external center**.
