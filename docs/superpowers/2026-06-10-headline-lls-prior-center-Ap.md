# Headline — the LLS prior *center* carries ~1σ of the DESI A_p

*2026-06-10. A focused finding doc (preliminary; the Phase-4 run is still completing). The closure
machinery is unbiased when the prior is right, but the **center of the LLS-incidence prior** — a quantity
we set, not measure from the P1D — moves the DESI cosmology amplitude A_p by **~1σ**. This is the dominant
*real-fit* A_p risk, and it is a prior/data problem, not a likelihood bug.*

---

## The claim, in one line

In the separate-inference closure, with the LLS prior centered on a **realistic (non-circular)** value
rather than the truth, **DESI A_p scatters by ~±1σ**, and changing *only* the LLS prior center at a fixed
fiducial moves A_p by **~0.8σ**. KS is much less sensitive. So **how well we know the LLS abundance of the
data sample sets a ~1σ floor on the DESI A_p error** unless it is pinned externally.

![LLS-center → A_p budget](../../figures/analysis/05_likelihood/lls_center_ap_budget.png)

- **(A)** At the **sim-population-mean** LLS prior center (non-circular — the prior is *not* told the
  truth), DESI A_p scatters ±~1σ across fiducials and **flips sign** (D_f3 −0.92σ, D_f4 −0.62σ, D_f6
  +1.02σ); KS stays within ~±0.4σ.
- **(B)** The clean cut: at the **same** fiducial (D_f6, n_s=0.966), changing *only* the LLS prior center
  from the truth to the sim-mean moves A_p from **+0.20σ → +1.02σ** — a **+0.82σ** shift from the prior
  center alone.

## Why this happens (the mechanism)

LLS absorption adds low-k power that is **interchangeable with the clustering amplitude** to the
likelihood — so A_p and the LLS amplitude α_LLS are degenerate (we saw corr ≈ −0.3 to −0.5). Critically,
**the data cannot pin the LLS amplitude on its own** (the LLS/subDLA templates are collinear over the
measured k-range), so α_LLS is effectively **set by its prior**. If the prior *center* is offset from the
sample's true LLS abundance, α_LLS is pulled to the wrong value and **A_p compensates** — by ~1σ for an
offset of order the sim-population spread.

DESI is more exposed than KS because DESI's low-k modes are exactly where the LLS damping wings and the
clustering amplitude overlap; KS's bias shows up more in n_s.

## Why it matters

- The closure **machinery is unbiased** — center the prior on the truth and A_p recovers in-gate (+0.20σ).
  So this is **not a likelihood bug**.
- But in the **real fit** we don't know the truth; we center the LLS prior on literature, and **the data
  sample's actual LLS abundance can differ** (selection bias — the PRIYA-KS paper's published α_LLS≈2 case
  is exactly this). So the LLS-prior-center accuracy is a **~1σ A_p error budget**.
- ⟹ The decisive real-fit lever is an **external LLS-abundance measurement of the specific sample**, per
  survey (not a shared/assumed center) — and the prior should be **moderately** informative so a slightly
  wrong center can be partly corrected by the data, not pinned.

## Caveats (this is preliminary)

- **n = 3 DESI fiducials**; the Phase-4 run is still completing (D_f7 + the sensitivity arms pending). The
  ±1σ is the current scatter, not a converged error budget.
- **Panel B mixes one confound:** the truth-center (+0.20σ) and sim-mean (+1.02σ) runs used *different
  noise seeds*, so part of the +0.82σ is noise, not purely the center. **The clean test is the
  `D_f6_lit` arm** (same mock, same noise, only the center → literature) — *pending* — which will isolate
  the center effect exactly. The forward-only Laplace scan (below) already showed the directional center
  dependence.
- The cross-fiducial ±1σ scatter (Panel A) is **center-offset + per-sim emulator-LOSO error + noise**
  combined; the center is *one* contributor, cleanly isolated only by the same-mock arm.
- Some chains are ESS-short (D_f6 tail 155 vs 400 target); the cert proper wants longer chains.

Supporting forward-only preview (the center × width dependence):

![LLS prior sensitivity (Laplace)](../../figures/analysis/05_likelihood/lls_prior_sensitivity.png)

## ⚠️ FUTURE CHECK — does this hold for a *flatter* prior?

**Open question to test (PI's note):** the ~1σ A_p sensitivity is measured at the *current* LLS prior
width (σ/μ=0.15 — fairly **strong**). A strong prior on a wrong center is exactly what pins α_LLS and
biases A_p. **Hypothesis:** a **flatter (looser) LLS prior** lets the data pull α_LLS toward the truth, so
the center offset matters *less* → the A_p shift shrinks — at the cost of a wider A_p posterior (since A_p
then absorbs more of the unconstrained LLS direction).

- **First data point:** the `D_f6_sig40` arm (σ_LLS 0.15→0.40, same mock as D_f6) — *pending*; compare its
  A_p bias to D_f6's +1.02σ.
- **Future work:** a fuller width scan (σ_LLS → wide/uniform) at a fixed non-circular center, mapping the
  **bias-vs-width trade-off** in full NUTS (the Laplace above is only directional and under-states the
  bias ~×3). The right LLS-prior width is where the center-mismatch bias is acceptable *and* A_p stays
  usefully constrained.

If a flatter prior removes most of the ~1σ sensitivity, the real-fit recipe is "loosen the LLS prior +
externally pin the center"; if it doesn't (A_p stays biased because the center still dominates), then the
external pin is mandatory and tightness is secondary. **This is the key thing to settle before the real
fit.**
