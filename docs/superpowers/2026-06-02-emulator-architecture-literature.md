# Lyα-forest P1D emulator architecture — literature review (PCA vs polynomial vs MLP; GP vs NN)

**Date:** 2026-06-02. Source: a deep-research workflow (5 search angles → fetch → 3-vote
adversarial verification → 70 web-verified, non-refuted claims quoting primary sources).
Question: for our Phase-2b emulator (~60 LF sims, 9 cosmo/IGM params + z + τ₀, P1D split
by HCD class, differentiable for HMC), is a **dense per-k MLP with no PCA** justified, or
should we use PCA / a Gaussian Process / a polynomial-chaos surrogate?

---

## 1. Polynomial surrogates — SUPERSEDED
Rogers, Bird, Peiris, Verde, Font-Ribera & Pontzen 2019, "An Emulator for the Lyman-α
Forest" (arXiv:1812.04654, JCAP 02(2019)050): GP emulator from **21 small simulations**
(Latin-hypercube sampling), ~1.5% typical / 4% worst-case accuracy, and **explicitly
outperforms prior quadratic-polynomial interpolation**. ⇒ Polynomial / polynomial-chaos is
the old approach, beaten by GP. **Do not use.**

## 2. PCA / dimensionality reduction — NOT the norm; per-k is standard
Fernandez, Ho & Bird 2022, "A multifidelity emulator for the Lyman-α forest flux power
spectrum" (arXiv:2207.06445, MNRAS 517,3200) — the **PRIYA-lineage** emulator — states
verbatim: *"Since our target summary statistic is a vector, we model each k bin of the flux
power spectrum with a separate GP. The primary reason for this choice"* is to preserve the
**scale-dependence**. That is per-k emulation, the **opposite of PCA compression**. ⇒ Our
"no PCA, keep per-k structure" choice **matches the PRIYA design philosophy**. ✓

## 3. GP vs NN — the field uses BOTH

**GP lineage (incumbent, Bird group / PRIYA):**
- Rogers/Bird 2019 (1812.04654): GP, 21 sims, beats polynomial.
- Fernandez, Ho & Bird 2022 (2207.06445): **GP, not NN** (no neural network in the paper);
  multi-fidelity **Kennedy–O'Hagan (KO / AR1) linear** model for the main results; per-k
  separate GPs; *"A GP provides closed-form expressions for predictions … naturally comes
  with uncertainty quantification."* Nonlinear deep-GP (**NARGP**, Perdikaris+2017) is in the
  appendix and gives only **~0.08%** improvement while needing more HF sims ⇒ linear preferred.
- Walther, Schöneberg, Chabanier, Armengaud et al. 2024 (arXiv:2412.05372, JCAP 05(2025)099):
  **GP emulator** for the **Lyssa** suite (18 high-res Nyx sims, 120 Mpc, 4096³), eBOSS P1D.

**NN lineage (modern, used by DESI):**
- Cabayol-García, Chaves-Montero, Font-Ribera & Pedersen 2023, "A neural network emulator for
  the Lyman-α 1D flux power spectrum" (arXiv:2305.19064, MNRAS 525,3499) — the **LaCE-NN**
  emulator: NN surrogate, **sub-percent precision across k∥=0.1–4 Mpc⁻¹, z=2–4.5**; used in
  DESI Lyα analyses.
- The **LaCE** package (`github.com/igmhub/LaCE`) ships **both** `gp_emulator.py` and
  `nn_emulator.py` (+ `nn_architecture.py`) — the field maintains GP and NN side by side.

**Training-set sizes:** GP works at ~18–21 sims; our ~60 LF is comfortable for either.

## 4. Assessment for our setup

| Choice | Verdict | Why |
|---|---|---|
| PCA compression | **Avoid** | PRIYA-lineage emulates per-k specifically to keep scale-dependence; PCA would smear low-k vs high-k. |
| Polynomial / PCE | **Avoid** | GP beats quadratic polynomial (Rogers 2019). |
| Dense per-k outputs | **Keep** | matches the per-k GP philosophy. |
| **MLP vs GP** | **MLP defensible; GP is the small-sample alternative** | see below. |

**Why MLP is well-motivated for *us* (beyond LaCE-NN precedent):** we emulate **~1240 outputs**
(f_nhi[30] + dN/dX[3] + 4×172 filtered class P1D + 3×172 Δ_c) with a **structural sum identity**
(`P_tier_p = Σ_c w_c·P_c^filt`), a **τ₀ dimension**, and **joint CDDF↔P1D coupling**. A shared
encoder + joint heads + end-to-end differentiability (for HMC) fits a single NN far better than
~1240 independent GPs. PRIYA's GP solves the *simpler* single-total-P1D problem (172 per-k GPs).

**The one real caveat (the small-sample regime):** with only ~60 sims, GP is more data-efficient
and gives **uncertainty for free**, whereas an MLP can overfit ~60 points and we must *construct*
the uncertainty (the k-fold LOSO error vector). Mitigations already in the plan: AdamW weight
decay, k-fold LOSO error vector (Task 14), held-out-α/τ₀ validation (Tasks 4, 15).

## 5. Recommendation
Keep the **dense per-k MLP, no PCA, no polynomial** — right for our richer per-class /
structural / differentiable problem, with the LaCE-NN precedent. Mitigate the small-sample
overfitting risk with the planned regularization + LOSO error vector. **Optional validation
baseline:** a per-k GP cross-check (PRIYA-native) on one or two output channels once the cache
lands, to confirm the MLP isn't leaving accuracy on the table at ~60 sims.

## Sources
Rogers/Bird 2019 (1812.04654); Fernandez, Ho & Bird 2022 (2207.06445, MNRAS 517,3200);
Walther et al. 2024 (2412.05372, JCAP 05(2025)099); Cabayol-García et al. 2023 (2305.19064,
MNRAS 525,3499); LaCE package (github.com/igmhub/LaCE). Method: deep-research workflow
wf_25749b73-8f9, 70 verified non-refuted claims.
