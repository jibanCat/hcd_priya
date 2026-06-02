# Lyα-forest P1D emulator architecture — literature review (PCA vs polynomial vs MLP; GP vs NN)

**Date:** 2026-06-02. Source: deep-research workflow `wf_25749b73-8f9` (5 angles, 16 primary
sources fetched, 76 claims → 25 verified by 3-vote adversarial check → 22 confirmed, 3 refuted,
6 synthesized findings). **This supersedes an earlier same-day draft that was built from a
partial journal read and over-stated the MLP case** — the corrected conclusions are below.

Question: for our Phase-2b emulator (~60 LF sims, 9 cosmo/IGM params + z + τ₀, P1D split by
HCD class, differentiable for HMC) — PCA / polynomial / MLP, and GP vs NN?

---

## 1. The field is GP-dominated; raw per-k MLP is the least-supported option
- **Rogers/Bird 2019** (arXiv:1812.04654, JCAP 02 050): **GP**, 21 LHS sims, 1.5%/4% accuracy;
  explicitly beats quadratic-polynomial *interpolation*. No PCA.
- **Fernandez, Ho & Bird 2022** (arXiv:2207.06445, MNRAS 517,3200) — PRIYA method paper: **GP**,
  multi-fidelity Kennedy–O'Hagan; per-k separate GPs (scale-dependent LF↔HF correlation); "neural"
  appears 0×; nonlinear NARGP no better (≤0.08%). No PCA.
- **Production PRIYA (Bird+2023, arXiv:2306.05471):** **GP**, and **ABANDONED per-k GPs** for a
  single GP-per-redshift across the full k range (memory/speed), negligible accuracy loss. ⇒ the
  "PRIYA per-k" design is the 2022 precursor, not the production 2023 emulator.
- **Walther+2024/2025** (arXiv:2412.05372, JCAP 05 099): **GP** (george, Matérn-5/2), 18 Nyx sims,
  one GP per z predicting all k bins.

## 2. The one NN emulator does NOT regress raw per-k bins — it regresses polynomial coefficients
**LaCE-NN, Cabayol-García et al. 2023** (arXiv:2305.19064, MNRAS 525,3499): a neural network P1D
emulator, **sub-percent (k∥=0.1–4 Mpc⁻¹, z=2–4.5), used in DESI**. Crucially, **both** the LaCE GP
and NN **compress P1D into low-order polynomial coefficients in log-P1D** — GP default
`emu_type='polyfit'`, `ndeg=4`; NN class `MDNemulator_polyfit` maps params → ~5–7 poly coeffs
(deg-5 for k≤4, deg-7 extended), reconstructing `P1D = yscalings·exp(poly(k, coeffs))`. **Not PCA,
not raw k-bins.** The `igmhub/LaCE` repo ships both `gp_emulator.py` (GPy) and `nn_emulator.py`
(PyTorch). So the modern NN precedent uses a **polynomial smoothing prior over k**, regressing a
handful of coefficients — exactly to cut output dimensionality + overfitting in the low-data regime.

## 3. PCA — avoid; not used in any Lyα P1D emulator
PCA appears only in CMB emulators (CosmoPower/-JAX), applied **selectively** to TE/lensing spectra
that have zero-crossings / are log-incompatible (512/64 components). The matter power spectrum
(closest smooth-positive analog to P1D) uses the **direct, no-PCA** NN mapping. A smooth, positive,
log-able P1D is precisely the case where PCA is *not* motivated. (Verifier caveat: the reason is
log-incompatibility, not smoothness; and CosmoPower is CMB, an analogy not direct Lyα precedent.)

## 4. Polynomial — two distinct roles (don't conflate)
- (a) Polynomial **interpolation over the parameter space** — a baseline GP beats (Rogers 2019).
  Avoid. Polynomial-chaos-expansion proper is used by **no** surveyed Lyα P1D emulator.
- (b) Polynomial **compression of the P1D vector over k** — the de facto LaCE smoothing layer
  (deg 4–7), regressed by either GP or NN. **Recommended** for smooth P1D.

## 5. Training-set regime
GP is proven viable at **18–21 sims** (Bird+2019, Walther+2024) — below our ~60. BUT those LHCs
varied **~5 params**; ours has **9 + z + τ₀**, so *dimensionality*, not sim count, is the real
constraint, and 60 sims in ~11-D is sparse. (Synthesis confidence: MEDIUM — no surveyed paper
matches our exact config: HCD-class split, explicit τ₀ dim, JAX-HMC differentiability.)

## 6. The HMC/differentiability axis is the LEAST-evidenced (and is our novel requirement)
**No surveyed Lyα P1D emulator performs gradient-based HMC/NUTS via autodiff** — GPs dominate and
are sampled with non-gradient methods. The CosmoPower claims asserting JAX-differentiable HMC were
**REFUTED** (1-2 votes) and are CMB anyway. So autodiff-HMC through the emulator is the genuine
reason to choose an NN, but it is *not* well-precedented in this field — which makes getting the
output representation right (smoothing prior) more important, not less.

---

## 7. Assessment + recommendation for our setup
| Axis | Verdict |
|---|---|
| PCA/SVD | **Avoid** — absent from all Lyα P1D emulators; smooth positive P1D doesn't motivate it. |
| Polynomial-chaos over params | **Avoid** — GP beats it; nobody uses it. |
| Raw dense per-k MLP, no smoothing | **Least-supported** — the field uses GP, or NN-on-poly-coeffs. |
| GP | Field-proven at low sims, free uncertainty; but **not autodiff-friendly** for our HMC, and awkward for our ~1240 structured outputs + τ₀ + joint heads. |
| **NN + polynomial-coefficient outputs (LaCE pattern)** | **Recommended** — keeps autodiff for HMC AND adds the smoothing prior critical at ~60 sims/11-D; the only NN precedent in the field does exactly this. |

**Concrete recommendation:** keep the NN (for the structural sum + τ₀ + joint CDDF/P1D + autodiff
needs that GPs can't easily meet), but **change Head B's output representation from raw 172 per-k
bins to ~5–7 polynomial coefficients of log-`P_c^filt` per class** (Chebyshev/Legendre in log-k),
reconstructing `P_c^filt` from the coefficients. This is the LaCE smoothing prior, not PCA, and is
the field-standard way to regularize a smooth P1D in the low-data regime.

**The one real tension (the report's open question):** polynomial-coefficient compression assumes
a smooth k-shape. The filtered `P_c^filt` is smooth/positive/log-able → fine. But the HCD delta
`Δ_c` **flips sign at low k** and has a damping-wing turnover a low-order log-polynomial cannot
capture. ⇒ Adopt poly-coeff compression for `P_c^filt` only; keep `Δ_c` in the dense/arcsinh form
(or a representation that handles the sign change). Validate that the compressed `P_c^filt` still
recovers the structural `P_tier_p = Σ_c w_c·P_c^filt` to the required precision.

## Open questions (from the report)
1. Does any DESI-era Lyα emulator actually do autodiff-HMC? (None found — we'd be early.)
2. How does poly-coeff compression interact with HCD-class-split P1D where contaminants add
   non-smooth k-structure (our `Δ_c` sign flip)?
3. At 9+z+τ₀ params with ~60 LF sims, does a GP stay accurate, or does dimensionality favor an
   NN coefficient-emulator? (Argues for NN.)
4. τ₀ as an emulator input (Walther/lym1d) vs analytic post-emulation rescale — which preserves
   differentiability most cleanly? (We use it as a Head-B input.)

## 8. DECISION (2026-06-02, user)
**Option A — raw per-k NN, NO fixed PCA/polynomial pre-compression, NO GP.** Rationale:
- The effective P1D training set is **~60×20 ≈ 1200 rows** (the τ₀/α augmentation), not 60; dN/dX
  is per-sim (60, τ₀-invariant) but adequate with **clever regularization**.
- The **broader cosmological power-spectrum / galaxy-clustering emulator field is NN-dominated**;
  the Lyα-specific GP dominance surveyed above is historical (Bird-group). The deep-research was
  Lyα-scoped, which biased toward GP — noted.
- Keep the dense per-k NN already built (Tasks 8–11) → **no rework** for Option A.

**Preferred refinement (better than fixed PCA/poly): in-network LEARNED latent reduction.** Add a
trainable low-rank output basis to Head B (`Head B → small code (n_basis≈8–16) → learned decoder →
172 k-bins`, i.e. learned end-to-end "PCA"). Advantages over fixed PCA/polynomial: learns the
optimal basis from data; regularizes via the bottleneck; fully autodiff for HMC; and a learned
basis **can represent the `Δ_c` low-k sign flip** that a fixed log-polynomial cannot — so it also
resolves §7's tension. Implement as a toggleable variant (`n_basis`) to A/B against the dense head
in the k-fold LOSO sweep. Regularize via: the bottleneck + AdamW weight decay + LOSO error vector +
early stopping.

## Sources
Rogers/Bird 2019 (1812.04654); Fernandez/Ho/Bird 2022 (2207.06445); PRIYA Bird+2023 (2306.05471);
Walther+2024/2025 (2412.05372); Cabayol-García+2023 (2305.19064, MNRAS 525,3499); LaCE
(github.com/igmhub/LaCE); CosmoPower-JAX (2305.06347). 3 refuted claims logged in the workflow result.
