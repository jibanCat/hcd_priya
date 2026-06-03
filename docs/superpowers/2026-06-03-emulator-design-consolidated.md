# Phase-2b Lyα P1D emulator — consolidated design (2026-06-03)

**Branch:** `phase2-emulator-jax` · **Status:** design consolidated post-referee + post-τ₀
+ post-k-representation decision; gating a sub-percent-feasibility review before real
tuning/training. Supersedes/integrates: spec `2026-05-29-phase2b-emulator-design.md`,
`2026-06-02-normalization-fix-research.md`, `2026-06-02-tau0-error-model-and-closure-design.md`,
`2026-06-02-training-walkthrough.md`. Memories: `phase2-tau0-cosmology-interaction`,
`phase2-hr-phase1-and-bins`.

## 1. Goal & data
Emulate the per-class filtered Lyα-forest P1D(k) for cosmological inference, marginalize
HCDs, return per-class effective dN/dX. **Inputs:** 9 PRIYA params `[ns, Ap, herei, heref,
alphaq, hub, omegamh2, hireionz, bhfeedback]` (thermal/pressure carried by the reionization
axes — see §6) + redshift z + mean-flux τ₀=−ln⟨F⟩. **Targets:** 4 absorber classes
(clean/LLS/subDLA/DLA). **Training:** the τ₀ cache — LF `observables_tau0_lf.h5` (21440 rows,
n_k=172, k=4e-4..0.069 s/km), HR `_hr.h5` (n_k=525, higher k_max), ~60 sims × 20 τ₀-rescale
levels × ~18 z. **Target data:** KODIAQ-SQUAD-XQ100 (high-res OQE, high-k → HR fidelity) +
DESI DR1 P1D (lower-res → LF fidelity).

## 2. Architecture (the baseline+residual normalization redesign)
Cosmology is only ~0.4% of the per-k log-variance (the (z,τ₀) spread is ~99.5%). A global
per-k σ standardization buries it. Fix = Kennedy-O'Hagan structured-mean + zero-mean
residual, IN-NETWORK:
- **Encoder** (θ,z,τ₀) → latent (the in-network data reduction).
- **θ-BLIND baseline head** `m̂(z,τ₀)` (input `[z,τ₀]` only; ∂m̂/∂θ≡0 exactly) — the
  (z,τ₀)-conditional MEAN log-P1D.
- **Cosmology residual head** `r̂(θ,z,τ₀)` — the within-cell cosmology signal in conditional
  σ_cosmo units; a joint, non-separable function so ∂²lnP/∂θ∂τ₀ is representable.
- **Reconstruction:** `logP̂ = (m̂·σ_marg + μ_marg) + σ_cosmo·r̂` → exp. All norm stats are
  **per-(class,k)** `(4,172)`: μ_marg, σ_marg (marginal, standardize baseline), σ_cosmo
  (per-k within-cell cosmology scale ≈0.077·σ_marg). θ-response = σ_cosmo·∂r̂/∂θ.
- **Identifiability:** the split is identifiable ONLY because the baseline is θ-blind (forces
  all θ-response through r̂). Verified: jacfwd(baseline)=0, reconstruction round-trip 8e-16,
  structural identity to PRIYA 1e-15.

## 3. Output / k-representation (DECISION 2026-06-03)
Decoder = learned **SVD low-rank basis** (n_basis≈12) `P = coeffs(4,n_basis)·basis(n_basis,K)`
(coeffs from the encoder = the DeepONet "branch") + a **DIFFERENTIABLE cubic-spline
interpolation** onto the data k-grid. This is grid-agnostic (train once, evaluate any survey
grid — no retrain), fully HMC-differentiable (spline = constant matrix `W` for a fixed
sim→data grid, so `∂P_data/∂θ = W·∂P_sim/∂θ`; the resolution window `C_res=1+2 f_res R² k²` is
applied pointwise at data k, so ∂P/∂k is not needed), the most accurate option (SVD ceiling
0.18-0.64% vs the continuous-trunk's 1.2-2.4×), is NOT the rejected polynomial (learned
data-driven basis), and does NOT foreclose metals (Nyquist-limited like any representation;
only a polynomial forecloses structurally). **k-range:** cut the likelihood to the validated
sim range (LF≤0.069, HR≤~0.2 s/km); KODIAQ high-k beyond → HR fidelity / cut. A continuous
**POD-DeepONet trunk** (learned φ_i(k)=trunk(γ(k)), bandlimited Fourier features) is a
validated, novel-in-Lyα design but DEFERRED — it costs ~1-2% accuracy today and earns its
keep only with future metal-variation sims (targeted-frequency band). Prototype: `0701ef8`.

## 4. HCD sector & likelihood
`P_obs(k,z) = P_tier_p + Σ_{c∈HCD} α_c·Δ_c(k,z)`, `P_tier_p = Σ_c w_c·P_c^filt` (structural,
bit-exact to PRIYA). Single per-class **α_c** = effective post-masking residual incidence;
P1D constrains it via the distinct per-class Δ_c damping-wing shapes; its posterior IS the
rough per-class effective dN/dX (analytic M₀-inverse `alpha_to_dndx`). α_c(z) low-order (PW14
or 2-node, matched to DESI DR1). Prior on α_c centered on Head-A's sim `w_c(dN/dX)`, wide.
**Δ_subDLA/Δ_DLA fix:** they carry the same ~86% (z,τ₀) burial the P_filt fix removed
(within-cell var frac 0.143/0.145 vs Δ_LLS 0.977) → apply the same baseline+residual (or
conditional-σ) treatment to protect the α_c posterior. Difference form is default.

## 5. Error model C_emu(c,k,z,τ₀) — τ₀-aware
Full design in `2026-06-02-tau0-error-model-and-closure-design.md`. Producer
(`fold_resid_neff`) gains a τ₀-band axis → `(4,K,Zb,Tb)`; `assemble_covariance` (a) UNITS FIX
— multiply the fractional σ by the predicted `P_c²` (the prior `Σ w_c²·σ²` dropped `P_c²`);
(b) smoothly τ₀-index σ(c,k,z,τ₀); (c) include the now-state-dependent **−½logdet C** term (a
real NUTS gradient on τ₀); (d) k×k off-diagonal — MVP diagonal+envelope inflation, shrinkage
upgrade gated on the closure χ². Mean-flux ⟨F⟩ prior on τ₀(z) at measurement width.

## 6. Why the τ₀×cosmology design is sound (the PI's concern, resolved)
The two-stage split preserves the τ₀×cosmology interaction (∂²lnP/∂θ∂τ₀ = σ_cosmo·∂²r̂/∂θ∂τ₀,
fully in r̂); it is field-standard (= ForestFlow/DESI DR1's "mean-flux baseline + cosmology
correction"); the thermal/pressure state is carried by separate reionization axes
(herei/heref/alphaq→T₀,γ; hireionz→k_F), so τ-rescaling only does the mean-flux-amplitude part
it does faithfully. **Empirically** the interaction is ≈ a scalar rescale for clean/LLS/subDLA
(≤5% amplitude, shape corr >0.98, ≤0.3% of cosmology variance) — homoscedastic σ_cosmo is fine
there. DLA is the exception (≤4× low-k swing, shape corr→0.90) → DLA-specific τ₀-resolved
σ_cosmo. The residual risk is all second-moment (τ₀-aware C_emu, edge loss weight), validated
by the closure test (§7).

## 7. Validation / the "unbiased" gate
- **Honest θ-tracking** referenced to the deployed baseline m̂ (NOT the flattered val-mean):
  the within-cell corr is 0.62-0.92 (not the flattered 0.97); the deployed error decomposes
  into residual-head fit (a)=0.18-0.22 of signal (good) + **baseline mis-fit (b)=0.64-0.81**
  (dominant) → §8.
- **∂P/∂θ** autodiff-vs-finite-diff fidelity per (class,k,z) — HMC integrates the gradient.
- **Closure/SBC** (the gate): mocks from held-out-sim TRUTH at OFF-LADDER τ₀, real HMC
  marginalizing τ₀(z)+α_c with the ⟨F⟩ prior; diagnostics = θ coverage, joint θ+τ₀ SBC (incl.
  the τ₀–θ eigenvector projection), recovered ρ(τ₀,θ) vs finite-diff Fisher, and the
  **bias-vs-ladder-position curve**; loose-prior stress test.
- LOSO 8-fold; Tier-P/clean vs PRIYA bit-anchor; w_c↔dN/dX round-trip; within-data-range
  spline interpolation accuracy.

## 8. The known bottleneck → sub-percent question (what this review must answer)
The deployed absolute error is **~7-10% (clean), baseline-limited**: term (b) — the θ-blind
baseline m̂ under-resolving the per-(z,τ₀) k-shape of the cell mean — dominates, amplified by
the tiny σ_cosmo. Root cause: the `inv_nc` loss weighting shrinks the baseline gradient to
~6e-3 (undertrained). This is also exactly the k-tilt visible in the pred-vs-true residual
panels (`walkthrough/06_pred_vs_true_p1d.png`). The fix (task #13): normalize masked_mse by
Σ(weight·mask) + train the baseline head hard (capacity/epochs), possibly anchor to the
empirical cell-means. **OPEN QUESTION for the review: is sub-percent absolute |P̂/P−1|
achievable with this design?** — bounded by (i) the SVD basis-rank ceiling per class (rank-12:
0.18-0.64% reconstruction; DLA may need higher rank), (ii) the baseline-head trainability with
the inv_nc fix, (iii) the irreducible finite-sample floor (60 sims × 20 τ₀), (iv) within-data-
range spline interpolation (sub-percent expected; the catastrophic low-k spike is at k<data
k_min, cut anyway).

## 9. Phasing (tasks #13-#16; gated by this review)
1. **#13** baseline-head accuracy (inv_nc fix + training) — THE bottleneck → sub-percent.
2. **#14** ∂P/∂θ fidelity + τ₀-aware C_emu (units, τ₀-band, logdet, diagonal-MVP).
3. **#16** staged-bug fix + Δ_subDLA/Δ_DLA conditional split + DLA τ₀-σ_cosmo + edge loss weight.
4. **#15** closure/SBC harness (the gate) + mean-flux prior; retrain all 8 folds.
5. SVD + differentiable-spline k-mapping (#18 decision) wired into the likelihood path.
6. k×k C_emu upgrade — follow-up, gated on closure χ².
