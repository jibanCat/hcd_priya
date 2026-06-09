# MF correction form-safety + dN/dX error-model review (Bayesian / PPL / identifiability lens)

**Author:** Bayesian/PPL statistician. **Date:** 2026-06-08. **Branch:** `phase2c-likelihood`.
**Charge:** (a) is the SEPARABLE + rank-1 MF correction form both SUFFICIENT (captures the measured
(z,τ₀) sign-flip) and SAFE (cannot alias the n_s tilt over 6 clustered HR sims)? (b) does a
θ-independent fixed-slope dN/dX/CDDF resolution correction preserve the α-incidence closure
unbiasedness (<0.17σ) with the incidence-prior widths unchanged?

Evidence reproduced from `mf_rescorr_vs_tau0_z.txt`, `mf_rescorr_loso.txt`, `mf_rescorr_per_class.txt`,
`inference.py`, `dndx_wc.py`. All numbers below are from the committed artifacts.

---

## (a) Form-safety: separable + rank-1 — VERDICT: **SUFFICIENT and SAFE, conditional on the T3 gate**

### A.1 Is it SUFFICIENT? — YES, the rank-1 term is the minimal form that represents the measured interaction.
The plan's form is `g(z,τ₀,k) = gbar_z(z,k) + gbar_tau(τ₀,k) + a(k)·u_z(z)·u_τ(τ₀)`. The question is
whether the single rank-1 term captures the **(z,τ₀) sign-flip** a pure-separable form cannot. I read
the interaction directly from `mf_rescorr_vs_tau0_z.txt` at k~0.0595 s/km (the high-k band where the
correction is largest):

| z | ρ(rung0, low-τ₀) | ρ(rung19, high-τ₀) | Δ_τ0 = high−low | level |
|---|---|---|---|---|
| 2.2 | 1.1064 | 1.0560 | **−0.050** | both >1 (LF deficient) |
| 3.6 | 1.0113 | 0.9925 | −0.019 | ~1 |
| 5.0 | 1.0231 | **0.9281** | **−0.095** | crosses BELOW 1 (LF over-predicts) |

The τ₀-slope of ρ (its dependence on τ₀) **changes with z**: it is −0.050 at z=2.2 and −0.095 at z=5.0,
and the *level* swings from a +5.6%/+10.6% boost at low z to a −7.2% suppression at z=5 high-τ₀. This is
a genuine, ~2× change in the τ₀-response across z = a rank-1 (z,τ₀) interaction: `g ⊃ a(k)·u_z(z)·u_τ(τ₀)`
with `u_τ` ~ the (decreasing-in-τ₀) shape and `u_z` ~ the (increasing-in-magnitude, sign-flipping) shape.
A pure separable `gbar_z(z,k) + gbar_tau(τ₀,k)` forces the τ₀-slope to be the SAME at every z (it is the
sum of a z-only and a τ₀-only function — `∂²g/∂z∂τ₀ ≡ 0`), so it CANNOT bend the τ₀-response from −0.050
to −0.095. **One** rank-1 term has exactly one cross-derivative DOF per k (`∂²g/∂z∂τ₀ = a(k)·u_z'(z)·u_τ'(τ₀)`),
which is precisely enough to represent a single monotone steepening of the τ₀-slope with z. So the
rank-1 form is the **minimal sufficient** representation of the measured interaction — neither
under- (separable) nor over- (free 2-D table) parametrized.

CAVEAT on sufficiency: a SINGLE rank-1 term assumes the interaction is *separable within itself*
(`u_z(z)·u_τ(τ₀)`), i.e. the z-shape of the steepening is the same at all τ₀. The table above is
consistent with that (the τ₀-slope grows roughly monotonically with z), but with only 18 z-bins × 3
sampled rungs I cannot exclude a second interaction mode. That is acceptable: a second rank-1 term over
6 sims is where aliasing risk spikes (more cross DOF), so **stop at rank-1** and let any residual
2nd-mode interaction be carried by the T4 z-resolved C_emu floor (which is z-resolved precisely to
absorb the un-modeled per-z sign-flip residual). Rank-1 + z-resolved floor is the right division of
labor.

### A.2 Is it SAFE (cannot alias the n_s tilt over 6 clustered sims)? — YES, by three structural facts, IF the T3 gate is enforced.

1. **The head is θ-INDEPENDENT by construction.** `gbar_z`, `gbar_tau`, and `a(k)·u_z·u_τ` all read
   ONLY `cond[9]=z_unit` and `cond[10]=τ₀` — never the 9 cosmology params. So `∂g/∂n_s ≡ 0` (T3 §3.3
   verifies this exactly with `jax.grad`). A θ-blind correction **cannot** bias the *closure* n_s
   (it is applied identically to all 60 closure sims, so it adds a constant to every sim's forward and
   drops out of the per-sim n_s residual to first order). This is the key safety property the retired
   `mlp` head LACKED — that head read the 9 params and over-attributed the LF→HR tilt to n_s (4σ).
   The separable+rank-1 head, being θ-blind, is structurally immune to the mlp failure mode.

2. **The DOF count is far below the aliasing threshold.** The mlp head had a full MLP (≫ 6 sims worth
   of params) fitting the θ-gradient → it aliased. The rank-1 interaction adds `a(k)` (K coeffs, but
   smoothed onto the 4-coeff Chebyshev basis → effectively 4 numbers) + `u_z` (a z-shape, ~few DOF) +
   `u_τ` (a τ₀-shape, ~few DOF) = O(10) DOF, fit on 6 HR sims × 18 z × 20 τ₀-rungs = thousands of
   (z,τ₀) cells. The interaction is fit in the (z,τ₀) plane where there is ABUNDANT data (the τ₀
   ladder has 20 rungs PER sim), NOT in the n_s direction where there are only 6 clustered points. **The
   rank-1 term is data-rich in its own (z,τ₀) domain** — this is why it does not need to borrow n_s
   information, the structural reason it differs from the mlp head (which was data-starved in θ).

3. **n_s and τ₀ are correlated (~−0.66), so aliasing is POSSIBLE in principle — which is why the T3
   gate is mandatory, not optional.** The one residual risk: even a θ-blind (z,τ₀) reshaping can alias
   the n_s tilt THROUGH the Fisher Jacobian, because the τ₀ axis the rank-1 term reshapes is ~−0.66
   degenerate with n_s. If `u_τ(τ₀)` happens to point along the τ₀-direction that the Fisher uses to
   absorb n_s, the reshaped (z,τ₀) response will move the n_s MAP even though the head is θ-blind. This
   is NOT caught by §3.3 (θ-blindness) — it is caught by T3 §3.1 (rank-1 ON-vs-OFF n_s delta ≤ 0.1σ)
   and §3.2 (τ₀-width inflation ≤ 5%). **So: SAFE conditional on the T3 gate passing all of G1–G4
   including the aliasing detector.** Without the gate, "θ-blind" alone does not prove no-aliasing.

### A.3 Recommended rank-1 CONSTRUCTION direction (Bayesian preference)
Build the rank-1 from the **SVD of the residual-after-separable**, but CONSTRAIN it, do not take the
free top singular vector:
1. Form `gbar_z(z,k)` and `gbar_tau(τ₀,k)` as the marginal means (the separable part).
2. Form the residual `R(z,τ₀,k) = ρ_measured(z,τ₀,k) − gbar_z − gbar_tau` (the part separability
   cannot fit).
3. **Project R onto the PHYSICS direction, not the free SVD.** The measured interaction is the
   τ₀-slope steepening with z (A.1). Take `u_τ` = the fixed, monotone-decreasing τ₀-shape (e.g. the
   normalized `(alpha_slope − mean)` over the 20 rungs — the mean-flux axis), and `u_z` = the
   z-projection `⟨R(z,·,k)·u_τ⟩_τ` (the per-z amplitude of that τ₀-shape). This makes the rank-1 a
   *physically interpretable* "τ₀-response that strengthens/flips with z" rather than a free SVD
   direction that could rotate INTO the n_s-degenerate τ₀ sub-direction.
   - WHY this is safer than free SVD: the free top-1 singular vector of a 6-sim residual can point
     anywhere the 6-sim noise pushes it, including along the n_s-aliasing direction (the most dangerous
     vector, T3 §4). Fixing `u_τ` to the *physical* mean-flux axis removes that freedom — the term can
     only modulate the (z,τ₀) response along the known physical axis, not rotate toward n_s.
4. Cap `a(k)` so the rank-1 contributes at most the measured interaction strength (~9pp at k~0.05),
   and verify `a(k)` exceeds the per-sim ρ noise (CoV 0.6–1.3%) — below that it is fitting noise; drop
   it (T3 §4).

If the constrained (physics-direction) rank-1 fails T3 §3.1, the fallback is **pure separable + the
T4 z-resolved floor** carrying the sign-flip as variance. Do NOT respond to a T3 failure by adding a
second rank-1 term or a free 2-D table.

---

## (b) dN/dX / CDDF fixed-slope resolution correction — VERDICT: **preserves the α-closure unbiasedness, with TWO flags**

### B.1 The structure (verified in `dndx_wc.py` + `inference.py`)
The forward maps dN/dX → α via `α_c = w_c_from_mu(mu_from_dndx(dN/dX, X̄))` (telescoping-Poisson),
then `P_obs = P_clean + Σ_c α_c(P_c − P_clean)`. The incidence PRIOR (`hcd_incidence_prior`) centers
α on the **OBSERVED** dN/dX (via `lit_over_sim_at_z`, a power-law-in-(1+z) literature/sim ratio), NOT
the sim's dN/dX, with fractional widths `HCD_PRIOR_FRAC_SIGMA = (0.15, 0.40, 0.10)`. The PI directive:
MF the sim's dN/dX/CDDF by a FIXED (θ-independent) slope-in-logN_HI resolution correction, KEEP the
prior widths unchanged.

### B.2 Why a θ-independent fixed-slope dN/dX correction PRESERVES closure unbiasedness — the argument
The α-incidence closure (<0.17σ, `phase2c-hcd-redesign`) is unbiased because the prior center tracks
the *quantity the data measures* (observed dN/dX) and the mock is drawn from the SAME forward the
likelihood uses (`predict_P_obs_and_cov_single_z` is the single source of truth — verified
`closure_mocks.py:122` calls it). A θ-independent resolution correction `r_c(z, logN_HI)` applied to
the **sim** dN/dX:
- enters ONLY through the fiducial `w_c_fid` that sets the prior CENTER and the structural α the mock
  is built from. It is a deterministic, θ-independent reshaping of where the prior sits in α-space.
- Because it is θ-INDEPENDENT, it carries **no gradient toward (n_s, A_p, τ₀)** — `∂r_c/∂θ ≡ 0` — so
  it cannot move the cosmology MAP. It shifts the *α* prior center, and the closure (mock drawn from
  the same shifted center) re-absorbs the shift exactly: prior center, mock truth, and likelihood
  forward all move together → the α residual stays centered → the n_s/A_p MAP is untouched. This is
  the SAME invariance that makes the existing `lit_over_sim_at_z` z-slope (already in production)
  closure-neutral: it too is a θ-independent reshaping of the prior center, and the closure is unbiased
  WITH it. A resolution fixed-slope is structurally identical — another θ-independent multiplier on the
  dN/dX that feeds w_c_fid. **So the <0.17σ result is preserved by construction**, provided:
  - the correction is applied CONSISTENTLY to the prior center AND the mock truth AND (if the
    likelihood reads sim dN/dX anywhere) the likelihood — i.e. it must be in the single forward, not
    bolted onto one side. (CHECK: it is applied at the `w_c_fid` / `snap_dNdX` ingestion point that
    BOTH the mock and the prior read, not separately.)
  - the prior WIDTHS stay unchanged (per directive) — they are fractional (σ/μ), so multiplying μ by a
    fixed slope scales σ proportionally; the *relative* width is preserved, which is what the closure's
    coverage depends on. Keeping the fractional widths fixed is the CORRECT choice for unbiasedness.

### B.3 FLAG 1 (bias risk — the renorm interaction): the fixed-slope on dN/dX is NOT exactly a fixed-slope on α
`w_c_corrected` renormalizes across all 4 classes (`dndx_wc.py:64`: `w / sum(w)`). A fixed-slope
correction on dN/dX_c (per class) → mu_c → w_c, then the renorm couples the classes: boosting DLA
dN/dX (the largest resolution effect — HR has MORE DLA, `mf_rescorr_per_class.txt` HR DLA lowz cnt
14481 vs LF 12612) reduces w_clean and the OTHER w_c through the shared denominator. So a "fixed slope
in logN_HI per class" does NOT map to a fixed shift in each α_c — it is a coupled, mildly nonlinear
reshaping. **This is not a bias risk for the CLOSURE** (the mock is drawn from the same renormalized
forward, so it cancels), but it IS a risk if the correction is characterized/validated as
"α_c → α_c × const" rather than "dN/dX_c → dN/dX_c × r_c(z,logN_HI) THEN re-run w_c_corrected." Spec
requirement: **apply the slope to dN/dX BEFORE `w_c_from_mu`/`w_c_corrected`, and let the renorm
flow** — do not approximate it as a direct α multiplier. The closure test (T6 step 4) will catch a
mis-placement (it would show as an α residual). Confirm the T6 closure re-run is done AFTER the
correction is in the shared forward.

### B.4 FLAG 2 (the prior center moves — is that intended?): the correction shifts where the data is "expected"
The prior center already tracks OBSERVED dN/dX (not sim). A *sim-side* resolution correction makes the
sim dN/dX match the (higher-resolution) truth better, which changes `w_c_fid` (the fiducial structural
weight) and therefore the prior CENTER (since center = `lit_over_sim_at_z · w_c_fid` for LLS/subDLA,
and `0.30·lit_over_sim·w_DLA` for DLA). So the resolution correction will SHIFT the α prior center.
This is intended and benign IF the literature/sim ratio `HCD_LIT_OVER_SIM` was calibrated against the
SAME (corrected or uncorrected) sim dN/dX. **Potential double-count:** if `HCD_LIT_OVER_SIM = (1.06,
0.76, 1.34)` was fit as (literature dN/dX)/(LF-sim dN/dX) and you now ALSO multiply the LF-sim dN/dX by
a resolution slope toward HR, the prior center = literature/sim_LF × resolution × w_c could
**double-apply** the sim→reality gap (once via lit_over_sim, once via resolution). Check:
`scripts/plot_dndx_vs_literature.py` — was the lit/sim ratio computed against LF or HR sim dN/dX? If
LF: the resolution correction is NEW information (LF→HR), distinct from sim→literature, and they
compose correctly (LF→HR→literature). If HR: the resolution correction is already implicit in the
lit/HR ratio and applying it again double-counts → either drop the resolution correction on dN/dX or
re-fit lit_over_sim against LF. **This is the one real bias risk and it is cheap to check** — confirm
the provenance of `HCD_LIT_OVER_SIM` before T6 step 3. (Same class of double-count flag the meta-review
raised for gbar-vs-res_corr, item I4.)

### B.5 Net dN/dX verdict
A θ-independent fixed-slope dN/dX/CDDF resolution correction **preserves the α-closure unbiasedness**
(it is a θ-blind reshaping of the prior center + mock, exactly like the existing `lit_over_sim_at_z`
z-slope, which the closure is already unbiased WITH), provided: (1) it is applied to dN/dX BEFORE the
telescoping `w_c_from_mu`/renorm and the renorm flows (FLAG 1); (2) it is in the SINGLE shared forward
(prior center + mock + likelihood), not bolted onto one side; (3) the `HCD_LIT_OVER_SIM` provenance is
LF-based so it composes with (not double-counts) the LF→HR resolution slope (FLAG 2 — the one genuine
bias risk, cheap to check). Keeping the prior widths unchanged (per directive) is the correct,
coverage-preserving choice. The T6-step-4 closure re-run is the empirical gate; FLAG 1/2 are what to
verify so that re-run comes back <0.17σ rather than surfacing a mis-wire.

---

## Summary verdicts
- **(a) Form:** separable + rank-1 is SUFFICIENT (minimal representation of the measured (z,τ₀)
  τ₀-slope-steepening sign-flip) and SAFE (θ-blind by construction → immune to the mlp failure mode;
  O(10) DOF fit in the data-rich (z,τ₀) plane, not the 6-sim n_s direction), **conditional on the T3
  aliasing gate** (the residual n_s↔τ₀ degeneracy route is real and only the through-MF Fisher
  ON/OFF + τ₀-width check can close it). Recommend a PHYSICS-CONSTRAINED rank-1 (`u_τ` = the mean-flux
  axis, not a free SVD vector) to remove the n_s-aliasing rotational freedom. Fallback on T3 failure =
  pure separable + the T4 z-resolved floor, never a 2nd rank-1 or a 2-D table.
- **(b) dN/dX:** the θ-independent fixed-slope correction preserves the <0.17σ α-closure unbiasedness
  with unchanged prior widths; two flags to verify (apply-before-renorm; LF-provenance of
  HCD_LIT_OVER_SIM to avoid double-counting the sim→reality gap).
